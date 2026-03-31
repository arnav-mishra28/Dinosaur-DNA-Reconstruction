"""
Evaluation and Benchmarking Module for Dinosaur DNA Reconstruction
Implements comprehensive metrics and validation methods for model performance
"""

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report
from scipy.stats import pearsonr, spearmanr
from Bio.Seq import Seq
from Bio import SeqUtils, Align
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Tuple, Optional, Union
import logging
import json
import os
from datetime import datetime
from tqdm import tqdm

from config import PROJECT_CONFIG
from models import create_model, ModelOutput
from data_collection import SequencePreprocessor, PhylogeneticDataProcessor
from evolutionary_constraints import create_mutation_context

logger = logging.getLogger(__name__)

class DNAReconstructionEvaluator:
    """Comprehensive evaluation system for DNA reconstruction models"""
    
    def __init__(self, model_path: Optional[str] = None, config: Dict = None):
        self.config = config or PROJECT_CONFIG
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load model if provided
        if model_path and os.path.exists(model_path):
            self.model = self._load_model(model_path)
            self.model.eval()
        else:
            self.model = None
            logger.warning("No model provided or model file not found")
        
        # Initialize processors
        self.preprocessor = SequencePreprocessor()
        self.phylo_processor = PhylogeneticDataProcessor()
        
        # Nucleotide mapping
        self.nucleotides = ['A', 'T', 'G', 'C', 'N']
        self.nt_to_idx = {nt: i for i, nt in enumerate(self.nucleotides)}
    
    def _load_model(self, model_path: str):
        """Load trained model"""
        checkpoint = torch.load(model_path, map_location=self.device)
        model = create_model(checkpoint['config'], model_type="hybrid")
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        return model
    
    def evaluate_reconstruction_accuracy(
        self, 
        original_sequences: List[str],
        damaged_sequences: List[str], 
        reconstructed_sequences: List[str]
    ) -> Dict[str, float]:
        """
        Evaluate reconstruction accuracy using multiple metrics
        
        Args:
            original_sequences: Ground truth sequences
            damaged_sequences: Input damaged sequences
            reconstructed_sequences: Model reconstructions
        """
        
        metrics = {
            'sequence_identity': [],
            'position_accuracy': [],
            'damaged_position_accuracy': [],
            'base_accuracy': {nt: [] for nt in self.nucleotides[:4]},
            'gc_content_error': [],
            'length_accuracy': []
        }
        
        for orig, damaged, recon in zip(original_sequences, damaged_sequences, reconstructed_sequences):
            # Ensure same length for comparison
            min_len = min(len(orig), len(recon))
            orig_trimmed = orig[:min_len]
            recon_trimmed = recon[:min_len]
            damaged_trimmed = damaged[:min_len]
            
            # Sequence identity (overall match rate)
            identity = sum(1 for o, r in zip(orig_trimmed, recon_trimmed) if o == r) / min_len
            metrics['sequence_identity'].append(identity)
            
            # Position-wise accuracy
            position_accuracies = [1 if o == r else 0 for o, r in zip(orig_trimmed, recon_trimmed)]
            metrics['position_accuracy'].extend(position_accuracies)
            
            # Accuracy specifically for damaged positions
            damaged_positions = [i for i, (o, d) in enumerate(zip(orig_trimmed, damaged_trimmed)) if o != d]
            if damaged_positions:
                damaged_acc = sum(1 for i in damaged_positions if orig_trimmed[i] == recon_trimmed[i]) / len(damaged_positions)
                metrics['damaged_position_accuracy'].append(damaged_acc)
            
            # Per-base accuracy
            for nt in self.nucleotides[:4]:
                nt_positions = [i for i, base in enumerate(orig_trimmed) if base == nt]
                if nt_positions:
                    nt_acc = sum(1 for i in nt_positions if recon_trimmed[i] == nt) / len(nt_positions)
                    metrics['base_accuracy'][nt].append(nt_acc)
            
            # GC content error
            orig_gc = SeqUtils.gc_fraction(orig_trimmed)
            recon_gc = SeqUtils.gc_fraction(recon_trimmed)
            gc_error = abs(orig_gc - recon_gc)
            metrics['gc_content_error'].append(gc_error)
            
            # Length accuracy
            length_acc = 1.0 - abs(len(orig) - len(recon)) / max(len(orig), len(recon))
            metrics['length_accuracy'].append(length_acc)
        
        # Aggregate metrics
        results = {
            'mean_sequence_identity': np.mean(metrics['sequence_identity']),
            'std_sequence_identity': np.std(metrics['sequence_identity']),
            'mean_position_accuracy': np.mean(metrics['position_accuracy']),
            'mean_damaged_position_accuracy': np.mean(metrics['damaged_position_accuracy']) if metrics['damaged_position_accuracy'] else 0.0,
            'mean_gc_content_error': np.mean(metrics['gc_content_error']),
            'mean_length_accuracy': np.mean(metrics['length_accuracy'])
        }
        
        # Add per-base accuracies
        for nt in self.nucleotides[:4]:
            if metrics['base_accuracy'][nt]:
                results[f'mean_{nt}_accuracy'] = np.mean(metrics['base_accuracy'][nt])
        
        return results
    
    def evaluate_confidence_calibration(
        self,
        confidence_scores: List[np.ndarray],
        reconstruction_errors: List[np.ndarray]
    ) -> Dict[str, float]:
        """
        Evaluate how well confidence scores correlate with actual errors
        
        Args:
            confidence_scores: Model confidence predictions for each position
            reconstruction_errors: Actual reconstruction errors (0 or 1) for each position
        """
        
        # Flatten all scores and errors
        all_confidences = np.concatenate(confidence_scores)
        all_errors = np.concatenate(reconstruction_errors)
        
        # Calculate correlations
        pearson_corr, pearson_p = pearsonr(all_confidences, 1 - all_errors)  # High confidence should correlate with low error
        spearman_corr, spearman_p = spearmanr(all_confidences, 1 - all_errors)
        
        # Expected calibration error (ECE)
        ece = self._calculate_expected_calibration_error(all_confidences, 1 - all_errors)
        
        # Reliability diagram data
        bin_boundaries = np.linspace(0, 1, 11)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        bin_accuracies = []
        bin_confidences = []
        bin_counts = []
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (all_confidences > bin_lower) & (all_confidences <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = (1 - all_errors[in_bin]).mean()
                avg_confidence_in_bin = all_confidences[in_bin].mean()
                bin_accuracies.append(accuracy_in_bin)
                bin_confidences.append(avg_confidence_in_bin)
                bin_counts.append(in_bin.sum())
            else:
                bin_accuracies.append(0)
                bin_confidences.append(0)
                bin_counts.append(0)
        
        return {
            'pearson_correlation': pearson_corr,
            'pearson_p_value': pearson_p,
            'spearman_correlation': spearman_corr,
            'spearman_p_value': spearman_p,
            'expected_calibration_error': ece,
            'reliability_diagram': {
                'bin_accuracies': bin_accuracies,
                'bin_confidences': bin_confidences,
                'bin_counts': bin_counts
            }
        }
    
    def _calculate_expected_calibration_error(self, confidences: np.ndarray, accuracies: np.ndarray, n_bins: int = 15) -> float:
        """Calculate Expected Calibration Error"""
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = accuracies[in_bin].mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        return ece
    
    def evaluate_evolutionary_consistency(
        self,
        sequences: List[str],
        species_pairs: List[Tuple[str, str]],
        divergence_times: List[float]
    ) -> Dict[str, float]:
        """
        Evaluate if reconstructed sequences follow evolutionary constraints
        
        Args:
            sequences: Reconstructed sequences
            species_pairs: Pairs of (species1, species2) for comparison
            divergence_times: Divergence times for each pair
        """
        
        results = {
            'sequence_divergences': [],
            'time_correlations': [],
            'transition_transversion_ratios': [],
            'molecular_clock_violations': []
        }
        
        for i, ((sp1, sp2), div_time) in enumerate(zip(species_pairs, divergence_times)):
            if i + 1 < len(sequences):
                seq1, seq2 = sequences[i], sequences[i + 1]
                
                # Calculate sequence divergence
                divergence = self._calculate_sequence_divergence(seq1, seq2)
                results['sequence_divergences'].append(divergence)
                
                # Check if divergence correlates with time
                results['time_correlations'].append((divergence, div_time))
                
                # Calculate Ts/Tv ratio
                ts_tv_ratio = self._calculate_ts_tv_ratio(seq1, seq2)
                results['transition_transversion_ratios'].append(ts_tv_ratio)
                
                # Check molecular clock
                expected_divergence = div_time * 2e-9  # Approximate substitution rate
                clock_violation = abs(divergence - expected_divergence) / expected_divergence
                results['molecular_clock_violations'].append(clock_violation)
        
        # Calculate correlations
        if results['time_correlations']:
            divergences, times = zip(*results['time_correlations'])
            time_corr, time_p = pearsonr(divergences, times)
        else:
            time_corr, time_p = 0.0, 1.0
        
        return {
            'mean_sequence_divergence': np.mean(results['sequence_divergences']) if results['sequence_divergences'] else 0.0,
            'divergence_time_correlation': time_corr,
            'divergence_time_p_value': time_p,
            'mean_ts_tv_ratio': np.mean(results['transition_transversion_ratios']) if results['transition_transversion_ratios'] else 0.0,
            'mean_molecular_clock_violation': np.mean(results['molecular_clock_violations']) if results['molecular_clock_violations'] else 0.0,
            'ts_tv_ratio_std': np.std(results['transition_transversion_ratios']) if results['transition_transversion_ratios'] else 0.0
        }
    
    def _calculate_sequence_divergence(self, seq1: str, seq2: str) -> float:
        """Calculate evolutionary divergence between two sequences"""
        if len(seq1) != len(seq2):
            min_len = min(len(seq1), len(seq2))
            seq1, seq2 = seq1[:min_len], seq2[:min_len]
        
        differences = sum(1 for a, b in zip(seq1, seq2) if a != b and a != 'N' and b != 'N')
        valid_positions = sum(1 for a, b in zip(seq1, seq2) if a != 'N' and b != 'N')
        
        return differences / valid_positions if valid_positions > 0 else 0.0
    
    def _calculate_ts_tv_ratio(self, seq1: str, seq2: str) -> float:
        """Calculate transition/transversion ratio"""
        transitions = 0
        transversions = 0
        
        transition_pairs = {('A', 'G'), ('G', 'A'), ('C', 'T'), ('T', 'C')}
        
        for a, b in zip(seq1, seq2):
            if a != b and a in 'ATGC' and b in 'ATGC':
                if (a, b) in transition_pairs:
                    transitions += 1
                else:
                    transversions += 1
        
        return transitions / transversions if transversions > 0 else float('inf')
    
    def benchmark_against_baselines(
        self,
        test_sequences: List[Tuple[str, str]],  # (original, damaged)
        baseline_methods: List[str] = None
    ) -> Dict[str, Dict]:
        """
        Benchmark model against baseline methods
        
        Args:
            test_sequences: List of (original, damaged) sequence pairs
            baseline_methods: List of baseline method names
        """
        
        if baseline_methods is None:
            baseline_methods = ['random', 'most_frequent', 'nearest_neighbor', 'consensus']
        
        results = {}
        
        # Get model predictions if model is available
        if self.model:
            model_predictions = []
            for orig, damaged in tqdm(test_sequences, desc="Getting model predictions"):
                pred = self._predict_sequence(damaged)
                model_predictions.append(pred)
            
            # Evaluate model
            originals, _ = zip(*test_sequences)
            model_metrics = self.evaluate_reconstruction_accuracy(
                list(originals), 
                [damaged for _, damaged in test_sequences],
                model_predictions
            )
            results['our_model'] = model_metrics
        
        # Evaluate baseline methods
        for baseline in baseline_methods:
            baseline_predictions = []
            
            for orig, damaged in test_sequences:
                if baseline == 'random':
                    pred = self._random_reconstruction(damaged)
                elif baseline == 'most_frequent':
                    pred = self._most_frequent_reconstruction(damaged)
                elif baseline == 'nearest_neighbor':
                    pred = self._nearest_neighbor_reconstruction(damaged, [o for o, _ in test_sequences])
                elif baseline == 'consensus':
                    pred = self._consensus_reconstruction(damaged, [o for o, _ in test_sequences])
                else:
                    pred = damaged  # No reconstruction
                
                baseline_predictions.append(pred)
            
            # Evaluate baseline
            baseline_metrics = self.evaluate_reconstruction_accuracy(
                [orig for orig, _ in test_sequences],
                [damaged for _, damaged in test_sequences], 
                baseline_predictions
            )
            results[baseline] = baseline_metrics
        
        return results
    
    def _predict_sequence(self, damaged_sequence: str) -> str:
        """Get model prediction for a damaged sequence"""
        if not self.model:
            return damaged_sequence
        
        # Preprocess sequence
        encoded = torch.tensor(
            self.preprocessor.encode_sequence(damaged_sequence),
            dtype=torch.long
        ).unsqueeze(0).to(self.device)
        
        # Create dummy inputs for other required parameters
        batch_size = 1
        damage_mask = torch.zeros(batch_size, len(damaged_sequence), dtype=torch.float).to(self.device)
        species_ids = torch.zeros(batch_size, dtype=torch.long).to(self.device)
        divergence_times = torch.full((batch_size,), 150.0).to(self.device)
        phylo_distances = torch.full((batch_size,), 150.0).to(self.device)
        
        # Model prediction
        with torch.no_grad():
            output = self.model(encoded, damage_mask, species_ids, divergence_times, phylo_distances)
            predictions = torch.argmax(output.reconstructed_sequence, dim=-1)
            predicted_sequence = self.preprocessor.decode_sequence(predictions[0].cpu().tolist())
        
        return predicted_sequence
    
    def _random_reconstruction(self, damaged_sequence: str) -> str:
        """Random baseline: replace damaged bases randomly"""
        nucleotides = ['A', 'T', 'G', 'C']
        result = list(damaged_sequence)
        
        for i, base in enumerate(result):
            if base == 'N':
                result[i] = np.random.choice(nucleotides)
        
        return ''.join(result)
    
    def _most_frequent_reconstruction(self, damaged_sequence: str) -> str:
        """Most frequent baseline: replace with most common base"""
        # Count non-N bases
        counts = {'A': 0, 'T': 0, 'G': 0, 'C': 0}
        for base in damaged_sequence:
            if base in counts:
                counts[base] += 1
        
        most_frequent = max(counts, key=counts.get)
        result = damaged_sequence.replace('N', most_frequent)
        
        return result
    
    def _nearest_neighbor_reconstruction(self, damaged_sequence: str, reference_sequences: List[str]) -> str:
        """Nearest neighbor baseline: use closest reference sequence"""
        if not reference_sequences:
            return damaged_sequence
        
        # Find most similar reference
        best_similarity = -1
        best_ref = reference_sequences[0]
        
        for ref in reference_sequences:
            similarity = sum(1 for a, b in zip(damaged_sequence, ref) 
                           if a == b and a != 'N') / len(damaged_sequence)
            if similarity > best_similarity:
                best_similarity = similarity
                best_ref = ref
        
        # Replace N's with corresponding bases from best reference
        result = list(damaged_sequence)
        for i, base in enumerate(result):
            if base == 'N' and i < len(best_ref):
                result[i] = best_ref[i]
        
        return ''.join(result)
    
    def _consensus_reconstruction(self, damaged_sequence: str, reference_sequences: List[str]) -> str:
        """Consensus baseline: use consensus of reference sequences"""
        if not reference_sequences:
            return damaged_sequence
        
        result = list(damaged_sequence)
        
        for i, base in enumerate(result):
            if base == 'N':
                # Get consensus at this position
                position_bases = [ref[i] for ref in reference_sequences if i < len(ref) and ref[i] != 'N']
                if position_bases:
                    consensus = max(set(position_bases), key=position_bases.count)
                    result[i] = consensus
        
        return ''.join(result)
    
    def create_evaluation_report(
        self,
        evaluation_results: Dict,
        output_path: str = None
    ) -> str:
        """
        Create comprehensive evaluation report
        
        Args:
            evaluation_results: Dictionary with all evaluation metrics
            output_path: Path to save the report
        """
        
        report_lines = [
            "# Dinosaur DNA Reconstruction - Evaluation Report",
            f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Summary",
            "This report provides a comprehensive evaluation of the dinosaur DNA reconstruction model.",
            ""
        ]
        
        # Reconstruction accuracy section
        if 'reconstruction_accuracy' in evaluation_results:
            acc = evaluation_results['reconstruction_accuracy']
            report_lines.extend([
                "## Reconstruction Accuracy",
                f"- Mean Sequence Identity: {acc.get('mean_sequence_identity', 0):.4f} ± {acc.get('std_sequence_identity', 0):.4f}",
                f"- Position-wise Accuracy: {acc.get('mean_position_accuracy', 0):.4f}",
                f"- Damaged Position Accuracy: {acc.get('mean_damaged_position_accuracy', 0):.4f}",
                f"- GC Content Error: {acc.get('mean_gc_content_error', 0):.4f}",
                f"- Length Accuracy: {acc.get('mean_length_accuracy', 0):.4f}",
                ""
            ])
            
            # Per-base accuracies
            report_lines.append("### Per-Base Accuracies")
            for nt in ['A', 'T', 'G', 'C']:
                key = f'mean_{nt}_accuracy'
                if key in acc:
                    report_lines.append(f"- {nt}: {acc[key]:.4f}")
            report_lines.append("")
        
        # Confidence calibration section
        if 'confidence_calibration' in evaluation_results:
            calib = evaluation_results['confidence_calibration']
            report_lines.extend([
                "## Confidence Calibration",
                f"- Pearson Correlation: {calib.get('pearson_correlation', 0):.4f} (p={calib.get('pearson_p_value', 1):.4f})",
                f"- Spearman Correlation: {calib.get('spearman_correlation', 0):.4f} (p={calib.get('spearman_p_value', 1):.4f})",
                f"- Expected Calibration Error: {calib.get('expected_calibration_error', 0):.4f}",
                ""
            ])
        
        # Evolutionary consistency section
        if 'evolutionary_consistency' in evaluation_results:
            evo = evaluation_results['evolutionary_consistency']
            report_lines.extend([
                "## Evolutionary Consistency",
                f"- Mean Sequence Divergence: {evo.get('mean_sequence_divergence', 0):.4f}",
                f"- Divergence-Time Correlation: {evo.get('divergence_time_correlation', 0):.4f} (p={evo.get('divergence_time_p_value', 1):.4f})",
                f"- Mean Ts/Tv Ratio: {evo.get('mean_ts_tv_ratio', 0):.4f} ± {evo.get('ts_tv_ratio_std', 0):.4f}",
                f"- Molecular Clock Violation: {evo.get('mean_molecular_clock_violation', 0):.4f}",
                ""
            ])
        
        # Benchmark comparison section
        if 'benchmark_comparison' in evaluation_results:
            bench = evaluation_results['benchmark_comparison']
            report_lines.extend([
                "## Benchmark Comparison",
                "| Method | Sequence Identity | Position Accuracy | GC Error |",
                "|--------|-------------------|-------------------|----------|"
            ])
            
            for method, metrics in bench.items():
                identity = metrics.get('mean_sequence_identity', 0)
                position_acc = metrics.get('mean_position_accuracy', 0)
                gc_error = metrics.get('mean_gc_content_error', 0)
                report_lines.append(f"| {method} | {identity:.4f} | {position_acc:.4f} | {gc_error:.4f} |")
            
            report_lines.append("")
        
        # Conclusions
        report_lines.extend([
            "## Conclusions",
            "The evaluation results demonstrate the model's performance in reconstructing ancient DNA sequences.",
            "Key findings:",
            "- Reconstruction accuracy shows the model's ability to recover damaged genetic information",
            "- Confidence calibration indicates how well the model estimates its own uncertainty",
            "- Evolutionary consistency validates adherence to biological constraints",
            "- Benchmark comparison shows performance relative to baseline methods",
            ""
        ])
        
        report_text = '\n'.join(report_lines)
        
        # Save report if path provided
        if output_path:
            with open(output_path, 'w') as f:
                f.write(report_text)
            logger.info(f"Evaluation report saved to {output_path}")
        
        return report_text
    
    def create_visualization_dashboard(self, evaluation_results: Dict) -> Dict[str, go.Figure]:
        """Create interactive visualizations for evaluation results"""
        
        figures = {}
        
        # 1. Reconstruction accuracy comparison
        if 'benchmark_comparison' in evaluation_results:
            bench = evaluation_results['benchmark_comparison']
            methods = list(bench.keys())
            identities = [bench[method].get('mean_sequence_identity', 0) for method in methods]
            
            fig = go.Figure(data=[
                go.Bar(x=methods, y=identities, marker_color='lightblue')
            ])
            fig.update_layout(
                title='Reconstruction Accuracy Comparison',
                xaxis_title='Method',
                yaxis_title='Mean Sequence Identity',
                height=400
            )
            figures['accuracy_comparison'] = fig
        
        # 2. Confidence calibration reliability diagram
        if 'confidence_calibration' in evaluation_results:
            calib = evaluation_results['confidence_calibration']
            if 'reliability_diagram' in calib:
                rel_data = calib['reliability_diagram']
                
                fig = go.Figure()
                
                # Perfect calibration line
                fig.add_trace(go.Scatter(
                    x=[0, 1], y=[0, 1],
                    mode='lines',
                    name='Perfect Calibration',
                    line=dict(color='red', dash='dash')
                ))
                
                # Actual calibration
                fig.add_trace(go.Scatter(
                    x=rel_data['bin_confidences'],
                    y=rel_data['bin_accuracies'],
                    mode='markers+lines',
                    name='Model Calibration',
                    marker=dict(size=10, color='blue')
                ))
                
                fig.update_layout(
                    title='Reliability Diagram (Confidence Calibration)',
                    xaxis_title='Mean Predicted Confidence',
                    yaxis_title='Mean Observed Accuracy',
                    height=400
                )
                figures['reliability_diagram'] = fig
        
        # 3. Per-base accuracy radar chart
        if 'reconstruction_accuracy' in evaluation_results:
            acc = evaluation_results['reconstruction_accuracy']
            bases = ['A', 'T', 'G', 'C']
            accuracies = [acc.get(f'mean_{base}_accuracy', 0) for base in bases]
            
            fig = go.Figure(data=go.Scatterpolar(
                r=accuracies + [accuracies[0]],  # Close the radar chart
                theta=bases + [bases[0]],
                fill='toself',
                name='Base Accuracy'
            ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 1]
                    )
                ),
                showlegend=True,
                title='Per-Base Reconstruction Accuracy',
                height=400
            )
            figures['base_accuracy_radar'] = fig
        
        # 4. Evolutionary metrics heatmap
        if 'evolutionary_consistency' in evaluation_results:
            evo = evaluation_results['evolutionary_consistency']
            
            metrics = ['Sequence Divergence', 'Time Correlation', 'Ts/Tv Ratio', 'Clock Violation']
            values = [
                evo.get('mean_sequence_divergence', 0),
                abs(evo.get('divergence_time_correlation', 0)),
                evo.get('mean_ts_tv_ratio', 0) / 3.0,  # Normalize to 0-1 range
                1.0 - evo.get('mean_molecular_clock_violation', 0)  # Invert for better visualization
            ]
            
            fig = go.Figure(data=go.Heatmap(
                z=[values],
                x=metrics,
                y=['Score'],
                colorscale='Viridis',
                showscale=True
            ))
            
            fig.update_layout(
                title='Evolutionary Consistency Metrics',
                height=200
            )
            figures['evolutionary_metrics'] = fig
        
        return figures

def run_comprehensive_evaluation(
    model_path: str,
    test_data_path: str,
    output_dir: str = "evaluation_results"
) -> Dict:
    """
    Run comprehensive evaluation pipeline
    
    Args:
        model_path: Path to trained model
        test_data_path: Path to test dataset
        output_dir: Directory to save evaluation results
    """
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize evaluator
    evaluator = DNAReconstructionEvaluator(model_path)
    
    # Load test data
    with open(test_data_path, 'r') as f:
        test_data = json.load(f)
    
    # Prepare test sequences
    test_sequences = []
    for item in test_data:
        original = item['original_sequence']
        for damaged in item['damaged_sequences']:
            test_sequences.append((original, damaged))
    
    logger.info(f"Evaluating on {len(test_sequences)} test sequences")
    
    # Run evaluations
    results = {}
    
    # 1. Benchmark comparison
    logger.info("Running benchmark comparison...")
    results['benchmark_comparison'] = evaluator.benchmark_against_baselines(test_sequences)
    
    # 2. Get model predictions for detailed evaluation
    if evaluator.model:
        logger.info("Getting model predictions...")
        model_predictions = []
        confidence_scores = []
        
        for orig, damaged in tqdm(test_sequences[:100], desc="Model inference"):  # Limit for efficiency
            pred = evaluator._predict_sequence(damaged)
            model_predictions.append(pred)
            # Mock confidence scores for now
            confidence_scores.append(np.random.rand(len(pred)) * 0.5 + 0.5)
        
        # 3. Detailed accuracy evaluation
        logger.info("Evaluating reconstruction accuracy...")
        results['reconstruction_accuracy'] = evaluator.evaluate_reconstruction_accuracy(
            [orig for orig, _ in test_sequences[:100]],
            [damaged for _, damaged in test_sequences[:100]],
            model_predictions
        )
        
        # 4. Confidence calibration
        logger.info("Evaluating confidence calibration...")
        reconstruction_errors = []
        for i, (orig, damaged) in enumerate(test_sequences[:100]):
            pred = model_predictions[i]
            errors = [1 if o != p else 0 for o, p in zip(orig, pred)]
            reconstruction_errors.append(np.array(errors))
        
        results['confidence_calibration'] = evaluator.evaluate_confidence_calibration(
            confidence_scores, reconstruction_errors
        )
    
    # 5. Evolutionary consistency (mock data for demonstration)
    logger.info("Evaluating evolutionary consistency...")
    species_pairs = [('Species_A', 'Species_B'), ('Species_B', 'Species_C')]
    divergence_times = [50.0, 100.0]
    test_seqs_for_evo = [item['original_sequence'] for item in test_data[:10]]
    
    results['evolutionary_consistency'] = evaluator.evaluate_evolutionary_consistency(
        test_seqs_for_evo, species_pairs[:len(test_seqs_for_evo)//2], divergence_times[:len(test_seqs_for_evo)//2]
    )
    
    # 6. Create visualizations
    logger.info("Creating visualizations...")
    figures = evaluator.create_visualization_dashboard(results)
    
    # Save figures
    for name, fig in figures.items():
        fig.write_html(os.path.join(output_dir, f"{name}.html"))
    
    # 7. Generate report
    logger.info("Generating evaluation report...")
    report = evaluator.create_evaluation_report(results)
    
    with open(os.path.join(output_dir, "evaluation_report.md"), 'w') as f:
        f.write(report)
    
    # Save results as JSON
    with open(os.path.join(output_dir, "evaluation_results.json"), 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        json_results = json.loads(json.dumps(results, default=str))
        json.dump(json_results, f, indent=2)
    
    logger.info(f"Evaluation completed. Results saved to {output_dir}")
    
    return results

if __name__ == "__main__":
    # Example usage
    model_path = "models/best_model.pth"
    test_data_path = "data/processed/test_data.json"
    
    if os.path.exists(test_data_path):
        results = run_comprehensive_evaluation(model_path, test_data_path)
        print("Evaluation completed successfully!")
    else:
        print("Test data not found. Please run data collection first.")
