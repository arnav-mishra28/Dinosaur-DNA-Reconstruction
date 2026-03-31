"""
Enhanced Evaluation Module for Dinosaur DNA Reconstruction
Features: Reconstruction accuracy, confidence calibration, evolutionary consistency,
phylogenetic likelihood, and comprehensive metrics
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import json
import logging
from pathlib import Path
from dataclasses import dataclass
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import pandas as pd
from tqdm import tqdm

from enhanced_config import config
from enhanced_models import create_model, EnhancedDinosaurDNAModel, HybridDinosaurModel

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class EvaluationResults:
    """Store comprehensive evaluation results."""
    
    # Basic reconstruction metrics
    sequence_identity: float = 0.0
    position_wise_accuracy: float = 0.0
    base_wise_precision: Dict[str, float] = None
    base_wise_recall: Dict[str, float] = None
    
    # Confidence and uncertainty
    confidence_calibration_error: float = 0.0
    expected_calibration_error: float = 0.0
    reliability_diagram: Dict = None
    
    # Evolutionary consistency
    transition_transversion_ratio: float = 0.0
    gc_content_preservation: float = 0.0
    codon_usage_similarity: float = 0.0
    
    # Phylogenetic metrics
    phylogenetic_likelihood: float = 0.0
    tree_consistency_score: float = 0.0
    
    # Mutation modeling
    mutation_rate_accuracy: float = 0.0
    damage_pattern_accuracy: float = 0.0
    
    # Overall scores
    weighted_accuracy: float = 0.0
    composite_score: float = 0.0

class SequenceEvaluator:
    """Evaluate DNA sequence reconstruction quality."""
    
    def __init__(self, model: torch.nn.Module, device: str = 'cpu'):
        self.model = model
        self.device = device
        self.model.eval()
        
        # DNA base mapping
        self.base_to_idx = {'A': 0, 'T': 1, 'G': 2, 'C': 3, 'N': 4}
        self.idx_to_base = {v: k for k, v in self.base_to_idx.items()}
        
    def sequence_identity(self, predicted: str, target: str) -> float:
        """Calculate sequence identity percentage."""
        if len(predicted) != len(target):
            # Align sequences (simple implementation)
            min_len = min(len(predicted), len(target))
            predicted = predicted[:min_len]
            target = target[:min_len]
        
        if len(predicted) == 0:
            return 0.0
        
        matches = sum(1 for p, t in zip(predicted, target) if p == t)
        return matches / len(predicted) * 100
    
    def position_wise_accuracy(self, predicted: torch.Tensor, target: torch.Tensor,
                              mask: torch.Tensor) -> float:
        """Calculate position-wise accuracy considering padding."""
        if predicted.shape != target.shape:
            min_len = min(predicted.size(-1), target.size(-1))
            predicted = predicted[..., :min_len]
            target = target[..., :min_len] 
            mask = mask[..., :min_len]
        
        # Get predictions
        pred_tokens = torch.argmax(predicted, dim=-1)
        
        # Apply mask
        valid_positions = mask.bool()
        correct_predictions = (pred_tokens == target) & valid_positions
        
        if valid_positions.sum() == 0:
            return 0.0
        
        return (correct_predictions.sum().float() / valid_positions.sum().float()).item() * 100
    
    def base_wise_metrics(self, predicted: str, target: str) -> Tuple[Dict, Dict]:
        """Calculate precision and recall for each DNA base."""
        bases = ['A', 'T', 'G', 'C']
        precision = {}
        recall = {}
        
        for base in bases:
            # True positives, false positives, false negatives
            tp = sum(1 for p, t in zip(predicted, target) if p == base and t == base)
            fp = sum(1 for p, t in zip(predicted, target) if p == base and t != base)
            fn = sum(1 for p, t in zip(predicted, target) if p != base and t == base)
            
            precision[base] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall[base] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        return precision, recall
    
    def confidence_calibration(self, confidences: torch.Tensor, 
                              accuracies: torch.Tensor,
                              n_bins: int = 10) -> Tuple[float, Dict]:
        """Calculate Expected Calibration Error (ECE) and reliability diagram."""
        confidences = confidences.flatten().cpu().numpy()
        accuracies = accuracies.flatten().cpu().numpy()
        
        # Remove invalid values
        mask = ~(np.isnan(confidences) | np.isnan(accuracies))
        confidences = confidences[mask]
        accuracies = accuracies[mask]
        
        if len(confidences) == 0:
            return float('inf'), {}
        
        # Create bins
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0.0
        reliability_diagram = {
            'bin_centers': [],
            'bin_accuracies': [],
            'bin_confidences': [],
            'bin_counts': []
        }
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # Find samples in this bin
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = accuracies[in_bin].mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                
                # Add to ECE
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
                
                # Store for reliability diagram
                reliability_diagram['bin_centers'].append((bin_lower + bin_upper) / 2)
                reliability_diagram['bin_accuracies'].append(accuracy_in_bin)
                reliability_diagram['bin_confidences'].append(avg_confidence_in_bin)
                reliability_diagram['bin_counts'].append(in_bin.sum())
        
        return ece * 100, reliability_diagram  # Convert to percentage

class EvolutionaryEvaluator:
    """Evaluate evolutionary consistency of reconstructed sequences."""
    
    def __init__(self):
        # Expected evolutionary parameters for vertebrates
        self.expected_ts_tv_ratio = 2.0  # Transition/Transversion ratio
        self.expected_gc_content = 0.42  # Approximate GC content for vertebrates
        
    def transition_transversion_ratio(self, original: str, reconstructed: str) -> float:
        """Calculate transition/transversion ratio in differences."""
        transitions = {'A': 'G', 'G': 'A', 'C': 'T', 'T': 'C'}
        transversions = {
            'A': ['C', 'T'], 'G': ['C', 'T'],
            'C': ['A', 'G'], 'T': ['A', 'G']
        }
        
        ts_count = 0
        tv_count = 0
        
        for orig, recon in zip(original, reconstructed):
            if orig != recon and orig in transitions and recon in transitions:
                if transitions[orig] == recon:
                    ts_count += 1
                elif recon in transversions.get(orig, []):
                    tv_count += 1
        
        if tv_count == 0:
            return float('inf') if ts_count > 0 else 0.0
        
        return ts_count / tv_count
    
    def gc_content_preservation(self, original: str, reconstructed: str) -> float:
        """Calculate how well GC content is preserved."""
        def gc_content(seq):
            gc_count = seq.count('G') + seq.count('C')
            return gc_count / len(seq) if len(seq) > 0 else 0
        
        orig_gc = gc_content(original)
        recon_gc = gc_content(reconstructed)
        
        # Return similarity as percentage
        return 100 - abs(orig_gc - recon_gc) * 100
    
    def codon_usage_similarity(self, original: str, reconstructed: str) -> float:
        """Calculate codon usage similarity (simplified)."""
        def get_codons(seq):
            codons = []
            for i in range(0, len(seq) - 2, 3):
                codon = seq[i:i+3]
                if len(codon) == 3 and all(base in 'ATGC' for base in codon):
                    codons.append(codon)
            return codons
        
        orig_codons = get_codons(original)
        recon_codons = get_codons(reconstructed)
        
        if not orig_codons or not recon_codons:
            return 0.0
        
        # Simple codon frequency comparison
        orig_freq = {codon: orig_codons.count(codon) / len(orig_codons) for codon in set(orig_codons)}
        recon_freq = {codon: recon_codons.count(codon) / len(recon_codons) for codon in set(recon_codons)}
        
        # Calculate similarity using cosine similarity
        all_codons = set(orig_freq.keys()) | set(recon_freq.keys())
        
        orig_vec = np.array([orig_freq.get(codon, 0) for codon in all_codons])
        recon_vec = np.array([recon_freq.get(codon, 0) for codon in all_codons])
        
        if np.linalg.norm(orig_vec) == 0 or np.linalg.norm(recon_vec) == 0:
            return 0.0
        
        cosine_sim = np.dot(orig_vec, recon_vec) / (np.linalg.norm(orig_vec) * np.linalg.norm(recon_vec))
        return cosine_sim * 100

class PhylogeneticEvaluator:
    """Evaluate phylogenetic consistency."""
    
    def __init__(self):
        self.species_tree = config.PHYLOGENETIC_TREE
        
    def phylogenetic_likelihood(self, sequences: Dict[str, str]) -> float:
        """Calculate phylogenetic likelihood (simplified)."""
        # This is a simplified implementation
        # In practice, you would use phylogenetic software like RAxML or IQ-TREE
        
        if len(sequences) < 2:
            return 0.0
        
        # Calculate pairwise distances
        distances = {}
        species_list = list(sequences.keys())
        
        for i, sp1 in enumerate(species_list):
            for j, sp2 in enumerate(species_list[i+1:], i+1):
                seq1, seq2 = sequences[sp1], sequences[sp2]
                distance = self._sequence_distance(seq1, seq2)
                distances[(sp1, sp2)] = distance
        
        # Compare with expected phylogenetic distances
        expected_distances = self._get_expected_distances()
        
        correlation = 0.0
        count = 0
        for (sp1, sp2), observed in distances.items():
            expected = expected_distances.get((sp1, sp2), expected_distances.get((sp2, sp1)))
            if expected is not None:
                correlation += abs(observed - expected)
                count += 1
        
        if count == 0:
            return 0.0
        
        # Convert to similarity score (higher is better)
        avg_error = correlation / count
        return max(0, 100 - avg_error * 100)
    
    def _sequence_distance(self, seq1: str, seq2: str) -> float:
        """Calculate sequence distance (simple p-distance)."""
        if len(seq1) != len(seq2):
            min_len = min(len(seq1), len(seq2))
            seq1, seq2 = seq1[:min_len], seq2[:min_len]
        
        if len(seq1) == 0:
            return 1.0
        
        differences = sum(1 for a, b in zip(seq1, seq2) if a != b)
        return differences / len(seq1)
    
    def _get_expected_distances(self) -> Dict[Tuple[str, str], float]:
        """Get expected phylogenetic distances between species."""
        # Simplified expected distances based on divergence times
        expected = {
            ('Gallus_gallus', 'Anas_platyrhynchos'): 0.15,
            ('Gallus_gallus', 'Struthio_camelus'): 0.25,
            ('Gallus_gallus', 'Alligator_mississippiensis'): 0.50,
            ('Anas_platyrhynchos', 'Struthio_camelus'): 0.20,
            ('Alligator_mississippiensis', 'Crocodylus_porosus'): 0.10,
            # Add more as needed
        }
        return expected

class ComprehensiveEvaluator:
    """Main evaluation class combining all metrics."""
    
    def __init__(self, model: torch.nn.Module, device: str = 'cpu'):
        self.model = model
        self.device = device
        
        # Individual evaluators
        self.sequence_eval = SequenceEvaluator(model, device)
        self.evolution_eval = EvolutionaryEvaluator()
        self.phylo_eval = PhylogeneticEvaluator()
        
        # Metric weights for composite score
        self.metric_weights = config.EVALUATION_CONFIG.get('metric_weights', {
            'sequence_identity': 0.3,
            'position_wise_accuracy': 0.25,
            'confidence_calibration': 0.15,
            'evolutionary_consistency': 0.15,
            'phylogenetic_likelihood': 0.15,
        })
    
    def evaluate_batch(self, batch: Dict, model_outputs: Dict) -> Dict[str, float]:
        """Evaluate a single batch."""
        metrics = {}
        
        # Extract data
        predicted_logits = model_outputs.get('logits', model_outputs.get('fused_logits'))
        target_ids = batch['target_ids']
        target_mask = batch['target_mask']
        confidences = model_outputs.get('confidence', torch.ones_like(predicted_logits[..., 0]))
        
        # Position-wise accuracy
        metrics['position_wise_accuracy'] = self.sequence_eval.position_wise_accuracy(
            predicted_logits, target_ids, target_mask
        )
        
        # Convert to sequences for other metrics
        predicted_seqs = self._logits_to_sequences(predicted_logits, target_mask)
        target_seqs = self._tokens_to_sequences(target_ids, target_mask)
        
        # Sequence identity
        seq_identities = []
        for pred_seq, target_seq in zip(predicted_seqs, target_seqs):
            identity = self.sequence_eval.sequence_identity(pred_seq, target_seq)
            seq_identities.append(identity)
        
        metrics['sequence_identity'] = np.mean(seq_identities) if seq_identities else 0.0
        
        # Confidence calibration
        accuracies = torch.tensor([
            self.sequence_eval.sequence_identity(pred, target) / 100
            for pred, target in zip(predicted_seqs, target_seqs)
        ], device=self.device)
        
        if confidences.numel() > 0 and accuracies.numel() > 0:
            ece, _ = self.sequence_eval.confidence_calibration(
                confidences.mean(dim=-1), accuracies
            )
            metrics['confidence_calibration_error'] = ece
        else:
            metrics['confidence_calibration_error'] = float('inf')
        
        return metrics
    
    def evaluate_dataset(self, dataloader, max_batches: Optional[int] = None) -> EvaluationResults:
        """Evaluate entire dataset."""
        logger.info("Starting comprehensive evaluation...")
        
        all_metrics = defaultdict(list)
        batch_count = 0
        
        self.model.eval()
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                if max_batches and batch_count >= max_batches:
                    break
                
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Forward pass
                try:
                    outputs = self.model(
                        input_ids=batch['input_ids'],
                        attention_mask=batch['attention_mask'],
                        species_ids=batch.get('species_ids'),
                        divergence_times=batch.get('divergence_times'),
                        taxonomy=batch.get('taxonomy'),
                    )
                    
                    # Evaluate batch
                    batch_metrics = self.evaluate_batch(batch, outputs)
                    
                    # Accumulate metrics
                    for metric, value in batch_metrics.items():
                        if not np.isnan(value) and not np.isinf(value):
                            all_metrics[metric].append(value)
                    
                    batch_count += 1
                    
                except Exception as e:
                    logger.warning(f"Error evaluating batch {batch_count}: {e}")
                    continue
        
        # Calculate final metrics
        results = EvaluationResults()
        
        # Average basic metrics
        results.sequence_identity = np.mean(all_metrics['sequence_identity']) if all_metrics['sequence_identity'] else 0.0
        results.position_wise_accuracy = np.mean(all_metrics['position_wise_accuracy']) if all_metrics['position_wise_accuracy'] else 0.0
        results.confidence_calibration_error = np.mean(all_metrics['confidence_calibration_error']) if all_metrics['confidence_calibration_error'] else float('inf')
        
        # Calculate composite score
        valid_metrics = {}
        for metric, weight in self.metric_weights.items():
            if hasattr(results, metric):
                value = getattr(results, metric)
                if not np.isnan(value) and not np.isinf(value):
                    valid_metrics[metric] = value
        
        if valid_metrics:
            # Normalize metrics to 0-100 scale and calculate weighted average
            normalized_scores = {}
            for metric, value in valid_metrics.items():
                if 'error' in metric:
                    # For error metrics, lower is better - convert to score
                    normalized_scores[metric] = max(0, 100 - value)
                else:
                    # For accuracy metrics, use as-is (assuming 0-100 scale)
                    normalized_scores[metric] = min(100, max(0, value))
            
            # Calculate weighted composite score
            total_weight = sum(self.metric_weights[metric] for metric in normalized_scores.keys())
            if total_weight > 0:
                results.composite_score = sum(
                    normalized_scores[metric] * self.metric_weights[metric] 
                    for metric in normalized_scores.keys()
                ) / total_weight
            else:
                results.composite_score = 0.0
        else:
            results.composite_score = 0.0
        
        logger.info(f"Evaluation completed. Composite score: {results.composite_score:.2f}")
        return results
    
    def _logits_to_sequences(self, logits: torch.Tensor, mask: torch.Tensor) -> List[str]:
        """Convert logits to DNA sequences."""
        predicted_tokens = torch.argmax(logits, dim=-1)
        return self._tokens_to_sequences(predicted_tokens, mask)
    
    def _tokens_to_sequences(self, tokens: torch.Tensor, mask: torch.Tensor) -> List[str]:
        """Convert token IDs to DNA sequences."""
        sequences = []
        
        for i in range(tokens.size(0)):
            seq_tokens = tokens[i]
            seq_mask = mask[i] if mask is not None else torch.ones_like(seq_tokens)
            
            # Get valid tokens
            valid_positions = seq_mask.bool()
            valid_tokens = seq_tokens[valid_positions]
            
            # Convert to sequence (simplified - assumes direct base mapping)
            sequence = ""
            for token in valid_tokens:
                token_id = token.item()
                if token_id < 4:  # A, T, G, C
                    base = ['A', 'T', 'G', 'C'][token_id]
                    sequence += base
                elif token_id == 4:  # N
                    sequence += 'N'
                # Skip special tokens
            
            sequences.append(sequence)
        
        return sequences
    
    def save_results(self, results: EvaluationResults, filepath: Path) -> None:
        """Save evaluation results to file."""
        # Convert to dictionary for JSON serialization
        results_dict = {
            'sequence_identity': results.sequence_identity,
            'position_wise_accuracy': results.position_wise_accuracy,
            'confidence_calibration_error': results.confidence_calibration_error,
            'composite_score': results.composite_score,
            # Add other metrics as needed
        }
        
        with open(filepath, 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        logger.info(f"Evaluation results saved to {filepath}")
    
    def create_evaluation_report(self, results: EvaluationResults, 
                               output_dir: Path) -> None:
        """Create comprehensive evaluation report with plots."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create summary report
        report_file = output_dir / "evaluation_report.txt"
        with open(report_file, 'w') as f:
            f.write("Dinosaur DNA Reconstruction - Evaluation Report\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Sequence Identity: {results.sequence_identity:.2f}%\n")
            f.write(f"Position-wise Accuracy: {results.position_wise_accuracy:.2f}%\n")
            f.write(f"Confidence Calibration Error: {results.confidence_calibration_error:.2f}%\n")
            f.write(f"Composite Score: {results.composite_score:.2f}/100\n")
            f.write("\n")
            f.write("Interpretation:\n")
            if results.composite_score >= 80:
                f.write("Excellent reconstruction quality\n")
            elif results.composite_score >= 60:
                f.write("Good reconstruction quality\n")
            elif results.composite_score >= 40:
                f.write("Moderate reconstruction quality\n")
            else:
                f.write("Poor reconstruction quality - needs improvement\n")
        
        logger.info(f"Evaluation report saved to {report_file}")

def main():
    """Main evaluation function."""
    logger.info("Starting model evaluation...")
    
    # Load model (you'll need to specify the checkpoint path)
    model_checkpoint = config.MODEL_DIR / "checkpoints" / "best_model.pt"
    
    if not model_checkpoint.exists():
        logger.error(f"Model checkpoint not found: {model_checkpoint}")
        logger.info("Please train the model first using enhanced_training.py")
        return
    
    # Load model
    model = create_model('hybrid')
    checkpoint = torch.load(model_checkpoint, map_location=config.DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(config.DEVICE)
    
    # Create evaluator
    evaluator = ComprehensiveEvaluator(model, config.DEVICE)
    
    # TODO: Load test dataset and create dataloader
    # This would require the test dataset preparation
    
    logger.info("Evaluation setup complete!")

if __name__ == "__main__":
    main()
