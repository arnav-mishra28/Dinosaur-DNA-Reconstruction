"""
Dinosaur DNA Reconstruction Inference Module
Handles loading trained models and generating reconstructed sequences
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from Bio import SeqIO, SeqUtils
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
import json
import os
from typing import Dict, List, Tuple, Optional, Union
import logging
from tqdm import tqdm
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from config import PROJECT_CONFIG, GENOMIC_FEATURES
from models import create_model, ModelOutput
from data_collection import SequencePreprocessor, PhylogeneticDataProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DinosaurDNAReconstructor:
    """Main class for reconstructing dinosaur DNA sequences"""
    
    def __init__(self, model_path: str, config: Dict = None):
        self.config = config or PROJECT_CONFIG
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load model
        self.model = self._load_model(model_path)
        self.model.eval()
        
        # Initialize processors
        self.preprocessor = SequencePreprocessor()
        self.phylo_processor = PhylogeneticDataProcessor()
        
        # Species mapping
        self.species_mapping = self._create_species_mapping()
        
        logger.info(f"DinosaurDNA Reconstructor initialized on {self.device}")
    
    def _load_model(self, model_path: str):
        """Load trained model from checkpoint"""
        checkpoint = torch.load(model_path, map_location=self.device)
        
        model = create_model(checkpoint['config'], model_type="hybrid")
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        
        logger.info(f"Model loaded from {model_path}")
        return model
    
    def _create_species_mapping(self) -> Dict[str, int]:
        """Create mapping from species names to IDs"""
        archosaur_species = (
            self.config["data_sources"]["archosaur_genomes"]["crocodilian"] +
            self.config["data_sources"]["archosaur_genomes"]["key_bird_species"]
        )
        
        return {species: idx for idx, species in enumerate(archosaur_species)}
    
    def reconstruct_sequence(
        self, 
        damaged_sequence: str,
        target_species: str = "Theropod_ancestor",
        reference_species: str = "Gallus gallus",
        confidence_threshold: float = 0.7,
        num_samples: int = 10
    ) -> Dict:
        """
        Reconstruct a damaged DNA sequence
        
        Args:
            damaged_sequence: Input damaged DNA sequence
            target_species: Target species for reconstruction
            reference_species: Reference species for phylogenetic context
            confidence_threshold: Minimum confidence for accepting predictions
            num_samples: Number of Monte Carlo samples for uncertainty estimation
        """
        
        # Preprocess input
        processed_input = self._preprocess_input(
            damaged_sequence, target_species, reference_species
        )
        
        # Multiple sampling for uncertainty quantification
        all_reconstructions = []
        all_confidences = []
        
        with torch.no_grad():
            for _ in range(num_samples):
                # Add noise for Monte Carlo sampling
                noisy_input = self._add_reconstruction_noise(processed_input)
                
                # Model inference
                model_output = self.model(**noisy_input)
                
                # Store results
                all_reconstructions.append(model_output.reconstructed_sequence)
                all_confidences.append(model_output.confidence_scores)
        
        # Aggregate results
        reconstruction_result = self._aggregate_reconstructions(
            all_reconstructions, all_confidences, confidence_threshold
        )
        
        # Post-process and validate
        final_result = self._postprocess_reconstruction(
            reconstruction_result, damaged_sequence, target_species
        )
        
        return final_result
    
    def _preprocess_input(
        self, 
        sequence: str, 
        target_species: str,
        reference_species: str
    ) -> Dict[str, torch.Tensor]:
        """Preprocess input sequence for model"""
        
        # Clean and encode sequence
        clean_sequence = sequence.upper().replace('-', 'N')
        
        # Pad/truncate to model's expected length
        max_length = self.config["model"]["transformer"]["max_seq_length"]
        if len(clean_sequence) > max_length:
            clean_sequence = clean_sequence[:max_length]
        else:
            clean_sequence += 'N' * (max_length - len(clean_sequence))
        
        # Encode sequence
        encoded_sequence = torch.tensor(
            self.preprocessor.encode_sequence(clean_sequence), 
            dtype=torch.long
        ).unsqueeze(0).to(self.device)
        
        # Create damage mask
        damage_mask = torch.zeros(1, max_length, dtype=torch.float).to(self.device)
        for i, base in enumerate(clean_sequence):
            if base == 'N':
                damage_mask[0, i] = 1.0
            elif base not in ['A', 'T', 'G', 'C']:
                damage_mask[0, i] = 0.5
        
        # Phylogenetic information
        species_id = self.species_mapping.get(reference_species, 0)
        divergence_time = self._estimate_divergence_time(target_species, reference_species)
        phylo_distance = self._estimate_phylo_distance(target_species, reference_species)
        
        return {
            'input_ids': encoded_sequence,
            'damaged_mask': damage_mask,
            'species_ids': torch.tensor([species_id], dtype=torch.long).to(self.device),
            'divergence_times': torch.tensor([divergence_time], dtype=torch.float).to(self.device),
            'phylo_distances': torch.tensor([phylo_distance], dtype=torch.float).to(self.device)
        }
    
    def _add_reconstruction_noise(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Add small amount of noise for Monte Carlo sampling"""
        noisy_inputs = {}
        
        for key, tensor in inputs.items():
            if key in ['divergence_times', 'phylo_distances']:
                # Add small Gaussian noise to continuous variables
                noise = torch.randn_like(tensor) * 0.05 * tensor
                noisy_inputs[key] = tensor + noise
            else:
                # Keep discrete variables unchanged
                noisy_inputs[key] = tensor
        
        return noisy_inputs
    
    def _aggregate_reconstructions(
        self, 
        reconstructions: List[torch.Tensor], 
        confidences: List[torch.Tensor],
        threshold: float
    ) -> Dict:
        """Aggregate multiple reconstruction samples"""
        
        # Stack predictions
        stacked_recons = torch.stack(reconstructions)  # (num_samples, batch, seq_len, vocab_size)
        stacked_confs = torch.stack(confidences)      # (num_samples, batch, seq_len)
        
        # Average predictions
        mean_logits = stacked_recons.mean(dim=0)
        mean_confidences = stacked_confs.mean(dim=0)
        
        # Uncertainty estimation (variance across samples)
        uncertainty = stacked_recons.var(dim=0).sum(dim=-1)  # Sum over vocab dimension
        
        # Most likely sequence
        predicted_sequence = torch.argmax(mean_logits, dim=-1)
        
        # Identify high-confidence positions
        confident_mask = mean_confidences.squeeze(0) > threshold
        
        return {
            'predicted_sequence': predicted_sequence.squeeze(0),
            'confidence_scores': mean_confidences.squeeze(0),
            'uncertainty_scores': uncertainty.squeeze(0),
            'confident_mask': confident_mask,
            'raw_logits': mean_logits.squeeze(0)
        }
    
    def _postprocess_reconstruction(
        self, 
        reconstruction: Dict, 
        original_damaged: str,
        target_species: str
    ) -> Dict:
        """Post-process reconstruction results"""
        
        # Convert predictions to sequence
        predicted_indices = reconstruction['predicted_sequence'].cpu().numpy()
        reconstructed_seq = self.preprocessor.decode_sequence(predicted_indices.tolist())
        
        # Calculate reconstruction metrics
        metrics = self._calculate_reconstruction_metrics(
            original_damaged, reconstructed_seq, reconstruction
        )
        
        # Generate alternative sequences (top-k sampling)
        alternative_sequences = self._generate_alternatives(
            reconstruction['raw_logits'], k=5
        )
        
        # Biological validation
        bio_validation = self._validate_biology(reconstructed_seq, target_species)
        
        return {
            'reconstructed_sequence': reconstructed_seq,
            'original_damaged': original_damaged,
            'confidence_scores': reconstruction['confidence_scores'].cpu().numpy(),
            'uncertainty_scores': reconstruction['uncertainty_scores'].cpu().numpy(),
            'confident_positions': reconstruction['confident_mask'].cpu().numpy(),
            'reconstruction_metrics': metrics,
            'alternative_sequences': alternative_sequences,
            'biological_validation': bio_validation,
            'target_species': target_species
        }
    
    def _calculate_reconstruction_metrics(
        self, 
        original: str, 
        reconstructed: str, 
        reconstruction_data: Dict
    ) -> Dict:
        """Calculate various reconstruction quality metrics"""
        
        # Basic sequence statistics
        original_clean = original.replace('N', '').replace('-', '')
        recon_clean = reconstructed[:len(original)]
        
        # GC content comparison
        original_gc = SeqUtils.GC(original_clean) if original_clean else 0
        recon_gc = SeqUtils.GC(recon_clean) if recon_clean else 0
        
        # Confidence statistics
        confidences = reconstruction_data['confidence_scores'].cpu().numpy()
        
        # Count reconstructed positions
        n_positions = len(original)
        n_missing = original.count('N') + original.count('-')
        n_reconstructed = n_missing
        
        return {
            'sequence_length': n_positions,
            'positions_reconstructed': n_reconstructed,
            'reconstruction_rate': n_reconstructed / n_positions if n_positions > 0 else 0,
            'mean_confidence': float(confidences.mean()),
            'min_confidence': float(confidences.min()),
            'max_confidence': float(confidences.max()),
            'high_confidence_positions': int((confidences > 0.8).sum()),
            'original_gc_content': original_gc,
            'reconstructed_gc_content': recon_gc,
            'gc_content_difference': abs(original_gc - recon_gc)
        }
    
    def _generate_alternatives(self, logits: torch.Tensor, k: int = 5) -> List[str]:
        """Generate alternative reconstruction sequences using top-k sampling"""
        alternatives = []
        
        for _ in range(k):
            # Temperature scaling for diversity
            temperature = np.random.uniform(0.8, 1.2)
            scaled_logits = logits / temperature
            
            # Sample from probability distribution
            probs = F.softmax(scaled_logits, dim=-1)
            sampled_indices = torch.multinomial(probs, 1).squeeze(-1)
            
            # Convert to sequence
            alt_sequence = self.preprocessor.decode_sequence(sampled_indices.cpu().numpy().tolist())
            alternatives.append(alt_sequence)
        
        return alternatives
    
    def _validate_biology(self, sequence: str, target_species: str) -> Dict:
        """Validate biological plausibility of reconstructed sequence"""
        validation_results = {}
        
        # Check for stop codons in coding regions
        stop_codons = ['TAA', 'TAG', 'TGA']
        stop_codon_count = sum(sequence.count(codon) for codon in stop_codons)
        validation_results['stop_codon_frequency'] = stop_codon_count / (len(sequence) // 3)
        
        # Codon usage bias check (simplified)
        validation_results['codon_usage_score'] = self._calculate_codon_usage_score(sequence)
        
        # GC content plausibility
        gc_content = SeqUtils.GC(sequence)
        expected_gc = self._get_expected_gc_content(target_species)
        validation_results['gc_content'] = gc_content
        validation_results['gc_content_deviation'] = abs(gc_content - expected_gc)
        validation_results['gc_content_plausible'] = abs(gc_content - expected_gc) < 10.0
        
        # Sequence complexity
        validation_results['sequence_complexity'] = self.preprocessor.calculate_complexity(sequence)
        validation_results['sequence_entropy'] = self.preprocessor.calculate_entropy(sequence)
        
        # Overall plausibility score
        plausibility_score = (
            (1.0 if validation_results['gc_content_plausible'] else 0.0) * 0.3 +
            min(validation_results['sequence_complexity'], 1.0) * 0.3 +
            min(validation_results['sequence_entropy'] / 2.0, 1.0) * 0.2 +
            max(0, 1.0 - validation_results['stop_codon_frequency'] * 10) * 0.2
        )
        
        validation_results['overall_plausibility'] = plausibility_score
        validation_results['plausibility_grade'] = (
            'Excellent' if plausibility_score > 0.8 else
            'Good' if plausibility_score > 0.6 else
            'Fair' if plausibility_score > 0.4 else
            'Poor'
        )
        
        return validation_results
    
    def _calculate_codon_usage_score(self, sequence: str) -> float:
        """Calculate codon usage bias score (simplified)"""
        if len(sequence) < 3:
            return 0.0
        
        # Count codons
        codon_counts = {}
        for i in range(0, len(sequence) - 2, 3):
            codon = sequence[i:i+3]
            if 'N' not in codon:
                codon_counts[codon] = codon_counts.get(codon, 0) + 1
        
        if not codon_counts:
            return 0.0
        
        # Calculate entropy (higher entropy = more uniform usage)
        total_codons = sum(codon_counts.values())
        entropy = 0.0
        for count in codon_counts.values():
            if count > 0:
                p = count / total_codons
                entropy -= p * np.log2(p)
        
        # Normalize by maximum possible entropy
        max_entropy = np.log2(len(codon_counts))
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
        
        return normalized_entropy
    
    def _get_expected_gc_content(self, species: str) -> float:
        """Get expected GC content for species"""
        gc_expectations = {
            'Theropod_ancestor': 42.0,
            'Tyrannosaurus_rex': 42.5,
            'Velociraptor': 41.8,
            'Triceratops': 43.0,
            'Gallus gallus': 41.5,
            'default': 42.0
        }
        
        return gc_expectations.get(species, gc_expectations['default'])
    
    def _estimate_divergence_time(self, target: str, reference: str) -> float:
        """Estimate divergence time between target and reference species"""
        # Simplified estimation - would use proper phylogenetic data in practice
        time_estimates = {
            ('Theropod_ancestor', 'Gallus gallus'): 150.0,
            ('Tyrannosaurus_rex', 'Gallus gallus'): 68.0,
            ('Velociraptor', 'Gallus gallus'): 70.0,
            ('Triceratops', 'Gallus gallus'): 68.0
        }
        
        return time_estimates.get((target, reference), 100.0)
    
    def _estimate_phylo_distance(self, target: str, reference: str) -> float:
        """Estimate phylogenetic distance"""
        # Simplified - would calculate from actual phylogenetic tree
        return self._estimate_divergence_time(target, reference) * 2.0
    
    def visualize_reconstruction(self, result: Dict, save_path: str = None) -> go.Figure:
        """Create comprehensive visualization of reconstruction results"""
        
        original = result['original_damaged']
        reconstructed = result['reconstructed_sequence'][:len(original)]
        confidences = result['confidence_scores'][:len(original)]
        
        # Create subplot figure
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=[
                'Sequence Comparison', 'Confidence Scores',
                'GC Content Analysis', 'Reconstruction Quality',
                'Position-wise Comparison', 'Biological Validation'
            ],
            specs=[[{"colspan": 2}, None],
                   [{"type": "scatter"}, {"type": "scatter"}],
                   [{"type": "scatter"}, {"type": "indicator"}]]
        )
        
        # 1. Sequence alignment visualization
        positions = list(range(min(50, len(original))))  # Show first 50 positions
        
        # Color code bases
        color_map = {'A': 'red', 'T': 'blue', 'G': 'green', 'C': 'orange', 'N': 'gray'}
        
        original_colors = [color_map.get(original[i], 'black') for i in positions]
        recon_colors = [color_map.get(reconstructed[i], 'black') for i in positions]
        
        # Add sequence traces
        fig.add_trace(
            go.Scatter(
                x=positions, y=[1] * len(positions),
                mode='markers+text',
                marker=dict(color=original_colors, size=15),
                text=[original[i] for i in positions],
                textposition="middle center",
                name="Original",
                showlegend=True
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=positions, y=[0] * len(positions),
                mode='markers+text',
                marker=dict(color=recon_colors, size=15),
                text=[reconstructed[i] for i in positions],
                textposition="middle center",
                name="Reconstructed",
                showlegend=True
            ),
            row=1, col=1
        )
        
        # 2. Confidence scores
        fig.add_trace(
            go.Scatter(
                x=positions,
                y=confidences[positions],
                mode='lines+markers',
                name="Confidence",
                line=dict(color='purple')
            ),
            row=2, col=1
        )
        
        # 3. GC content sliding window
        window_size = 10
        gc_original = []
        gc_recon = []
        window_positions = []
        
        for i in range(0, len(original) - window_size, window_size//2):
            window_orig = original[i:i+window_size].replace('N', '').replace('-', '')
            window_recon = reconstructed[i:i+window_size].replace('N', '').replace('-', '')
            
            if window_orig and window_recon:
                gc_original.append(SeqUtils.GC(window_orig))
                gc_recon.append(SeqUtils.GC(window_recon))
                window_positions.append(i + window_size//2)
        
        fig.add_trace(
            go.Scatter(
                x=window_positions, y=gc_original,
                mode='lines', name="Original GC%",
                line=dict(color='blue', dash='dash')
            ),
            row=2, col=2
        )
        
        fig.add_trace(
            go.Scatter(
                x=window_positions, y=gc_recon,
                mode='lines', name="Reconstructed GC%",
                line=dict(color='red')
            ),
            row=2, col=2
        )
        
        # 4. Position-wise accuracy
        matches = [1 if original[i] == reconstructed[i] or original[i] == 'N' 
                  else 0 for i in range(len(original))]
        
        fig.add_trace(
            go.Scatter(
                x=list(range(len(matches))), y=matches,
                mode='markers',
                marker=dict(
                    color=confidences[:len(matches)],
                    colorscale='RdYlBu',
                    size=8,
                    colorbar=dict(title="Confidence")
                ),
                name="Match/Mismatch"
            ),
            row=3, col=1
        )
        
        # 5. Overall quality indicator
        overall_score = result['biological_validation']['overall_plausibility']
        
        fig.add_trace(
            go.Indicator(
                mode="gauge+number+delta",
                value=overall_score * 100,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Biological Plausibility"},
                delta={'reference': 80},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 50], 'color': "lightgray"},
                        {'range': [50, 80], 'color': "gray"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                }
            ),
            row=3, col=2
        )
        
        # Update layout
        fig.update_layout(
            title=f"Dinosaur DNA Reconstruction: {result['target_species']}",
            height=800,
            showlegend=True
        )
        
        if save_path:
            fig.write_html(save_path)
            logger.info(f"Visualization saved to {save_path}")
        
        return fig

def main():
    """Example usage of the DinosaurDNA Reconstructor"""
    
    # Example damaged sequence (mitochondrial fragment)
    damaged_sequence = """
    NNNGATCACAGGTCTATCACCCTATTAACCACTCACGGGAGCTCTCCATGCATTTGGTATTTTCGTCTGGGGGTATGCAC
    GCGATAGCATTGCGAGACGCTGGAGCCGGAGCACCCTATGTCGCANNNNNNNNNNTAATGCTCTAGTTTTGCTCCGGTG
    CCAGGGTGAACATTATTNNNNNNCTATAANNNNNNNNNNNNNNGATGGCTATTTNNNNNNNNNTTCCANNNNNNNNNNN
    """
    
    # Initialize reconstructor (you'll need a trained model)
    model_path = os.path.join(PROJECT_CONFIG["paths"]["models"], "best_model.pth")
    
    if not os.path.exists(model_path):
        logger.error(f"Model not found at {model_path}. Train the model first.")
        return
    
    reconstructor = DinosaurDNAReconstructor(model_path)
    
    # Reconstruct sequence
    logger.info("Starting DNA reconstruction...")
    result = reconstructor.reconstruct_sequence(
        damaged_sequence=damaged_sequence,
        target_species="Theropod_ancestor",
        reference_species="Gallus gallus",
        confidence_threshold=0.7,
        num_samples=10
    )
    
    # Print results
    print("\n" + "="*80)
    print("DINOSAUR DNA RECONSTRUCTION RESULTS")
    print("="*80)
    print(f"Target Species: {result['target_species']}")
    print(f"Sequence Length: {result['reconstruction_metrics']['sequence_length']}")
    print(f"Positions Reconstructed: {result['reconstruction_metrics']['positions_reconstructed']}")
    print(f"Mean Confidence: {result['reconstruction_metrics']['mean_confidence']:.3f}")
    print(f"Biological Plausibility: {result['biological_validation']['plausibility_grade']}")
    print(f"GC Content: {result['biological_validation']['gc_content']:.1f}%")
    
    print("\nReconstructed Sequence (first 200bp):")
    print(result['reconstructed_sequence'][:200])
    
    print("\nOriginal Damaged (first 200bp):")
    print(result['original_damaged'][:200])
    
    # Create visualization
    fig = reconstructor.visualize_reconstruction(
        result, 
        save_path=os.path.join(PROJECT_CONFIG["paths"]["figures"], "reconstruction_results.html")
    )
    
    # Save detailed results
    output_path = os.path.join(PROJECT_CONFIG["paths"]["outputs"], "reconstruction_results.json")
    with open(output_path, 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        json_result = {}
        for key, value in result.items():
            if isinstance(value, np.ndarray):
                json_result[key] = value.tolist()
            else:
                json_result[key] = value
        
        json.dump(json_result, f, indent=2)
    
    logger.info(f"Results saved to {output_path}")

if __name__ == "__main__":
    main()
