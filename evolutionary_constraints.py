"""
Advanced Evolutionary Constraints and Mutation Models
Implements sophisticated transition matrices, mutation probabilities, and evolutionary constraints
Based on paleogenomics research and ancient DNA studies
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
import logging
from dataclasses import dataclass
from scipy.linalg import expm
import json

logger = logging.getLogger(__name__)

@dataclass
class MutationContext:
    """Context for mutation predictions"""
    nucleotide_context: torch.Tensor  # Local sequence context
    gc_content: torch.Tensor         # Local GC content
    cpg_density: torch.Tensor        # CpG island density
    repetitive_content: torch.Tensor # Repetitive element content
    time_since_divergence: torch.Tensor  # Evolutionary time
    temperature: torch.Tensor        # Ancient environmental conditions

class AdvancedMutationModel(nn.Module):
    """
    Advanced mutation model incorporating multiple evolutionary factors
    Based on real paleogenomics research and ancient DNA degradation patterns
    """
    
    def __init__(self, d_model: int = 512, context_window: int = 7):
        super().__init__()
        self.d_model = d_model
        self.context_window = context_window
        
        # Base transition matrix (Kimura 2-parameter + extensions)
        self.base_rate_matrix = nn.Parameter(
            torch.zeros(4, 4), requires_grad=True
        )
        
        # Context-dependent mutation networks
        self.context_encoder = nn.Sequential(
            nn.Conv1d(4, 64, kernel_size=context_window, padding=context_window//2),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(128, d_model)
        )
        
        # Environmental factor encoders
        self.gc_encoder = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, 64)
        )
        
        self.cpg_encoder = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(), 
            nn.Linear(32, 64)
        )
        
        self.time_encoder = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, 128)
        )
        
        # Temperature/environmental encoder
        self.env_encoder = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, 64)
        )
        
        # Fusion network for context-dependent rates
        fusion_input_dim = d_model + 64 + 64 + 128 + 64
        self.fusion_network = nn.Sequential(
            nn.Linear(fusion_input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 16),  # 4x4 rate matrix
            nn.Softplus()  # Ensure positive rates
        )
        
        # Ancient DNA damage patterns
        self.damage_model = AncientDNADamageModel()
        
        # Initialize with biologically realistic values
        self._initialize_rates()
    
    def _initialize_rates(self):
        """Initialize rate matrix with known biological constraints"""
        # Based on observed vertebrate mutation patterns
        # Transition/transversion ratio ~2.0, with C->T bias from deamination
        
        with torch.no_grad():
            # Initialize symmetric rate matrix (will be modified by context)
            initial_rates = torch.tensor([
                [0.0, 0.05, 0.15, 0.08],   # A -> T,G,C (purine->pyrimidine, transitions)
                [0.05, 0.0, 0.08, 0.25],   # T -> A,G,C (C->T from deamination)
                [0.15, 0.08, 0.0, 0.05],   # G -> A,T,C (transitions favored)
                [0.08, 0.25, 0.05, 0.0]    # C -> A,T,G (high C->T rate)
            ])
            
            self.base_rate_matrix.data = initial_rates
    
    def forward(self, context: MutationContext) -> torch.Tensor:
        """
        Predict context-dependent mutation rates
        
        Args:
            context: MutationContext with all relevant factors
            
        Returns:
            4x4 rate matrix for each position (batch_size, seq_len, 4, 4)
        """
        batch_size, seq_len = context.nucleotide_context.shape[:2]
        
        # Encode local sequence context
        context_features = self.context_encoder(
            context.nucleotide_context.float().transpose(1, 2)
        )
        
        # Encode environmental factors
        gc_features = self.gc_encoder(context.gc_content.unsqueeze(-1))
        cpg_features = self.cpg_encoder(context.cpg_density.unsqueeze(-1))
        time_features = self.time_encoder(context.time_since_divergence.unsqueeze(-1))
        env_features = self.env_encoder(context.temperature.unsqueeze(-1))
        
        # Combine all features
        combined_features = torch.cat([
            context_features,
            gc_features,
            cpg_features, 
            time_features,
            env_features
        ], dim=-1)
        
        # Predict context-specific rate modulations
        rate_modulations = self.fusion_network(combined_features)
        rate_modulations = rate_modulations.view(batch_size, 4, 4)
        
        # Apply to base rate matrix
        base_rates = self.base_rate_matrix.unsqueeze(0).expand(batch_size, -1, -1)
        context_rates = base_rates * rate_modulations
        
        # Normalize rows (ensure proper rate matrix)
        context_rates = self._normalize_rate_matrix(context_rates)
        
        # Add ancient DNA damage if specified
        if hasattr(context, 'ancient_damage') and context.ancient_damage:
            damage_rates = self.damage_model(context_rates, context.time_since_divergence)
            context_rates = damage_rates
        
        # Expand to sequence length
        context_rates = context_rates.unsqueeze(1).expand(-1, seq_len, -1, -1)
        
        return context_rates
    
    def _normalize_rate_matrix(self, rate_matrix: torch.Tensor) -> torch.Tensor:
        """Normalize rate matrix to ensure valid transition probabilities"""
        # Set diagonal to negative sum of off-diagonal elements
        off_diagonal_sum = rate_matrix.sum(dim=-1, keepdim=True) - torch.diagonal(rate_matrix, dim1=-2, dim2=-1).unsqueeze(-1)
        
        # Zero out diagonal
        rate_matrix = rate_matrix * (1 - torch.eye(4, device=rate_matrix.device))
        
        # Set diagonal
        diagonal_indices = torch.arange(4, device=rate_matrix.device)
        rate_matrix[:, diagonal_indices, diagonal_indices] = -off_diagonal_sum.squeeze(-1)
        
        return rate_matrix

class AncientDNADamageModel(nn.Module):
    """
    Models ancient DNA damage patterns observed in paleogenomics
    Implements C->T deamination, purine loss, and fragmentation effects
    """
    
    def __init__(self):
        super().__init__()
        
        # Temperature-dependent damage rates
        self.damage_rate_predictor = nn.Sequential(
            nn.Linear(2, 32),  # temperature + time
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.Sigmoid()
        )
        
        # Position-dependent damage (5'/3' end effects)
        self.position_effect = nn.Parameter(torch.ones(1000))  # Max sequence length
        
    def forward(self, base_rates: torch.Tensor, time: torch.Tensor, 
                temperature: torch.Tensor = None) -> torch.Tensor:
        """
        Apply ancient DNA damage patterns to base mutation rates
        
        Args:
            base_rates: Base mutation rate matrix (batch_size, 4, 4)
            time: Time since death/preservation (batch_size,)
            temperature: Average preservation temperature (batch_size,)
        """
        if temperature is None:
            temperature = torch.zeros_like(time)  # Assume cold preservation
        
        batch_size = base_rates.shape[0]
        device = base_rates.device
        
        # Predict damage rates based on time and temperature
        damage_input = torch.stack([time, temperature], dim=-1)
        damage_multiplier = self.damage_rate_predictor(damage_input)
        
        # Apply specific ancient DNA damage patterns
        damaged_rates = base_rates.clone()
        
        # C->T deamination (highly enhanced in ancient DNA)
        c_to_t_enhancement = 3.0 + 2.0 * damage_multiplier[:, 0]  # Enhanced 3-5x
        damaged_rates[:, 3, 1] *= c_to_t_enhancement  # C -> T
        
        # G->A deamination (reverse complement)
        g_to_a_enhancement = 2.5 + 1.5 * damage_multiplier[:, 1]
        damaged_rates[:, 2, 0] *= g_to_a_enhancement  # G -> A
        
        # Purine loss (A,G -> gaps, modeled as increased transition to N)
        purine_loss_rate = damage_multiplier[:, 2] * 0.1
        damaged_rates[:, 0, :] *= (1 + purine_loss_rate.unsqueeze(-1))  # A loss
        damaged_rates[:, 2, :] *= (1 + purine_loss_rate.unsqueeze(-1))  # G loss
        
        # Re-normalize
        return self._normalize_matrix(damaged_rates)
    
    def _normalize_matrix(self, matrix: torch.Tensor) -> torch.Tensor:
        """Normalize mutation rate matrix"""
        # Ensure rows sum to zero (continuous time Markov chain property)
        row_sums = matrix.sum(dim=-1, keepdim=True)
        matrix_diagonal_mask = torch.eye(4, device=matrix.device).bool()
        
        # Set off-diagonal elements
        off_diagonal_matrix = matrix * (~matrix_diagonal_mask).float()
        
        # Set diagonal to negative sum of off-diagonal
        diagonal_values = -off_diagonal_matrix.sum(dim=-1)
        matrix = off_diagonal_matrix
        matrix[:, matrix_diagonal_mask] = diagonal_values.flatten()
        
        return matrix

class PhylogeneticConstraintModel(nn.Module):
    """
    Enforces phylogenetic constraints on mutations
    Uses known species relationships to constrain evolution
    """
    
    def __init__(self, num_species: int = 20, embedding_dim: int = 256):
        super().__init__()
        
        # Species-specific rate modifiers
        self.species_embeddings = nn.Embedding(num_species, embedding_dim)
        
        # Phylogenetic distance encoder
        self.distance_encoder = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, embedding_dim)
        )
        
        # Rate modification network
        self.rate_modifier = nn.Sequential(
            nn.Linear(embedding_dim * 2, 256),
            nn.ReLU(),
            nn.Linear(256, 16),  # 4x4 rate matrix
            nn.Tanh()  # Allow both positive and negative modulations
        )
        
        # Molecular clock constraints
        self.clock_predictor = nn.Sequential(
            nn.Linear(embedding_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Softplus()
        )
    
    def forward(self, base_rates: torch.Tensor, species_id: torch.Tensor,
                phylo_distance: torch.Tensor, divergence_time: torch.Tensor) -> torch.Tensor:
        """
        Apply phylogenetic constraints to mutation rates
        
        Args:
            base_rates: Base rate matrix (batch_size, 4, 4) or (batch_size, seq_len, 4, 4)
            species_id: Species identifier (batch_size,)
            phylo_distance: Phylogenetic distance from reference (batch_size,)
            divergence_time: Time since divergence (batch_size,)
        """
        batch_size = species_id.shape[0]
        
        # Get species-specific embeddings
        species_emb = self.species_embeddings(species_id)
        
        # Encode phylogenetic distance
        distance_emb = self.distance_encoder(phylo_distance.unsqueeze(-1))
        
        # Combine species and distance information
        combined_emb = torch.cat([species_emb, distance_emb], dim=-1)
        
        # Predict rate modifications
        rate_mods = self.rate_modifier(combined_emb)
        rate_mods = rate_mods.view(batch_size, 4, 4)
        
        # Apply molecular clock constraint
        clock_rates = self.clock_predictor(species_emb).squeeze(-1)
        time_scaling = divergence_time * clock_rates / 100.0  # Normalize
        
        # Scale base rates by phylogenetic factors
        if len(base_rates.shape) == 3:  # (batch_size, 4, 4)
            phylo_rates = base_rates * (1.0 + 0.1 * rate_mods) * time_scaling.unsqueeze(-1).unsqueeze(-1)
        else:  # (batch_size, seq_len, 4, 4)
            seq_len = base_rates.shape[1]
            rate_mods = rate_mods.unsqueeze(1).expand(-1, seq_len, -1, -1)
            time_scaling = time_scaling.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            phylo_rates = base_rates * (1.0 + 0.1 * rate_mods) * time_scaling
        
        return phylo_rates

class TransitionMatrixGenerator(nn.Module):
    """
    Generates transition probability matrices from rate matrices
    Handles matrix exponentiation for continuous-time evolution
    """
    
    def __init__(self, max_time: float = 500.0, time_steps: int = 100):
        super().__init__()
        self.max_time = max_time
        self.time_steps = time_steps
        
        # Pre-compute time points for efficiency
        self.register_buffer('time_points', 
                           torch.linspace(0, max_time, time_steps))
    
    def forward(self, rate_matrix: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        """
        Convert rate matrix to transition probability matrix
        
        Args:
            rate_matrix: Continuous-time rate matrix (batch_size, 4, 4) or (batch_size, seq_len, 4, 4)
            time: Evolution time (batch_size,)
            
        Returns:
            Transition probability matrix
        """
        batch_size = rate_matrix.shape[0]
        device = rate_matrix.device
        
        if len(rate_matrix.shape) == 3:  # Single rate matrix per batch
            # Matrix exponentiation: P(t) = exp(Q*t)
            transition_matrices = []
            
            for i in range(batch_size):
                Qt = rate_matrix[i] * time[i]
                try:
                    # Use torch.linalg.matrix_exp for matrix exponentiation
                    P_t = torch.linalg.matrix_exp(Qt)
                    transition_matrices.append(P_t)
                except:
                    # Fallback to approximation if matrix_exp fails
                    P_t = self._matrix_exp_approximation(Qt)
                    transition_matrices.append(P_t)
            
            return torch.stack(transition_matrices)
        
        else:  # Rate matrix per sequence position
            seq_len = rate_matrix.shape[1]
            transition_matrices = []
            
            for i in range(batch_size):
                batch_transitions = []
                for j in range(seq_len):
                    Qt = rate_matrix[i, j] * time[i]
                    try:
                        P_t = torch.linalg.matrix_exp(Qt)
                        batch_transitions.append(P_t)
                    except:
                        P_t = self._matrix_exp_approximation(Qt)
                        batch_transitions.append(P_t)
                
                transition_matrices.append(torch.stack(batch_transitions))
            
            return torch.stack(transition_matrices)
    
    def _matrix_exp_approximation(self, A: torch.Tensor, terms: int = 10) -> torch.Tensor:
        """
        Approximate matrix exponential using Taylor series
        exp(A) ≈ I + A + A²/2! + A³/3! + ...
        """
        device = A.device
        I = torch.eye(A.shape[-1], device=device)
        
        result = I.clone()
        power = I.clone()
        factorial = 1.0
        
        for k in range(1, terms + 1):
            factorial *= k
            power = torch.matmul(power, A)
            result += power / factorial
        
        return result

def create_mutation_context(
    sequences: torch.Tensor,
    species_info: Dict,
    environmental_data: Optional[Dict] = None
) -> MutationContext:
    """
    Create mutation context from input data
    
    Args:
        sequences: Input sequences (batch_size, seq_len)
        species_info: Dictionary with species information
        environmental_data: Optional environmental data
    """
    batch_size, seq_len = sequences.shape
    device = sequences.device
    
    # Calculate GC content
    gc_content = calculate_gc_content(sequences)
    
    # Calculate CpG density
    cpg_density = calculate_cpg_density(sequences)
    
    # Get repetitive content (simplified)
    repetitive_content = torch.zeros(batch_size, device=device)
    
    # One-hot encode sequences for context
    nucleotide_context = F.one_hot(sequences, num_classes=5)[:, :, :4].float()
    
    # Get temporal information
    divergence_times = torch.tensor([
        species_info.get('divergence_time', 150.0) for _ in range(batch_size)
    ], device=device)
    
    # Environmental conditions (default to cold preservation)
    if environmental_data:
        temperature = torch.tensor([
            environmental_data.get('temperature', -10.0) for _ in range(batch_size)
        ], device=device)
    else:
        temperature = torch.full((batch_size,), -10.0, device=device)
    
    return MutationContext(
        nucleotide_context=nucleotide_context,
        gc_content=gc_content,
        cpg_density=cpg_density,
        repetitive_content=repetitive_content,
        time_since_divergence=divergence_times,
        temperature=temperature
    )

def calculate_gc_content(sequences: torch.Tensor, window_size: int = 100) -> torch.Tensor:
    """Calculate local GC content"""
    # G=2, C=3 in standard encoding
    gc_mask = (sequences == 2) | (sequences == 3)
    
    # Use conv1d for sliding window
    gc_counts = F.conv1d(
        gc_mask.float().unsqueeze(1),
        torch.ones(1, 1, window_size, device=sequences.device) / window_size,
        padding=window_size // 2
    )
    
    return gc_counts.squeeze(1)

def calculate_cpg_density(sequences: torch.Tensor) -> torch.Tensor:
    """Calculate CpG dinucleotide density"""
    batch_size, seq_len = sequences.shape
    device = sequences.device
    
    # Find CpG dinucleotides (C=3, G=2)
    cpg_positions = ((sequences[:, :-1] == 3) & (sequences[:, 1:] == 2)).float()
    
    # Calculate density in sliding windows
    density = F.avg_pool1d(
        cpg_positions.unsqueeze(1),
        kernel_size=min(100, seq_len // 2),
        stride=1,
        padding=50
    )
    
    # Pad to match sequence length
    if density.shape[-1] < seq_len:
        padding = seq_len - density.shape[-1]
        density = F.pad(density, (0, padding))
    elif density.shape[-1] > seq_len:
        density = density[:, :, :seq_len]
    
    return density.squeeze(1)

# Testing and validation functions
def test_mutation_models():
    """Test the mutation models with synthetic data"""
    batch_size = 4
    seq_len = 200
    d_model = 512
    
    # Create synthetic data
    sequences = torch.randint(0, 4, (batch_size, seq_len))
    species_info = {'divergence_time': 150.0}
    
    # Create mutation context
    context = create_mutation_context(sequences, species_info)
    
    # Test advanced mutation model
    mutation_model = AdvancedMutationModel(d_model)
    rate_matrices = mutation_model(context)
    
    print(f"Rate matrices shape: {rate_matrices.shape}")
    print(f"Sample rate matrix:\n{rate_matrices[0, 0]}")
    
    # Test transition matrix generation
    transition_gen = TransitionMatrixGenerator()
    times = torch.tensor([10.0, 50.0, 100.0, 200.0])
    transitions = transition_gen(rate_matrices[:, 0], times)
    
    print(f"Transition matrices shape: {transitions.shape}")
    print(f"Sample transition matrix:\n{transitions[0]}")
    
    # Check that probabilities sum to 1
    row_sums = transitions.sum(dim=-1)
    print(f"Row sums (should be ~1.0): {row_sums[0]}")
    
    return True

if __name__ == "__main__":
    test_mutation_models()
