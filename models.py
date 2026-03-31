"""
Neural Network Models for Dinosaur DNA Reconstruction
Implements transformer-based architecture with phylogenetic and evolutionary constraints
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from config import PROJECT_CONFIG

@dataclass
class ModelOutput:
    """Output container for model predictions"""
    reconstructed_sequence: torch.Tensor
    confidence_scores: torch.Tensor
    mutation_probabilities: torch.Tensor
    phylogenetic_embedding: torch.Tensor
    attention_weights: torch.Tensor

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer with evolutionary time dimension"""
    
    def __init__(self, d_model: int, max_len: int = 10000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class PhylogeneticEmbedding(nn.Module):
    """Embedding layer that incorporates phylogenetic relationships"""
    
    def __init__(self, embedding_dim: int, num_species: int = 50):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.species_embedding = nn.Embedding(num_species, embedding_dim)
        self.time_embedding = nn.Linear(1, embedding_dim)
        self.distance_embedding = nn.Linear(1, embedding_dim)
        
        # Learnable phylogenetic tree structure
        self.tree_attention = nn.MultiheadAttention(embedding_dim, 8)
        
    def forward(self, species_ids: torch.Tensor, divergence_times: torch.Tensor, 
                phylo_distances: torch.Tensor) -> torch.Tensor:
        """
        Args:
            species_ids: Species identifier (batch_size,)
            divergence_times: Time since divergence in Mya (batch_size,)
            phylo_distances: Phylogenetic distances (batch_size,)
        """
        batch_size = species_ids.size(0)
        
        # Species-specific embeddings
        species_emb = self.species_embedding(species_ids)
        
        # Time-dependent evolution
        time_emb = self.time_embedding(divergence_times.unsqueeze(-1))
        
        # Phylogenetic distance encoding
        dist_emb = self.distance_embedding(phylo_distances.unsqueeze(-1))
        
        # Combine embeddings
        phylo_emb = species_emb + time_emb + dist_emb
        
        # Self-attention to capture phylogenetic relationships
        phylo_emb = phylo_emb.unsqueeze(1)  # Add sequence dimension
        phylo_emb, _ = self.tree_attention(phylo_emb, phylo_emb, phylo_emb)
        
        return phylo_emb.squeeze(1)

class EvolutionaryConstraintLayer(nn.Module):
    """Layer that enforces evolutionary constraints on mutations"""
    
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        
        # Learnable mutation matrices for different evolutionary contexts
        self.transition_matrix = nn.Parameter(torch.eye(4))  # A, T, G, C
        self.transversion_matrix = nn.Parameter(torch.eye(4))
        
        # Context-dependent mutation rates
        self.mutation_rate_predictor = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 16),  # 4x4 mutation matrix
            nn.Softmax(dim=-1)
        )
        
        # Initialize based on known biological constraints
        self._initialize_mutation_matrices()
    
    def _initialize_mutation_matrices(self):
        """Initialize mutation matrices with biological priors"""
        constraints = PROJECT_CONFIG["evolutionary_constraints"]["mutation_probabilities"]
        
        # Transition bias (Ts/Tv ratio ~2.0)
        with torch.no_grad():
            # A, T, G, C indices: 0, 1, 2, 3
            transition_rates = torch.tensor([
                [0.0, 0.07, 0.15, 0.06],   # A -> T,G,C
                [0.07, 0.0, 0.06, 0.15],   # T -> A,G,C  
                [0.15, 0.06, 0.0, 0.06],   # G -> A,T,C
                [0.06, 0.15, 0.06, 0.0]    # C -> A,T,G
            ])
            
            self.transition_matrix.data = transition_rates
    
    def forward(self, sequence_repr: torch.Tensor, phylo_context: torch.Tensor) -> torch.Tensor:
        """
        Apply evolutionary constraints to sequence representation
        
        Args:
            sequence_repr: Sequence representation (batch, seq_len, d_model)
            phylo_context: Phylogenetic context (batch, d_model)
        """
        batch_size, seq_len, d_model = sequence_repr.shape
        
        # Predict context-dependent mutation rates
        phylo_expanded = phylo_context.unsqueeze(1).expand(-1, seq_len, -1)
        mutation_context = torch.cat([sequence_repr, phylo_expanded], dim=-1)
        
        # Get mutation probabilities for each position
        mutation_rates = self.mutation_rate_predictor(mutation_context)
        mutation_rates = mutation_rates.view(batch_size, seq_len, 4, 4)
        
        # Apply transition/transversion bias
        constrained_rates = (mutation_rates * 0.7 + 
                           self.transition_matrix.unsqueeze(0).unsqueeze(0) * 0.3)
        
        return sequence_repr, constrained_rates

class DinosaurDNATransformer(nn.Module):
    """Main transformer model for dinosaur DNA reconstruction"""
    
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        model_config = config["model"]["transformer"]
        
        self.d_model = model_config["d_model"]
        self.n_heads = model_config["n_heads"]
        self.n_layers = model_config["n_layers"]
        self.vocab_size = model_config["vocab_size"]
        
        # Embedding layers
        self.token_embedding = nn.Embedding(self.vocab_size, self.d_model)
        self.positional_encoding = PositionalEncoding(self.d_model, 
                                                    model_config["max_seq_length"])
        
        # Phylogenetic components
        self.phylo_embedding = PhylogeneticEmbedding(
            config["model"]["phylogenetic"]["embedding_dim"]
        )
        
        # Project phylogenetic embedding to model dimension
        self.phylo_projection = nn.Linear(
            config["model"]["phylogenetic"]["embedding_dim"], 
            self.d_model
        )
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.n_heads,
            dim_feedforward=model_config["d_ff"],
            dropout=model_config["dropout"],
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=self.n_layers
        )
        
        # Evolutionary constraint layer
        self.evolution_layer = EvolutionaryConstraintLayer(self.d_model)
        
        # Output layers
        self.output_projection = nn.Linear(self.d_model, self.vocab_size)
        self.confidence_head = nn.Linear(self.d_model, 1)
        
        # Reconstruction head for missing/damaged regions
        self.reconstruction_head = nn.Sequential(
            nn.Linear(self.d_model * 2, self.d_model),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.d_model, self.vocab_size)
        )
        
    def forward(self, input_ids: torch.Tensor, damaged_mask: torch.Tensor,
                species_ids: torch.Tensor, divergence_times: torch.Tensor,
                phylo_distances: torch.Tensor) -> ModelOutput:
        """
        Forward pass of the DinosaurDNA transformer
        
        Args:
            input_ids: Tokenized DNA sequence (batch_size, seq_len)
            damaged_mask: Mask indicating damaged/missing positions (batch_size, seq_len)
            species_ids: Species identifiers (batch_size,)
            divergence_times: Evolutionary time distances (batch_size,)
            phylo_distances: Phylogenetic distances (batch_size,)
        """
        batch_size, seq_len = input_ids.shape
        
        # Token embeddings
        token_embs = self.token_embedding(input_ids)  # (batch, seq_len, d_model)
        
        # Add positional encoding
        sequence_repr = self.positional_encoding(token_embs.transpose(0, 1)).transpose(0, 1)
        
        # Phylogenetic context
        phylo_context = self.phylo_embedding(species_ids, divergence_times, phylo_distances)
        phylo_context = self.phylo_projection(phylo_context)  # (batch, d_model)
        
        # Add phylogenetic context to sequence representation
        phylo_expanded = phylo_context.unsqueeze(1).expand(-1, seq_len, -1)
        sequence_repr = sequence_repr + phylo_expanded
        
        # Transformer encoding
        attention_mask = self._create_attention_mask(damaged_mask)
        encoded = self.transformer_encoder(sequence_repr, src_key_padding_mask=attention_mask)
        
        # Apply evolutionary constraints
        constrained_repr, mutation_probs = self.evolution_layer(encoded, phylo_context)
        
        # Generate outputs
        output_logits = self.output_projection(constrained_repr)
        confidence_scores = torch.sigmoid(self.confidence_head(constrained_repr))
        
        # Reconstruction for damaged regions
        damaged_positions = damaged_mask.bool()
        if damaged_positions.any():
            # Context for reconstruction
            context_repr = torch.cat([
                constrained_repr, 
                phylo_expanded
            ], dim=-1)
            
            reconstruction_logits = self.reconstruction_head(context_repr)
            
            # Blend original and reconstructed predictions
            final_logits = torch.where(
                damaged_positions.unsqueeze(-1), 
                reconstruction_logits, 
                output_logits
            )
        else:
            final_logits = output_logits
        
        return ModelOutput(
            reconstructed_sequence=final_logits,
            confidence_scores=confidence_scores.squeeze(-1),
            mutation_probabilities=mutation_probs,
            phylogenetic_embedding=phylo_context,
            attention_weights=None  # Would need to extract from transformer
        )
    
    def _create_attention_mask(self, damaged_mask: torch.Tensor) -> torch.Tensor:
        """Create attention mask for damaged positions"""
        # Don't attend to heavily damaged positions
        return damaged_mask > 0.8

class MarkovChainEvolutionModel(nn.Module):
    """Markov chain model for evolutionary transitions"""
    
    def __init__(self, order: int = 3):
        super().__init__()
        self.order = order
        self.vocab_size = 4  # A, T, G, C
        
        # Context-dependent transition matrices
        self.context_size = self.vocab_size ** order
        self.transition_matrices = nn.Parameter(
            torch.zeros(self.context_size, self.vocab_size, self.vocab_size)
        )
        
        # Time-dependent evolution rates
        self.rate_predictor = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, self.context_size),
            nn.Softplus()
        )
        
        self._initialize_transitions()
    
    def _initialize_transitions(self):
        """Initialize with biologically realistic transition probabilities"""
        with torch.no_grad():
            for i in range(self.context_size):
                # Add small random noise to break symmetry
                self.transition_matrices[i] = torch.eye(self.vocab_size) * 0.7
                self.transition_matrices[i] += torch.rand(self.vocab_size, self.vocab_size) * 0.1
                
                # Normalize
                self.transition_matrices[i] = F.softmax(self.transition_matrices[i], dim=-1)
    
    def forward(self, sequence: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        """
        Predict evolution of sequence over time
        
        Args:
            sequence: Input DNA sequence (batch_size, seq_len)
            time: Evolutionary time in millions of years (batch_size,)
        """
        batch_size, seq_len = sequence.shape
        
        # Get time-dependent rates
        rates = self.rate_predictor(time.unsqueeze(-1))  # (batch, context_size)
        
        # Apply Markov transitions
        evolved_logits = torch.zeros(batch_size, seq_len, self.vocab_size, device=sequence.device)
        
        for i in range(self.order, seq_len):
            # Get context (previous nucleotides)
            context = sequence[:, i-self.order:i]  # (batch, order)
            
            # Convert context to index
            context_indices = self._sequence_to_context_index(context)  # (batch,)
            
            # Get appropriate transition matrix for each sample
            for b in range(batch_size):
                ctx_idx = context_indices[b]
                rate = rates[b, ctx_idx]
                
                # Apply transition matrix scaled by rate and time
                transition_matrix = self.transition_matrices[ctx_idx]
                time_scaled_matrix = torch.matrix_power(
                    transition_matrix, 
                    int(rate * time[b] + 1)
                )
                
                current_nucleotide = sequence[b, i-1]
                evolved_logits[b, i] = time_scaled_matrix[current_nucleotide]
        
        return evolved_logits
    
    def _sequence_to_context_index(self, context: torch.Tensor) -> torch.Tensor:
        """Convert sequence context to single index"""
        batch_size, order = context.shape
        indices = torch.zeros(batch_size, dtype=torch.long, device=context.device)
        
        for i in range(order):
            indices += context[:, i] * (self.vocab_size ** (order - i - 1))
        
        return indices

class HybridDinosaurModel(nn.Module):
    """Hybrid model combining transformer and Markov chain approaches"""
    
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        
        # Main transformer model
        self.transformer = DinosaurDNATransformer(config)
        
        # Markov evolution model
        self.markov_model = MarkovChainEvolutionModel(order=3)
        
        # Fusion layer
        self.fusion_layer = nn.Sequential(
            nn.Linear(config["model"]["transformer"]["vocab_size"] * 2, 
                     config["model"]["transformer"]["vocab_size"]),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(config["model"]["transformer"]["vocab_size"], 
                     config["model"]["transformer"]["vocab_size"])
        )
        
        # Uncertainty estimation
        self.uncertainty_head = nn.Linear(
            config["model"]["transformer"]["d_model"], 
            config["model"]["transformer"]["vocab_size"]
        )
    
    def forward(self, input_ids: torch.Tensor, damaged_mask: torch.Tensor,
                species_ids: torch.Tensor, divergence_times: torch.Tensor,
                phylo_distances: torch.Tensor) -> ModelOutput:
        """Forward pass combining transformer and Markov predictions"""
        
        # Transformer prediction
        transformer_output = self.transformer(
            input_ids, damaged_mask, species_ids, 
            divergence_times, phylo_distances
        )
        
        # Markov chain prediction
        markov_logits = self.markov_model(input_ids, divergence_times)
        
        # Combine predictions
        combined_input = torch.cat([
            transformer_output.reconstructed_sequence,
            markov_logits
        ], dim=-1)
        
        fused_logits = self.fusion_layer(combined_input)
        
        # Update output
        return ModelOutput(
            reconstructed_sequence=fused_logits,
            confidence_scores=transformer_output.confidence_scores,
            mutation_probabilities=transformer_output.mutation_probabilities,
            phylogenetic_embedding=transformer_output.phylogenetic_embedding,
            attention_weights=transformer_output.attention_weights
        )

def create_model(config: Dict, model_type: str = "hybrid") -> nn.Module:
    """Factory function to create models"""
    if model_type == "transformer":
        return DinosaurDNATransformer(config)
    elif model_type == "markov":
        return MarkovChainEvolutionModel()
    elif model_type == "hybrid":
        return HybridDinosaurModel(config)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
