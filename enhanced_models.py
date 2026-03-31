"""
Enhanced Neural Network Models for Dinosaur DNA Reconstruction
Features: DNABERT integration, Multi-head attention, Variable-length sequences,
Multi-species phylogenetic context, Enhanced mutation modeling
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from transformers import (
        AutoTokenizer, AutoModel, AutoConfig,
        BertConfig, BertModel, BertTokenizer
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    print("Warning: Transformers library not available. Using basic implementation.")
    TRANSFORMERS_AVAILABLE = False
    
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import math
from enhanced_config import config

class DNABERTTokenizer:
    """Custom tokenizer for DNA sequences with k-mer encoding."""
    
    def __init__(self, kmer_size: int = 6):
        self.kmer_size = kmer_size
        self.vocab = self._create_vocabulary()
        self.token_to_id = {token: i for i, token in enumerate(self.vocab)}
        self.id_to_token = {i: token for token, i in self.token_to_id.items()}
        
    def _create_vocabulary(self) -> List[str]:
        """Create k-mer vocabulary."""
        bases = ['A', 'T', 'G', 'C', 'N']  # Include N for unknown bases
        
        # Generate all possible k-mers
        vocab = ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]']
        
        def generate_kmers(k):
            if k == 1:
                return bases
            smaller_kmers = generate_kmers(k - 1)
            return [kmer + base for kmer in smaller_kmers for base in bases]
        
        vocab.extend(generate_kmers(self.kmer_size))
        return vocab
    
    def encode(self, sequence: str, max_length: int = 512) -> Dict[str, torch.Tensor]:
        """Encode DNA sequence to k-mer tokens."""
        sequence = sequence.upper().replace('U', 'T')  # RNA to DNA
        
        # Generate k-mers
        kmers = []
        for i in range(len(sequence) - self.kmer_size + 1):
            kmer = sequence[i:i + self.kmer_size]
            kmers.append(kmer)
        
        # Convert to token IDs
        token_ids = [self.token_to_id.get('[CLS]', 1)]  # Start token
        for kmer in kmers[:max_length - 2]:
            token_ids.append(self.token_to_id.get(kmer, 1))  # UNK for unknown k-mers
        token_ids.append(self.token_to_id.get('[SEP]', 3))  # End token
        
        # Pad to max length
        attention_mask = [1] * len(token_ids)
        while len(token_ids) < max_length:
            token_ids.append(0)  # PAD token
            attention_mask.append(0)
        
        return {
            'input_ids': torch.tensor(token_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
        }

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for variable-length sequences."""
    
    def __init__(self, d_model: int, max_seq_length: int = 2048):
        super().__init__()
        
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:x.size(0), :]

class MultiHeadAttention(nn.Module):
    """Enhanced multi-head attention with DNA-specific modifications."""
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)
        
        # DNA-specific: learn attention bias for complementary bases
        self.complement_bias = nn.Parameter(torch.randn(4, 4) * 0.1)  # A,T,G,C
        
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, _ = query.size()
        
        # Linear transformations
        Q = self.w_q(query).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        # Apply mask if provided
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        context = torch.matmul(attention_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        return self.w_o(context)

class FeedForward(nn.Module):
    """Enhanced feed-forward network with GELU activation."""
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.gelu(self.linear1(x))))

class EnhancedTransformerBlock(nn.Module):
    """Transformer block with enhanced features for DNA modeling."""
    
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-attention with residual connection
        attn_output = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward with residual connection
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x

class SpeciesEmbedding(nn.Module):
    """Multi-species phylogenetic embeddings with hierarchical structure."""
    
    def __init__(self, num_species: int, embedding_dim: int, phylo_dim: int = 64):
        super().__init__()
        self.species_embedding = nn.Embedding(num_species, embedding_dim)
        self.phylo_embedding = nn.Embedding(100, phylo_dim)  # Divergence time bins
        
        # Hierarchical taxonomy embeddings
        self.kingdom_embedding = nn.Embedding(10, 32)  # Animalia, etc.
        self.phylum_embedding = nn.Embedding(20, 32)   # Chordata, etc.
        self.class_embedding = nn.Embedding(30, 32)    # Reptilia, Aves, etc.
        self.order_embedding = nn.Embedding(50, 32)    # Theropoda, etc.
        
        self.fusion_layer = nn.Linear(embedding_dim + phylo_dim + 4*32, embedding_dim)
        
    def forward(self, species_ids: torch.Tensor, divergence_times: torch.Tensor,
                taxonomy: Optional[Dict[str, torch.Tensor]] = None) -> torch.Tensor:
        # Basic species embeddings
        species_emb = self.species_embedding(species_ids)
        phylo_emb = self.phylo_embedding(divergence_times)
        
        # Handle missing taxonomy gracefully
        if taxonomy is None:
            # Use default taxonomy values
            batch_size = species_ids.size(0)
            device = species_ids.device
            taxonomy = {
                'kingdom': torch.zeros(batch_size, device=device, dtype=torch.long),
                'phylum': torch.zeros(batch_size, device=device, dtype=torch.long), 
                'class': torch.zeros(batch_size, device=device, dtype=torch.long),
                'order': torch.zeros(batch_size, device=device, dtype=torch.long),
            }
        
        # Hierarchical taxonomy embeddings
        kingdom_emb = self.kingdom_embedding(taxonomy['kingdom'])
        phylum_emb = self.phylum_embedding(taxonomy['phylum'])
        class_emb = self.class_embedding(taxonomy['class'])
        order_emb = self.order_embedding(taxonomy['order'])
        
        # Concatenate and fuse all embeddings
        combined = torch.cat([species_emb, phylo_emb, kingdom_emb, phylum_emb, 
                             class_emb, order_emb], dim=-1)
        
        return self.fusion_layer(combined)

class MutationProbabilityLayer(nn.Module):
    """Learn mutation probability biases with evolutionary constraints."""
    
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        
        # Learnable mutation matrices
        self.transition_matrix = nn.Parameter(torch.eye(4) * 0.9 + 
                                            torch.ones(4, 4) * 0.025)  # A,T,G,C
        
        # Context-dependent mutation rates (trinucleotide context)
        self.context_conv = nn.Conv1d(d_model, 64, kernel_size=3, padding=1)
        self.mutation_predictor = nn.Linear(64, 16)  # 4x4 mutation matrix
        
        # Ancient DNA damage patterns
        self.damage_predictor = nn.Linear(d_model, 3)  # C->T, G->A, depurination
        
        # Temperature and time dependency
        self.temp_time_layer = nn.Linear(2, 16)  # temperature, time inputs
        
    def forward(self, x: torch.Tensor, temperature: torch.Tensor = None,
                time: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        batch_size, seq_len, _ = x.size()
        
        # Context-dependent mutations
        context_features = F.relu(self.context_conv(x.transpose(1, 2))).transpose(1, 2)
        context_mutations = self.mutation_predictor(context_features)
        
        # Ancient DNA damage
        damage_rates = torch.sigmoid(self.damage_predictor(x))
        
        # Time and temperature effects
        if temperature is not None and time is not None:
            temp_time = torch.stack([temperature, time], dim=-1)
            temp_time_effect = torch.sigmoid(self.temp_time_layer(temp_time))
            temp_time_effect = temp_time_effect.unsqueeze(1).expand(-1, seq_len, -1)
        else:
            temp_time_effect = torch.ones(batch_size, seq_len, 16, device=x.device)
        
        return {
            'base_transitions': self.transition_matrix,
            'context_mutations': context_mutations,
            'damage_rates': damage_rates,
            'temp_time_effect': temp_time_effect,
        }

class EnhancedDinosaurDNAModel(nn.Module):
    """
    Enhanced Transformer model with DNABERT backbone for dinosaur DNA reconstruction.
    Features: Multi-head attention, variable-length sequences, multi-species support,
    phylogenetic context, and enhanced mutation modeling.
    """
    
    def __init__(self, vocab_size: int = 4096, d_model: int = 768, num_heads: int = 12,
                 num_layers: int = 12, d_ff: int = 3072, max_seq_length: int = 2048,
                 num_species: int = 50, dropout: float = 0.1):
        super().__init__()
        
        self.d_model = d_model
        self.max_seq_length = max_seq_length
        
        # DNABERT tokenizer and embeddings
        self.tokenizer = DNABERTTokenizer(kmer_size=config.DNABERT_KMER)
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        
        # Positional encoding for variable-length sequences
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)
        
        # Multi-species phylogenetic embeddings
        self.species_embedding = SpeciesEmbedding(num_species, d_model)
        
        # Enhanced transformer layers
        self.transformer_blocks = nn.ModuleList([
            EnhancedTransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        # Mutation probability modeling
        self.mutation_layer = MutationProbabilityLayer(d_model)
        
        # Output layers
        self.norm = nn.LayerNorm(d_model)
        self.output_projection = nn.Linear(d_model, vocab_size)
        
        # Confidence estimation
        self.confidence_head = nn.Linear(d_model, 1)
        
        # Evolutionary constraint head
        self.evolution_head = nn.Linear(d_model, 4)  # ATGC probabilities
        
        # Phylogenetic likelihood head
        self.phylo_head = nn.Linear(d_model, 1)
        
        self.dropout = nn.Dropout(dropout)
        
    def create_padding_mask(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Create padding mask for variable-length sequences."""
        batch_size, seq_len = x.size()
        mask = torch.arange(seq_len, device=x.device)[None, :] < lengths[:, None]
        return mask.unsqueeze(1).unsqueeze(2)  # [batch, 1, 1, seq_len]
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None,
                species_ids: torch.Tensor = None, divergence_times: torch.Tensor = None,
                taxonomy: Dict[str, torch.Tensor] = None,
                temperature: torch.Tensor = None, time: torch.Tensor = None,
                return_attention: bool = False) -> Dict[str, torch.Tensor]:
        
        batch_size, seq_len = input_ids.size()
        
        # Token embeddings
        x = self.token_embedding(input_ids)  # [batch, seq, d_model]
        x = x * math.sqrt(self.d_model)  # Scale embeddings
        
        # Add positional encoding
        x = self.positional_encoding(x)
        
        # Add species/phylogenetic context if provided
        if species_ids is not None:
            species_emb = self.species_embedding(species_ids, divergence_times, taxonomy)
            species_emb = species_emb.unsqueeze(1).expand(-1, seq_len, -1)
            x = x + species_emb
        
        x = self.dropout(x)
        
        # Apply transformer layers
        attention_weights = []
        for transformer in self.transformer_blocks:
            x = transformer(x, attention_mask)
            if return_attention:
                # Note: This is a simplified version - in practice you'd need to
                # modify the transformer blocks to return attention weights
                attention_weights.append(None)
        
        # Apply layer normalization
        x = self.norm(x)
        
        # Get mutation probabilities
        mutation_info = self.mutation_layer(x, temperature, time)
        
        # Generate outputs
        outputs = {
            'hidden_states': x,
            'logits': self.output_projection(x),
            'confidence': torch.sigmoid(self.confidence_head(x)),
            'evolution_probs': F.softmax(self.evolution_head(x), dim=-1),
            'phylo_likelihood': self.phylo_head(x),
            'mutation_info': mutation_info,
        }
        
        if return_attention:
            outputs['attention_weights'] = attention_weights
        
        return outputs
    
    def encode_sequence(self, sequence: str) -> Dict[str, torch.Tensor]:
        """Encode a DNA sequence for input to the model."""
        return self.tokenizer.encode(sequence, self.max_seq_length)
    
    def decode_sequence(self, token_ids: torch.Tensor) -> str:
        """Decode token IDs back to DNA sequence."""
        tokens = [self.tokenizer.id_to_token.get(int(id_), '[UNK]') 
                 for id_ in token_ids]
        
        # Remove special tokens and reconstruct sequence
        sequence = ""
        for token in tokens:
            if token not in ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]']:
                if len(sequence) == 0:
                    sequence = token
                else:
                    sequence += token[-1]  # Add last character for k-mer overlap
        
        return sequence

class HybridDinosaurModel(nn.Module):
    """
    Hybrid model combining enhanced transformer with Markov chains
    and evolutionary constraints for robust DNA reconstruction.
    """
    
    def __init__(self, transformer_config: dict, markov_order: int = 3):
        super().__init__()
        
        self.transformer = EnhancedDinosaurDNAModel(**transformer_config)
        self.markov_order = markov_order
        
        # Markov chain transition matrices
        self.markov_transitions = nn.ModuleDict({
            f'order_{i}': nn.Linear(4**i, 4) for i in range(1, markov_order + 1)
        })
        
        # Fusion layer to combine transformer and Markov predictions
        self.fusion_layer = nn.Linear(8, 4)  # Transformer + Markov -> ATGC
        
        # Evolutionary constraint layers
        self.gc_content_predictor = nn.Linear(768, 1)
        self.codon_bias_predictor = nn.Linear(768, 64)  # 64 codons
        
    def get_markov_context(self, sequence: torch.Tensor, position: int,
                          order: int) -> torch.Tensor:
        """Extract Markov chain context for given position and order."""
        if position < order:
            # Pad with zeros for beginning of sequence
            context = torch.zeros(4**order, device=sequence.device)
            context[0] = 1  # Default to 'A' context
            return context
        
        # Extract k-mer context
        context_seq = sequence[position-order:position]
        context_index = 0
        for i, base in enumerate(context_seq):
            context_index += int(base) * (4 ** (order - i - 1))
        
        context_vector = torch.zeros(4**order, device=sequence.device)
        context_vector[context_index] = 1
        return context_vector
    
    def forward(self, input_ids: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        # Get transformer outputs
        transformer_outputs = self.transformer(input_ids, **kwargs)
        transformer_logits = transformer_outputs['logits']
        hidden_states = transformer_outputs['hidden_states']
        
        batch_size, seq_len, vocab_size = transformer_logits.size()
        
        # Compute Markov predictions for each position
        markov_logits = torch.zeros(batch_size, seq_len, 4, device=input_ids.device)
        
        for batch_idx in range(batch_size):
            for pos in range(seq_len):
                # Combine predictions from different Markov orders
                markov_pred = torch.zeros(4, device=input_ids.device)
                total_weight = 0
                
                for order in range(1, self.markov_order + 1):
                    if pos >= order:
                        context = self.get_markov_context(input_ids[batch_idx], pos, order)
                        order_pred = self.markov_transitions[f'order_{order}'](context)
                        weight = 1.0 / order  # Higher order gets lower weight
                        markov_pred += weight * F.softmax(order_pred, dim=-1)
                        total_weight += weight
                
                if total_weight > 0:
                    markov_logits[batch_idx, pos] = markov_pred / total_weight
        
        # Combine transformer and Markov predictions
        # Note: This assumes transformer outputs 4-class logits for ATGC
        transformer_probs = F.softmax(transformer_logits[:, :, :4], dim=-1)
        combined_input = torch.cat([transformer_probs, markov_logits], dim=-1)
        fused_logits = self.fusion_layer(combined_input)
        
        # Add evolutionary constraints
        gc_content = torch.sigmoid(self.gc_content_predictor(hidden_states))
        codon_bias = F.softmax(self.codon_bias_predictor(hidden_states), dim=-1)
        
        # Update outputs
        transformer_outputs.update({
            'fused_logits': fused_logits,
            'markov_logits': markov_logits,
            'gc_content': gc_content,
            'codon_bias': codon_bias,
        })
        
        return transformer_outputs

def create_model(model_type: str = 'hybrid') -> nn.Module:
    """Factory function to create the appropriate model."""
    
    if model_type == 'transformer':
        return EnhancedDinosaurDNAModel(**config.TRANSFORMER_CONFIG)
    
    elif model_type == 'hybrid':
        return HybridDinosaurModel(
            transformer_config=config.TRANSFORMER_CONFIG,
            markov_order=3
        )
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")

if __name__ == "__main__":
    # Test model creation
    model = create_model('hybrid')
    print(f"Created hybrid model with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Test forward pass
    dummy_input = torch.randint(0, 1000, (2, 100))  # Batch size 2, sequence length 100
    dummy_mask = torch.ones(2, 1, 1, 100)
    
    with torch.no_grad():
        outputs = model(dummy_input, attention_mask=dummy_mask)
        print(f"Output shapes:")
        for key, value in outputs.items():
            if isinstance(value, torch.Tensor):
                print(f"  {key}: {value.shape}")
