"""
Enhanced Configuration for Dinosaur DNA Reconstruction System
Optimized for Linux/WSL environment with DNABERT integration
"""

import os
import torch
from pathlib import Path

class EnhancedConfig:
    # =====================================================
    # SYSTEM CONFIGURATION (WSL/Linux optimized)
    # =====================================================
    
    # Base paths - Use CURRENT directory structure (no new folders!)
    BASE_DIR = Path("/mnt/d/MY WORK/Dinosaur DNA Reconstruction")  # Your actual nested folder path
    DATA_DIR = BASE_DIR / "data"
    MODEL_DIR = BASE_DIR / "models" 
    OUTPUT_DIR = BASE_DIR / "outputs"
    CACHE_DIR = BASE_DIR / "cache"
    LOG_DIR = BASE_DIR / "logs"
    
    # Create directories if they don't exist
    for directory in [BASE_DIR, DATA_DIR, MODEL_DIR, OUTPUT_DIR, CACHE_DIR, LOG_DIR]:
        directory.mkdir(parents=True, exist_ok=True)
    
    # Device configuration - auto-detect GPU/CPU
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = min(4, os.cpu_count() or 1)  # Reduced for WSL
    
    # Memory management for WSL (conservative settings)
    MAX_MEMORY_GB = 6  # Conservative for 8GB systems
    BATCH_SIZE_MULTIPLIER = 1  # Start with 1x batch size
    
    # =====================================================
    # ENHANCED MODEL ARCHITECTURE
    # =====================================================
    
    # DNABERT Configuration
    DNABERT_MODEL = "zhihan1996/DNABERT-2-117M"  # Pre-trained DNA model
    DNABERT_MAX_LENGTH = 512
    DNABERT_KMER = 6  # k-mer size for DNA tokenization
    
    # Multi-Head Attention Enhanced Transformer
    TRANSFORMER_CONFIG = {
        'vocab_size': 4096,  # Extended for DNABERT tokens
        'hidden_size': 768,  # Match DNABERT dimensions
        'num_attention_heads': 12,  # Multi-head attention
        'num_hidden_layers': 12,
        'intermediate_size': 3072,  # Feed-forward layer size
        'hidden_dropout_prob': 0.1,
        'attention_probs_dropout_prob': 0.1,
        'max_position_embeddings': 1024,  # Variable length sequences
        'type_vocab_size': 16,  # Species type embeddings
        'initializer_range': 0.02,
        'layer_norm_eps': 1e-12,
        'gradient_checkpointing': True,  # Memory optimization
    }
    
    # Enhanced Multi-Species Architecture
    SPECIES_CONFIG = {
        'num_species': 50,  # Support for 50 different species
        'species_embedding_dim': 128,
        'phylogenetic_embedding_dim': 64,
        'divergence_time_bins': 20,  # Discrete time periods
        'use_hierarchical_species': True,
    }
    
    # Variable Length Sequence Handling
    SEQUENCE_CONFIG = {
        'min_length': 100,
        'max_length': 2048,  # Support very long sequences
        'padding_strategy': 'max_length',
        'truncation_strategy': 'longest_first',
        'dynamic_padding': True,
        'use_attention_mask': True,
    }
    
    # =====================================================
    # EVOLUTIONARY AND MUTATION MODELS
    # =====================================================
    
    # Enhanced Mutation Probability Bias
    MUTATION_CONFIG = {
        'base_mutation_rates': {
            'A->G': 0.25,  # Transition (high probability)
            'G->A': 0.25,
            'C->T': 0.25,
            'T->C': 0.25,
            'A->C': 0.05,  # Transversion (low probability)
            'A->T': 0.05,
            'C->G': 0.05,
            'G->C': 0.05,
            'G->T': 0.05,
            'T->G': 0.05,
            'C->A': 0.05,
            'T->A': 0.05,
        },
        'context_dependent': True,
        'trinucleotide_context': True,  # Consider 3-mer context
        'cpg_island_bias': 2.0,  # CpG sites mutate faster
        'ancient_dna_damage': {
            'c_to_t_rate': 0.3,  # Cytosine deamination
            'g_to_a_rate': 0.15,  # Guanine oxidation
            'depurination_rate': 0.1,  # A/G loss
        }
    }
    
    # Evolution Constraints
    EVOLUTION_CONFIG = {
        'molecular_clock_rate': 1e-9,  # Mutations per site per year
        'generation_time': 20,  # Years per generation
        'effective_population_size': 10000,
        'selection_coefficients': {
            'synonymous': 0.0,
            'nonsynonymous': -0.01,
            'nonsense': -0.1,
        },
        'codon_usage_bias': True,
        'gc_content_constraint': True,
        'phylogenetic_constraints': {
            'enforce_tree_consistency': True,
            'use_parsimony': True,
            'likelihood_threshold': -1000,
        }
    }
    
    # =====================================================
    # TRAINING CONFIGURATION
    # =====================================================
    
    TRAINING_CONFIG = {
        # Basic training parameters  
        'epochs': 50,  # Reduced for initial testing
        'batch_size': 8,  # Reduced for WSL memory constraints
        'learning_rate': 5e-5,  # Reduced for stability
        'weight_decay': 1e-5,
        'warmup_steps': 500,  # Reduced
        'gradient_accumulation_steps': 8,  # Effective batch size = 64
        
        # Advanced optimization
        'optimizer': 'adamw',
        'scheduler': 'cosine_with_restarts',
        'mixed_precision': True,  # FP16 for memory efficiency
        'gradient_clipping': 1.0,
        
        # Multi-task learning
        'loss_weights': {
            'reconstruction': 1.0,
            'evolutionary_consistency': 0.3,
            'phylogenetic_likelihood': 0.2,
            'mutation_prediction': 0.5,
            'confidence_estimation': 0.1,
        },
        
        # Regularization
        'dropout_rate': 0.1,
        'label_smoothing': 0.05,
        'data_augmentation': True,
        
        # Checkpointing and monitoring
        'save_every_n_steps': 500,
        'eval_every_n_steps': 250,
        'early_stopping_patience': 10,
        'use_wandb': True,
    }
    
    # =====================================================
    # DATA COLLECTION AND PROCESSING
    # =====================================================
    
    # NCBI Dataset Configuration
    NCBI_CONFIG = {
        'email': 'syslee320@gmail.com',  # CHANGE THIS TO YOUR EMAIL!
        'api_key': None,  # Optional but recommended
        'databases': ['nucleotide', 'genome', 'sra'],
        'max_sequences_per_species': 1000,
        'min_sequence_length': 500,
        'quality_threshold': 0.9,
        
        # Target species for training data
        'target_species': [
            # Modern birds (closest to theropods)
            'Gallus_gallus',  # Chicken
            'Anas_platyrhynchos',  # Duck  
            'Columba_livia',  # Pigeon
            'Taeniopygia_guttata',  # Finch
            
            # Ratites (primitive birds)
            'Struthio_camelus',  # Ostrich
            'Dromaius_novaehollandiae',  # Emu
            
            # Crocodilians (archosaur outgroup)
            'Alligator_mississippiensis',  # Alligator
            'Crocodylus_porosus',  # Saltwater crocodile
            
            # Other reptiles for phylogenetic context
            'Python_bivittatus',  # Python
            'Anolis_carolinensis',  # Anole lizard
        ],
        
        # Data processing parameters
        'chunk_size': 500000,  # Reduced for memory efficiency
        'parallel_downloads': 2,  # Reduced for WSL
        'retry_attempts': 3,
        'cache_sequences': True,
    }
    
    # =====================================================
    # EVALUATION METRICS
    # =====================================================
    
    EVALUATION_CONFIG = {
        'metrics': [
            'sequence_identity',
            'position_wise_accuracy', 
            'confidence_calibration',
            'evolutionary_consistency',
            'phylogenetic_likelihood',
            'mutation_rate_accuracy',
            'gc_content_preservation',
            'codon_usage_similarity',
        ],
        'confidence_thresholds': [0.5, 0.7, 0.8, 0.9, 0.95],
        'bootstrap_samples': 1000,
        'cross_validation_folds': 5,
        'statistical_tests': ['permutation', 'bootstrap', 'wilcoxon'],
    }
    
    # =====================================================
    # PHYLOGENETIC TREE STRUCTURE
    # =====================================================
    
    # Simplified evolutionary tree with divergence times (million years ago)
    PHYLOGENETIC_TREE = {
        'root': {
            'name': 'Archosaur_ancestor',
            'time': 250,  # Permian-Triassic
            'children': {
                'Crocodylomorpha': {
                    'time': 220,
                    'children': {
                        'Alligator_mississippiensis': {'time': 0},
                        'Crocodylus_porosus': {'time': 0},
                        'Gavialis_gangeticus': {'time': 0},
                    }
                },
                'Avemetatarsalia': {
                    'time': 240,
                    'children': {
                        'Dinosauria': {
                            'time': 230,
                            'children': {
                                'Theropoda': {
                                    'time': 200,
                                    'children': {
                                        'Tyrannosaurus_rex': {'time': 68},
                                        'Velociraptor_mongoliensis': {'time': 75},
                                        'Aves': {
                                            'time': 150,
                                            'children': {
                                                'Gallus_gallus': {'time': 0},
                                                'Struthio_camelus': {'time': 0},
                                                'Anas_platyrhynchos': {'time': 0},
                                            }
                                        }
                                    }
                                },
                                'Sauropoda': {
                                    'time': 210,
                                    'children': {
                                        'Brontosaurus_excelsus': {'time': 150},
                                        'Diplodocus_longus': {'time': 155},
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    
    # =====================================================
    # WEB INTERFACE CONFIGURATION
    # =====================================================
    
    WEB_CONFIG = {
        'streamlit': {
            'port': 8501,
            'host': '0.0.0.0',  # Allow access from Windows host
            'theme': 'dark',
            'max_upload_size': 200,  # MB
        },
        'fastapi': {
            'port': 8000,
            'host': '0.0.0.0',
            'docs_url': '/docs',
            'redoc_url': '/redoc',
        },
        'features': {
            'file_upload': True,
            'batch_processing': True,
            'real_time_visualization': True,
            'export_formats': ['fasta', 'json', 'csv', 'xlsx', 'genbank'],
            'api_rate_limiting': True,
        }
    }
    
    # =====================================================
    # LOGGING AND MONITORING
    # =====================================================
    
    LOGGING_CONFIG = {
        'level': 'INFO',
        'format': '%(asctime)s | %(levelname)s | %(module)s | %(message)s',
        'handlers': ['console', 'file', 'rotating_file'],
        'log_file': LOG_DIR / 'dinosaur_dna.log',
        'max_file_size': '100MB',
        'backup_count': 5,
        'capture_warnings': True,
    }

# Create global config instance
config = EnhancedConfig()

# Validation function
def validate_config():
    """Validate configuration settings."""
    assert config.BASE_DIR.exists(), f"Base directory {config.BASE_DIR} not accessible"
    assert config.DEVICE in ['cuda', 'cpu'], "Invalid device specification"
    assert 0 < config.TRAINING_CONFIG['batch_size'] <= 128, "Invalid batch size"
    print(f"✓ Configuration validated - Using device: {config.DEVICE}")
    print(f"✓ Base directory: {config.BASE_DIR}")
    return True

if __name__ == "__main__":
    validate_config()