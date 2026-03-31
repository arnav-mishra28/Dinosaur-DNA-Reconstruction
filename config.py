# Dinosaur DNA Reconstruction Project Configuration
# Based on scientific research on archosaur genome evolution and paleogenomics

PROJECT_CONFIG = {
    "project_name": "DinosaurDNA_Reconstruction",
    "version": "1.0.0",
    "description": "ML system for reconstructing dinosaur DNA using modern bird and archosaur genomic data",
    
    # Model Architecture Parameters
    "model": {
        "transformer": {
            "d_model": 512,
            "n_heads": 8,
            "n_layers": 12,
            "d_ff": 2048,
            "dropout": 0.1,
            "max_seq_length": 8192,
            "vocab_size": 16  # A, T, G, C, N, + special tokens
        },
        "phylogenetic": {
            "embedding_dim": 256,
            "tree_depth": 20,
            "divergence_time_encoding": True
        },
        "evolutionary": {
            "mutation_matrix_size": 4,  # A, T, G, C
            "use_transition_bias": True,
            "time_dependent_rates": True
        }
    },
    
    # Training Parameters
    "training": {
        "batch_size": 32,
        "learning_rate": 1e-4,
        "weight_decay": 1e-5,
        "epochs": 100,
        "warmup_steps": 1000,
        "gradient_clip": 1.0,
        "validation_split": 0.2,
        "early_stopping_patience": 10
    },
    
    # Data Sources - Based on research findings
    "data_sources": {
        "avian_genomes": {
            "b10k_database": "https://b10k.genomics.cn/",
            "gigadb_avian": "http://gigadb.org/dataset/101000",
            "avianbase": "https://avianbase.sanger.ac.uk/",
            "ncbi_birds": "https://www.ncbi.nlm.nih.gov/datasets/genome/?taxon=8782"
        },
        "archosaur_genomes": {
            "crocodilian": [
                "Alligator mississippiensis",  # American Alligator
                "Crocodylus porosus",          # Saltwater Crocodile
                "Gavialis gangeticus"          # Indian Gharial
            ],
            "key_bird_species": [
                "Gallus gallus",               # Chicken (reference)
                "Taeniopygia guttata",         # Zebra Finch
                "Struthio camelus",            # Ostrich (paleognath)
                "Tinamus guttatus",            # Tinamou (paleognath)
                "Falco peregrinus",            # Peregrine Falcon
                "Corvus brachyrhynchos"        # American Crow
            ]
        }
    },
    
    # Evolutionary Constraints - Based on paleogenomics research
    "evolutionary_constraints": {
        "mutation_probabilities": {
            # Transition vs Transversion bias (Ts/Tv = ~2.0 in vertebrates)
            "transitions": {
                "A_to_G": 0.15,
                "G_to_A": 0.15,
                "C_to_T": 0.15,  # High due to cytosine deamination
                "T_to_C": 0.10
            },
            "transversions": {
                "A_to_C": 0.06,
                "A_to_T": 0.07,
                "G_to_C": 0.06,
                "G_to_T": 0.07,
                "C_to_A": 0.06,
                "C_to_G": 0.06,
                "T_to_A": 0.06,
                "T_to_G": 0.06
            }
        },
        "ancient_dna_damage": {
            # Patterns observed in ancient DNA
            "cytosine_deamination": 0.3,   # C->T at 5' ends
            "purine_loss": 0.2,            # A,G loss creating gaps
            "fragment_length": {
                "mean": 50,
                "std": 20,
                "min": 25,
                "max": 150
            }
        },
        "divergence_times": {
            # Million years ago (mya) - based on fossil and molecular data
            "archosaur_split": 240,        # Archosaur common ancestor
            "dino_crocodile_split": 240,   # Dinosaur-crocodile divergence
            "theropod_emergence": 200,     # Early theropod dinosaurs
            "bird_emergence": 150,         # Earliest known birds
            "modern_bird_radiation": 65    # Post-K-Pg boundary
        }
    },
    
    # Sequence Processing
    "sequence_processing": {
        "min_sequence_length": 100,
        "max_sequence_length": 10000,
        "sliding_window": 512,
        "overlap": 128,
        "quality_threshold": 0.8,
        "gap_threshold": 0.1
    },
    
    # File Paths
    "paths": {
    "data_dir": "./data",
    "raw_data": "./data/raw",
    "processed_data": "./data/processed",
    "models": "./models",
    "outputs": "./outputs",
    "logs": "./logs",
    "figures": "./figures"
}
}

# Genomic Features of Interest - Based on comparative genomics research
GENOMIC_FEATURES = {
    "conserved_regions": [
        "16S_rRNA", "12S_rRNA", "Cytochrome_b", "Cytochrome_oxidase_I",
        "NADH_dehydrogenase", "ATP_synthase"
    ],
    "developmental_genes": [
        "HOX_clusters", "SHH_pathway", "WNT_pathway", "FGF_pathway",
        "NOTCH_pathway", "BMP_pathway"
    ],
    "metabolic_pathways": [
        "glycolysis", "TCA_cycle", "oxidative_phosphorylation",
        "fatty_acid_metabolism", "amino_acid_metabolism"
    ],
    "structural_genes": [
        "collagen_genes", "keratin_genes", "bone_morphogenetic_proteins",
        "calcium_binding_proteins"
    ]
}

# Species-specific genome characteristics
GENOME_CHARACTERISTICS = {
    "typical_bird_genome": {
        "size_gb": 1.2,  # Gigabases
        "genes": 15000,
        "chromosomes": 80,  # Including microchromosomes
        "gc_content": 0.42
    },
    "crocodilian_genome": {
        "size_gb": 2.3,
        "genes": 23000,
        "chromosomes": 32,
        "gc_content": 0.44
    },
    "estimated_theropod": {
        "size_gb": 1.5,  # Between bird and crocodile
        "genes": 17000,
        "chromosomes": 60,
        "gc_content": 0.43
    }
}
