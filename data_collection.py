"""
Data Collection and Preprocessing Module
Handles downloading genomic data from various sources and preprocessing for ML models
"""

import os
import requests
import pandas as pd
import numpy as np
from Bio import SeqIO, Entrez, SeqUtils
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
import gzip
import json
from tqdm import tqdm
import logging
from typing import List, Dict, Tuple, Optional
from urllib.parse import urljoin
import time
import random

from config import PROJECT_CONFIG, GENOMIC_FEATURES

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GenomeDataCollector:
    """Collects genomic data from multiple sources"""
    
    def __init__(self, email: str = "researcher@example.com"):
        self.email = email
        Entrez.email = email
        self.data_dir = PROJECT_CONFIG["paths"]["raw_data"]
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Species list from config
        self.target_species = (
            PROJECT_CONFIG["data_sources"]["archosaur_genomes"]["crocodilian"] +
            PROJECT_CONFIG["data_sources"]["archosaur_genomes"]["key_bird_species"]
        )
    
    def download_ncbi_genome(self, species_name: str, retries: int = 3) -> Optional[str]:
        """Download genome data from NCBI"""
        try:
            # Search for genome assembly
            search_term = f"{species_name}[Organism] AND latest[filter]"
            search_handle = Entrez.esearch(db="assembly", term=search_term, retmax=1)
            search_results = Entrez.read(search_handle)
            search_handle.close()
            
            if not search_results["IdList"]:
                logger.warning(f"No genome found for {species_name}")
                return None
            
            # Get assembly details
            assembly_id = search_results["IdList"][0]
            summary_handle = Entrez.esummary(db="assembly", id=assembly_id)
            summary = Entrez.read(summary_handle, validate=False)
            summary_handle.close()
            
            # Download genome FASTA
            ftp_path = summary["DocumentSummarySet"]["DocumentSummary"][0]["FtpPath_RefSeq"]
            if not ftp_path:
                ftp_path = summary["DocumentSummarySet"]["DocumentSummary"][0]["FtpPath_GenBank"]
            
            if ftp_path:
                filename = os.path.basename(ftp_path) + "_genomic.fna.gz"
                download_url = ftp_path + "/" + filename
                
                # Download file
                output_path = os.path.join(self.data_dir, f"{species_name.replace(' ', '_')}_genome.fna.gz")
                
                if not os.path.exists(output_path):
                    logger.info(f"Downloading genome for {species_name}")
                    response = requests.get(download_url, stream=True)
                    response.raise_for_status()
                    
                    with open(output_path, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                
                return output_path
                
        except Exception as e:
            logger.error(f"Error downloading genome for {species_name}: {e}")
            if retries > 0:
                time.sleep(5)
                return self.download_ncbi_genome(species_name, retries - 1)
            return None
    
    def download_mitochondrial_genomes(self) -> Dict[str, str]:
        """Download mitochondrial genomes for target species"""
        mt_genomes = {}
        
        for species in self.target_species:
            try:
                # Search for mitochondrial genome
                search_term = f"{species}[Organism] AND mitochondrion[filter] AND complete[title]"
                search_handle = Entrez.esearch(db="nucleotide", term=search_term, retmax=1)
                search_results = Entrez.read(search_handle)
                search_handle.close()
                
                if search_results["IdList"]:
                    seq_id = search_results["IdList"][0]
                    
                    # Fetch sequence
                    fetch_handle = Entrez.efetch(db="nucleotide", id=seq_id, rettype="fasta")
                    sequence_record = SeqIO.read(fetch_handle, "fasta")
                    fetch_handle.close()
                    
                    # Save to file
                    output_path = os.path.join(self.data_dir, f"{species.replace(' ', '_')}_mitochondrion.fasta")
                    SeqIO.write(sequence_record, output_path, "fasta")
                    mt_genomes[species] = output_path
                    
                    logger.info(f"Downloaded mitochondrial genome for {species}")
                    time.sleep(1)  # Be nice to NCBI servers
                    
            except Exception as e:
                logger.error(f"Error downloading mitochondrial genome for {species}: {e}")
        
        return mt_genomes
    
    def create_synthetic_ancient_dna(self, modern_sequence: str, damage_rate: float = 0.3) -> str:
        """Simulate ancient DNA damage patterns"""
        sequence = list(modern_sequence.upper())
        damage_params = PROJECT_CONFIG["evolutionary_constraints"]["ancient_dna_damage"]
        
        for i, base in enumerate(sequence):
            if random.random() < damage_rate:
                if base == 'C' and random.random() < damage_params["cytosine_deamination"]:
                    # C->T deamination (especially at 5' ends)
                    if i < len(sequence) * 0.1 or i > len(sequence) * 0.9:
                        sequence[i] = 'T'
                elif base in ['A', 'G'] and random.random() < damage_params["purine_loss"]:
                    # Purine loss
                    sequence[i] = 'N'
        
        return ''.join(sequence)
    
    def fragment_sequences(self, sequence: str, mean_length: int = 50) -> List[str]:
        """Fragment sequences to simulate ancient DNA preservation"""
        fragments = []
        damage_params = PROJECT_CONFIG["evolutionary_constraints"]["ancient_dna_damage"]["fragment_length"]
        
        pos = 0
        while pos < len(sequence):
            fragment_length = max(
                damage_params["min"],
                min(
                    damage_params["max"],
                    int(np.random.normal(damage_params["mean"], damage_params["std"]))
                )
            )
            
            fragment = sequence[pos:pos + fragment_length]
            if len(fragment) >= damage_params["min"]:
                fragments.append(fragment)
            
            # Random gap between fragments
            pos += fragment_length + random.randint(10, 100)
        
        return fragments

class SequencePreprocessor:
    """Preprocesses sequences for machine learning"""
    
    def __init__(self):
        self.nucleotide_to_int = {'A': 0, 'T': 1, 'G': 2, 'C': 3, 'N': 4}
        self.int_to_nucleotide = {v: k for k, v in self.nucleotide_to_int.items()}
        self.special_tokens = {
            'PAD': 5, 'UNK': 6, 'MASK': 7, 'CLS': 8, 'SEP': 9,
            'DAMAGE': 10, 'MISSING': 11, 'UNCERTAIN': 12
        }
        
    def encode_sequence(self, sequence: str) -> List[int]:
        """Convert DNA sequence to integer encoding"""
        return [self.nucleotide_to_int.get(base, self.special_tokens['UNK']) 
                for base in sequence.upper()]
    
    def decode_sequence(self, encoded: List[int]) -> str:
        """Convert integer encoding back to DNA sequence"""
        return ''.join([self.int_to_nucleotide.get(token, 'N') for token in encoded])
    
    def create_sliding_windows(self, sequence: str, window_size: int = 512, overlap: int = 128) -> List[str]:
        """Create overlapping windows from sequence"""
        windows = []
        step = window_size - overlap
        
        for i in range(0, len(sequence) - window_size + 1, step):
            windows.append(sequence[i:i + window_size])
        
        # Add the last segment if it's long enough
        if len(sequence) % step > window_size // 2:
            windows.append(sequence[-window_size:])
        
        return windows
    
    def calculate_sequence_features(self, sequence: str) -> Dict[str, float]:
        """Calculate various sequence features"""
        sequence = sequence.upper()
        length = len(sequence)
        
        if length == 0:
            return {}
        
        features = {
            'length': length,
            'gc_content': (sequence.count('G') + sequence.count('C')) / length,
            'at_content': (sequence.count('A') + sequence.count('T')) / length,
            'n_content': sequence.count('N') / length,
            'complexity': self.calculate_complexity(sequence),
            'entropy': self.calculate_entropy(sequence)
        }
        
        # Dinucleotide frequencies
        dinucleotides = ['AA', 'AT', 'AG', 'AC', 'TA', 'TT', 'TG', 'TC',
                        'GA', 'GT', 'GG', 'GC', 'CA', 'CT', 'CG', 'CC']
        
        for dinuc in dinucleotides:
            features[f'{dinuc}_freq'] = sequence.count(dinuc) / max(1, length - 1)
        
        return features
    
    def calculate_complexity(self, sequence: str, window: int = 64) -> float:
        """Calculate linguistic complexity of sequence"""
        if len(sequence) < window:
            return 0.0
        
        complexities = []
        for i in range(len(sequence) - window + 1):
            subseq = sequence[i:i + window]
            vocab_size = len(set(subseq))
            max_vocab = min(4, len(subseq))  # A, T, G, C
            complexity = vocab_size / max_vocab if max_vocab > 0 else 0
            complexities.append(complexity)
        
        return np.mean(complexities)
    
    def calculate_entropy(self, sequence: str) -> float:
        """Calculate Shannon entropy of sequence"""
        if not sequence:
            return 0.0
        
        counts = {}
        for base in sequence:
            counts[base] = counts.get(base, 0) + 1
        
        length = len(sequence)
        entropy = 0.0
        
        for count in counts.values():
            if count > 0:
                p = count / length
                entropy -= p * np.log2(p)
        
        return entropy

class PhylogeneticDataProcessor:
    """Handles phylogenetic relationships and evolutionary distances"""
    
    def __init__(self):
        self.divergence_times = PROJECT_CONFIG["evolutionary_constraints"]["divergence_times"]
        self.species_tree = self.build_species_tree()
    
    def build_species_tree(self) -> Dict[str, Dict]:
        """Build a phylogenetic tree structure"""
        # Simplified tree based on scientific consensus
        tree = {
            'Archosauria': {
                'divergence_time': 240,
                'children': {
                    'Pseudosuchia': {
                        'divergence_time': 240,
                        'children': {
                            'Alligator mississippiensis': {'divergence_time': 0},
                            'Crocodylus porosus': {'divergence_time': 80},
                            'Gavialis gangeticus': {'divergence_time': 100}
                        }
                    },
                    'Avemetatarsalia': {
                        'divergence_time': 240,
                        'children': {
                            'Dinosauria': {
                                'divergence_time': 200,
                                'children': {
                                    'Theropoda': {
                                        'divergence_time': 180,
                                        'children': {
                                            'Aves': {
                                                'divergence_time': 150,
                                                'children': {
                                                    'Paleognathae': {
                                                        'divergence_time': 100,
                                                        'children': {
                                                            'Struthio camelus': {'divergence_time': 0},
                                                            'Tinamus guttatus': {'divergence_time': 20}
                                                        }
                                                    },
                                                    'Neognathae': {
                                                        'divergence_time': 85,
                                                        'children': {
                                                            'Gallus gallus': {'divergence_time': 0},
                                                            'Taeniopygia guttata': {'divergence_time': 50},
                                                            'Falco peregrinus': {'divergence_time': 50},
                                                            'Corvus brachyrhynchos': {'divergence_time': 50}
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        return tree
    
    def get_evolutionary_distance(self, species1: str, species2: str) -> float:
        """Calculate evolutionary distance between two species"""
        # Find common ancestor and sum branch lengths
        path1 = self.find_path_to_species(species1, self.species_tree)
        path2 = self.find_path_to_species(species2, self.species_tree)
        
        if not path1 or not path2:
            return float('inf')
        
        # Find common ancestor
        common_ancestor_idx = 0
        for i, (node1, node2) in enumerate(zip(path1, path2)):
            if node1 != node2:
                break
            common_ancestor_idx = i
        
        # Calculate distance
        distance1 = sum(node.get('divergence_time', 0) for node in path1[common_ancestor_idx:])
        distance2 = sum(node.get('divergence_time', 0) for node in path2[common_ancestor_idx:])
        
        return distance1 + distance2
    
    def find_path_to_species(self, species: str, tree: Dict, path: List = None) -> Optional[List]:
        """Find path from root to species in tree"""
        if path is None:
            path = []
        
        if species in tree:
            return path + [tree[species]]
        
        for key, subtree in tree.items():
            if isinstance(subtree, dict) and 'children' in subtree:
                result = self.find_path_to_species(species, subtree['children'], path + [subtree])
                if result is not None:
                    return result
        
        return None

def main():
    """Main data collection and preprocessing pipeline"""
    logger.info("Starting dinosaur DNA reconstruction data pipeline")
    
    # Initialize components
    collector = GenomeDataCollector()
    preprocessor = SequencePreprocessor()
    phylo_processor = PhylogeneticDataProcessor()
    
    # Create output directories
    for path in PROJECT_CONFIG["paths"].values():
        os.makedirs(path, exist_ok=True)
    
    # Download mitochondrial genomes (faster and more manageable)
    logger.info("Downloading mitochondrial genomes...")
    mt_genomes = collector.download_mitochondrial_genomes()
    
    # Process sequences
    processed_data = []
    
    for species, file_path in tqdm(mt_genomes.items(), desc="Processing sequences"):
        try:
            # Read sequence
            record = SeqIO.read(file_path, "fasta")
            sequence = str(record.seq)
            
            # Create windows
            windows = preprocessor.create_sliding_windows(sequence)
            
            for window in windows:
                # Calculate features
                features = preprocessor.calculate_sequence_features(window)
                
                # Create synthetic ancient DNA samples
                damaged_sequences = []
                for damage_rate in [0.1, 0.3, 0.5]:  # Different damage levels
                    damaged = collector.create_synthetic_ancient_dna(window, damage_rate)
                    fragments = collector.fragment_sequences(damaged)
                    damaged_sequences.extend(fragments)
                
                # Add to dataset
                processed_data.append({
                    'species': species,
                    'original_sequence': window,
                    'damaged_sequences': damaged_sequences,
                    'features': features,
                    'encoded_sequence': preprocessor.encode_sequence(window)
                })
        
        except Exception as e:
            logger.error(f"Error processing {species}: {e}")
    
    # Save processed data
    output_path = os.path.join(PROJECT_CONFIG["paths"]["processed_data"], "training_data.json")
    with open(output_path, 'w') as f:
        json.dump(processed_data, f, indent=2)
    
    logger.info(f"Processed data saved to {output_path}")
    logger.info(f"Total sequences processed: {len(processed_data)}")

if __name__ == "__main__":
    main()
