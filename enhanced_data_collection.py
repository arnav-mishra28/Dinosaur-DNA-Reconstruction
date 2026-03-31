"""
Enhanced Data Collection for Dinosaur DNA Reconstruction
Features: Real NCBI datasets, multi-species support, phylogenetic context,
variable-length sequences, and ancient DNA damage simulation
"""

import os
import asyncio
import aiohttp
import pandas as pd
import numpy as np
try:
    from Bio import Entrez, SeqIO
    from Bio.Seq import Seq
    from Bio.SeqRecord import SeqRecord
    BIOPYTHON_AVAILABLE = True
except ImportError:
    print("Warning: BioPython not available. Please install: pip install biopython")
    BIOPYTHON_AVAILABLE = False
    
from pathlib import Path
import json
import gzip
from typing import Dict, List, Tuple, Optional, Generator
import logging
from tqdm import tqdm
import time
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
from enhanced_config import config

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NCBIDataCollector:
    """Enhanced NCBI data collector with async support and caching."""
    
    def __init__(self, email: str, api_key: Optional[str] = None):
        if not BIOPYTHON_AVAILABLE:
            raise ImportError("BioPython is required. Install with: pip install biopython")
            
        self.email = email
        self.api_key = api_key
        Entrez.email = email
        if api_key:
            Entrez.api_key = api_key
        
        self.cache_dir = config.CACHE_DIR / "ncbi_sequences"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Rate limiting for NCBI API
        self.last_request_time = 0
        self.request_delay = 0.34 if api_key else 1.0  # Seconds between requests
        
    def _cache_key(self, query: str, database: str) -> str:
        """Generate cache key for query."""
        return hashlib.md5(f"{query}_{database}".encode()).hexdigest()
    
    def _load_from_cache(self, cache_key: str) -> Optional[List[Dict]]:
        """Load results from cache if available."""
        cache_file = self.cache_dir / f"{cache_key}.json.gz"
        if cache_file.exists():
            try:
                with gzip.open(cache_file, 'rt') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load cache {cache_file}: {e}")
        return None
    
    def _save_to_cache(self, cache_key: str, data: List[Dict]) -> None:
        """Save results to cache."""
        cache_file = self.cache_dir / f"{cache_key}.json.gz"
        try:
            with gzip.open(cache_file, 'wt') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save cache {cache_file}: {e}")
    
    async def _rate_limit(self):
        """Ensure rate limiting for NCBI API."""
        current_time = time.time()
        elapsed = current_time - self.last_request_time
        if elapsed < self.request_delay:
            await asyncio.sleep(self.request_delay - elapsed)
        self.last_request_time = time.time()
    
    def search_sequences(self, species: str, database: str = "nucleotide",
                        max_results: int = 1000, min_length: int = 500) -> List[str]:
        """Search for sequences of a given species."""
        cache_key = self._cache_key(f"{species}_{max_results}_{min_length}", database)
        
        # Try cache first
        cached_results = self._load_from_cache(cache_key)
        if cached_results:
            logger.info(f"Loaded {len(cached_results)} sequences from cache for {species}")
            return [item['id'] for item in cached_results]
        
        try:
            # Convert species name format for NCBI query
            species_query = species.replace('_', ' ')
            
            # Construct search query  
            query = f'"{species_query}"[Organism] AND {min_length}:{10000000}[SLEN]'
            
            logger.info(f"Searching NCBI {database} for {species_query}...")
            
            # Search for sequence IDs
            handle = Entrez.esearch(
                db=database,
                term=query,
                retmax=max_results,
                sort="relevance"
            )
            search_results = Entrez.read(handle)
            handle.close()
            
            sequence_ids = search_results["IdList"]
            logger.info(f"Found {len(sequence_ids)} sequences for {species_query}")
            
            # Cache the results
            cache_data = [{"id": seq_id, "species": species} for seq_id in sequence_ids]
            self._save_to_cache(cache_key, cache_data)
            
            return sequence_ids
            
        except Exception as e:
            logger.error(f"Failed to search sequences for {species}: {e}")
            return []
    
    def fetch_sequences(self, sequence_ids: List[str], batch_size: int = 100) -> List[SeqRecord]:
        """Fetch sequence records in batches."""
        sequences = []
        
        for i in tqdm(range(0, len(sequence_ids), batch_size), 
                     desc="Fetching sequences"):
            batch_ids = sequence_ids[i:i + batch_size]
            
            try:
                # Rate limiting
                time.sleep(self.request_delay)
                
                # Fetch batch
                handle = Entrez.efetch(
                    db="nucleotide",
                    id=batch_ids,
                    rettype="fasta",
                    retmode="text"
                )
                
                batch_sequences = list(SeqIO.parse(handle, "fasta"))
                sequences.extend(batch_sequences)
                handle.close()
                
                logger.info(f"Fetched batch {i//batch_size + 1}: {len(batch_sequences)} sequences")
                
            except Exception as e:
                logger.error(f"Failed to fetch batch {i//batch_size + 1}: {e}")
                continue
        
        return sequences
    
    def download_species_data(self, species: str, max_sequences: int = 1000) -> List[SeqRecord]:
        """Download all data for a single species."""
        logger.info(f"Downloading data for {species}")
        
        # Search for sequences
        sequence_ids = self.search_sequences(
            species, 
            max_results=max_sequences,
            min_length=config.NCBI_CONFIG['min_sequence_length']
        )
        
        if not sequence_ids:
            logger.warning(f"No sequences found for {species}")
            return []
        
        # Limit to requested number
        sequence_ids = sequence_ids[:max_sequences]
        
        # Fetch sequences
        sequences = self.fetch_sequences(sequence_ids, batch_size=50)
        
        # Filter by quality and length
        filtered_sequences = []
        for seq in sequences:
            if (len(seq.seq) >= config.NCBI_CONFIG['min_sequence_length'] and
                self._calculate_quality(seq) >= config.NCBI_CONFIG['quality_threshold']):
                filtered_sequences.append(seq)
        
        logger.info(f"Downloaded {len(filtered_sequences)} quality sequences for {species}")
        return filtered_sequences
    
    def _calculate_quality(self, seq_record: SeqRecord) -> float:
        """Calculate sequence quality score (0-1)."""
        sequence = str(seq_record.seq).upper()
        
        # Count valid nucleotides
        valid_bases = sum(1 for base in sequence if base in 'ATGC')
        total_bases = len(sequence)
        
        if total_bases == 0:
            return 0.0
        
        return valid_bases / total_bases

class AncientDNASimulator:
    """Simulate ancient DNA damage patterns and degradation."""
    
    def __init__(self, random_seed: int = 42):
        self.rng = np.random.RandomState(random_seed)
        
        # Ancient DNA damage parameters
        self.damage_patterns = {
            'c_to_t_5prime': 0.15,  # Cytosine deamination at 5' end
            'g_to_a_3prime': 0.08,  # Guanine oxidation at 3' end  
            'depurination': 0.05,   # A/G loss
            'fragmentation': 0.1,   # Random breaks
        }
        
        # Time-dependent parameters
        self.degradation_rates = {
            'base_rate': 1e-6,     # Per year
            'temperature_factor': 1.5,  # Per 10°C increase
            'humidity_factor': 1.3,     # Per 10% RH increase
        }
    
    def simulate_ancient_damage(self, sequence: str, age_years: int,
                               temperature: float = 10.0, humidity: float = 60.0) -> str:
        """Simulate ancient DNA damage based on age and conditions."""
        sequence = sequence.upper()
        damaged_seq = list(sequence)
        
        # Calculate damage probability based on age and conditions
        temp_factor = (temperature / 10) * self.degradation_rates['temperature_factor']
        humidity_factor = (humidity / 10) * self.degradation_rates['humidity_factor']
        time_factor = age_years * self.degradation_rates['base_rate']
        
        damage_prob = min(0.8, time_factor * temp_factor * humidity_factor)
        
        for i, base in enumerate(damaged_seq):
            if self.rng.random() < damage_prob:
                # Apply specific damage patterns
                if base == 'C' and i < len(damaged_seq) // 2:  # 5' end bias
                    if self.rng.random() < self.damage_patterns['c_to_t_5prime']:
                        damaged_seq[i] = 'T'
                
                elif base == 'G' and i >= len(damaged_seq) // 2:  # 3' end bias
                    if self.rng.random() < self.damage_patterns['g_to_a_3prime']:
                        damaged_seq[i] = 'A'
                
                elif base in 'AG':  # Depurination
                    if self.rng.random() < self.damage_patterns['depurination']:
                        damaged_seq[i] = 'N'
        
        return ''.join(damaged_seq)
    
    def fragment_sequence(self, sequence: str, mean_fragment_length: int = 150,
                         std_fragment_length: int = 50) -> List[str]:
        """Fragment sequence to simulate ancient DNA degradation."""
        fragments = []
        sequence_length = len(sequence)
        
        start = 0
        while start < sequence_length:
            # Generate random fragment length
            fragment_length = max(50, int(self.rng.normal(
                mean_fragment_length, std_fragment_length
            )))
            
            end = min(start + fragment_length, sequence_length)
            fragment = sequence[start:end]
            
            if len(fragment) >= 50:  # Minimum fragment size
                fragments.append(fragment)
            
            start = end
        
        return fragments

class PhylogeneticContext:
    """Generate phylogenetic context for multi-species training."""
    
    def __init__(self):
        self.species_tree = config.PHYLOGENETIC_TREE
        self.species_to_id = self._build_species_mapping()
        
    def _build_species_mapping(self) -> Dict[str, int]:
        """Build mapping from species names to IDs."""
        species_mapping = {}
        species_id = 0
        
        def traverse(node, current_id=0):
            nonlocal species_id
            if 'children' in node:
                for child_name, child_node in node['children'].items():
                    species_mapping[child_name] = species_id
                    species_id += 1
                    traverse(child_node, species_id)
            return current_id
        
        traverse(self.species_tree['root'])
        return species_mapping
    
    def get_species_context(self, species_name: str) -> Dict[str, int]:
        """Get phylogenetic context for a species."""
        # Convert species name format
        species_key = species_name.replace(' ', '_')
        
        # Find species in tree and get taxonomic info
        context = {
            'species_id': self.species_to_id.get(species_key, 0),
            'kingdom': 0,    # Animalia
            'phylum': 0,     # Chordata  
            'class': 0,      # Will be set based on species
            'order': 0,      # Will be set based on species
        }
        
        # Simple taxonomic classification
        if 'gallus' in species_key.lower() or 'anas' in species_key.lower():
            context['class'] = 1  # Aves
            context['order'] = 1  # Galliformes/Anseriformes
        elif 'alligator' in species_key.lower() or 'crocodylus' in species_key.lower():
            context['class'] = 2  # Reptilia
            context['order'] = 2  # Crocodilia
        elif 'tyrannosaurus' in species_key.lower() or 'velociraptor' in species_key.lower():
            context['class'] = 2  # Reptilia (extinct)
            context['order'] = 3  # Theropoda
        
        return context
    
    def get_divergence_time(self, species1: str, species2: str) -> int:
        """Get divergence time between two species in millions of years."""
        # Simplified divergence time calculation
        # In practice, this would use the phylogenetic tree structure
        
        species1_key = species1.replace(' ', '_')
        species2_key = species2.replace(' ', '_')
        
        # Default divergence times (simplified)
        if species1_key == species2_key:
            return 0
        elif ('gallus' in species1_key and 'anas' in species2_key) or \
             ('gallus' in species2_key and 'anas' in species1_key):
            return 50  # Bird divergence
        elif ('gallus' in species1_key or 'anas' in species1_key) and \
             ('alligator' in species2_key or 'crocodylus' in species2_key):
            return 250  # Bird-crocodile divergence
        else:
            return 100  # Default divergence

class EnhancedDataProcessor:
    """Process and prepare data for training with enhanced features."""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.ncbi_collector = NCBIDataCollector(config.NCBI_CONFIG['email'])
        self.dna_simulator = AncientDNASimulator()
        self.phylo_context = PhylogeneticContext()
        
    def collect_all_species_data(self) -> Dict[str, List[SeqRecord]]:
        """Collect data for all target species."""
        all_data = {}
        
        logger.info(f"Collecting data for {len(config.NCBI_CONFIG['target_species'])} species")
        
        for species in tqdm(config.NCBI_CONFIG['target_species'], desc="Species"):
            try:
                sequences = self.ncbi_collector.download_species_data(
                    species, 
                    max_sequences=config.NCBI_CONFIG['max_sequences_per_species']
                )
                
                if sequences:
                    all_data[species] = sequences
                    logger.info(f"Collected {len(sequences)} sequences for {species}")
                else:
                    logger.warning(f"No data collected for {species}")
                    
            except Exception as e:
                logger.error(f"Failed to collect data for {species}: {e}")
                continue
        
        return all_data
    
    def create_training_pairs(self, all_data: Dict[str, List[SeqRecord]]) -> List[Dict]:
        """Create training pairs with damaged and original sequences."""
        training_pairs = []
        
        logger.info("Creating training pairs with ancient DNA simulation")
        
        for species, sequences in all_data.items():
            species_context = self.phylo_context.get_species_context(species)
            
            for seq_record in tqdm(sequences, desc=f"Processing {species}"):
                original_seq = str(seq_record.seq).upper()
                
                # Skip sequences that are too short or have too many Ns
                if len(original_seq) < 100 or original_seq.count('N') / len(original_seq) > 0.1:
                    continue
                
                # Create multiple damaged versions with different parameters
                for age in [50000, 100000, 500000, 1000000]:  # Different ages
                    for temp in [5, 15, 25]:  # Different temperatures
                        # Simulate ancient damage
                        damaged_seq = self.dna_simulator.simulate_ancient_damage(
                            original_seq, age, temp
                        )
                        
                        # Fragment the sequence
                        fragments = self.dna_simulator.fragment_sequence(damaged_seq)
                        
                        for fragment in fragments:
                            if len(fragment) >= config.SEQUENCE_CONFIG['min_length']:
                                # Create training example
                                training_pair = {
                                    'original_sequence': original_seq,
                                    'damaged_sequence': fragment,
                                    'species': species,
                                    'species_context': species_context,
                                    'age_years': age,
                                    'temperature': temp,
                                    'sequence_id': seq_record.id,
                                    'length': len(fragment),
                                }
                                
                                training_pairs.append(training_pair)
        
        logger.info(f"Created {len(training_pairs)} training pairs")
        return training_pairs
    
    def create_multi_species_examples(self, training_pairs: List[Dict]) -> List[Dict]:
        """Create multi-species training examples with phylogenetic context."""
        multi_species_examples = []
        
        logger.info("Creating multi-species phylogenetic examples")
        
        # Group by species
        species_groups = {}
        for pair in training_pairs:
            species = pair['species']
            if species not in species_groups:
                species_groups[species] = []
            species_groups[species].append(pair)
        
        # Create multi-species combinations
        species_list = list(species_groups.keys())
        
        for i, primary_species in enumerate(species_list):
            primary_examples = species_groups[primary_species]
            
            # Find related species for phylogenetic context
            related_species = []
            for j, other_species in enumerate(species_list):
                if i != j:
                    divergence_time = self.phylo_context.get_divergence_time(
                        primary_species, other_species
                    )
                    if divergence_time < 300:  # Within 300 million years
                        related_species.append((other_species, divergence_time))
            
            # Sort by divergence time (closest first)
            related_species.sort(key=lambda x: x[1])
            
            # Create examples with phylogenetic context
            for primary_example in primary_examples[:100]:  # Limit examples per species
                context_sequences = []
                context_species = []
                divergence_times = []
                
                # Add primary sequence
                context_sequences.append(primary_example['damaged_sequence'])
                context_species.append(primary_example['species'])
                divergence_times.append(0)
                
                # Add related species context (up to 3 related species)
                for related_sp, div_time in related_species[:3]:
                    if related_sp in species_groups:
                        related_examples = species_groups[related_sp]
                        if related_examples:
                            related_example = random.choice(related_examples)
                            context_sequences.append(related_example['damaged_sequence'])
                            context_species.append(related_sp)
                            divergence_times.append(div_time)
                
                # Create multi-species training example
                multi_example = {
                    'target_sequence': primary_example['original_sequence'],
                    'damaged_sequence': primary_example['damaged_sequence'],
                    'context_sequences': context_sequences,
                    'context_species': context_species,
                    'divergence_times': divergence_times,
                    'primary_species': primary_species,
                    'age_years': primary_example['age_years'],
                    'temperature': primary_example['temperature'],
                }
                
                multi_species_examples.append(multi_example)
        
        logger.info(f"Created {len(multi_species_examples)} multi-species examples")
        return multi_species_examples
    
    def save_processed_data(self, training_pairs: List[Dict], 
                           multi_species_examples: List[Dict]) -> None:
        """Save processed data to files."""
        logger.info("Saving processed data...")
        
        # Save training pairs
        pairs_file = self.output_dir / "training_pairs.json.gz"
        with gzip.open(pairs_file, 'wt') as f:
            json.dump(training_pairs, f, indent=2)
        
        # Save multi-species examples
        multi_file = self.output_dir / "multi_species_examples.json.gz"
        with gzip.open(multi_file, 'wt') as f:
            json.dump(multi_species_examples, f, indent=2)
        
        # Create summary statistics
        stats = {
            'total_training_pairs': len(training_pairs),
            'total_multi_species_examples': len(multi_species_examples),
            'species_counts': {},
            'average_sequence_length': 0,
            'damage_distribution': {},
        }
        
        # Calculate statistics
        for pair in training_pairs:
            species = pair['species']
            stats['species_counts'][species] = stats['species_counts'].get(species, 0) + 1
        
        if training_pairs:
            avg_length = sum(pair['length'] for pair in training_pairs) / len(training_pairs)
            stats['average_sequence_length'] = avg_length
        
        # Save statistics
        stats_file = self.output_dir / "data_statistics.json"
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        
        logger.info(f"Saved data to {self.output_dir}")
        logger.info(f"Training pairs: {len(training_pairs)}")
        logger.info(f"Multi-species examples: {len(multi_species_examples)}")

async def main():
    """Main data collection pipeline."""
    logger.info("Starting enhanced data collection pipeline")
    
    # Initialize processor
    processor = EnhancedDataProcessor(config.DATA_DIR)
    
    # Collect all species data
    all_data = processor.collect_all_species_data()
    
    if not all_data:
        logger.error("No data collected! Check your NCBI configuration.")
        return
    
    # Create training pairs
    training_pairs = processor.create_training_pairs(all_data)
    
    # Create multi-species examples
    multi_species_examples = processor.create_multi_species_examples(training_pairs)
    
    # Save processed data
    processor.save_processed_data(training_pairs, multi_species_examples)
    
    logger.info("Data collection pipeline completed successfully!")

if __name__ == "__main__":
    asyncio.run(main())
