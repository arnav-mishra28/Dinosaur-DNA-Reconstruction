"""
Training Pipeline for Dinosaur DNA Reconstruction
Implements custom loss functions with evolutionary and phylogenetic constraints
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import numpy as np
import json
import os
from typing import Dict, List, Tuple, Optional
import logging
from tqdm import tqdm
import wandb
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

from config import PROJECT_CONFIG
from models import create_model, ModelOutput
from data_collection import SequencePreprocessor, PhylogeneticDataProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DinosaurDNADataset(Dataset):
    """Dataset class for dinosaur DNA reconstruction"""
    
    def __init__(self, data_path: str, max_length: int = 512):
        self.data_path = data_path
        self.max_length = max_length
        self.preprocessor = SequencePreprocessor()
        
        # Load processed data
        with open(data_path, 'r') as f:
            self.data = json.load(f)
        
        # Create species to ID mapping
        self.species_to_id = {
            species: idx for idx, species in enumerate(
                set(item['species'] for item in self.data)
            )
        }
        
        # Prepare dataset items
        self.items = self._prepare_items()
        
    def _prepare_items(self) -> List[Dict]:
        """Prepare individual training items from the data"""
        items = []
        phylo_processor = PhylogeneticDataProcessor()
        
        for data_item in self.data:
            species = data_item['species']
            original_seq = data_item['original_sequence']
            damaged_sequences = data_item['damaged_sequences']
            
            for damaged_seq in damaged_sequences:
                # Pad/truncate sequences
                original_padded = self._pad_sequence(original_seq)
                damaged_padded = self._pad_sequence(damaged_seq)
                
                # Create damage mask
                damage_mask = self._create_damage_mask(original_padded, damaged_padded)
                
                # Encode sequences
                original_encoded = self.preprocessor.encode_sequence(original_padded)
                damaged_encoded = self.preprocessor.encode_sequence(damaged_padded)
                
                # Phylogenetic information
                species_id = self.species_to_id[species]
                
                # Calculate evolutionary distances (simplified)
                divergence_time = self._get_divergence_time(species)
                phylo_distance = self._get_phylo_distance(species)
                
                items.append({
                    'original_sequence': torch.tensor(original_encoded, dtype=torch.long),
                    'damaged_sequence': torch.tensor(damaged_encoded, dtype=torch.long),
                    'damage_mask': torch.tensor(damage_mask, dtype=torch.float),
                    'species_id': torch.tensor(species_id, dtype=torch.long),
                    'divergence_time': torch.tensor(divergence_time, dtype=torch.float),
                    'phylo_distance': torch.tensor(phylo_distance, dtype=torch.float),
                    'species_name': species
                })
        
        return items
    
    def _pad_sequence(self, sequence: str) -> str:
        """Pad or truncate sequence to max_length"""
        if len(sequence) > self.max_length:
            return sequence[:self.max_length]
        else:
            return sequence + 'N' * (self.max_length - len(sequence))
    
    def _create_damage_mask(self, original: str, damaged: str) -> List[float]:
        """Create mask indicating level of damage at each position"""
        mask = []
        for orig, dmg in zip(original, damaged):
            if orig == dmg:
                mask.append(0.0)  # No damage
            elif dmg == 'N':
                mask.append(1.0)  # Complete loss
            else:
                mask.append(0.5)  # Substitution
        return mask
    
    def _get_divergence_time(self, species: str) -> float:
        """Get divergence time for species (simplified)"""
        # Simplified mapping - in reality would use phylogenetic tree
        bird_times = {
            'Gallus gallus': 65.0,
            'Taeniopygia guttata': 65.0,
            'Struthio camelus': 100.0,
            'Tinamus guttatus': 100.0,
            'Falco peregrinus': 65.0,
            'Corvus brachyrhynchos': 65.0
        }
        
        crocodile_times = {
            'Alligator mississippiensis': 240.0,
            'Crocodylus porosus': 240.0,
            'Gavialis gangeticus': 240.0
        }
        
        return bird_times.get(species, crocodile_times.get(species, 150.0))
    
    def _get_phylo_distance(self, species: str) -> float:
        """Get phylogenetic distance (simplified)"""
        # Distance from theropod ancestor
        if 'gallus' in species.lower() or 'chicken' in species.lower():
            return 150.0  # Close to theropod ancestor
        elif any(bird in species.lower() for bird in ['falcon', 'crow', 'finch']):
            return 180.0  # More derived birds
        elif any(paleo in species.lower() for paleo in ['ostrich', 'tinamou']):
            return 200.0  # Paleognath birds
        else:  # Crocodilians
            return 240.0  # Most distant
    
    def __len__(self) -> int:
        return len(self.items)
    
    def __getitem__(self, idx: int) -> Dict:
        return self.items[idx]

class EvolutionaryLoss(nn.Module):
    """Custom loss function incorporating evolutionary constraints"""
    
    def __init__(self, alpha: float = 1.0, beta: float = 0.5, gamma: float = 0.3):
        super().__init__()
        self.alpha = alpha  # Reconstruction loss weight
        self.beta = beta    # Evolutionary constraint weight  
        self.gamma = gamma  # Phylogenetic consistency weight
        
        self.cross_entropy = nn.CrossEntropyLoss(reduction='none')
        self.mse_loss = nn.MSELoss()
        
    def forward(self, model_output: ModelOutput, targets: Dict) -> Dict[str, torch.Tensor]:
        """
        Compute multi-component evolutionary loss
        
        Args:
            model_output: Model predictions
            targets: Ground truth data
        """
        device = model_output.reconstructed_sequence.device
        
        # Basic reconstruction loss
        recon_logits = model_output.reconstructed_sequence
        target_sequences = targets['original_sequence']
        damage_mask = targets['damage_mask']
        
        # Cross-entropy loss for each position
        batch_size, seq_len, vocab_size = recon_logits.shape
        recon_logits_flat = recon_logits.view(-1, vocab_size)
        target_sequences_flat = target_sequences.view(-1)
        
        position_losses = self.cross_entropy(recon_logits_flat, target_sequences_flat)
        position_losses = position_losses.view(batch_size, seq_len)
        
        # Weight losses by damage level (focus on damaged regions)
        weighted_losses = position_losses * (1.0 + damage_mask)
        reconstruction_loss = weighted_losses.mean()
        
        # Evolutionary constraint loss
        mutation_probs = model_output.mutation_probabilities
        evolutionary_loss = self._evolutionary_constraint_loss(
            mutation_probs, targets['divergence_time']
        )
        
        # Phylogenetic consistency loss
        phylo_loss = self._phylogenetic_consistency_loss(
            model_output.phylogenetic_embedding, 
            targets['phylo_distance']
        )
        
        # Confidence calibration loss
        confidence_loss = self._confidence_calibration_loss(
            model_output.confidence_scores, position_losses.detach(), damage_mask
        )
        
        # Total loss
        total_loss = (self.alpha * reconstruction_loss + 
                     self.beta * evolutionary_loss + 
                     self.gamma * phylo_loss +
                     0.1 * confidence_loss)
        
        return {
            'total_loss': total_loss,
            'reconstruction_loss': reconstruction_loss,
            'evolutionary_loss': evolutionary_loss,
            'phylogenetic_loss': phylo_loss,
            'confidence_loss': confidence_loss
        }
    
    def _evolutionary_constraint_loss(self, mutation_probs: torch.Tensor, 
                                    divergence_times: torch.Tensor) -> torch.Tensor:
        """Penalize violations of evolutionary constraints"""
        batch_size, seq_len, from_base, to_base = mutation_probs.shape
        
        # Known transition/transversion ratios
        ts_tv_ratio = 2.0
        
        # Extract transition and transversion probabilities
        transitions = torch.stack([
            mutation_probs[:, :, 0, 2],  # A->G
            mutation_probs[:, :, 2, 0],  # G->A
            mutation_probs[:, :, 1, 3],  # T->C
            mutation_probs[:, :, 3, 1]   # C->T
        ], dim=-1).mean(dim=-1)
        
        transversions = torch.stack([
            mutation_probs[:, :, 0, 1], mutation_probs[:, :, 0, 3],  # A->T,C
            mutation_probs[:, :, 2, 1], mutation_probs[:, :, 2, 3],  # G->T,C
            mutation_probs[:, :, 1, 0], mutation_probs[:, :, 1, 2],  # T->A,G
            mutation_probs[:, :, 3, 0], mutation_probs[:, :, 3, 2]   # C->A,G
        ], dim=-1).mean(dim=-1)
        
        # Penalize deviations from expected Ts/Tv ratio
        predicted_ratio = transitions / (transversions + 1e-8)
        ratio_loss = torch.abs(predicted_ratio - ts_tv_ratio).mean()
        
        # Time-dependent constraint: more mutations for longer divergence
        time_normalized = divergence_times / 200.0  # Normalize by ~200 Mya
        expected_mutation_rate = time_normalized.unsqueeze(1).expand(-1, seq_len)
        
        total_mutation_rate = mutation_probs.sum(dim=-1).mean(dim=-1)  # Average over positions and source bases
        time_constraint_loss = self.mse_loss(total_mutation_rate, expected_mutation_rate)
        
        return ratio_loss + time_constraint_loss
    
    def _phylogenetic_consistency_loss(self, phylo_embeddings: torch.Tensor,
                                     phylo_distances: torch.Tensor) -> torch.Tensor:
        """Ensure phylogenetically related species have similar embeddings"""
        batch_size = phylo_embeddings.shape[0]
        
        if batch_size < 2:
            return torch.tensor(0.0, device=phylo_embeddings.device)
        
        # Compute pairwise distances in embedding space
        embedding_distances = torch.cdist(phylo_embeddings, phylo_embeddings, p=2)
        
        # Normalize phylogenetic distances
        phylo_dist_matrix = torch.abs(phylo_distances.unsqueeze(0) - phylo_distances.unsqueeze(1))
        phylo_dist_normalized = phylo_dist_matrix / phylo_dist_matrix.max()
        
        # Penalize mismatches between embedding and phylogenetic distances
        distance_correlation_loss = self.mse_loss(
            embedding_distances / embedding_distances.max(),
            phylo_dist_normalized
        )
        
        return distance_correlation_loss
    
    def _confidence_calibration_loss(self, confidence_scores: torch.Tensor,
                                   reconstruction_errors: torch.Tensor,
                                   damage_mask: torch.Tensor) -> torch.Tensor:
        """Calibrate confidence scores with actual errors"""
        # High confidence should correlate with low reconstruction error
        error_normalized = reconstruction_errors / (reconstruction_errors.max() + 1e-8)
        expected_confidence = 1.0 - error_normalized
        
        # Weight by damage level
        weights = 1.0 + damage_mask
        
        calibration_loss = (weights * (confidence_scores - expected_confidence) ** 2).mean()
        
        return calibration_loss

class DinosaurDNATrainer:
    """Main training class for dinosaur DNA reconstruction"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Initialize model
        self.model = create_model(config, model_type="hybrid")
        self.model.to(self.device)
        
        # Loss function
        self.criterion = EvolutionaryLoss()
        
        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config["training"]["learning_rate"],
            weight_decay=config["training"]["weight_decay"]
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=2
        )
        
        # Metrics tracking
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        
        # Create output directories
        os.makedirs(config["paths"]["models"], exist_ok=True)
        os.makedirs(config["paths"]["logs"], exist_ok=True)
    
    def create_dataloaders(self, data_path: str) -> Tuple[DataLoader, DataLoader]:
        """Create training and validation dataloaders"""
        dataset = DinosaurDNADataset(data_path)
        
        # Train/validation split
        val_size = int(len(dataset) * self.config["training"]["validation_split"])
        train_size = len(dataset) - val_size
        
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config["training"]["batch_size"],
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config["training"]["batch_size"],
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        return train_loader, val_loader
    
    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        
        epoch_losses = {'total': [], 'reconstruction': [], 'evolutionary': [], 'phylogenetic': []}
        epoch_accuracy = []
        
        pbar = tqdm(dataloader, desc="Training")
        for batch_idx, batch in enumerate(pbar):
            # Move to device
            batch = {k: v.to(self.device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            
            # Forward pass
            self.optimizer.zero_grad()
            
            model_output = self.model(
                input_ids=batch['damaged_sequence'],
                damaged_mask=batch['damage_mask'],
                species_ids=batch['species_id'],
                divergence_times=batch['divergence_time'],
                phylo_distances=batch['phylo_distance']
            )
            
            # Compute loss
            loss_dict = self.criterion(model_output, batch)
            loss = loss_dict['total_loss']
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 
                self.config["training"]["gradient_clip"]
            )
            self.optimizer.step()
            
            # Track metrics
            epoch_losses['total'].append(loss.item())
            epoch_losses['reconstruction'].append(loss_dict['reconstruction_loss'].item())
            epoch_losses['evolutionary'].append(loss_dict['evolutionary_loss'].item())
            epoch_losses['phylogenetic'].append(loss_dict['phylogenetic_loss'].item())
            
            # Accuracy calculation
            with torch.no_grad():
                predictions = torch.argmax(model_output.reconstructed_sequence, dim=-1)
                targets = batch['original_sequence']
                accuracy = (predictions == targets).float().mean().item()
                epoch_accuracy.append(accuracy)
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f"{loss.item():.4f}",
                'Acc': f"{accuracy:.3f}"
            })
        
        return {
            'total_loss': np.mean(epoch_losses['total']),
            'reconstruction_loss': np.mean(epoch_losses['reconstruction']),
            'evolutionary_loss': np.mean(epoch_losses['evolutionary']),
            'phylogenetic_loss': np.mean(epoch_losses['phylogenetic']),
            'accuracy': np.mean(epoch_accuracy)
        }
    
    def validate_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """Validate for one epoch"""
        self.model.eval()
        
        epoch_losses = {'total': [], 'reconstruction': [], 'evolutionary': [], 'phylogenetic': []}
        epoch_accuracy = []
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            pbar = tqdm(dataloader, desc="Validation")
            for batch in pbar:
                # Move to device
                batch = {k: v.to(self.device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
                
                # Forward pass
                model_output = self.model(
                    input_ids=batch['damaged_sequence'],
                    damaged_mask=batch['damage_mask'],
                    species_ids=batch['species_id'],
                    divergence_times=batch['divergence_time'],
                    phylo_distances=batch['phylo_distance']
                )
                
                # Compute loss
                loss_dict = self.criterion(model_output, batch)
                
                # Track metrics
                epoch_losses['total'].append(loss_dict['total_loss'].item())
                epoch_losses['reconstruction'].append(loss_dict['reconstruction_loss'].item())
                epoch_losses['evolutionary'].append(loss_dict['evolutionary_loss'].item())
                epoch_losses['phylogenetic'].append(loss_dict['phylogenetic_loss'].item())
                
                # Accuracy
                predictions = torch.argmax(model_output.reconstructed_sequence, dim=-1)
                targets = batch['original_sequence']
                accuracy = (predictions == targets).float().mean().item()
                epoch_accuracy.append(accuracy)
                
                # Collect for detailed metrics
                all_predictions.extend(predictions.cpu().numpy().flatten())
                all_targets.extend(targets.cpu().numpy().flatten())
        
        # Calculate additional metrics
        f1 = f1_score(all_targets, all_predictions, average='weighted')
        
        return {
            'total_loss': np.mean(epoch_losses['total']),
            'reconstruction_loss': np.mean(epoch_losses['reconstruction']),
            'evolutionary_loss': np.mean(epoch_losses['evolutionary']),
            'phylogenetic_loss': np.mean(epoch_losses['phylogenetic']),
            'accuracy': np.mean(epoch_accuracy),
            'f1_score': f1
        }
    
    def train(self, data_path: str):
        """Main training loop"""
        logger.info("Starting training...")
        
        # Create dataloaders
        train_loader, val_loader = self.create_dataloaders(data_path)
        
        # Initialize wandb
        wandb.init(
            project="dinosaur-dna-reconstruction",
            config=self.config,
            name=f"hybrid_model_{self.config['version']}"
        )
        
        patience = 0
        best_epoch = 0
        
        for epoch in range(self.config["training"]["epochs"]):
            logger.info(f"Epoch {epoch+1}/{self.config['training']['epochs']}")
            
            # Training
            train_metrics = self.train_epoch(train_loader)
            
            # Validation
            val_metrics = self.validate_epoch(val_loader)
            
            # Learning rate scheduling
            self.scheduler.step()
            
            # Log metrics
            wandb.log({
                'epoch': epoch,
                'train_loss': train_metrics['total_loss'],
                'train_accuracy': train_metrics['accuracy'],
                'val_loss': val_metrics['total_loss'],
                'val_accuracy': val_metrics['accuracy'],
                'val_f1': val_metrics['f1_score'],
                'learning_rate': self.optimizer.param_groups[0]['lr']
            })
            
            logger.info(f"Train Loss: {train_metrics['total_loss']:.4f}, "
                       f"Val Loss: {val_metrics['total_loss']:.4f}, "
                       f"Val Acc: {val_metrics['accuracy']:.3f}")
            
            # Early stopping and model saving
            if val_metrics['total_loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['total_loss']
                best_epoch = epoch
                patience = 0
                
                # Save best model
                self.save_model(f"best_model_epoch_{epoch}.pth")
            else:
                patience += 1
            
            # Early stopping
            if patience >= self.config["training"]["early_stopping_patience"]:
                logger.info(f"Early stopping at epoch {epoch}")
                break
            
            # Save checkpoint
            if (epoch + 1) % 10 == 0:
                self.save_model(f"checkpoint_epoch_{epoch}.pth")
        
        logger.info(f"Training completed. Best epoch: {best_epoch}")
        wandb.finish()
    
    def save_model(self, filename: str):
        """Save model checkpoint"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config,
            'best_val_loss': self.best_val_loss
        }
        
        filepath = os.path.join(self.config["paths"]["models"], filename)
        torch.save(checkpoint, filepath)
        logger.info(f"Model saved: {filepath}")

def main():
    """Main training script"""
    # Load configuration
    config = PROJECT_CONFIG
    
    # Initialize trainer
    trainer = DinosaurDNATrainer(config)
    
    # Data path
    data_path = os.path.join(config["paths"]["processed_data"], "training_data.json")
    
    if not os.path.exists(data_path):
        logger.error(f"Training data not found at {data_path}. Run data_collection.py first.")
        return
    
    # Start training
    trainer.train(data_path)

if __name__ == "__main__":
    main()
