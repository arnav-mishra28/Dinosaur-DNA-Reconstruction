"""
Enhanced Training Pipeline for Dinosaur DNA Reconstruction
Features: DNABERT integration, multi-head attention training, real NCBI datasets,
variable-length sequences, multi-species phylogenetic context
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import GradScaler, autocast

import json
import gzip
import numpy as np
from pathlib import Path
import logging
from tqdm import tqdm
import wandb
from typing import Dict, List, Tuple, Optional
import random
import time
from dataclasses import dataclass
from collections import defaultdict

from enhanced_config import config
from enhanced_models import create_model, EnhancedDinosaurDNAModel, HybridDinosaurModel

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TrainingMetrics:
    """Track training metrics."""
    epoch: int = 0
    step: int = 0
    train_loss: float = 0.0
    val_loss: float = 0.0
    reconstruction_accuracy: float = 0.0
    confidence_calibration: float = 0.0
    phylogenetic_likelihood: float = 0.0
    learning_rate: float = 0.0

class DinosaurDNADataset(Dataset):
    """Enhanced dataset for dinosaur DNA reconstruction training."""
    
    def __init__(self, data_file: Path, model_tokenizer, max_length: int = 512,
                 augment: bool = True, multi_species: bool = True):
        self.data_file = data_file
        self.tokenizer = model_tokenizer
        self.max_length = max_length
        self.augment = augment
        self.multi_species = multi_species
        
        # Load data
        self.data = self._load_data()
        self.species_to_id = self._build_species_mapping()
        
        logger.info(f"Loaded {len(self.data)} examples from {data_file}")
        
    def _load_data(self) -> List[Dict]:
        """Load training data from compressed JSON."""
        try:
            with gzip.open(self.data_file, 'rt') as f:
                data = json.load(f)
            return data
        except Exception as e:
            logger.error(f"Failed to load data from {self.data_file}: {e}")
            return []
    
    def _build_species_mapping(self) -> Dict[str, int]:
        """Build species name to ID mapping."""
        species_names = set()
        for example in self.data:
            if self.multi_species:
                species_names.add(example['primary_species'])
                species_names.update(example.get('context_species', []))
            else:
                species_names.add(example['species'])
        
        return {species: idx for idx, species in enumerate(sorted(species_names))}
    
    def _augment_sequence(self, sequence: str) -> str:
        """Apply data augmentation to sequence."""
        if not self.augment or random.random() > 0.5:
            return sequence
        
        # Random mutations (low rate)
        if random.random() < 0.1:
            sequence_list = list(sequence.upper())
            for i in range(len(sequence_list)):
                if random.random() < 0.001:  # 0.1% mutation rate
                    sequence_list[i] = random.choice(['A', 'T', 'G', 'C'])
            sequence = ''.join(sequence_list)
        
        # Random masking
        if random.random() < 0.2:
            sequence_list = list(sequence)
            mask_length = min(10, len(sequence_list) // 20)
            start_pos = random.randint(0, len(sequence_list) - mask_length)
            for i in range(start_pos, start_pos + mask_length):
                sequence_list[i] = 'N'
            sequence = ''.join(sequence_list)
        
        return sequence
    
    def _tokenize_sequence(self, sequence: str) -> Dict[str, torch.Tensor]:
        """Tokenize DNA sequence using the model's tokenizer."""
        # Ensure sequence is uppercase and clean
        sequence = sequence.upper().replace('U', 'T')
        
        # Apply augmentation
        sequence = self._augment_sequence(sequence)
        
        # Truncate if too long
        if len(sequence) > self.max_length - 2:  # Account for special tokens
            sequence = sequence[:self.max_length - 2]
        
        # Tokenize using the model's tokenizer
        return self.tokenizer.encode(sequence, max_length=self.max_length)
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        example = self.data[idx]
        
        if self.multi_species:
            # Multi-species example
            damaged_seq = example['damaged_sequence']
            target_seq = example['target_sequence']
            
            # Tokenize sequences
            damaged_tokens = self._tokenize_sequence(damaged_seq)
            target_tokens = self._tokenize_sequence(target_seq)
            
            # Species information
            primary_species_id = self.species_to_id.get(example['primary_species'], 0)
            context_species_ids = [
                self.species_to_id.get(species, 0) 
                for species in example.get('context_species', [])
            ]
            
            # Phylogenetic context
            divergence_times = example.get('divergence_times', [0])
            
            # Convert to tensors
            result = {
                'input_ids': damaged_tokens['input_ids'],
                'attention_mask': damaged_tokens['attention_mask'],
                'target_ids': target_tokens['input_ids'],
                'target_mask': target_tokens['attention_mask'],
                'species_id': torch.tensor(primary_species_id, dtype=torch.long),
                'divergence_time': torch.tensor(divergence_times[0] if divergence_times else 0, 
                                              dtype=torch.long),
                'age_years': torch.tensor(example.get('age_years', 100000), dtype=torch.float),
                'temperature': torch.tensor(example.get('temperature', 15.0), dtype=torch.float),
            }
            
            # Taxonomy information (simplified)
            result.update({
                'kingdom': torch.tensor(0, dtype=torch.long),  # Animalia
                'phylum': torch.tensor(0, dtype=torch.long),   # Chordata
                'class': torch.tensor(0, dtype=torch.long),    # Will be determined by species
                'order': torch.tensor(0, dtype=torch.long),    # Will be determined by species
            })
            
        else:
            # Single species example
            damaged_seq = example['damaged_sequence']
            target_seq = example['original_sequence']
            
            # Tokenize sequences
            damaged_tokens = self._tokenize_sequence(damaged_seq)
            target_tokens = self._tokenize_sequence(target_seq)
            
            # Species information
            species_id = self.species_to_id.get(example['species'], 0)
            
            result = {
                'input_ids': damaged_tokens['input_ids'],
                'attention_mask': damaged_tokens['attention_mask'],
                'target_ids': target_tokens['input_ids'],
                'target_mask': target_tokens['attention_mask'],
                'species_id': torch.tensor(species_id, dtype=torch.long),
                'divergence_time': torch.tensor(0, dtype=torch.long),
                'age_years': torch.tensor(example.get('age_years', 100000), dtype=torch.float),
                'temperature': torch.tensor(example.get('temperature', 15.0), dtype=torch.float),
                'kingdom': torch.tensor(0, dtype=torch.long),
                'phylum': torch.tensor(0, dtype=torch.long),
                'class': torch.tensor(0, dtype=torch.long),
                'order': torch.tensor(0, dtype=torch.long),
            }
        
        return result

def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Custom collate function for variable-length sequences."""
    
    # Get maximum lengths in batch
    max_input_len = max(item['input_ids'].size(0) for item in batch)
    max_target_len = max(item['target_ids'].size(0) for item in batch)
    
    # Prepare batch tensors
    batch_size = len(batch)
    
    # Input sequences
    input_ids = torch.zeros(batch_size, max_input_len, dtype=torch.long)
    attention_mask = torch.zeros(batch_size, max_input_len, dtype=torch.long)
    
    # Target sequences
    target_ids = torch.zeros(batch_size, max_target_len, dtype=torch.long)
    target_mask = torch.zeros(batch_size, max_target_len, dtype=torch.long)
    
    # Other fields
    species_ids = torch.zeros(batch_size, dtype=torch.long)
    divergence_times = torch.zeros(batch_size, dtype=torch.long)
    age_years = torch.zeros(batch_size, dtype=torch.float)
    temperatures = torch.zeros(batch_size, dtype=torch.float)
    
    # Taxonomy
    kingdoms = torch.zeros(batch_size, dtype=torch.long)
    phylums = torch.zeros(batch_size, dtype=torch.long)
    classes = torch.zeros(batch_size, dtype=torch.long)
    orders = torch.zeros(batch_size, dtype=torch.long)
    
    # Fill tensors
    for i, item in enumerate(batch):
        input_len = item['input_ids'].size(0)
        target_len = item['target_ids'].size(0)
        
        input_ids[i, :input_len] = item['input_ids']
        attention_mask[i, :input_len] = item['attention_mask']
        
        target_ids[i, :target_len] = item['target_ids']
        target_mask[i, :target_len] = item['target_mask']
        
        species_ids[i] = item['species_id']
        divergence_times[i] = item['divergence_time']
        age_years[i] = item['age_years']
        temperatures[i] = item['temperature']
        
        kingdoms[i] = item['kingdom']
        phylums[i] = item['phylum']
        classes[i] = item['class']
        orders[i] = item['order']
    
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'target_ids': target_ids,
        'target_mask': target_mask,
        'species_ids': species_ids,
        'divergence_times': divergence_times,
        'age_years': age_years,
        'temperatures': temperatures,
        'taxonomy': {
            'kingdom': kingdoms,
            'phylum': phylums,
            'class': classes,
            'order': orders,
        }
    }

class EnhancedLoss(nn.Module):
    """Multi-component loss function for dinosaur DNA reconstruction."""
    
    def __init__(self, weights: Dict[str, float]):
        super().__init__()
        self.weights = weights
        self.cross_entropy = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()
        
    def reconstruction_loss(self, logits: torch.Tensor, targets: torch.Tensor,
                           target_mask: torch.Tensor) -> torch.Tensor:
        """Reconstruction loss for DNA sequences."""
        # Reshape for loss computation
        vocab_size = logits.size(-1)
        logits_flat = logits.view(-1, vocab_size)
        targets_flat = targets.view(-1)
        
        # Apply mask
        mask_flat = target_mask.view(-1)
        valid_positions = mask_flat > 0
        
        if valid_positions.sum() == 0:
            return torch.tensor(0.0, device=logits.device)
        
        return self.cross_entropy(
            logits_flat[valid_positions], 
            targets_flat[valid_positions]
        )
    
    def evolutionary_consistency_loss(self, evolution_probs: torch.Tensor) -> torch.Tensor:
        """Enforce evolutionary constraints (e.g., Ts/Tv ratio)."""
        # Expected transition/transversion ratio (~2.0 for vertebrates)
        batch_size, seq_len, _ = evolution_probs.shape
        
        # A<->G and C<->T are transitions
        # All others are transversions
        transitions = evolution_probs[:, :, [0, 2]] + evolution_probs[:, :, [1, 3]]  # Simplified
        transversions = evolution_probs[:, :, :].sum(dim=-1) - transitions.sum(dim=-1)
        
        # Target Ts/Tv ratio of 2.0
        target_ratio = 2.0
        actual_ratio = transitions.sum(dim=-1) / (transversions + 1e-8)
        
        return self.mse_loss(actual_ratio, torch.full_like(actual_ratio, target_ratio))
    
    def phylogenetic_likelihood_loss(self, phylo_likelihood: torch.Tensor) -> torch.Tensor:
        """Phylogenetic consistency loss."""
        # Encourage higher likelihood scores
        return -phylo_likelihood.mean()
    
    def confidence_calibration_loss(self, confidence: torch.Tensor, 
                                   accuracy: torch.Tensor) -> torch.Tensor:
        """Calibration loss to align confidence with actual accuracy."""
        return self.mse_loss(confidence.squeeze(), accuracy)
    
    def mutation_prediction_loss(self, mutation_info: Dict, 
                                age_years: torch.Tensor, 
                                temperature: torch.Tensor) -> torch.Tensor:
        """Loss for mutation rate prediction accuracy."""
        # Simplified mutation rate prediction loss
        damage_rates = mutation_info['damage_rates']
        
        # Expected damage increases with age and temperature
        expected_damage = torch.sigmoid(age_years / 1000000 + temperature / 50)
        predicted_damage = damage_rates.mean(dim=(-2, -1))  # Average over sequence
        
        return self.mse_loss(predicted_damage, expected_damage)
    
    def forward(self, model_outputs: Dict, targets: Dict) -> Dict[str, torch.Tensor]:
        """Compute total loss and individual components."""
        
        losses = {}
        
        # Reconstruction loss
        if 'fused_logits' in model_outputs:
            logits = model_outputs['fused_logits']  # Use fused logits for hybrid model
        else:
            logits = model_outputs['logits']
        
        losses['reconstruction'] = self.reconstruction_loss(
            logits, targets['target_ids'], targets['target_mask']
        )
        
        # Evolutionary consistency loss
        if 'evolution_probs' in model_outputs:
            losses['evolutionary_consistency'] = self.evolutionary_consistency_loss(
                model_outputs['evolution_probs']
            )
        else:
            losses['evolutionary_consistency'] = torch.tensor(0.0, device=logits.device)
        
        # Phylogenetic likelihood loss
        if 'phylo_likelihood' in model_outputs:
            losses['phylogenetic_likelihood'] = self.phylogenetic_likelihood_loss(
                model_outputs['phylo_likelihood']
            )
        else:
            losses['phylogenetic_likelihood'] = torch.tensor(0.0, device=logits.device)
        
        # Confidence calibration loss (simplified)
        if 'confidence' in model_outputs:
            # Calculate position-wise accuracy as proxy
            predicted = torch.argmax(logits, dim=-1)
            accuracy = (predicted == targets['target_ids']).float() * targets['target_mask']
            accuracy = accuracy.sum(dim=-1) / (targets['target_mask'].sum(dim=-1) + 1e-8)
            
            losses['confidence_calibration'] = self.confidence_calibration_loss(
                model_outputs['confidence'].mean(dim=1),  # Average over sequence
                accuracy
            )
        else:
            losses['confidence_calibration'] = torch.tensor(0.0, device=logits.device)
        
        # Mutation prediction loss
        if 'mutation_info' in model_outputs:
            losses['mutation_prediction'] = self.mutation_prediction_loss(
                model_outputs['mutation_info'],
                targets['age_years'],
                targets['temperatures']
            )
        else:
            losses['mutation_prediction'] = torch.tensor(0.0, device=logits.device)
        
        # Compute weighted total loss
        total_loss = sum(
            self.weights.get(key, 1.0) * loss 
            for key, loss in losses.items()
        )
        
        losses['total'] = total_loss
        return losses

class EnhancedTrainer:
    """Enhanced trainer with multi-GPU support, mixed precision, and monitoring."""
    
    def __init__(self, model: nn.Module, train_dataset: Dataset, val_dataset: Dataset,
                 config_dict: Dict):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.config = config_dict
        
        # Setup device and distributed training
        self.device = torch.device(config.DEVICE)
        self.is_distributed = torch.cuda.device_count() > 1
        
        if self.is_distributed:
            dist.init_process_group(backend='nccl')
            self.model = DDP(model.to(self.device))
        else:
            self.model = model.to(self.device)
        
        # Setup data loaders
        self.train_loader = self._create_data_loader(train_dataset, shuffle=True)
        self.val_loader = self._create_data_loader(val_dataset, shuffle=False)
        
        # Setup optimizer and scheduler
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        
        # Setup loss function
        self.loss_fn = EnhancedLoss(self.config['loss_weights'])
        
        # Setup mixed precision training
        self.scaler = GradScaler() if self.config['mixed_precision'] else None
        
        # Setup logging
        if self.config['use_wandb']:
            wandb.init(
                project="dinosaur-dna-reconstruction",
                config=self.config,
                name=f"enhanced-training-{int(time.time())}"
            )
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.metrics_history = []
        
        logger.info(f"Trainer initialized - Device: {self.device}")
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def _create_data_loader(self, dataset: Dataset, shuffle: bool) -> DataLoader:
        """Create data loader with appropriate sampler."""
        sampler = DistributedSampler(dataset) if self.is_distributed else None
        
        return DataLoader(
            dataset,
            batch_size=self.config['batch_size'],
            shuffle=(shuffle and sampler is None),
            sampler=sampler,
            num_workers=config.NUM_WORKERS,
            collate_fn=collate_fn,
            pin_memory=True,
        )
    
    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer with weight decay."""
        return AdamW(
            self.model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay'],
            eps=1e-8,
        )
    
    def _create_scheduler(self):
        """Create learning rate scheduler."""
        return CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=self.config['warmup_steps'],
            T_mult=2,
            eta_min=self.config['learning_rate'] * 0.01,
        )
    
    def _forward_pass(self, batch: Dict) -> Tuple[Dict, Dict]:
        """Forward pass through model."""
        # Prepare inputs
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        species_ids = batch['species_ids'].to(self.device)
        divergence_times = batch['divergence_times'].to(self.device)
        taxonomy = {k: v.to(self.device) for k, v in batch['taxonomy'].items()}
        age_years = batch['age_years'].to(self.device)
        temperatures = batch['temperatures'].to(self.device)
        
        # Forward pass
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            species_ids=species_ids,
            divergence_times=divergence_times,
            taxonomy=taxonomy,
            temperature=temperatures,
            time=age_years / 365.25,  # Convert to years
        )
        
        # Prepare targets
        targets = {
            'target_ids': batch['target_ids'].to(self.device),
            'target_mask': batch['target_mask'].to(self.device),
            'age_years': age_years,
            'temperatures': temperatures,
        }
        
        return outputs, targets
    
    def train_step(self, batch: Dict) -> Dict[str, float]:
        """Single training step."""
        self.model.train()
        
        if self.scaler:
            with autocast():
                outputs, targets = self._forward_pass(batch)
                loss_dict = self.loss_fn(outputs, targets)
        else:
            outputs, targets = self._forward_pass(batch)
            loss_dict = self.loss_fn(outputs, targets)
        
        # Backward pass
        total_loss = loss_dict['total']
        
        if self.scaler:
            self.scaler.scale(total_loss).backward()
            
            if (self.global_step + 1) % self.config['gradient_accumulation_steps'] == 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config['gradient_clipping']
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                self.scheduler.step()
        else:
            total_loss.backward()
            
            if (self.global_step + 1) % self.config['gradient_accumulation_steps'] == 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['gradient_clipping']
                )
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.scheduler.step()
        
        return {k: v.item() for k, v in loss_dict.items()}
    
    def validation_step(self, batch: Dict) -> Dict[str, float]:
        """Single validation step."""
        self.model.eval()
        
        with torch.no_grad():
            outputs, targets = self._forward_pass(batch)
            loss_dict = self.loss_fn(outputs, targets)
        
        return {k: v.item() for k, v in loss_dict.items()}
    
    def train_epoch(self) -> TrainingMetrics:
        """Train for one epoch."""
        self.model.train()
        epoch_losses = defaultdict(list)
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}")
        
        for batch in progress_bar:
            loss_dict = self.train_step(batch)
            
            for key, value in loss_dict.items():
                epoch_losses[key].append(value)
            
            self.global_step += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{loss_dict['total']:.4f}",
                'lr': f"{self.optimizer.param_groups[0]['lr']:.2e}"
            })
            
            # Log to wandb
            if self.config['use_wandb'] and self.global_step % 10 == 0:
                wandb.log({
                    f"train/{key}": value for key, value in loss_dict.items()
                }, step=self.global_step)
        
        # Calculate epoch averages
        avg_losses = {key: np.mean(values) for key, values in epoch_losses.items()}
        
        return TrainingMetrics(
            epoch=self.current_epoch,
            step=self.global_step,
            train_loss=avg_losses['total'],
            learning_rate=self.optimizer.param_groups[0]['lr']
        )
    
    def validate(self) -> TrainingMetrics:
        """Run validation."""
        self.model.eval()
        epoch_losses = defaultdict(list)
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                loss_dict = self.validation_step(batch)
                
                for key, value in loss_dict.items():
                    epoch_losses[key].append(value)
        
        # Calculate averages
        avg_losses = {key: np.mean(values) for key, values in epoch_losses.items()}
        
        # Log to wandb
        if self.config['use_wandb']:
            wandb.log({
                f"val/{key}": value for key, value in avg_losses.items()
            }, step=self.global_step)
        
        return TrainingMetrics(
            epoch=self.current_epoch,
            step=self.global_step,
            val_loss=avg_losses['total']
        )
    
    def save_checkpoint(self, metrics: TrainingMetrics, is_best: bool = False) -> None:
        """Save model checkpoint."""
        checkpoint_dir = config.MODEL_DIR / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'epoch': metrics.epoch,
            'global_step': metrics.step,
            'model_state_dict': self.model.module.state_dict() if self.is_distributed else self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config,
            'metrics': metrics,
        }
        
        # Save regular checkpoint
        checkpoint_file = checkpoint_dir / f"checkpoint_epoch_{metrics.epoch}.pt"
        torch.save(checkpoint, checkpoint_file)
        
        # Save best model
        if is_best:
            best_file = checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_file)
            logger.info(f"Saved best model with validation loss: {metrics.val_loss:.4f}")
        
        logger.info(f"Saved checkpoint: {checkpoint_file}")
    
    def train(self) -> None:
        """Main training loop."""
        logger.info("Starting training...")
        
        patience_counter = 0
        
        for epoch in range(self.config['epochs']):
            self.current_epoch = epoch
            
            # Train epoch
            train_metrics = self.train_epoch()
            
            # Validate
            val_metrics = self.validate()
            
            # Check for improvement
            is_best = val_metrics.val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics.val_loss
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Save checkpoint
            combined_metrics = TrainingMetrics(
                epoch=epoch,
                step=self.global_step,
                train_loss=train_metrics.train_loss,
                val_loss=val_metrics.val_loss,
                learning_rate=train_metrics.learning_rate
            )
            
            self.save_checkpoint(combined_metrics, is_best)
            self.metrics_history.append(combined_metrics)
            
            # Log metrics
            logger.info(f"Epoch {epoch}: Train Loss: {train_metrics.train_loss:.4f}, "
                       f"Val Loss: {val_metrics.val_loss:.4f}, LR: {train_metrics.learning_rate:.2e}")
            
            # Early stopping
            if patience_counter >= self.config['early_stopping_patience']:
                logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                break
        
        logger.info("Training completed!")

def main():
    """Main training function."""
    logger.info("Starting enhanced dinosaur DNA training pipeline")
    
    # Load datasets
    train_data_file = config.DATA_DIR / "multi_species_examples.json.gz"
    val_data_file = config.DATA_DIR / "training_pairs.json.gz"  # Use single species for validation
    
    if not train_data_file.exists():
        logger.error(f"Training data file not found: {train_data_file}")
        logger.info("Please run enhanced_data_collection.py first")
        return
    
    # Create model
    model = create_model(model_type='hybrid')
    logger.info(f"Created model with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Create datasets
    train_dataset = DinosaurDNADataset(
        train_data_file, 
        model.transformer.tokenizer if hasattr(model, 'transformer') else model.tokenizer,
        max_length=config.SEQUENCE_CONFIG['max_length'],
        augment=True,
        multi_species=True
    )
    
    val_dataset = DinosaurDNADataset(
        val_data_file,
        model.transformer.tokenizer if hasattr(model, 'transformer') else model.tokenizer,
        max_length=config.SEQUENCE_CONFIG['max_length'],
        augment=False,
        multi_species=False
    )
    
    # Create trainer
    trainer = EnhancedTrainer(model, train_dataset, val_dataset, config.TRAINING_CONFIG)
    
    # Start training
    trainer.train()

if __name__ == "__main__":
    main()
