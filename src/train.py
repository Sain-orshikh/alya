"""
Training script with two-stage fine-tuning strategy.
Stage 1: Train only the classification head (frozen backbone)
Stage 2: Fine-tune entire network with low learning rate
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from tqdm import tqdm
import os
from datetime import datetime
import config
from src.model import create_model, count_parameters
from src.dataset import get_data_loaders


class EmotionTrainer:
    """Trainer class for two-stage fine-tuning of emotion detector."""
    
    def __init__(self, model=None, device=config.DEVICE):
        """
        Args:
            model: EmotionDetector model instance
            device: torch device
        """
        self.device = device
        self.model = model or create_model(freeze_backbone=True)
        self.criterion = nn.CrossEntropyLoss()
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
        }
    
    def _get_optimizer(self, stage=1):
        """Get optimizer for current stage."""
        if stage == 1:
            # Stage 1: Only train the head
            optimizer = optim.Adam(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=config.STAGE1_LR,
                weight_decay=config.WEIGHT_DECAY
            )
        else:
            # Stage 2: Discriminative fine-tuning with SGD
            param_groups = self.model.get_discriminative_lr_groups(config.STAGE2_LR)
            optimizer = optim.SGD(
                param_groups,
                momentum=config.STAGE2_MOMENTUM,
                weight_decay=config.WEIGHT_DECAY
            )
        return optimizer
    
    def train_epoch(self, train_loader, optimizer, stage=1):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        progress_bar = tqdm(train_loader, desc=f"Stage {stage} - Training")
        
        for images, labels in progress_bar:
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            
            progress_bar.set_postfix({'loss': loss.item(), 'acc': 100 * correct / total})
        
        avg_loss = total_loss / len(train_loader)
        avg_acc = 100 * correct / total
        
        return avg_loss, avg_acc
    
    def validate(self, val_loader):
        """Validate on validation set."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            progress_bar = tqdm(val_loader, desc="Validating")
            for images, labels in progress_bar:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
                
                progress_bar.set_postfix({'loss': loss.item(), 'acc': 100 * correct / total})
        
        avg_loss = total_loss / len(val_loader)
        avg_acc = 100 * correct / total
        
        return avg_loss, avg_acc
    
    def train_stage(self, train_loader, val_loader, num_epochs, stage=1):
        """
        Train for one stage.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of epochs
            stage: Stage number (1 or 2)
        """
        optimizer = self._get_optimizer(stage)
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=3,
            verbose=True
        )
        
        print(f"\n{'='*60}")
        print(f"Stage {stage} Training")
        print(f"{'='*60}")
        print(f"Epochs: {num_epochs}")
        print(f"Optimizer: {type(optimizer).__name__}")
        print(f"Device: {self.device}")
        
        # Print model parameters status
        params = count_parameters(self.model)
        print(f"Trainable parameters: {params['trainable']:,}")
        print(f"{'='*60}\n")
        
        for epoch in range(num_epochs):
            print(f"\nEpoch [{epoch+1}/{num_epochs}]")
            
            # Train
            train_loss, train_acc = self.train_epoch(train_loader, optimizer, stage)
            
            # Validate
            val_loss, val_acc = self.validate(val_loader)
            
            # Store history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self.save_checkpoint(config.BEST_MODEL_PATH, stage=stage)
                print(f"✓ Best model saved (Val Loss: {val_loss:.4f})")
            else:
                self.patience_counter += 1
            
            # Early stopping
            if self.patience_counter >= config.EARLY_STOPPING_PATIENCE:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
            
            # Learning rate scheduling
            scheduler.step(val_loss)
    
    def train_two_stage(self, train_loader, val_loader):
        """
        Execute two-stage training:
        Stage 1: Train head only (frozen backbone)
        Stage 2: Fine-tune entire network
        """
        print("\n" + "="*60)
        print("TWO-STAGE FINE-TUNING STRATEGY")
        print("="*60)
        
        # Stage 1: Train head with frozen backbone
        print("\n[STAGE 1] Training classification head (frozen backbone)")
        self.train_stage(
            train_loader, val_loader,
            num_epochs=config.STAGE1_EPOCHS,
            stage=1
        )
        
        # Stage 2: Fine-tune entire network
        print("\n[STAGE 2] Fine-tuning entire network")
        self.model.unfreeze_backbone()
        self.model.freeze_early_layers(num_layers=2)  # Keep layer1 and layer2 frozen
        
        self.train_stage(
            train_loader, val_loader,
            num_epochs=config.STAGE2_EPOCHS,
            stage=2
        )
        
        # Save final model
        self.save_checkpoint(config.TRAINED_MODEL_PATH, final=True)
        print(f"\n✓ Training complete! Final model saved.")
    
    def save_checkpoint(self, path, stage=None, final=False):
        """Save model checkpoint."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'history': self.history,
        }
        if stage:
            checkpoint['stage'] = stage
        if final:
            checkpoint['final'] = True
        
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.history = checkpoint.get('history', self.history)
        print(f"✓ Model loaded from {path}")


def main():
    """Main training script."""
    print("Loading data...")
    loaders = get_data_loaders(
        batch_size=config.STAGE1_BATCH_SIZE,
        num_workers=4
    )
    
    print("\nCreating model...")
    model = create_model(freeze_backbone=True)
    params = count_parameters(model)
    print(f"Model Parameters: {params['total']:,} (Trainable: {params['trainable']:,})")
    
    print("\nInitializing trainer...")
    trainer = EmotionTrainer(model, device=config.DEVICE)
    
    print("\nStarting two-stage training...")
    trainer.train_two_stage(loaders['train'], loaders['val'])
    
    print("\nTraining complete!")


if __name__ == '__main__':
    main()
