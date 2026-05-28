"""
04_train.py - Training Script for Crosswalk Classifier

This script handles:
  - Model initialization with proper seed setting
  - Training loop with loss tracking
  - Validation on test set during training
  - Checkpoint saving
  - GPU/CPU device handling
  - Early stopping to prevent overfitting

Training strategy:
  - Use BCE loss for binary classification
  - Adam optimizer (adaptive learning rates)
  - Learning rate scheduling with ReduceLROnPlateau
  - Track both loss and accuracy metrics
  - Save best model based on validation accuracy
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import json
from typing import Dict, Tuple, Optional
import sys
from datetime import datetime

# Import from local modules
from dataset import create_dataloaders
from model import create_model, set_seed


class Trainer:
    """Handles model training, validation, and checkpoint management."""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        test_loader: DataLoader,
        device: torch.device,
        dropout_rate: float = 0.5,
        seed: int = 42,
        checkpoint_dir: str = "./checkpoints"
    ):
        """
        Initialize the Trainer.
        
        Args:
            model: PyTorch model to train
            train_loader: Training data loader
            test_loader: Test/validation data loader
            device: Device to train on (cpu or cuda)
            dropout_rate: Dropout rate for model
            seed: Random seed for reproducibility
            checkpoint_dir: Directory to save model checkpoints
        """
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.seed = seed
        
        # Create checkpoint directory
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Loss function with class weights to handle severe imbalance
        # Dataset: 313 crosswalk vs 18,887 no-crosswalk (ratio 1:60)
        # Weights inversely proportional to class frequency
        class_weights = torch.tensor([60.0, 1.0]).to(device)  # Adjust if rebalanced!
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        print(f"✓ Weighted loss enabled (class weights: {class_weights.cpu().tolist()})")
        
        # ==========================================
        # 🔧 FIX 1: DIFFERENTIAL LEARNING RATES
        # ==========================================
        # Optimizer: Adam using parameter groups from model.py
        # This keeps the backbone LR low (1e-5) and the custom head higher (1e-4)
        self.optimizer = optim.Adam(
            self.model.get_parameter_groups(),
            weight_decay=1e-5  # L2 regularization
        )
        
        # Learning rate scheduler: reduce LR when validation plateaus
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',  # Maximize accuracy
            factor=0.5,  # Reduce LR by 50%
            patience=3,  # Wait 3 epochs before reducing
        )
        
        # Training history
        self.history = {
            "train_loss": [],
            "train_accuracy": [],
            "test_loss": [],
            "test_accuracy": []
        }
        
        print(f"\n🎯 Trainer initialized:")
        print(f"   Device: {device}")
        print(f"   Optimizer: Adam (with differential parameter groups)")
        print(f"   Loss function: CrossEntropyLoss")
    
    def train_epoch(self) -> Tuple[float, float]:
        """Train for one epoch."""
        self.model.train()  # Set to training mode
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (images, labels) in enumerate(self.train_loader):
            # Move data to device
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()  # Clear gradients
            logits = self.model(images)
            loss = self.criterion(logits, labels)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Track metrics
            total_loss += loss.item()
            _, predicted = torch.max(logits.data, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            
            # Print progress every N batches
            if (batch_idx + 1) % max(1, len(self.train_loader) // 5) == 0:
                print(f"   Batch {batch_idx + 1}/{len(self.train_loader)} | "
                      f"Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100 * correct / total
        
        return avg_loss, accuracy
    
    def validate(self) -> Tuple[float, float]:
        """Validate on test set."""
        self.model.eval()  # Set to evaluation mode
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():  # Don't compute gradients
            for images, labels in self.test_loader:
                # Move data to device
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                logits = self.model(images)
                loss = self.criterion(logits, labels)
                
                # Track metrics
                total_loss += loss.item()
                _, predicted = torch.max(logits.data, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
        
        avg_loss = total_loss / len(self.test_loader)
        accuracy = 100 * correct / total
        
        return avg_loss, accuracy
    
    def train(
        self,
        num_epochs: int = 20,
        early_stopping_patience: int = 5
    ) -> Dict:
        """Full training loop with early stopping."""
        print(f"\n{'='*60}")
        print(f"🚀 Starting Training ({num_epochs} epochs)")
        print(f"{'='*60}\n")
        
        best_test_acc = 0.0
        best_epoch = 0
        patience_counter = 0
        
        for epoch in range(num_epochs):
            print(f"Epoch {epoch + 1}/{num_epochs}")
            
            # Train
            train_loss, train_acc = self.train_epoch()
            
            # Validate
            test_loss, test_acc = self.validate()
            
            # Store history
            self.history["train_loss"].append(train_loss)
            self.history["train_accuracy"].append(train_acc)
            self.history["test_loss"].append(test_loss)
            self.history["test_accuracy"].append(test_acc)
            
            # Print metrics
            print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"  Test Loss:  {test_loss:.4f} | Test Acc:  {test_acc:.2f}%")
            print()
            
            # Learning rate scheduling
            self.scheduler.step(test_acc)
            
            # Early stopping check
            if test_acc > best_test_acc:
                best_test_acc = test_acc
                best_epoch = epoch
                patience_counter = 0
                
                # Save best model
                self._save_checkpoint(epoch, test_acc)
                print(f"  ✓ Best model saved (Val Acc: {test_acc:.2f}%)\n")
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print(f"\n⏸️  Early stopping triggered at epoch {epoch + 1}")
                    print(f"  Best validation accuracy: {best_test_acc:.2f}% (epoch {best_epoch + 1})")
                    
                    # ==========================================
                    # 🔧 FIX 2: RESTORE THE BEST MODEL
                    # ==========================================
                    print("  Restoring the best version of the model from disk...")
                    best_model_path = self.checkpoint_dir / f"model_epoch{best_epoch+1}_acc{best_test_acc:.2f}.pth"
                    if best_model_path.exists():
                        checkpoint = torch.load(best_model_path)
                        self.model.load_state_dict(checkpoint['model_state_dict'])
                    break
        
        print(f"\n{'='*60}")
        print(f"✅ Training Complete!")
        print(f"   Best Accuracy: {best_test_acc:.2f}% (epoch {best_epoch + 1})")
        print(f"{'='*60}\n")
        
        return self.history
    
    def _save_checkpoint(self, epoch: int, accuracy: float) -> None:
        """Save model checkpoint."""
        checkpoint_path = self.checkpoint_dir / f"model_epoch{epoch+1}_acc{accuracy:.2f}.pth"
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'accuracy': accuracy
        }, checkpoint_path)
    
    def save_history(self, filepath: str = "training_history.json") -> None:
        """Save training history to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.history, f, indent=2)
        print(f"✓ Training history saved to {filepath}")


def main():
    """Main training function."""
    
    # ========== CONFIGURATION ==========
    BATCH_SIZE = 32
    DROPOUT_RATE = 0.5
    NUM_EPOCHS = 20
    SEED = 42
    EARLY_STOPPING_PATIENCE = 5
    
    TRAIN_DIR = "./data/train"
    TEST_DIR = "./data/test"
    CHECKPOINT_DIR = "./checkpoints"
    # ===================================
    
    # Set seed for reproducibility
    set_seed(SEED)
    
    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🖥️  Device: {device}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   CUDA: {torch.version.cuda}")
    
    try:
        # Create dataloaders
        print(f"\n📊 Loading data...")
        train_loader, test_loader = create_dataloaders(
            train_dir=TRAIN_DIR,
            test_dir=TEST_DIR,
            batch_size=BATCH_SIZE,
            seed=SEED
        )
        
        # Create model
        print(f"\n🏗️  Building model...")
        model = create_model(
            device=device,
            dropout_rate=DROPOUT_RATE,
            pretrained=True,
            freeze_backbone=True, # Ensure your backbone is frozen per the differential rates logic
            seed=SEED
        )
        
        # Create trainer and train
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            device=device,
            dropout_rate=DROPOUT_RATE,
            seed=SEED,
            checkpoint_dir=CHECKPOINT_DIR
        )
        
        history = trainer.train(
            num_epochs=NUM_EPOCHS,
            early_stopping_patience=EARLY_STOPPING_PATIENCE
        )
        
        # Save history
        trainer.save_history("training_history.json")
        
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print(f"   Make sure you've run 01_data_split.py first to create train/test directories.")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        raise


if __name__ == "__main__":
    main()