import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import json
import sys
import numpy as np

# NEW IMPORTS FOR METRICS AND VISUALIZATION
from sklearn.metrics import f1_score, confusion_matrix, precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt
import seaborn as sns

from dataset import create_dataloaders
from model import create_model, set_seed

class Trainer:
    def __init__(self, model, train_loader, test_loader, device, checkpoint_dir="./checkpoints"):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Softened weights to balance Precision and Recall
        class_weights = torch.tensor([15.0, 1.0]).to(device)
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        
        self.optimizer = optim.Adam(self.model.get_parameter_groups(), weight_decay=1e-4)
        
        # Monitor MAX F1-score for learning rate reduction
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=2
        )
        self.history = {"train_loss": [], "val_loss": [], "val_f1": []}
    
    def train_epoch(self):
        self.model.train()
        total_loss = 0.0
        
        for batch_idx, (images, labels) in enumerate(self.train_loader):
            images, labels = images.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            logits = self.model(images)
            loss = self.criterion(logits, labels)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            total_loss += loss.item()
            
            # --- THE FANCY COMMANDLINE UPDATES ---
            # Prints progress 5 times per epoch
            if (batch_idx + 1) % max(1, len(self.train_loader) // 5) == 0:
                print(f"   Batch {batch_idx + 1:03d}/{len(self.train_loader):03d} | "
                      f"Current Loss: {loss.item():.4f}")
                
        return total_loss / len(self.train_loader)
    
    def validate(self):
        self.model.eval()
        total_loss = 0.0
        
        all_preds = []
        all_labels = []
        all_probs = [] # Keep track of probabilities for the PR curve
        
        with torch.no_grad():
            for images, labels in self.test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                logits = self.model(images)
                loss = self.criterion(logits, labels)
                total_loss += loss.item()
                
                # Get probabilities using Softmax
                probs = torch.softmax(logits, dim=1)
                
                _, predicted = torch.max(logits.data, 1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                # Store probability of the positive class (Class 0: crosswalk)
                all_probs.extend(probs[:, 0].cpu().numpy())
                
        avg_loss = total_loss / len(self.test_loader)
        
        # Calculate Macro F1 Score (balances performance across both majority and minority classes)
        val_f1 = f1_score(all_labels, all_preds, average='macro')
        
        return avg_loss, val_f1, all_labels, all_preds, all_probs
    
    def train(self, num_epochs=20, patience=5):
        best_f1 = 0.0
        best_epoch = 0
        patience_counter = 0
        
        best_labels, best_preds, best_probs = None, None, None
        
        for epoch in range(num_epochs):
            train_loss = self.train_epoch()
            val_loss, val_f1, labels, preds, probs = self.validate()
            
            print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val F1-Score: {val_f1:.4f}")
            
            # Step the scheduler based on F1-Score
            self.scheduler.step(val_f1)
            
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["val_f1"].append(val_f1)
            
            # Early stopping based on F1-SCORE, not accuracy
            if val_f1 > best_f1:
                best_f1 = val_f1
                best_epoch = epoch
                patience_counter = 0
                best_labels, best_preds, best_probs = labels, preds, probs
                
                torch.save(self.model.state_dict(), self.checkpoint_dir / "best_model.pth")
                print(f"  ✓ Best model saved! (F1: {best_f1:.4f})")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"\n⏸️ Early stopping at epoch {epoch + 1}")
                    self.model.load_state_dict(torch.load(self.checkpoint_dir / "best_model.pth"))
                    break
                    
        # Generate Visualizations at the end of training
        print("\n📊 Generating Evaluation Visualizations...")
        self.plot_evaluation_metrics(best_labels, best_preds, best_probs)
        
        return self.history

    def plot_evaluation_metrics(self, labels, preds, probs):
        """Creates the visualizations that the judges will actually care about."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 1. Confusion Matrix
        cm = confusion_matrix(labels, preds)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1,
                    xticklabels=['Crosswalk (0)', 'No Crosswalk (1)'],
                    yticklabels=['Crosswalk (0)', 'No Crosswalk (1)'])
        ax1.set_title('Confusion Matrix')
        ax1.set_ylabel('Actual Label')
        ax1.set_xlabel('Predicted Label')
        
        # 2. Precision-Recall Curve (Crucial for imbalanced data)
        # Note: Scikit-learn expects the positive class to be '1'. 
        # In your dataset, '0' is crosswalk. So we invert labels for the PR curve.
        inverted_labels = [1 if l == 0 else 0 for l in labels]
        
        precision, recall, _ = precision_recall_curve(inverted_labels, probs)
        pr_auc = average_precision_score(inverted_labels, probs)
        
        ax2.plot(recall, precision, color='darkorange', lw=2, 
                 label=f'PR curve (area = {pr_auc:.3f})')
        ax2.set_xlim([0.0, 1.0])
        ax2.set_ylim([0.0, 1.05])
        ax2.set_xlabel('Recall (Found all Crosswalks?)')
        ax2.set_ylabel('Precision (Are they actually Crosswalks?)')
        ax2.set_title('Precision-Recall Curve for Crosswalks')
        ax2.legend(loc="lower left")
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(self.checkpoint_dir / 'evaluation_metrics.png')
        print(f"✓ Saved visualizations to {self.checkpoint_dir / 'evaluation_metrics.png'}")
    def save_history(self, filepath: str = "training_history.json") -> None:
        with open(filepath, 'w') as f:
            json.dump(self.history, f, indent=2)
        print(f"✓ Training history saved to {filepath}")


def main():
    BATCH_SIZE = 32
    DROPOUT_RATE = 0.4
    NUM_EPOCHS = 20
    SEED = 42
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = create_dataloaders("./data/train", "./data/test", batch_size=BATCH_SIZE)
    
    model = create_model(device, dropout_rate=DROPOUT_RATE, pretrained=True, freeze_backbone=False)
    
    trainer = Trainer(model, train_loader, test_loader, device)
    trainer.train(num_epochs=NUM_EPOCHS)
    trainer.save_history("training_history.json")

if __name__ == "__main__":
    main()