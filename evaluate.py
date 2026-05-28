"""
05_evaluate.py - Model Evaluation and Visualization

This script provides comprehensive evaluation of the trained model:
  - Load trained model checkpoint
  - Compute metrics on test set (accuracy, precision, recall, F1)
  - Generate Confusion Matrix
  - Plot training/validation loss and accuracy
  - Per-class performance analysis

These visualizations are essential for your hackathon report!
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix, classification_report, accuracy_score,
    precision_recall_fscore_support, roc_auc_score, roc_curve
)
import seaborn as sns
from typing import Dict, Tuple
import sys

# Import from local modules
from dataset import create_dataloaders
from model import create_model, set_seed


class ModelEvaluator:
    """Evaluate trained model and generate visualizations."""
    
    def __init__(
        self,
        model: nn.Module,
        test_loader: DataLoader,
        device: torch.device,
        class_names: list = ["Crosswalk", "No-Crosswalk"]
    ):
        """
        Initialize evaluator.
        
        Args:
            model: Trained PyTorch model
            test_loader: Test data loader
            device: Device to evaluate on
            class_names: Names for the two classes
        """
        self.model = model
        self.test_loader = test_loader
        self.device = device
        self.class_names = class_names
        self.criterion = nn.CrossEntropyLoss()
    
    def evaluate(self) -> Dict:
        """
        Full evaluation on test set.
        
        Returns:
            Dictionary with evaluation metrics
        """
        self.model.eval()
        
        all_logits = []
        all_labels = []
        total_loss = 0.0
        
        print("\n📊 Evaluating model on test set...")
        
        with torch.no_grad():
            for images, labels in self.test_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                logits = self.model(images)
                loss = self.criterion(logits, labels)
                
                total_loss += loss.item()
                all_logits.append(logits.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
        
        # Combine all batches
        all_logits = np.concatenate(all_logits, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        all_preds = np.argmax(all_logits, axis=1)
        
        # Compute metrics
        avg_loss = total_loss / len(self.test_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='weighted'
        )
        
        # Per-class metrics
        per_class_metrics = {}
        for class_idx, class_name in enumerate(self.class_names):
            mask = all_labels == class_idx
            if mask.sum() > 0:
                class_acc = (all_preds[mask] == all_labels[mask]).mean()
                per_class_metrics[class_name] = {
                    'accuracy': float(class_acc),
                    'count': int(mask.sum())
                }
        
        # Compute ROC-AUC for binary classification
        try:
            probabilities = torch.softmax(torch.tensor(all_logits), dim=1).numpy()
            roc_auc = roc_auc_score(all_labels, probabilities[:, 1])
        except:
            roc_auc = None
        
        metrics = {
            'test_loss': float(avg_loss),
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'roc_auc': float(roc_auc) if roc_auc else None,
            'per_class': per_class_metrics,
            'confusion_matrix': confusion_matrix(all_labels, all_preds).tolist(),
            'predictions': all_preds.tolist(),
            'ground_truth': all_labels.tolist()
        }
        
        return metrics, all_logits, all_labels
    
    def print_report(self, metrics: Dict) -> None:
        """Print detailed evaluation report."""
        print(f"\n{'='*60}")
        print(f"📈 TEST SET EVALUATION REPORT")
        print(f"{'='*60}\n")
        
        print(f"Overall Metrics:")
        print(f"  Loss:      {metrics['test_loss']:.4f}")
        print(f"  Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1-Score:  {metrics['f1_score']:.4f}")
        if metrics['roc_auc']:
            print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
        
        print(f"\nPer-Class Performance:")
        for class_name, class_metrics in metrics['per_class'].items():
            print(f"  {class_name}:")
            print(f"    Accuracy: {class_metrics['accuracy']:.4f}")
            print(f"    Samples:  {class_metrics['count']}")
        
        print(f"\nConfusion Matrix:")
        cm = np.array(metrics['confusion_matrix'])
        print(f"                 Predicted")
        print(f"                {self.class_names[0]:>15} {self.class_names[1]:>15}")
        for i, row in enumerate(cm):
            print(f"Actual {self.class_names[i]:>10} {row[0]:>15} {row[1]:>15}")
        
        print(f"\n{'='*60}\n")
    
    def plot_confusion_matrix(self, metrics: Dict, save_path: str = "confusion_matrix.png") -> None:
        """
        Plot and save confusion matrix heatmap.
        
        Args:
            metrics: Evaluation metrics dictionary
            save_path: Path to save the figure
        """
        cm = np.array(metrics['confusion_matrix'])
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.class_names,
                   yticklabels=self.class_names,
                   cbar_kws={'label': 'Count'})
        plt.title('Confusion Matrix - Crosswalk Classifier', fontsize=14, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Confusion matrix saved to {save_path}")
        plt.close()
    
    def plot_training_curves(self, history_path: str = "training_history.json", 
                            save_path: str = "training_curves.png") -> None:
        """
        Plot training and validation loss/accuracy curves.
        
        Args:
            history_path: Path to training history JSON file
            save_path: Path to save the figure
        """
        # Load history
        with open(history_path, 'r') as f:
            history = json.load(f)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Loss plot
        axes[0].plot(history['train_loss'], label='Train Loss', linewidth=2, marker='o', markersize=4)
        axes[0].plot(history['test_loss'], label='Test Loss', linewidth=2, marker='s', markersize=4)
        axes[0].set_xlabel('Epoch', fontsize=12)
        axes[0].set_ylabel('Loss', fontsize=12)
        axes[0].set_title('Training & Validation Loss', fontsize=13, fontweight='bold')
        axes[0].legend(fontsize=11)
        axes[0].grid(True, alpha=0.3)
        
        # Accuracy plot
        axes[1].plot(history['train_accuracy'], label='Train Accuracy', linewidth=2, marker='o', markersize=4)
        axes[1].plot(history['test_accuracy'], label='Test Accuracy', linewidth=2, marker='s', markersize=4)
        axes[1].set_xlabel('Epoch', fontsize=12)
        axes[1].set_ylabel('Accuracy (%)', fontsize=12)
        axes[1].set_title('Training & Validation Accuracy', fontsize=13, fontweight='bold')
        axes[1].legend(fontsize=11)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Training curves saved to {save_path}")
        plt.close()
    
    def plot_roc_curve(self, all_logits: np.ndarray, all_labels: np.ndarray,
                      save_path: str = "roc_curve.png") -> None:
        """
        Plot ROC curve for binary classification.
        
        Args:
            all_logits: Model output logits
            all_labels: Ground truth labels
            save_path: Path to save the figure
        """
        probabilities = torch.softmax(torch.tensor(all_logits), dim=1).numpy()
        fpr, tpr, thresholds = roc_curve(all_labels, probabilities[:, 1])
        roc_auc = roc_auc_score(all_labels, probabilities[:, 1])
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curve - Crosswalk Classifier', fontsize=13, fontweight='bold')
        plt.legend(loc="lower right", fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ ROC curve saved to {save_path}")
        plt.close()


def load_checkpoint(checkpoint_path: str, model: nn.Module, device: torch.device) -> None:
    """Load model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"✓ Model loaded from {checkpoint_path}")
    print(f"  Epoch: {checkpoint['epoch'] + 1}, Accuracy: {checkpoint['accuracy']:.2f}%")


def main():
    """Main evaluation function."""
    
    # ========== CONFIGURATION ==========
    TEST_DIR = "./data/test"
    CHECKPOINT_PATH = None  # Will be auto-detected if None
    BATCH_SIZE = 32
    SEED = 42
    # ===================================
    
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🖥️  Device: {device}\n")
    
    try:
        # Create test dataloader
        print(f"📊 Loading test data...")
        _, test_loader = create_dataloaders(
            train_dir="./data/train",
            test_dir=TEST_DIR,
            batch_size=BATCH_SIZE,
            seed=SEED
        )
        
        # Create model
        print(f"\n🏗️  Building model...")
        model = create_model(device=device, seed=SEED)
        
        # Load checkpoint
        if CHECKPOINT_PATH is None:
            checkpoint_dir = Path("./checkpoints")
            checkpoints = sorted(checkpoint_dir.glob("*.pth"))
            if not checkpoints:
                print("❌ No checkpoints found in ./checkpoints/")
                sys.exit(1)
            CHECKPOINT_PATH = str(checkpoints[-1])  # Load latest checkpoint
        
        load_checkpoint(CHECKPOINT_PATH, model, device)
        
        # Evaluate
        evaluator = ModelEvaluator(model, test_loader, device)
        metrics, all_logits, all_labels = evaluator.evaluate()
        
        # Print report
        evaluator.print_report(metrics)
        
        # Generate visualizations
        print(f"\n🎨 Generating visualizations...")
        evaluator.plot_confusion_matrix(metrics)
        
        # Check if training history exists
        history_path = Path("training_history.json")
        if history_path.exists():
            evaluator.plot_training_curves(str(history_path))
            evaluator.plot_roc_curve(all_logits, all_labels)
        else:
            print(f"⚠️  training_history.json not found. Skipping training curves.")
        
        # Save metrics
        with open("evaluation_metrics.json", 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"✓ Metrics saved to evaluation_metrics.json")
        
        print(f"\n✅ Evaluation complete!\n")
        
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        raise


if __name__ == "__main__":
    main()
