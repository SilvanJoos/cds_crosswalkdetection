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
    def __init__(self, model: nn.Module, test_loader: DataLoader, device: torch.device, class_names: list = ["Crosswalk (0)", "No-Crosswalk (1)"]):
        self.model = model
        self.test_loader = test_loader
        self.device = device
        self.class_names = class_names
        self.criterion = nn.CrossEntropyLoss()
    
    def evaluate(self) -> Dict:
        self.model.eval()
        all_logits = []
        all_labels = []
        total_loss = 0.0
        
        print("\n📊 Evaluating model on test set...")
        
        with torch.no_grad():
            for images, labels in self.test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                logits = self.model(images)
                loss = self.criterion(logits, labels)
                
                total_loss += loss.item()
                all_logits.append(logits.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
        
        all_logits = np.concatenate(all_logits, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        all_preds = np.argmax(all_logits, axis=1)
        
        avg_loss = total_loss / len(self.test_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        
        # Calculate Macro F1 (aligns with what Optuna and Training optimized for)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='macro'
        )
        
        per_class_metrics = {}
        for class_idx, class_name in enumerate(self.class_names):
            mask = all_labels == class_idx
            if mask.sum() > 0:
                class_acc = (all_preds[mask] == all_labels[mask]).mean()
                per_class_metrics[class_name] = {
                    'accuracy': float(class_acc),
                    'count': int(mask.sum())
                }
        
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
            'f1_score_macro': float(f1),
            'roc_auc': float(roc_auc) if roc_auc else None,
            'per_class': per_class_metrics,
            'confusion_matrix': confusion_matrix(all_labels, all_preds).tolist(),
        }
        
        return metrics, all_logits, all_labels
    
    def print_report(self, metrics: Dict) -> None:
        print(f"\n{'='*60}")
        print(f"📈 TEST SET EVALUATION REPORT")
        print(f"{'='*60}\n")
        
        print(f"Overall Metrics:")
        print(f"  Loss:           {metrics['test_loss']:.4f}")
        print(f"  Accuracy:       {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
        print(f"  Macro F1-Score: {metrics['f1_score_macro']:.4f}")
        if metrics['roc_auc']:
            print(f"  ROC-AUC:        {metrics['roc_auc']:.4f}")
        
        print(f"\nPer-Class Performance:")
        for class_name, class_metrics in metrics['per_class'].items():
            print(f"  {class_name}:")
            print(f"    Accuracy: {class_metrics['accuracy']:.4f}")
            print(f"    Samples:  {class_metrics['count']}")
        
        print(f"\nConfusion Matrix:")
        cm = np.array(metrics['confusion_matrix'])
        print(f"                 Predicted")
        print(f"                {'Crosswalk':>15} {'No-Crosswalk':>15}")
        for i, row in enumerate(cm):
            print(f"Actual {self.class_names[i].split()[0]:>10} {row[0]:>15} {row[1]:>15}")
        
        print(f"\n{'='*60}\n")
    
    def plot_confusion_matrix(self, metrics: Dict, all_labels: np.ndarray = None, all_logits: np.ndarray = None, save_path: str = "confusion_matrix.png") -> None:
        cm = np.array(metrics['confusion_matrix'])
        if all_labels is not None and all_logits is not None:
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=self.class_names, yticklabels=self.class_names, ax=axes[0])
            axes[0].set_title('Confusion Matrix - F1 Optimized Model')
            axes[0].set_ylabel('True Label')
            axes[0].set_xlabel('Predicted Label')
            
            try:
                from sklearn.metrics import precision_recall_curve, auc
                probabilities = torch.softmax(torch.tensor(all_logits), dim=1).numpy()
                precision, recall, _ = precision_recall_curve(all_labels, probabilities[:, 1])
                pr_auc = auc(recall, precision)
                
                axes[1].plot(recall, precision, color='b', lw=2, label=f'PR curve (AUC = {pr_auc:.3f})')
                axes[1].set_xlabel('Recall')
                axes[1].set_ylabel('Precision')
                axes[1].set_title('Precision-Recall Curve')
                axes[1].legend(loc="lower left")
                axes[1].grid(True, alpha=0.3)
            except Exception as e:
                print(f"Could not plot PR curve: {e}")
                
            plt.tight_layout()
        else:
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=self.class_names, yticklabels=self.class_names)
            plt.title('Confusion Matrix - F1 Optimized Model')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.tight_layout()
            
        plt.savefig(save_path, dpi=300)
        print(f"✓ Evaluation plots saved to {save_path}")
        plt.close()
    
    def plot_training_curves(self, history_path: str = "training_history.json", save_path: str = "training_curves.png") -> None:
        with open(history_path, 'r') as f:
            history = json.load(f)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Determine which keys exist (handles both old and new training scripts)
        val_loss_key = 'val_loss' if 'val_loss' in history else 'test_loss'
        
        # Plot Loss
        axes[0].plot(history.get('train_loss', []), label='Train Loss', lw=2, marker='o')
        if val_loss_key in history:
            axes[0].plot(history[val_loss_key], label='Validation Loss', lw=2, marker='s')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training & Validation Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot Metric (F1 or Accuracy)
        if 'val_f1' in history:
            axes[1].plot(history['val_f1'], label='Validation Macro F1', color='green', lw=2, marker='s')
            axes[1].set_ylabel('F1-Score')
            axes[1].set_title('Validation F1-Score Progression')
        else:
            val_acc_key = 'test_accuracy' if 'test_accuracy' in history else 'val_accuracy'
            axes[1].plot(history.get('train_accuracy', []), label='Train Accuracy', lw=2, marker='o')
            if val_acc_key in history:
                axes[1].plot(history[val_acc_key], label='Validation Accuracy', lw=2, marker='s')
            axes[1].set_ylabel('Accuracy (%)')
            axes[1].set_title('Training & Validation Accuracy')
            
        axes[1].set_xlabel('Epoch')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        print(f"✓ Training curves saved to {save_path}")
        plt.close()

def load_checkpoint(checkpoint_path: str, model: nn.Module, device: torch.device) -> None:
    # NEW: Directly loads the state_dict, skipping the old dictionary keys
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    print(f"✓ Weights loaded successfully from {checkpoint_path}")

def main():
    TEST_DIR = "./data/test"
    CHECKPOINT_PATH = None
    BATCH_SIZE = 32
    SEED = 42
    
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🖥️  Device: {device}\n")
    
    try:
        print(f"📊 Loading test data...")
        _, test_loader = create_dataloaders("./data/train", TEST_DIR, batch_size=BATCH_SIZE, seed=SEED)
        
        print(f"\n🏗️  Building model...")
        # NEW: Aligned with the exact setup you used to train (unfrozen, 0.4 dropout)
        model = create_model(device=device, dropout_rate=0.4, freeze_backbone=False, seed=SEED)
        
        if CHECKPOINT_PATH is None:
            checkpoint_dir = Path("./checkpoints")
            checkpoints = sorted(checkpoint_dir.glob("*.pth"))
            if not checkpoints:
                print("❌ No checkpoints found in ./checkpoints/")
                sys.exit(1)
            CHECKPOINT_PATH = str(checkpoints[-1])
        
        load_checkpoint(CHECKPOINT_PATH, model, device)
        
        eval_dir = Path("eval") / Path(CHECKPOINT_PATH).stem
        eval_dir.mkdir(parents=True, exist_ok=True)
        
        evaluator = ModelEvaluator(model, test_loader, device)
        metrics, all_logits, all_labels = evaluator.evaluate()
        evaluator.print_report(metrics)
        
        print(f"\n🎨 Generating visualizations in {eval_dir}...")
        evaluator.plot_confusion_matrix(metrics, all_labels=all_labels, all_logits=all_logits, save_path=str(eval_dir / "confusion_matrix.png"))
        
        history_path = Path("training_history.json")
        if history_path.exists():
            evaluator.plot_training_curves(str(history_path), save_path=str(eval_dir / "training_curves.png"))
        
        metrics_file = eval_dir / "evaluation_metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
            
        print(f"✓ Metrics saved to {metrics_file}")
        print(f"\n✅ Evaluation complete!\n")
        
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        raise

if __name__ == "__main__":
    main()