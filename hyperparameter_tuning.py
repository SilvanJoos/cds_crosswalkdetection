"""
06_hyperparameter_tuning.py - Optuna Hyperparameter Optimization

This script uses Optuna to automatically find optimal hyperparameters:
  - Learning rate (1e-5 to 1e-2)
  - Batch size (16, 32, 64)
  - Dropout rate (0.1 to 0.7)
  - Weight decay for L2 regularization
  - Number of epochs before early stopping

Why Optuna?
  - Efficient sampling (Bayesian optimization)
  - Automatic pruning (stops unpromising trials early)
  - Easy visualization and export
  - Reproducible results with seeds

This is essential for your hackathon report - show the optimization process!
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import optuna
from optuna.trial import Trial
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
import numpy as np
from pathlib import Path
import json
from typing import Dict, Tuple
import sys

# Import from local modules
from dataset import create_dataloaders
from model import create_model, set_seed


class OptunaTrainer:
    """Trainer for Optuna hyperparameter optimization trials."""
    
    def __init__(
        self,
        train_loader: DataLoader,
        test_loader: DataLoader,
        device: torch.device,
        seed: int = 42
    ):
        """Initialize Optuna trainer."""
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.seed = seed
        self.criterion = nn.CrossEntropyLoss()
    
    def train_and_evaluate(
        self,
        learning_rate: float,
        dropout_rate: float,
        weight_decay: float,
        batch_size: int,
        max_epochs: int = 15,
        trial: Trial = None
    ) -> float:
        """
        Train model with given hyperparameters and return validation accuracy.
        
        Args:
            learning_rate: Learning rate for optimizer
            dropout_rate: Dropout probability
            weight_decay: L2 regularization weight
            batch_size: Batch size
            max_epochs: Maximum epochs to train
            trial: Optuna trial object for pruning
            
        Returns:
            Best validation accuracy achieved
        """
        set_seed(self.seed)
        
        # Create model with current hyperparameters
        model = create_model(
            device=self.device,
            dropout_rate=dropout_rate,
            pretrained=True,
            freeze_backbone=True,
            seed=self.seed
        )
        
        # Create optimizer and scheduler
        optimizer = optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=0.5,
            patience=2,
            verbose=False
        )
        
        best_val_acc = 0.0
        patience_counter = 0
        early_stopping_patience = 3
        
        for epoch in range(max_epochs):
            # Train
            model.train()
            train_loss = 0.0
            for images, labels in self.train_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                optimizer.zero_grad()
                logits = model(images)
                loss = self.criterion(logits, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                train_loss += loss.item()
            
            # Validate
            model.eval()
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for images, labels in self.test_loader:
                    images = images.to(self.device)
                    labels = labels.to(self.device)
                    
                    logits = model(images)
                    _, predicted = torch.max(logits.data, 1)
                    val_correct += (predicted == labels).sum().item()
                    val_total += labels.size(0)
            
            val_acc = val_correct / val_total
            scheduler.step(val_acc)
            
            # Track best accuracy
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= early_stopping_patience:
                break
            
            # Optuna pruning
            if trial is not None:
                trial.report(val_acc, epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned()
        
        return best_val_acc


def objective(trial: Trial, trainer: OptunaTrainer) -> float:
    """
    Objective function for Optuna optimization.
    
    Optuna will call this function repeatedly with different hyperparameters
    and find the combination that maximizes validation accuracy.
    
    Args:
        trial: Optuna trial object
        trainer: OptunaTrainer instance
        
    Returns:
        Validation accuracy (to be maximized)
    """
    
    # Suggest hyperparameters
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.7)
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
    
    print(f"\n{'='*70}")
    print(f"Trial {trial.number}")
    print(f"  Learning Rate: {learning_rate:.2e}")
    print(f"  Dropout Rate:  {dropout_rate:.3f}")
    print(f"  Weight Decay:  {weight_decay:.2e}")
    print(f"  Batch Size:    {batch_size}")
    print(f"{'='*70}")
    
    try:
        # Recreate dataloaders with new batch size
        train_loader, test_loader = create_dataloaders(
            train_dir="./data/train",
            test_dir="./data/test",
            batch_size=batch_size,
            seed=42
        )
        
        trainer.train_loader = train_loader
        trainer.test_loader = test_loader
        
        # Train and evaluate
        val_accuracy = trainer.train_and_evaluate(
            learning_rate=learning_rate,
            dropout_rate=dropout_rate,
            weight_decay=weight_decay,
            batch_size=batch_size,
            max_epochs=15,
            trial=trial
        )
        
        print(f"  ✓ Best Validation Accuracy: {val_accuracy:.4f} ({val_accuracy*100:.2f}%)\n")
        
        return val_accuracy
        
    except optuna.TrialPruned:
        print(f"  ⏸️  Trial pruned\n")
        raise
    except Exception as e:
        print(f"  ❌ Error in trial: {e}\n")
        raise


def main():
    """Main Optuna optimization function."""
    
    # ========== CONFIGURATION ==========
    N_TRIALS = 20  # Number of trials to run (increase for better results)
    N_JOBS = 1  # Parallel jobs (1 for GPU safety)
    SEED = 42
    STUDY_NAME = "crosswalk_optimization"
    # ===================================
    
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"\n{'='*70}")
    print(f"🔬 OPTUNA HYPERPARAMETER OPTIMIZATION")
    print(f"{'='*70}")
    print(f"Device: {device}")
    print(f"Number of trials: {N_TRIALS}")
    print(f"{'='*70}\n")
    
    try:
        # Create initial dataloaders (batch_size will be adjusted per trial)
        print(f"📊 Loading data...")
        train_loader, test_loader = create_dataloaders(
            train_dir="./data/train",
            test_dir="./data/test",
            batch_size=32,  # Default, will be changed by trials
            seed=SEED
        )
        
        # Create trainer
        trainer = OptunaTrainer(
            train_loader=train_loader,
            test_loader=test_loader,
            device=device,
            seed=SEED
        )
        
        # Create Optuna study
        sampler = TPESampler(seed=SEED)  # TPE: Tree-structured Parzen Estimator
        pruner = MedianPruner()  # Prune underperforming trials
        
        study = optuna.create_study(
            direction='maximize',  # Maximize accuracy
            sampler=sampler,
            pruner=pruner,
            study_name=STUDY_NAME
        )
        
        # Optimize
        print(f"🚀 Starting optimization...\n")
        study.optimize(
            lambda trial: objective(trial, trainer),
            n_trials=N_TRIALS,
            n_jobs=N_JOBS,
            show_progress_bar=True
        )
        
        # Print results
        print(f"\n{'='*70}")
        print(f"✅ OPTIMIZATION COMPLETE")
        print(f"{'='*70}\n")
        
        best_trial = study.best_trial
        
        print(f"📊 Best Trial:")
        print(f"  Trial Number: {best_trial.number}")
        print(f"  Best Accuracy: {best_trial.value:.4f} ({best_trial.value*100:.2f}%)\n")
        
        print(f"🎯 Best Hyperparameters:")
        for param, value in best_trial.params.items():
            if 'rate' in param.lower():
                print(f"  {param}: {value:.4f}")
            elif 'decay' in param.lower():
                print(f"  {param}: {value:.2e}")
            else:
                print(f"  {param}: {value}")
        
        print(f"\n{'='*70}\n")
        
        # Save results
        results = {
            'best_accuracy': float(best_trial.value),
            'best_params': best_trial.params,
            'best_trial': best_trial.number,
            'n_trials': len(study.trials)
        }
        
        with open("optuna_results.json", 'w') as f:
            json.dump(results, f, indent=2)
        print(f"✓ Results saved to optuna_results.json")
        
        # Plot optimization history
        try:
            fig = optuna.visualization.plot_optimization_history(study).to_html()
            with open("optuna_history.html", 'w') as f:
                f.write(fig)
            print(f"✓ Optimization history saved to optuna_history.html")
        except:
            print(f"⚠️  Could not generate optimization history plot")
        
        # Plot parameter importance
        try:
            fig = optuna.visualization.plot_param_importances(study).to_html()
            with open("optuna_param_importance.html", 'w') as f:
                f.write(fig)
            print(f"✓ Parameter importance saved to optuna_param_importance.html")
        except:
            print(f"⚠️  Could not generate parameter importance plot")
        
        print(f"\n💡 Next Step: Use the best hyperparameters in 04_train.py for final training!\n")
        
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        raise


if __name__ == "__main__":
    main()
