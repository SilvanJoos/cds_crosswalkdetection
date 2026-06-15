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
import sys

# NEW IMPORT for F1 Score
from sklearn.metrics import f1_score

# Import from local modules
from dataset import create_dataloaders
from model import create_model, set_seed

class OptunaTrainer:
    def __init__(self, train_loader: DataLoader, test_loader: DataLoader, device: torch.device, seed: int = 42):
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.seed = seed
        
        # ALIGNED WITH TRAIN.PY: Injecting softened class weights
        class_weights = torch.tensor([15.0, 1.0]).to(device)
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    def train_and_evaluate(
        self,
        backbone_lr: float,
        head_lr: float,
        dropout_rate: float,
        weight_decay: float,
        batch_size: int,
        max_epochs: int = 15,
        trial: Trial = None
    ) -> float:
        set_seed(self.seed)
        
        # ALIGNED: freeze_backbone MUST be False for differential LRs to work
        model = create_model(
            device=self.device,
            dropout_rate=dropout_rate,
            pretrained=True,
            freeze_backbone=False, 
            seed=self.seed
        )
        
        # ALIGNED: Setup differential parameter groups based on Optuna's suggestions
        backbone_params = list(model.backbone.parameters())
        head_params = list(model.fc1.parameters()) + list(model.bn1.parameters()) + list(model.fc2.parameters())
        
        optimizer = optim.Adam([
            {'params': backbone_params, 'lr': backbone_lr},
            {'params': head_params, 'lr': head_lr}
        ], weight_decay=weight_decay)
        
        # Monitor MAX F1-score
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=2
        )
        
        best_val_f1 = 0.0
        patience_counter = 0
        early_stopping_patience = 3
        
        for epoch in range(max_epochs):
            # Train
            model.train()
            for images, labels in self.train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                logits = model(images)
                loss = self.criterion(logits, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            
            # Validate
            model.eval()
            all_preds = []
            all_labels = []
            
            with torch.no_grad():
                for images, labels in self.test_loader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    logits = model(images)
                    _, predicted = torch.max(logits.data, 1)
                    
                    all_preds.extend(predicted.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
            
            # ALIGNED: Track F1-Score instead of Accuracy
            val_f1 = f1_score(all_labels, all_preds, average='macro')
            scheduler.step(val_f1)
            
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= early_stopping_patience:
                break
            
            # Optuna pruning based on F1
            if trial is not None:
                trial.report(val_f1, epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned()
        
        return best_val_f1

def objective(trial: Trial, trainer: OptunaTrainer) -> float:
    # ALIGNED: Optuna now searches for TWO learning rates
    backbone_lr = trial.suggest_float('backbone_lr', 1e-6, 1e-4, log=True) # Keep it small
    head_lr = trial.suggest_float('head_lr', 1e-4, 1e-2, log=True)         # Can be larger
    
    dropout_rate = trial.suggest_float('dropout_rate', 0.2, 0.6)
    weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-3, log=True)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
    
    print(f"\n{'='*70}")
    print(f"Trial {trial.number}")
    print(f"  Backbone LR:   {backbone_lr:.2e}")
    print(f"  Head LR:       {head_lr:.2e}")
    print(f"  Dropout Rate:  {dropout_rate:.3f}")
    print(f"  Weight Decay:  {weight_decay:.2e}")
    print(f"  Batch Size:    {batch_size}")
    print(f"{'='*70}")
    
    try:
        train_loader, test_loader = create_dataloaders(
            train_dir="./data/train", test_dir="./data/test",
            batch_size=batch_size, seed=42
        )
        trainer.train_loader = train_loader
        trainer.test_loader = test_loader
        
        val_f1 = trainer.train_and_evaluate(
            backbone_lr=backbone_lr,
            head_lr=head_lr,
            dropout_rate=dropout_rate,
            weight_decay=weight_decay,
            batch_size=batch_size,
            max_epochs=15,
            trial=trial
        )
        
        print(f"  ✓ Best Validation F1-Score: {val_f1:.4f}\n")
        return val_f1
        
    except optuna.TrialPruned:
        print(f"  ⏸️  Trial pruned\n")
        raise

def main():
    N_TRIALS = 20
    N_JOBS = 1
    SEED = 42
    STUDY_NAME = "crosswalk_f1_optimization"
    
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"\n{'='*70}")
    print(f"🔬 OPTUNA F1-SCORE HYPERPARAMETER OPTIMIZATION")
    print(f"{'='*70}")
    
    try:
        train_loader, val_loader = create_dataloaders("./data/train", "./data/val", batch_size=32, seed=SEED)
        trainer = OptunaTrainer(train_loader, val_loader, device, SEED)
        
        sampler = TPESampler(seed=SEED)
        pruner = MedianPruner()
        
        study = optuna.create_study(
            direction='maximize', # We want the MAXIMUM F1-Score
            sampler=sampler,
            pruner=pruner,
            study_name=STUDY_NAME
        )
        
        study.optimize(lambda trial: objective(trial, trainer), n_trials=N_TRIALS, n_jobs=N_JOBS)
        
        best_trial = study.best_trial
        
        print(f"\n✅ OPTIMIZATION COMPLETE")
        print(f"📊 Best Trial Number: {best_trial.number}")
        print(f"🎯 Best F1-Score: {best_trial.value:.4f}\n")
        print(f"💡 Best Hyperparameters:")
        for param, value in best_trial.params.items():
            print(f"  {param}: {value}")
        best_params_path = "best_optuna_params.json"
        with open(best_params_path, 'w') as f:
            # We dump exactly the dictionary of the best parameters
            json.dump(best_trial.params, f, indent=4)
            
        print(f"\n✓ Successfully exported the optimal parameters to {best_params_path}")
        
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        raise

if __name__ == "__main__":
    main()()