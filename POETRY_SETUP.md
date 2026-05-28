# 🚀 Poetry Setup Guide for Crosswalk Classifier

## What is Poetry?
Poetry is a Python dependency manager that:
- Tracks exact versions of all packages (reproducible environment)
- Manages virtual environments automatically
- Creates a `poetry.lock` file (like `requirements.txt` but more deterministic)
- Perfect for hackathon projects!

## Installation

### Option 1: Install Poetry (Recommended)
```powershell
# Install Poetry via pip
pip install poetry

# Verify installation
poetry --version
```

### Option 2: Already have Poetry?
```powershell
poetry --version
```

## Setup Your Environment

### Step 1: Create Virtual Environment and Install Dependencies
```powershell
cd C:\Users\silva\Documents\DeepLearning

# Install all dependencies from pyproject.toml
poetry install
```

This will:
- Create a virtual environment specific to your project
- Install all packages with exact versions
- Create a `poetry.lock` file for reproducibility

### Step 2: Run Commands with Poetry

**Option A: Activate the virtual environment**
```powershell
poetry shell

# Now run Python commands directly
python 00_rebalance_dataset.py
python 04_train.py
python 05_evaluate.py
```

**Option B: Run directly without activating** (recommended)
```powershell
poetry run python 00_rebalance_dataset.py
poetry run python 04_train.py
poetry run python 05_evaluate.py
```

## Your Project Files

```
DeepLearning/
├── pyproject.toml          ← Poetry configuration (with all dependencies)
├── poetry.lock             ← Auto-generated, version-locked dependencies
├── 00_rebalance_dataset.py ← Handle class imbalance
├── 01_data_split.py        ← Split train/test (ALREADY RUN ✓)
├── 02_model.py             ← Model architecture
├── 03_dataset.py           ← Custom dataset with augmentation
├── 04_train.py             ← Training script (uses weighted loss)
├── 05_evaluate.py          ← Evaluation & visualizations
├── 06_hyperparameter_tuning.py ← Optuna optimization
└── README.md               ← Full documentation
```

## Next Steps

### 1. Install Dependencies
```powershell
poetry install
```

### 2. Analyze Class Imbalance (Optional)
```powershell
poetry run python 00_rebalance_dataset.py
```

This will show you the imbalance problem and offer solutions.

### 3. Train Model
```powershell
poetry run python 04_train.py
```

This uses **weighted loss** to handle the 1:60 class imbalance automatically.

### 4. Evaluate Results
```powershell
poetry run python 05_evaluate.py
```

### 5. Optimize Hyperparameters (Optional)
```powershell
poetry run python 06_hyperparameter_tuning.py
```

## Troubleshooting

### Poetry not found?
```powershell
pip install poetry
```

### Want to add a new package?
```powershell
poetry add package_name
```

### Want to update all packages?
```powershell
poetry update
```

### Check what's installed?
```powershell
poetry show
```

---

## About Class Imbalance Handling

Your dataset has a **severe class imbalance** (313 vs 18,887 or 1.6% crosswalks):

### ✅ What I've Implemented:
- **Weighted CrossEntropyLoss** in `04_train.py`
- Class weights: [60.0, 1.0] (crosswalk gets 60x penalty for misclassification)
- This ensures the model actually learns to detect crosswalks, not just say "no crosswalk"

### 🔧 If You Want to Rebalance Data:
Run `00_rebalance_dataset.py` for options:
1. **Weighted loss** (already implemented) ✓ RECOMMENDED
2. **Stratified resampling** (1:3 ratio, balanced training)
3. **Oversampling** (duplicate minority class)
4. **Undersampling** (remove majority class)

### 📊 Why This Matters for Your Report:
Explain in your hackathon report:
- The imbalance problem (313 vs 18,887)
- Your solution (weighted loss)
- How it affected metrics (precision/recall trade-off)
- Why accuracy alone is misleading (always predict "no-crosswalk" → 98% accurate!)

---

## Quick Reference

| Command | Description |
|---------|-------------|
| `poetry install` | Install dependencies |
| `poetry shell` | Activate virtual environment |
| `poetry run python script.py` | Run script without activating shell |
| `poetry add package` | Add new dependency |
| `poetry show` | List installed packages |
| `poetry lock` | Update poetry.lock file |

---

Ready to train! 🚀
