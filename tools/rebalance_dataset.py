import os
import shutil
import random
from pathlib import Path
from typing import Tuple
import numpy as np


def analyze_imbalance(train_dir: str, test_dir: str) -> dict:
    """Analyze class imbalance in dataset."""
    stats = {}
    
    for split_name, split_dir in [("train", train_dir), ("test", test_dir)]:
        y_count = len(list(Path(split_dir).glob("y/*")))
        n_count = len(list(Path(split_dir).glob("n/*")))
        total = y_count + n_count
        ratio = n_count / y_count if y_count > 0 else 0
        
        stats[split_name] = {
            "crosswalk": y_count,
            "no_crosswalk": n_count,
            "total": total,
            "ratio": ratio,
            "crosswalk_pct": 100 * y_count / total if total > 0 else 0
        }
    
    return stats


def print_analysis(stats: dict) -> None:
    """Print detailed imbalance analysis."""
    print(f"\n{'='*70}")
    print(f"📊 CLASS IMBALANCE ANALYSIS")
    print(f"{'='*70}\n")
    
    for split, data in stats.items():
        print(f"{split.upper()} SET:")
        print(f"  Crosswalk (y):     {data['crosswalk']:>6} ({data['crosswalk_pct']:>5.2f}%)")
        print(f"  No-Crosswalk (n):  {data['no_crosswalk']:>6}")
        print(f"  Imbalance Ratio:   1:{data['ratio']:.1f}")
        print(f"  Total:             {data['total']:>6}\n")
    
    print(f"{'='*70}")
    print(f"⚠️  ANALYSIS: Dataset is SEVERELY imbalanced (1:60 ratio)")
    print(f"{'='*70}\n")
    
    print(f"🔴 Problem: Naive model predicting all \"no-crosswalk\" would achieve:")
    print(f"   Accuracy: {stats['train']['no_crosswalk'] / stats['train']['total'] * 100:.2f}% ❌")
    print(f"   (But we need to actually DETECT crosswalks!)\n")


def oversample_minority(
    train_dir: str,
    target_ratio: float = 1.0,
    output_dir: str = "./data_rebalanced_oversample"
) -> None:
    """
    Oversample minority class by duplicating images.
    
    Args:
        train_dir: Path to training directory
        target_ratio: Target minority/majority ratio (1.0 = balanced)
        output_dir: Directory to save rebalanced data
    """
    print(f"\n🔄 STRATEGY 1: OVERSAMPLING (Duplicate minority class)")
    print(f"{'='*70}\n")
    
    train_path = Path(train_dir)
    output_path = Path(output_dir) / "train"
    
    # Count images
    y_images = list((train_path / "y").glob("*"))
    n_images = list((train_path / "n").glob("*"))
    
    y_count = len(y_images)
    n_count = len(n_images)
    
    # Calculate how many duplicates needed
    target_y_count = int(n_count * target_ratio)
    duplicates_needed = max(0, target_y_count - y_count)
    
    print(f"Original counts:")
    print(f"  Crosswalk (y): {y_count}")
    print(f"  No-Crosswalk (n): {n_count}")
    print(f"  Ratio: 1:{n_count/y_count:.2f}\n")
    
    print(f"Target ratio: 1:{target_ratio:.2f}")
    print(f"Duplicates needed: {duplicates_needed}\n")
    
    # Copy original files
    (output_path / "y").mkdir(parents=True, exist_ok=True)
    (output_path / "n").mkdir(parents=True, exist_ok=True)
    
    for img in n_images:
        shutil.copy2(img, output_path / "n" / img.name)
    
    for img in y_images:
        shutil.copy2(img, output_path / "y" / img.name)
    
    # Create duplicates with modified names
    for i in range(duplicates_needed):
        source = random.choice(y_images)
        # Create copy with suffix
        new_name = f"{source.stem}_dup{i}{source.suffix}"
        shutil.copy2(source, output_path / "y" / new_name)
    
    final_y = len(list((output_path / "y").glob("*")))
    print(f"✓ Oversampling complete!")
    print(f"  Final crosswalk count: {final_y}")
    print(f"  Final ratio: 1:{n_count/final_y:.2f}\n")


def undersample_majority(
    train_dir: str,
    target_ratio: float = 3.0,
    output_dir: str = "./data_rebalanced_undersample"
) -> None:
    """
    Undersample majority class by removing images.
    
    Args:
        train_dir: Path to training directory
        target_ratio: Target majority/minority ratio (3.0 = 1:3)
        output_dir: Directory to save rebalanced data
    """
    print(f"\n🔄 STRATEGY 2: UNDERSAMPLING (Remove majority class)")
    print(f"{'='*70}\n")
    
    train_path = Path(train_dir)
    output_path = Path(output_dir) / "train"
    
    # Count images
    y_images = list((train_path / "y").glob("*"))
    n_images = list((train_path / "n").glob("*"))
    
    y_count = len(y_images)
    n_count = len(n_images)
    
    # Calculate target n_count
    target_n_count = int(y_count * target_ratio)
    keep_count = target_n_count
    
    print(f"Original counts:")
    print(f"  Crosswalk (y): {y_count}")
    print(f"  No-Crosswalk (n): {n_count}")
    print(f"  Ratio: 1:{n_count/y_count:.2f}\n")
    
    print(f"Target ratio: 1:{target_ratio:.2f}")
    print(f"Removing: {n_count - keep_count} images\n")
    
    # Create output directories
    (output_path / "y").mkdir(parents=True, exist_ok=True)
    (output_path / "n").mkdir(parents=True, exist_ok=True)
    
    # Copy all crosswalk images
    for img in y_images:
        shutil.copy2(img, output_path / "y" / img.name)
    
    # Copy subset of no-crosswalk images
    selected_n = random.sample(n_images, keep_count)
    for img in selected_n:
        shutil.copy2(img, output_path / "n" / img.name)
    
    print(f"✓ Undersampling complete!")
    print(f"  Final no-crosswalk count: {keep_count}")
    print(f"  Final ratio: 1:{keep_count/y_count:.2f}\n")


def stratified_resample(
    train_dir: str,
    test_dir: str,
    target_ratio: float = 3.0,
    output_dir: str = "./data_rebalanced_stratified"
) -> None:
    """
    Hybrid approach: Balance to intermediate ratio on TRAINING set,
    keep test set balanced for fair evaluation.
    
    Args:
        train_dir: Path to training directory
        test_dir: Path to test directory
        target_ratio: Target majority/minority ratio (3.0 = 1:3)
        output_dir: Directory to save rebalanced data
    """
    print(f"\n🔄 STRATEGY 3: STRATIFIED RESAMPLING (Hybrid)")
    print(f"{'='*70}\n")
    
    train_path = Path(train_dir)
    test_path = Path(test_dir)
    output_path = Path(output_dir)
    
    # Count original
    y_train = list((train_path / "y").glob("*"))
    n_train = list((train_path / "n").glob("*"))
    
    y_test = list((test_path / "y").glob("*"))
    n_test = list((test_path / "n").glob("*"))
    
    y_train_count = len(y_train)
    n_train_count = len(n_train)
    
    target_n_count = int(y_train_count * target_ratio)
    
    print(f"Original TRAIN set:")
    print(f"  Crosswalk (y): {y_train_count}")
    print(f"  No-Crosswalk (n): {n_train_count}")
    print(f"  Ratio: 1:{n_train_count/y_train_count:.2f}\n")
    
    print(f"Rebalancing to: 1:{target_ratio:.2f}")
    print(f"Removing: {n_train_count - target_n_count} images\n")
    
    # Create rebalanced dataset
    for split_name, y_files, n_files in [
        ("train", y_train, random.sample(n_train, target_n_count)),
        ("test", y_test, n_test)  # Keep test as-is for fair evaluation
    ]:
        split_path = output_path / split_name
        (split_path / "y").mkdir(parents=True, exist_ok=True)
        (split_path / "n").mkdir(parents=True, exist_ok=True)
        
        for img in y_files:
            shutil.copy2(img, split_path / "y" / img.name)
        for img in n_files:
            shutil.copy2(img, split_path / "n" / img.name)
    
    print(f"✓ Stratified resampling complete!")
    print(f"  TRAIN - Crosswalk: {len(y_train)}, No-Crosswalk: {target_n_count}")
    print(f"  TRAIN - Ratio: 1:{target_n_count/len(y_train):.2f}")
    print(f"  TEST - Kept as original for fair evaluation\n")


def main():
    """Main rebalancing function."""
    
    # Setup project root for local paths
    import sys
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    sys.path.append(str(PROJECT_ROOT))
    
    TRAIN_DIR = str(PROJECT_ROOT / "data" / "train")
    TEST_DIR = str(PROJECT_ROOT / "data" / "test")
    
    # Analyze current imbalance
    stats = analyze_imbalance(TRAIN_DIR, TEST_DIR)
    print_analysis(stats)
    
    print(f"\n🎯 RECOMMENDED APPROACH:")
    print(f"{'='*70}")
    print(f"\nUse WEIGHTED LOSS (already in train.py) +")
    print(f"Optional STRATIFIED RESAMPLING for better training\n")
    
    print(f"Weighted loss is simple and effective:")
    print(f"  - Class weight for crosswalk = 60 (1 / 313 * 19200)")
    print(f"  - Class weight for no-crosswalk = 1")
    print(f"\nNo data modification needed! Model learns to weight")
    print(f"minority class mistakes 60x more heavily.\n")
    
    print(f"{'='*70}\n")
    
    # Ask user what to do
    print(f"\n💡 OPTIONS:")
    print(f"  1. Continue with weighted loss (RECOMMENDED) → No rebalancing needed")
    print(f"  2. Apply stratified resampling (1:3 ratio) → Faster training, less severe imbalance")
    print(f"  3. Apply oversampling → Best for very small datasets")
    print(f"  4. Apply undersampling → Faster training, loses data\n")
    
    choice = input("Choose rebalancing strategy (1-4, default=1): ").strip() or "1"
    
    if choice == "2":
        output_path = str(PROJECT_ROOT / "data_rebalanced_stratified")
        stratified_resample(TRAIN_DIR, TEST_DIR, target_ratio=3.0, output_dir=output_path)
        print(f"✅ Rebalanced data ready in {output_path}/")
    elif choice == "3":
        output_path = str(PROJECT_ROOT / "data_rebalanced_oversample")
        oversample_minority(TRAIN_DIR, target_ratio=1.0, output_dir=output_path)
        print(f"✅ Oversampled data ready in {output_path}/\n")
    elif choice == "4":
        output_path = str(PROJECT_ROOT / "data_rebalanced_undersample")
        undersample_majority(TRAIN_DIR, target_ratio=3.0, output_dir=output_path)
        print(f"✅ Undersampled data ready in {output_path}/\n")
    else:
        print(f"✓ Using weighted loss in training (no rebalancing applied)")
        print(f"  This is the recommended approach!\n")


if __name__ == "__main__":
    main()
