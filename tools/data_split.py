"""
01_data_split.py - Data Preparation: Train/Val/Test Split

This script splits the crosswalk dataset into train, val, and test directories.
It assumes your original data is organized in two folders:
  - y/ : images containing crosswalks (positive class)
  - n/ : images without crosswalks (negative class)

Output structure:
  - data/train/y/ : training crosswalk images
  - data/train/n/ : training non-crosswalk images
  - data/val/y/   : validation crosswalk images
  - data/val/n/   : validation non-crosswalk images
  - data/test/y/  : testing crosswalk images
  - data/test/n/  : testing non-crosswalk images

Note: This script ensures reproducibility by using a fixed random seed.
"""

import os
import shutil
import random
from pathlib import Path
from typing import Tuple, Dict


def set_seed(seed: int = 42) -> None:
    """
    Set random seed for reproducibility across Python and libraries.
    
    Args:
        seed: Random seed value (default: 42)
    """
    random.seed(seed)
    print(f"✓ Random seed set to {seed} for reproducibility")


def split_dataset(
    data_dir: str,
    output_base_dir: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42
) -> Tuple[int, Dict]:
    """
    Split images from source directories into train/val/test folders.
    
    Args:
        data_dir: Path to folder containing 'y' and 'n' subdirectories
        output_base_dir: Path where 'train', 'val', and 'test' directories will be created
        train_ratio: Proportion of data for training (default: 0.7)
        val_ratio: Proportion of data for validation (default: 0.15)
        test_ratio: Proportion of data for testing (default: 0.15)
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (total_images_split, images_per_class_dict)
    """
    set_seed(seed)
    
    # Create output directory structure
    base_path = Path(output_base_dir)
    train_path = base_path / "train"
    val_path = base_path / "val"
    test_path = base_path / "test"
    
    # Create all required directories
    for class_label in ["y", "n"]:
        (train_path / class_label).mkdir(parents=True, exist_ok=True)
        (val_path / class_label).mkdir(parents=True, exist_ok=True)
        (test_path / class_label).mkdir(parents=True, exist_ok=True)
    
    print(f"\n📁 Created output directory structure at: {output_base_dir}")
    print(f"   Train/Val/Test split ratio: {train_ratio:.0%} / {val_ratio:.0%} / {test_ratio:.0%}\n")
    
    # Process each class folder
    total_images = 0
    class_stats = {}
    
    for class_label in ["y", "n"]:
        source_class_dir = Path(data_dir) / class_label
        
        if not source_class_dir.exists():
            print(f"⚠️  Warning: {source_class_dir} does not exist. Skipping.")
            continue
        
        # Get all image files (common image formats)
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
        image_files = [
            f for f in source_class_dir.iterdir()
            if f.suffix.lower() in image_extensions
        ]
        
        if not image_files:
            print(f"⚠️  No images found in {source_class_dir}")
            continue
        
        # Shuffle and split
        random.shuffle(image_files)
        total = len(image_files)
        train_idx = int(total * train_ratio)
        val_idx = train_idx + int(total * val_ratio)
        
        train_files = image_files[:train_idx]
        val_files = image_files[train_idx:val_idx]
        test_files = image_files[val_idx:]
        
        # Copy files to respective directories
        for img_file in train_files:
            shutil.copy2(img_file, train_path / class_label / img_file.name)
            
        for img_file in val_files:
            shutil.copy2(img_file, val_path / class_label / img_file.name)
        
        for img_file in test_files:
            shutil.copy2(img_file, test_path / class_label / img_file.name)
        
        # Log statistics
        class_name = "Crosswalk (y)" if class_label == "y" else "No Crosswalk (n)"
        print(f"📊 {class_name}:")
        print(f"   Total: {total} | Train: {len(train_files)} | Val: {len(val_files)} | Test: {len(test_files)}")
        
        total_images += total
        class_stats[class_label] = {
            "total": total,
            "train": len(train_files),
            "val": len(val_files),
            "test": len(test_files)
        }
    
    return total_images, class_stats


def print_summary(total_images: int, class_stats: dict) -> None:
    """Print a summary of the data split."""
    print(f"\n{'='*60}")
    print(f"✅ Dataset Split Complete!")
    print(f"{'='*60}")
    print(f"Total images processed: {total_images}")
    print(f"Train/Val/Test split: {sum(s['train'] for s in class_stats.values())} / {sum(s['val'] for s in class_stats.values())} / {sum(s['test'] for s in class_stats.values())}")
    print(f"\n📁 Directory structure ready for training:")
    print(f"   data/")
    print(f"   ├── train/")
    print(f"   │   ├── y/  ({class_stats.get('y', {}).get('train', 0)} images)")
    print(f"   │   └── n/  ({class_stats.get('n', {}).get('train', 0)} images)")
    print(f"   ├── val/")
    print(f"   │   ├── y/  ({class_stats.get('y', {}).get('val', 0)} images)")
    print(f"   │   └── n/  ({class_stats.get('n', {}).get('val', 0)} images)")
    print(f"   └── test/")
    print(f"       ├── y/  ({class_stats.get('y', {}).get('test', 0)} images)")
    print(f"       └── n/  ({class_stats.get('n', {}).get('test', 0)} images)")


if __name__ == "__main__":
    # Setup project root for local paths
    import sys
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    sys.path.append(str(PROJECT_ROOT))
    
    # ========== CONFIGURATION ==========
    # Path to your source data with 'y' and 'n' subdirectories
    SOURCE_DATA_DIR = str(PROJECT_ROOT / "data" / "working")  # Update this to your data location
    
    # Output directory where train/val/test split will be created
    OUTPUT_DIR = str(PROJECT_ROOT / "data")
    
    # Train/Val/Test split ratio
    TRAIN_RATIO = 0.7
    VAL_RATIO = 0.15
    TEST_RATIO = 0.15
    
    # Random seed for reproducibility
    RANDOM_SEED = 42
    # ===================================
    
    try:
        print("\n🚀 Starting data split process...\n")
        total, stats = split_dataset(
            data_dir=SOURCE_DATA_DIR,
            output_base_dir=OUTPUT_DIR,
            train_ratio=TRAIN_RATIO,
            val_ratio=VAL_RATIO,
            test_ratio=TEST_RATIO,
            seed=RANDOM_SEED
        )
        print_summary(total, stats)
        
    except Exception as e:
        print(f"❌ Error during data split: {e}")
        raise
