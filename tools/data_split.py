"""
01_data_split.py - Data Preparation: Train/Test Split

This script splits the crosswalk dataset into train and test directories.
It assumes your original data is organized in two folders:
  - y/ : images containing crosswalks (positive class)
  - n/ : images without crosswalks (negative class)

Output structure:
  - data/train/y/ : training crosswalk images
  - data/train/n/ : training non-crosswalk images
  - data/test/y/  : testing crosswalk images
  - data/test/n/  : testing non-crosswalk images

Note: This script ensures reproducibility by using a fixed random seed.
"""

import os
import shutil
import random
from pathlib import Path
from typing import Tuple


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
    train_ratio: float = 0.8,
    seed: int = 42
) -> Tuple[int, int]:
    """
    Split images from source directories into train/test folders.
    
    Args:
        data_dir: Path to folder containing 'y' and 'n' subdirectories
        output_base_dir: Path where 'train' and 'test' directories will be created
        train_ratio: Proportion of data for training (default: 0.8 for 80/20 split)
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (total_images_split, images_per_class_dict)
    """
    set_seed(seed)
    
    # Create output directory structure
    base_path = Path(output_base_dir)
    train_path = base_path / "train"
    test_path = base_path / "test"
    
    # Create all required directories
    for class_label in ["y", "n"]:
        (train_path / class_label).mkdir(parents=True, exist_ok=True)
        (test_path / class_label).mkdir(parents=True, exist_ok=True)
    
    print(f"\n📁 Created output directory structure at: {output_base_dir}")
    print(f"   Train/test split ratio: {train_ratio:.0%} / {1-train_ratio:.0%}\n")
    
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
        split_idx = int(len(image_files) * train_ratio)
        train_files = image_files[:split_idx]
        test_files = image_files[split_idx:]
        
        # Copy files to respective directories
        for img_file in train_files:
            shutil.copy2(img_file, train_path / class_label / img_file.name)
        
        for img_file in test_files:
            shutil.copy2(img_file, test_path / class_label / img_file.name)
        
        # Log statistics
        class_name = "Crosswalk (y)" if class_label == "y" else "No Crosswalk (n)"
        print(f"📊 {class_name}:")
        print(f"   Total: {len(image_files)} | Train: {len(train_files)} | Test: {len(test_files)}")
        
        total_images += len(image_files)
        class_stats[class_label] = {
            "total": len(image_files),
            "train": len(train_files),
            "test": len(test_files)
        }
    
    return total_images, class_stats


def print_summary(total_images: int, class_stats: dict) -> None:
    """Print a summary of the data split."""
    print(f"\n{'='*60}")
    print(f"✅ Dataset Split Complete!")
    print(f"{'='*60}")
    print(f"Total images processed: {total_images}")
    print(f"Train/Test split: {sum(s['train'] for s in class_stats.values())} / {sum(s['test'] for s in class_stats.values())}")
    print(f"\n📁 Directory structure ready for training:")
    print(f"   data/")
    print(f"   ├── train/")
    print(f"   │   ├── y/  ({class_stats.get('y', {}).get('train', 0)} images)")
    print(f"   │   └── n/  ({class_stats.get('n', {}).get('train', 0)} images)")
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
    
    # Output directory where train/test split will be created
    OUTPUT_DIR = str(PROJECT_ROOT / "data")
    
    # Train/test split ratio (80% train, 20% test)
    TRAIN_TEST_RATIO = 0.8
    
    # Random seed for reproducibility
    RANDOM_SEED = 42
    # ===================================
    
    try:
        print("\n🚀 Starting data split process...\n")
        total, stats = split_dataset(
            data_dir=SOURCE_DATA_DIR,
            output_base_dir=OUTPUT_DIR,
            train_ratio=TRAIN_TEST_RATIO,
            seed=RANDOM_SEED
        )
        print_summary(total, stats)
        
    except Exception as e:
        print(f"❌ Error during data split: {e}")
        raise
