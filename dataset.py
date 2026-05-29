"""
03_dataset.py - Custom PyTorch Dataset with Data Augmentation

This module provides:
  - CrosswalkDataset: Custom PyTorch Dataset class for loading crosswalk images
  - Data augmentation for training set: random flips, rotations, brightness/contrast
  - Augmentation is critical for small datasets to prevent overfitting
  - For satellite imagery, rotations make sense because orientation doesn't matter

Key augmentation techniques:
  - Horizontal/Vertical flips: Important because crosswalks can be oriented any way
  - Rotations (0°, 90°, 180°, 270°): Satellite images are rotation-invariant
  - Color jitter: Compensates for different lighting conditions/seasons
  - Normalization: ImageNet statistics (for ResNet18 pre-training)
"""

import os
from pathlib import Path
from typing import Tuple, Optional, Callable
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np


class CrosswalkDataset(Dataset):
    """
    PyTorch Dataset for binary crosswalk classification.
    
    Loads images from directory structure:
        root/
        ├── y/  (class 0: crosswalk present)
        └── n/  (class 1: no crosswalk)
    
    Applies optional augmentation to improve model robustness.
    """
    
    def __init__(
        self,
        root_dir: str,
        augment: bool = False,
        target_size: Tuple[int, int] = (224, 224),
        seed: int = 42
    ):
        """
        Initialize the CrosswalkDataset.
        
        Args:
            root_dir: Path to root directory containing 'y' and 'n' subdirectories
            augment: Apply data augmentation (True for training, False for testing)
            target_size: Resize all images to this size (default: 224x224 for ResNet18)
            seed: Random seed for reproducibility
        """
        self.root_dir = Path(root_dir)
        self.augment = augment
        self.target_size = target_size
        self.seed = seed
        
        # Image file extensions to support
        self.image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
        
        # Class labels: y → 0 (crosswalk), n → 1 (no-crosswalk)
        self.class_to_idx = {"y": 0, "n": 1}
        self.idx_to_class = {0: "y", 1: "n"}
        
        # Load image paths and labels
        self.images = []
        self.labels = []
        self._load_images()
        
        # Define normalization (ImageNet statistics - ResNet18 was trained on this)
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        
        print(f"✓ Dataset loaded from {root_dir}")
        print(f"  Total images: {len(self.images)}")
        print(f"  Crosswalk (y): {sum(1 for l in self.labels if l == 0)}")
        print(f"  No-Crosswalk (n): {sum(1 for l in self.labels if l == 1)}")
        print(f"  Augmentation: {'ON' if augment else 'OFF'}")
    
    def _load_images(self) -> None:
        """Recursively load all image paths and their labels."""
        for class_name, class_idx in self.class_to_idx.items():
            class_dir = self.root_dir / class_name
            
            if not class_dir.exists():
                print(f"⚠️  Warning: {class_dir} does not exist")
                continue
            
            # Get all image files
            for img_file in class_dir.iterdir():
                if img_file.suffix.lower() in self.image_extensions:
                    self.images.append(img_file)
                    self.labels.append(class_idx)
    
    def _get_augmentation_transform(self) -> transforms.Compose:
        """
        Define data augmentation pipeline for training.
        
        Augmentations chosen for satellite imagery:
        - RandomHorizontalFlip/VerticalFlip: Crosswalks orientation-agnostic
        - RandomRotation: Satellite view is rotation-invariant
        - ColorJitter: Compensate for seasonal/lighting variations
        - RandomAffine: Additional geometric variations
        
        Returns:
            Composed torchvision transforms
        """
        return transforms.Compose([
            transforms.Resize(self.target_size),
            
            # Geometric augmentations (flips and rotations)
            transforms.RandomHorizontalFlip(p=0.5),      # Horizontal flip
            transforms.RandomVerticalFlip(p=0.5),        # Vertical flip
            transforms.RandomRotation(degrees=90),       # 0-90 degree random rotation
            
            # Color augmentations (handle lighting/seasonal variations)
            transforms.ColorJitter(
                brightness=0.2,  # ±20% brightness
                contrast=0.2,    # ±20% contrast
                saturation=0.2,  # ±20% saturation
                hue=0.1          # ±10% hue
            ),
            
            # Convert to tensor and normalize
            transforms.ToTensor(),
            self.normalize
        ])
    
    def _get_test_transform(self) -> transforms.Compose:
        """
        Define test/validation transform (no augmentation).
        
        Only resizing and normalization for consistent evaluation.
        
        Returns:
            Composed torchvision transforms
        """
        return transforms.Compose([
            transforms.Resize(self.target_size),
            transforms.ToTensor(),
            self.normalize
        ])
    
    def __len__(self) -> int:
        """Return total number of images in dataset."""
        return len(self.images)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get image and label by index.
        
        Args:
            idx: Image index
            
        Returns:
            Tuple of (image_tensor, label)
        """
        # Load image
        img_path = self.images[idx]
        try:
            image = Image.open(img_path).convert("RGB")  # Ensure RGB (no alpha channel)
        except Exception as e:
            print(f"❌ Error loading {img_path}: {e}")
            # Return a black image as fallback
            image = Image.new("RGB", self.target_size, color=(0, 0, 0))
        
        # Get label
        label = self.labels[idx]
        
        # Apply transforms
        if self.augment:
            transform = self._get_augmentation_transform()
        else:
            transform = self._get_test_transform()
        
        image = transform(image)
        
        return image, label


def create_dataloaders(
    train_dir: str,
    test_dir: str,
    batch_size: int = 128,
    num_workers: int = 8,
    seed: int = 42
) -> Tuple[DataLoader, DataLoader]:
    """
    Create training and test DataLoaders.
    
    Args:
        train_dir: Path to training data (contains y/ and n/ subdirectories)
        test_dir: Path to test data
        batch_size: Batch size for both loaders
        num_workers: Number of worker processes (0 = main process only)
        seed: Random seed
        
    Returns:
        Tuple of (train_loader, test_loader)
    """
    # Set seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Create datasets
    train_dataset = CrosswalkDataset(
        root_dir=train_dir,
        augment=True,  # Augmentation ON for training
        seed=seed
    )
    
    test_dataset = CrosswalkDataset(
        root_dir=test_dir,
        augment=False,  # NO augmentation for testing
        seed=seed
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True  # Speed up GPU transfer
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"\n✓ DataLoaders created:")
    print(f"  Train batches: {len(train_loader)} @ batch_size={batch_size}")
    print(f"  Test batches: {len(test_loader)} @ batch_size={batch_size}")
    
    return train_loader, test_loader


if __name__ == "__main__":
    # Test the dataset
    import sys
    
    # Use dummy data directory for testing
    test_root = Path("./data/asconalocarno")
    
    if test_root.exists():
        print(f"\n📁 Loading dataset from: {test_root}\n")
        
        # Create dataset without augmentation
        dataset = CrosswalkDataset(
            root_dir=str(test_root),
            augment=False,
            target_size=(224, 224)
        )
        
        if len(dataset) > 0:
            # Try to load a sample
            img, label = dataset[0]
            print(f"\n✓ Sample loaded successfully!")
            print(f"  Image shape: {img.shape}")
            print(f"  Label: {label} ({'Crosswalk' if label == 0 else 'No-Crosswalk'})")
        else:
            print("⚠️  Dataset is empty. Make sure images are in y/ and n/ subdirectories.")
    else:
        print(f"⚠️  Test directory not found: {test_root}")
        print("   Please ensure data/asconalocarno/ exists with y/ and n/ subdirectories.")
