"""
03_dataset.py - Custom PyTorch Dataset with Data Augmentation and Visualization

This module provides:
  - CrosswalkDataset: Custom PyTorch Dataset class for loading crosswalk images
  - Data augmentation for training set: random flips, rotations, brightness/contrast
  - Saves augmented images to an inspection folder ("img_2") for visual validation
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
    """
    
    def __init__(
        self,
        root_dir: str,
        augment: bool = False,
        target_size: Tuple[int, int] = (224, 224),
        seed: int = 42,
        save_aug_dir: str = "img_2"  # <-- Added parameter for saving augmented images
    ):
        self.root_dir = Path(root_dir)
        self.augment = augment
        self.target_size = target_size
        self.seed = seed
        self.save_aug_dir = Path(save_aug_dir)
        
        # Create the inspection directory if augmentations are ON
        if self.augment:
            self.save_aug_dir.mkdir(parents=True, exist_ok=True)
            print(f"👁️ Saving visual augmentations to: {self.save_aug_dir.absolute()}")

        # Image file extensions to support
        self.image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
        
        # Class labels: y → 0 (crosswalk), n → 1 (no-crosswalk)
        self.class_to_idx = {"y": 0, "n": 1}
        
        # Load image paths and labels
        self.images = []
        self.labels = []
        self._load_images()
        
        # Define normalization (ImageNet statistics)
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        
        print(f"✓ Dataset loaded from {root_dir}")
        print(f"  Total images: {len(self.images)}")
        print(f"  Augmentation: {'ON' if augment else 'OFF'}")
    
    def _load_images(self) -> None:
        """Recursively load all image paths and their labels."""
        for class_name, class_idx in self.class_to_idx.items():
            class_dir = self.root_dir / class_name
            if not class_dir.exists():
                continue
            for img_file in class_dir.iterdir():
                if img_file.suffix.lower() in self.image_extensions:
                    self.images.append(img_file)
                    self.labels.append(class_idx)

    def _get_visual_transforms(self) -> transforms.Compose:
        """
        Step 1: Pipeline for visual augmentations (stays as PIL Image).
        These are the changes we actually want to look at.
        """
        return transforms.Compose([
            transforms.Resize(self.target_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=90),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1
            )
        ])

    def _get_tensor_transforms(self) -> transforms.Compose:
        """
        Step 2: Convert to tensor and normalize for the model.
        """
        return transforms.Compose([
            transforms.ToTensor(),
            self.normalize
        ])
    
    def _get_test_transform(self) -> transforms.Compose:
        """Test/validation transform (no augmentation)."""
        return transforms.Compose([
            transforms.Resize(self.target_size),
            transforms.ToTensor(),
            self.normalize
        ])
    
    def __len__(self) -> int:
        return len(self.images)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path = self.images[idx]
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"❌ Error loading {img_path}: {e}")
            image = Image.new("RGB", self.target_size, color=(0, 0, 0))
        
        label = self.labels[idx]
        
        if self.augment:
            # 1. Apply visual augmentations
            vis_transform = self._get_visual_transforms()
            augmented_pil_image = vis_transform(image)
            
            # 2. Save the image to img_2 folder for inspection
            original_filename = img_path.name
            save_path = self.save_aug_dir / f"aug_{idx}_{original_filename}"
            augmented_pil_image.save(save_path)
            
            # 3. Convert to normalized tensor for PyTorch
            tensor_transform = self._get_tensor_transforms()
            final_image = tensor_transform(augmented_pil_image)
        else:
            # Test set gets standard pipeline
            transform = self._get_test_transform()
            final_image = transform(image)
            
        return final_image, label


def create_dataloaders(
    train_dir: str,
    test_dir: str,
    batch_size: int = 32,
    num_workers: int = 0,
    seed: int = 42,
    save_aug_dir: str = "img_2"
) -> Tuple[DataLoader, DataLoader]:
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    train_dataset = CrosswalkDataset(
        root_dir=train_dir,
        augment=True,
        seed=seed,
        save_aug_dir=save_aug_dir
    )
    
    test_dataset = CrosswalkDataset(
        root_dir=test_dir,
        augment=False,
        seed=seed
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    return train_loader, test_loader


if __name__ == "__main__":
    import sys
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    sys.path.append(str(PROJECT_ROOT))
    
    # Change this to your real dataset path!
    test_root = PROJECT_ROOT / "data" / "asconalocarno"
    save_aug_dir = PROJECT_ROOT / "img_2"
    
    if test_root.exists():
        print(f"\n📁 Loading dataset from: {test_root}\n")
        
        # Turn augment to True so the saving feature is triggered
        dataset = CrosswalkDataset(
            root_dir=str(test_root),
            augment=True, 
            target_size=(224, 224),
            save_aug_dir=str(save_aug_dir)
        )
        
        target_count = 300
        available_images = len(dataset)
        
        if available_images > 0:
            # Prevent crashing if your test folder has fewer than 300 images
            limit = min(target_count, available_images)
            print(f"⚙️ Generating {limit} augmented images...")
            
            # Loop through the dataset
            for i in range(limit):
                # Fetching the item triggers the augmentation and saves the image
                img, label = dataset[i]
                
                # Print a progress update every 50 images so you know it's working
                if (i + 1) % 50 == 0:
                    print(f"  Processed {i + 1}/{limit} images...")
                    
            print(f"\n✓ Done! Check the '{save_aug_dir.name}' folder to see all {limit} images.")
        else:
            print("⚠️  Dataset is empty. Make sure images are in y/ and n/ subdirectories.")
    else:
        print(f"⚠️  Test directory not found: {test_root}")