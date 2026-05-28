"""
02_model.py - Model Architecture: Transfer Learning with ResNet18

This module defines a binary classification model for crosswalk detection using:
  - ResNet18 as the backbone (pre-trained on ImageNet)
  - Frozen base layers to leverage learned features
  - Custom classification head with maximum 3 linear layers
  - Dropout for regularization to prevent overfitting

The architecture:
  ResNet18 backbone → Global Average Pooling → FC1 (512→256) → Dropout
                   → FC2 (256→128) → Dropout → FC3 (128→2) [binary classification]
"""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Optional


def set_seed(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility across PyTorch, CUDA, and NumPy.
    
    This is CRITICAL for hackathon evaluation - ensures your model is deterministic.
    
    Args:
        seed: Random seed value (default: 42)
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"✓ PyTorch random seed set to {seed} for reproducibility")


class CrosswalkClassifier(nn.Module):
    """
    Binary classifier for detecting crosswalks in satellite imagery.
    
    Architecture:
    - Backbone: ResNet18 (pre-trained on ImageNet)
    - Base layers: FROZEN to use learned features
    - Classification head: 3 linear layers with dropout
    
    This transfer learning approach works well because:
    1. ResNet18 learns basic shape/color patterns from ImageNet
    2. Frozen layers preserve these general features
    3. Custom head learns specific crosswalk patterns
    4. Small dataset + frozen backbone prevents overfitting
    """
    
    def __init__(
        self,
        num_classes: int = 2,
        dropout_rate: float = 0.5,
        pretrained: bool = True,
        freeze_backbone: bool = True
    ):
        """
        Initialize the crosswalk classifier.
        
        Args:
            num_classes: Number of output classes (2 for binary: crosswalk/no-crosswalk)
            dropout_rate: Dropout probability for regularization (default: 0.5)
            pretrained: Load pre-trained ImageNet weights (default: True)
            freeze_backbone: Freeze ResNet18 base layers (default: True)
        """
        super(CrosswalkClassifier, self).__init__()
        
        # ========== BACKBONE: Pre-trained ResNet18 ==========
        self.backbone = models.resnet18(pretrained=pretrained)
        
        # Get the number of features from ResNet18's final layer
        in_features = self.backbone.fc.in_features  # 512 for ResNet18
        
        # Replace the final classification layer (we'll add our own head)
        self.backbone.fc = nn.Identity()  # Remove original FC layer
        
        # Freeze backbone parameters if requested
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            print(f"✓ ResNet18 backbone frozen ({self._count_parameters(self.backbone):,} frozen params)")
        else:
            print(f"✓ ResNet18 backbone unfrozen (fine-tuning enabled)")
        
        # ========== CUSTOM CLASSIFICATION HEAD (3 layers max) ==========
        # Layer 1: 512 → 256
        self.fc1 = nn.Linear(in_features, 256)
        self.bn1 = nn.BatchNorm1d(256)  # Batch norm for stability
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_rate)
        
        # Layer 2: 256 → 128
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout_rate)
        
        # Layer 3: 128 → num_classes (binary output)
        self.fc3 = nn.Linear(128, num_classes)
        
        print(f"✓ Custom head created: {in_features} → 256 → 128 → {num_classes}")
        print(f"✓ Dropout rate: {dropout_rate}")
        print(f"✓ Total trainable params: {self._count_parameters(self, trainable=True):,}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor of shape (batch_size, 3, height, width)
            
        Returns:
            Logits of shape (batch_size, num_classes)
        """
        # Backbone feature extraction
        features = self.backbone(x)  # (batch, 512)
        
        # Custom head with dropout
        x = self.fc1(features)       # (batch, 256)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        
        x = self.fc2(x)              # (batch, 128)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        
        x = self.fc3(x)              # (batch, 2) - binary classification
        
        return x
    
    @staticmethod
    def _count_parameters(model: nn.Module, trainable: bool = False) -> int:
        """Count total or trainable parameters in a model."""
        if trainable:
            return sum(p.numel() for p in model.parameters() if p.requires_grad)
        return sum(p.numel() for p in model.parameters())
    
    def get_parameter_groups(self):
        """
        Get parameter groups for differential learning rates.
        Useful for fine-tuning: lower LR for backbone, higher for head.
        
        Returns:
            List of parameter groups for optimizer
        """
        backbone_params = list(self.backbone.parameters())
        head_params = list(self.fc1.parameters()) + \
                     list(self.bn1.parameters()) + \
                     list(self.fc2.parameters()) + \
                     list(self.bn2.parameters()) + \
                     list(self.fc3.parameters())
        
        return [
            {'params': backbone_params, 'lr': 1e-4},  # Low LR for backbone
            {'params': head_params, 'lr': 1e-3}       # Higher LR for custom head
        ]


def create_model(
    device: torch.device,
    dropout_rate: float = 0.5,
    pretrained: bool = True,
    freeze_backbone: bool = True,
    seed: int = 42
) -> CrosswalkClassifier:
    """
    Factory function to create and initialize a CrosswalkClassifier.
    
    Args:
        device: Torch device (cpu or cuda)
        dropout_rate: Dropout probability
        pretrained: Load ImageNet weights
        freeze_backbone: Freeze base layers
        seed: Random seed
        
    Returns:
        Initialized CrosswalkClassifier model on the specified device
    """
    set_seed(seed)
    model = CrosswalkClassifier(
        num_classes=2,
        dropout_rate=dropout_rate,
        pretrained=pretrained,
        freeze_backbone=freeze_backbone
    )
    model.to(device)
    return model


if __name__ == "__main__":
    # Test the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🖥️  Device: {device}\n")
    
    model = create_model(device, dropout_rate=0.5)
    
    # Test forward pass
    dummy_input = torch.randn(4, 3, 224, 224).to(device)  # Batch of 4 images
    output = model(dummy_input)
    
    print(f"\n✓ Model test successful!")
    print(f"  Input shape: {dummy_input.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Output (logits): {output.detach().cpu()}")
