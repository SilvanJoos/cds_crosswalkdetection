import torch
import torch.nn as nn
import torchvision.models as models

def set_seed(seed: int = 42) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"✓ PyTorch random seed set to {seed}")

class CrosswalkClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int = 2,
        dropout_rate: float = 0.4,  # REDUCED from 0.8
        pretrained: bool = True,
        freeze_backbone: bool = False  # SET TO FALSE to enable fine-tuning
    ):
        super(CrosswalkClassifier, self).__init__()
        
        self.backbone = models.resnet18(pretrained=pretrained)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # SIMPLIFIED HEAD: 2 Layers instead of 3 to prevent overfitting
        self.fc1 = nn.Linear(in_features, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_rate)
        
        self.fc2 = nn.Linear(256, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        x = self.dropout1(self.relu1(self.bn1(self.fc1(features))))
        x = self.fc2(x)
        return x
    
    def get_parameter_groups(self):
        backbone_params = list(self.backbone.parameters())
        head_params = list(self.fc1.parameters()) + \
                     list(self.bn1.parameters()) + \
                     list(self.fc2.parameters())
        
        return [
            {'params': backbone_params, 'lr': 1e-5},  # Very low LR for backbone
            {'params': head_params, 'lr': 1e-3}       # Standard LR for custom head
        ]

def create_model(device, dropout_rate=0.4, pretrained=True, freeze_backbone=False, seed=42):
    set_seed(seed)
    model = CrosswalkClassifier(
        num_classes=2, dropout_rate=dropout_rate, 
        pretrained=pretrained, freeze_backbone=freeze_backbone
    )
    return model.to(device)


if __name__ == "__main__":
    # Test the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🖥️  Device: {device}\n")
    
    model = create_model(device, dropout_rate=0.4)
    
    # Test forward pass
    dummy_input = torch.randn(4, 3, 224, 224).to(device)  # Batch of 4 images
    output = model(dummy_input)
    
    print(f"\n✓ Model test successful!")
    print(f"  Input shape: {dummy_input.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Output (logits): {output.detach().cpu()}")
