"""
CNN architectures for lightning classification.
- ResNet-50: Primary (transfer learning)
- UNet: Optional (segmentation)
"""

import torch
import torch.nn as nn
import torchvision.models as models


class LightningResNet50(nn.Module):
    """
    ResNet-50 patch classifier for lightning detection.
    
    Transfer learning approach:
    - Load ImageNet pretrained ResNet-50
    - Adapt first conv layer for multi-channel input
    - Replace final FC layer for binary classification
    - Dropout for regularization
    """
    
    def __init__(self, num_input_channels=3, num_classes=1, dropout_rate=0.5, pretrained=True):
        """
        Args:
            num_input_channels (int): 3 (IR, WV, VIS) or other combinations
            num_classes (int): 1 (binary sigmoid output)
            dropout_rate (float): Dropout probability
        """
        super(LightningResNet50, self).__init__()
        
        # Load pretrained ResNet-50 when available; otherwise fall back to a random init.
        if pretrained:
            try:
                self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
            except Exception:
                self.backbone = models.resnet50(weights=None)
        else:
            self.backbone = models.resnet50(weights=None)
        
        # Adapt first conv layer for multi-channel input
        if num_input_channels != 3:
            original_conv1 = self.backbone.conv1
            self.backbone.conv1 = nn.Conv2d(
                num_input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
            )
            # Initialize new weights: average of original or repeat
            with torch.no_grad():
                if num_input_channels < 3:
                    # Take first N channels from pretrained
                    self.backbone.conv1.weight[:, :num_input_channels, :, :] = \
                        original_conv1.weight[:, :num_input_channels, :, :]
                elif num_input_channels > 3:
                    # Replicate channels to fill extra ones
                    self.backbone.conv1.weight[:, :3, :, :] = original_conv1.weight
                    for i in range(3, num_input_channels):
                        self.backbone.conv1.weight[:, i:i+1, :, :] = \
                            original_conv1.weight[:, 0:1, :, :]
        
        # Replace final FC layer
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.6),
            nn.Linear(128, num_classes),
            nn.Sigmoid()  # Binary output [0, 1]
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): shape (batch_size, C, H, W)
        
        Returns:
            torch.Tensor: shape (batch_size, 1), values in [0, 1]
        """
        return self.backbone(x)


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.
    
    Paper: "Focal Loss for Dense Object Detection"
    https://arxiv.org/abs/1708.02002
    
    Focus on hard examples; down-weight easy ones.
    """
    
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        """
        Args:
            alpha (float): Weighting factor in range [0, 1]
            gamma (float): Exponent for the modulating factor (1 - p_t)^gamma
            reduction (str): 'mean', 'sum', or 'none'
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs (torch.Tensor): Predicted probabilities, shape (N, 1)
            targets (torch.Tensor): Ground truth labels, shape (N, 1)
        
        Returns:
            torch.Tensor: Scalar loss value
        """
        # Clip predictions to avoid log(0)
        inputs = torch.clamp(inputs, min=1e-7, max=1 - 1e-7)
        
        # Compute binary cross entropy
        bce = -targets * torch.log(inputs) - (1 - targets) * torch.log(1 - inputs)
        
        # Compute modulating factor (1 - p_t)^gamma
        p_t = torch.where(targets == 1, inputs, 1 - inputs)
        modulating_factor = (1 - p_t).pow(self.gamma)
        
        # Focal loss
        focal_loss = self.alpha * modulating_factor * bce
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


if __name__ == '__main__':
    # Test model initialization and forward pass
    model = LightningResNet50(num_input_channels=3)
    x = torch.randn(16, 3, 64, 64)  # Batch of 16 samples
    y = model(x)
    print(f"Output shape: {y.shape}")
    print(f"Output range: [{y.min():.4f}, {y.max():.4f}]")
    
    # Test Focal Loss
    criterion = FocalLoss(alpha=0.25, gamma=2.0)
    targets = torch.randint(0, 2, (16, 1)).float()
    loss = criterion(y, targets)
    print(f"Focal Loss: {loss.item():.4f}")
