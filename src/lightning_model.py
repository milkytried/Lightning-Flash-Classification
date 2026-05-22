"""
Lightning detection model using metadata features.
Uses simple MLP for classification of strike characteristics.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LightningMetadataClassifier(nn.Module):
    """
    Simple MLP for lightning detection from metadata.
    
    Input: [latitude, longitude, amplitude, strike_type] (4 features)
    Output: Binary classification (lightning present = 1, absent = 0)
    """
    
    def __init__(self, input_size: int = 4, hidden_size: int = 256, dropout: float = 0.5):
        super().__init__()
        
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.dropout1 = nn.Dropout(dropout)
        
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.bn2 = nn.BatchNorm1d(hidden_size // 2)
        self.dropout2 = nn.Dropout(dropout)
        
        self.fc3 = nn.Linear(hidden_size // 2, hidden_size // 4)
        self.bn3 = nn.BatchNorm1d(hidden_size // 4)
        self.dropout3 = nn.Dropout(dropout)
        
        self.fc4 = nn.Linear(hidden_size // 4, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, 4) metadata features
        
        Returns:
            (batch_size, 1) sigmoid probabilities
        """
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout2(x)
        
        x = self.fc3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.dropout3(x)
        
        x = self.fc4(x)
        x = torch.sigmoid(x)  # Binary classification
        
        return x


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance (5M+ positive vs 8K negative)."""
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            predictions: (batch_size, 1) sigmoid probabilities [0, 1]
            targets: (batch_size, 1) binary labels {0, 1}
        """
        # BCE loss
        bce_loss = F.binary_cross_entropy(predictions, targets, reduction='none')
        
        # Focal loss weighting
        p_t = torch.where(targets == 1, predictions, 1 - predictions)
        focal_weight = (1 - p_t) ** self.gamma
        
        # Apply alpha weighting for class imbalance
        alpha_weight = torch.where(targets == 1, self.alpha, 1 - self.alpha)
        
        focal_loss = alpha_weight * focal_weight * bce_loss
        
        return focal_loss.mean()
