"""
Unit tests for model_arch module.
"""

import pytest
import torch

from src.model_arch import LightningResNet50, FocalLoss


def test_resnet50_initialization():
    """Test model initializes without errors."""
    model = LightningResNet50(num_input_channels=3)
    assert model is not None
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")


def test_forward_pass_shape():
    """Test forward pass returns correct shape."""
    model = LightningResNet50(num_input_channels=3)
    x = torch.randn(16, 3, 64, 64)
    y = model(x)
    
    assert y.shape == torch.Size([16, 1])
    assert (y >= 0).all() and (y <= 1).all()


def test_forward_pass_multi_channel():
    """Test forward pass with non-standard channel count."""
    model = LightningResNet50(num_input_channels=5)  # IR, WV, VIS, custom1, custom2
    x = torch.randn(8, 5, 64, 64)
    y = model(x)
    
    assert y.shape == torch.Size([8, 1])
    assert (y >= 0).all() and (y <= 1).all()


def test_forward_pass_single_channel():
    """Test forward pass with single channel (IR only)."""
    model = LightningResNet50(num_input_channels=1)
    x = torch.randn(4, 1, 64, 64)
    y = model(x)
    
    assert y.shape == torch.Size([4, 1])


def test_model_gradients():
    """Test that gradients flow through model."""
    model = LightningResNet50(num_input_channels=3)
    x = torch.randn(8, 3, 64, 64, requires_grad=True)
    y = model(x)
    loss = y.mean()
    loss.backward()
    
    # Check gradients exist
    for name, param in model.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"


def test_focal_loss_initialization():
    """Test Focal Loss initializes without errors."""
    criterion = FocalLoss(alpha=0.25, gamma=2.0)
    assert criterion is not None


def test_focal_loss_computation():
    """Test Focal Loss computes without errors."""
    criterion = FocalLoss(alpha=0.25, gamma=2.0)
    predictions = torch.sigmoid(torch.randn(32, 1))
    targets = torch.randint(0, 2, (32, 1)).float()
    
    loss = criterion(predictions, targets)
    assert not torch.isnan(loss)
    assert loss > 0
    assert loss.item() > 0


def test_focal_loss_value_range():
    """Test Focal Loss returns positive value."""
    criterion = FocalLoss(alpha=0.25, gamma=2.0)
    predictions = torch.rand(64, 1)
    targets = torch.randint(0, 2, (64, 1)).float()
    
    loss = criterion(predictions, targets)
    assert loss.item() >= 0


def test_focal_loss_perfect_prediction():
    """Test Focal Loss on perfect predictions."""
    criterion = FocalLoss(alpha=0.25, gamma=2.0)
    
    # Perfect predictions
    predictions = torch.ones(16, 1) * 0.99  # Almost all 1
    targets = torch.ones(16, 1)  # All 1
    
    loss = criterion(predictions, targets)
    # Loss should be low (but not exactly 0 due to clipping)
    assert loss.item() < 0.1


def test_model_eval_mode():
    """Test model can be switched to eval mode."""
    model = LightningResNet50(num_input_channels=3)
    model.eval()
    
    x = torch.randn(8, 3, 64, 64)
    
    # Forward pass in eval mode
    with torch.no_grad():
        y = model(x)
    
    assert y.shape == torch.Size([8, 1])


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
