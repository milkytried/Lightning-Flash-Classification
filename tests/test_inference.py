"""
Unit tests for inference module.
"""

import pytest
import torch
import numpy as np
import tempfile
from pathlib import Path

from src.model_arch import LightningResNet50
from src.inference import LightningPredictor


@pytest.fixture
def dummy_model_and_config():
    """Create a dummy model and config for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Create and save a dummy model
        model = LightningResNet50(num_input_channels=3, pretrained=False)
        model_path = tmpdir / "dummy_model.pth"
        torch.save(model.state_dict(), model_path)
        
        # Create a dummy config
        config_path = tmpdir / "config.yaml"
        config_content = """
model:
  num_input_channels: 3
  
train:
  device: "cpu"
"""
        with open(config_path, 'w') as f:
            f.write(config_content)
        
        yield str(model_path), str(config_path)


def test_predictor_initialization(dummy_model_and_config):
    """Test predictor initializes without errors."""
    model_path, config_path = dummy_model_and_config
    predictor = LightningPredictor(model_path, config_path=config_path, device='cpu')
    assert predictor is not None
    assert predictor.device.type == 'cpu'


def test_predictor_missing_config():
    """Test predictor raises FileNotFoundError for missing config."""
    with pytest.raises(FileNotFoundError, match="Config not found"):
        LightningPredictor('dummy_model.pth', config_path='nonexistent.yaml', device='cpu')


def test_predictor_missing_model(dummy_model_and_config):
    """Test predictor raises FileNotFoundError for missing model."""
    _, config_path = dummy_model_and_config
    with pytest.raises(FileNotFoundError, match="Model weights not found"):
        LightningPredictor('nonexistent_model.pth', config_path=config_path, device='cpu')


def test_predict_single_image(dummy_model_and_config):
    """Test single image prediction."""
    model_path, config_path = dummy_model_and_config
    predictor = LightningPredictor(model_path, config_path=config_path, device='cpu')
    
    # Create dummy image
    image = np.random.rand(3, 64, 64).astype(np.float32)
    result = predictor.predict(image)
    
    assert isinstance(result, dict)
    assert 'probability' in result
    assert 'prediction' in result
    assert 'confidence' in result
    assert 'threshold' in result
    
    # Check value ranges
    assert 0 <= result['probability'] <= 1
    assert result['prediction'] in [0, 1]
    assert 0 <= result['confidence'] <= 1


def test_predict_single_image_tensor(dummy_model_and_config):
    """Test prediction with torch.Tensor input."""
    model_path, config_path = dummy_model_and_config
    predictor = LightningPredictor(model_path, config_path=config_path, device='cpu')
    
    # Create dummy tensor
    image = torch.rand(3, 64, 64)
    result = predictor.predict(image)
    
    assert isinstance(result, dict)
    assert 0 <= result['probability'] <= 1


def test_predict_hwc_format(dummy_model_and_config):
    """Test prediction with (H, W, C) format."""
    model_path, config_path = dummy_model_and_config
    predictor = LightningPredictor(model_path, config_path=config_path, device='cpu')
    
    # Create image in (H, W, C) format
    image = np.random.rand(64, 64, 3).astype(np.float32)
    result = predictor.predict(image)
    
    assert isinstance(result, dict)
    assert 0 <= result['probability'] <= 1


def test_predict_batch(dummy_model_and_config):
    """Test batch inference."""
    model_path, config_path = dummy_model_and_config
    predictor = LightningPredictor(model_path, config_path=config_path, device='cpu')
    
    # Create batch of dummy images
    batch_size = 5
    images = [np.random.rand(3, 64, 64).astype(np.float32) for _ in range(batch_size)]
    
    results = predictor.predict_batch(images, batch_size=2)
    
    assert len(results) == batch_size
    assert all(isinstance(r, dict) for r in results)
    assert all(0 <= r['probability'] <= 1 for r in results)


def test_predict_batch_single_image(dummy_model_and_config):
    """Test batch inference with single image."""
    model_path, config_path = dummy_model_and_config
    predictor = LightningPredictor(model_path, config_path=config_path, device='cpu')
    
    # Single image in a list
    images = [np.random.rand(3, 64, 64).astype(np.float32)]
    results = predictor.predict_batch(images, batch_size=16)
    
    assert len(results) == 1
    assert 0 <= results[0]['probability'] <= 1


def test_predict_consistency(dummy_model_and_config):
    """Test that predictions are consistent for same input."""
    model_path, config_path = dummy_model_and_config
    predictor = LightningPredictor(model_path, config_path=config_path, device='cpu')
    
    image = np.ones((3, 64, 64), dtype=np.float32) * 0.5
    
    result1 = predictor.predict(image)
    result2 = predictor.predict(image)
    
    # Predictions should be identical
    assert result1['probability'] == result2['probability']
    assert result1['prediction'] == result2['prediction']


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
