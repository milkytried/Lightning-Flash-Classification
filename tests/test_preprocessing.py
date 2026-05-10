"""
Unit tests for preprocessing module.
"""

import pytest
import numpy as np
import h5py
import tempfile
from pathlib import Path
from datetime import datetime
import pandas as pd

from src.preprocessing import HimawariPreprocessor


@pytest.fixture
def preprocessor():
    """Create a temporary preprocessor instance for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Create dummy directories
        himawari_dir = tmpdir / "himawari"
        himawari_dir.mkdir()
        
        # Create dummy MMD CSV
        mmd_csv = tmpdir / "mmd_lightning.csv"
        mmd_data = pd.DataFrame({
            'timestamp': ['2023-01-01 12:00:00'],
            'latitude': [10.0],
            'longitude': [110.0],
            'intensity': [1.0]
        })
        mmd_data.to_csv(mmd_csv, index=False)
        
        output_h5 = tmpdir / "dataset.h5"
        
        processor = HimawariPreprocessor(
            raw_himawari_dir=str(himawari_dir),
            raw_mmd_csv=str(mmd_csv),
            output_h5=str(output_h5),
            region_bbox=[100.0, 120.0, -5.0, 15.0],
            patch_size=64,
            num_channels=3
        )
        
        yield processor


def test_initialization(preprocessor):
    """Test preprocessor initializes correctly."""
    assert preprocessor.patch_size == 64
    assert preprocessor.num_channels == 3
    assert preprocessor.downsample_ratio == 0.2


def test_normalize_channels(preprocessor):
    """Test channel normalization."""
    # Create test image with 3 channels
    image = np.zeros((3, 100, 100), dtype=np.float32)
    
    # Set values
    image[0] = 255.0  # IR
    image[1] = 255.0  # WV
    image[2] = 0.5    # VIS
    
    normalized = preprocessor._normalize_channels(image.copy())
    
    # Check values are in [0, 1]
    assert normalized.min() >= 0.0
    assert normalized.max() <= 1.0


def test_create_patches(preprocessor):
    """Test patch creation."""
    image = np.random.rand(3, 256, 256).astype(np.float32)
    patches = preprocessor.create_patches(image, stride=64)
    
    # Verify patch shape
    assert len(patches) > 0
    for patch in patches:
        assert patch.shape == (3, 64, 64)


def test_label_patch(preprocessor):
    """Test patch labeling."""
    # Create dummy lightning data
    lightning_df = pd.DataFrame({
        'timestamp': [datetime(2023, 1, 1, 12, 0, 0)],
        'latitude': [10.0],
        'longitude': [110.0]
    })
    
    # Test labeling
    timestamp = datetime(2023, 1, 1, 12, 0, 0)
    label = preprocessor.label_patch(
        patch_center_lat=10.0,
        patch_center_lon=110.0,
        timestamp=timestamp,
        lightning_df=lightning_df
    )
    
    assert label in [0, 1]


def test_balance_dataset(preprocessor):
    """Test dataset balancing."""
    # Create imbalanced dataset (90% negative, 10% positive)
    patches = [np.random.rand(3, 64, 64) for _ in range(100)]
    labels = [0] * 90 + [1] * 10
    
    balanced_patches, balanced_labels = preprocessor._balance_dataset(patches, labels)
    
    # Check balance improved
    assert len(balanced_patches) == len(balanced_labels)
    assert len(balanced_patches) < len(patches)  # Downsampled


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
