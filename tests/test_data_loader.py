"""
Unit tests for data_loader module.
"""

import pytest
import torch
import numpy as np
import h5py
import tempfile
from pathlib import Path

from src.data_loader import HDF5Dataset, create_data_loaders


@pytest.fixture
def dummy_hdf5_file():
    """Create a dummy HDF5 file for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        hdf5_path = Path(tmpdir) / "test_dataset.h5"
        
        # Create dummy dataset
        num_samples = 100
        with h5py.File(hdf5_path, 'w') as f:
            f.create_dataset('images', data=np.random.rand(num_samples, 64, 64, 3).astype(np.float32))
            f.create_dataset('labels', data=np.random.randint(0, 2, num_samples).astype(np.float32))
            f.create_dataset('train_indices', data=np.arange(80))
            f.create_dataset('val_indices', data=np.arange(80, 90))
            f.create_dataset('test_indices', data=np.arange(90, 100))
        
        yield str(hdf5_path)


def test_hdf5_dataset_initialization(dummy_hdf5_file):
    """Test HDF5Dataset initializes without errors."""
    dataset = HDF5Dataset(dummy_hdf5_file, split='train', augment=False)
    assert dataset is not None
    assert len(dataset) == 80


def test_hdf5_dataset_val_split(dummy_hdf5_file):
    """Test val split initialization."""
    dataset = HDF5Dataset(dummy_hdf5_file, split='val', augment=False)
    assert len(dataset) == 10


def test_hdf5_dataset_test_split(dummy_hdf5_file):
    """Test test split initialization."""
    dataset = HDF5Dataset(dummy_hdf5_file, split='test', augment=False)
    assert len(dataset) == 10


def test_getitem_returns_correct_shapes(dummy_hdf5_file):
    """Test __getitem__ returns correct tensor shapes."""
    dataset = HDF5Dataset(dummy_hdf5_file, split='train', augment=False)
    image, label = dataset[0]
    
    assert isinstance(image, torch.Tensor)
    assert isinstance(label, torch.Tensor)
    assert image.shape == torch.Size([3, 64, 64])
    assert label.shape == torch.Size([])


def test_getitem_value_range(dummy_hdf5_file):
    """Test __getitem__ returns values in valid range."""
    dataset = HDF5Dataset(dummy_hdf5_file, split='train', augment=False)
    image, label = dataset[0]
    
    # Image should be in [0, 1] or similar
    assert image.min() >= 0
    assert image.max() <= 1
    
    # Label should be 0 or 1
    assert label in [0, 1]


def test_create_data_loaders(dummy_hdf5_file):
    """Test create_data_loaders returns dict with all splits."""
    loaders = create_data_loaders(dummy_hdf5_file, batch_size=16)
    
    assert 'train' in loaders
    assert 'val' in loaders
    assert 'test' in loaders
    assert all(hasattr(loader, '__iter__') for loader in loaders.values())


def test_train_loader_batch_shape(dummy_hdf5_file):
    """Test train loader returns correct batch shapes."""
    loaders = create_data_loaders(dummy_hdf5_file, batch_size=16)
    train_loader = loaders['train']
    
    for images, labels in train_loader:
        assert images.shape[0] <= 16  # Batch size
        assert images.shape[1:] == torch.Size([3, 64, 64])  # Image shape
        assert labels.shape[0] == images.shape[0]  # Same batch size
        break


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
