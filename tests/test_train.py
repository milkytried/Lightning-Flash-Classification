"""
Unit tests for train module.
"""

import pytest
import torch
import numpy as np
import h5py
import tempfile
from pathlib import Path

from src.train import set_seed, train_epoch, validate
from src.model_arch import LightningResNet50, FocalLoss
from src.data_loader import HDF5Dataset, DataLoader


@pytest.fixture
def dummy_hdf5_file():
    """Create a dummy HDF5 file for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        hdf5_path = Path(tmpdir) / "test_dataset.h5"
        
        num_samples = 100
        with h5py.File(hdf5_path, 'w') as f:
            f.create_dataset('images', data=np.random.rand(num_samples, 64, 64, 3).astype(np.float32))
            f.create_dataset('labels', data=np.random.randint(0, 2, num_samples).astype(np.float32))
            f.create_dataset('train_indices', data=np.arange(80))
            f.create_dataset('val_indices', data=np.arange(80, 90))
            f.create_dataset('test_indices', data=np.arange(90, 100))
        
        yield str(hdf5_path)


def test_set_seed_reproducibility():
    """Test set_seed produces reproducible results."""
    set_seed(42)
    vals1 = torch.randn(10)
    
    set_seed(42)
    vals2 = torch.randn(10)
    
    assert torch.allclose(vals1, vals2)


def test_train_epoch(dummy_hdf5_file):
    """Test train_epoch runs without errors."""
    device = torch.device('cpu')
    model = LightningResNet50(num_input_channels=3).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = FocalLoss(alpha=0.25, gamma=2.0)
    
    # Create data loader
    dataset = HDF5Dataset(dummy_hdf5_file, split='train', augment=False)
    train_loader = DataLoader(dataset, batch_size=16, shuffle=False)
    
    # Run one epoch
    loss = train_epoch(model, train_loader, optimizer, criterion, device)
    
    assert isinstance(loss, float)
    assert loss > 0
    assert not np.isnan(loss)


def test_validate(dummy_hdf5_file):
    """Test validate function runs without errors."""
    device = torch.device('cpu')
    model = LightningResNet50(num_input_channels=3).to(device)
    criterion = FocalLoss(alpha=0.25, gamma=2.0)
    
    # Create data loader
    dataset = HDF5Dataset(dummy_hdf5_file, split='val', augment=False)
    val_loader = DataLoader(dataset, batch_size=16, shuffle=False)
    
    # Run validation
    loss, preds, labels = validate(model, val_loader, criterion, device)
    
    assert isinstance(loss, float)
    assert loss > 0
    assert isinstance(preds, np.ndarray)
    assert isinstance(labels, np.ndarray)
    assert len(preds) == len(labels)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
