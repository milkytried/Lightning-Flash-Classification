"""
Data loader for real Met Department Malaysia lightning dataset.
Supports both the original metadata feature set and a clean lat/lon/time-only variant.
"""

import torch
import h5py
import numpy as np
from typing import Tuple
from datetime import datetime
from torch.utils.data import Dataset, DataLoader


class LightningMetadataDataset(Dataset):
    """Load lightning strike metadata as features."""
    
    def __init__(self, hdf5_path: str, split: str = 'train', feature_mode: str = 'metadata'):
        """
        Args:
            hdf5_path: Path to HDF5 dataset
            split: 'train', 'val', or 'test'
            feature_mode: 'metadata' or 'clean'
        """
        self.hdf5_path = hdf5_path
        self.split = split
        self.feature_mode = feature_mode
        
        with h5py.File(hdf5_path, 'r') as f:
            self.split_indices = f[f'{split}_indices'][:]
    
    def __len__(self):
        return len(self.split_indices)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            (features, label)
        """
        global_idx = self.split_indices[idx]
        
        with h5py.File(self.hdf5_path, 'r') as f:
            latitude = f['latitudes'][global_idx]
            longitude = f['longitudes'][global_idx]
            amplitude = f['amplitudes'][global_idx]
            strike_type = f['strike_types'][global_idx]
            date_value = f['dates'][global_idx]
            label = f['labels'][global_idx]
            
            # Encode strike type: Cloud=0, Ground=1, None=2
            if isinstance(strike_type, bytes):
                strike_type = strike_type.decode('utf-8')
            
            if strike_type == 'Cloud':
                strike_code = 0.0
            elif strike_type == 'Ground':
                strike_code = 1.0
            else:
                strike_code = 2.0
            
            # Normalize to roughly [-1, 1] range
            lat_norm = (latitude - 3.0) / 3.0  # Center: ~3°N, range ~1-6°
            lon_norm = (longitude - 102.0) / 2.5  # Center: ~102°E, range ~100-104°
            amp_norm = np.clip(amplitude / 10.0, -1.0, 1.0)  # Amplitude typically -20 to +10

            if isinstance(date_value, bytes):
                date_value = date_value.decode('utf-8')
            date_obj = datetime.strptime(str(date_value), '%Y-%m-%d')
            month_norm = (date_obj.month - 6.5) / 5.5
            day_of_year_norm = (date_obj.timetuple().tm_yday - 183.0) / 182.0
            season = ((date_obj.month % 12) // 3) / 3.0

            if self.feature_mode == 'clean':
                features = torch.tensor(
                    [lat_norm, lon_norm, month_norm, day_of_year_norm, season],
                    dtype=torch.float32
                )
            else:
                # Preserve the current metadata model feature set for comparison.
                features = torch.tensor(
                    [lat_norm, lon_norm, amp_norm, strike_code],
                    dtype=torch.float32
                )
            
            label_tensor = torch.tensor(label, dtype=torch.float32)
            
            return features, label_tensor


def create_lightning_loaders(hdf5_path: str, batch_size: int = 512, feature_mode: str = 'metadata') -> dict:
    """Create train/val/test DataLoaders for lightning dataset."""
    
    train_dataset = LightningMetadataDataset(hdf5_path, split='train', feature_mode=feature_mode)
    val_dataset = LightningMetadataDataset(hdf5_path, split='val', feature_mode=feature_mode)
    test_dataset = LightningMetadataDataset(hdf5_path, split='test', feature_mode=feature_mode)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )
    
    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader,
    }
