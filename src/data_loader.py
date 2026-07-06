"""Minimal compatibility wrapper for the legacy HDF5 dataset loader used by older tests."""

from __future__ import annotations

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class HDF5Dataset(Dataset):
    def __init__(self, hdf5_path, split='train', augment=False):
        self.hdf5_path = hdf5_path
        self.split = split
        self.augment = augment
        with h5py.File(hdf5_path, 'r') as f:
            self.indices = f[f'{split}_indices'][:]
            self.images = f['images'][:]
            self.labels = f['labels'][:]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        global_idx = int(self.indices[idx])
        image = self.images[global_idx].astype(np.float32)
        label = float(self.labels[global_idx])
        image_tensor = torch.from_numpy(image).permute(2, 0, 1) / 255.0
        label_tensor = torch.tensor(label, dtype=torch.float32)
        return image_tensor, label_tensor


def create_data_loaders(hdf5_path, batch_size=16):
    return {
        'train': DataLoader(HDF5Dataset(hdf5_path, split='train', augment=False), batch_size=batch_size, shuffle=False),
        'val': DataLoader(HDF5Dataset(hdf5_path, split='val', augment=False), batch_size=batch_size, shuffle=False),
        'test': DataLoader(HDF5Dataset(hdf5_path, split='test', augment=False), batch_size=batch_size, shuffle=False),
    }
