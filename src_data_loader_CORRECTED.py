"""
HDF5-based data loader for efficient batch loading during training.
Implements lazy loading to minimize GPU memory overhead.
"""

from typing import Dict, Tuple, Optional
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2


class HDF5Dataset(Dataset):
    """
    Lazy-loading HDF5 dataset for Lightning classification.
    
    Features:
    - Only loads batch into memory when accessed
    - Applies augmentation on CPU (frees GPU)
    - Supports train/val/test splits via indices
    """
    
    def __init__(self, hdf5_path: str, split: str = 'train', augment: bool = False) -> None:
        """
        Args:
            hdf5_path (str): Path to HDF5 dataset file
            split (str): 'train', 'val', or 'test'
            augment (bool): Apply data augmentation (train only)
        
        Raises:
            FileNotFoundError: If HDF5 file does not exist
            KeyError: If required datasets not found in HDF5 file
            ValueError: If split is not valid
        """
        if split not in ['train', 'val', 'test']:
            raise ValueError(f"Split must be 'train', 'val', or 'test', got {split}")
        
        self.hdf5_path = hdf5_path
        self.split = split
        self.augment = augment
        
        # Load metadata (not actual data)
        try:
            with h5py.File(hdf5_path, 'r') as f:
                self.num_samples_total = f['images'].shape[0]
                self.image_shape = f['images'].shape[1:]
                
                # Check for split indices
                split_key = f'{split}_indices'
                if split_key not in f:
                    raise KeyError(f"Dataset missing '{split_key}'. Available keys: {list(f.keys())}")
                
                self.split_indices = f[split_key][:]
        except FileNotFoundError:
            raise FileNotFoundError(f"HDF5 file not found: {hdf5_path}")
        except (KeyError, OSError) as e:
            raise RuntimeError(f"Error reading HDF5 file {hdf5_path}: {str(e)}")
        
        self.num_samples = len(self.split_indices)
        
        # Augmentation pipeline (CPU-based)
        if augment:
            self.transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Rotate(limit=15, p=0.5),
                A.GaussNoise(p=0.2),
                A.ToFloat(),
                ToTensorV2(),
            ], is_check_shapes=False)
        else:
            self.transform = A.Compose([
                A.ToFloat(),
                ToTensorV2(),
            ], is_check_shapes=False)
    
    def __len__(self) -> int:
        """Return number of samples in split."""
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Lazy load single sample from disk.
        
        Args:
            idx (int): Index within split
        
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (image, label) tensors
        
        Raises:
            IndexError: If idx is out of bounds
        """
        if idx < 0 or idx >= self.num_samples:
            raise IndexError(f"Index {idx} out of bounds for split with {self.num_samples} samples")
        
        # Get global index in HDF5 file
        global_idx = self.split_indices[idx]
        
        # Lazy load from disk
        try:
            with h5py.File(self.hdf5_path, 'r') as f:
                image = f['images'][global_idx]
                label = f['labels'][global_idx]
        except (OSError, KeyError) as e:
            raise RuntimeError(f"Error reading sample {idx} (global {global_idx}): {str(e)}")
        
        # Convert to float32
        image = image.astype(np.float32)
        label = np.float32(label)
        
        # Apply augmentation AND conversion to tensor through transform pipeline
        augmented = self.transform(image=image)
        image = augmented['image']
        
        return image, torch.tensor(label, dtype=torch.float32)


def create_data_loaders(
    hdf5_path: str, 
    batch_size: int = 16, 
    num_workers: int = 0
) -> Dict[str, DataLoader]:
    """
    Create train/val/test data loaders.
    
    Args:
        hdf5_path (str): Path to HDF5 dataset
        batch_size (int): Batch size (16 for RTX 3050 with 8GB VRAM)
        num_workers (int): Number of worker processes (0 for Windows, ≤2 for Linux)
    
    Returns:
        Dict[str, DataLoader]: Dictionary with keys 'train', 'val', 'test'
    
    Raises:
        FileNotFoundError: If HDF5 file does not exist
        ValueError: If batch_size is invalid
    """
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    
    if not isinstance(num_workers, int) or num_workers < 0:
        raise ValueError(f"num_workers must be non-negative integer, got {num_workers}")
    
    try:
        train_dataset = HDF5Dataset(hdf5_path, split='train', augment=True)
        val_dataset = HDF5Dataset(hdf5_path, split='val', augment=False)
        test_dataset = HDF5Dataset(hdf5_path, split='test', augment=False)
    except (FileNotFoundError, RuntimeError, KeyError) as e:
        raise RuntimeError(f"Failed to create datasets: {str(e)}")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,  # Faster GPU transfer
        drop_last=False,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    
    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader,
    }


if __name__ == '__main__':
    # Test: Load one batch
    try:
        loaders = create_data_loaders('data/processed/dataset.h5', batch_size=16)
        train_loader = loaders['train']
        
        for images, labels in train_loader:
            print(f"Batch shape: {images.shape}")
            print(f"Labels shape: {labels.shape}")
            print(f"Label values: {labels[:5]}")
            break
    except FileNotFoundError:
        print("Dataset not found. Run preprocessing first.")
    except Exception as e:
        print(f"Error: {e}")
