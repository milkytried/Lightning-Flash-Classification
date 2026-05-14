"""
HDF5-based data loader for efficient batch loading during training.
Implements lazy loading to minimize GPU memory overhead.
"""

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Dict
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
            FileNotFoundError: If HDF5 file doesn't exist
            ValueError: If required datasets missing
        """
        self.hdf5_path = hdf5_path
        self.split = split
        self.augment = augment
        
        # Load metadata (not actual data) with validation
        try:
            with h5py.File(hdf5_path, 'r') as f:
                # Validate required datasets exist
                required_datasets = ['images', 'labels', f'{split}_indices']
                for dataset in required_datasets:
                    if dataset not in f:
                        raise ValueError(
                            f"Missing required dataset '{dataset}' in HDF5 file. "
                            f"Available: {list(f.keys())}"
                        )
                
                self.num_samples_total = f['images'].shape[0]
                self.image_shape = f['images'].shape[1:]
                self.split_indices = f[f'{split}_indices'][:]
        except FileNotFoundError:
            raise FileNotFoundError(f"HDF5 file not found: {hdf5_path}")
        except Exception as e:
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
            ])
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Lazy load single sample from disk.
        
        Args:
            idx (int): Index within split
        
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (image, label)
        """
        try:
            # Get global index in HDF5 file
            global_idx = self.split_indices[idx]
            
            # Lazy load from disk
            with h5py.File(self.hdf5_path, 'r') as f:
                image = f['images'][global_idx].copy()  # Copy to prevent issues
                label = f['labels'][global_idx]
            
            # Convert to float32
            image = image.astype(np.float32)
            label = np.float32(label)
            
            # Apply consistent augmentation pipeline
            augmented = self.transform(image=image)
            image = augmented['image']
            
            return image, torch.tensor(label, dtype=torch.float32)
        except Exception as e:
            raise RuntimeError(f"Error loading sample {idx}: {str(e)}")


def create_data_loaders(hdf5_path: str, batch_size: int = 16, num_workers: int = 0) -> Dict[str, DataLoader]:
    """
    Create train/val/test data loaders.
    
    Args:
        hdf5_path (str): Path to HDF5 dataset
        batch_size (int): Batch size (16 for RTX 3050)
        num_workers (int): Number of worker processes (0 for Windows)
    
    Returns:
        Dict[str, DataLoader]: {'train': DataLoader, 'val': DataLoader, 'test': DataLoader}
    """
    
    train_dataset = HDF5Dataset(hdf5_path, split='train', augment=True)
    val_dataset = HDF5Dataset(hdf5_path, split='val', augment=False)
    test_dataset = HDF5Dataset(hdf5_path, split='test', augment=False)
    
    # Only pin memory when an accelerator is available.
    use_pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
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
