"""
PyTorch DataLoader for Himawari-8 satellite image patches.

Loads 64×64 image patches with binary labels for CNN training.
"""

import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from PIL import Image
from typing import Tuple, Dict, Optional
import logging
from albumentations import (
    HorizontalFlip, VerticalFlip, Rotate, GaussNoise, Compose,
    Normalize, pytorch
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HimawariPatchDataset(Dataset):
    """
    PyTorch Dataset for Himawari-8 satellite image patches.
    
    Loads 64×64 patches with binary labels (lightning vs no lightning).
    """
    
    def __init__(self, dataset_csv: str, split: str = 'train', augment: bool = True):
        """
        Initialize dataset.
        
        Args:
            dataset_csv (str): Path to satellite_dataset.csv
            split (str): 'train', 'val', or 'test'
            augment (bool): Apply augmentations to training set
        """
        # Load CSV
        self.df = pd.read_csv(dataset_csv)
        
        # Filter by split
        self.df = self.df[self.df['split'] == split].reset_index(drop=True)
        
        self.split = split
        self.augment = augment and (split == 'train')
        
        logger.info(f"HimawariPatchDataset initialized")
        logger.info(f"  Split: {split}")
        logger.info(f"  Samples: {len(self.df)}")
        logger.info(f"  Positive: {(self.df['label'] == 1).sum()}")
        logger.info(f"  Negative: {(self.df['label'] == 0).sum()}")
        logger.info(f"  Augmentation: {self.augment}")
        
        # Initialize augmentation pipeline
        self.transform = self._get_augmentation_pipeline()
    
    def _get_augmentation_pipeline(self):
        """Create augmentation pipeline using albumentations."""
        if self.augment:
            return Compose([
                HorizontalFlip(p=0.5),
                VerticalFlip(p=0.5),
                Rotate(limit=15, p=0.5),
                GaussNoise(p=0.3),
                Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
                pytorch.ToTensorV2()
            ])
        else:
            # No augmentation for val/test
            return Compose([
                Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
                pytorch.ToTensorV2()
            ])
    
    def __len__(self) -> int:
        """Return number of samples."""
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get single sample.
        
        Args:
            idx (int): Sample index
        
        Returns:
            (image_tensor, label_tensor) tuple
            - image_tensor: (3, 64, 64) float tensor, normalized
            - label_tensor: scalar tensor (0 or 1)
        """
        row = self.df.iloc[idx]
        
        patch_path = row['path']
        label = row['label']
        
        # Load patch
        try:
            img = Image.open(patch_path).convert('RGB')
            patch = np.array(img, dtype=np.uint8)
        except Exception as e:
            logger.error(f"Error loading {patch_path}: {e}")
            # Return black patch as fallback
            patch = np.zeros((64, 64, 3), dtype=np.uint8)
        
        # Apply augmentation
        if self.transform:
            patch = self.transform(image=patch)['image']
        else:
            # Manual normalization if no transform
            patch = patch.astype(np.float32) / 255.0
            patch = torch.from_numpy(patch).permute(2, 0, 1)
        
        label_tensor = torch.tensor(label, dtype=torch.long)
        
        return patch, label_tensor


def create_himawari_loaders(dataset_csv: str, batch_size: int = 32, 
                           num_workers: int = 0) -> Dict[str, DataLoader]:
    """
    Create train/val/test DataLoaders.
    
    Args:
        dataset_csv (str): Path to satellite_dataset.csv
        batch_size (int): Batch size
        num_workers (int): Number of worker processes
    
    Returns:
        Dictionary with 'train', 'val', 'test' DataLoaders
    """
    loaders = {}
    
    for split in ['train', 'val', 'test']:
        dataset = HimawariPatchDataset(dataset_csv, split=split, augment=(split == 'train'))
        
        shuffle = (split == 'train')
        drop_last = (split == 'train')
        
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=num_workers,
            pin_memory=True
        )
        
        loaders[split] = loader
        logger.info(f"Created {split} loader: {len(dataset)} samples, {len(loader)} batches")
    
    return loaders


# Example usage
if __name__ == '__main__':
    # Create loaders
    dataset_csv = 'data/processed/satellite_dataset.csv'
    
    if os.path.exists(dataset_csv):
        print("Creating data loaders...")
        loaders = create_himawari_loaders(dataset_csv, batch_size=16)
        
        # Test loading a batch
        print("\nTesting train loader...")
        train_loader = loaders['train']
        
        batch_idx, (images, labels) = next(enumerate(train_loader))
        print(f"  Batch shape: {images.shape}")
        print(f"  Label shape: {labels.shape}")
        print(f"  Labels (first 5): {labels[:5].tolist()}")
        print(f"  Label distribution: {torch.bincount(labels)}")
    else:
        print(f"Dataset CSV not found: {dataset_csv}")
        print("Run satellite_dataset_builder.py first")
