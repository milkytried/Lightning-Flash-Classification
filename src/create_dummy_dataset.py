"""
Utility to create dummy HDF5 dataset for testing the training pipeline.

This script generates a small test dataset with realistic shapes and properties
for validating the entire training workflow without requiring real satellite data.
"""

import h5py
import numpy as np
from pathlib import Path
from typing import Dict, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_dummy_dataset(
    output_path: str = 'data/processed/dataset.h5',
    num_samples: int = 500,
    patch_size: int = 64,
    num_channels: int = 3,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    positive_ratio: float = 0.2
) -> None:
    """
    Create dummy HDF5 dataset for testing.
    
    Args:
        output_path (str): Path to save HDF5 file
        num_samples (int): Total number of samples
        patch_size (int): Size of image patches (64x64)
        num_channels (int): Number of channels (3: IR, WV, VIS)
        train_ratio (float): Training set ratio
        val_ratio (float): Validation set ratio
        test_ratio (float): Test set ratio
        positive_ratio (float): Ratio of positive (lightning) samples
    """
    
    # Create output directory
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Creating dummy dataset: {num_samples} samples")
    
    # Generate random images
    images = np.random.rand(num_samples, num_channels, patch_size, patch_size).astype(np.float32)
    
    # Normalize to [0, 1]
    images = np.clip(images, 0, 1)
    
    # Generate labels (20% positive, 80% negative)
    labels = np.random.rand(num_samples) < positive_ratio
    labels = labels.astype(np.float32)
    
    logger.info(f"Label distribution: {np.sum(labels)/len(labels)*100:.1f}% positive")
    
    # Create train/val/test split
    indices = np.random.permutation(num_samples)
    
    train_size = int(num_samples * train_ratio)
    val_size = int(num_samples * val_ratio)
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]
    
    logger.info(f"Split: train={len(train_indices)}, val={len(val_indices)}, test={len(test_indices)}")
    
    # Save to HDF5
    with h5py.File(output_path, 'w') as f:
        # Datasets
        f.create_dataset('images', data=images, compression='gzip', compression_opts=4)
        f.create_dataset('labels', data=labels, compression='gzip')
        
        # Indices
        f.create_dataset('train_indices', data=train_indices)
        f.create_dataset('val_indices', data=val_indices)
        f.create_dataset('test_indices', data=test_indices)
        
        # Metadata
        f.attrs['num_channels'] = num_channels
        f.attrs['patch_size'] = patch_size
        f.attrs['total_samples'] = num_samples
        f.attrs['train_samples'] = len(train_indices)
        f.attrs['val_samples'] = len(val_indices)
        f.attrs['test_samples'] = len(test_indices)
        f.attrs['positive_ratio'] = positive_ratio
    
    logger.info(f"Dataset saved to {output_path}")
    
    # Verify
    with h5py.File(output_path, 'r') as f:
        logger.info(f"Verification:")
        logger.info(f"  Images shape: {f['images'].shape}")
        logger.info(f"  Labels shape: {f['labels'].shape}")
        logger.info(f"  Train indices: {len(f['train_indices'])}")
        logger.info(f"  Val indices: {len(f['val_indices'])}")
        logger.info(f"  Test indices: {len(f['test_indices'])}")


if __name__ == '__main__':
    # Create small test dataset
    create_dummy_dataset(
        output_path='data/processed/dataset.h5',
        num_samples=500,
        patch_size=64,
        num_channels=3,
        positive_ratio=0.2
    )
    
    print("\n✅ Dummy dataset created successfully!")
    print("Next step: Run training with: python -m src.train")
