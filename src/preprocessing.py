"""
Preprocessing module for Himawari-8 satellite imagery and MMD lightning data.

Converts raw netCDF4 satellite images and CSV lightning records into HDF5 dataset
optimized for GPU training.

Pipeline:
1. Read Himawari-8 netCDF4 files (IR, WV, VIS channels)
2. Read MMD Lightning CSV (timestamps, lat/lon)
3. Crop to Malaysia region
4. Create patches (64x64 pixels)
5. Label with lightning occurrence (0/1)
6. Handle class imbalance (downsampling)
7. Split into train/val/test
8. Save to HDF5 with lazy-loading indices
"""

import os
import numpy as np
import h5py
import pandas as pd
import xarray as xr
from pathlib import Path
from typing import Tuple, Dict, List, Optional
from datetime import datetime, timedelta
from tqdm import tqdm
import logging
from collections import Counter

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HimawariPreprocessor:
    """
    Preprocess Himawari-8 satellite data and MMD lightning records.
    
    Attributes:
        raw_himawari_dir (str): Directory containing netCDF4 Himawari-8 files
        raw_mmd_csv (str): Path to MMD Lightning CSV file
        output_h5 (str): Output HDF5 file path
        region_bbox (list): [lon_min, lon_max, lat_min, lat_max] for Malaysia
        patch_size (int): Size of image patches (64x64)
        num_channels (int): Number of channels (3: IR, WV, VIS)
        lead_time (int): Lead time in minutes for lightning occurrence
        downsample_ratio (float): Downsample negative (no lightning) samples
    """
    
    def __init__(
        self,
        raw_himawari_dir: str,
        raw_mmd_csv: str,
        output_h5: str,
        region_bbox: list = [100.0, 120.0, -5.0, 15.0],
        patch_size: int = 64,
        num_channels: int = 3,
        lead_time: int = 30,
        downsample_ratio: float = 0.2
    ):
        """Initialize preprocessor."""
        self.raw_himawari_dir = Path(raw_himawari_dir)
        self.raw_mmd_csv = Path(raw_mmd_csv)
        self.output_h5 = Path(output_h5)
        self.region_bbox = region_bbox
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.lead_time = lead_time
        self.downsample_ratio = downsample_ratio
        
        # Create output directory
        self.output_h5.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"HimawariPreprocessor initialized")
        logger.info(f"  Region: {region_bbox}")
        logger.info(f"  Patch size: {patch_size}x{patch_size}")
        logger.info(f"  Lead time: {lead_time} minutes")
    
    def load_himawari_file(self, filepath: str) -> Optional[np.ndarray]:
        """
        Load Himawari-8 netCDF4 file and extract channels.
        
        Args:
            filepath (str): Path to netCDF4 file
        
        Returns:
            np.ndarray: Shape (3, H, W) for IR, WV, VIS channels
        """
        try:
            ds = xr.open_dataset(filepath)
            
            # Extract channels (adjust variable names based on actual Himawari data)
            # Common variable names: ir_brightness_temp, wv_brightness_temp, vis_reflectance
            ir = ds['ir_brightness_temp'].values if 'ir_brightness_temp' in ds else ds['IR'].values
            wv = ds['wv_brightness_temp'].values if 'wv_brightness_temp' in ds else ds['WV'].values
            vis = ds['vis_reflectance'].values if 'vis_reflectance' in ds else ds['VIS'].values
            
            # Stack channels
            image = np.stack([ir, wv, vis], axis=0).astype(np.float32)
            
            # Normalize
            image = self._normalize_channels(image)
            
            ds.close()
            return image
        except Exception as e:
            logger.warning(f"Failed to load {filepath}: {e}")
            return None
    
    def _normalize_channels(self, image: np.ndarray) -> np.ndarray:
        """Normalize channels to [0, 1] range."""
        # IR: 180-330 K → [0, 1]
        image[0] = np.clip((image[0] - 180) / (330 - 180), 0, 1)
        # WV: 180-330 K → [0, 1]
        image[1] = np.clip((image[1] - 180) / (330 - 180), 0, 1)
        # VIS: 0-1 already in [0, 1]
        image[2] = np.clip(image[2], 0, 1)
        
        return image
    
    def load_mmd_lightning(self) -> pd.DataFrame:
        """
        Load MMD Lightning CSV.
        
        Expected columns: timestamp, latitude, longitude, intensity (optional)
        
        Returns:
            pd.DataFrame: Lightning records with parsed timestamps
        """
        try:
            df = pd.read_csv(self.raw_mmd_csv)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Filter to region
            df = df[
                (df['longitude'] >= self.region_bbox[0]) & 
                (df['longitude'] <= self.region_bbox[1]) &
                (df['latitude'] >= self.region_bbox[2]) & 
                (df['latitude'] <= self.region_bbox[3])
            ]
            
            logger.info(f"Loaded {len(df)} lightning records in region")
            return df
        except Exception as e:
            logger.error(f"Failed to load MMD CSV: {e}")
            raise
    
    def create_patches(
        self,
        image: np.ndarray,
        stride: Optional[int] = None
    ) -> List[np.ndarray]:
        """
        Extract patches from image with stride.
        
        Args:
            image (np.ndarray): Shape (C, H, W)
            stride (int): Stride between patches (default: patch_size/2)
        
        Returns:
            List[np.ndarray]: List of patches, each shape (C, patch_size, patch_size)
        """
        if stride is None:
            stride = self.patch_size // 2
        
        c, h, w = image.shape
        patches = []
        
        for i in range(0, h - self.patch_size, stride):
            for j in range(0, w - self.patch_size, stride):
                patch = image[:, i:i+self.patch_size, j:j+self.patch_size]
                patches.append(patch)
        
        return patches
    
    def label_patch(
        self,
        patch_center_lat: float,
        patch_center_lon: float,
        timestamp: datetime,
        lightning_df: pd.DataFrame
    ) -> int:
        """
        Determine if patch has lightning occurrence.
        
        Args:
            patch_center_lat (float): Patch center latitude
            patch_center_lon (float): Patch center longitude
            timestamp (datetime): Image timestamp
            lightning_df (pd.DataFrame): Lightning records
        
        Returns:
            int: 1 if lightning within lead_time and spatial distance, 0 otherwise
        """
        # Time window: image time to lead_time minutes later
        time_window = [timestamp, timestamp + timedelta(minutes=self.lead_time)]
        
        # Spatial distance threshold: ~10 km ≈ 0.1 degrees
        spatial_threshold = 0.1
        
        # Filter lightning in time and space
        lightning_match = lightning_df[
            (lightning_df['timestamp'] >= time_window[0]) &
            (lightning_df['timestamp'] <= time_window[1]) &
            (np.abs(lightning_df['latitude'] - patch_center_lat) <= spatial_threshold) &
            (np.abs(lightning_df['longitude'] - patch_center_lon) <= spatial_threshold)
        ]
        
        return 1 if len(lightning_match) > 0 else 0
    
    def preprocess_dataset(
        self,
        split_ratios: Dict[str, float] = {'train': 0.7, 'val': 0.15, 'test': 0.15}
    ) -> None:
        """
        Full preprocessing pipeline.
        
        Args:
            split_ratios (dict): Train/val/test split ratios
        """
        logger.info("Starting preprocessing pipeline...")
        
        # Load lightning data
        lightning_df = self.load_mmd_lightning()
        
        # Collect all patches and labels
        all_patches = []
        all_labels = []
        patch_metadata = []
        
        # Process Himawari-8 files
        himawari_files = sorted(self.raw_himawari_dir.glob("*.nc"))
        logger.info(f"Found {len(himawari_files)} Himawari-8 files")
        
        for filepath in tqdm(himawari_files, desc="Processing Himawari-8 files"):
            image = self.load_himawari_file(str(filepath))
            if image is None:
                continue
            
            # Extract timestamp from filename (adjust as needed)
            try:
                timestamp = datetime.strptime(Path(filepath).stem, "%Y%m%d_%H%M")
            except:
                continue
            
            # Create patches
            patches = self.create_patches(image)
            
            # Label each patch
            for idx, patch in enumerate(patches):
                # Calculate patch center (approximate)
                patch_center_lat = (self.region_bbox[2] + self.region_bbox[3]) / 2
                patch_center_lon = (self.region_bbox[0] + self.region_bbox[1]) / 2
                
                label = self.label_patch(
                    patch_center_lat, patch_center_lon, timestamp, lightning_df
                )
                
                all_patches.append(patch)
                all_labels.append(label)
                patch_metadata.append({
                    'timestamp': timestamp,
                    'file': filepath.name,
                    'patch_idx': idx
                })
        
        logger.info(f"Created {len(all_patches)} patches")
        
        # Handle class imbalance
        all_patches, all_labels = self._balance_dataset(all_patches, all_labels)
        logger.info(f"After balancing: {len(all_patches)} patches")
        
        # Convert to numpy
        images = np.array(all_patches, dtype=np.float32)
        labels = np.array(all_labels, dtype=np.float32)
        
        # Create train/val/test split
        self._create_splits(images, labels, split_ratios)
        
        logger.info(f"Dataset saved to {self.output_h5}")
    
    def _balance_dataset(
        self,
        patches: List[np.ndarray],
        labels: List[int]
    ) -> Tuple[List[np.ndarray], List[int]]:
        """Handle class imbalance by downsampling negatives."""
        # Count classes
        label_counts = Counter(labels)
        logger.info(f"Class distribution before balancing: {dict(label_counts)}")
        
        # Separate positive and negative samples
        pos_indices = [i for i, l in enumerate(labels) if l == 1]
        neg_indices = [i for i, l in enumerate(labels) if l == 0]
        
        # Downsample negatives
        num_negatives_keep = int(len(pos_indices) / (1 - self.downsample_ratio))
        if len(neg_indices) > num_negatives_keep:
            neg_indices = np.random.choice(
                neg_indices, size=num_negatives_keep, replace=False
            )
        
        # Combine
        balanced_indices = pos_indices + list(neg_indices)
        balanced_patches = [patches[i] for i in balanced_indices]
        balanced_labels = [labels[i] for i in balanced_indices]
        
        label_counts_balanced = Counter(balanced_labels)
        logger.info(f"Class distribution after balancing: {dict(label_counts_balanced)}")
        
        return balanced_patches, balanced_labels
    
    def _create_splits(
        self,
        images: np.ndarray,
        labels: np.ndarray,
        split_ratios: Dict[str, float]
    ) -> None:
        """Create train/val/test splits and save to HDF5."""
        n = len(images)
        
        # Shuffle
        indices = np.random.permutation(n)
        images = images[indices]
        labels = labels[indices]
        
        # Calculate split points
        train_size = int(n * split_ratios['train'])
        val_size = int(n * split_ratios['val'])
        
        train_indices = np.arange(0, train_size)
        val_indices = np.arange(train_size, train_size + val_size)
        test_indices = np.arange(train_size + val_size, n)
        
        logger.info(f"Split sizes: train={len(train_indices)}, val={len(val_indices)}, test={len(test_indices)}")
        
        # Save to HDF5
        with h5py.File(self.output_h5, 'w') as f:
            # Datasets
            f.create_dataset('images', data=images, compression='gzip', compression_opts=4)
            f.create_dataset('labels', data=labels, compression='gzip')
            
            # Indices for splits
            f.create_dataset('train_indices', data=train_indices)
            f.create_dataset('val_indices', data=val_indices)
            f.create_dataset('test_indices', data=test_indices)
            
            # Metadata
            f.attrs['num_channels'] = self.num_channels
            f.attrs['patch_size'] = self.patch_size
            f.attrs['total_samples'] = n
            f.attrs['train_samples'] = len(train_indices)
            f.attrs['val_samples'] = len(val_indices)
            f.attrs['test_samples'] = len(test_indices)


def preprocess_from_config(config_path: str = 'config.yaml') -> None:
    """
    Run preprocessing from YAML config.
    
    Args:
        config_path (str): Path to config.yaml
    """
    import yaml
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    preprocessor = HimawariPreprocessor(
        raw_himawari_dir=config['data']['raw_himawari_dir'],
        raw_mmd_csv=config['data']['raw_mmd_csv'],
        output_h5=config['data']['processed_dataset'],
        region_bbox=config['data']['region_bbox'],
        patch_size=config['preprocessing']['patch_size'],
        num_channels=config['preprocessing']['num_channels'],
        lead_time=config['data']['lead_time_window'][1],
        downsample_ratio=config['preprocessing']['downsample_ratio']
    )
    
    preprocessor.preprocess_dataset()


if __name__ == '__main__':
    # Example usage (requires real data)
    import yaml
    
    # Load config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Verify data exists
    raw_himawari = Path(config['data']['raw_himawari_dir'])
    raw_mmd = Path(config['data']['raw_mmd_csv'])
    
    if not raw_himawari.exists():
        print(f"ERROR: Himawari data directory not found: {raw_himawari}")
        print("Please download Himawari-8 data from JMA archive")
        exit(1)
    
    if not raw_mmd.exists():
        print(f"ERROR: MMD Lightning CSV not found: {raw_mmd}")
        print("Please obtain MMD Lightning Detection System data")
        exit(1)
    
    # Run preprocessing
    print("Starting preprocessing pipeline...")
    preprocess_from_config('config.yaml')
    print("Preprocessing complete!")
