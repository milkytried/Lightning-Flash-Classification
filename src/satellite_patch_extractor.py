"""
Extract 64×64 satellite image patches from Himawari-8 PNGs.

Creates positive samples (at lightning locations) and negative samples 
(random areas without lightning) for training CNN classifier.
"""

import numpy as np
import os
from PIL import Image
from pathlib import Path
from typing import Tuple, List, Optional, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SatellitePatchExtractor:
    """Extract image patches from satellite PNGs with lightning labels."""
    
    def __init__(self, png_loader, output_dir: str = 'data/processed/patches', 
                 patch_size: int = 64, lead_time_minutes: int = 60):
        """
        Initialize patch extractor.
        
        Args:
            png_loader: HimawariPNGLoader instance
            output_dir: Directory to save patches
            patch_size: Size of patches (64×64)
            lead_time_minutes: Time window for labeling lightning
        """
        self.png_loader = png_loader
        self.output_dir = Path(output_dir)
        self.patch_size = patch_size
        self.lead_time_minutes = lead_time_minutes
        
        # Create output directories
        for split in ['train', 'val', 'test']:
            for label in ['positive', 'negative']:
                (self.output_dir / split / label).mkdir(parents=True, exist_ok=True)
        
        logger.info(f"SatellitePatchExtractor initialized")
        logger.info(f"  Output: {self.output_dir}")
        logger.info(f"  Patch size: {patch_size}×{patch_size}")
        logger.info(f"  Lead time: {lead_time_minutes} minutes")
    
    def extract_patch_at_location(self, png_array: np.ndarray, 
                                  center_lat: float, center_lon: float,
                                  patch_size: int = 64) -> Tuple[Optional[np.ndarray], Tuple[int, int]]:
        """
        Extract patch centered at lightning location.
        
        Args:
            png_array: (H, W, 3) numpy array of PNG image
            center_lat: Latitude of lightning strike
            center_lon: Longitude of lightning strike
            patch_size: Size of patch
        
        Returns:
            (patch, (x, y)) tuple where patch is (patch_size, patch_size, 3) or None if out of bounds
            (x, y) are center pixel coordinates
        """
        # Validate coordinates
        if not self.png_loader.validate_coordinates(center_lat, center_lon):
            logger.debug(f"Coordinates ({center_lat}, {center_lon}) out of bounds")
            return None, (0, 0)
        
        # Convert to pixel coordinates
        x, y = self.png_loader.latlon_to_pixel(center_lat, center_lon)
        
        # Extract patch with boundary checking
        half_size = patch_size // 2
        y_min = max(0, y - half_size)
        y_max = min(png_array.shape[0], y + half_size)
        x_min = max(0, x - half_size)
        x_max = min(png_array.shape[1], x + half_size)
        
        # Skip if patch is too close to boundary
        if (y_max - y_min) < patch_size * 0.9 or (x_max - x_min) < patch_size * 0.9:
            logger.debug(f"Patch too close to boundary at ({x}, {y})")
            return None, (x, y)
        
        patch = png_array[y_min:y_max, x_min:x_max, :].copy()
        
        return patch, (x, y)
    
    def extract_negative_patches(self, png_array: np.ndarray, 
                                lightning_locs: List[Tuple[int, int]],
                                n_samples: int = 5,
                                patch_size: int = 64,
                                exclusion_radius: int = 100) -> List[Tuple[np.ndarray, Tuple[int, int]]]:
        """
        Extract random patches from areas WITHOUT lightning.
        
        Args:
            png_array: (H, W, 3) numpy array
            lightning_locs: List of (x, y) pixel coordinates of lightning
            n_samples: Number of negative patches to extract
            patch_size: Size of patches
            exclusion_radius: Radius around lightning to exclude
        
        Returns:
            List of (patch, (x, y)) tuples
        """
        negative_patches = []
        half_size = patch_size // 2
        
        # Create mask of valid sample locations (not too close to lightning)
        h, w = png_array.shape[:2]
        valid_mask = np.ones((h, w), dtype=bool)
        
        # Mark areas near lightning as invalid
        for lx, ly in lightning_locs:
            y_min = max(0, ly - exclusion_radius)
            y_max = min(h, ly + exclusion_radius)
            x_min = max(0, lx - exclusion_radius)
            x_max = min(w, lx + exclusion_radius)
            valid_mask[y_min:y_max, x_min:x_max] = False
        
        # Also mark boundary regions as invalid (too close to edge)
        boundary = patch_size
        valid_mask[:boundary, :] = False
        valid_mask[-boundary:, :] = False
        valid_mask[:, :boundary] = False
        valid_mask[:, -boundary:] = False
        
        # Sample random valid locations
        valid_coords = np.where(valid_mask)
        
        if len(valid_coords[0]) == 0:
            logger.warning(f"No valid locations for negative patches")
            return negative_patches
        
        n_valid = len(valid_coords[0])
        n_to_sample = min(n_samples, n_valid)
        
        sample_indices = np.random.choice(n_valid, size=n_to_sample, replace=False)
        
        for idx in sample_indices:
            y = valid_coords[0][idx]
            x = valid_coords[1][idx]
            
            # Extract patch
            y_min = max(0, y - half_size)
            y_max = min(h, y + half_size)
            x_min = max(0, x - half_size)
            x_max = min(w, x + half_size)
            
            if (y_max - y_min) >= patch_size * 0.9 and (x_max - x_min) >= patch_size * 0.9:
                patch = png_array[y_min:y_max, x_min:x_max, :].copy()
                negative_patches.append((patch, (x, y)))
        
        return negative_patches
    
    def save_patch(self, patch: np.ndarray, output_path: Path) -> bool:
        """
        Save patch as PNG.
        
        Args:
            patch: (H, W, 3) numpy array
            output_path: Path to save PNG
        
        Returns:
            True if successful
        """
        try:
            # Ensure patch is uint8
            if patch.dtype != np.uint8:
                patch = np.clip(patch, 0, 255).astype(np.uint8)
            
            # Save as PNG
            img = Image.fromarray(patch, mode='RGB')
            img.save(output_path)
            
            return True
        except Exception as e:
            logger.error(f"Error saving patch to {output_path}: {e}")
            return False
    
    def process_png_for_dataset(self, png_array: np.ndarray, 
                               png_datetime,
                               lightning_df,
                               split: str = 'train',
                               n_negative_per_positive: int = 1) -> Dict:
        """
        Extract patches from single PNG image.
        
        Args:
            png_array: (H, W, 3) numpy array of PNG
            png_datetime: Datetime of PNG
            lightning_df: DataFrame with lightning records (must have timestamp, latitude, longitude)
            split: 'train', 'val', or 'test'
            n_negative_per_positive: Number of negative patches per positive
        
        Returns:
            Dictionary with:
            - n_positive: Number of positive patches extracted
            - n_negative: Number of negative patches extracted
            - patches: List of (patch_path, label, x, y, lat, lon) tuples
        """
        result = {
            'n_positive': 0,
            'n_negative': 0,
            'patches': []
        }
        
        if lightning_df is None or len(lightning_df) == 0:
            logger.debug(f"No lightning records for {png_datetime}")
            return result
        
        # Filter lightning records within lead time window
        time_min = png_datetime - pd.Timedelta(minutes=self.lead_time_minutes)
        time_max = png_datetime + pd.Timedelta(minutes=self.lead_time_minutes)
        
        # Assuming lightning_df has 'timestamp' column
        mask = (lightning_df['timestamp'] >= time_min) & (lightning_df['timestamp'] <= time_max)
        window_df = lightning_df[mask]
        
        if len(window_df) == 0:
            logger.debug(f"No lightning in lead time window")
            return result
        
        lightning_locs = []
        
        # Extract positive patches (at lightning locations)
        for _, row in window_df.iterrows():
            lat = row['latitude']
            lon = row['longitude']
            
            patch, (x, y) = self.extract_patch_at_location(png_array, lat, lon, self.patch_size)
            
            if patch is not None:
                # Save patch
                patch_id = f"{png_datetime.strftime('%Y%m%d_%H%M%S')}_{len(result['patches'])}"
                patch_path = self.output_dir / split / 'positive' / f"{patch_id}.png"
                
                if self.save_patch(patch, patch_path):
                    result['patches'].append({
                        'path': str(patch_path),
                        'label': 1,
                        'x': x,
                        'y': y,
                        'lat': lat,
                        'lon': lon,
                        'split': split
                    })
                    result['n_positive'] += 1
                    lightning_locs.append((x, y))
        
        # Extract negative patches
        if result['n_positive'] > 0:
            n_negative = result['n_positive'] * n_negative_per_positive
            negative_patches = self.extract_negative_patches(
                png_array, lightning_locs, n_negative, self.patch_size
            )
            
            for i, (patch, (x, y)) in enumerate(negative_patches):
                # Save patch
                patch_id = f"{png_datetime.strftime('%Y%m%d_%H%M%S')}_neg_{i}"
                patch_path = self.output_dir / split / 'negative' / f"{patch_id}.png"
                
                if self.save_patch(patch, patch_path):
                    lat, lon = self.png_loader.pixel_to_latlon(x, y)
                    result['patches'].append({
                        'path': str(patch_path),
                        'label': 0,
                        'x': x,
                        'y': y,
                        'lat': lat,
                        'lon': lon,
                        'split': split
                    })
                    result['n_negative'] += 1
        
        return result


# Example usage
if __name__ == '__main__':
    from himawari_png_loader import HimawariPNGLoader
    from lightning_csv_parser import LightningCSVParser
    import pandas as pd
    
    # Initialize loaders
    png_loader = HimawariPNGLoader('data/raw/himawari8_pngs')
    csv_parser = LightningCSVParser('data/raw/himawari8_pngs')
    
    # Initialize extractor
    extractor = SatellitePatchExtractor(png_loader, 'data/processed/patches')
    
    # Load a PNG
    pngs = png_loader.find_png_files()
    if pngs:
        png_path, png_dt = pngs[-1]
        png_array = png_loader.load_png(png_path)
        
        # Load lightning for this date
        lightning_df = csv_parser.load_all_lightning(
            pd.Timestamp(png_dt).normalize(),
            pd.Timestamp(png_dt).normalize() + pd.Timedelta(days=1)
        )
        
        # Extract patches
        result = extractor.process_png_for_dataset(png_array, png_dt, lightning_df, split='test')
        
        print(f"\nExtracted patches:")
        print(f"  Positive: {result['n_positive']}")
        print(f"  Negative: {result['n_negative']}")
        print(f"  Total: {len(result['patches'])}")
