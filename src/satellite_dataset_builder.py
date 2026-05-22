"""
Build satellite patch dataset index.

Iterates over PNG files, extracts patches, and creates indexed CSV with
train/val/test split (time-based to avoid temporal leakage).
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SatelliteDatasetBuilder:
    """Build indexed satellite patch dataset."""
    
    def __init__(self, png_loader, csv_parser, patch_extractor,
                 output_dir: str = 'data/processed',
                 train_ratio: float = 0.7,
                 val_ratio: float = 0.15,
                 test_ratio: float = 0.15):
        """
        Initialize dataset builder.
        
        Args:
            png_loader: HimawariPNGLoader instance
            csv_parser: LightningCSVParser instance
            patch_extractor: SatellitePatchExtractor instance
            output_dir: Directory for output files
            train_ratio: Fraction of dates for training
            val_ratio: Fraction of dates for validation
            test_ratio: Fraction of dates for testing
        """
        self.png_loader = png_loader
        self.csv_parser = csv_parser
        self.patch_extractor = patch_extractor
        self.output_dir = Path(output_dir)
        
        # Validate ratios
        total = train_ratio + val_ratio + test_ratio
        if abs(total - 1.0) > 0.001:
            raise ValueError(f"Ratios must sum to 1.0, got {total}")
        
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        
        logger.info(f"SatelliteDatasetBuilder initialized")
        logger.info(f"  Train/Val/Test: {train_ratio:.1%} / {val_ratio:.1%} / {test_ratio:.1%}")
    
    def _assign_split(self, date: datetime) -> str:
        """
        Assign split based on date (time-based, not random).
        
        Splits dataset chronologically to avoid temporal leakage.
        
        Args:
            date: Datetime of image
        
        Returns:
            'train', 'val', or 'test'
        """
        # This will be updated when we know date range
        # For now, use simple heuristic based on datetime
        return 'train'  # Default
    
    def build_dataset(self, sample_limit: Optional[int] = None,
                      start_date: Optional[datetime] = None,
                      end_date: Optional[datetime] = None) -> pd.DataFrame:
        """
        Build complete dataset by extracting patches from all PNGs.
        
        Args:
            sample_limit: Maximum number of PNGs to process (for testing)
            start_date: Start date filter
            end_date: End date filter
        
        Returns:
            DataFrame with all patch metadata
        """
        # Find PNG files
        png_files = self.png_loader.find_png_files()
        
        logger.info(f"Found {len(png_files)} PNG files")
        
        if start_date or end_date:
            png_files = [
                (p, dt) for p, dt in png_files
                if (not start_date or dt >= start_date) and (not end_date or dt <= end_date)
            ]
            logger.info(f"After date filtering: {len(png_files)} PNGs")
        
        if sample_limit:
            png_files = png_files[:sample_limit]
            logger.info(f"Limited to {len(png_files)} PNGs")
        
        # Determine date range for split assignment
        if png_files:
            dates = [dt for _, dt in png_files]
            min_date = min(dates)
            max_date = max(dates)
            date_range = (max_date - min_date).total_seconds()
            
            train_end = min_date + timedelta(seconds=date_range * self.train_ratio)
            val_end = train_end + timedelta(seconds=date_range * self.val_ratio)
            
            logger.info(f"Date range: {min_date} to {max_date}")
            logger.info(f"Train cutoff: {train_end}")
            logger.info(f"Val cutoff: {val_end}")
        
        # Load all lightning records (this is expensive but done once)
        logger.info("Loading all lightning records...")
        all_lightning_df = self.csv_parser.load_all_lightning()
        logger.info(f"Loaded {len(all_lightning_df)} lightning records")
        
        # Extract patches
        all_patches = []
        stats = {'total_pngs': 0, 'processed_pngs': 0, 'total_patches': 0}
        
        for i, (png_path, png_dt) in enumerate(png_files):
            if i % max(1, len(png_files) // 10) == 0:
                logger.info(f"Progress: {i}/{len(png_files)}")
            
            stats['total_pngs'] += 1
            
            try:
                # Load PNG
                png_array = self.png_loader.load_png(png_path)
                
                # Filter lightning records for this PNG's date
                png_date_start = pd.Timestamp(png_dt).normalize()
                png_date_end = png_date_start + pd.Timedelta(days=1)
                
                time_mask = (all_lightning_df['timestamp'] >= png_date_start) & \
                            (all_lightning_df['timestamp'] < png_date_end)
                daily_lightning_df = all_lightning_df[time_mask]
                
                # Assign split based on date
                if png_dt < train_end:
                    split = 'train'
                elif png_dt < val_end:
                    split = 'val'
                else:
                    split = 'test'
                
                # Extract patches
                result = self.patch_extractor.process_png_for_dataset(
                    png_array, png_dt, daily_lightning_df, split=split, n_negative_per_positive=1
                )
                
                # Collect patches
                all_patches.extend(result['patches'])
                stats['total_patches'] += len(result['patches'])
                stats['processed_pngs'] += 1
                
            except Exception as e:
                logger.warning(f"Error processing {png_path}: {e}")
                continue
        
        # Create DataFrame
        if all_patches:
            df = pd.DataFrame(all_patches)
            df = df.sort_values('split').reset_index(drop=True)
            
            logger.info(f"\nDataset Statistics:")
            logger.info(f"  Total PNGs processed: {stats['processed_pngs']}/{stats['total_pngs']}")
            logger.info(f"  Total patches: {len(df)}")
            logger.info(f"  Positive: {(df['label'] == 1).sum()}")
            logger.info(f"  Negative: {(df['label'] == 0).sum()}")
            logger.info(f"\nSplit distribution:")
            logger.info(f"  Train: {(df['split'] == 'train').sum()}")
            logger.info(f"  Val: {(df['split'] == 'val').sum()}")
            logger.info(f"  Test: {(df['split'] == 'test').sum()}")
            
            return df
        else:
            logger.warning("No patches extracted")
            return pd.DataFrame()
    
    def save_dataset_index(self, df: pd.DataFrame, output_path: Optional[Path] = None) -> Path:
        """
        Save dataset index to CSV.
        
        Args:
            df: DataFrame with patch metadata
            output_path: Path to save CSV (default: data/processed/satellite_dataset.csv)
        
        Returns:
            Path to saved file
        """
        if output_path is None:
            output_path = self.output_dir / 'satellite_dataset.csv'
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        df.to_csv(output_path, index=False)
        
        logger.info(f"Saved dataset index to {output_path}")
        
        return output_path


# Example usage
if __name__ == '__main__':
    from himawari_png_loader import HimawariPNGLoader
    from lightning_csv_parser import LightningCSVParser
    from satellite_patch_extractor import SatellitePatchExtractor
    
    # Initialize components
    png_loader = HimawariPNGLoader('data/raw/himawari8_pngs')
    csv_parser = LightningCSVParser('data/raw/himawari8_pngs')
    patch_extractor = SatellitePatchExtractor(png_loader, 'data/processed/patches')
    
    # Initialize builder
    builder = SatelliteDatasetBuilder(png_loader, csv_parser, patch_extractor)
    
    # Build dataset (with limit for testing)
    print("Building satellite patch dataset...")
    print("(Using sample_limit=2 for quick test)")
    df = builder.build_dataset(sample_limit=2)
    
    if len(df) > 0:
        print(f"\nSample records:")
        print(df.head(10))
        
        # Save index
        output_path = builder.save_dataset_index(df)
        print(f"\nDataset index saved to: {output_path}")
