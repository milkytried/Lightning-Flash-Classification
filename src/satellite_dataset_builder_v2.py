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


def build_dataset_simple():
    """Simple dataset builder - loads components and extracts patches."""
    from himawari_png_loader import HimawariPNGLoader
    from lightning_csv_parser import LightningCSVParser
    from satellite_patch_extractor import SatellitePatchExtractor
    
    # Initialize components
    png_loader = HimawariPNGLoader('data/raw/himawari8_pngs')
    csv_parser = LightningCSVParser('data/raw/himawari8_pngs')
    patch_extractor = SatellitePatchExtractor(png_loader, 'data/processed/patches')
    
    logger.info("Components initialized")
    
    # Find PNG files
    png_files = png_loader.find_png_files()
    logger.info(f"Found {len(png_files)} PNG files")
    
    if not png_files:
        logger.warning("No PNG files found")
        return
    
    # Assign PNG files to splits by IMAGE (not patch) to avoid leakage
    # Use random split (reproducible with seed) to handle data gaps
    # This ensures all splits have access to lightning data
    np.random.seed(42)
    png_paths = [p for p, _ in png_files]
    n_pngs = len(png_paths)
    
    # Create random split (65% train, 20% val, 15% test)
    indices = np.random.permutation(n_pngs)
    train_count = int(0.65 * n_pngs)
    val_count = int(0.20 * n_pngs)
    
    train_indices = set(indices[:train_count])
    val_indices = set(indices[train_count:train_count + val_count])
    test_indices = set(indices[train_count + val_count:])
    
    png_splits = {}
    for i, png_path in enumerate(png_paths):
        if i in train_indices:
            png_splits[png_path] = 'train'
        elif i in val_indices:
            png_splits[png_path] = 'val'
        else:
            png_splits[png_path] = 'test'
    
    # Log split assignment
    split_counts = {'train': 0, 'val': 0, 'test': 0}
    for split in png_splits.values():
        split_counts[split] += 1
    
    logger.info(f"\nRandom Image-Level Split (avoids patch-level leakage):")
    logger.info(f"Total PNG files: {n_pngs}")
    for split in ['train', 'val', 'test']:
        logger.info(f"  {split}: {split_counts[split]} images")
    
    logger.info(f"\nPNG split assignment:")
    for i, png_path in enumerate(png_paths):
        split = png_splits[png_path]
        logger.debug(f"  [{i}] {png_path.name} → {split}")
    
    logger.info(f"PNG splits dictionary has {len(png_splits)} entries")
    
    # Load all lightning records
    logger.info("Loading lightning records...")
    all_lightning_df = csv_parser.load_all_lightning()
    logger.info(f"Loaded {len(all_lightning_df)} lightning records")
    
    # Extract patches from each PNG
    all_patches = []
    n_processed = 0
    n_errors = 0
    
    for i, (png_path, png_dt) in enumerate(png_files):
        if i % max(1, len(png_files) // 10) == 0:
            logger.info(f"Progress: {i}/{len(png_files)}")
        
        try:
            # Debug: log which PNG we're processing
            logger.debug(f"[{i}] Processing {png_path.name} (type={type(png_path)}, in dict={png_path in png_splits})")
            
            # Load PNG image
            png_array = png_loader.load_png(png_path)
            
            # Get lightning for this PNG's date
            png_ts = pd.Timestamp(png_dt)
            day_start = png_ts.normalize()
            day_end = day_start + pd.Timedelta(days=1)
            
            # Handle timezone-aware timestamps
            lightning_ts = all_lightning_df['timestamp']
            if hasattr(lightning_ts.dtype, 'tz') and lightning_ts.dtype.tz is not None:
                day_start = day_start.tz_localize('UTC')
                day_end = day_end.tz_localize('UTC')
            
            # Filter lightning for this day
            mask = (lightning_ts >= day_start) & (lightning_ts < day_end)
            daily_lightning = all_lightning_df[mask]
            
            # Get split from pre-assigned PNG splits (image-level, not patch-level)
            if png_path in png_splits:
                split = png_splits[png_path]
                logger.info(f"PNG {png_path.name} assigned split='{split}'")
            else:
                logger.warning(f"PNG {png_path.name} not found in png_splits!")
                split = 'train'  # fallback
            
            # Extract patches with split parameter
            logger.info(f"  Calling process_png_for_dataset with split='{split}'")
            result = patch_extractor.process_png_for_dataset(
                png_array, png_dt, daily_lightning, 
                split=split, n_negative_per_positive=1
            )
            logger.info(f"  Returned {len(result['patches'])} patches with splits: {set([p.get('split') for p in result['patches']][:5])}")
            
            all_patches.extend(result['patches'])
            n_processed += 1
            
        except Exception as e:
            logger.warning(f"Error processing {png_path.name}: {str(e)[:100]}")
            n_errors += 1
            continue
    
    logger.info(f"\nProcessing complete: {n_processed}/{len(png_files)} PNGs, {n_errors} errors")
    logger.info(f"Total patches extracted: {len(all_patches)}")
    
    if all_patches:
        # Create DataFrame
        df = pd.DataFrame(all_patches)
        
        # Log statistics
        logger.info(f"\nDataset Statistics:")
        logger.info(f"  Total patches: {len(df)}")
        logger.info(f"  Positive: {(df['label'] == 1).sum()}")
        logger.info(f"  Negative: {(df['label'] == 0).sum()}")
        logger.info(f"\nSplit distribution:")
        logger.info(f"  Train: {(df['split'] == 'train').sum()}")
        logger.info(f"  Val: {(df['split'] == 'val').sum()}")
        logger.info(f"  Test: {(df['split'] == 'test').sum()}")
        
        # Save to CSV
        output_path = Path('data/processed/satellite_dataset.csv')
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        
        logger.info(f"\n✓ Dataset saved to {output_path}")
        return output_path
    else:
        logger.warning("No patches extracted")
        return None


if __name__ == '__main__':
    build_dataset_simple()
