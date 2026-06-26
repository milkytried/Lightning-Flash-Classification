"""
Ingest real Met Department Malaysia lightning data (2023-2026) into HDF5.

This script:
1. Reads all lightning CSV files from 4-year dataset
2. Parses lightning strikes with location & timestamp
3. Creates labeled HDF5 dataset (positive = lightning, negative = no lightning)
4. Generates 70/15/15 train/val/test splits
"""

import os
import pandas as pd
import numpy as np
import h5py
from pathlib import Path
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

rng = np.random.default_rng(42)


def scan_lightning_csvs(data_root: str = "data/raw/himawari8_pngs"):
    """Scan all lightning CSV files and extract strike information."""
    
    data_path = Path(data_root)
    csv_files = list(data_path.rglob("raw data all.csv"))
    
    logger.info(f"Found {len(csv_files)} lightning CSV files")
    
    all_strikes = []
    
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            
            # Extract year, month, day from path
            parts = str(csv_file).split(os.sep)
            year = parts[-5]  # e.g., "2023"
            month_day = parts[-3]  # e.g., "01 JAN"
            day_num = parts[-2]  # e.g., "1"
            
            # Add source metadata
            df['source_file'] = str(csv_file)
            df['year'] = year
            df['month_day'] = month_day
            
            all_strikes.append(df)
            
        except Exception as e:
            logger.warning(f"Error reading {csv_file}: {e}")
            continue
    
    # Combine all strikes
    strikes_df = pd.concat(all_strikes, ignore_index=True)
    logger.info(f"Total strikes loaded: {len(strikes_df)}")
    
    return strikes_df


def create_labeled_dataset(strikes_df, output_path: str = "data/processed/lightning_dataset.h5"):
    """
    Create HDF5 dataset with lightning labels.
    
    Strategy:
    - For each day with lightning strikes, label positive
    - For days without strikes, label negative
    - Create synthetic negative samples from regions without strikes
    """
    
    logger.info("Creating labeled dataset...")
    
    # Get unique days
    strikes_df = strikes_df.copy()
    strikes_df['date'] = pd.to_datetime(strikes_df['Date/Time']).dt.date
    unique_days = strikes_df['date'].unique()
    
    logger.info(f"Unique days with lightning: {len(unique_days)}")
    
    # For each day, create labels: 1 if lightning, 0 if no lightning
    day_labels = {}
    for day in unique_days:
        day_labels[str(day)] = 1
    
    # Create positive labels (days WITH lightning)
    positive_labels = []
    
    for day in unique_days:
        day_str = str(day)
        day_strikes = strikes_df[strikes_df['date'] == pd.to_datetime(day).date()]
        
        for _, strike in day_strikes.iterrows():
            positive_labels.append({
                'label': 1,
                'date': day_str,
                'latitude': strike['Latitude'],
                'longitude': strike['Longitude'],
                'amplitude': strike['Amplitude'],
                'strike_type': strike['Cloud or Ground'],
                'timestamp': strike['Date/Time']
            })
    
    logger.info(f"Positive samples (with lightning): {len(positive_labels)}")
    
    # Create negative samples from truly no-strike days and realistic metadata values.
    # This avoids the previous leakage pattern where negatives were encoded with an obvious
    # amplitude of 0 and strike_type = 'None'.
    days_range = pd.date_range('2023-01-01', '2026-03-31', freq='D')
    days_without_strikes = [str(d.date()) for d in days_range if str(d.date()) not in day_labels]
    
    logger.info(f"Days without lightning records: {len(days_without_strikes)}")
    
    negative_labels = []
    
    # Sample regions within Malaysia bounding box using a realistic spatial prior
    lat_min, lat_max = 0.85, 6.73  # Peninsular Malaysia
    lon_min, lon_max = 99.6, 104.4
    
    positive_latitudes = strikes_df['Latitude'].astype(float).dropna().to_numpy()
    positive_longitudes = strikes_df['Longitude'].astype(float).dropna().to_numpy()
    positive_amplitudes = strikes_df['Amplitude'].astype(float).dropna().to_numpy()
    positive_types = strikes_df['Cloud or Ground'].fillna('Cloud').astype(str)
    type_choices = ['Cloud', 'Ground']
    type_probs = positive_types.value_counts(normalize=True).reindex(type_choices, fill_value=0.5).to_numpy()
    type_probs = type_probs / type_probs.sum()
    
    # Generate negative samples: roughly 3x positive count
    num_negatives = min(len(positive_labels) * 3, len(days_without_strikes) * 10)
    
    for i in range(num_negatives):
        amp_sample = float(rng.normal(loc=0.0, scale=max(np.std(positive_amplitudes) / 3.0, 1.5)))
        amp_sample = float(np.clip(amp_sample, -20.0, 20.0))
        
        negative_labels.append({
            'label': 0,
            'date': days_without_strikes[i % len(days_without_strikes)],
            'latitude': float(rng.normal(loc=np.mean(positive_latitudes), scale=max(np.std(positive_latitudes), 0.5))),
            'longitude': float(rng.normal(loc=np.mean(positive_longitudes), scale=max(np.std(positive_longitudes), 0.5))),
            'amplitude': amp_sample,
            'strike_type': str(rng.choice(type_choices, p=type_probs)),
            'timestamp': 'N/A'
        })
    
    logger.info(
        "Negative sampling uses no-strike dates with Gaussian amplitude values and a realistic cloud/ground strike-type mix"
    )
    
    logger.info(f"Negative samples (no lightning): {len(negative_labels)}")
    
    # Combine all labels
    all_labels = positive_labels + negative_labels
    labels_df = pd.DataFrame(all_labels)
    
    # Shuffle
    labels_df = labels_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    logger.info(f"Total samples: {len(labels_df)}")
    logger.info(f"Positive: {(labels_df['label'] == 1).sum()}, Negative: {(labels_df['label'] == 0).sum()}")
    
    # Create HDF5 with metadata
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with h5py.File(str(output_path), 'w') as f:
        # Store labels
        f.create_dataset('labels', data=labels_df['label'].values, dtype=np.float32)
        
        # Store metadata
        f.create_dataset('dates', (len(labels_df),), dtype=h5py.string_dtype(encoding='utf-8'))
        f['dates'][:] = labels_df['date'].values
        
        f.create_dataset('latitudes', data=labels_df['latitude'].values, dtype=np.float32)
        f.create_dataset('longitudes', data=labels_df['longitude'].values, dtype=np.float32)
        f.create_dataset('amplitudes', data=labels_df['amplitude'].values, dtype=np.float32)
        
        f.create_dataset('strike_types', (len(labels_df),), dtype=h5py.string_dtype(encoding='utf-8'))
        f['strike_types'][:] = labels_df['strike_type'].values
        
        # Create train/val/test splits
        num_samples = len(labels_df)
        split_1 = int(0.7 * num_samples)
        split_2 = int(0.85 * num_samples)
        
        indices = np.arange(num_samples)
        
        f.create_dataset('train_indices', data=indices[:split_1], dtype=np.int32)
        f.create_dataset('val_indices', data=indices[split_1:split_2], dtype=np.int32)
        f.create_dataset('test_indices', data=indices[split_2:], dtype=np.int32)
        
        logger.info(f"Created HDF5: {output_path}")
        logger.info(f"  Train: {len(indices[:split_1])}, Val: {len(indices[split_1:split_2])}, Test: {len(indices[split_2:])}")
    
    return labels_df


def main():
    logger.info("=" * 60)
    logger.info("LIGHTNING DATASET INGESTION (2023-2026)")
    logger.info("=" * 60)
    
    # Scan all lightning CSVs
    strikes_df = scan_lightning_csvs("data/raw/himawari8_pngs")
    
    # Create labeled dataset
    labels_df = create_labeled_dataset(strikes_df, "data/processed/lightning_dataset.h5")
    
    logger.info("=" * 60)
    logger.info("✅ Ingestion complete!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
