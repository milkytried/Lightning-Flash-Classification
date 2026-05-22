"""
Malaysian Meteorological Department (MMD) lightning CSV parser.

Reads lightning strike records from CSV files and provides organized
access to timestamp, location, and amplitude data.
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LightningCSVParser:
    """Parse and load MMD lightning CSV files."""
    
    def __init__(self, csv_root_dir: str = 'data/raw/himawari8_pngs'):
        """
        Initialize CSV parser.
        
        Args:
            csv_root_dir (str): Root directory containing nested CSV files
        """
        self.csv_root_dir = Path(csv_root_dir)
        self.csv_files = []
        
        # Find all CSV files
        self._find_csvs()
        
        logger.info(f"LightningCSVParser initialized")
        logger.info(f"  Root: {self.csv_root_dir}")
        logger.info(f"  Found {len(self.csv_files)} CSV files")
    
    def _find_csvs(self):
        """Find all CSV files recursively."""
        csv_list = []
        
        for root, dirs, files in os.walk(self.csv_root_dir):
            for file in files:
                if file.lower().endswith('.csv'):
                    # Skip non-lightning CSVs
                    if any(x in file.lower() for x in ['raw data', 'raw_data', 'lightning']):
                        filepath = Path(root) / file
                        csv_list.append(filepath)
        
        self.csv_files = sorted(csv_list)
        
        if self.csv_files:
            logger.info(f"First CSV:  {self.csv_files[0]}")
            logger.info(f"Last CSV:   {self.csv_files[-1]}")
    
    def parse_csv(self, filepath: Path) -> Optional[pd.DataFrame]:
        """
        Parse single CSV file.
        
        Expected columns: Date/Time, Latitude, Longitude, Amplitude, Cloud or Ground, ...
        
        Args:
            filepath (Path): Path to CSV file
        
        Returns:
            DataFrame with columns:
            - timestamp (datetime)
            - latitude (float)
            - longitude (float)
            - amplitude (float)
            - strike_type (str)
            
            Returns None if parsing fails
        """
        try:
            df = pd.read_csv(filepath)
            
            # Standardize column names (strip whitespace)
            df.columns = df.columns.str.strip()
            
            logger.debug(f"Parsing: {filepath.name}, shape={df.shape}")
            logger.debug(f"  Columns: {df.columns.tolist()[:5]}...")
            
            # Extract required columns
            required_cols = {
                'timestamp': ['Date/Time', 'DateTime', 'Datetime'],
                'latitude': ['Latitude', 'Lat'],
                'longitude': ['Longitude', 'Lon', 'Longitude '],
                'amplitude': ['Amplitude', 'Amp'],
                'strike_type': ['Cloud or Ground', 'Strike Type', 'Type']
            }
            
            result_dict = {}
            
            # Map columns
            for key, options in required_cols.items():
                found = False
                for col_option in options:
                    if col_option in df.columns:
                        result_dict[key] = df[col_option].copy()
                        found = True
                        break
                
                if not found:
                    logger.warning(f"Column {key} not found in {filepath.name}")
                    return None
            
            # Parse timestamp
            try:
                result_dict['timestamp'] = pd.to_datetime(result_dict['timestamp'])
            except Exception as e:
                logger.warning(f"Failed to parse timestamp: {e}")
                return None
            
            # Convert numeric columns
            result_dict['latitude'] = pd.to_numeric(result_dict['latitude'], errors='coerce')
            result_dict['longitude'] = pd.to_numeric(result_dict['longitude'], errors='coerce')
            result_dict['amplitude'] = pd.to_numeric(result_dict['amplitude'], errors='coerce')
            
            # Clean strike type (remove extra spaces)
            result_dict['strike_type'] = result_dict['strike_type'].astype(str).str.strip()
            
            result_df = pd.DataFrame(result_dict)
            
            # Remove rows with NaN in critical columns
            result_df = result_df.dropna(subset=['timestamp', 'latitude', 'longitude'])
            
            logger.debug(f"  Parsed: {len(result_df)} records")
            
            return result_df
        
        except Exception as e:
            logger.error(f"Error parsing {filepath}: {e}")
            return None
    
    def load_all_lightning(self, start_date: Optional[datetime] = None, 
                          end_date: Optional[datetime] = None) -> pd.DataFrame:
        """
        Load all lightning records from all CSV files.
        
        Args:
            start_date (datetime, optional): Filter records after this date
            end_date (datetime, optional): Filter records before this date
        
        Returns:
            Combined DataFrame sorted by timestamp
        """
        all_dfs = []
        
        logger.info(f"Loading {len(self.csv_files)} CSV files...")
        
        for i, csv_path in enumerate(self.csv_files):
            if i % 50 == 0:
                logger.info(f"  Progress: {i}/{len(self.csv_files)}")
            
            df = self.parse_csv(csv_path)
            if df is not None and len(df) > 0:
                all_dfs.append(df)
        
        if not all_dfs:
            logger.warning("No CSV files successfully parsed")
            return pd.DataFrame()
        
        # Combine all dataframes
        combined_df = pd.concat(all_dfs, ignore_index=True)
        combined_df = combined_df.sort_values('timestamp').reset_index(drop=True)
        
        logger.info(f"Combined: {len(combined_df)} total lightning records")
        
        # Filter by date if specified
        if start_date:
            combined_df = combined_df[combined_df['timestamp'] >= start_date]
            logger.info(f"After start_date filter: {len(combined_df)} records")
        
        if end_date:
            combined_df = combined_df[combined_df['timestamp'] <= end_date]
            logger.info(f"After end_date filter: {len(combined_df)} records")
        
        # Log statistics
        if len(combined_df) > 0:
            logger.info(f"Date range: {combined_df['timestamp'].min()} to {combined_df['timestamp'].max()}")
            logger.info(f"Lat range: {combined_df['latitude'].min():.2f} to {combined_df['latitude'].max():.2f}")
            logger.info(f"Lon range: {combined_df['longitude'].min():.2f} to {combined_df['longitude'].max():.2f}")
            logger.info(f"Strike types: {combined_df['strike_type'].value_counts().to_dict()}")
        
        return combined_df
    
    def load_lightning_by_date(self, target_date: datetime) -> pd.DataFrame:
        """
        Load lightning records for a specific date.
        
        Args:
            target_date (datetime): Target date
        
        Returns:
            DataFrame with records on that date
        """
        start = datetime.combine(target_date.date(), datetime.min.time())
        end = start + timedelta(days=1)
        
        all_df = self.load_all_lightning(start, end)
        
        logger.info(f"Loaded {len(all_df)} lightning records for {target_date.date()}")
        
        return all_df
    
    def get_lightning_in_window(self, png_datetime: datetime, 
                               window_minutes: int = 60) -> pd.DataFrame:
        """
        Get lightning records within a time window around PNG timestamp.
        
        Args:
            png_datetime (datetime): Center timestamp
            window_minutes (int): Time window in minutes (±window_minutes from center)
        
        Returns:
            DataFrame with lightning records in window
        """
        start = png_datetime - timedelta(minutes=window_minutes)
        end = png_datetime + timedelta(minutes=window_minutes)
        
        all_df = self.load_all_lightning(start, end)
        
        return all_df
    
    def get_lightning_stats(self) -> Dict:
        """
        Get statistics about lightning data.
        
        Returns:
            Dictionary with statistics
        """
        all_df = self.load_all_lightning()
        
        if len(all_df) == 0:
            return {}
        
        stats = {
            'total_records': len(all_df),
            'date_min': all_df['timestamp'].min(),
            'date_max': all_df['timestamp'].max(),
            'lat_min': all_df['latitude'].min(),
            'lat_max': all_df['latitude'].max(),
            'lon_min': all_df['longitude'].min(),
            'lon_max': all_df['longitude'].max(),
            'amp_min': all_df['amplitude'].min(),
            'amp_max': all_df['amplitude'].max(),
            'amp_mean': all_df['amplitude'].mean(),
            'strike_types': all_df['strike_type'].value_counts().to_dict(),
            'num_csv_files': len(self.csv_files)
        }
        
        return stats


# Example usage
if __name__ == '__main__':
    parser = LightningCSVParser('data/raw/himawari8_pngs')
    
    # Get statistics
    stats = parser.get_lightning_stats()
    print("\nLightning Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Load first day of data
    print("\n\nLoading first day...")
    first_day_df = parser.load_all_lightning(
        datetime(2023, 1, 1),
        datetime(2023, 1, 2)
    )
    if len(first_day_df) > 0:
        print(f"First few records:")
        print(first_day_df.head())
