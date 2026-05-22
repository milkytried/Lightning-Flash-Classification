"""
Himawari-8 PNG image loader with geographic coordinate mapping.

Loads satellite images from disk and provides coordinate transformation
between lat/lon and pixel coordinates.
"""

import os
import numpy as np
from PIL import Image
from pathlib import Path
from datetime import datetime
from typing import Tuple, Optional, Dict, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HimawariPNGLoader:
    """Load Himawari-8 PNG files with geographic metadata and coordinate mapping."""
    
    def __init__(self, png_dir: str, config: Optional[Dict] = None):
        """
        Initialize PNG loader with directory and geographic config.
        
        Args:
            png_dir (str): Root directory containing PNG files
            config (dict): Geographic configuration with keys:
                - png_lat_min (float): Minimum latitude
                - png_lat_max (float): Maximum latitude  
                - png_lon_min (float): Minimum longitude
                - png_lon_max (float): Maximum longitude
                - png_width (int): Image width in pixels (default 950)
                - png_height (int): Image height in pixels (default 800)
                
                If None, uses Malaysia region defaults.
        """
        self.png_dir = Path(png_dir)
        
        # Default config for Malaysia region (empirically determined from data)
        self.config = {
            'png_lat_min': 1.0,
            'png_lat_max': 6.5,
            'png_lon_min': 99.5,
            'png_lon_max': 120.0,
            'png_width': 950,
            'png_height': 800
        }
        
        # Override with user config if provided
        if config:
            self.config.update(config)
        
        logger.info(f"HimawariPNGLoader initialized")
        logger.info(f"  Directory: {self.png_dir}")
        logger.info(f"  Lat bounds: [{self.config['png_lat_min']}, {self.config['png_lat_max']}]")
        logger.info(f"  Lon bounds: [{self.config['png_lon_min']}, {self.config['png_lon_max']}]")
        logger.info(f"  Image size: {self.config['png_width']}x{self.config['png_height']}")
    
    def find_png_files(self) -> List[Tuple[Path, Optional[datetime]]]:
        """
        Recursively find all PNG files in directory.
        
        Attempts to extract datetime from filename or creation date.
        
        Returns:
            List of (path, datetime) tuples, sorted by datetime
        """
        png_files = []
        
        for root, dirs, files in os.walk(self.png_dir):
            for file in files:
                if file.lower().endswith('.png'):
                    filepath = Path(root) / file
                    
                    # Try to extract datetime from filename
                    dt = self._parse_filename_datetime(file)
                    
                    # Fallback to file modification time
                    if dt is None:
                        stat_info = filepath.stat()
                        dt = datetime.fromtimestamp(stat_info.st_mtime)
                    
                    png_files.append((filepath, dt))
        
        # Sort by datetime
        png_files.sort(key=lambda x: x[1] if x[1] else datetime.min)
        
        logger.info(f"Found {len(png_files)} PNG files")
        if png_files:
            logger.info(f"  First: {png_files[0][0].name} ({png_files[0][1]})")
            logger.info(f"  Last:  {png_files[-1][0].name} ({png_files[-1][1]})")
        
        return png_files
    
    def _parse_filename_datetime(self, filename: str) -> Optional[datetime]:
        """
        Extract datetime from PNG filename.
        
        Supports formats like:
        - 12_May_Himawari.png → 2026-05-12
        - 22_May_Himawari.png → 2026-05-22
        
        Args:
            filename (str): PNG filename
        
        Returns:
            datetime object or None if parsing fails
        """
        # Pattern: DD_Mon_Himawari.png
        try:
            parts = filename.replace('_Himawari.png', '').split('_')
            if len(parts) >= 2:
                day = int(parts[0])
                month_str = parts[1]
                
                # Map month names
                months = {
                    'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
                    'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12
                }
                
                if month_str in months:
                    month = months[month_str]
                    # Assume current year (2026)
                    year = 2026
                    return datetime(year, month, day)
        except:
            pass
        
        return None
    
    def load_png(self, filepath) -> np.ndarray:
        """
        Load PNG image as numpy array.
        
        Args:
            filepath: Path to PNG file
        
        Returns:
            numpy array (H, W, 3) with dtype uint8, RGB channels
        """
        img = Image.open(filepath)
        
        # Ensure RGB
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        array = np.array(img)
        
        logger.debug(f"Loaded PNG: {filepath.name}, shape={array.shape}")
        
        return array
    
    def latlon_to_pixel(self, lat: float, lon: float) -> Tuple[int, int]:
        """
        Convert lat/lon coordinates to pixel coordinates (x, y).
        
        Assumes linear mapping from geographic bounds to image dimensions.
        
        Args:
            lat (float): Latitude in degrees N
            lon (float): Longitude in degrees E
        
        Returns:
            (x, y) tuple with pixel coordinates (0-indexed)
        """
        # Normalize to [0, 1]
        x_norm = (lon - self.config['png_lon_min']) / (self.config['png_lon_max'] - self.config['png_lon_min'])
        y_norm = (self.config['png_lat_max'] - lat) / (self.config['png_lat_max'] - self.config['png_lat_min'])
        
        # Convert to pixel coordinates
        x_pixel = int(np.clip(x_norm * self.config['png_width'], 0, self.config['png_width'] - 1))
        y_pixel = int(np.clip(y_norm * self.config['png_height'], 0, self.config['png_height'] - 1))
        
        return x_pixel, y_pixel
    
    def pixel_to_latlon(self, x: int, y: int) -> Tuple[float, float]:
        """
        Convert pixel coordinates to lat/lon.
        
        Inverse of latlon_to_pixel.
        
        Args:
            x (int): Pixel x coordinate
            y (int): Pixel y coordinate
        
        Returns:
            (lat, lon) tuple
        """
        # Normalize to [0, 1]
        x_norm = x / self.config['png_width']
        y_norm = y / self.config['png_height']
        
        # Convert to lat/lon
        lon = self.config['png_lon_min'] + x_norm * (self.config['png_lon_max'] - self.config['png_lon_min'])
        lat = self.config['png_lat_max'] - y_norm * (self.config['png_lat_max'] - self.config['png_lat_min'])
        
        return lat, lon
    
    def validate_coordinates(self, lat: float, lon: float) -> bool:
        """
        Check if coordinates are within PNG bounds.
        
        Args:
            lat (float): Latitude
            lon (float): Longitude
        
        Returns:
            True if within bounds, False otherwise
        """
        lat_ok = self.config['png_lat_min'] <= lat <= self.config['png_lat_max']
        lon_ok = self.config['png_lon_min'] <= lon <= self.config['png_lon_max']
        return lat_ok and lon_ok


# Example usage
if __name__ == '__main__':
    loader = HimawariPNGLoader('data/raw/himawari8_pngs')
    
    # Find PNG files
    pngs = loader.find_png_files()
    
    if pngs:
        # Load first PNG
        png_path, png_dt = pngs[0]
        image = loader.load_png(png_path)
        print(f"Loaded: {png_path.name}, shape={image.shape}, dtype={image.dtype}")
        
        # Test coordinate mapping
        test_lat, test_lon = 3.5, 101.5  # Kuala Lumpur area
        x, y = loader.latlon_to_pixel(test_lat, test_lon)
        print(f"Lat/Lon ({test_lat}, {test_lon}) → Pixel ({x}, {y})")
        
        # Inverse mapping
        lat_back, lon_back = loader.pixel_to_latlon(x, y)
        print(f"Pixel ({x}, {y}) → Lat/Lon ({lat_back:.2f}, {lon_back:.2f})")
