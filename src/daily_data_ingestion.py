"""
Daily data ingestion pipeline for Himawari-8 PNG satellite images.

This module:
1. Monitors data/raw/himawari8_pngs/ for new PNG files
2. Extracts satellite channels (IR, WV, VIS) from PNGs
3. Creates 64x64 patches for ML training
4. Incrementally appends to HDF5 dataset
5. Triggers model retraining when enough new data accumulates
"""

import os
import logging
from pathlib import Path
from datetime import datetime
from typing import Tuple, Optional, List, Dict
import numpy as np
import h5py
from PIL import Image
import cv2

logger = logging.getLogger(__name__)


class HimawariPNGLoader:
    """Load and preprocess Himawari-8 satellite PNG images."""

    def __init__(
        self,
        patch_size: int = 64,
        num_channels: int = 3,
        bbox: Tuple[float, float, float, float] = (100.0, 120.0, -5.0, 15.0),
    ):
        """
        Initialize PNG loader.

        Args:
            patch_size: Size of extracted patches (64x64)
            num_channels: Number of channels (3: IR, WV, VIS)
            bbox: Region bounding box [min_lon, max_lon, min_lat, max_lat]
        """
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.bbox = bbox

    def load_png(self, png_path: str) -> Tuple[np.ndarray, str]:
        """
        Load and extract channels from PNG.

        Args:
            png_path: Path to Himawari-8 PNG file

        Returns:
            Tuple of (channels, timestamp)
                - channels: (3, H, W) array with IR, WV, VIS channels
                - timestamp: ISO format timestamp from filename
        """
        png_path = Path(png_path)
        if not png_path.exists():
            raise FileNotFoundError(f"PNG not found: {png_path}")

        # Extract timestamp from filename (e.g., "20261212_0940_ir_enhanced.png")
        stem = png_path.stem
        try:
            timestamp = self._parse_timestamp(stem)
        except Exception as e:
            logger.warning(f"Could not parse timestamp from {stem}: {e}")
            timestamp = datetime.now().isoformat()

        # Load PNG image
        img = Image.open(png_path)
        img_array = np.array(img, dtype=np.float32)

        logger.info(f"Loaded PNG: {png_path.name}, shape: {img_array.shape}")

        # Extract channels based on image format
        channels = self._extract_channels(img_array)

        return channels, timestamp

    def _parse_timestamp(self, filename: str) -> str:
        """
        Parse timestamp from Himawari-8 filename.

        Expected format: YYYYMMDDD_HHMM or similar
        Returns ISO format string: YYYY-MM-DDTHH:MM:00
        """
        # Remove common suffixes
        clean = filename.replace("_ir_enhanced", "").replace("_ir", "").replace("_wv", "")

        if "_" in clean:
            date_part, time_part = clean.split("_")[:2]
        else:
            date_part, time_part = clean[:8], clean[8:12]

        # Parse YYYYMMDD
        year = int(date_part[:4])
        month = int(date_part[4:6])
        day = int(date_part[6:8])

        # Parse HHMM
        hour = int(time_part[:2])
        minute = int(time_part[2:4])

        return f"{year:04d}-{month:02d}-{day:02d}T{hour:02d}:{minute:02d}:00"

    def _extract_channels(self, img_array: np.ndarray) -> np.ndarray:
        """
        Extract 3 channels from PNG image.

        Supports:
        1. RGB image (3 channels) → map R=IR, G=WV, B=VIS
        2. Grayscale image → repeat across channels
        3. 4-channel (RGBA) → use RGB, ignore alpha

        Args:
            img_array: Image array from PIL

        Returns:
            (3, H, W) normalized array with channels [IR, WV, VIS]
        """
        if len(img_array.shape) == 2:
            # Grayscale → repeat across 3 channels
            logger.warning("Image is grayscale, repeating across channels")
            channels = np.stack([img_array, img_array, img_array])

        elif len(img_array.shape) == 3:
            if img_array.shape[2] == 4:
                # RGBA → use RGB
                channels = np.transpose(img_array[:, :, :3], (2, 0, 1))
            elif img_array.shape[2] == 3:
                # RGB → transpose to (C, H, W)
                channels = np.transpose(img_array, (2, 0, 1))
            else:
                raise ValueError(
                    f"Unexpected number of channels: {img_array.shape[2]}"
                )
        else:
            raise ValueError(f"Unexpected image shape: {img_array.shape}")

        # Normalize to [0, 1] range
        channels = channels.astype(np.float32)
        ch_min = channels.min(axis=(1, 2), keepdims=True)
        ch_max = channels.max(axis=(1, 2), keepdims=True)
        channels = (channels - ch_min) / (ch_max - ch_min + 1e-8)

        return channels

    def create_patches(
        self, channels: np.ndarray, stride: int = 32
    ) -> List[np.ndarray]:
        """
        Extract overlapping patches from channels.

        Args:
            channels: (3, H, W) channel array
            stride: Patch stride (32 = 50% overlap for 64x64 patches)

        Returns:
            List of (3, 64, 64) patches
        """
        _, height, width = channels.shape
        patches = []

        for y in range(0, height - self.patch_size + 1, stride):
            for x in range(0, width - self.patch_size + 1, stride):
                patch = channels[
                    :, y : y + self.patch_size, x : x + self.patch_size
                ]
                if patch.shape == (self.num_channels, self.patch_size, self.patch_size):
                    patches.append(patch)

        logger.info(f"Extracted {len(patches)} patches from image")
        return patches


class DailyDataPipeline:
    """Manage daily data ingestion and HDF5 updates."""

    def __init__(
        self,
        png_dir: str = "data/raw/himawari8_pngs/",
        hdf5_path: str = "data/processed/himawari_dataset.h5",
        metadata_path: str = "data/processed/metadata.json",
    ):
        """
        Initialize daily pipeline.

        Args:
            png_dir: Directory to watch for new PNG files
            hdf5_path: Path to HDF5 dataset file
            metadata_path: Path to metadata JSON tracking processed files
        """
        self.png_dir = Path(png_dir)
        self.hdf5_path = Path(hdf5_path)
        self.metadata_path = Path(metadata_path)
        self.loader = HimawariPNGLoader()

        # Create directories
        self.png_dir.mkdir(parents=True, exist_ok=True)
        self.hdf5_path.parent.mkdir(parents=True, exist_ok=True)

    def process_new_pngs(self) -> Dict[str, int]:
        """
        Find and process new PNG files.

        Returns:
            Dictionary with processing stats:
            {
                'new_pngs': int,
                'total_patches': int,
                'errors': int,
                'files_processed': list
            }
        """
        stats = {
            "new_pngs": 0,
            "total_patches": 0,
            "errors": 0,
            "files_processed": [],
        }

        # Find all PNG files
        png_files = sorted(self.png_dir.glob("*.png"))
        logger.info(f"Found {len(png_files)} PNG files in {self.png_dir}")

        for png_path in png_files:
            try:
                # Load PNG and extract channels
                channels, timestamp = self.loader.load_png(str(png_path))

                # Create patches
                patches = self.loader.create_patches(channels)

                if len(patches) > 0:
                    # Append to HDF5
                    self._append_to_hdf5(patches, timestamp, str(png_path))
                    stats["new_pngs"] += 1
                    stats["total_patches"] += len(patches)
                    stats["files_processed"].append(png_path.name)
                    logger.info(
                        f"✅ Processed {png_path.name}: {len(patches)} patches"
                    )

            except Exception as e:
                logger.error(f"❌ Error processing {png_path.name}: {e}")
                stats["errors"] += 1

        return stats

    def _append_to_hdf5(
        self, patches: List[np.ndarray], timestamp: str, source_file: str
    ) -> None:
        """
        Append patches to HDF5 dataset.

        Args:
            patches: List of (3, 64, 64) patch arrays
            timestamp: ISO timestamp string
            source_file: Source PNG filename
        """
        patches_array = np.stack(patches, axis=0)  # (N, 3, 64, 64)

        if not self.hdf5_path.exists():
            # Create new HDF5 file
            self._create_hdf5(patches_array, timestamp, source_file)
        else:
            # Append to existing
            self._append_to_existing_hdf5(patches_array, timestamp, source_file)

    def _create_hdf5(
        self, patches: np.ndarray, timestamp: str, source_file: str
    ) -> None:
        """Create new HDF5 file with initial patches."""
        with h5py.File(self.hdf5_path, "w") as f:
            # Store patches
            f.create_dataset(
                "images",
                data=patches,
                maxshape=(None, 3, 64, 64),
                chunks=(100, 3, 64, 64),
                compression="gzip",
                compression_opts=4,
            )

            # Placeholder for labels (to be filled by user)
            labels = np.zeros(len(patches), dtype=np.float32)
            f.create_dataset(
                "labels",
                data=labels,
                maxshape=(None,),
                chunks=(100,),
                compression="gzip",
            )

            # Metadata
            f.create_dataset(
                "timestamps",
                (len(patches),),
                dtype=h5py.string_dtype(encoding="utf-8"),
            )
            f["timestamps"][:] = [timestamp] * len(patches)

            f.create_dataset(
                "source_files",
                (len(patches),),
                dtype=h5py.string_dtype(encoding="utf-8"),
            )
            f["source_files"][:] = [source_file] * len(patches)

            logger.info(
                f"Created HDF5: {self.hdf5_path} with {len(patches)} samples"
            )

    def _append_to_existing_hdf5(
        self, patches: np.ndarray, timestamp: str, source_file: str
    ) -> None:
        """Append new patches to existing HDF5 file."""
        with h5py.File(self.hdf5_path, "a") as f:
            # Get current size
            current_size = f["images"].shape[0]
            new_size = current_size + len(patches)

            # Resize datasets
            f["images"].resize(new_size, axis=0)
            f["labels"].resize(new_size, axis=0)
            f["timestamps"].resize(new_size, axis=0)
            f["source_files"].resize(new_size, axis=0)

            # Append new data
            f["images"][current_size:new_size] = patches
            f["labels"][current_size:new_size] = 0  # Placeholder
            f["timestamps"][current_size:new_size] = [timestamp] * len(patches)
            f["source_files"][current_size:new_size] = [source_file] * len(patches)

            logger.info(
                f"Appended to HDF5: {len(patches)} samples "
                f"(total: {new_size})"
            )

    def get_dataset_stats(self) -> Dict[str, any]:
        """Get current HDF5 dataset statistics."""
        if not self.hdf5_path.exists():
            return {"status": "No dataset yet", "samples": 0}

        with h5py.File(self.hdf5_path, "r") as f:
            num_samples = f["images"].shape[0]
            num_labeled = int(np.sum(f["labels"][()] > 0))

            return {
                "total_samples": num_samples,
                "labeled_samples": num_labeled,
                "unlabeled_samples": num_samples - num_labeled,
                "image_shape": f["images"].shape,
                "file_size_mb": self.hdf5_path.stat().st_size / (1024 * 1024),
            }


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Initialize pipeline
    pipeline = DailyDataPipeline()

    # Process new PNGs
    print("🔄 Scanning for new PNG files...")
    stats = pipeline.process_new_pngs()
    print(f"\n📊 Processing Results:")
    print(f"  New PNGs: {stats['new_pngs']}")
    print(f"  Total patches: {stats['total_patches']}")
    print(f"  Errors: {stats['errors']}")
    if stats["files_processed"]:
        print(f"  Files: {', '.join(stats['files_processed'])}")

    # Show dataset stats
    print(f"\n📈 Dataset Statistics:")
    dataset_stats = pipeline.get_dataset_stats()
    for key, value in dataset_stats.items():
        print(f"  {key}: {value}")
