"""
Daily data ingestion pipeline for Himawari-8 PNG satellite images.

This module:
1. Monitors data/raw/himawari8_pngs/ for new PNG files
2. Extracts satellite channels (IR, WV, VIS) from PNGs
3. Creates 64x64 patches for ML training
4. Incrementally appends to HDF5 dataset
5. Triggers model retraining when enough new data accumulates
"""

import logging
from pathlib import Path
from datetime import datetime
from typing import Tuple, Optional, List, Dict
import numpy as np
import h5py
from PIL import Image

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

        Supports common filename formats such as:
        - YYYYMMDD_HHMM
        - YYYYMMDDHHMM
        - DD_Mon_Himawari (falls back to 00:00, current year)

        Returns ISO format string: YYYY-MM-DDTHH:MM:00
        """
        # Remove common suffixes
        clean = (
            filename
            .replace("_ir_enhanced", "")
            .replace("_ir", "")
            .replace("_wv", "")
            .replace("_vis", "")
        )

        parts = clean.split("_")

        # Format: YYYYMMDD_HHMM
        if len(parts) >= 2 and len(parts[0]) == 8 and parts[0].isdigit() and len(parts[1]) == 4 and parts[1].isdigit():
            year = int(parts[0][:4])
            month = int(parts[0][4:6])
            day = int(parts[0][6:8])
            hour = int(parts[1][:2])
            minute = int(parts[1][2:4])
            return f"{year:04d}-{month:02d}-{day:02d}T{hour:02d}:{minute:02d}:00"

        # Format: YYYYMMDDHHMM
        if len(clean) >= 12 and clean[:12].isdigit():
            compact = clean[:12]
            year = int(compact[:4])
            month = int(compact[4:6])
            day = int(compact[6:8])
            hour = int(compact[8:10])
            minute = int(compact[10:12])
            return f"{year:04d}-{month:02d}-{day:02d}T{hour:02d}:{minute:02d}:00"

        # Format: DD_Mon_Himawari (time defaults to 00:00)
        if len(parts) >= 2 and parts[0].isdigit() and parts[1].isalpha():
            day = int(parts[0])
            month = datetime.strptime(parts[1][:3].title(), "%b").month

            year = datetime.now().year
            for part in parts:
                if len(part) == 4 and part.isdigit():
                    year = int(part)
                    break

            hour = 0
            minute = 0
            for part in parts:
                if len(part) == 4 and part.isdigit() and not (1900 <= int(part) <= 2100):
                    hour = int(part[:2])
                    minute = int(part[2:4])
                    break

            return f"{year:04d}-{month:02d}-{day:02d}T{hour:02d}:{minute:02d}:00"

        raise ValueError(f"Unsupported filename timestamp format: {filename}")

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
            "files_skipped_existing": 0,
        }

        # Find all PNG files
        png_files = sorted(self.png_dir.glob("*.png"))
        logger.info(f"Found {len(png_files)} PNG files in {self.png_dir}")

        processed_png_names = self._get_processed_png_names()

        for png_path in png_files:
            if png_path.name in processed_png_names:
                stats["files_skipped_existing"] += 1
                logger.info(f"Skipped already processed PNG: {png_path.name}")
                continue

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

    def _get_processed_png_names(self) -> set[str]:
        """Return PNG basenames that were already ingested into HDF5."""
        if not self.hdf5_path.exists():
            return set()

        try:
            with h5py.File(self.hdf5_path, "r") as f:
                if "source_files" not in f:
                    return set()

                names = set()
                for item in f["source_files"][:]:
                    if isinstance(item, bytes):
                        value = item.decode("utf-8", errors="ignore")
                    else:
                        value = str(item)
                    if value:
                        names.add(Path(value).name)

                return names
        except Exception as e:
            logger.warning(f"Could not read processed file list from HDF5: {e}")
            return set()

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
            self._ensure_resizable_string_dataset(f, "timestamps")
            self._ensure_resizable_string_dataset(f, "source_files")

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

    def _ensure_resizable_string_dataset(self, h5_file: h5py.File, name: str) -> None:
        """Ensure string metadata dataset is chunked and resizable for append operations."""
        if name not in h5_file:
            current_size = h5_file["images"].shape[0]
            h5_file.create_dataset(
                name,
                (current_size,),
                maxshape=(None,),
                chunks=(max(1, min(100, current_size)),),
                dtype=h5py.string_dtype(encoding="utf-8"),
            )
            h5_file[name][:] = [""] * current_size
            return

        dataset = h5_file[name]
        if dataset.maxshape and dataset.maxshape[0] is None and dataset.chunks is not None:
            return

        # Migrate fixed-size/non-chunked dataset to a resizable chunked layout.
        data = dataset[:]
        del h5_file[name]
        h5_file.create_dataset(
            name,
            data=data,
            maxshape=(None,),
            chunks=(max(1, min(100, len(data))),),
            dtype=h5py.string_dtype(encoding="utf-8"),
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
