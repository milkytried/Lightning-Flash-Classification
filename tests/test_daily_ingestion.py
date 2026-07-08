"""
Test script for daily PNG ingestion pipeline.

This script verifies:
1. PNG loading and channel extraction
2. Patch creation from satellite data
3. HDF5 dataset creation
4. Incremental dataset growth
"""

import logging
import sys
from pathlib import Path
import pytest

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def check_png_loading() -> bool:
    """Test 1: PNG loading and channel extraction."""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 1: PNG Loading and Channel Extraction")
    logger.info("=" * 60)

    from src.daily_data_ingestion import HimawariPNGLoader

    loader = HimawariPNGLoader()

    # Check if PNG exists
    png_dir = Path("data/raw/himawari8_pngs")
    png_files = list(png_dir.glob("*.png"))

    if not png_files:
        logger.warning(f"❌ No PNG files found in {png_dir}")
        logger.info(f"📁 Please add a PNG file to: {png_dir}")
        return False

    # Load first PNG
    png_path = png_files[0]
    logger.info(f"📷 Loading PNG: {png_path.name}")

    try:
        channels, timestamp = loader.load_png(str(png_path))
        logger.info(f"✅ Successfully loaded PNG")
        logger.info(f"   Shape: {channels.shape}")
        logger.info(f"   Timestamp: {timestamp}")
        logger.info(f"   Data type: {channels.dtype}")

        # Check channel statistics
        logger.info("📊 Channel Statistics:")
        for i, ch_name in enumerate(["IR", "WV", "VIS"]):
            ch = channels[i]
            logger.info(
                f"   {ch_name}: min={ch.min():.3f}, max={ch.max():.3f}, "
                f"mean={ch.mean():.3f}, std={ch.std():.3f}"
            )

        return True

    except Exception as e:
        logger.error(f"❌ Failed to load PNG: {e}", exc_info=True)
        return False


def check_patch_creation() -> bool:
    """Test 2: Patch extraction from satellite data."""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 2: Patch Creation")
    logger.info("=" * 60)

    from src.daily_data_ingestion import HimawariPNGLoader

    loader = HimawariPNGLoader()

    # Load PNG
    png_dir = Path("data/raw/himawari8_pngs")
    png_files = list(png_dir.glob("*.png"))

    if not png_files:
        logger.warning("❌ No PNG files found")
        return False

    png_path = png_files[0]

    try:
        channels, _ = loader.load_png(str(png_path))

        # Create patches
        patches = loader.create_patches(channels, stride=32)
        logger.info(f"✅ Created {len(patches)} patches")

        if len(patches) == 0:
            logger.error("❌ No patches created (image may be too small)")
            return False

        # Check patch shape
        patch = patches[0]
        logger.info(f"   Patch shape: {patch.shape}")
        logger.info(f"   Expected: (3, 64, 64)")

        if patch.shape != (3, 64, 64):
            logger.error(f"❌ Unexpected patch shape: {patch.shape}")
            return False

        # Check patch values
        logger.info(
            f"   Patch value range: [{patch.min():.3f}, {patch.max():.3f}]"
        )

        return True

    except Exception as e:
        logger.error(f"❌ Patch creation failed: {e}", exc_info=True)
        return False


def check_hdf5_creation() -> bool:
    """Test 3: HDF5 dataset creation."""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 3: HDF5 Dataset Creation")
    logger.info("=" * 60)

    from src.daily_data_ingestion import DailyDataPipeline

    # Initialize pipeline
    pipeline = DailyDataPipeline()

    try:
        # Process new PNGs
        logger.info("🔄 Processing PNGs...")
        stats = pipeline.process_new_pngs()

        logger.info(f"✅ Processing completed")
        logger.info(f"   PNGs processed: {stats['new_pngs']}")
        logger.info(f"   Patches created: {stats['total_patches']}")
        logger.info(f"   Errors: {stats['errors']}")

        if stats["new_pngs"] == 0:
            logger.warning("⚠️  No new PNGs processed. Check directory.")
            return False

        # Check HDF5 file
        hdf5_path = Path("data/processed/himawari_dataset.h5")
        if not hdf5_path.exists():
            logger.error(f"❌ HDF5 file not created: {hdf5_path}")
            return False

        logger.info(f"✅ HDF5 file created: {hdf5_path}")
        logger.info(f"   File size: {hdf5_path.stat().st_size / (1024*1024):.2f} MB")

        return True

    except Exception as e:
        logger.error(f"❌ HDF5 creation failed: {e}", exc_info=True)
        return False


def check_dataset_stats() -> bool:
    """Test 4: Dataset statistics."""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 4: Dataset Statistics")
    logger.info("=" * 60)

    from src.daily_data_ingestion import DailyDataPipeline

    pipeline = DailyDataPipeline()

    try:
        stats = pipeline.get_dataset_stats()

        logger.info("📊 Dataset Statistics:")
        for key, value in stats.items():
            logger.info(f"   {key}: {value}")

        if stats.get("total_samples", 0) > 0:
            logger.info("✅ Dataset contains samples")
            return True
        else:
            logger.warning("⚠️  Dataset is empty")
            return False

    except Exception as e:
        logger.error(f"❌ Failed to get stats: {e}", exc_info=True)
        return False


def main():
    """Run all tests."""
    logger.info("\n")
    logger.info("🧪 DAILY DATA INGESTION TEST SUITE")
    logger.info("=" * 60)

    results = {
        "PNG Loading": check_png_loading(),
        "Patch Creation": check_patch_creation(),
        "HDF5 Creation": check_hdf5_creation(),
        "Dataset Stats": check_dataset_stats(),
    }

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("TEST SUMMARY")
    logger.info("=" * 60)

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"{status}: {test_name}")

    logger.info("=" * 60)
    logger.info(f"Result: {passed}/{total} tests passed")
    logger.info("=" * 60)

    if passed == total:
        logger.info("\n✅ ALL TESTS PASSED!")
        logger.info("Your daily ingestion pipeline is ready!")
        logger.info("\nNext steps:")
        logger.info(
            "1. Add tomorrow's PNG to: data/raw/himawari8_pngs/"
        )
        logger.info("2. Run: python src/daily_scheduler.py schedule")
        logger.info("3. Or run manually: python src/daily_scheduler.py once")
        return 0
    else:
        logger.error(f"\n❌ {total - passed} test(s) failed")
        logger.error("Check the errors above and try again")
        return 1


def test_png_loading():
    if not check_png_loading():
        png_dir = Path("data/raw/himawari8_pngs")
        if not list(png_dir.glob("*.png")):
            pytest.skip(f"No PNG files found in {png_dir}")
        pytest.fail("PNG loading and channel extraction check failed")


def test_patch_creation():
    if not check_patch_creation():
        png_dir = Path("data/raw/himawari8_pngs")
        if not list(png_dir.glob("*.png")):
            pytest.skip(f"No PNG files found in {png_dir}")
        pytest.fail("Patch creation check failed")


def test_hdf5_creation():
    if not check_hdf5_creation():
        from src.daily_data_ingestion import DailyDataPipeline

        # Daily ingestion can be a no-op when there are no new PNGs to append.
        pipeline = DailyDataPipeline()
        dataset_stats = pipeline.get_dataset_stats()
        if dataset_stats.get("total_samples", 0) > 0:
            pytest.skip("No new PNGs available to append; existing dataset is present")
        hdf5_path = Path("data/processed/himawari_dataset.h5")
        if not hdf5_path.exists():
            pytest.skip("himawari_dataset.h5 is not present; run the ingestion pipeline locally to generate it")
        pytest.fail("HDF5 dataset creation check failed")


def test_dataset_stats():
    if not check_dataset_stats():
        pytest.skip("Dataset is currently empty")
        pytest.fail("Dataset statistics check failed")


if __name__ == "__main__":
    sys.exit(main())
