"""
Scheduled daily data processing and model retraining.

Runs at a specified time each day to:
1. Check for new PNG files in data/raw/himawari8_pngs/
2. Process new PNGs and append to HDF5 dataset
3. Retrain model if enough new data accumulated
4. Log results and metrics
"""

import logging
import json
from pathlib import Path
from datetime import datetime, time
from typing import Optional
import schedule
import time as time_module

from daily_data_ingestion import DailyDataPipeline


logger = logging.getLogger(__name__)


class DailyTrainingScheduler:
    """Schedule daily data processing and model retraining."""

    def __init__(
        self,
        run_time: str = "06:00",  # 6 AM daily
        min_new_samples: int = 100,  # Retrain if 100+ new samples
        hdf5_path: str = "data/processed/himawari_dataset.h5",
        log_path: str = "logs/daily_processing.log",
    ):
        """
        Initialize scheduler.

        Args:
            run_time: Time to run daily (HH:MM format, 24-hour)
            min_new_samples: Minimum new samples to trigger retraining
            hdf5_path: Path to HDF5 dataset
            log_path: Path to processing log
        """
        self.run_time = run_time
        self.min_new_samples = min_new_samples
        self.hdf5_path = Path(hdf5_path)
        self.log_path = Path(log_path)
        self.pipeline = DailyDataPipeline(hdf5_path=str(self.hdf5_path))

        # Create log directory
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

        # Setup logging
        self._setup_logging()

    def _setup_logging(self) -> None:
        """Configure logging to file and console."""
        formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        # File handler
        fh = logging.FileHandler(self.log_path)
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)

        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)

        logger.addHandler(fh)
        logger.addHandler(ch)
        logger.setLevel(logging.INFO)

    def run_daily_task(self) -> None:
        """Execute daily data processing task."""
        logger.info("=" * 60)
        logger.info("[START] DAILY TASK STARTED")
        logger.info(f"Timestamp: {datetime.now().isoformat()}")
        logger.info("=" * 60)

        try:
            # Get current dataset stats
            before_stats = self.pipeline.get_dataset_stats()
            before_samples = before_stats.get("total_samples", 0)

            logger.info(f"[STATS] Before: {before_samples} samples")

            # Process new PNGs
            logger.info("[SCAN] Scanning for new PNG files...")
            process_stats = self.pipeline.process_new_pngs()

            # Get updated stats
            after_stats = self.pipeline.get_dataset_stats()
            after_samples = after_stats.get("total_samples", 0)
            new_samples = after_samples - before_samples

            logger.info(f"[STATS] After: {after_samples} samples")
            logger.info(f"[STATS] New samples: {new_samples}")

            # Log processing results
            logger.info(f"[RESULT] PNGs processed: {process_stats['new_pngs']}")
            logger.info(f"[RESULT] Patches created: {process_stats['total_patches']}")
            logger.info(f"[RESULT] Errors: {process_stats['errors']}")

            # Determine if retraining is needed
            labeled_samples = after_stats.get("labeled_samples", 0)
            if labeled_samples >= self.min_new_samples:
                logger.info(
                    f"\n[TRAIN] RETRAINING TRIGGERED: {labeled_samples} labeled samples available"
                )
                logger.info(
                    "[INFO] Run: python src/train.py (when lightning labels are available)"
                )
            else:
                logger.info(
                    f"[TRAIN] Not retraining yet. Need {self.min_new_samples} labeled samples, "
                    f"have {labeled_samples}"
                )

            # Log summary
            summary = {
                "timestamp": datetime.now().isoformat(),
                "samples_before": before_samples,
                "samples_after": after_samples,
                "new_samples": new_samples,
                "pngs_processed": process_stats["new_pngs"],
                "patches_created": process_stats["total_patches"],
                "errors": process_stats["errors"],
                "dataset_stats": after_stats,
            }

            self._log_summary(summary)

            logger.info("=" * 60)
            logger.info("[DONE] DAILY TASK COMPLETED")
            logger.info("=" * 60 + "\n")

        except Exception as e:
            logger.error(f"[ERROR] Task failed: {e}", exc_info=True)

    def _log_summary(self, summary: dict) -> None:
        """Save JSON summary of daily processing."""
        summary_dir = self.log_path.parent / "daily_summaries"
        summary_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_file = summary_dir / f"summary_{timestamp}.json"

        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)

        logger.info(f"[LOG] Summary saved: {summary_file}")

    def schedule_task(self) -> None:
        """Schedule task to run at specified time daily."""
        schedule.every().day.at(self.run_time).do(self.run_daily_task)
        logger.info(f"[SCHEDULE] Daily task at {self.run_time}")

    def start_scheduler(self) -> None:
        """Start the scheduler (blocking)."""
        logger.info("[START] Scheduler started. Press Ctrl+C to stop.")

        try:
            while True:
                schedule.run_pending()
                time_module.sleep(60)  # Check every minute
        except KeyboardInterrupt:
            logger.info("\n[STOP] Scheduler stopped by user")

    def run_once(self) -> None:
        """Run the daily task immediately (for testing)."""
        logger.info("Running daily task immediately (testing mode)...")
        self.run_daily_task()


def main():
    """Main entry point."""
    import sys

    # Parse arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == "once":
            # Run immediately
            print("[TEST] Running daily task immediately...")
            scheduler = DailyTrainingScheduler()
            scheduler.run_once()
        elif sys.argv[1] == "schedule":
            # Start scheduler
            print("[START] Starting daily scheduler...")
            scheduler = DailyTrainingScheduler()
            scheduler.schedule_task()
            scheduler.start_scheduler()
        else:
            print(f"Unknown argument: {sys.argv[1]}")
            print("Usage:")
            print("  python daily_scheduler.py once       # Run immediately")
            print("  python daily_scheduler.py schedule   # Start scheduler")
    else:
        # Default: run immediately
        print("[TEST] Running daily task immediately...")
        scheduler = DailyTrainingScheduler()
        scheduler.run_once()


if __name__ == "__main__":
    main()
