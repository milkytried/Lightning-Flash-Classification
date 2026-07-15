"""Create an immutable inventory and ten-minute availability calendar for supplied MMD CSVs."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path

import pandas as pd

LOGGER = logging.getLogger(__name__)
REQUIRED = {"Date/Time", "Latitude", "Longitude", "Cloud or Ground"}


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def create_inventory(mmd_root: Path, output_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    files = sorted(mmd_root.rglob("raw data all.csv"))
    if not files:
        raise FileNotFoundError(f"No MMD files found below {mmd_root}")
    inventory_rows: list[dict] = []
    ground_bins_by_date: dict[str, set[pd.Timestamp]] = {}
    valid_dates: set[str] = set()

    for path in files:
        base = {"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)}
        try:
            frame = pd.read_csv(path, low_memory=False)
            missing = sorted(REQUIRED.difference(frame.columns))
            if missing:
                raise ValueError(f"missing required columns: {missing}")
            timestamps = pd.to_datetime(frame["Date/Time"], utc=True, errors="coerce")
            ground = frame["Cloud or Ground"].astype(str).str.casefold().eq("ground")
            valid_timestamp = timestamps.notna()
            dates = sorted(timestamps[valid_timestamp].dt.strftime("%Y-%m-%d").unique())
            for date in dates:
                valid_dates.add(date)
                ground_bins_by_date.setdefault(date, set())
            for timestamp in timestamps[valid_timestamp & ground].dt.floor("10min"):
                ground_bins_by_date.setdefault(timestamp.strftime("%Y-%m-%d"), set()).add(timestamp)
            inventory_rows.append({
                **base,
                "status": "valid",
                "error": "",
                "dates": ";".join(dates),
                "row_count": int(len(frame)),
                "timestamp_min_utc": timestamps.min().isoformat() if valid_timestamp.any() else "",
                "timestamp_max_utc": timestamps.max().isoformat() if valid_timestamp.any() else "",
                "invalid_timestamp_count": int((~valid_timestamp).sum()),
                "ground_event_count": int((ground & valid_timestamp).sum()),
            })
        except Exception as exc:
            LOGGER.exception("Unreadable or malformed MMD file: %s", path)
            inventory_rows.append({**base, "status": "invalid", "error": str(exc), "dates": "", "row_count": 0,
                                   "timestamp_min_utc": "", "timestamp_max_utc": "", "invalid_timestamp_count": 0,
                                   "ground_event_count": 0})

    inventory = pd.DataFrame(inventory_rows).sort_values("path").reset_index(drop=True)
    bin_rows: list[dict] = []
    for date in sorted(valid_dates):
        day_start = pd.Timestamp(date, tz="UTC")
        active = ground_bins_by_date.get(date, set())
        for timestamp in pd.date_range(day_start, periods=144, freq="10min"):
            bin_rows.append({
                "frame_timestamp_utc": timestamp.isoformat(),
                "date": date,
                "mmd_file_valid_for_date": True,
                "recorded_ground_event": timestamp in active,
                "category": "active" if timestamp in active else "zero_recorded",
                "missing_date_policy": "missing dates are unknown and excluded",
            })
    bins = pd.DataFrame(bin_rows)
    output_root.mkdir(parents=True, exist_ok=True)
    inventory_path = output_root / "mmd_inventory.csv"
    bins_path = output_root / "mmd_ten_minute_bins.csv"
    inventory.to_csv(inventory_path, index=False)
    bins.to_csv(bins_path, index=False)
    summary = {
        "schema_version": "v2-phase1-inventory-1",
        "mmd_root": str(mmd_root),
        "discovered_files": len(files),
        "valid_files": int(inventory.status.eq("valid").sum()),
        "invalid_files": int(inventory.status.ne("valid").sum()),
        "rows": int(inventory.row_count.sum()),
        "ground_events": int(inventory.ground_event_count.sum()),
        "valid_dates": len(valid_dates),
        "active_ten_minute_bins": int(bins.recorded_ground_event.sum()),
        "zero_recorded_ten_minute_bins": int((~bins.recorded_ground_event).sum()),
        "missing_dates_policy": "unknown; never negative or calm",
        "timestamps": "UTC",
        "inventory_sha256": sha256_file(inventory_path),
        "bins_sha256": sha256_file(bins_path),
    }
    (output_root / "mmd_inventory_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return inventory, bins, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mmd-root", type=Path, default=Path("data/raw/himawari8_pngs"))
    parser.add_argument("--output-root", type=Path, default=Path("data/processed/v2"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    _, _, summary = create_inventory(args.mmd_root, args.output_root)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

