"""Build only the deterministic 100-frame Version 2 pilot from its frozen frame ledger."""

from __future__ import annotations

import argparse
import ctypes
import os
import subprocess
import hashlib
import json
import logging
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from PIL import Image
from pyresample.geometry import AreaDefinition
from satpy import Scene

from src.build_satellite_dataset import FrameSlot, band_to_uint8, build_hsd_key, download_frame, patch_black_fraction
from src.create_mmd_inventory import sha256_file
from src.mmd_spatiotemporal_index import MMDSpatiotemporalIndex

LOGGER = logging.getLogger(__name__)

def working_set_bytes() -> int:
    """Return current process working set through Windows process telemetry."""
    output = subprocess.check_output([
        "powershell", "-NoProfile", "-Command",
        f"(Get-Process -Id {os.getpid()}).WorkingSet64",
    ], text=True)
    return int(output.strip())


def load_config(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def study_area(config: dict) -> AreaDefinition:
    mask, resolution = config["study_mask"], float(config["satellite"]["degrees_per_pixel"])
    width = int(math.ceil((mask["longitude_max"] - mask["longitude_min"]) / resolution))
    height = int(math.ceil((mask["latitude_max"] - mask["latitude_min"]) / resolution))
    return AreaDefinition("v2_empirical_peninsular", "V2 conservative empirical Peninsular Malaysia mask",
                          "v2_empirical_peninsular", {"proj": "longlat", "datum": "WGS84"}, width, height,
                          (mask["longitude_min"], mask["latitude_min"], mask["longitude_max"], mask["latitude_max"]))


def load_frame(files: list[Path], config: dict) -> np.ndarray:
    bands = config["satellite"]["bands"]
    scene = Scene(filenames=[str(path) for path in files], reader="ahi_hsd")
    scene.load(bands, calibration="brightness_temperature")
    result = scene.resample(study_area(config))
    return np.stack([band_to_uint8(result[band].values, band) for band in bands], axis=-1)


def lonlat_to_pixel(lat: float, lon: float, shape: tuple[int, int], mask: dict) -> tuple[int, int]:
    height, width = shape
    x = int(round((lon - mask["longitude_min"]) / (mask["longitude_max"] - mask["longitude_min"]) * (width - 1)))
    y = int(round((mask["latitude_max"] - lat) / (mask["latitude_max"] - mask["latitude_min"]) * (height - 1)))
    return x, y


def pixel_to_lonlat(x: int, y: int, shape: tuple[int, int], mask: dict) -> tuple[float, float]:
    height, width = shape
    lon = mask["longitude_min"] + x / (width - 1) * (mask["longitude_max"] - mask["longitude_min"])
    lat = mask["latitude_max"] - y / (height - 1) * (mask["latitude_max"] - mask["latitude_min"])
    return lat, lon


def crop(image: np.ndarray, x: int, y: int, size: int) -> tuple[np.ndarray | None, tuple[int, int, int, int]]:
    half = size // 2
    x0, y0, x1, y1 = x - half, y - half, x - half + size, y - half + size
    if x0 < 0 or y0 < 0 or x1 > image.shape[1] or y1 > image.shape[0]:
        return None, (x0, y0, x1, y1)
    return image[y0:y1, x0:x1], (x0, y0, x1, y1)


def save_patch(path: Path, patch: np.ndarray) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(patch.astype(np.uint8), mode="RGB").save(path)
    return sha256_file(path)


def build_pilot(config_path: Path, allow_download: bool = True) -> tuple[pd.DataFrame, dict]:
    started = time.perf_counter()
    config = load_config(config_path)
    out, sat, labels, frames_cfg = config["outputs"], config["satellite"], config["labels"], config["frames"]
    pilot_ledger = pd.read_csv(out["pilot_frame_ledger"])
    pilot_ledger["frame_timestamp_utc"] = pd.to_datetime(pilot_ledger.frame_timestamp_utc, utc=True)
    if len(pilot_ledger) != int(frames_cfg["pilot_total"]):
        raise ValueError("Pilot ledger does not contain the configured number of frames")
    index = MMDSpatiotemporalIndex.from_inventory(Path(out["inventory_csv"]), config["study_mask"])
    root = Path(out["pilot_root"])
    rng = np.random.default_rng(int(frames_cfg["random_seed"]))
    rows, downloaded_bytes, downloaded_files = [], 0, 0
    peak_working_set_bytes = working_set_bytes()
    for frame_number, frame in pilot_ledger.iterrows():
        ts = pd.Timestamp(frame.frame_timestamp_utc).tz_convert("UTC")
        slot = FrameSlot(ts)
        before = {}
        for band in sat["bands"]:
            for segment in sat["segments"]:
                local = Path(config["inputs"]["himawari_cache"]) / slot.bucket / build_hsd_key(slot, band, int(segment))
                before[str(local)] = local.stat().st_size if local.exists() else None
        if not allow_download and not all(value is not None for value in before.values()):
            raise FileNotFoundError(f"Pilot frame requires download but downloads are disabled: {ts}")
        files = download_frame(slot, sat["bands"], sat["segments"], config["inputs"]["himawari_cache"])
        for path in files:
            if before[str(path)] is None:
                downloaded_files += 1; downloaded_bytes += path.stat().st_size
        image = load_frame(files, config)
        patch_size = int(sat["patch_size"]); half = patch_size // 2; mask = config["study_mask"]
        positive_strikes = index.query_window(ts, int(labels["positive_window"]["start_minutes"]), int(labels["positive_window"]["end_minutes"]))
        if frame.category == "active":
            positive_strikes = positive_strikes.sort_values(["timestamp", "lat", "lon"], kind="mergesort").head(int(frames_cfg["max_positive_patches_per_active_frame"]))
            for pos_number, strike in positive_strikes.iterrows():
                x, y = lonlat_to_pixel(float(strike.lat), float(strike.lon), image.shape[:2], mask)
                patch, bounds = crop(image, x, y, patch_size)
                if patch is None or patch_black_fraction(patch) > 0.02:
                    continue
                path = root / "patches" / frame.split / "positive" / f"{slot.date}_{slot.hhmm}_pos_{pos_number}.png"
                digest = save_patch(path, patch)
                rows.append(sample_row(frame, files, ts, 1, "recorded ground strike in positive [t,t+10m) window", x, y, bounds,
                                       float(strike.lat), float(strike.lon), patch, digest, None, None, path, config))
        target_negatives = int(frames_cfg["max_hard_negative_patches_per_active_frame"] if frame.category == "active" else frames_cfg["zero_recorded_patches_per_frame"])
        accepted, attempts = 0, 0
        while accepted < target_negatives and attempts < target_negatives * 1000:
            attempts += 1
            x = int(rng.integers(half, image.shape[1] - half)); y = int(rng.integers(half, image.shape[0] - half))
            lat, lon = pixel_to_lonlat(x, y, image.shape[:2], mask)
            query = index.patch_query(ts, lat, lon, patch_size, float(sat["degrees_per_pixel"]), float(labels["safety_margin_km"]),
                                      int(labels["negative_window"]["start_minutes"]), int(labels["negative_window"]["end_minutes"]))
            if not query["clear"]:
                continue
            patch, bounds = crop(image, x, y, patch_size)
            if patch is None or patch_black_fraction(patch) > 0.02:
                continue
            reason = "full crop plus safety margin clear of MMD-recorded ground strikes in configured temporal window"
            path = root / "patches" / frame.split / "negative" / f"{slot.date}_{slot.hhmm}_neg_{accepted}.png"
            digest = save_patch(path, patch)
            rows.append(sample_row(frame, files, ts, 0, reason, x, y, bounds, lat, lon, patch, digest,
                                   query["nearest_distance_km"], query["nearest_time_difference_minutes"], path, config))
            accepted += 1
        peak_working_set_bytes = max(peak_working_set_bytes, working_set_bytes())
        LOGGER.info("Pilot frame %d/%d %s category=%s samples=%d", frame_number + 1, len(pilot_ledger), ts, frame.category,
                    sum(1 for row in rows if row["frame_timestamp_utc"] == ts.isoformat()))
    manifest = pd.DataFrame(rows)
    manifest_path = Path(out["pilot_manifest"])
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(manifest_path, index=False)
    elapsed = time.perf_counter() - started
    patch_bytes = sum(Path(path).stat().st_size for path in manifest.path)
    metrics = {
        "pilot_frames_planned": len(pilot_ledger), "pilot_frames_built": int(manifest.frame_id.nunique()), "patches": len(manifest),
        "class_counts": manifest.label.value_counts().sort_index().to_dict(), "category_counts": manifest.frame_category.value_counts().to_dict(),
        "elapsed_seconds": elapsed, "downloaded_files": downloaded_files, "downloaded_bytes": downloaded_bytes,
        "patch_bytes": patch_bytes, "peak_working_set_bytes": peak_working_set_bytes, "peak_working_set_gib": peak_working_set_bytes / 2**30, "manifest_sha256": sha256_file(manifest_path),
        "frame_ledger_sha256": sha256_file(Path(out["pilot_frame_ledger"])),
        "peak_process_memory_note": "Measured in-process after each sequential frame with Windows GetProcessMemoryInfo.",
        "study_mask_statement": config["description"],
    }
    (root / "build_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return manifest, metrics


def sample_row(frame, files, ts, label, reason, x, y, bounds, lat, lon, patch, digest, nearest_distance, nearest_delta, path, config):
    x0, y0, x1, y1 = bounds; mask = config["study_mask"]; shape = (int(math.ceil((mask["latitude_max"]-mask["latitude_min"])/config["satellite"]["degrees_per_pixel"])), int(math.ceil((mask["longitude_max"]-mask["longitude_min"])/config["satellite"]["degrees_per_pixel"])))
    north, west = pixel_to_lonlat(x0, y0, shape, mask); south, east = pixel_to_lonlat(x1 - 1, y1 - 1, shape, mask)
    return {"path": str(path), "label": label, "split": frame.split, "frame_id": f"H09_{ts.strftime('%Y%m%d_%H%M')}",
            "frame_timestamp_utc": ts.isoformat(), "date": ts.strftime("%Y-%m-%d"), "storm_id": frame.storm_id,
            "frame_category": frame.category, "label_reason": reason, "centre_lat": lat, "centre_lon": lon, "x": x, "y": y,
            "crop_x0": x0, "crop_y0": y0, "crop_x1": x1, "crop_y1": y1, "crop_north": north, "crop_south": south,
            "crop_west": west, "crop_east": east, "nearest_strike_distance_km": nearest_distance,
            "nearest_strike_time_difference_minutes": nearest_delta, "source_himawari_files": ";".join(str(item) for item in files),
            "sha256": digest, "mean_B08": float(patch[:,:,0].mean()), "mean_B13": float(patch[:,:,1].mean()),
            "mean_B15": float(patch[:,:,2].mean()), "min_B13": int(patch[:,:,1].min()), "ledger_hash": frame.ledger_hash,
            "study_mask_kind": config["study_mask"]["kind"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/v2_minimum.yaml"))
    parser.add_argument("--no-download", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    _, metrics = build_pilot(args.config, allow_download=not args.no_download)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()


