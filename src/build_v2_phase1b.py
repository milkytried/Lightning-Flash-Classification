"""Build the isolated Phase 1B pilot from its frozen science-first frame ledger."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.build_satellite_dataset import FrameSlot, build_hsd_key, download_frame, patch_black_fraction
from src.build_v2_pilot import crop, load_frame, lonlat_to_pixel, pixel_to_lonlat, save_patch, working_set_bytes
from src.create_mmd_inventory import sha256_file
from src.mmd_spatiotemporal_index import MMDSpatiotemporalIndex

LOGGER = logging.getLogger(__name__)


def _atomic_json(path: Path, value: dict) -> None:
    """Write a restart checkpoint without ever exposing a partial JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _ledger_identity(ledger_path: Path, configuration_hash: str) -> str:
    payload = f"{sha256_file(ledger_path)}|{configuration_hash}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def retained_strikes(index: MMDSpatiotemporalIndex, timestamp: pd.Timestamp, spacing_km: float, cap: int) -> tuple[pd.DataFrame, dict]:
    strikes = index.query_window(timestamp, 0, 10).copy().sort_values(["timestamp", "lat", "lon"], kind="mergesort")
    original = len(strikes)
    strikes["dedup_key"] = strikes.timestamp.astype(str) + "|" + strikes.lat.round(5).astype(str) + "|" + strikes.lon.round(5).astype(str)
    strikes = strikes.drop_duplicates("dedup_key")
    deduplicated = original - len(strikes)
    retained_indices = []
    for idx, row in strikes.iterrows():
        if retained_indices:
            previous = strikes.loc[retained_indices]
            distance = index.haversine_km(previous.lat.to_numpy(), previous.lon.to_numpy(), row.lat, row.lon)
            if float(distance.min()) < spacing_km:
                continue
        retained_indices.append(idx)
        if len(retained_indices) >= cap:
            break
    retained = strikes.loc[retained_indices].copy()
    return retained, {"candidate": original, "deduplicated": deduplicated, "spacing_or_cap_removed": int(len(strikes) - len(retained)), "retained": len(retained)}


def matched_negative(index, timestamp, target_lat, target_lon, image, config, rng):
    mask, sat, labels, sampling = config["study_mask"], config["satellite"], config["labels"], config["phase1b"]["sampling"]
    grid = float(sampling["geographic_grid_degrees"]); size = int(sat["patch_size"]); half = size // 2
    lat_cell = np.floor((target_lat - mask["latitude_min"]) / grid); lon_cell = np.floor((target_lon - mask["longitude_min"]) / grid)
    offsets = [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, 1), (-1, 1), (1, -1)]
    for level, (dy, dx) in enumerate(offsets):
        lat0 = mask["latitude_min"] + (lat_cell + dy) * grid; lon0 = mask["longitude_min"] + (lon_cell + dx) * grid
        lat_low, lat_high = max(lat0, mask["latitude_min"]), min(lat0 + grid, mask["latitude_max"])
        lon_low, lon_high = max(lon0, mask["longitude_min"]), min(lon0 + grid, mask["longitude_max"])
        if lat_low >= lat_high or lon_low >= lon_high:
            continue
        for _ in range(200):
            lat = float(rng.uniform(lat_low, lat_high))
            lon = float(rng.uniform(lon_low, lon_high))
            x, y = lonlat_to_pixel(lat, lon, image.shape[:2], mask)
            if x < half or y < half or x >= image.shape[1] - half or y >= image.shape[0] - half:
                continue
            query = index.patch_query(timestamp, lat, lon, size, float(sat["degrees_per_pixel"]), float(labels["safety_margin_km"]),
                                      int(labels["negative_window"]["start_minutes"]), int(labels["negative_window"]["end_minutes"]))
            if not query["clear"]:
                continue
            patch, bounds = crop(image, x, y, size)
            if patch is None or patch_black_fraction(patch) > 0.02:
                continue
            return {"lat": lat, "lon": lon, "x": x, "y": y, "patch": patch, "bounds": bounds,
                    "match_level": "same_grid_cell" if level == 0 else "neighbour_grid_cell",
                    "target_lat": target_lat, "target_lon": target_lon, **query}
    return None


def row_record(frame, files, timestamp, label, reason, centre_lat, centre_lon, x, y, bounds, patch, digest, path, config,
               target_lat=None, target_lon=None, match_level=None, nearest_distance=None, nearest_delta=None):
    x0, y0, x1, y1 = bounds; mask = config["study_mask"]
    north, west = pixel_to_lonlat(x0, y0, (280, 235), mask); south, east = pixel_to_lonlat(x1 - 1, y1 - 1, (280, 235), mask)
    boundary_distance = min(centre_lat - mask["latitude_min"], mask["latitude_max"] - centre_lat,
                            centre_lon - mask["longitude_min"], mask["longitude_max"] - centre_lon) * 111.0
    return {"path": str(path), "label": label, "split": frame.split, "frame_id": f"H09_{timestamp.strftime('%Y%m%d_%H%M')}",
            "frame_timestamp_utc": timestamp.isoformat(), "date": timestamp.strftime("%Y-%m-%d"), "storm_id": frame.storm_id,
            "frame_category": frame.category, "label_reason": reason, "centre_lat": centre_lat, "centre_lon": centre_lon,
            "x": x, "y": y, "crop_x0": x0, "crop_y0": y0, "crop_x1": x1, "crop_y1": y1,
            "crop_north": north, "crop_south": south, "crop_west": west, "crop_east": east,
            "geographic_grid_cell": f"{int((centre_lat-mask['latitude_min'])//0.5)}_{int((centre_lon-mask['longitude_min'])//0.5)}",
            "distance_to_study_mask_boundary_km": boundary_distance, "matched_target_lat": target_lat, "matched_target_lon": target_lon,
            "geographic_match_level": match_level, "nearest_strike_distance_km": nearest_distance,
            "nearest_strike_time_difference_minutes": nearest_delta, "source_himawari_files": ";".join(str(item) for item in files),
            "sha256": digest, "mean_B08": float(patch[:,:,0].mean()), "mean_B13": float(patch[:,:,1].mean()),
            "mean_B15": float(patch[:,:,2].mean()), "min_B13": int(patch[:,:,1].min()),
            "configuration_hash": config["phase1b"]["configuration_hash"], "label_rule_version": config["labels"]["label_rule_version"]}


def build(config_path: Path, allow_download: bool = True, resume: bool = True) -> tuple[pd.DataFrame, dict]:
    started = time.perf_counter(); config = yaml.safe_load(config_path.read_text(encoding="utf-8")); phase = config["phase1b"]
    ledger = pd.read_csv(phase["outputs"]["frame_ledger"]); ledger["frame_timestamp_utc"] = pd.to_datetime(ledger.frame_timestamp_utc, utc=True)
    index = MMDSpatiotemporalIndex.from_inventory(Path(config["outputs"]["inventory_csv"]), config["study_mask"])
    rng = np.random.default_rng(int(config["frames"]["random_seed"])); root = Path(phase["outputs"]["root"])
    sampling = phase["sampling"]; spacing = float(sampling["positive_minimum_centre_spacing_km"]); cap = int(sampling["positive_cap_per_frame"])
    # Freeze spatial targets before image/cache work.
    targets_by_split = defaultdict(list); retained_by_frame = {}; removal_stats = {}
    for frame in ledger[ledger.category.eq("active")].itertuples():
        strikes, stats = retained_strikes(index, frame.frame_timestamp_utc, spacing, cap)
        retained_by_frame[frame.frame_timestamp_utc.isoformat()] = strikes; removal_stats[frame.frame_timestamp_utc.isoformat()] = stats
        targets_by_split[frame.split].extend([(float(row.lat), float(row.lon)) for row in strikes.itertuples()])
    target_positions = {split: 0 for split in targets_by_split}; rows = []; unavailable = []; downloaded_bytes = 0; downloaded_files = 0; peak = working_set_bytes()
    checkpoint_path = root / "build_checkpoint.json"
    identity = _ledger_identity(Path(phase["outputs"]["frame_ledger"]), phase["configuration_hash"])
    start_number = 1
    if resume and checkpoint_path.exists():
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if checkpoint.get("identity") != identity:
            raise RuntimeError("Phase 1B checkpoint does not match the frozen ledger/configuration")
        start_number = int(checkpoint["next_frame_number"])
        rows = checkpoint["rows"]
        unavailable = checkpoint["unavailable"]
        downloaded_bytes = int(checkpoint["downloaded_bytes"])
        downloaded_files = int(checkpoint["downloaded_files"])
        peak = max(peak, int(checkpoint["peak_working_set_bytes"]))
        target_positions.update({key: int(value) for key, value in checkpoint["target_positions"].items()})
        rng.bit_generator.state = checkpoint["rng_state"]
        LOGGER.info("Resuming Phase 1B at frame %d/%d with %d samples", start_number, len(ledger), len(rows))
    for number, frame in enumerate(ledger.itertuples(), 1):
        if number < start_number:
            continue
        timestamp = pd.Timestamp(frame.frame_timestamp_utc).tz_convert("UTC"); slot = FrameSlot(timestamp)
        before = {}
        for band in config["satellite"]["bands"]:
            for segment in config["satellite"]["segments"]:
                local = Path(config["inputs"]["himawari_cache"]) / slot.bucket / build_hsd_key(slot, band, int(segment)); before[str(local)] = local.stat().st_size if local.exists() else None
        try:
            if not allow_download and not all(value is not None for value in before.values()): raise FileNotFoundError("download disabled")
            files = download_frame(slot, config["satellite"]["bands"], config["satellite"]["segments"], config["inputs"]["himawari_cache"])
            for path in files:
                if before[str(path)] is None: downloaded_files += 1; downloaded_bytes += path.stat().st_size
            image = load_frame(files, config)
        except Exception as exc:
            LOGGER.exception("Unavailable frozen Phase 1B frame %s", timestamp); unavailable.append({"timestamp": timestamp.isoformat(), "error": str(exc)}); continue
        frame_rows = 0
        if frame.category == "active":
            strikes = retained_by_frame[timestamp.isoformat()]
            for positive_number, strike in enumerate(strikes.itertuples()):
                x, y = lonlat_to_pixel(strike.lat, strike.lon, image.shape[:2], config["study_mask"]); patch, bounds = crop(image, x, y, int(config["satellite"]["patch_size"]))
                if patch is None or patch_black_fraction(patch) > 0.02: continue
                path = root / "patches" / frame.split / "positive" / f"{slot.date}_{slot.hhmm}_pos_{positive_number}.png"; digest = save_patch(path, patch)
                rows.append(row_record(frame, files, timestamp, 1, "deduplicated and spacing-retained MMD ground strike in [t,t+10m)", strike.lat, strike.lon, x, y, bounds, patch, digest, path, config)); frame_rows += 1
                negative = matched_negative(index, timestamp, strike.lat, strike.lon, image, config, rng)
                if negative:
                    npath = root / "patches" / frame.split / "negative" / f"{slot.date}_{slot.hhmm}_hard_{positive_number}.png"; ndigest = save_patch(npath, negative["patch"])
                    rows.append(row_record(frame, files, timestamp, 0, "spatially matched active-frame negative clear under frozen [t-20m,t+30m) rule",
                        negative["lat"], negative["lon"], negative["x"], negative["y"], negative["bounds"], negative["patch"], ndigest, npath, config,
                        strike.lat, strike.lon, negative["match_level"], negative["nearest_distance_km"], negative["nearest_time_difference_minutes"])); frame_rows += 1
        else:
            targets = targets_by_split[frame.split]
            for zero_number in range(int(sampling["zero_recorded_patches_per_frame"])):
                target = targets[target_positions[frame.split] % len(targets)]; target_positions[frame.split] += 1
                negative = matched_negative(index, timestamp, target[0], target[1], image, config, rng)
                if not negative: continue
                path = root / "patches" / frame.split / "negative" / f"{slot.date}_{slot.hhmm}_zero_{zero_number}.png"; digest = save_patch(path, negative["patch"])
                rows.append(row_record(frame, files, timestamp, 0, "positive-distribution-matched zero-recorded-frame negative clear under frozen rule",
                    negative["lat"], negative["lon"], negative["x"], negative["y"], negative["bounds"], negative["patch"], digest, path, config,
                    target[0], target[1], negative["match_level"], negative["nearest_distance_km"], negative["nearest_time_difference_minutes"])); frame_rows += 1
        peak = max(peak, working_set_bytes()); LOGGER.info("Phase1B frame %d/%d %s samples=%d", number, len(ledger), timestamp, frame_rows)
        _atomic_json(checkpoint_path, {"identity": identity, "next_frame_number": number + 1, "rows": rows,
                     "unavailable": unavailable, "downloaded_bytes": downloaded_bytes, "downloaded_files": downloaded_files,
                     "peak_working_set_bytes": peak, "target_positions": target_positions, "rng_state": rng.bit_generator.state})
    manifest = pd.DataFrame(rows); manifest_path = Path(phase["outputs"]["manifest"]); manifest_path.parent.mkdir(parents=True, exist_ok=True); manifest.to_csv(manifest_path, index=False)
    unavailable_path = root / "unavailable_frames.json"; unavailable_path.write_text(json.dumps(unavailable, indent=2) + "\n", encoding="utf-8")
    metrics = {"phase": "V2 Phase 1B", "frames_planned": len(ledger), "frames_built": int(manifest.frame_id.nunique()), "unavailable_frames": len(unavailable),
               "patches": len(manifest), "class_by_split": {"|".join(map(str,key)): int(value) for key,value in manifest.groupby(["split","label"]).size().items()},
               "positive_frames_by_split": manifest[manifest.label.eq(1)].groupby("split").frame_id.nunique().to_dict(),
               "elapsed_seconds": time.perf_counter()-started, "downloaded_files": downloaded_files, "downloaded_bytes": downloaded_bytes,
               "peak_working_set_bytes": peak, "disk_patch_bytes": int(sum(Path(path).stat().st_size for path in manifest.path)),
               "manifest_sha256": sha256_file(manifest_path), "ledger_sha256": sha256_file(Path(phase["outputs"]["frame_ledger"])),
               "configuration_hash": phase["configuration_hash"], "positive_removal_totals": {key: int(sum(item[key] for item in removal_stats.values())) for key in ["candidate","deduplicated","spacing_or_cap_removed","retained"]}}
    (root / "build_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    checkpoint_path.unlink(missing_ok=True)
    return manifest, metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--config", type=Path, default=Path("configs/v2_minimum.yaml")); parser.add_argument("--no-download", action="store_true"); parser.add_argument("--no-resume", action="store_true"); args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    _, metrics = build(args.config, not args.no_download, not args.no_resume); print(json.dumps(metrics, indent=2))


if __name__ == "__main__": main()
