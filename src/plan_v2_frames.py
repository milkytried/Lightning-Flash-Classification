"""Create the immutable V2 candidate frame ledger and deterministic 100-frame pilot plan."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.cluster import DBSCAN

from src.build_satellite_dataset import FrameSlot, build_hsd_key
from src.create_mmd_inventory import sha256_file
from src.mmd_spatiotemporal_index import MMDSpatiotemporalIndex

LOGGER = logging.getLogger(__name__)


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def split_for_timestamp(timestamp: pd.Timestamp, config: dict) -> str:
    ts = pd.Timestamp(timestamp).tz_convert("UTC")
    if ts <= pd.Timestamp(config["splits"]["train_end_utc"]):
        return "train"
    if ts <= pd.Timestamp(config["splits"]["validation_end_utc"]):
        return "val"
    return "test"


def cached_and_keys(timestamp: pd.Timestamp, config: dict) -> tuple[bool, list[str], list[str]]:
    slot = FrameSlot(pd.Timestamp(timestamp).tz_convert("UTC"))
    cache = Path(config["inputs"]["himawari_cache"])
    keys, paths = [], []
    for band in config["satellite"]["bands"]:
        for segment in config["satellite"]["segments"]:
            key = build_hsd_key(slot, band, int(segment))
            keys.append(f"{slot.bucket}/{key}")
            paths.append(str(cache / slot.bucket / key))
    return all(Path(path).exists() for path in paths), keys, paths


def index_bin_summary(index: MMDSpatiotemporalIndex) -> pd.DataFrame:
    rows = []
    for bin_ns, idx in sorted(index.bin_indices.items()):
        rows.append({"frame_timestamp_utc": pd.Timestamp(bin_ns, tz="UTC"),
                     "in_mask_ground_strike_count": int(len(idx)),
                     "strike_centroid_lat": float(index.latitudes[idx].mean()),
                     "strike_centroid_lon": float(index.longitudes[idx].mean())})
    return pd.DataFrame(rows)


def assign_storm_groups(active: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, dict]:
    params = config["storm_clustering"]
    rows = active.copy().sort_values("frame_timestamp_utc").reset_index(drop=True)
    timestamps = pd.to_datetime(rows.frame_timestamp_utc, utc=True)
    time_minutes = timestamps.to_numpy(dtype="datetime64[ns]").astype("int64") / 60e9
    mean_lat = float(rows.strike_centroid_lat.mean())
    features = np.column_stack([
        time_minutes / float(params["temporal_scale_minutes"]),
        rows.strike_centroid_lon.to_numpy() * 111.32 * np.cos(np.radians(mean_lat)) / float(params["spatial_scale_km"]),
        rows.strike_centroid_lat.to_numpy() * 111.32 / float(params["spatial_scale_km"]),
    ])
    labels = DBSCAN(eps=float(params["eps"]), min_samples=int(params["min_samples"]), algorithm="kd_tree").fit_predict(features)
    rows["storm_id"] = [f"DERIVED_STORM_{label:06d}" for label in labels]
    rows["storm_assignment_reason"] = "DBSCAN on ten-minute in-mask strike centroids; derived analytical group, not official storm ID"
    latest = rows.groupby("storm_id").frame_timestamp_utc.max()
    storm_split = {storm: split_for_timestamp(pd.Timestamp(ts), config) for storm, ts in latest.items()}
    rows["split"] = rows.storm_id.map(storm_split)
    return rows, {**params, "derived_storm_count": int(rows.storm_id.nunique()), "split_anchor": "latest frame in derived group"}


def stratified_pick(rows: pd.DataFrame, count: int, seed: int, prefer_cached: bool) -> pd.DataFrame:
    """Deterministically maximize cache reuse, then diversify month/local-hour strata."""
    if count > len(rows):
        raise ValueError(f"Requested {count} frames from only {len(rows)} candidates")
    rng = np.random.default_rng(seed)
    candidates = rows.copy()
    candidates["_random"] = rng.random(len(candidates))
    candidates["_stratum"] = candidates.month.astype(str) + "|" + (candidates.malaysia_local_hour // 4).astype(str)
    picked = []
    cache_levels = [True, False] if prefer_cached else [False, True]
    for cache_level in cache_levels:
        pool = candidates[candidates.satellite_files_cached.eq(cache_level)]
        groups = {key: group.sort_values("_random").copy() for key, group in pool.groupby("_stratum")}
        while len(picked) < count:
            available = [(key, group) for key, group in groups.items() if len(group)]
            if not available:
                break
            key, group = min(available, key=lambda item: (sum(1 for row in picked if row["_stratum"] == item[0]), item[0]))
            picked.append(group.iloc[0].to_dict())
            groups[key] = group.iloc[1:]
        if len(picked) == count:
            break
    return pd.DataFrame(picked).drop(columns=["_random", "_stratum"], errors="ignore")


def select_pilot_category(rows: pd.DataFrame, config: dict, seed: int) -> pd.DataFrame:
    parts = []
    for offset, (split, count) in enumerate(config["frames"]["pilot_split_per_category"].items()):
        parts.append(stratified_pick(rows[rows.split.eq(split)], int(count), seed + offset, True))
    return pd.concat(parts, ignore_index=True)
def create_frame_plan(config_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    config = load_config(config_path)
    outputs = config["outputs"]
    bins = pd.read_csv(outputs["bins_csv"])
    bins["frame_timestamp_utc"] = pd.to_datetime(bins.frame_timestamp_utc, utc=True)
    index = MMDSpatiotemporalIndex.from_inventory(Path(outputs["inventory_csv"]), config["study_mask"])
    ledger = bins.merge(index_bin_summary(index), on="frame_timestamp_utc", how="left")
    ledger["in_mask_ground_strike_count"] = ledger.in_mask_ground_strike_count.fillna(0).astype(int)
    ledger["category"] = np.where(ledger.in_mask_ground_strike_count.gt(0), "active", "zero_recorded")
    active, storm_meta = assign_storm_groups(ledger[ledger.category.eq("active")], config)
    zero = ledger[ledger.category.eq("zero_recorded")].copy()
    zero["storm_id"] = zero.frame_timestamp_utc.dt.strftime("ZERO_RECORDED_%Y%m%d_%H%M")
    zero["storm_assignment_reason"] = "No in-mask recorded ground strike; independent frame group, not an official storm ID"
    zero["split"] = zero.frame_timestamp_utc.map(lambda ts: split_for_timestamp(ts, config))
    ledger = pd.concat([active, zero], ignore_index=True).sort_values("frame_timestamp_utc").reset_index(drop=True)
    ledger["date"] = ledger.frame_timestamp_utc.dt.strftime("%Y-%m-%d")
    ledger["utc_hour"] = ledger.frame_timestamp_utc.dt.hour
    ledger["malaysia_local_hour"] = (ledger.utc_hour + 8) % 24
    ledger["month"] = ledger.frame_timestamp_utc.dt.month
    checks = [cached_and_keys(ts, config) for ts in ledger.frame_timestamp_utc]
    ledger["satellite_files_cached"] = [item[0] for item in checks]
    ledger["satellite_file_availability"] = np.where(ledger.satellite_files_cached, "complete_local_cache", "not_cached_download_required")
    ledger["required_noaa_object_keys"] = [";".join(item[1]) for item in checks]
    ledger["required_local_paths"] = [";".join(item[2]) for item in checks]
    ledger["random_selection_seed"] = int(config["frames"]["random_seed"])
    ledger["selection_reason"] = "eligible supplied-MMD-date candidate; category determined only from in-mask recorded strikes"
    ledger["ledger_version"] = config["version"]
    ledger["study_mask_statement"] = config["description"]
    output_csv = Path(outputs["frame_ledger_csv"])
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    ledger_hash = hashlib.sha256(ledger.to_csv(index=False).encode()).hexdigest()
    ledger["ledger_hash"] = ledger_hash
    ledger.to_csv(output_csv, index=False)

    frames = config["frames"]
    active_pilot = select_pilot_category(ledger[ledger.category.eq("active")], config, int(frames["random_seed"]))
    zero_pilot = select_pilot_category(ledger[ledger.category.eq("zero_recorded")], config, int(frames["random_seed"]) + 10)
    pilot = pd.concat([active_pilot, zero_pilot], ignore_index=True).sort_values("frame_timestamp_utc").reset_index(drop=True)
    pilot["pilot_selected"] = True
    pilot["pilot_selection_reason"] = "deterministic stratified category/month/local-hour/split selection; local cache preferred"
    pilot_path = Path(outputs["pilot_frame_ledger"])
    pilot_path.parent.mkdir(parents=True, exist_ok=True)
    pilot.to_csv(pilot_path, index=False)
    metadata = {
        "ledger_version": config["version"], "ledger_hash": ledger_hash, "ledger_file_sha256": sha256_file(output_csv),
        "candidate_frames": len(ledger), "categories": ledger.category.value_counts().to_dict(), "splits": ledger.split.value_counts().to_dict(),
        "storm_clustering": storm_meta, "storm_groups_crossing_splits": int((ledger.groupby("storm_id").split.nunique() > 1).sum()),
        "pilot_frames": len(pilot), "pilot_categories": pilot.category.value_counts().to_dict(), "pilot_splits": pilot.split.value_counts().to_dict(),
        "pilot_cached_frames": int(pilot.satellite_files_cached.sum()), "pilot_download_frames": int((~pilot.satellite_files_cached).sum()),
        "mask_statement": config["description"], "missing_dates_policy": "unknown and excluded",
    }
    Path(outputs["frame_ledger_json"]).write_text(json.dumps({"metadata": metadata, "records": json.loads(ledger.to_json(orient="records", date_format="iso"))}, indent=2) + "\n", encoding="utf-8")
    Path("report/V2_FRAME_PLAN.md").write_text(
        "# Version 2 frame plan\n\n" + config["description"] + "\n\n"
        f"Candidate ledger: {len(ledger):,} frames; file SHA-256 `{metadata['ledger_file_sha256']}`. Categories: {metadata['categories']}. "
        "Missing dates are unknown and excluded. Zero-recorded does not mean physically lightning-free.\n\n"
        "## Preliminary split and storm groups\n\n"
        f"Centroid DBSCAN produced {storm_meta['derived_storm_count']:,} derived analytical groups; none crosses a split. Split counts: {metadata['splits']}. "
        "These are not official storm identifiers.\n\n## Pilot selection\n\n"
        f"Selected {len(pilot)} frames: {metadata['pilot_categories']}; splits {metadata['pilot_splits']}; "
        f"{metadata['pilot_cached_frames']} cached and {metadata['pilot_download_frames']} requiring download. "
        "Selection is deterministic and stratified across category, split, month, and local-time block; it does not use earliest-frame selection.\n",
        encoding="utf-8")
    return ledger, pilot, metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/v2_minimum.yaml"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    _, _, metadata = create_frame_plan(args.config)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()



