"""Plan the science-first Phase 1B pilot before considering cache availability."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.create_mmd_inventory import sha256_file
from src.mmd_spatiotemporal_index import MMDSpatiotemporalIndex

LOGGER = logging.getLogger(__name__)


def config_hash(config: dict) -> str:
    value = json.loads(json.dumps(config))
    value["phase1b"]["configuration_hash"] = None
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(canonical).hexdigest()


def time_block(hour: int) -> str:
    return ("00:00-05:59" if hour < 6 else "06:00-11:59" if hour < 12
            else "12:00-17:59" if hour < 18 else "18:00-23:59")


def retained_positive_count(index: MMDSpatiotemporalIndex, timestamp: pd.Timestamp, spacing_km: float, cap: int) -> tuple[int, int, int]:
    bin_ns = int(pd.Timestamp(timestamp).tz_convert("UTC").value)
    indices = index.bin_indices.get(bin_ns, np.empty(0, dtype=np.int64))
    original = len(indices)
    if not original:
        return 0, 0, 0
    order = np.lexsort((index.longitudes[indices], index.latitudes[indices], index.timestamps_ns[indices]))
    indices = indices[order]
    keys = np.column_stack((index.timestamps_ns[indices], np.round(index.latitudes[indices], 5), np.round(index.longitudes[indices], 5)))
    keep = np.ones(len(indices), dtype=bool)
    keep[1:] = np.any(keys[1:] != keys[:-1], axis=1)
    indices = indices[keep]
    after_dedup = len(indices)
    retained: list[tuple[float, float]] = []
    for strike_index in indices:
        strike_lat = float(index.latitudes[strike_index]); strike_lon = float(index.longitudes[strike_index])
        if retained:
            lat = np.asarray([item[0] for item in retained]); lon = np.asarray([item[1] for item in retained])
            if float(index.haversine_km(lat, lon, strike_lat, strike_lon).min()) < spacing_km:
                continue
        retained.append((strike_lat, strike_lon))
        if len(retained) >= cap:
            break
    return len(retained), original - after_dedup, after_dedup - len(retained)


def select_active(candidates: pd.DataFrame, count: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = candidates.copy()
    rows["_random"] = rng.random(len(rows))
    top = rows.sort_values(["retained_positive_count", "_random"], ascending=[False, True]).head(max(count * 15, 1000))
    cached = rows[rows.satellite_files_cached]
    rows = pd.concat([top, cached]).drop_duplicates("frame_timestamp_utc").copy()
    chosen, dates, storms, strata = [], Counter(), Counter(), Counter()
    while len(chosen) < count and len(rows):
        def score(row):
            stratum = (row.month, row.local_time_block)
            return (120 * (dates[row.date] == 0) + 80 * (storms[row.storm_id] == 0)
                    - 18 * strata[stratum] + 3 * min(row.retained_positive_count, 10)
                    + 10.0 * bool(row.satellite_files_cached) + row._random)
        idx = max(rows.index, key=lambda value: score(rows.loc[value]))
        row = rows.loc[idx]
        chosen.append(row); dates[row.date] += 1; storms[row.storm_id] += 1; strata[(row.month, row.local_time_block)] += 1
        rows = rows.drop(idx)
    return pd.DataFrame(chosen).drop(columns="_random")


def select_zero(candidates: pd.DataFrame, active: pd.DataFrame, count: int, seed: int) -> pd.DataFrame:
    """Match active frame month/time strata; cache is only a tie-breaker."""
    rng = np.random.default_rng(seed)
    rows = candidates.copy(); rows["_random"] = rng.random(len(rows))
    random_pool = rows.sort_values("_random").head(max(count * 30, 1500))
    cached = rows[rows.satellite_files_cached]
    rows = pd.concat([random_pool, cached]).drop_duplicates("frame_timestamp_utc").copy()
    target = Counter(zip(active.month, active.local_time_block)); selected = Counter(); chosen = []
    while len(chosen) < count and len(rows):
        def score(row):
            key = (row.month, row.local_time_block)
            deficit = target[key] / max(len(active), 1) - selected[key] / max(count, 1)
            return 100 * deficit + 20 * (sum(item.date == row.date for item in chosen) == 0) + 10.0 * bool(row.satellite_files_cached) + row._random
        idx = max(rows.index, key=lambda value: score(rows.loc[value]))
        row = rows.loc[idx]; chosen.append(row); selected[(row.month, row.local_time_block)] += 1; rows = rows.drop(idx)
    return pd.DataFrame(chosen).drop(columns="_random")


def plan(config_path: Path) -> tuple[pd.DataFrame, dict]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    phase = config["phase1b"]; cfg_hash = config_hash(config)
    if phase["configuration_hash"] not in {"PENDING", cfg_hash}:
        raise ValueError("Configuration hash does not match canonical Phase 1B configuration")
    ledger = pd.read_csv(config["outputs"]["frame_ledger_csv"])
    ledger["frame_timestamp_utc"] = pd.to_datetime(ledger.frame_timestamp_utc, utc=True)
    ledger["local_time_block"] = ledger.malaysia_local_hour.map(time_block)
    index = MMDSpatiotemporalIndex.from_inventory(Path(config["outputs"]["inventory_csv"]), config["study_mask"])
    sampling, frame_cfg = phase["sampling"], phase["frames"]
    spacing, cap = float(sampling["positive_minimum_centre_spacing_km"]), int(sampling["positive_cap_per_frame"])
    minimum = int(frame_cfg["minimum_spaced_positives_per_selected_active_frame"])
    active_pool = ledger[ledger.category.eq("active") & ledger.in_mask_ground_strike_count.ge(minimum)].copy()
    counts = [retained_positive_count(index, row.frame_timestamp_utc, spacing, cap) for row in active_pool.itertuples()]
    active_pool[["retained_positive_count", "deduplicated_positive_count", "spacing_removed_positive_count"]] = counts
    active_pool = active_pool[active_pool.retained_positive_count.ge(minimum)].copy()

    selected = []
    for offset, (split, quotas) in enumerate(frame_cfg["per_split"].items()):
        active = select_active(active_pool[active_pool.split.eq(split)], int(quotas["active"]), int(config["frames"]["random_seed"]) + offset)
        zero = select_zero(ledger[ledger.split.eq(split) & ledger.category.eq("zero_recorded")], active,
                           int(quotas["zero_recorded"]), int(config["frames"]["random_seed"]) + 20 + offset)
        zero["retained_positive_count"] = 0; zero["deduplicated_positive_count"] = 0; zero["spacing_removed_positive_count"] = 0
        selected.extend([active, zero])
    frozen = pd.concat(selected, ignore_index=True).sort_values("frame_timestamp_utc").reset_index(drop=True)
    frozen["phase"] = "V2 Phase 1B"
    frozen["label_rule_version"] = config["labels"]["label_rule_version"]
    frozen["configuration_hash"] = cfg_hash
    frozen["selection_reason"] = np.where(frozen.category.eq("active"),
        "science-first positive-support/date/storm/month/local-time diversity; cache only tie-breaker",
        "month/local-time matched to selected active frames; cache only tie-breaker")
    frozen["desired_before_cache_check"] = True
    content_hash = hashlib.sha256(frozen.to_csv(index=False).encode()).hexdigest()
    frozen["frozen_ledger_content_hash"] = content_hash
    out = Path(phase["outputs"]["frame_ledger"]); out.parent.mkdir(parents=True, exist_ok=True); frozen.to_csv(out, index=False)
    split_summary = {}
    blockers = []
    for split, rows in frozen.groupby("split"):
        active = rows[rows.category.eq("active")]
        split_summary[split] = {"frames": len(rows), "active_frames": len(active), "zero_recorded_frames": int(rows.category.eq("zero_recorded").sum()),
                                "active_dates": int(active.date.nunique()), "storm_groups": int(active.storm_id.nunique()),
                                "planned_positive_capacity": int(active.retained_positive_count.sum()), "cached_frames": int(rows.satellite_files_cached.sum()),
                                "download_frames": int((~rows.satellite_files_cached).sum())}
        if split in {"val", "test"}:
            if split_summary[split]["active_dates"] < frame_cfg["minimum_active_dates_validation_test"]: blockers.append(f"{split} active dates below minimum")
            if split_summary[split]["storm_groups"] < frame_cfg["minimum_storm_groups_validation_test"]: blockers.append(f"{split} storms below minimum")
            if split_summary[split]["planned_positive_capacity"] < 100: blockers.append(f"{split} positive capacity below 100")
    metadata = {"phase": "V2 Phase 1B", "configuration_hash": cfg_hash, "ledger_content_hash": content_hash,
                "ledger_file_sha256": sha256_file(out), "frames": len(frozen), "split_summary": split_summary,
                "category_counts": frozen.category.value_counts().to_dict(), "cache_checked_after_selection": True,
                "download_frames": int((~frozen.satellite_files_cached).sum()), "blockers": blockers,
                "positive_spacing_km": spacing, "positive_cap_per_frame": cap,
                "study_mask_statement": config["description"], "storm_ids_official": False}
    Path(phase["outputs"]["frame_ledger_json"]).write_text(json.dumps({"metadata": metadata, "records": json.loads(frozen.to_json(orient="records", date_format="iso"))}, indent=2) + "\n", encoding="utf-8")
    report = "# Version 2 Phase 1B frame plan\n\nThe temporal label rule was frozen before sampling: positive `[t,t+10m)`, negative exclusion `[t-20m,t+30m)`, full 64x64 crop plus 10 km margin.\n\n"
    report += f"Configuration hash: `{cfg_hash}`. Frozen ledger SHA-256: `{metadata['ledger_file_sha256']}`. Frames: {len(frozen)}; categories {metadata['category_counts']}.\n\n"
    report += "## Split support\n\n" + "\n".join(f"- {key}: {value}" for key, value in split_summary.items()) + "\n\n"
    report += f"Cache was checked only after science-first selection. {metadata['download_frames']} desired frames require download. Blockers: {blockers or 'none'}. Derived storm groups are not official identifiers.\n"
    Path("report/V2_PHASE1B_FRAME_PLAN.md").write_text(report, encoding="utf-8")
    Path("report/V2_PHASE1B_FRAME_PLAN.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return frozen, metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--config", type=Path, default=Path("configs/v2_minimum.yaml")); args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    _, metadata = plan(args.config); print(json.dumps(metadata, indent=2))


if __name__ == "__main__": main()
