"""Plan the science-first full Phase 2 dataset before considering cache availability."""

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
    value["full_build"]["configuration_hash"] = None
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
    """Vectorized equivalent of the frozen Phase 1B greedy diversity score."""
    rng = np.random.default_rng(seed); source = candidates.copy(); source["_random"] = rng.random(len(source))
    top = source.sort_values(["retained_positive_count", "_random"], ascending=[False, True]).head(max(count * 15, 1000))
    rows = pd.concat([top, source[source.satellite_files_cached]]).drop_duplicates("frame_timestamp_utc").reset_index(drop=True)
    dates, storms, strata = Counter(), Counter(), Counter(); available=np.ones(len(rows),dtype=bool); chosen=[]
    date_values=rows.date.astype(str).to_numpy(); storm_values=rows.storm_id.astype(str).to_numpy(); months=rows.month.to_numpy(); blocks=rows.local_time_block.astype(str).to_numpy()
    retained=np.minimum(rows.retained_positive_count.to_numpy(float),10); cached=rows.satellite_files_cached.to_numpy(bool); random_values=rows._random.to_numpy(float)
    for _ in range(min(count,len(rows))):
        date_new=np.fromiter((dates[item]==0 for item in date_values),dtype=float,count=len(rows)); storm_new=np.fromiter((storms[item]==0 for item in storm_values),dtype=float,count=len(rows))
        stratum_counts=np.fromiter((strata[(month,block)] for month,block in zip(months,blocks)),dtype=float,count=len(rows))
        scores=120*date_new+80*storm_new-18*stratum_counts+3*retained+10*cached+random_values; scores[~available]=-np.inf
        pos=int(np.argmax(scores)); available[pos]=False; chosen.append(pos); dates[date_values[pos]]+=1; storms[storm_values[pos]]+=1; strata[(months[pos],blocks[pos])]+=1
    return rows.iloc[chosen].drop(columns="_random")

def select_zero(candidates: pd.DataFrame, active: pd.DataFrame, count: int, seed: int) -> pd.DataFrame:
    """Vectorized month/time matching; cache remains only a tie-breaker."""
    rng=np.random.default_rng(seed); source=candidates.copy(); source["_random"]=rng.random(len(source))
    random_pool=source.sort_values("_random").head(max(count*30,1500)); rows=pd.concat([random_pool,source[source.satellite_files_cached]]).drop_duplicates("frame_timestamp_utc").reset_index(drop=True)
    target=Counter(zip(active.month,active.local_time_block)); selected=Counter(); selected_dates=set(); available=np.ones(len(rows),dtype=bool); chosen=[]
    months=rows.month.to_numpy(); blocks=rows.local_time_block.astype(str).to_numpy(); dates=rows.date.astype(str).to_numpy(); cached=rows.satellite_files_cached.to_numpy(bool); random_values=rows._random.to_numpy(float)
    for _ in range(min(count,len(rows))):
        deficit=np.fromiter((target[(m,b)]/max(len(active),1)-selected[(m,b)]/max(count,1) for m,b in zip(months,blocks)),dtype=float,count=len(rows))
        date_new=np.fromiter((item not in selected_dates for item in dates),dtype=float,count=len(rows)); scores=100*deficit+20*date_new+10*cached+random_values; scores[~available]=-np.inf
        pos=int(np.argmax(scores)); available[pos]=False; chosen.append(pos); selected[(months[pos],blocks[pos])]+=1; selected_dates.add(dates[pos])
    return rows.iloc[chosen].drop(columns="_random")

def plan(config_path: Path) -> tuple[pd.DataFrame, dict]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    phase = config["full_build"]; cfg_hash = config_hash(config)
    if phase["configuration_hash"] not in {"PENDING", cfg_hash}:
        raise ValueError("Configuration hash does not match canonical Phase 2 full configuration")
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
        active = select_active(active_pool[active_pool.split.eq(split)], int(quotas["active"]), int(phase["seeds"]["frame_selection"]) + offset)
        zero = select_zero(ledger[ledger.split.eq(split) & ledger.category.eq("zero_recorded")], active,
                           int(quotas["zero_recorded"]), int(phase["seeds"]["frame_selection"]) + 20 + offset)
        zero["retained_positive_count"] = 0; zero["deduplicated_positive_count"] = 0; zero["spacing_removed_positive_count"] = 0
        selected.extend([active, zero])
    frozen = pd.concat(selected, ignore_index=True).sort_values("frame_timestamp_utc").reset_index(drop=True)
    frozen["phase"] = "V2 Phase 2 full dataset"
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
            if split_summary[split]["active_dates"] < phase["decision_rules"]["minimum_active_dates_validation_test"]: blockers.append(f"{split} active dates below minimum")
            if split_summary[split]["storm_groups"] < phase["decision_rules"]["minimum_storm_groups_validation_test"]: blockers.append(f"{split} storms below minimum")
            if split_summary[split]["planned_positive_capacity"] < 500: blockers.append(f"{split} positive capacity below 500")
            if split_summary[split]["active_frames"] < 75: blockers.append(f"{split} active frames below 75")
    metadata = {"phase": "V2 Phase 2 full dataset", "configuration_hash": cfg_hash, "ledger_content_hash": content_hash,
                "ledger_file_sha256": sha256_file(out), "frames": len(frozen), "split_summary": split_summary,
                "category_counts": frozen.category.value_counts().to_dict(), "cache_checked_after_selection": True,
                "download_frames": int((~frozen.satellite_files_cached).sum()), "blockers": blockers,
                "positive_spacing_km": spacing, "positive_cap_per_frame": cap,
                "study_mask_statement": config["description"], "storm_ids_official": False}
    Path(phase["outputs"]["frame_ledger_json"]).write_text(json.dumps({"metadata": metadata, "records": json.loads(frozen.to_json(orient="records", date_format="iso"))}, indent=2) + "\n", encoding="utf-8")
    report = "# Version 2 Phase 2 full frame plan\n\nThe temporal label rule was frozen before sampling: positive `[t,t+10m)`, negative exclusion `[t-20m,t+30m)`, full 64x64 crop plus 10 km margin. The frame list is frozen before cache/download decisions.\n\n"
    report += f"Configuration hash: `{cfg_hash}`. Frozen ledger SHA-256: `{metadata['ledger_file_sha256']}`. Frames: {len(frozen)}; categories {metadata['category_counts']}.\n\n"
    report += "## Split support\n\n" + "\n".join(f"- {key}: {value}" for key, value in split_summary.items()) + "\n\n"
    report += f"Cache was checked only after science-first selection. {metadata['download_frames']} desired frames require download. Blockers: {blockers or 'none'}. Derived storm groups are not official identifiers.\n"
    Path("report/V2_FULL_FRAME_PLAN.md").write_text(report, encoding="utf-8")
    Path("report/V2_FULL_FRAME_PLAN.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return frozen, metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--config", type=Path, default=Path("configs/v2_full.yaml")); args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    _, metadata = plan(args.config); print(json.dumps(metadata, indent=2))


if __name__ == "__main__": main()
