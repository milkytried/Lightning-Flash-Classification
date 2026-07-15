from pathlib import Path

import pandas as pd
import yaml

from src.plan_v2_frames import split_for_timestamp, stratified_pick


def config():
    return yaml.safe_load(Path("configs/v2_minimum.yaml").read_text(encoding="utf-8"))


def test_chronological_split_prefers_later_test_dates():
    cfg = config()
    assert split_for_timestamp(pd.Timestamp("2024-01-01T00:00:00Z"), cfg) == "train"
    assert split_for_timestamp(pd.Timestamp("2025-01-15T00:00:00Z"), cfg) == "val"
    assert split_for_timestamp(pd.Timestamp("2025-03-15T00:00:00Z"), cfg) == "test"


def test_stratified_selection_is_deterministic_and_cache_first():
    rows = pd.DataFrame({"month": [1, 1, 2, 2], "malaysia_local_hour": [0, 8, 0, 8],
                         "satellite_files_cached": [True, False, True, False], "value": range(4)})
    first = stratified_pick(rows, 2, seed=42, prefer_cached=True)
    second = stratified_pick(rows, 2, seed=42, prefer_cached=True)
    assert first.value.tolist() == second.value.tolist()
    assert first.satellite_files_cached.all()


def test_artifact_ledger_has_no_storm_or_frame_split_overlap():
    path = Path("data/processed/v2/pilot/frame_ledger.csv")
    if not path.exists():
        return
    ledger = pd.read_csv(path)
    assert (ledger.groupby("storm_id").split.nunique() == 1).all()
    assert (ledger.groupby("frame_timestamp_utc").split.nunique() == 1).all()
    assert set(ledger.split) == {"train", "val", "test"}
