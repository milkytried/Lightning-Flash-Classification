import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest
import yaml


CONFIG = Path("configs/v2_full.yaml")
LEDGER = Path("data/processed/v2/full/frame_ledger.csv")
MANIFEST = Path("data/processed/v2/full/manifest.csv")


def test_full_configuration_hash_is_frozen_and_valid():
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    recorded = config["full_build"]["configuration_hash"]
    config["full_build"]["configuration_hash"] = None
    actual = hashlib.sha256(
        json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert actual == recorded


def test_full_ledger_is_frozen_and_split_independent():
    if not LEDGER.exists():
        pytest.skip("full Phase 2 frame ledger is generated locally and not committed")
    ledger = pd.read_csv(LEDGER)
    assert len(ledger) == 1200
    assert ledger.frame_timestamp_utc.is_unique
    assert (ledger.groupby("date").split.nunique() == 1).all()
    assert (ledger.groupby("storm_id").split.nunique() == 1).all()
    sources = {split: set(";".join(part.required_noaa_object_keys.astype(str)).split(";")) for split, part in ledger.groupby("split")}
    assert not (sources["train"] & sources["val"])
    assert not (sources["train"] & sources["test"])
    assert not (sources["val"] & sources["test"])


def test_full_manifest_has_no_duplicate_crop_within_or_across_splits():
    if not MANIFEST.exists():
        pytest.skip("full manifest has not been built yet")
    frame = pd.read_csv(MANIFEST)
    key = ["frame_id", "x", "y", "crop_x0", "crop_y0", "crop_x1", "crop_y1", "label"]
    assert not frame.duplicated(key).any()
    bounds = ["frame_id", "crop_x0", "crop_y0", "crop_x1", "crop_y1"]
    assert not frame.duplicated(bounds).any()



def test_full_required_auxiliary_ledgers_exist_with_protocol_columns():
    if not LEDGER.exists():
        pytest.skip("full Phase 2 generated ledgers are not present in CI")
    required = {
        Path("data/processed/v2/full/noaa_object_inventory.csv"): {
            "frame_timestamp_utc",
            "frame_id",
            "split",
            "category",
            "required_noaa_object_key",
            "required_local_path",
            "exists_on_disk",
            "configuration_hash",
        },
        Path("data/processed/v2/full/download_ledger.csv"): {
            "frame_number",
            "timestamp",
            "new_files",
            "new_bytes",
            "status",
        },
        Path("data/processed/v2/full/excluded_frames.csv"): {
            "frame_number",
            "timestamp",
            "frame_id",
            "exclusion_reason",
            "exclusion_stage",
        },
    }
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        pytest.skip("Phase 2 auxiliary ledgers have not been materialized yet: " + ", ".join(missing))
    for path, columns in required.items():
        frame = pd.read_csv(path)
        assert columns.issubset(frame.columns), path
