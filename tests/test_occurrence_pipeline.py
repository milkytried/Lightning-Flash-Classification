import json
from pathlib import Path

import pandas as pd
import pytest

from src.build_occurrence_dataset import GridConfig, build_occurrence_dataset
from src.evaluate_occurrence_baselines import evaluate


def _write_strike_csv(csv_path: Path):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {"Date/Time": "2023-01-01 00:10:00", "Latitude": 2.0, "Longitude": 101.0},
        {"Date/Time": "2023-01-01 01:15:00", "Latitude": 2.2, "Longitude": 101.2},
        {"Date/Time": "2023-01-02 03:05:00", "Latitude": 3.1, "Longitude": 102.3},
        {"Date/Time": "2023-01-03 04:30:00", "Latitude": 4.0, "Longitude": 103.0},
        {"Date/Time": "2023-01-04 12:45:00", "Latitude": 2.8, "Longitude": 101.8},
        {"Date/Time": "2023-01-05 17:50:00", "Latitude": 5.0, "Longitude": 104.0},
    ]
    pd.DataFrame(rows).to_csv(csv_path, index=False)


def test_occurrence_dataset_split_is_strictly_chronological(tmp_path):
    data_root = tmp_path / "data/raw/himawari8_pngs"
    csv_path = data_root / "2023" / "JAN" / "1" / "raw data all.csv"
    _write_strike_csv(csv_path)

    output_csv = tmp_path / "occurrence_dataset.csv"
    output_stats = tmp_path / "occurrence_stats.json"

    grid = GridConfig(
        lat_min=0.5,
        lat_max=6.0,
        lon_min=99.5,
        lon_max=104.5,
        lat_step=0.5,
        lon_step=0.5,
    )

    df = build_occurrence_dataset(
        data_root=data_root,
        output_csv=output_csv,
        output_stats_json=output_stats,
        grid=grid,
        time_freq="1H",
        negative_ratio=2,
        seed=42,
    )

    assert len(df) > 0
    assert output_csv.exists()
    assert output_stats.exists()

    train_times = pd.to_datetime(df[df["split"] == "train"]["time_bin"])
    val_times = pd.to_datetime(df[df["split"] == "val"]["time_bin"])
    test_times = pd.to_datetime(df[df["split"] == "test"]["time_bin"])

    if len(train_times) and len(val_times):
        assert train_times.max() < val_times.min()
    if len(val_times) and len(test_times):
        assert val_times.max() < test_times.min()


def test_occurrence_baseline_excludes_circular_columns(tmp_path):
    # Build a tiny valid occurrence dataset table directly for baseline eval.
    df = pd.DataFrame(
        [
            {"time_bin": "2023-01-01 00:00:00", "split": "train", "latitude": 2.0, "longitude": 101.0, "month": 1, "hour": 0, "day_of_year": 1, "season": 0, "strike_count": 1, "label": 1},
            {"time_bin": "2023-01-01 01:00:00", "split": "train", "latitude": 2.5, "longitude": 101.5, "month": 1, "hour": 1, "day_of_year": 1, "season": 0, "strike_count": 0, "label": 0},
            {"time_bin": "2023-01-02 00:00:00", "split": "val", "latitude": 3.0, "longitude": 102.0, "month": 1, "hour": 0, "day_of_year": 2, "season": 0, "strike_count": 1, "label": 1},
            {"time_bin": "2023-01-02 01:00:00", "split": "val", "latitude": 3.5, "longitude": 102.5, "month": 1, "hour": 1, "day_of_year": 2, "season": 0, "strike_count": 0, "label": 0},
            {"time_bin": "2023-01-03 00:00:00", "split": "test", "latitude": 4.0, "longitude": 103.0, "month": 1, "hour": 0, "day_of_year": 3, "season": 0, "strike_count": 1, "label": 1},
            {"time_bin": "2023-01-03 01:00:00", "split": "test", "latitude": 4.5, "longitude": 103.5, "month": 1, "hour": 1, "day_of_year": 3, "season": 0, "strike_count": 0, "label": 0},
        ]
    )

    dataset_csv = tmp_path / "occurrence_dataset.csv"
    metrics_json = tmp_path / "occurrence_baseline_metrics.json"
    df.to_csv(dataset_csv, index=False)

    result = evaluate(dataset_csv, metrics_json)
    assert metrics_json.exists()

    used_features = set(result["features"])
    forbidden = {"amplitude", "strike_type", "strike_count", "label"}
    assert used_features.isdisjoint(forbidden)

    parsed = json.loads(metrics_json.read_text(encoding="utf-8"))
    assert "test_metrics" in parsed
    assert "pr_auc_no_strike" in parsed["test_metrics"]
