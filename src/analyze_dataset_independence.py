"""Quantify frame-, date-, and spatial-cluster independence in the patch manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN


EARTH_RADIUS_KM = 6371.0088


def distribution(values: pd.Series | np.ndarray) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    if not len(array):
        raise ValueError("Cannot summarize an empty distribution")
    q1, median, q3 = np.quantile(array, [0.25, 0.5, 0.75])
    return {
        "count": int(len(array)),
        "median": float(median),
        "q1": float(q1),
        "q3": float(q3),
        "iqr": float(q3 - q1),
        "max": int(np.max(array)),
        "mean": float(np.mean(array)),
    }


def count_spatial_clusters(
    positive_rows: pd.DataFrame,
    eps_km: float = 20.0,
) -> tuple[int, dict[str, int]]:
    """Count within-frame haversine DBSCAN clusters overall and by split."""

    cluster_counts = {split: 0 for split in sorted(positive_rows["split"].unique())}
    total = 0
    for (split, _frame_id), frame_rows in positive_rows.groupby(
        ["split", "frame_id"], sort=False
    ):
        coordinates = frame_rows[["lat", "lon"]].to_numpy(dtype=np.float64)
        labels = DBSCAN(
            eps=eps_km / EARTH_RADIUS_KM,
            min_samples=1,
            metric="haversine",
            algorithm="ball_tree",
        ).fit_predict(np.radians(coordinates))
        count = int(np.unique(labels).size)
        cluster_counts[str(split)] += count
        total += count
    return total, cluster_counts


def analyze_manifest(manifest: pd.DataFrame, eps_km: float = 20.0) -> dict[str, Any]:
    required = {"label", "split", "timestamp", "frame_id", "lat", "lon"}
    missing = sorted(required.difference(manifest.columns))
    if missing:
        raise ValueError(f"Dataset manifest is missing required columns: {missing}")

    rows = manifest.copy()
    rows["split"] = rows["split"].astype(str).str.casefold()
    rows["timestamp_utc"] = pd.to_datetime(
        rows["timestamp"], utc=True, errors="coerce"
    )
    if rows["timestamp_utc"].isna().any():
        raise ValueError("Dataset manifest contains invalid timestamps")
    if rows["frame_id"].isna().any():
        raise ValueError("Dataset manifest contains missing frame identifiers")

    mixed_frames = rows.groupby("frame_id")["split"].nunique()
    if (mixed_frames > 1).any():
        raise ValueError("A source frame occurs in more than one dataset split")

    positive_rows = rows[rows["label"].astype(int) == 1].copy()
    coordinates = positive_rows[["lat", "lon"]].to_numpy(dtype=np.float64)
    if not np.isfinite(coordinates).all():
        raise ValueError("Positive manifest rows contain missing or non-finite coordinates")

    splits = sorted(rows["split"].unique())
    patches_per_frame = rows.groupby("frame_id").size()
    positive_per_frame = positive_rows.groupby("frame_id").size()
    total_clusters, clusters_per_split = count_spatial_clusters(
        positive_rows, eps_km=eps_km
    )

    per_split: dict[str, Any] = {}
    for split in splits:
        split_rows = rows[rows["split"] == split]
        split_positive = positive_rows[positive_rows["split"] == split]
        per_split[split] = {
            "patches": int(len(split_rows)),
            "positive_patches": int(len(split_positive)),
            "distinct_source_frames": int(split_rows["frame_id"].nunique()),
            "distinct_dates": int(split_rows["timestamp_utc"].dt.date.nunique()),
            "patches_per_frame": distribution(
                split_rows.groupby("frame_id").size()
            ),
            "positive_distinct_frames": int(
                split_positive["frame_id"].nunique()
            ),
            "positives_per_positive_frame": distribution(
                split_positive.groupby("frame_id").size()
            ),
            "positive_spatial_clusters": int(
                clusters_per_split.get(split, 0)
            ),
        }

    distinct_dates = int(rows["timestamp_utc"].dt.date.nunique())
    positive_frames = int(positive_rows["frame_id"].nunique())
    positive_patches = int(len(positive_rows))
    return {
        "method": {
            "source_frame_column": "frame_id",
            "date_source_column": "timestamp",
            "positive_coordinate_columns": ["lat", "lon"],
            "spatial_clustering": (
                "DBSCAN within each source frame using haversine distance"
            ),
            "dbscan_eps_km": float(eps_km),
            "dbscan_min_samples": 1,
            "earth_radius_km": EARTH_RADIUS_KM,
            "quartile_method": "NumPy linear quantiles",
            "interpretation": (
                "Spatial clusters are a coarse within-frame proxy for "
                "convective events, not tracked meteorological storm objects "
                "across time."
            ),
        },
        "overall": {
            "patches": int(len(rows)),
            "positive_patches": positive_patches,
            "distinct_source_frames": int(rows["frame_id"].nunique()),
            "distinct_dates": distinct_dates,
            "patches_per_frame": distribution(patches_per_frame),
            "positive_distinct_frames": positive_frames,
            "positives_per_positive_frame": distribution(positive_per_frame),
            "positive_spatial_clusters": int(total_clusters),
        },
        "per_split": per_split,
        "viva_headline": (
            f"The {positive_patches:,} positive patches derive from "
            f"{positive_frames:,} distinct source frames across "
            f"{distinct_dates:,} dates, representing approximately "
            f"{total_clusters:,} spatially distinct within-frame convective "
            "clusters at a 20 km DBSCAN scale."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-csv",
        type=Path,
        default=Path("data/processed/satellite_dataset.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/dataset_independence.json"),
    )
    parser.add_argument("--eps-km", type=float, default=20.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.eps_km <= 0:
        raise ValueError("--eps-km must be positive")
    manifest = pd.read_csv(args.dataset_csv)
    result = analyze_manifest(manifest, eps_km=args.eps_km)
    result["dataset_manifest"] = str(args.dataset_csv)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    print(result["viva_headline"])
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
