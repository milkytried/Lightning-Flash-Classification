"""
Build a real occurrence dataset from lightning strike logs.

This dataset targets occurrence prediction with genuinely independent features:
- spatial coordinates (grid cell centers)
- time-derived features (month, hour, day-of-year, season)

Labels are built on space-time cells:
- label=1 if >=1 strike happened in the cell during the time bin
- label=0 if no strike happened in that cell during the time bin

Negative samples are real no-strike cells sampled from the same observed time bins.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class GridConfig:
    lat_min: float = 0.5
    lat_max: float = 7.0
    lon_min: float = 99.0
    lon_max: float = 105.0
    lat_step: float = 0.25
    lon_step: float = 0.25


def scan_lightning_csvs(data_root: Path) -> pd.DataFrame:
    """Scan all CSVs and return normalized strike records."""
    csv_files = list(data_root.rglob("raw data all.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No strike CSV files found under: {data_root}")

    frames: list[pd.DataFrame] = []
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
        except Exception:
            continue
        frames.append(df)

    if not frames:
        raise RuntimeError("CSV scan completed but no valid strike tables were parsed")

    strikes = pd.concat(frames, ignore_index=True)
    required = ["Date/Time", "Latitude", "Longitude"]
    missing = [c for c in required if c not in strikes.columns]
    if missing:
        raise ValueError(f"Strike CSV missing required columns: {missing}")

    strikes = strikes[required].copy()
    strikes["Date/Time"] = pd.to_datetime(strikes["Date/Time"], errors="coerce")
    strikes["Latitude"] = pd.to_numeric(strikes["Latitude"], errors="coerce")
    strikes["Longitude"] = pd.to_numeric(strikes["Longitude"], errors="coerce")
    strikes = strikes.dropna(subset=["Date/Time", "Latitude", "Longitude"])
    return strikes


def _grid_index(values: pd.Series, vmin: float, vmax: float, step: float) -> pd.Series:
    idx = np.floor((values - vmin) / step).astype(int)
    max_idx = int(np.floor((vmax - vmin) / step)) - 1
    return idx.clip(lower=0, upper=max_idx)


def _cell_center(idx: pd.Series, vmin: float, step: float) -> pd.Series:
    return vmin + (idx + 0.5) * step


def _time_features(ts: pd.Series) -> pd.DataFrame:
    out = pd.DataFrame(index=ts.index)
    out["month"] = ts.dt.month.astype(int)
    out["hour"] = ts.dt.hour.astype(int)
    out["day_of_year"] = ts.dt.dayofyear.astype(int)
    out["season"] = ((out["month"] % 12) // 3).astype(int)
    return out


def build_occurrence_dataset(
    data_root: Path,
    output_csv: Path,
    output_stats_json: Path,
    grid: GridConfig,
    time_freq: str = "1H",
    negative_ratio: int = 3,
    seed: int = 42,
) -> pd.DataFrame:
    """Create a real occurrence dataset with sampled no-strike cells."""
    rng = np.random.default_rng(seed)
    time_freq = time_freq.replace("H", "h")

    strikes = scan_lightning_csvs(data_root)
    strikes = strikes[
        (strikes["Latitude"] >= grid.lat_min)
        & (strikes["Latitude"] <= grid.lat_max)
        & (strikes["Longitude"] >= grid.lon_min)
        & (strikes["Longitude"] <= grid.lon_max)
    ].copy()

    strikes["time_bin"] = strikes["Date/Time"].dt.floor(time_freq)
    strikes["lat_idx"] = _grid_index(strikes["Latitude"], grid.lat_min, grid.lat_max, grid.lat_step)
    strikes["lon_idx"] = _grid_index(strikes["Longitude"], grid.lon_min, grid.lon_max, grid.lon_step)

    grouped = (
        strikes.groupby(["time_bin", "lat_idx", "lon_idx"], as_index=False)
        .size()
        .rename(columns={"size": "strike_count"})
    )
    grouped["label"] = 1

    lat_bins = int(np.floor((grid.lat_max - grid.lat_min) / grid.lat_step))
    lon_bins = int(np.floor((grid.lon_max - grid.lon_min) / grid.lon_step))
    all_cells = np.array([(i, j) for i in range(lat_bins) for j in range(lon_bins)], dtype=int)

    negatives: list[dict] = []
    for time_bin, pos in grouped.groupby("time_bin"):
        pos_cells = set(zip(pos["lat_idx"].tolist(), pos["lon_idx"].tolist()))
        candidates = np.array([cell for cell in all_cells if (int(cell[0]), int(cell[1])) not in pos_cells], dtype=int)
        if candidates.size == 0:
            continue

        n_pos = len(pos)
        n_neg = min(len(candidates), negative_ratio * n_pos)
        picked = candidates[rng.choice(len(candidates), size=n_neg, replace=False)]

        for lat_idx, lon_idx in picked:
            negatives.append(
                {
                    "time_bin": time_bin,
                    "lat_idx": int(lat_idx),
                    "lon_idx": int(lon_idx),
                    "strike_count": 0,
                    "label": 0,
                }
            )

    neg_df = pd.DataFrame(negatives)
    full = pd.concat([grouped, neg_df], ignore_index=True)

    full["latitude"] = _cell_center(full["lat_idx"], grid.lat_min, grid.lat_step)
    full["longitude"] = _cell_center(full["lon_idx"], grid.lon_min, grid.lon_step)

    time_df = _time_features(pd.to_datetime(full["time_bin"]))
    full = pd.concat([full, time_df], axis=1)

    # Chronological split by unique time bins.
    unique_times = np.sort(full["time_bin"].unique())
    n_times = len(unique_times)
    t1 = unique_times[int(0.70 * n_times)]
    t2 = unique_times[int(0.85 * n_times)]

    full["split"] = np.where(
        full["time_bin"] < t1,
        "train",
        np.where(full["time_bin"] < t2, "val", "test"),
    )

    full = full[
        [
            "time_bin",
            "split",
            "latitude",
            "longitude",
            "month",
            "hour",
            "day_of_year",
            "season",
            "strike_count",
            "label",
        ]
    ].sort_values(["time_bin", "latitude", "longitude"]) 

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    full.to_csv(output_csv, index=False)

    stats = {
        "rows": int(len(full)),
        "positives": int((full["label"] == 1).sum()),
        "negatives": int((full["label"] == 0).sum()),
        "negative_to_positive_ratio": float((full["label"] == 0).sum() / max(1, (full["label"] == 1).sum())),
        "split_counts": {
            s: int((full["split"] == s).sum()) for s in ["train", "val", "test"]
        },
        "grid": {
            "lat_min": grid.lat_min,
            "lat_max": grid.lat_max,
            "lon_min": grid.lon_min,
            "lon_max": grid.lon_max,
            "lat_step": grid.lat_step,
            "lon_step": grid.lon_step,
        },
        "time_freq": time_freq,
        "source_rows": int(len(strikes)),
    }
    output_stats_json.parent.mkdir(parents=True, exist_ok=True)
    output_stats_json.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    return full


def parse_args():
    p = argparse.ArgumentParser(description="Build real grid-based occurrence dataset")
    p.add_argument("--data-root", type=Path, default=Path("data/raw/himawari8_pngs"))
    p.add_argument("--output-csv", type=Path, default=Path("data/processed/occurrence_dataset.csv"))
    p.add_argument("--output-stats", type=Path, default=Path("results/occurrence_dataset_stats.json"))
    p.add_argument("--time-freq", type=str, default="1H")
    p.add_argument("--negative-ratio", type=int, default=3)
    p.add_argument("--lat-min", type=float, default=0.5)
    p.add_argument("--lat-max", type=float, default=7.0)
    p.add_argument("--lon-min", type=float, default=99.0)
    p.add_argument("--lon-max", type=float, default=105.0)
    p.add_argument("--lat-step", type=float, default=0.25)
    p.add_argument("--lon-step", type=float, default=0.25)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    grid = GridConfig(
        lat_min=args.lat_min,
        lat_max=args.lat_max,
        lon_min=args.lon_min,
        lon_max=args.lon_max,
        lat_step=args.lat_step,
        lon_step=args.lon_step,
    )
    df = build_occurrence_dataset(
        data_root=args.data_root,
        output_csv=args.output_csv,
        output_stats_json=args.output_stats,
        grid=grid,
        time_freq=args.time_freq,
        negative_ratio=args.negative_ratio,
        seed=args.seed,
    )
    print(f"Saved occurrence dataset: {args.output_csv}")
    print(f"Rows: {len(df):,}")
    print(f"Positives: {(df['label'] == 1).sum():,}")
    print(f"Negatives: {(df['label'] == 0).sum():,}")


if __name__ == "__main__":
    main()
