"""Validate an aligned Himawari satellite patch dataset.

This is intended to be run after a clean rebuild to confirm that generated
patches are balanced, chronological, loadable, and free of black/no-data leakage.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from build_satellite_dataset import chronological_split
from himawari_data_loader import create_himawari_loaders


def black_fraction(path: Path, black_threshold: int) -> tuple[float, bool]:
    arr = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)
    nonfinite = not np.isfinite(arr).all()
    black = np.all(arr < black_threshold, axis=2)
    return float(black.mean()), nonfinite


def summarize_black(scan: pd.DataFrame, label: int, name: str) -> None:
    values = scan.loc[scan["label"] == label, "black_fraction"]
    print(f"\n{name}")
    print("count", int(values.count()))
    print("mean", f"{values.mean():.6f}")
    print("median", f"{values.median():.6f}")
    print("max", f"{values.max():.6f}")
    print("gt_02pct", int((values > 0.02).sum()))
    print("gt_30pct", int((values > 0.30).sum()))
    print("gt_50pct", int((values > 0.50).sum()))
    print("exact_100pct", int((values == 1.0).sum()))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="data/processed/satellite_dataset.csv")
    parser.add_argument("--black-threshold", type=int, default=8)
    parser.add_argument("--max-black-fraction", type=float, default=0.02)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")

    manifest = pd.read_csv(manifest_path)
    required = {"path", "label", "split", "timestamp", "frame_id"}
    missing_columns = required - set(manifest.columns)
    if missing_columns:
        raise ValueError(f"Manifest missing required columns: {sorted(missing_columns)}")

    rows: list[dict[str, object]] = []
    missing_paths: list[str] = []
    nonfinite_count = 0
    for _, row in manifest.iterrows():
        patch_path = Path(row["path"])
        if not patch_path.exists():
            missing_paths.append(str(patch_path))
            continue
        fraction, nonfinite = black_fraction(patch_path, args.black_threshold)
        if nonfinite:
            nonfinite_count += 1
        rows.append({"label": int(row["label"]), "split": row["split"], "black_fraction": fraction})

    scan = pd.DataFrame(rows)
    print("rows", len(manifest))
    print("scanned", len(scan))
    print("missing_paths", len(missing_paths))
    print("nonfinite_pngs", nonfinite_count)
    if missing_paths:
        print("first_missing_paths", missing_paths[:5])

    summarize_black(scan, 0, "NEGATIVE")
    summarize_black(scan, 1, "POSITIVE")

    print("\nlabel balance by split")
    print(manifest.groupby(["split", "label"]).size().to_string())

    manifest = manifest.copy()
    manifest["ts"] = pd.to_datetime(manifest["timestamp"], utc=True)
    manifest["date"] = manifest["ts"].dt.date
    expected_splits = manifest["ts"].map(chronological_split)
    split_mismatches = int((manifest["split"].astype(str) != expected_splits).sum())
    print("\nsplit_mismatches", split_mismatches)

    print("\ndate ranges")
    print(manifest.groupby("split")["ts"].agg(["min", "max", "count"]).to_string())

    date_sets = {split: set(part["date"]) for split, part in manifest.groupby("split")}
    overlaps: dict[str, int] = {}
    split_names = sorted(date_sets)
    for index, split_a in enumerate(split_names):
        for split_b in split_names[index + 1 :]:
            key = f"date_overlap_{split_a}_{split_b}"
            overlaps[key] = len(date_sets[split_a] & date_sets[split_b])
            print(key, overlaps[key])

    loaders = create_himawari_loaders(
        str(manifest_path),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    print("\nloader_batches", {name: len(loader) for name, loader in loaders.items()})
    print("loader_samples", {name: len(loader.dataset) for name, loader in loaders.items()})

    too_black = int((scan["black_fraction"] > args.max_black_fraction).sum())
    errors = []
    if missing_paths:
        errors.append(f"{len(missing_paths)} missing patch paths")
    if nonfinite_count:
        errors.append(f"{nonfinite_count} PNGs contain non-finite values")
    if split_mismatches:
        errors.append(f"{split_mismatches} split mismatches")
    overlap_total = sum(overlaps.values())
    if overlap_total:
        errors.append(f"{overlap_total} overlapping split dates")
    if too_black:
        errors.append(f"{too_black} patches exceed black fraction {args.max_black_fraction}")

    if errors:
        raise SystemExit("VALIDATION FAILED: " + "; ".join(errors))
    print("\nVALIDATION PASSED")


if __name__ == "__main__":
    main()
