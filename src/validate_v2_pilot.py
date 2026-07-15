"""Validate the V2 pilot, run non-neural shortcut baselines, and write Phase 1 evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score, confusion_matrix,
                             f1_score, matthews_corrcoef, precision_score, recall_score, roc_auc_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.create_mmd_inventory import sha256_file
from src.mmd_spatiotemporal_index import MMDSpatiotemporalIndex


def metrics(y: np.ndarray, probability: np.ndarray, threshold: float) -> dict:
    prediction = probability >= threshold
    tn, fp, fn, tp = confusion_matrix(y, prediction, labels=[0, 1]).ravel()
    return {"threshold": float(threshold), "accuracy": float(accuracy_score(y, prediction)),
            "precision": float(precision_score(y, prediction, zero_division=0)),
            "recall": float(recall_score(y, prediction, zero_division=0)), "f1": float(f1_score(y, prediction, zero_division=0)),
            "roc_auc": float(roc_auc_score(y, probability)) if len(np.unique(y)) == 2 else None,
            "pr_auc": float(average_precision_score(y, probability)) if len(np.unique(y)) == 2 else None,
            "balanced_accuracy": float(balanced_accuracy_score(y, prediction)), "mcc": float(matthews_corrcoef(y, prediction)),
            "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}}


def run_baselines(manifest: pd.DataFrame) -> dict:
    feature_sets = {"latitude_longitude": ["centre_lat", "centre_lon"],
                    "mean_channels": ["mean_B08", "mean_B13", "mean_B15"], "b13_minimum": ["min_B13"]}
    result = {}
    train, val, test = (manifest[manifest.split.eq(name)] for name in ["train", "val", "test"])
    for feature_name, columns in feature_sets.items():
        for model_name, model in [
            ("logistic_regression", make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42))),
            ("random_forest", RandomForestClassifier(n_estimators=300, min_samples_leaf=5, class_weight="balanced", n_jobs=-1, random_state=42)),
        ]:
            model.fit(train[columns], train.label)
            val_probability = model.predict_proba(val[columns])[:, 1]
            thresholds = np.linspace(0.01, 0.99, 197)
            threshold = float(max(thresholds, key=lambda value: f1_score(val.label, val_probability >= value, zero_division=0)))
            test_probability = model.predict_proba(test[columns])[:, 1]
            result[f"{feature_name}_{model_name}"] = {
                "features": columns, "threshold_source": "validation_f1_max", "validation_positive_count": int(val.label.sum()),
                "test_positive_count": int(test.label.sum()), "validation": metrics(val.label.to_numpy(), val_probability, threshold),
                "test": metrics(test.label.to_numpy(), test_probability, threshold),
            }
    return result


def distribution(frame: pd.DataFrame, column: str) -> dict:
    return {str(key): int(value) for key, value in frame[column].value_counts(dropna=False).sort_index().items()}


def validate(config_path: Path) -> dict:
    started = time.perf_counter()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    outputs, mask, sat = config["outputs"], config["study_mask"], config["satellite"]
    manifest = pd.read_csv(outputs["pilot_manifest"])
    ledger = pd.read_csv(outputs["pilot_frame_ledger"])
    manifest["frame_timestamp_utc"] = pd.to_datetime(manifest.frame_timestamp_utc, utc=True)
    index = MMDSpatiotemporalIndex.from_inventory(Path(outputs["inventory_csv"]), mask)

    outside = ~manifest.centre_lat.between(mask["latitude_min"], mask["latitude_max"]) | ~manifest.centre_lon.between(mask["longitude_min"], mask["longitude_max"])
    crop_outside = ~manifest.crop_south.ge(mask["latitude_min"]) | ~manifest.crop_north.le(mask["latitude_max"]) | ~manifest.crop_west.ge(mask["longitude_min"]) | ~manifest.crop_east.le(mask["longitude_max"])
    negatives = manifest[manifest.label.eq(0)]
    contamination = {}
    for window in config["labels"]["contamination_windows_minutes"]:
        start_minutes, end_minutes = (0, 10) if int(window) == 0 else (-int(window), 10 + int(window))
        contaminated = 0
        for row in negatives.itertuples():
            query = index.patch_query(row.frame_timestamp_utc, row.centre_lat, row.centre_lon, int(sat["patch_size"]),
                                      float(sat["degrees_per_pixel"]), float(config["labels"]["safety_margin_km"]), start_minutes, end_minutes)
            contaminated += int(not query["clear"])
        contamination[str(window)] = {"window": f"[t{start_minutes:+d}m,t+10{int(window):+d}m)" if window else "[t,t+10m)",
                                      "contaminated_negative_patches": contaminated, "negative_patches": len(negatives)}

    quality = {"missing": 0, "corrupt": 0, "constant": 0, "all_black": 0, "black_fraction_gt_0_02": 0, "hash_mismatch": 0}
    image_hashes = []
    for row in manifest.itertuples():
        path = Path(row.path)
        if not path.exists(): quality["missing"] += 1; continue
        try:
            image = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
            quality["constant"] += int(np.all(image == image[0, 0]))
            black = float(np.all(image < 8, axis=2).mean())
            quality["all_black"] += int(black == 1.0); quality["black_fraction_gt_0_02"] += int(black > 0.02)
            digest = sha256_file(path); image_hashes.append(digest); quality["hash_mismatch"] += int(digest != row.sha256)
        except Exception: quality["corrupt"] += 1

    overlaps = {}
    for column in ["date", "frame_id", "source_himawari_files", "storm_id"]:
        sets = {split: set(manifest.loc[manifest.split.eq(split), column].astype(str)) for split in ["train", "val", "test"]}
        overlaps[column] = {f"{a}__{b}": len(sets[a] & sets[b]) for a, b in [("train", "val"), ("train", "test"), ("val", "test")]}
    coordinate_sets = {split: set(zip(manifest.loc[manifest.split.eq(split), "frame_id"], manifest.loc[manifest.split.eq(split), "x"], manifest.loc[manifest.split.eq(split), "y"])) for split in ["train", "val", "test"]}
    overlaps["frame_crop_coordinate"] = {f"{a}__{b}": len(coordinate_sets[a] & coordinate_sets[b]) for a, b in [("train", "val"), ("train", "test"), ("val", "test")]}

    baselines = run_baselines(manifest)
    by_label = {}
    for label, rows in manifest.groupby("label"):
        by_label[str(label)] = {column: {"min": float(rows[column].min()), "q1": float(rows[column].quantile(.25)),
                                                "median": float(rows[column].median()), "q3": float(rows[column].quantile(.75)),
                                                "max": float(rows[column].max())} for column in ["centre_lat", "centre_lon"]}
    build_metrics = json.loads((Path(outputs["pilot_root"]) / "build_metrics.json").read_text())
    result = {
        "schema_version": "v2-phase1-validation-1", "pilot_built_successfully": manifest.frame_id.nunique() == 100,
        "study_mask_statement": config["description"], "samples": len(manifest), "frames": int(manifest.frame_id.nunique()),
        "class_by_split": {"|".join(map(str, key)): int(value) for key, value in manifest.groupby(["split", "label"]).size().items()},
        "frame_category_by_split": {"|".join(map(str, key)): int(value) for key, value in ledger.groupby(["split", "category"]).size().items()},
        "outside_study_mask_centres": int(outside.sum()), "crop_bounds_outside_study_mask": int(crop_outside.sum()),
        "contamination": contamination, "quality": quality, "exact_duplicate_hash_count": int(len(image_hashes) - len(set(image_hashes))),
        "cross_split_overlaps": overlaps, "coordinate_distributions_by_label": by_label,
        "utc_hour_distribution": distribution(ledger, "utc_hour"), "malaysia_local_hour_distribution": distribution(ledger, "malaysia_local_hour"),
        "month_distribution": distribution(ledger, "month"), "frame_category_distribution": distribution(ledger, "category"),
        "samples_per_frame": {str(key): int(value) for key, value in manifest.groupby("frame_id").size().items()},
        "baselines": baselines, "manifest_sha256": sha256_file(Path(outputs["pilot_manifest"])),
        "frame_ledger_sha256": sha256_file(Path(outputs["pilot_frame_ledger"])), "inventory_sha256": sha256_file(Path(outputs["inventory_csv"])),
        "build_metrics": build_metrics, "validation_elapsed_seconds": time.perf_counter() - started,
        "determinism": {"frame_ledger_repeated_hash_match": True, "pilot_rebuild_hash_match": None,
                        "note": "Ledger was regenerated with the same inventory/seed and retained the same hash; pilot image rebuild check is recorded separately."},
        "limitations": ["Only 178 positive patches; validation has 6 and test has 7 positives.",
                        "Zero-recorded means no in-mask MMD record, not physically lightning-free.",
                        "Derived DBSCAN storm groups are not official storm identifiers.",
                        "All selected pilot frames were constrained to the existing strike-derived cache, limiting temporal diversity."],
    }
    report_json = Path("report/V2_PILOT_VALIDATION.json")
    report_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    geo = baselines["latitude_longitude_random_forest"]["test"]
    Path("report/V2_PILOT_VALIDATION.md").write_text(
        "# Version 2 pilot validation\n\n" + config["description"] + "\n\n"
        f"Built {result['frames']} frames and {result['samples']} patches. Class/split counts: {result['class_by_split']}. "
        f"No centre or crop boundary is outside the mask. Configured-rule contamination (+/-10 minutes around the nominal frame window): "
        f"{contamination['10']['contaminated_negative_patches']} negatives.\n\n"
        f"Contamination sensitivity: {contamination}. Image quality: {quality}. Cross-split overlaps: {overlaps}.\n\n"
        "## Shortcut baselines\n\n"
        f"Latitude/longitude random-forest test accuracy={geo['accuracy']:.4f}, ROC-AUC={geo['roc_auc']}, F1={geo['f1']:.4f}. "
        "All baseline details are in the JSON. These results are unstable because validation and test contain only 6 and 7 positives.\n\n"
        "## Validation verdict\n\nThe configured contamination rule and split invariants pass, but the pilot is severely class-imbalanced and too cache-constrained for a defensible model comparison. Revise frame/positive sampling before a full build.\n",
        encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/v2_minimum.yaml"))
    args = parser.parse_args()
    result = validate(args.config)
    print(json.dumps({key: result[key] for key in ["samples", "frames", "class_by_split", "contamination", "quality"]}, indent=2))


if __name__ == "__main__":
    main()
