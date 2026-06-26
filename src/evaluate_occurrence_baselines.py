"""
Evaluate clean tabular baselines on the real occurrence dataset.

This script avoids strike-derived features and reports rare-event metrics
focused on the minority class (no-strike).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler


def minority_metrics(y_true: np.ndarray, prob_pos: np.ndarray, threshold: float = 0.5) -> dict:
    pred_pos = (prob_pos >= threshold).astype(int)
    y_no = (y_true == 0).astype(int)
    pred_no = (pred_pos == 0).astype(int)
    return {
        "precision_no_strike": float(precision_score(y_no, pred_no, zero_division=0)),
        "recall_no_strike": float(recall_score(y_no, pred_no, zero_division=0)),
        "f1_no_strike": float(f1_score(y_no, pred_no, zero_division=0)),
        "pr_auc_no_strike": float(average_precision_score(y_no, 1.0 - prob_pos)),
        "support_no_strike": int(y_no.sum()),
        "support_strike": int((y_true == 1).sum()),
    }


def evaluate(dataset_csv: Path, out_json: Path) -> dict:
    df = pd.read_csv(dataset_csv)
    required = {"split", "label", "latitude", "longitude", "month", "hour", "day_of_year", "season"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset missing required columns: {sorted(missing)}")

    feat_cols = ["latitude", "longitude", "month", "hour", "day_of_year", "season"]

    train_df = df[df["split"] == "train"].copy()
    val_df = df[df["split"] == "val"].copy()
    test_df = df[df["split"] == "test"].copy()

    x_train = train_df[feat_cols].to_numpy(dtype=float)
    y_train = train_df["label"].to_numpy(dtype=int)
    x_val = val_df[feat_cols].to_numpy(dtype=float)
    y_val = val_df["label"].to_numpy(dtype=int)
    x_test = test_df[feat_cols].to_numpy(dtype=float)
    y_test = test_df["label"].to_numpy(dtype=int)

    scaler = StandardScaler()
    x_train_s = scaler.fit_transform(x_train)
    x_val_s = scaler.transform(x_val)
    x_test_s = scaler.transform(x_test)

    model = LogisticRegression(max_iter=1000)
    model.fit(x_train_s, y_train)

    val_prob = model.predict_proba(x_val_s)[:, 1]
    test_prob = model.predict_proba(x_test_s)[:, 1]

    val_metrics = minority_metrics(y_val, val_prob, threshold=0.5)
    test_metrics = minority_metrics(y_test, test_prob, threshold=0.5)

    baseline_always_strike = {
        "accuracy": float((y_test == 1).mean()),
        "test_no_strike_base_rate": float((y_test == 0).mean()),
        "no_strike_precision": 0.0,
        "no_strike_recall": 0.0,
        "no_strike_f1": 0.0,
        "no_strike_pr_auc": float((y_test == 0).mean()),
    }

    result = {
        "dataset": str(dataset_csv).replace("\\", "/"),
        "features": feat_cols,
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
        "always_predict_strike_baseline": baseline_always_strike,
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate clean occurrence baselines")
    p.add_argument("--dataset-csv", type=Path, default=Path("data/processed/occurrence_dataset.csv"))
    p.add_argument("--output-json", type=Path, default=Path("results/occurrence_baseline_metrics.json"))
    return p.parse_args()


def main():
    args = parse_args()
    result = evaluate(args.dataset_csv, args.output_json)
    print("Saved baseline metrics:", args.output_json)
    print("Validation no-strike metrics:", result["validation_metrics"])
    print("Test no-strike metrics:", result["test_metrics"])


if __name__ == "__main__":
    main()
