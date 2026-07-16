from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay, confusion_matrix

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.v2_phase3_common import sha256_file, write_json


def plot_confusion(labels, preds, title, path):
    cm = confusion_matrix(labels, preds, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4, 3.6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], labels=["No lightning", "Lightning"], rotation=20, ha="right")
    ax.set_yticks([0, 1], labels=["No lightning", "Lightning"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Observed")
    ax.set_title(title)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def plot_probability_hist(labels, probs, threshold, title, path):
    fig, ax = plt.subplots(figsize=(5, 3.6))
    ax.hist(probs[labels == 0], bins=30, alpha=0.65, label="No lightning", color="#4575b4")
    ax.hist(probs[labels == 1], bins=30, alpha=0.65, label="Lightning", color="#d73027")
    ax.axvline(threshold, color="black", linestyle="--", linewidth=1.2, label=f"Threshold {threshold:.3f}")
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Patch count")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def plot_reliability(labels, probs, title, path):
    fig, ax = plt.subplots(figsize=(4.5, 3.6))
    frac_pos, mean_pred = calibration_curve(labels, probs, n_bins=10, strategy="uniform")
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    ax.plot(mean_pred, frac_pos, marker="o", label="Model")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed positive fraction")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def generate_for_split(split_name, prediction_dir, output_root):
    output_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for csv_path in sorted(Path(prediction_dir).glob("*.csv")):
        frame = pd.read_csv(csv_path)
        labels = frame["label"].to_numpy(int)
        probs = frame["probability"].to_numpy(float)
        threshold = float(frame["threshold"].iloc[0])
        preds = (probs >= threshold).astype(int)
        run = csv_path.stem
        safe = run.replace("\\", "_").replace("/", "_")
        if len(np.unique(labels)) == 2:
            fig, ax = plt.subplots(figsize=(4.8, 3.8))
            RocCurveDisplay.from_predictions(labels, probs, ax=ax, name=run)
            ax.plot([0, 1], [0, 1], "k--", linewidth=1)
            ax.set_title(f"{split_name} ROC: {run}")
            fig.tight_layout()
            roc_path = output_root / f"{safe}_roc.png"
            fig.savefig(roc_path, dpi=300)
            plt.close(fig)

            fig, ax = plt.subplots(figsize=(4.8, 3.8))
            PrecisionRecallDisplay.from_predictions(labels, probs, ax=ax, name=run)
            ax.set_title(f"{split_name} precision-recall: {run}")
            fig.tight_layout()
            pr_path = output_root / f"{safe}_precision_recall.png"
            fig.savefig(pr_path, dpi=300)
            plt.close(fig)
        else:
            roc_path = pr_path = None
        cm_path = output_root / f"{safe}_confusion_matrix.png"
        hist_path = output_root / f"{safe}_probability_histogram.png"
        rel_path = output_root / f"{safe}_reliability.png"
        plot_confusion(labels, preds, f"{split_name} confusion: {run}", cm_path)
        plot_probability_hist(labels, probs, threshold, f"{split_name} probabilities: {run}", hist_path)
        plot_reliability(labels, probs, f"{split_name} reliability: {run}", rel_path)
        cm = confusion_matrix(labels, preds, labels=[0, 1])
        cm_csv = output_root / f"{safe}_confusion_matrix.csv"
        pd.DataFrame(cm, index=["observed_0", "observed_1"], columns=["predicted_0", "predicted_1"]).to_csv(cm_csv)
        for path in [roc_path, pr_path, cm_path, hist_path, rel_path, cm_csv]:
            if path is not None:
                rows.append({"split": split_name, "run": run, "artifact": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size})
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--controlled-dir", default="results/v2/phase3/controlled_test_predictions")
    parser.add_argument("--natural-dir", default="results/v2/phase3/natural_prevalence_predictions")
    parser.add_argument("--output-root", default="results/v2/phase3/figures")
    args = parser.parse_args()
    root = Path(args.output_root)
    rows = []
    rows += generate_for_split("controlled_test", args.controlled_dir, root / "controlled_test")
    rows += generate_for_split("natural_prevalence", args.natural_dir, root / "natural_prevalence")
    manifest = {"artifacts": rows, "artifact_count": len(rows)}
    write_json(root / "artifact_hashes.json", manifest)
    print(json.dumps({"artifact_count": len(rows), "hash_manifest": str(root / "artifact_hashes.json")}, indent=2))


if __name__ == "__main__":
    main()
