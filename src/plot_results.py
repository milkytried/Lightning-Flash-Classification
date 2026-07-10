"""Generate report-ready plots for the frozen satellite CNN run.

The script reads finalized metrics from JSON and loads the saved checkpoint only
for read-only inference on the held-out test split to recover probabilities.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, roc_curve

from himawari_data_loader import create_himawari_loaders
from train_satellite import SatelliteTrainer


FIGURE_SPECS = {
    "training_curves.png": "Training and validation loss by epoch.",
    "validation_metrics.png": "Validation accuracy and ROC-AUC by epoch.",
    "confusion_matrix.png": "Held-out test confusion matrix at the validation-selected threshold.",
    "roc_curve.png": "Held-out test ROC curve from saved-checkpoint inference.",
    "baseline_comparison.png": "Old frozen-backbone baseline versus new aligned-dataset frozen model.",
    "meteorological_metrics.png": "Meteorological verification metrics on the held-out test split.",
    "probability_histogram.png": "Held-out test probability distribution by true class.",
}


def require_file(path: Path, label: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"Expected {label} to be a file: {path}")
    return path


def load_metrics(path: Path) -> dict[str, Any]:
    require_file(path, "metrics JSON")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def get_history(metrics: dict[str, Any]) -> dict[str, list[float]]:
    history = metrics.get("history")
    if not isinstance(history, dict):
        raise ValueError("Metrics JSON does not contain a history object")
    required = [
        "epoch",
        "train_loss",
        "val_loss",
        "val_accuracy_at_0_5",
        "val_roc_auc",
    ]
    missing = [key for key in required if key not in history]
    if missing:
        raise ValueError(f"Metrics history is missing required keys: {missing}")
    return history


def get_test_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    test_metrics = metrics.get("test_metrics_at_frozen_validation_threshold")
    if not isinstance(test_metrics, dict):
        raise ValueError("Metrics JSON does not contain test_metrics_at_frozen_validation_threshold")
    return test_metrics


def plot_training_curves(metrics: dict[str, Any], output_dir: Path) -> None:
    history = get_history(metrics)
    epochs = np.asarray(history["epoch"], dtype=int)
    train_loss = np.asarray(history["train_loss"], dtype=float)
    val_loss = np.asarray(history["val_loss"], dtype=float)
    best_index = int(np.argmin(val_loss))
    best_epoch = int(epochs[best_index])

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.plot(epochs, train_loss, marker="o", label="Train loss")
    ax.plot(epochs, val_loss, marker="s", label="Validation loss")
    ax.axvline(best_epoch, color="tab:green", linestyle="--", linewidth=1.5, label=f"Best epoch {best_epoch}")
    ax.scatter([best_epoch], [val_loss[best_index]], color="tab:green", zorder=5)
    ax.set_title("Frozen Satellite CNN Training Curves")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.25)
    ax.legend()
    save_figure(fig, output_dir / "training_curves.png")


def plot_validation_metrics(metrics: dict[str, Any], output_dir: Path) -> None:
    history = get_history(metrics)
    epochs = np.asarray(history["epoch"], dtype=int)
    val_acc = np.asarray(history["val_accuracy_at_0_5"], dtype=float)
    val_auc = np.asarray(history["val_roc_auc"], dtype=float)

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.plot(epochs, val_auc, marker="o", label="Validation ROC-AUC")
    ax.plot(epochs, val_acc, marker="s", label="Validation accuracy")
    ax.set_title("Validation Metrics by Epoch")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Score")
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower right")
    save_figure(fig, output_dir / "validation_metrics.png")


def plot_confusion_matrix(metrics: dict[str, Any], output_dir: Path) -> None:
    confusion = get_test_metrics(metrics).get("confusion_matrix")
    if not isinstance(confusion, dict):
        raise ValueError("Test metrics do not contain a confusion_matrix object")
    tn = int(confusion["tn"])
    fp = int(confusion["fp"])
    fn = int(confusion["fn"])
    tp = int(confusion["tp"])
    matrix = np.asarray([[tn, fp], [fn, tp]], dtype=int)

    fig, ax = plt.subplots(figsize=(5.8, 5.2))
    image = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="Count")
    labels = ["No-Lightning", "Lightning"]
    ax.set_xticks([0, 1], labels=labels)
    ax.set_yticks([0, 1], labels=labels)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title("Test Confusion Matrix")

    threshold = matrix.max() / 2.0
    for row in range(2):
        for col in range(2):
            color = "white" if matrix[row, col] > threshold else "black"
            ax.text(col, row, f"{matrix[row, col]:,}", ha="center", va="center", color=color, fontsize=12, fontweight="bold")
    save_figure(fig, output_dir / "confusion_matrix.png")


def collect_probabilities(checkpoint_path: Path, dataset_csv: Path, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
    require_file(checkpoint_path, "model checkpoint")
    require_file(dataset_csv, "dataset manifest")

    trainer = SatelliteTrainer(
        model_path=str(checkpoint_path),
        device="cpu",
        freeze_backbone=True,
        pretrained=False,
    )
    trainer.load_model()

    loaders = create_himawari_loaders(
        dataset_csv=str(dataset_csv),
        batch_size=batch_size,
        num_workers=0,
    )
    if "test" not in loaders:
        raise ValueError("create_himawari_loaders did not return a test loader")

    probabilities: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    trainer.model.eval()
    with torch.no_grad():
        for images, batch_labels in loaders["test"]:
            images = images.to(trainer.device)
            outputs = trainer.model(images).squeeze(-1).detach().cpu().numpy()
            probabilities.append(outputs.astype(np.float64, copy=False))
            labels.append(batch_labels.detach().cpu().numpy().astype(np.int64, copy=False))

    if not probabilities:
        raise ValueError("The test loader produced no batches")
    return np.concatenate(probabilities), np.concatenate(labels)


def plot_roc_curve(probabilities: np.ndarray, labels: np.ndarray, json_auc: float, output_dir: Path) -> float:
    fpr, tpr, _ = roc_curve(labels, probabilities)
    auc = float(roc_auc_score(labels, probabilities))

    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    ax.plot(fpr, tpr, color="tab:blue", linewidth=2, label=f"Test ROC-AUC = {auc:.4f}")
    ax.plot([0, 1], [0, 1], color="0.45", linestyle="--", linewidth=1.2, label="Chance")
    ax.set_title("Held-Out Test ROC Curve")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower right", title=f"JSON AUC {json_auc:.4f}")
    save_figure(fig, output_dir / "roc_curve.png")
    return auc


def plot_baseline_comparison(metrics: dict[str, Any], output_dir: Path) -> None:
    old = metrics.get("old_frozen_backbone_baseline")
    if not isinstance(old, dict):
        raise ValueError("Metrics JSON does not contain old_frozen_backbone_baseline")
    new = get_test_metrics(metrics)
    metric_keys = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    labels = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
    old_values = np.asarray([float(old[key]) for key in metric_keys])
    new_values = np.asarray([float(new[key]) for key in metric_keys])

    x = np.arange(len(metric_keys))
    width = 0.36
    fig, ax = plt.subplots(figsize=(8.2, 4.9))
    old_bars = ax.bar(x - width / 2, old_values, width, label="Old baseline", color="tab:gray")
    new_bars = ax.bar(x + width / 2, new_values, width, label="New frozen model", color="tab:blue")
    ax.set_title("Frozen-Backbone Baseline Comparison")
    ax.set_ylabel("Score")
    ax.set_ylim(0.0, 1.05)
    ax.set_xticks(x, labels=labels)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(loc="lower right")
    for bars in (old_bars, new_bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f"{height:.3f}", xy=(bar.get_x() + bar.get_width() / 2, height), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)
    save_figure(fig, output_dir / "baseline_comparison.png")


def plot_meteorological_metrics(metrics: dict[str, Any], output_dir: Path) -> None:
    test_metrics = get_test_metrics(metrics)
    metric_keys = ["pod", "far", "csi", "tss", "hss"]
    labels = ["POD", "FAR", "CSI", "TSS", "HSS"]
    values = np.asarray([float(test_metrics[key]) for key in metric_keys])

    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    bars = ax.bar(labels, values, color=["tab:green", "tab:red", "tab:blue", "tab:purple", "tab:orange"])
    ax.set_title("Meteorological Metrics on Test Split")
    ax.set_ylabel("Score")
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, axis="y", alpha=0.25)
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f"{height:.3f}", xy=(bar.get_x() + bar.get_width() / 2, height), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9)
    save_figure(fig, output_dir / "meteorological_metrics.png")


def plot_probability_histogram(probabilities: np.ndarray, labels: np.ndarray, threshold: float, output_dir: Path) -> None:
    bins = np.linspace(0.0, 1.0, 31)
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    ax.hist(probabilities[labels == 0], bins=bins, alpha=0.72, label="True No-Lightning", color="tab:gray", edgecolor="white")
    ax.hist(probabilities[labels == 1], bins=bins, alpha=0.68, label="True Lightning", color="tab:blue", edgecolor="white")
    ax.axvline(threshold, color="black", linestyle="--", linewidth=1.6, label=f"Threshold {threshold:.2f}")
    ax.set_title("Test Predicted Probability Distribution")
    ax.set_xlabel("Predicted probability of lightning")
    ax.set_ylabel("Patch count")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()
    save_figure(fig, output_dir / "probability_histogram.png")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate report plots for satellite CNN results.")
    parser.add_argument("--metrics-json", type=Path, default=Path("results/satellite_frozen_cpu_metrics.json"))
    parser.add_argument("--checkpoint", type=Path, default=Path("models/satellite_resnet50_frozen_cpu_best.pth"))
    parser.add_argument("--dataset-csv", type=Path, default=Path("data/processed/satellite_dataset.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/figures"))
    parser.add_argument("--batch-size", type=int, default=128)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = load_metrics(args.metrics_json)
    test_metrics = get_test_metrics(metrics)
    json_auc = float(test_metrics["roc_auc"])
    threshold = float(metrics.get("selected_threshold", test_metrics.get("threshold", 0.5)))

    plot_training_curves(metrics, args.output_dir)
    plot_validation_metrics(metrics, args.output_dir)
    plot_confusion_matrix(metrics, args.output_dir)
    plot_baseline_comparison(metrics, args.output_dir)
    plot_meteorological_metrics(metrics, args.output_dir)

    probabilities, labels = collect_probabilities(args.checkpoint, args.dataset_csv, args.batch_size)
    recomputed_auc = plot_roc_curve(probabilities, labels, json_auc, args.output_dir)
    plot_probability_histogram(probabilities, labels, threshold, args.output_dir)

    print("Wrote figures:")
    for filename in FIGURE_SPECS:
        path = args.output_dir / filename
        if not path.exists() or path.stat().st_size <= 0:
            raise RuntimeError(f"Figure was not written or is empty: {path}")
        print(f"{path} {path.stat().st_size} bytes")
    print(f"Recomputed test ROC-AUC: {recomputed_auc:.6f}")
    print(f"JSON test ROC-AUC: {json_auc:.6f}")


if __name__ == "__main__":
    main()
