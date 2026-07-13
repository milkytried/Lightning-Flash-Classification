"""Generate report-ready plots for the frozen satellite CNN run.

The script reads finalized metrics from JSON and loads the saved checkpoint only
for read-only inference on the held-out test split to recover probabilities.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score, roc_curve

from himawari_data_loader import HimawariPatchDataset, create_himawari_loaders
from train_satellite import SatelliteTrainer


FIGURE_SPECS = {
    "example_input_patches.png": "Held-out Himawari-9 patches by true class.",
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


def select_example_input_rows(
    manifest: pd.DataFrame,
    samples_per_class: int = 4,
    seed: int = 42,
) -> pd.DataFrame:
    """Select temporally diverse test patches with at most one row per frame."""

    required = {"path", "label", "split", "timestamp", "frame_id"}
    missing = sorted(required.difference(manifest.columns))
    if missing:
        raise ValueError(f"Dataset manifest is missing required columns: {missing}")

    test_rows = manifest[manifest["split"].astype(str).str.casefold() == "test"].copy()
    test_rows = test_rows.reset_index(drop=True)
    test_rows["_test_index"] = np.arange(len(test_rows), dtype=int)
    test_rows["_frame_timestamp"] = pd.to_datetime(test_rows["timestamp"], utc=True, errors="coerce")
    if test_rows["_frame_timestamp"].isna().any():
        bad_paths = test_rows.loc[test_rows["_frame_timestamp"].isna(), "path"].head(5).tolist()
        raise ValueError(f"Test manifest contains invalid frame timestamps for: {bad_paths}")
    if test_rows["frame_id"].isna().any():
        raise ValueError("Test manifest contains rows without frame_id")

    selected_groups: list[pd.DataFrame] = []
    for label, class_name in ((1, "lightning"), (0, "no-lightning")):
        class_rows = test_rows[test_rows["label"].astype(int) == label].copy()
        rng = np.random.default_rng(np.random.SeedSequence([seed, label]))

        one_per_frame = []
        for _frame_id, frame_rows in class_rows.groupby("frame_id", sort=False):
            frame_rows = frame_rows.sort_values(["path", "_test_index"], kind="stable")
            chosen_position = int(rng.integers(0, len(frame_rows)))
            one_per_frame.append(frame_rows.iloc[chosen_position])

        if one_per_frame:
            distinct_frames = (
                pd.DataFrame(one_per_frame)
                .sort_values(["_frame_timestamp", "frame_id"], kind="stable")
                .reset_index(drop=True)
            )
        else:
            distinct_frames = class_rows.iloc[0:0].copy()

        distinct_count = len(distinct_frames)
        if distinct_count < samples_per_class:
            warnings.warn(
                f"Requested {samples_per_class} {class_name} examples but the test split "
                f"contains only {distinct_count} distinct source frames; using one patch "
                "from each available frame without duplication.",
                RuntimeWarning,
                stacklevel=2,
            )
            chosen_frames = distinct_frames
        else:
            spaced_indices = np.rint(
                np.linspace(0, distinct_count - 1, num=samples_per_class)
            ).astype(int)
            chosen_frames = distinct_frames.iloc[spaced_indices].copy()

        selected_groups.append(chosen_frames)

    if not selected_groups:
        return test_rows.iloc[0:0].copy()
    return pd.concat(selected_groups, ignore_index=True)


def verify_selected_probability_mapping(
    checkpoint_path: Path,
    dataset_csv: Path,
    selections: list[dict[str, Any]],
    tolerance: float = 1e-6,
) -> list[dict[str, Any]]:
    """Re-infer selected paths individually and verify their indexed scores."""

    require_file(checkpoint_path, "model checkpoint")
    require_file(dataset_csv, "dataset manifest")
    verifier = SatelliteTrainer(
        model_path=str(checkpoint_path),
        device="cpu",
        freeze_backbone=True,
        pretrained=False,
    )
    verifier.load_model()
    verifier.model.eval()

    test_dataset = HimawariPatchDataset(
        dataset_csv=str(dataset_csv),
        split="test",
        augment=False,
    )
    transform = test_dataset.transform
    verification_rows = []
    with torch.no_grad():
        for selection in selections:
            patch_path = require_file(Path(selection["path"]), "selected held-out patch")
            with Image.open(patch_path) as image:
                patch = np.asarray(image.convert("RGB"), dtype=np.uint8)
            image_tensor = transform(image=patch)["image"].unsqueeze(0).to(verifier.device)
            direct_probability = float(
                verifier.model(image_tensor).reshape(-1)[0].detach().cpu().item()
            )
            indexed_probability = float(selection["predicted_probability"])
            absolute_error = abs(direct_probability - indexed_probability)
            if absolute_error > tolerance:
                raise AssertionError(
                    "Selected-patch probability mapping mismatch for "
                    f"{patch_path}: indexed={indexed_probability:.10f}, "
                    f"direct={direct_probability:.10f}, error={absolute_error:.3g}"
                )
            verification_rows.append(
                {
                    "path": str(patch_path),
                    "indexed_probability": indexed_probability,
                    "direct_path_probability": direct_probability,
                    "absolute_error": absolute_error,
                    "tolerance": float(tolerance),
                    "passed": True,
                }
            )
    return verification_rows


def plot_example_input_patches(
    dataset_csv: Path,
    output_dir: Path,
    probabilities: np.ndarray,
    checkpoint_path: Path,
    threshold: float = 0.51,
    samples_per_class: int = 4,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Plot auditable, temporally diverse held-out examples for Figure 5.1."""

    require_file(dataset_csv, "dataset manifest")
    manifest = pd.read_csv(dataset_csv)
    test_rows = manifest[manifest["split"].astype(str).str.casefold() == "test"].reset_index(drop=True)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if len(probabilities) != len(test_rows):
        raise ValueError(
            "Probability count does not match test manifest rows: "
            f"{len(probabilities)} probabilities for {len(test_rows)} rows"
        )

    selected = select_example_input_rows(
        manifest,
        samples_per_class=samples_per_class,
        seed=seed,
    ).copy()
    if not selected.empty:
        selected["probability"] = probabilities[selected["_test_index"].to_numpy(dtype=int)]

    selections = [
        {
            "label": int(row["label"]),
            "class_name": "Lightning" if int(row["label"]) == 1 else "No lightning",
            "path": str(row["path"]),
            "frame_id": str(row["frame_id"]),
            "frame_timestamp": pd.Timestamp(row["_frame_timestamp"]).isoformat(),
            "test_index": int(row["_test_index"]),
            "predicted_probability": float(row["probability"]),
            "predicted_label": int(float(row["probability"]) >= threshold),
            "correct_at_threshold": bool(
                int(float(row["probability"]) >= threshold) == int(row["label"])
            ),
        }
        for _, row in selected.iterrows()
    ]
    verification_rows = verify_selected_probability_mapping(
        checkpoint_path,
        dataset_csv,
        selections,
        tolerance=1e-6,
    )

    fig, axes = plt.subplots(2, samples_per_class, figsize=(2.5 * samples_per_class, 5.6))
    row_titles = ["Lightning", "No lightning"]
    for row_index, label in enumerate((1, 0)):
        row_samples = [item for item in selections if item["label"] == label]
        for col_index in range(samples_per_class):
            ax = axes[row_index, col_index]
            ax.axis("off")
            if col_index >= len(row_samples):
                ax.text(0.5, 0.5, "No distinct frame", ha="center", va="center", color="0.45")
                continue

            selection = row_samples[col_index]
            patch_path = require_file(Path(selection["path"]), "held-out patch")
            with Image.open(patch_path) as image:
                ax.imshow(image.convert("RGB"))
            annotation_color = (
                "tab:green" if selection["correct_at_threshold"] else "tab:red"
            )
            ax.set_title(
                f"p = {selection['predicted_probability']:.2f}",
                fontsize=10,
                color=annotation_color,
                fontweight="bold",
            )
            if col_index == 0:
                ax.text(
                    -0.12,
                    0.5,
                    row_titles[row_index],
                    transform=ax.transAxes,
                    ha="right",
                    va="center",
                    fontsize=11,
                    fontweight="bold",
                )

    fig.suptitle(
        "Held-Out Himawari-9 Infrared Patches "
        "(R=B08 6.2um, G=B13 10.4um, B=B15 12.4um)\n"
        f"Decision threshold = {threshold:.2f}; green = correct, red = misclassified"
    )
    save_figure(fig, output_dir / "example_input_patches.png")

    sidecar = {
        "seed": int(seed),
        "decision_threshold": float(threshold),
        "samples_per_class_requested": int(samples_per_class),
        "frame_identifier_column": "frame_id",
        "frame_timestamp_column": "timestamp",
        "dataset_manifest": str(dataset_csv),
        "checkpoint": str(checkpoint_path),
        "probability_mapping_verification": {
            "method": "individual inference from each selected patch path",
            "tolerance": 1e-6,
            "passed": True,
            "comparisons": verification_rows,
        },
        "selections": selections,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    selection_path = output_dir / "example_input_patches_selection.json"
    with selection_path.open("w", encoding="utf-8") as handle:
        json.dump(sidecar, handle, indent=2)
        handle.write("\n")
    return selections

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


def summarize_positive_probabilities(
    probabilities: np.ndarray,
    labels: np.ndarray,
) -> dict[str, float]:
    """Summarize held-out positive-class probability calibration."""

    positive_probabilities = np.asarray(probabilities, dtype=np.float64)[
        np.asarray(labels, dtype=np.int64) == 1
    ]
    if not len(positive_probabilities):
        raise ValueError("Cannot summarize positive probabilities: no positive labels")
    q1, median, q3 = np.quantile(positive_probabilities, [0.25, 0.5, 0.75])
    return {
        "median": float(median),
        "q1": float(q1),
        "q3": float(q3),
        "fraction_above_0_9": float(np.mean(positive_probabilities > 0.9)),
    }

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
    parser.add_argument("--metrics-json", type=Path, default=Path("results/satellite_frozen_cpu_clean_metrics.json"))
    parser.add_argument("--checkpoint", type=Path, default=Path("models/satellite_resnet50_frozen_cpu_clean_best.pth"))
    parser.add_argument("--dataset-csv", type=Path, default=Path("data/processed/satellite_dataset.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/figures"))
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
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
    test_manifest = pd.read_csv(args.dataset_csv)
    test_labels = (
        test_manifest[test_manifest["split"].astype(str).str.casefold() == "test"]["label"]
        .to_numpy(dtype=np.int64)
    )
    if not np.array_equal(labels, test_labels):
        raise RuntimeError("Inference labels do not align with test manifest row order")
    plot_example_input_patches(
        args.dataset_csv,
        args.output_dir,
        probabilities,
        checkpoint_path=args.checkpoint,
        threshold=threshold,
        seed=args.seed,
    )
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
    positive_summary = summarize_positive_probabilities(probabilities, labels)
    print(
        "Positive-class test probabilities: "
        f"median={positive_summary['median']:.6f}, "
        f"IQR=[{positive_summary['q1']:.6f}, {positive_summary['q3']:.6f}], "
        f"fraction_above_0.9={positive_summary['fraction_above_0_9']:.6f}"
    )
    print(f"Example selection: {args.output_dir / 'example_input_patches_selection.json'}")


if __name__ == "__main__":
    main()
