"""Generate Version 2 FYP report figures from frozen artifacts.

This script intentionally regenerates per-sample scores from the saved selected
checkpoint. Scalar targets are used only as assertions so stale plots fail
loudly instead of silently propagating wrong numbers.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from io import StringIO
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.v2_phase3_common import SmallCNN, create_v2_loader, run_inference  # noqa: E402


CHECKPOINT = ROOT / "models/v2/phase3/small_cnn_seed2026_bce_pos_weight_train_split_none_best.pth"
EXPECTED_SHA256 = "888696cb7f6d1543875795fca0deec2aaf5b0e54157692633b619e17f216ce1a"
MAIN_MANIFEST = ROOT / "data/processed/v2/full/manifest.csv"
NATURAL_MANIFEST = ROOT / "data/processed/v2/natural_prevalence_test/manifest.csv"
HISTORY_JSON = ROOT / "results/v2/phase3/training_history/small_cnn_seed2026_bce_pos_weight_train_split_none.json"
OUTPUT_DIR = ROOT / "figures"
SCORE_CACHE = OUTPUT_DIR / "v2_selected_scores_cache.npz"
THRESHOLD = 0.8307269811630249
TOL = 0.001

TARGETS = {
    "controlled_accuracy": 0.9556,
    "controlled_roc_auc": 0.9835,
    "controlled_pr_auc": 0.9662,
    "controlled_tn": 1934,
    "controlled_fp": 53,
    "controlled_fn": 69,
    "controlled_tp": 689,
    "controlled_pod": 0.909,
    "controlled_far": 0.071,
    "controlled_csi": 0.850,
    "controlled_tss": 0.882,
    "controlled_hss": 0.888,
    "natural_tn": 1838,
    "natural_fp": 35,
    "natural_fn": 185,
    "natural_tp": 417,
    "natural_pod": 0.693,
    "natural_far": 0.077,
    "natural_csi": 0.655,
    "natural_tss": 0.674,
    "natural_hss": 0.736,
}

PALETTE = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "purple": "#CC79A7",
    "sky": "#56B4E9",
    "vermillion": "#D55E00",
    "grey": "#666666",
    "light_grey": "#E6E6E6",
}


@dataclass
class Scores:
    labels: np.ndarray
    probs: np.ndarray
    metadata: pd.DataFrame


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assert_close(name: str, value: float, target: float, rows: list[tuple[str, float, float, str]]) -> None:
    passed = abs(float(value) - float(target)) <= TOL
    rows.append((name, float(value), float(target), "PASS" if passed else "FAIL"))
    if not passed:
        raise AssertionError(f"{name} recomputed {value:.6f} disagrees with target {target:.6f}")


def assert_equal(name: str, value: int, target: int, rows: list[tuple[str, float, float, str]]) -> None:
    passed = int(value) == int(target)
    rows.append((name, float(value), float(target), "PASS" if passed else "FAIL"))
    if not passed:
        raise AssertionError(f"{name} recomputed {value} disagrees with target {target}")


def load_model() -> SmallCNN:
    if not CHECKPOINT.exists():
        raise FileNotFoundError(f"Missing checkpoint: {CHECKPOINT}")
    actual = sha256_file(CHECKPOINT)
    if actual != EXPECTED_SHA256:
        raise RuntimeError(f"Checkpoint SHA-256 mismatch: {actual} != {EXPECTED_SHA256}")
    model = SmallCNN()
    state = torch.load(CHECKPOINT, map_location="cpu")
    model.load_state_dict(state["model_state_dict"] if isinstance(state, dict) and "model_state_dict" in state else state)
    model.eval()
    return model


def infer_manifest(model: SmallCNN, manifest: Path, split: str | None) -> Scores:
    loader = create_v2_loader(manifest, split=split, architecture="small_cnn", batch_size=128, augment=False, shuffle=False)
    predictions = run_inference(model, loader, torch.device("cpu"))
    return Scores(labels=predictions.labels, probs=predictions.probs, metadata=predictions.metadata)


def load_or_create_scores() -> tuple[Scores, Scores]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cache_ok = SCORE_CACHE.exists()
    if cache_ok:
        cache = np.load(SCORE_CACHE, allow_pickle=True)
        if cache["checkpoint_sha256"].item() == EXPECTED_SHA256 and float(cache["threshold"]) == THRESHOLD:
            controlled = Scores(
                labels=cache["controlled_labels"],
                probs=cache["controlled_probs"],
                metadata=pd.read_json(StringIO(cache["controlled_metadata"].item())),
            )
            natural = Scores(
                labels=cache["natural_labels"],
                probs=cache["natural_probs"],
                metadata=pd.read_json(StringIO(cache["natural_metadata"].item())),
            )
            return controlled, natural

    model = load_model()
    controlled = infer_manifest(model, MAIN_MANIFEST, split="test")
    natural = infer_manifest(model, NATURAL_MANIFEST, split="natural_prevalence_test")
    np.savez_compressed(
        SCORE_CACHE,
        checkpoint_sha256=EXPECTED_SHA256,
        threshold=THRESHOLD,
        controlled_labels=controlled.labels,
        controlled_probs=controlled.probs,
        controlled_metadata=controlled.metadata.to_json(),
        natural_labels=natural.labels,
        natural_probs=natural.probs,
        natural_metadata=natural.metadata.to_json(),
    )
    return controlled, natural


def confusion_values(labels: np.ndarray, probs: np.ndarray) -> tuple[int, int, int, int]:
    preds = (probs >= THRESHOLD).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    return int(tn), int(fp), int(fn), int(tp)


def skill_scores(tn: int, fp: int, fn: int, tp: int) -> dict[str, float]:
    pod = tp / (tp + fn)
    far = fp / (tp + fp)
    csi = tp / (tp + fp + fn)
    fpr = fp / (fp + tn)
    tss = pod - fpr
    total = tn + fp + fn + tp
    expected = ((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn)) / total
    hss = 2 * (tp * tn - fp * fn) / ((tp + fn) * (fn + tn) + (tp + fp) * (fp + tn))
    return {"POD": pod, "FAR": far, "CSI": csi, "TSS": tss, "HSS": hss, "expected_random": expected}


def validate_scores(controlled: Scores, natural: Scores) -> list[tuple[str, float, float, str]]:
    rows: list[tuple[str, float, float, str]] = []
    c_tn, c_fp, c_fn, c_tp = confusion_values(controlled.labels, controlled.probs)
    n_tn, n_fp, n_fn, n_tp = confusion_values(natural.labels, natural.probs)
    assert_equal("controlled_tn", c_tn, TARGETS["controlled_tn"], rows)
    assert_equal("controlled_fp", c_fp, TARGETS["controlled_fp"], rows)
    assert_equal("controlled_fn", c_fn, TARGETS["controlled_fn"], rows)
    assert_equal("controlled_tp", c_tp, TARGETS["controlled_tp"], rows)
    assert_equal("natural_tn", n_tn, TARGETS["natural_tn"], rows)
    assert_equal("natural_fp", n_fp, TARGETS["natural_fp"], rows)
    assert_equal("natural_fn", n_fn, TARGETS["natural_fn"], rows)
    assert_equal("natural_tp", n_tp, TARGETS["natural_tp"], rows)

    assert_close("controlled_accuracy", accuracy_score(controlled.labels, controlled.probs >= THRESHOLD), TARGETS["controlled_accuracy"], rows)
    assert_close("controlled_roc_auc", roc_auc_score(controlled.labels, controlled.probs), TARGETS["controlled_roc_auc"], rows)
    assert_close("controlled_pr_auc", average_precision_score(controlled.labels, controlled.probs), TARGETS["controlled_pr_auc"], rows)

    for prefix, values in [("controlled", (c_tn, c_fp, c_fn, c_tp)), ("natural", (n_tn, n_fp, n_fn, n_tp))]:
        scores = skill_scores(*values)
        for metric in ["pod", "far", "csi", "tss", "hss"]:
            assert_close(f"{prefix}_{metric}", scores[metric.upper()], TARGETS[f"{prefix}_{metric}"], rows)
    return rows


def apply_style() -> None:
    plt.rcParams.update({
        "font.size": 11,
        "axes.grid": False,
        "grid.color": PALETTE["light_grey"],
        "grid.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 100,
        "savefig.dpi": 300,
    })


def savefig(name: str) -> None:
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / name, dpi=300, bbox_inches="tight")
    plt.close()


def fig_example_patches(controlled: Scores) -> None:
    frame = controlled.metadata.copy()
    frame["label"] = controlled.labels
    frame["probability"] = controlled.probs
    frame["prediction"] = (controlled.probs >= THRESHOLD).astype(int)
    tp = frame[(frame["label"] == 1) & (frame["prediction"] == 1)].copy()
    tn = frame[(frame["label"] == 0) & (frame["prediction"] == 0)].copy()
    rng = np.random.default_rng(42)

    selected: list[pd.Series] = []
    used_frames: set[str] = set()
    for candidates in [tp, tn]:
        rows: list[pd.Series] = []
        for _, row in candidates.sample(frac=1, random_state=int(rng.integers(0, 1_000_000))).iterrows():
            frame_id = str(row["frame_id"])
            if frame_id in used_frames:
                continue
            rows.append(row)
            used_frames.add(frame_id)
            if len(rows) == 4:
                break
        if len(rows) < 4:
            raise RuntimeError("Could not sample four examples from distinct frames for one class")
        selected.extend(rows)
    if len(used_frames) < 8:
        raise RuntimeError(f"Expected 8 distinct source frames, got {len(used_frames)}")

    fig, axes = plt.subplots(2, 4, figsize=(6.5, 3.4))
    for ax, row in zip(axes.flat, selected):
        image = Image.open(ROOT / row["path"]).convert("RGB")
        ax.imshow(image)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel(f"p={row['probability']:.3f}", labelpad=2)
    axes[0, 0].set_ylabel("Lightning", rotation=90, labelpad=14)
    axes[1, 0].set_ylabel("No lightning", rotation=90, labelpad=14)
    savefig("fig_5_1_example_patches.png")


def fig_training_loss() -> None:
    if not HISTORY_JSON.exists():
        print(f"TODO: missing history JSON, skipped fig_5_2_training_loss.png: {HISTORY_JSON}")
        return
    history = json.loads(HISTORY_JSON.read_text(encoding="utf-8"))["history"]
    epochs = [item["epoch"] for item in history]
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    ax.plot(epochs, [item["train_loss"] for item in history], color=PALETTE["blue"], label="Train loss")
    ax.plot(epochs, [item["validation_loss"] for item in history], color=PALETTE["orange"], label="Validation loss")
    ax.axvline(25, color=PALETTE["grey"], linestyle="--", linewidth=1.2)
    ax.annotate("Best epoch 25", xy=(25, min(item["validation_loss"] for item in history)), xytext=(26, 0.34),
                arrowprops={"arrowstyle": "->", "color": PALETTE["grey"]}, color=PALETTE["grey"])
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(True, axis="y")
    ax.legend(frameon=False)
    savefig("fig_5_2_training_loss.png")


def fig_val_metrics() -> None:
    if not HISTORY_JSON.exists():
        print(f"TODO: missing history JSON, skipped fig_5_3_val_metrics.png: {HISTORY_JSON}")
        return
    history = json.loads(HISTORY_JSON.read_text(encoding="utf-8"))["history"]
    best_pr = max(item["validation_pr_auc"] for item in history)
    if abs(best_pr - 0.9588) > TOL:
        raise AssertionError(f"Best validation PR-AUC {best_pr:.6f} disagrees with target 0.9588")
    epochs = [item["epoch"] for item in history]
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    ax.plot(epochs, [item["validation_roc_auc"] for item in history], color=PALETTE["blue"], label="ROC-AUC")
    ax.plot(epochs, [item["validation_pr_auc"] for item in history], color=PALETTE["green"], label="PR-AUC")
    ax.plot(epochs, [item["validation_accuracy_at_0_5"] for item in history], color=PALETTE["orange"], label="Accuracy at 0.5")
    ax.axvline(25, color=PALETTE["grey"], linestyle="--", linewidth=1.2)
    ax.annotate("Best PR-AUC 0.9588", xy=(25, best_pr), xytext=(14, 0.90),
                arrowprops={"arrowstyle": "->", "color": PALETTE["grey"]}, color=PALETTE["grey"])
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation metric")
    ax.set_ylim(0.78, 1.0)
    ax.grid(True, axis="y")
    ax.legend(frameon=False, ncol=3)
    savefig("fig_5_3_val_metrics.png")


def draw_confusion(ax: plt.Axes, tn: int, fp: int, fn: int, tp: int) -> None:
    matrix = np.array([[tn, fp], [fn, tp]])
    row_pct = matrix / matrix.sum(axis=1, keepdims=True) * 100
    ax.imshow(matrix, cmap="Blues")
    ax.set_xticks([0, 1], ["No lightning", "Lightning"])
    ax.set_yticks([0, 1], ["No lightning", "Lightning"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    for y in range(2):
        for x in range(2):
            color = "white" if matrix[y, x] > matrix.max() * 0.55 else "black"
            ax.text(x, y, f"{matrix[y, x]}\n{row_pct[y, x]:.1f}%", ha="center", va="center", color=color)


def fig_confusion_controlled(controlled: Scores) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4.6))
    draw_confusion(ax, *confusion_values(controlled.labels, controlled.probs))
    savefig("fig_5_4_confusion_controlled.png")


def fig_roc_pr_controlled(controlled: Scores) -> None:
    fpr, tpr, roc_thresholds = roc_curve(controlled.labels, controlled.probs)
    precision, recall, pr_thresholds = precision_recall_curve(controlled.labels, controlled.probs)
    roc_auc = roc_auc_score(controlled.labels, controlled.probs)
    pr_auc = average_precision_score(controlled.labels, controlled.probs)
    preds = controlled.probs >= THRESHOLD
    tp = np.sum((preds == 1) & (controlled.labels == 1))
    fp = np.sum((preds == 1) & (controlled.labels == 0))
    fn = np.sum((preds == 0) & (controlled.labels == 1))
    tn = np.sum((preds == 0) & (controlled.labels == 0))
    op_fpr = fp / (fp + tn)
    op_tpr = tp / (tp + fn)
    op_precision = tp / (tp + fp)
    op_recall = op_tpr

    fig, axes = plt.subplots(1, 2, figsize=(6.5, 3.4))
    axes[0].plot(fpr, tpr, color=PALETTE["blue"], label=f"AUC={roc_auc:.4f}")
    axes[0].plot([0, 1], [0, 1], color=PALETTE["grey"], linestyle=":", linewidth=1)
    axes[0].scatter([op_fpr], [op_tpr], color=PALETTE["orange"], zorder=3, label="Threshold")
    axes[0].set_xlabel("False positive rate")
    axes[0].set_ylabel("True positive rate")
    axes[0].legend(frameon=False, loc="lower right")
    axes[0].grid(True)

    axes[1].plot(recall, precision, color=PALETTE["green"], label=f"PR-AUC={pr_auc:.4f}")
    axes[1].scatter([op_recall], [op_precision], color=PALETTE["orange"], zorder=3, label="Threshold")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_xlim(0, 1)
    axes[1].set_ylim(0, 1.02)
    axes[1].legend(frameon=False, loc="lower left")
    axes[1].grid(True)
    savefig("fig_5_5_roc_pr_controlled.png")


def fig_v1_v2_comparison() -> None:
    metrics = ["Accuracy", "ROC-AUC", "PR-AUC"]
    v1 = [0.9095, 0.9681, np.nan]
    v2 = [0.9556, 0.9835, 0.9662]
    x = np.arange(len(metrics))
    width = 0.35
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    bars1 = ax.bar(x - width / 2, np.nan_to_num(v1, nan=0), width, color=PALETTE["grey"], label="V1 diagnostic baseline")
    bars2 = ax.bar(x + width / 2, v2, width, color=PALETTE["blue"], label="V2 controlled")
    bars1[-1].set_alpha(0.15)
    ax.text(x[-1] - width / 2, 0.05, "n/a", ha="center", va="bottom", color=PALETTE["grey"])
    for bars, values in [(bars1, v1), (bars2, v2)]:
        for bar, value in zip(bars, values):
            if np.isfinite(value):
                ax.text(bar.get_x() + bar.get_width() / 2, value + 0.01, f"{value:.3f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x, metrics)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Score")
    ax.grid(True, axis="y")
    ax.legend(frameon=False, loc="lower right")
    savefig("fig_5_6_v1_v2_comparison.png")


def fig_met_skill_scores(controlled: Scores, natural: Scores) -> None:
    c_scores = skill_scores(*confusion_values(controlled.labels, controlled.probs))
    n_scores = skill_scores(*confusion_values(natural.labels, natural.probs))
    metrics = ["POD", "FAR", "CSI", "TSS", "HSS"]
    x = np.arange(len(metrics))
    width = 0.35
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    c_values = [c_scores[m] for m in metrics]
    n_values = [n_scores[m] for m in metrics]
    bars1 = ax.bar(x - width / 2, c_values, width, color=PALETTE["blue"], label="Controlled")
    bars2 = ax.bar(x + width / 2, n_values, width, color=PALETTE["orange"], label="Natural prevalence")
    for bars in [bars1, bars2]:
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.015, f"{bar.get_height():.3f}",
                    ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x, metrics)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.grid(True, axis="y")
    ax.legend(frameon=False, loc="lower right")
    savefig("fig_5_7_met_skill_scores.png")


def fig_probability_histogram(controlled: Scores) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    bins = np.linspace(0, 1, 31)
    ax.hist(controlled.probs[controlled.labels == 0], bins=bins, alpha=0.65, color=PALETTE["blue"],
            label="No lightning", density=False)
    ax.hist(controlled.probs[controlled.labels == 1], bins=bins, alpha=0.65, color=PALETTE["orange"],
            label="Lightning", density=False)
    ax.axvline(THRESHOLD, color=PALETTE["grey"], linestyle="--", linewidth=1.4, label=f"Threshold {THRESHOLD:.4f}")
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Patch count")
    ax.grid(True, axis="y")
    ax.legend(frameon=False)
    savefig("fig_5_8_prob_histogram.png")


def print_assertion_table(rows: list[tuple[str, float, float, str]]) -> None:
    print("\nRecomputed metric assertions")
    print(f"{'metric':<28} {'recomputed':>12} {'target':>12} {'status':>8}")
    for name, value, target, status in rows:
        print(f"{name:<28} {value:>12.6f} {target:>12.6f} {status:>8}")


def print_outputs() -> None:
    print("\nGenerated figures")
    for path in sorted(OUTPUT_DIR.glob("fig_5_*.png")):
        print(f"{path.name:<38} {path.stat().st_size:>10} bytes")


def main() -> None:
    apply_style()
    controlled, natural = load_or_create_scores()
    rows = validate_scores(controlled, natural)
    fig_example_patches(controlled)
    fig_training_loss()
    fig_val_metrics()
    fig_confusion_controlled(controlled)
    fig_roc_pr_controlled(controlled)
    fig_v1_v2_comparison()
    fig_met_skill_scores(controlled, natural)
    fig_probability_histogram(controlled)
    print_assertion_table(rows)
    print_outputs()


if __name__ == "__main__":
    main()
