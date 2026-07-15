"""Shared Version 2 Phase 3 neural-network utilities.

Models in this module return raw logits. Probability conversion happens only in
evaluation code via ``torch.sigmoid``. The same dataset/preprocessing path is
used for training, validation, controlled-test inference, natural-prevalence
inference, and standalone inference.
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models
import yaml
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
SMALL_CNN_MEAN = [0.5, 0.5, 0.5]
SMALL_CNN_STD = [0.5, 0.5, 0.5]


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_yaml(path: str | Path) -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def environment_record() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torchvision": getattr(tv_models, "__version__", None),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_build": torch.version.cuda,
        "device_count": torch.cuda.device_count(),
    }


class V2PatchDataset(Dataset):
    """V2 manifest-backed patch dataset with explicit shared preprocessing."""

    def __init__(
        self,
        manifest_csv: str | Path,
        split: str | None,
        architecture: str,
        augment: bool = False,
        indices: Iterable[int] | None = None,
    ) -> None:
        self.manifest_csv = Path(manifest_csv)
        self.frame = pd.read_csv(self.manifest_csv)
        if split is not None and "split" in self.frame.columns:
            self.frame = self.frame[self.frame["split"].astype(str).eq(split)].copy()
        if indices is not None:
            self.frame = self.frame.iloc[list(indices)].copy()
        self.frame = self.frame.reset_index(drop=True)
        self.split = split
        self.architecture = architecture
        self.augment = augment
        if architecture == "frozen_resnet50":
            self.mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(3, 1, 1)
            self.std = torch.tensor(IMAGENET_STD, dtype=torch.float32).view(3, 1, 1)
        elif architecture == "small_cnn":
            self.mean = torch.tensor(SMALL_CNN_MEAN, dtype=torch.float32).view(3, 1, 1)
            self.std = torch.tensor(SMALL_CNN_STD, dtype=torch.float32).view(3, 1, 1)
        else:
            raise ValueError(f"Unknown architecture: {architecture}")

    def __len__(self) -> int:
        return len(self.frame)

    def load_tensor_for_path(self, path: str | Path, augment: bool | None = None) -> torch.Tensor:
        image_path = Path(path)
        if not image_path.exists():
            raise FileNotFoundError(f"Patch image does not exist: {image_path}")
        with Image.open(image_path) as opened:
            image = opened.convert("RGB")
            if image.size != (64, 64):
                raise ValueError(f"Expected 64x64 patch, got {image.size} for {image_path}")
            array = np.asarray(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()
        use_aug = self.augment if augment is None else augment
        if use_aug:
            if random.random() < 0.5:
                tensor = torch.flip(tensor, dims=[2])
            if random.random() < 0.5:
                tensor = torch.flip(tensor, dims=[1])
            rotations = random.randint(0, 3)
            if rotations:
                tensor = torch.rot90(tensor, k=rotations, dims=[1, 2])
        return (tensor - self.mean) / self.std

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        row = self.frame.iloc[index]
        tensor = self.load_tensor_for_path(row["path"])
        label = torch.tensor(float(row["label"]), dtype=torch.float32)
        meta = {
            "path": row["path"],
            "date": row.get("date", ""),
            "frame_id": row.get("frame_id", ""),
            "storm_id": row.get("storm_id", ""),
            "frame_category": row.get("frame_category", ""),
        }
        return tensor, label, meta


def create_v2_loader(
    manifest_csv: str | Path,
    split: str | None,
    architecture: str,
    batch_size: int,
    augment: bool = False,
    shuffle: bool = False,
    num_workers: int = 0,
) -> DataLoader:
    dataset = V2PatchDataset(manifest_csv, split=split, architecture=architecture, augment=augment)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=torch.cuda.is_available(),
    )


class SmallCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.25),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x)).squeeze(1)


class FrozenResNet50(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        weights = tv_models.ResNet50_Weights.IMAGENET1K_V2
        self.weights_enum = "ResNet50_Weights.IMAGENET1K_V2"
        self.backbone = tv_models.resnet50(weights=weights)
        in_features = self.backbone.fc.in_features
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.fc = nn.Linear(in_features, 1)
        for param in self.backbone.fc.parameters():
            param.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x).squeeze(1)


def build_model(architecture: str) -> nn.Module:
    if architecture == "small_cnn":
        return SmallCNN()
    if architecture == "frozen_resnet50":
        return FrozenResNet50()
    raise ValueError(f"Unknown architecture: {architecture}")


def parameter_counts(model: nn.Module) -> dict[str, int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    total = trainable + frozen
    return {"trainable": int(trainable), "frozen": int(frozen), "total": int(total)}


def make_loss(loss_name: str, train_labels: np.ndarray) -> tuple[nn.Module, dict[str, Any]]:
    if loss_name == "bce_unweighted":
        return nn.BCEWithLogitsLoss(), {"loss": loss_name, "pos_weight": None}
    if loss_name == "bce_pos_weight_train_split":
        positives = float(np.sum(train_labels == 1))
        negatives = float(np.sum(train_labels == 0))
        if positives <= 0:
            raise ValueError("Cannot compute pos_weight with zero positive training examples")
        pos_weight = negatives / positives
        return nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, dtype=torch.float32)), {
            "loss": loss_name,
            "pos_weight": pos_weight,
        }
    raise ValueError(f"Unknown loss: {loss_name}")


def ece_score(labels: np.ndarray, probs: np.ndarray, bins: int = 10) -> float:
    labels = np.asarray(labels)
    probs = np.asarray(probs)
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = len(labels)
    error = 0.0
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (probs >= low) & (probs < high if high < 1.0 else probs <= high)
        if np.any(mask):
            error += float(mask.mean()) * abs(float(probs[mask].mean()) - float(labels[mask].mean()))
    return float(error) if total else 0.0


def classification_metrics(labels: np.ndarray, probs: np.ndarray, threshold: float) -> dict[str, Any]:
    labels = np.asarray(labels).astype(int)
    probs = np.asarray(probs, dtype=float)
    preds = (probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    far = fp / (tp + fp) if (tp + fp) else 0.0
    pod = tp / (tp + fn) if (tp + fn) else 0.0
    hss_den = ((tp + fn) * (fn + tn)) + ((tp + fp) * (fp + tn))
    hss = 2 * ((tp * tn) - (fp * fn)) / hss_den if hss_den else 0.0
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(labels, preds)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, preds)),
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall_pod": float(recall_score(labels, preds, zero_division=0)),
        "f1": float(f1_score(labels, preds, zero_division=0)),
        "specificity": float(specificity),
        "false_positive_rate_fpr": float(fpr),
        "false_discovery_ratio_far": float(far),
        "mcc": float(matthews_corrcoef(labels, preds)),
        "roc_auc": float(roc_auc_score(labels, probs)) if np.unique(labels).size == 2 else None,
        "pr_auc": float(average_precision_score(labels, probs)) if np.unique(labels).size == 2 else None,
        "brier_score": float(brier_score_loss(labels, probs)),
        "expected_calibration_error": ece_score(labels, probs),
        "hss": float(hss),
        "tss": float(pod - fpr),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def select_threshold(labels: np.ndarray, probs: np.ndarray) -> tuple[float, dict[str, Any]]:
    candidates = np.unique(np.r_[0.0, np.linspace(0.01, 0.99, 99), probs, 1.0])
    scores = np.array([f1_score(labels, probs >= item, zero_division=0) for item in candidates])
    best = np.flatnonzero(scores == scores.max())[-1]
    threshold = float(candidates[best])
    return threshold, classification_metrics(labels, probs, threshold)


def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    logits_t = torch.tensor(logits, dtype=torch.float32)
    labels_t = torch.tensor(labels, dtype=torch.float32)
    log_temp = torch.zeros(1, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_temp], lr=0.1, max_iter=100)

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        temp = torch.exp(log_temp).clamp(0.05, 20.0)
        loss = F.binary_cross_entropy_with_logits(logits_t / temp, labels_t)
        loss.backward()
        return loss

    optimizer.step(closure)
    temperature = float(torch.exp(log_temp).detach().clamp(0.05, 20.0).item())
    return {"temperature": temperature, "method": "validation_logits_temperature_scaling"}


def apply_temperature(logits: np.ndarray, temperature: float) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.asarray(logits, dtype=float) / temperature))


@dataclass
class PredictionFrame:
    labels: np.ndarray
    logits: np.ndarray
    probs: np.ndarray
    metadata: pd.DataFrame


def collate_metadata(items: list[dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(items)


def run_inference(model: nn.Module, loader: DataLoader, device: torch.device) -> PredictionFrame:
    model.eval()
    all_logits: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    metadata: list[dict[str, Any]] = []
    with torch.no_grad():
        for images, labels, metas in loader:
            images = images.to(device)
            logits = model(images).detach().cpu().numpy()
            all_logits.append(logits)
            all_labels.append(labels.numpy())
            batch_size = len(labels)
            for idx in range(batch_size):
                metadata.append({key: metas[key][idx] for key in metas})
    logits_arr = np.concatenate(all_logits)
    labels_arr = np.concatenate(all_labels).astype(int)
    probs = 1.0 / (1.0 + np.exp(-logits_arr))
    return PredictionFrame(labels=labels_arr, logits=logits_arr, probs=probs, metadata=pd.DataFrame(metadata))


def save_predictions(path: str | Path, predictions: PredictionFrame, threshold: float, calibrated_probs: np.ndarray | None = None) -> None:
    frame = predictions.metadata.copy()
    frame["label"] = predictions.labels
    frame["logit"] = predictions.logits
    frame["probability"] = predictions.probs
    frame["threshold"] = threshold
    frame["prediction"] = (predictions.probs >= threshold).astype(int)
    if calibrated_probs is not None:
        frame["calibrated_probability"] = calibrated_probs
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def clustered_bootstrap(frame: pd.DataFrame, labels: np.ndarray, probs: np.ndarray, threshold: float, cluster: str, repeats: int = 500, seed: int = 42) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    groups = frame[cluster].astype(str).to_numpy()
    unique = np.unique(groups)
    draws: list[dict[str, Any]] = []
    for _ in range(repeats):
        chosen = rng.choice(unique, size=len(unique), replace=True)
        indices = np.concatenate([np.flatnonzero(groups == item) for item in chosen])
        draws.append(classification_metrics(labels[indices], probs[indices], threshold))
    output: dict[str, Any] = {}
    for key in [
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall_pod",
        "f1",
        "mcc",
        "roc_auc",
        "pr_auc",
        "brier_score",
        "expected_calibration_error",
        "false_discovery_ratio_far",
        "false_positive_rate_fpr",
        "hss",
        "tss",
    ]:
        values = np.array([item[key] for item in draws if item[key] is not None], dtype=float)
        output[key] = {
            "lower": float(np.quantile(values, 0.025)),
            "median": float(np.quantile(values, 0.5)),
            "upper": float(np.quantile(values, 0.975)),
            "valid_replicates": int(values.size),
        }
    return output


def subgroup_metrics(manifest: pd.DataFrame, labels: np.ndarray, probs: np.ndarray, threshold: float, columns: list[str]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for column in columns:
        if column not in manifest.columns:
            continue
        result[column] = {}
        for value, part in manifest.groupby(column, dropna=False):
            idx = part.index.to_numpy()
            if len(idx) < 2:
                continue
            result[column][str(value)] = classification_metrics(labels[idx], probs[idx], threshold)
    return result


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    payload: dict[str, Any],
) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), **payload}, path)
    return sha256_file(path)


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
