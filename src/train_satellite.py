"""
Train and evaluate the Himawari satellite ResNet-50 patch classifier.

This script uses the aligned manifest splits in data/processed/satellite_dataset.csv:
train for optimization, val for early stopping and threshold selection, and test for
one final held-out evaluation.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from tqdm import tqdm

from himawari_data_loader import create_himawari_loaders
from model_arch import FocalLoss, LightningResNet50

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:  # pragma: no cover - tensorboard is optional at import time
    SummaryWriter = None


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


OLD_FROZEN_BASELINE = {
    "accuracy": 0.8765,
    "roc_auc": 0.9199,
    "precision": 0.8601,
    "recall": 0.8993,
    "f1": 0.8792,
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def positive_class_metrics(labels: np.ndarray, probs: np.ndarray, threshold: float) -> dict[str, Any]:
    preds = (probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()

    pod = tp / (tp + fn) if (tp + fn) else 0.0
    far = fp / (tp + fp) if (tp + fp) else 0.0
    csi = tp / (tp + fn + fp) if (tp + fn + fp) else 0.0
    pofd = fp / (fp + tn) if (fp + tn) else 0.0
    tss = pod - pofd
    hss_den = ((tp + fn) * (fn + tn)) + ((tp + fp) * (fp + tn))
    hss = (2 * ((tp * tn) - (fp * fn)) / hss_den) if hss_den else 0.0

    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(labels, preds)),
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall": float(recall_score(labels, preds, zero_division=0)),
        "f1": float(f1_score(labels, preds, zero_division=0)),
        "roc_auc": float(roc_auc_score(labels, probs)) if len(np.unique(labels)) > 1 else 0.0,
        "pod": float(pod),
        "far": float(far),
        "csi": float(csi),
        "tss": float(tss),
        "hss": float(hss),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
    }


def select_threshold(labels: np.ndarray, probs: np.ndarray, metric: str = "f1") -> tuple[float, dict[str, Any]]:
    best_threshold = 0.5
    best_metrics = positive_class_metrics(labels, probs, best_threshold)
    best_score = best_metrics[metric]

    for threshold in np.linspace(0.05, 0.95, 181):
        metrics = positive_class_metrics(labels, probs, float(threshold))
        if metrics[metric] > best_score:
            best_threshold = float(threshold)
            best_metrics = metrics
            best_score = metrics[metric]

    return best_threshold, best_metrics


class SatelliteTrainer:
    """Trainer for ResNet-50 on aligned Himawari patches."""

    def __init__(
        self,
        model_path: str = "models/satellite_resnet50_frozen_cpu_best.pth",
        device: str = "cpu",
        use_focal_loss: bool = True,
        freeze_backbone: bool = False,
        backbone_lr: float = 1e-5,
        head_lr: float = 1e-3,
        dropout_rate: float = 0.5,
        pretrained: bool = True,
        seed: int = 42,
    ):
        self.model_path = Path(model_path)
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        self.device = torch.device(device)
        self.use_focal_loss = use_focal_loss
        self.freeze_backbone = freeze_backbone
        self.backbone_lr = backbone_lr
        self.head_lr = head_lr
        self.seed = seed

        self.model = LightningResNet50(
            num_input_channels=3,
            num_classes=1,
            dropout_rate=dropout_rate,
            pretrained=pretrained,
        )
        self.configure_trainable_layers()
        self.model = self.model.to(self.device)

        if use_focal_loss:
            self.criterion = FocalLoss(alpha=0.25, gamma=2.0, reduction="mean")
        else:
            self.criterion = nn.BCELoss()

        logger.info("SatelliteTrainer initialized")
        logger.info("  Device: %s", self.device)
        logger.info("  Model: LightningResNet50")
        logger.info("  Freeze backbone: %s", freeze_backbone)
        logger.info("  Loss: %s", "Focal Loss" if use_focal_loss else "BCELoss")

    def configure_trainable_layers(self) -> None:
        for param in self.model.parameters():
            param.requires_grad = True

        if self.freeze_backbone:
            for name, param in self.model.backbone.named_parameters():
                param.requires_grad = name.startswith("fc.")

    def build_optimizer(self) -> optim.Optimizer:
        head_params = list(self.model.backbone.fc.parameters())
        head_param_ids = {id(param) for param in head_params}
        backbone_params = [
            param
            for param in self.model.parameters()
            if id(param) not in head_param_ids and param.requires_grad
        ]

        param_groups = []
        if backbone_params:
            param_groups.append({"params": backbone_params, "lr": self.backbone_lr, "name": "backbone"})
        param_groups.append({"params": head_params, "lr": self.head_lr, "name": "head"})

        logger.info("Optimizer: Adam")
        for group in param_groups:
            logger.info("  Param group %-8s lr=%g params=%d", group["name"], group["lr"], len(group["params"]))
        return optim.Adam(param_groups)

    def train_epoch(self, train_loader, optimizer, max_batches: int | None = None) -> float:
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(train_loader, desc="Training")
        for batch_idx, (images, labels) in enumerate(pbar):
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True).float()

            optimizer.zero_grad(set_to_none=True)
            probs = self.model(images).squeeze(-1)
            loss = self.criterion(probs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

            if max_batches is not None and batch_idx + 1 >= max_batches:
                break

        return total_loss / max(1, num_batches)

    def evaluate_loader(self, loader, split_name: str = "val") -> tuple[float, np.ndarray, np.ndarray]:
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        all_probs = []
        all_labels = []

        with torch.no_grad():
            pbar = tqdm(loader, desc=f"Evaluating {split_name}")
            for images, labels in pbar:
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True).float()

                probs = self.model(images).squeeze(-1)
                loss = self.criterion(probs, labels)

                total_loss += loss.item()
                num_batches += 1
                all_probs.append(probs.detach().cpu().numpy())
                all_labels.append(labels.detach().cpu().numpy())
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        return total_loss / max(1, num_batches), np.concatenate(all_probs), np.concatenate(all_labels)

    def run_gpu_sanity_check(self, dataset_csv: str, batch_size: int = 32, max_batches: int = 2) -> None:
        if self.device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for the GPU sanity check; not falling back to CPU.")

        loaders = create_himawari_loaders(dataset_csv, batch_size=batch_size, num_workers=0)
        optimizer = self.build_optimizer()
        loss = self.train_epoch(loaders["train"], optimizer, max_batches=max_batches)
        allocated_gb = torch.cuda.memory_allocated(self.device) / (1024**3)
        logger.info("GPU sanity check completed: batches=%d loss=%.6f allocated_vram_gb=%.3f", max_batches, loss, allocated_gb)

    def train(
        self,
        dataset_csv: str,
        num_epochs: int = 50,
        batch_size: int = 32,
        eval_batch_size: int = 128,
        early_stopping_patience: int = 5,
        log_dir: str = "logs/satellite_frozen_cpu",
        results_json: str = "results/satellite_frozen_cpu_metrics.json",
    ) -> dict[str, Any]:
        logger.info("Starting training")
        logger.info("  Dataset: %s", dataset_csv)
        logger.info("  Epochs: %d", num_epochs)
        logger.info("  Train batch size: %d", batch_size)
        logger.info("  Eval batch size: %d", eval_batch_size)

        train_loader = create_himawari_loaders(dataset_csv, batch_size=batch_size, num_workers=0)["train"]
        eval_loaders = create_himawari_loaders(dataset_csv, batch_size=eval_batch_size, num_workers=0)
        val_loader = eval_loaders["val"]
        test_loader = eval_loaders["test"]

        optimizer = self.build_optimizer()
        writer = SummaryWriter(log_dir=log_dir) if SummaryWriter is not None else None
        if writer is None:
            logger.warning("TensorBoard SummaryWriter unavailable; continuing without TensorBoard logs")

        history: dict[str, list[float | int]] = {
            "epoch": [],
            "train_loss": [],
            "val_loss": [],
            "val_accuracy_at_0_5": [],
            "val_precision_at_0_5": [],
            "val_recall_at_0_5": [],
            "val_f1_at_0_5": [],
            "val_roc_auc": [],
        }
        best_val_loss = float("inf")
        patience_count = 0

        for epoch in range(1, num_epochs + 1):
            logger.info("")
            logger.info("Epoch %d/%d", epoch, num_epochs)
            train_loss = self.train_epoch(train_loader, optimizer)
            val_loss, val_probs, val_labels = self.evaluate_loader(val_loader, split_name="val")
            val_metrics = positive_class_metrics(val_labels, val_probs, threshold=0.5)

            logger.info("Train Loss: %.4f", train_loss)
            logger.info("Val Loss:   %.4f", val_loss)
            logger.info(
                "Val @0.5: acc=%.4f precision=%.4f recall=%.4f f1=%.4f roc_auc=%.4f",
                val_metrics["accuracy"],
                val_metrics["precision"],
                val_metrics["recall"],
                val_metrics["f1"],
                val_metrics["roc_auc"],
            )

            history["epoch"].append(epoch)
            history["train_loss"].append(float(train_loss))
            history["val_loss"].append(float(val_loss))
            history["val_accuracy_at_0_5"].append(val_metrics["accuracy"])
            history["val_precision_at_0_5"].append(val_metrics["precision"])
            history["val_recall_at_0_5"].append(val_metrics["recall"])
            history["val_f1_at_0_5"].append(val_metrics["f1"])
            history["val_roc_auc"].append(val_metrics["roc_auc"])

            if writer is not None:
                writer.add_scalar("loss/train", train_loss, epoch)
                writer.add_scalar("loss/val", val_loss, epoch)
                for key in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
                    writer.add_scalar(f"val/{key}_at_0_5", val_metrics[key], epoch)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_count = 0
                self.save_model(epoch=epoch, val_loss=val_loss)
                logger.info("Best model saved (val_loss=%.4f)", val_loss)
            else:
                patience_count += 1
                logger.info("No improvement (%d/%d)", patience_count, early_stopping_patience)
                if patience_count >= early_stopping_patience:
                    logger.info("Early stopping triggered")
                    break

        self.load_model()
        val_loss, val_probs, val_labels = self.evaluate_loader(val_loader, split_name="val")
        threshold, val_threshold_metrics = select_threshold(val_labels, val_probs, metric="f1")
        test_loss, test_probs, test_labels = self.evaluate_loader(test_loader, split_name="test")
        test_metrics = positive_class_metrics(test_labels, test_probs, threshold=threshold)

        results = {
            "run_timestamp": datetime.now().isoformat(),
            "dataset_csv": dataset_csv,
            "model_path": str(self.model_path),
            "log_dir": log_dir,
            "freeze_backbone": self.freeze_backbone,
            "backbone_lr": self.backbone_lr,
            "head_lr": self.head_lr,
            "seed": self.seed,
            "best_val_loss": float(best_val_loss),
            "final_val_loss": float(val_loss),
            "final_test_loss": float(test_loss),
            "selected_threshold_source": "validation_f1_max",
            "selected_threshold": float(threshold),
            "validation_metrics_at_selected_threshold": val_threshold_metrics,
            "test_metrics_at_frozen_validation_threshold": test_metrics,
            "old_frozen_backbone_baseline": OLD_FROZEN_BASELINE,
            "history": history,
        }

        Path(results_json).parent.mkdir(parents=True, exist_ok=True)
        with open(results_json, "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2)
        self.save_history(history)

        if writer is not None:
            writer.close()

        logger.info("Selected threshold on validation only: %.3f", threshold)
        logger.info("Held-out TEST metrics: %s", json.dumps(test_metrics, indent=2))
        logger.info("Metrics JSON saved to %s", results_json)
        return results

    def save_model(self, epoch: int | None = None, val_loss: float | None = None) -> None:
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "epoch": epoch,
            "val_loss": val_loss,
            "freeze_backbone": self.freeze_backbone,
            "backbone_lr": self.backbone_lr,
            "head_lr": self.head_lr,
            "seed": self.seed,
        }
        torch.save(checkpoint, self.model_path)
        logger.info("Model saved to %s", self.model_path)

    def load_model(self) -> None:
        checkpoint = torch.load(self.model_path, map_location=self.device)
        state = checkpoint.get("model_state_dict", checkpoint)
        self.model.load_state_dict(state)
        logger.info("Model loaded from %s", self.model_path)

    def save_history(self, history: dict[str, list[float | int]]) -> None:
        run_name = self.model_path.stem
        if run_name.endswith("_best"):
            run_name = run_name[:-5]
        history_path = self.model_path.parent / f"{run_name}_training_history.json"
        with open(history_path, "w", encoding="utf-8") as fh:
            json.dump(history, fh, indent=2)
        logger.info("History saved to %s", history_path)


def resolve_device(requested: str, require_cuda: bool) -> str:
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is False.")
        return "cuda"
    if requested == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if require_cuda:
            raise RuntimeError("CUDA required but unavailable.")
        return "cpu"
    if requested == "cpu" and require_cuda:
        raise RuntimeError("CUDA required but device was set to CPU.")
    return requested


def print_cuda_info() -> None:
    available = torch.cuda.is_available()
    print(f"torch.cuda.is_available(): {available}")
    if available:
        idx = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(idx)
        print(f"GPU name: {props.name}")
        print(f"Total VRAM GB: {props.total_memory / (1024**3):.2f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train ResNet-50 on aligned Himawari satellite patches")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument("--device", type=str, default=None, choices=["cpu", "cuda", "auto"])
    parser.add_argument("--model-path", type=str, default="models/satellite_resnet50_frozen_cpu_best.pth")
    parser.add_argument("--results-json", type=str, default="results/satellite_frozen_cpu_metrics.json")
    parser.add_argument("--log-dir", type=str, default="logs/satellite_frozen_cpu")
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--backbone-lr", type=float, default=None)
    parser.add_argument("--head-lr", type=float, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--cuda-info", action="store_true")
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--sanity-steps", type=int, default=0)
    args = parser.parse_args()

    config = load_config(args.config)
    train_cfg = config.get("train", {})
    model_cfg = config.get("model", {})
    data_cfg = config.get("data", {})

    dataset_csv = args.dataset or data_cfg.get("satellite_dataset_csv", "data/processed/satellite_dataset.csv")
    epochs = args.epochs or int(train_cfg.get("max_epochs", 50))
    batch_size = args.batch_size or int(train_cfg.get("batch_size", 32))
    eval_batch_size = args.eval_batch_size or int(train_cfg.get("eval_batch_size", 128))
    device = resolve_device(args.device or train_cfg.get("device", "auto"), require_cuda=args.require_cuda)
    freeze_backbone = bool(args.freeze_backbone or model_cfg.get("freeze_backbone", False))
    backbone_lr = args.backbone_lr or float(train_cfg.get("backbone_learning_rate", 1e-5))
    head_lr = args.head_lr or float(train_cfg.get("head_learning_rate", 1e-3))
    patience = args.patience or int(train_cfg.get("early_stopping_patience", 5))
    seed = args.seed or int(train_cfg.get("seed", 42))

    set_seed(seed)

    if args.cuda_info:
        print_cuda_info()

    trainer = SatelliteTrainer(
        model_path=args.model_path,
        device=device,
        use_focal_loss=True,
        freeze_backbone=freeze_backbone,
        backbone_lr=backbone_lr,
        head_lr=head_lr,
        dropout_rate=float(model_cfg.get("dropout", 0.5)),
        pretrained=bool(model_cfg.get("pretrained", True)),
        seed=seed,
    )

    if args.sanity_steps:
        trainer.run_gpu_sanity_check(dataset_csv, batch_size=batch_size, max_batches=args.sanity_steps)
        return

    trainer.train(
        dataset_csv=dataset_csv,
        num_epochs=epochs,
        batch_size=batch_size,
        eval_batch_size=eval_batch_size,
        early_stopping_patience=patience,
        log_dir=args.log_dir,
        results_json=args.results_json,
    )


if __name__ == "__main__":
    main()
