import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from train_satellite import SatelliteTrainer, positive_class_metrics, select_threshold


def test_positive_class_metrics_include_meteorology_scores():
    labels = np.array([1, 1, 0, 0])
    probs = np.array([0.9, 0.4, 0.8, 0.1])

    metrics = positive_class_metrics(labels, probs, threshold=0.5)

    assert metrics["confusion_matrix"] == {"tn": 1, "fp": 1, "fn": 1, "tp": 1}
    assert metrics["pod"] == pytest.approx(0.5)
    assert metrics["far"] == pytest.approx(0.5)
    assert metrics["csi"] == pytest.approx(1 / 3)
    assert "tss" in metrics
    assert "hss" in metrics


def test_select_threshold_uses_validation_labels_only():
    labels = np.array([1, 1, 0, 0])
    probs = np.array([0.9, 0.7, 0.6, 0.1])

    threshold, metrics = select_threshold(labels, probs, metric="f1")

    assert 0.6 < threshold <= 0.7
    assert metrics["f1"] == pytest.approx(1.0)


def test_satellite_trainer_uses_discriminative_lr_groups():
    trainer = SatelliteTrainer(
        device="cpu",
        freeze_backbone=False,
        backbone_lr=1e-5,
        head_lr=1e-3,
        pretrained=False,
    )

    optimizer = trainer.build_optimizer()

    assert len(optimizer.param_groups) == 2
    assert optimizer.param_groups[0]["lr"] == pytest.approx(1e-5)
    assert optimizer.param_groups[1]["lr"] == pytest.approx(1e-3)


def test_save_history_derives_name_from_model_path(tmp_path):
    trainer = SatelliteTrainer(
        model_path=str(tmp_path / "satellite_resnet50_frozen_cpu_best.pth"),
        device="cpu",
        freeze_backbone=True,
        pretrained=False,
    )

    trainer.save_history({"epoch": [1], "train_loss": [0.1]})

    assert (tmp_path / "satellite_resnet50_frozen_cpu_training_history.json").exists()
