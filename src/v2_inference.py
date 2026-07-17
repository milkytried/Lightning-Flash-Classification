from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.v2_phase3_common import V2PatchDataset, apply_temperature, build_model, sha256_file

VERSION_LABEL = "Version 2 — Frozen Corrected Scientific Experiment"
SELECTED_RUN = "small_cnn_seed2026_bce_pos_weight_train_split_none"
DEFAULT_UNLOCK = "report/V2_PHASE3_TEST_UNLOCK.json"


def selected_model_record(unlock_path: str | Path = DEFAULT_UNLOCK) -> dict[str, Any]:
    unlock = json.loads(Path(unlock_path).read_text(encoding="utf-8"))
    for item in unlock["final_models"]:
        if item["run_name"] == SELECTED_RUN:
            return item
    raise KeyError(f"Selected run {SELECTED_RUN} not found in {unlock_path}")


def load_selected_model(unlock_path: str | Path = DEFAULT_UNLOCK, device: str = "cpu", verify_hash: bool = True) -> tuple[torch.nn.Module, dict[str, Any]]:
    record = selected_model_record(unlock_path)
    checkpoint_path = Path(record["checkpoint"])
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Frozen checkpoint is missing: {checkpoint_path}")
    if verify_hash:
        actual = sha256_file(checkpoint_path)
        if actual != record["checkpoint_sha256"]:
            raise ValueError(f"Checkpoint hash mismatch: {actual} != {record['checkpoint_sha256']}")
    model = build_model(record["architecture"]).to(torch.device(device))
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, record


def infer_patch(path: str | Path, unlock_path: str | Path = DEFAULT_UNLOCK, device: str = "cpu", verify_hash: bool = True) -> dict[str, Any]:
    model, record = load_selected_model(unlock_path, device=device, verify_hash=verify_hash)
    # Reuse the official V2 shared preprocessing. The temporary one-row dataset is
    # only a wrapper around the deterministic path loader; missing/corrupt/wrong-size
    # inputs raise instead of being replaced.
    dataset = V2PatchDataset.__new__(V2PatchDataset)
    dataset.architecture = record["architecture"]
    dataset.augment = False
    if record["architecture"] != "small_cnn":
        raise ValueError(f"Official selected model must be small_cnn, got {record['architecture']}")
    dataset.mean = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32).view(3, 1, 1)
    dataset.std = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32).view(3, 1, 1)
    tensor = dataset.load_tensor_for_path(path, augment=False).unsqueeze(0).to(torch.device(device))
    with torch.no_grad():
        logit = float(model(tensor).detach().cpu().numpy()[0])
    probability = float(1.0 / (1.0 + np.exp(-logit)))
    temperature = float(record["temperature_scaling"]["temperature"])
    calibrated_probability = float(apply_temperature(np.array([logit]), temperature)[0])
    threshold = float(record["selected_threshold"])
    return {
        "version_label": VERSION_LABEL,
        "run_name": record["run_name"],
        "architecture": record["architecture"],
        "checkpoint": record["checkpoint"],
        "checkpoint_sha256": record["checkpoint_sha256"],
        "threshold_source": "validation_f1_max_frozen_in_test_unlock",
        "threshold": threshold,
        "temperature": temperature,
        "input_path": str(path),
        "channel_order": ["B08", "B13", "B15"],
        "input_shape": [3, 64, 64],
        "preprocessing": "uint8 RGB patch scaled to [0,1], then small-CNN normalization mean/std [0.5,0.5,0.5]",
        "logit": logit,
        "probability": probability,
        "calibrated_probability": calibrated_probability,
        "classification": int(probability >= threshold),
        "bounded_claim": "Research classification of MMD-recorded cloud-to-ground lightning associations, not operational warning.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run frozen Version 2 selected-model inference for one 64x64 B08/B13/B15 patch.")
    parser.add_argument("patch")
    parser.add_argument("--unlock", default=DEFAULT_UNLOCK)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--no-verify-hash", action="store_true")
    args = parser.parse_args()
    print(json.dumps(infer_patch(args.patch, args.unlock, args.device, verify_hash=not args.no_verify_hash), indent=2))


if __name__ == "__main__":
    main()
