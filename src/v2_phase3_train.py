"""Version 2 Phase 3 preregistered neural-network training and evaluation.

The script has separate stages:

* ``train`` trains all preregistered validation-only runs and writes checkpoints.
* ``unlock`` records the frozen validation choices before test inference.
* ``evaluate`` runs controlled-test and natural-prevalence inference only after
  the unlock record exists.
* ``finalize`` writes the final comparison/decision from frozen outputs.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

if __package__ is None or __package__ == '':
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.v2_phase3_common import (
    apply_temperature,
    build_model,
    classification_metrics,
    clustered_bootstrap,
    create_v2_loader,
    environment_record,
    fit_temperature,
    make_loss,
    now_iso,
    parameter_counts,
    read_yaml,
    run_inference,
    save_checkpoint,
    save_predictions,
    select_threshold,
    set_seed,
    sha256_file,
    sha256_text,
    subgroup_metrics,
    write_json,
)


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "UNKNOWN"


def config_hash(config: dict[str, Any]) -> str:
    return sha256_text(json.dumps(config, sort_keys=True))


def ensure_test_unlock_exists() -> dict[str, Any]:
    path = Path("report/V2_PHASE3_TEST_UNLOCK.json")
    if not path.exists():
        raise SystemExit("Test inference is locked. Run the train stage and create V2_PHASE3_TEST_UNLOCK first.")
    return json.loads(path.read_text(encoding="utf-8"))


def run_name(architecture: str, seed: int, loss_name: str, augmentation: str) -> str:
    return f"{architecture}_seed{seed}_{loss_name}_{augmentation}"


def validation_selection_key(run: dict[str, Any]) -> tuple[float, float]:
    return (float(run["validation_pr_auc"]), -float(run["validation_loss"]))


def select_primary_runs(runs: list[dict[str, Any]], seeds: list[int]) -> dict[str, Any]:
    """Select one loss/augmentation option per architecture using validation only."""
    selections: dict[str, Any] = {}
    final_runs: list[dict[str, Any]] = []
    for architecture in ["small_cnn", "frozen_resnet50"]:
        arch_runs = [item for item in runs if item["architecture"] == architecture]
        candidate_keys = sorted({(item["loss_name"], item["augmentation"]) for item in arch_runs})
        candidate_rows = []
        for loss_name, augmentation in candidate_keys:
            subset = [item for item in arch_runs if item["loss_name"] == loss_name and item["augmentation"] == augmentation]
            if len(subset) != len(seeds):
                continue
            pr_values = np.array([item["validation_pr_auc"] for item in subset], dtype=float)
            loss_values = np.array([item["validation_loss"] for item in subset], dtype=float)
            candidate_rows.append({
                "architecture": architecture,
                "loss_name": loss_name,
                "augmentation": augmentation,
                "mean_validation_pr_auc": float(pr_values.mean()),
                "std_validation_pr_auc": float(pr_values.std(ddof=0)),
                "range_validation_pr_auc": [float(pr_values.min()), float(pr_values.max())],
                "mean_validation_loss": float(loss_values.mean()),
                "runs": [item["run_name"] for item in subset],
            })
        if not candidate_rows:
            raise SystemExit(f"No complete validation candidate set found for {architecture}.")
        selected = sorted(candidate_rows, key=lambda item: (item["mean_validation_pr_auc"], -item["mean_validation_loss"]))[-1]
        selected_runs = [
            item for item in arch_runs
            if item["loss_name"] == selected["loss_name"] and item["augmentation"] == selected["augmentation"]
        ]
        selected_runs = sorted(selected_runs, key=lambda item: seeds.index(int(item["seed"])))
        final_runs.extend(selected_runs)
        selections[architecture] = {"selected": selected, "candidates": candidate_rows, "final_runs": [item["run_name"] for item in selected_runs]}
    return {"architecture_selections": selections, "final_runs": final_runs}


def required_run_artifacts(config: dict[str, Any], name: str, checkpoint_path: Path) -> list[Path]:
    return [
        checkpoint_path,
        Path(config["outputs"]["training_history"]) / f"{name}.json",
        Path(config["outputs"]["validation_predictions"]) / f"{name}.csv",
        Path(config["outputs"]["threshold_records"]) / f"{name}.json",
        Path(config["outputs"]["calibration_records"]) / f"{name}.json",
        Path(config["outputs"]["resolved_configs"]) / f"{name}.json",
        Path(config["outputs"]["resource_reports"]) / f"{name}.json",
        Path(config["outputs"]["root"]) / "run_status" / f"{name}.json",
    ]


def load_completed_run_if_valid(config: dict[str, Any], name: str, checkpoint_path: Path) -> dict[str, Any] | None:
    status_path = Path(config["outputs"]["root"]) / "run_status" / f"{name}.json"
    if not status_path.exists():
        return None
    status = json.loads(status_path.read_text(encoding="utf-8"))
    if status.get("status") != "completed":
        raise SystemExit(f"Previous Phase 3 run {name} is marked {status.get('status')}; review before resuming.")
    missing = [str(path) for path in required_run_artifacts(config, name, checkpoint_path) if not path.exists()]
    if missing:
        raise SystemExit(f"Run {name} was marked complete but required artifacts are missing: {missing}")
    return status["run_record"]


def mark_run_status(config: dict[str, Any], name: str, status: str, extra: dict[str, Any]) -> None:
    status_path = Path(config["outputs"]["root"]) / "run_status" / f"{name}.json"
    write_json(status_path, {"run_name": name, "status": status, "updated_at_utc": now_iso(), **extra})


def current_working_set_bytes() -> int | None:
    if not sys.platform.startswith("win"):
        return None
    try:
        import ctypes
        import ctypes.wintypes as wt

        class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
            _fields_ = [
                ("cb", wt.DWORD),
                ("PageFaultCount", wt.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = PROCESS_MEMORY_COUNTERS()
        counters.cb = ctypes.sizeof(counters)
        handle = ctypes.windll.kernel32.GetCurrentProcess()
        ok = ctypes.windll.psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb)
        return int(counters.PeakWorkingSetSize) if ok else None
    except Exception:
        return None

def train_one_epoch(model, loader, criterion, optimizer, device) -> float:
    model.train()
    losses = []
    for images, labels, _ in tqdm(loader, desc="train", leave=False):
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite training loss encountered")
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu().item()))
    return float(np.mean(losses)) if losses else 0.0


def val_metrics_from_logits(labels: np.ndarray, logits: np.ndarray, loss: float, lr: float) -> dict[str, Any]:
    probs = 1.0 / (1.0 + np.exp(-logits))
    at_half = classification_metrics(labels, probs, 0.5)
    return {
        "validation_loss": float(loss),
        "validation_roc_auc": float(roc_auc_score(labels, probs)) if np.unique(labels).size == 2 else None,
        "validation_pr_auc": float(average_precision_score(labels, probs)) if np.unique(labels).size == 2 else None,
        "validation_accuracy_at_0_5": at_half["accuracy"],
        "validation_precision_at_0_5": at_half["precision"],
        "validation_recall_at_0_5": at_half["recall_pod"],
        "validation_f1_at_0_5": at_half["f1"],
        "validation_mcc_at_0_5": at_half["mcc"],
        "learning_rate": float(lr),
    }


def evaluate_loss(model, loader, criterion, device) -> tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    losses = []
    labels_all = []
    logits_all = []
    with torch.no_grad():
        for images, labels, _ in tqdm(loader, desc="val", leave=False):
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite validation loss encountered")
            losses.append(float(loss.detach().cpu().item()))
            labels_all.append(labels.detach().cpu().numpy())
            logits_all.append(logits.detach().cpu().numpy())
    return float(np.mean(losses)) if losses else 0.0, np.concatenate(labels_all), np.concatenate(logits_all)


def train_stage(args: argparse.Namespace) -> None:
    config = read_yaml(args.config)
    phase_config = read_yaml(config["inputs"]["v2_full_config"])
    root = Path(config["outputs"]["root"])
    checkpoint_root = Path(config["outputs"]["checkpoints"])
    for key in ["resolved_configs", "training_history", "validation_predictions", "calibration_records", "threshold_records", "resource_reports"]:
        Path(config["outputs"][key]).mkdir(parents=True, exist_ok=True)
    (root / "run_status").mkdir(parents=True, exist_ok=True)
    checkpoint_root.mkdir(parents=True, exist_ok=True)

    controlled_manifest = config["inputs"]["controlled_manifest"]
    train_df = pd.read_csv(controlled_manifest)
    train_labels = train_df.loc[train_df["split"].eq("train"), "label"].to_numpy()
    train_cfg_hash = config_hash(config)
    device = torch.device(args.device or config["training"]["device"])
    source_commit = git_commit()
    manifest_hashes = {
        "controlled_manifest_sha256": sha256_file(config["inputs"]["controlled_manifest"]),
        "natural_prevalence_manifest_sha256": sha256_file(config["inputs"]["natural_prevalence_manifest"]),
    }
    all_runs: list[dict[str, Any]] = []

    for architecture in ["small_cnn", "frozen_resnet50"]:
        for seed in config["training"]["seeds"]:
            for loss_name in config["training"]["loss_candidates"]:
                for augmentation in config["training"]["augmentation_candidates"]:
                    set_seed(int(seed))
                    name = run_name(architecture, int(seed), loss_name, augmentation)
                    model = build_model(architecture).to(device)
                    criterion, loss_record = make_loss(loss_name, train_labels)
                    criterion = criterion.to(device)
                    train_loader = create_v2_loader(
                        controlled_manifest,
                        "train",
                        architecture,
                        int(config["training"]["batch_size"]),
                        augment=(augmentation == "right_angle_flips_rotations"),
                        shuffle=True,
                    )
                    val_loader = create_v2_loader(
                        controlled_manifest,
                        "val",
                        architecture,
                        int(config["training"]["eval_batch_size"]),
                        augment=False,
                        shuffle=False,
                    )
                    optimizer = torch.optim.AdamW(
                        [p for p in model.parameters() if p.requires_grad],
                        lr=float(config["training"]["learning_rate"]),
                        weight_decay=float(config["training"]["weight_decay"]),
                    )
                    scheduler = ReduceLROnPlateau(
                        optimizer,
                        mode="max",
                        factor=float(config["training"]["scheduler_factor"]),
                        patience=int(config["training"]["scheduler_patience"]),
                    )
                    best_metric = -np.inf
                    best_val_loss = np.inf
                    best_epoch = 0
                    stale = 0
                    history: list[dict[str, Any]] = []
                    start = time.time()
                    checkpoint_path = checkpoint_root / f"{name}_best.pth"
                    completed = load_completed_run_if_valid(config, name, checkpoint_path)
                    if completed is not None:
                        all_runs.append(completed)
                        print(json.dumps({"run_name": name, "status": "resumed_completed"}, indent=2))
                        continue
                    mark_run_status(config, name, "running", {
                        "architecture": architecture,
                        "seed": int(seed),
                        "loss_name": loss_name,
                        "augmentation": augmentation,
                        "source_commit": source_commit,
                        **manifest_hashes,
                    })
                    early_stopping_reason = "max_epochs_reached"
                    try:
                        for epoch in range(1, int(config["training"]["max_epochs"]) + 1):
                            train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
                            val_loss, val_labels, val_logits = evaluate_loss(model, val_loader, criterion, device)
                            metrics = val_metrics_from_logits(val_labels, val_logits, val_loss, optimizer.param_groups[0]["lr"])
                            metrics["epoch"] = epoch
                            metrics["train_loss"] = train_loss
                            history.append(metrics)
                            selection = metrics["validation_pr_auc"] or -np.inf
                            scheduler.step(selection)
                            better = selection > best_metric or (selection == best_metric and val_loss < best_val_loss)
                            if better:
                                best_metric = selection
                                best_val_loss = val_loss
                                best_epoch = epoch
                                stale = 0
                                save_checkpoint(
                                    checkpoint_path,
                                    model,
                                    {
                                        "architecture": architecture,
                                        "seed": seed,
                                        "loss_name": loss_name,
                                        "augmentation": augmentation,
                                        "epoch": epoch,
                                        "validation_pr_auc": selection,
                                        "validation_loss": val_loss,
                                        "training_config_sha256": train_cfg_hash,
                                        "source_commit": source_commit,
                                        "parameter_counts": parameter_counts(model),
                                    },
                                )
                            else:
                                stale += 1
                                if stale >= int(config["training"]["early_stopping_patience"]):
                                    early_stopping_reason = "patience_exhausted"
                                    break
                    except Exception as exc:
                        mark_run_status(config, name, "failed", {"error": repr(exc), "source_commit": source_commit, **manifest_hashes})
                        raise
                    elapsed = time.time() - start

                    checkpoint = torch.load(checkpoint_path, map_location=device)
                    model.load_state_dict(checkpoint["model_state_dict"])
                    val_loss, val_labels, val_logits = evaluate_loss(model, val_loader, criterion, device)
                    val_probs = 1.0 / (1.0 + np.exp(-val_logits))
                    threshold, threshold_metrics = select_threshold(val_labels, val_probs)
                    temp_record = fit_temperature(val_logits, val_labels)
                    calibrated_probs = apply_temperature(val_logits, temp_record["temperature"])
                    val_pred_path = Path(config["outputs"]["validation_predictions"]) / f"{name}.csv"
                    val_predictions = run_inference(model, val_loader, device)
                    save_predictions(val_pred_path, val_predictions, threshold, calibrated_probs=calibrated_probs)
                    checkpoint_hash = save_checkpoint(
                        checkpoint_path,
                        model,
                        {
                            **{k: v for k, v in checkpoint.items() if k != "model_state_dict"},
                            "best_epoch": best_epoch,
                            "selected_threshold": threshold,
                            "threshold_objective": config["training"]["threshold_objective"],
                            "threshold_metrics_validation": threshold_metrics,
                            "temperature_scaling": temp_record,
                            "checkpoint_finalized_at_utc": now_iso(),
                        },
                    )
                    run_record = {
                        "run_name": name,
                        "architecture": architecture,
                        "seed": int(seed),
                        "loss_name": loss_name,
                        "augmentation": augmentation,
                        "best_epoch": best_epoch,
                        "validation_loss": float(val_loss),
                        "validation_pr_auc": float(average_precision_score(val_labels, val_probs)),
                        "validation_roc_auc": float(roc_auc_score(val_labels, val_probs)),
                        "selected_threshold": threshold,
                        "temperature_scaling": temp_record,
                        "checkpoint": str(checkpoint_path),
                        "checkpoint_sha256": checkpoint_hash,
                        "training_seconds": elapsed,
                        "peak_working_set_bytes": current_working_set_bytes(),
                        "device": str(device),
                        "training_config_sha256": train_cfg_hash,
                        "source_commit": source_commit,
                        **manifest_hashes,
                        "early_stopping_reason": early_stopping_reason,
                        "parameter_counts": parameter_counts(model),
                        "pretrained_weight_provenance": getattr(model, "weights_enum", None) or getattr(getattr(model, "backbone", None), "weights_enum", None),
                        "pretrained_weight_loaded": architecture != "frozen_resnet50" or bool(getattr(model, "weights_enum", None)),
                    }
                    all_runs.append(run_record)
                    write_json(Path(config["outputs"]["training_history"]) / f"{name}.json", {"history": history, "run": run_record})
                    write_json(Path(config["outputs"]["threshold_records"]) / f"{name}.json", {"threshold": threshold, "source": "validation_f1_max", "metrics": threshold_metrics})
                    write_json(Path(config["outputs"]["calibration_records"]) / f"{name}.json", temp_record)
                    write_json(Path(config["outputs"]["resolved_configs"]) / f"{name}.json", {"training_config": config, "phase2_config": phase_config, "run": run_record})
                    write_json(Path(config["outputs"]["resource_reports"]) / f"{name}.json", {"runtime_seconds": elapsed, "peak_working_set_bytes": run_record["peak_working_set_bytes"], "environment": environment_record()})
                    mark_run_status(config, name, "completed", {"run_record": run_record, "source_commit": source_commit, **manifest_hashes})
                    print(json.dumps(run_record, indent=2))

    root.mkdir(parents=True, exist_ok=True)
    seeds = [int(item) for item in config["training"]["seeds"]]
    selection = select_primary_runs(all_runs, seeds)
    final_runs = selection["final_runs"]
    training_report = {
        "created_at_utc": now_iso(),
        "training_config_sha256": train_cfg_hash,
        "source_commit": source_commit,
        "runs": final_runs,
        "candidate_runs": all_runs,
        "validation_selection": selection["architecture_selections"],
        "selection_rule": "Choose one loss/augmentation option per architecture by mean validation PR-AUC across preregistered seeds, using mean validation loss as tie-breaker; report all seeds without selecting by test.",
        "test_inference_status": "locked",
        **manifest_hashes,
    }
    write_json("report/V2_PHASE3_TRAINING.json", training_report)
    write_validation_selection_report(training_report)
    rows = [
        {
            "run": item["run_name"],
            "val_pr_auc": f"{item['validation_pr_auc']:.4f}",
            "val_roc_auc": f"{item['validation_roc_auc']:.4f}",
            "threshold": f"{item['selected_threshold']:.4f}",
            "checkpoint": item["checkpoint"],
        }
        for item in final_runs
    ]
    Path("report/V2_PHASE3_TRAINING.md").write_text(
        "# V2 Phase 3 Training\n\n"
        "Controlled-test and natural-prevalence inference remain locked. All choices below were made from training/validation only.\n\n"
        + markdown_table(rows)
        + "\n",
        encoding="utf-8",
    )


def summarize_by_architecture(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for architecture in sorted({item["architecture"] for item in runs}):
        subset = [item for item in runs if item["architecture"] == architecture]
        pr = np.array([item["validation_pr_auc"] for item in subset], dtype=float)
        roc = np.array([item["validation_roc_auc"] for item in subset], dtype=float)
        rows.append({
            "architecture": architecture,
            "runs": len(subset),
            "validation_pr_auc_mean": float(pr.mean()),
            "validation_pr_auc_std": float(pr.std(ddof=0)),
            "validation_pr_auc_range": [float(pr.min()), float(pr.max())],
            "validation_roc_auc_mean": float(roc.mean()),
            "validation_roc_auc_std": float(roc.std(ddof=0)),
            "validation_roc_auc_range": [float(roc.min()), float(roc.max())],
        })
    return rows


def write_validation_selection_report(training_report: dict[str, Any]) -> None:
    final_runs = training_report["runs"]
    candidate_runs = training_report["candidate_runs"]
    payload = {
        "created_at_utc": now_iso(),
        "source_commit": training_report["source_commit"],
        "training_config_sha256": training_report["training_config_sha256"],
        "controlled_manifest_sha256": training_report["controlled_manifest_sha256"],
        "natural_prevalence_manifest_sha256": training_report["natural_prevalence_manifest_sha256"],
        "completion_status": {item["run_name"]: "completed" for item in candidate_runs},
        "candidate_runs": candidate_runs,
        "final_primary_runs": final_runs,
        "validation_selection": training_report["validation_selection"],
        "architecture_summary_final_runs": summarize_by_architecture(final_runs),
        "ensemble_rule": "average seed probabilities is preregistered; no test predictions inspected at selection time",
        "test_use_statement": "No controlled-test or natural-prevalence predictions were inspected before this validation-selection report.",
    }
    write_json("report/V2_PHASE3_VALIDATION_SELECTION.json", payload)
    rows = [{
        "run": item["run_name"],
        "architecture": item["architecture"],
        "seed": item["seed"],
        "loss": item["loss_name"],
        "augmentation": item["augmentation"],
        "best_epoch": item["best_epoch"],
        "val_pr_auc": f"{item['validation_pr_auc']:.4f}",
        "val_roc_auc": f"{item['validation_roc_auc']:.4f}",
        "threshold": f"{item['selected_threshold']:.4f}",
        "checkpoint_sha256": item["checkpoint_sha256"],
    } for item in final_runs]
    selected = []
    for architecture, info in training_report["validation_selection"].items():
        item = info["selected"]
        selected.append({
            "architecture": architecture,
            "loss": item["loss_name"],
            "augmentation": item["augmentation"],
            "mean_val_pr_auc": f"{item['mean_validation_pr_auc']:.4f}",
            "mean_val_loss": f"{item['mean_validation_loss']:.4f}",
        })
    Path("report/V2_PHASE3_VALIDATION_SELECTION.md").write_text(
        "# V2 Phase 3 Validation Selection\n\n"
        "Selection used validation predictions only. Controlled-test and natural-prevalence predictions remain locked.\n\n"
        "## Selected Loss/Augmentation by Architecture\n\n"
        + markdown_table(selected)
        + "\n\n## Six Primary Runs for Test Unlock\n\n"
        + markdown_table(rows)
        + "\n",
        encoding="utf-8",
    )


def markdown_table(rows: list[dict[str, object]]) -> str:
    if not rows:
        return ""
    columns = list(rows[0].keys())
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(col, "")) for col in columns) + " |")
    return "\n".join(lines)


def unlock_stage(args: argparse.Namespace) -> None:
    config = read_yaml(args.config)
    training_path = Path("report/V2_PHASE3_TRAINING.json")
    if not training_path.exists():
        raise SystemExit("Training report missing; cannot unlock test inference.")
    training = json.loads(training_path.read_text(encoding="utf-8"))
    selection_path = Path("report/V2_PHASE3_VALIDATION_SELECTION.json")
    if not selection_path.exists():
        raise SystemExit("Validation-selection report missing; cannot unlock test inference.")
    expected_candidates = 2 * len(config["training"]["seeds"]) * len(config["training"]["loss_candidates"]) * len(config["training"]["augmentation_candidates"])
    expected_final = 2 * len(config["training"]["seeds"])
    if len(training.get("candidate_runs", [])) != expected_candidates:
        raise SystemExit(f"Expected {expected_candidates} candidate runs before unlock, found {len(training.get('candidate_runs', []))}.")
    if len(training["runs"]) != expected_final:
        raise SystemExit(f"Expected {expected_final} primary runs before unlock, found {len(training['runs'])}.")
    payload = {
        "created_at_utc": now_iso(),
        "source_commit": git_commit(),
        "training_config_sha256": training["training_config_sha256"],
        "controlled_manifest_sha256": training["controlled_manifest_sha256"],
        "natural_prevalence_manifest_sha256": training["natural_prevalence_manifest_sha256"],
        "validation_selection_report_sha256": sha256_file(selection_path),
        "final_models": training["runs"],
        "final_thresholds": {item["run_name"]: item["selected_threshold"] for item in training["runs"]},
        "final_calibration": {item["run_name"]: item["temperature_scaling"] for item in training["runs"]},
        "ensemble_rule": config["training"]["ensemble_rule"] if config["training"].get("ensemble_preregistered") else None,
        "validation_selection_rationale": training["selection_rule"],
        "test_use_statement": "No controlled-test or natural-prevalence labels or predictions were used for architecture, loss, epoch, hyperparameter, seed, calibration, ensemble, or threshold selection.",
    }
    write_json("report/V2_PHASE3_TEST_UNLOCK.json", payload)
    Path("report/V2_PHASE3_TEST_UNLOCK.md").write_text(
        "# V2 Phase 3 Test Unlock\n\n"
        f"Created: `{payload['created_at_utc']}`\n\n"
        + markdown_table(
            [
                {
                    "run": item["run_name"],
                    "checkpoint": item["checkpoint"],
                    "sha256": item["checkpoint_sha256"],
                    "threshold": item["selected_threshold"],
                    "temperature": item["temperature_scaling"]["temperature"],
                }
                for item in training["runs"]
            ]
        )
        + "\n\n"
        + payload["test_use_statement"]
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2))


def evaluate_manifest(config: dict[str, Any], run: dict[str, Any], manifest_path: str, split: str | None, output_dir: Path, device: torch.device) -> dict[str, Any]:
    architecture = run["architecture"]
    model = build_model(architecture).to(device)
    checkpoint = torch.load(run["checkpoint"], map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    loader = create_v2_loader(manifest_path, split, architecture, int(config["training"]["eval_batch_size"]), augment=False, shuffle=False)
    predictions = run_inference(model, loader, device)
    threshold = float(run["selected_threshold"])
    calibrated = apply_temperature(predictions.logits, float(run["temperature_scaling"]["temperature"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    save_predictions(output_dir / f"{run['run_name']}.csv", predictions, threshold, calibrated_probs=calibrated)
    metadata = predictions.metadata.copy()
    metadata["label"] = predictions.labels
    at_half = classification_metrics(predictions.labels, predictions.probs, 0.5)
    at_selected = classification_metrics(predictions.labels, predictions.probs, threshold)
    at_calibrated = classification_metrics(predictions.labels, calibrated, threshold)
    clusters = {}
    for cluster in ["date", "frame_id", "storm_id"]:
        if cluster in metadata.columns:
            clusters[cluster] = clustered_bootstrap(metadata, predictions.labels, predictions.probs, threshold, cluster, repeats=500, seed=42)
    per_group = subgroup_metrics(metadata, predictions.labels, predictions.probs, threshold, ["date", "frame_id", "storm_id", "frame_category"])
    return {
        "run_name": run["run_name"],
        "architecture": architecture,
        "seed": run["seed"],
        "threshold": threshold,
        "metrics_at_0_5": at_half,
        "metrics_at_validation_threshold": at_selected,
        "metrics_calibrated_at_validation_threshold": at_calibrated,
        "clustered_confidence_intervals": clusters,
        "subgroups": per_group,
        "prediction_csv": str(output_dir / f"{run['run_name']}.csv"),
    }


def evaluate_stage(args: argparse.Namespace) -> None:
    config = read_yaml(args.config)
    unlock = ensure_test_unlock_exists()
    device = torch.device(args.device or config["training"]["device"])
    controlled_results = []
    natural_results = []
    for run in unlock["final_models"]:
        controlled_results.append(
            evaluate_manifest(
                config,
                run,
                config["inputs"]["controlled_manifest"],
                "test",
                Path(config["outputs"]["controlled_predictions"]),
                device,
            )
        )
        natural_results.append(
            evaluate_manifest(
                config,
                run,
                config["inputs"]["natural_prevalence_manifest"],
                None,
                Path(config["outputs"]["natural_predictions"]),
                device,
            )
        )
    controlled_payload = {
        "created_at_utc": now_iso(),
        "test_unlock_hash": sha256_file("report/V2_PHASE3_TEST_UNLOCK.json"),
        "results": controlled_results,
    }
    natural_manifest = pd.read_csv(config["inputs"]["natural_prevalence_manifest"])
    natural_payload = {
        "created_at_utc": now_iso(),
        "test_unlock_hash": sha256_file("report/V2_PHASE3_TEST_UNLOCK.json"),
        "recorded_positive_prevalence": float(natural_manifest["label"].mean()),
        "results": natural_results,
    }
    write_json("report/V2_PHASE3_CONTROLLED_TEST.json", controlled_payload)
    write_json("report/V2_PHASE3_NATURAL_PREVALENCE.json", natural_payload)
    Path("report/V2_PHASE3_CONTROLLED_TEST.md").write_text(render_eval_md("V2 Phase 3 Controlled Test", controlled_results), encoding="utf-8")
    Path("report/V2_PHASE3_NATURAL_PREVALENCE.md").write_text(render_eval_md("V2 Phase 3 Natural Prevalence", natural_results), encoding="utf-8")
    print(json.dumps({"controlled": controlled_payload, "natural": natural_payload}, indent=2))


def render_eval_md(title: str, results: list[dict[str, Any]]) -> str:
    rows = []
    for item in results:
        metrics = item["metrics_at_validation_threshold"]
        rows.append(
            {
                "run": item["run_name"],
                "accuracy": f"{metrics['accuracy']:.4f}",
                "precision": f"{metrics['precision']:.4f}",
                "recall_pod": f"{metrics['recall_pod']:.4f}",
                "far": f"{metrics['false_discovery_ratio_far']:.4f}",
                "fpr": f"{metrics['false_positive_rate_fpr']:.4f}",
                "roc_auc": f"{metrics['roc_auc']:.4f}",
                "pr_auc": f"{metrics['pr_auc']:.4f}",
                "hss": f"{metrics['hss']:.4f}",
                "tss": f"{metrics['tss']:.4f}",
            }
        )
    return f"# {title}\n\nValidation-selected thresholds were frozen before this inference.\n\n" + markdown_table(rows) + "\n"


def finalize_stage(args: argparse.Namespace) -> None:
    controlled = json.loads(Path("report/V2_PHASE3_CONTROLLED_TEST.json").read_text(encoding="utf-8"))
    natural = json.loads(Path("report/V2_PHASE3_NATURAL_PREVALENCE.json").read_text(encoding="utf-8"))
    baselines = json.loads(Path("report/V2_FULL_BASELINES.json").read_text(encoding="utf-8"))
    rows = []
    for name, item in baselines.items():
        rows.append({"model": name, "source": "phase2_baseline", **item["test"]})
    for item in controlled["results"]:
        rows.append({"model": item["run_name"], "source": "phase3_neural", **item["metrics_at_validation_threshold"]})
    frame = pd.DataFrame(rows)
    neural = frame[frame["source"].eq("phase3_neural")]
    simple_image = frame[frame["model"].astype(str).str.startswith(("mean_channel", "b13_min"))]
    geographic = frame[frame["model"].astype(str).str.contains("latlon")]
    best_neural = neural.sort_values("pr_auc", ascending=False).iloc[0].to_dict() if len(neural) else {}
    best_image = simple_image.sort_values("pr_auc", ascending=False).iloc[0].to_dict() if len(simple_image) else {}
    best_geo = geographic.sort_values("pr_auc", ascending=False).iloc[0].to_dict() if len(geographic) else {}
    decision = "Version 2 evaluation is inconclusive"
    if best_neural and best_image and best_geo:
        if best_neural["pr_auc"] <= best_image["pr_auc"] + 0.02:
            decision = "Version 2 does not demonstrate a meaningful deep-learning advantage"
        elif best_neural["pr_auc"] > best_image["pr_auc"] + 0.05 and best_neural["pr_auc"] > best_geo["pr_auc"] + 0.05:
            decision = "Version 2 demonstrates meaningful image-based discrimination"
        else:
            decision = "Version 2 demonstrates limited advantage over simple image baselines"
    payload = {
        "created_at_utc": now_iso(),
        "decision": decision,
        "best_neural_controlled_test": best_neural,
        "best_simple_image_baseline": best_image,
        "best_geographic_baseline": best_geo,
        "controlled_test_results": controlled["results"],
        "natural_prevalence_results": natural["results"],
        "supported_claim": "Classification of MMD-recorded cloud-to-ground lightning associations from Himawari-9 image patches within a conservative empirical Peninsular Malaysia study region.",
        "unsupported_claims": [
            "Physical proof that lightning was absent",
            "Operational warning-system performance",
            "Real-time deployment readiness",
            "Generalization outside the study region",
            "True lightning nowcasting",
        ],
    }
    write_json("report/V2_PHASE3_FINAL_COMPARISON.json", {"comparison_rows": rows})
    write_json("report/V2_PHASE3_FINAL_DECISION.json", payload)
    Path("report/V2_PHASE3_FINAL_COMPARISON.md").write_text("# V2 Phase 3 Final Comparison\n\n" + frame.drop(columns=["confusion_matrix"], errors="ignore").to_markdown(index=False) + "\n", encoding="utf-8")
    Path("report/V2_PHASE3_FINAL_DECISION.md").write_text("# V2 Phase 3 Final Decision\n\n" + f"`{decision}`\n\n" + payload["supported_claim"] + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["train", "unlock", "evaluate", "finalize"])
    parser.add_argument("--config", default="configs/v2_training.yaml")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    if args.stage == "train":
        train_stage(args)
    elif args.stage == "unlock":
        unlock_stage(args)
    elif args.stage == "evaluate":
        evaluate_stage(args)
    elif args.stage == "finalize":
        finalize_stage(args)


if __name__ == "__main__":
    main()

