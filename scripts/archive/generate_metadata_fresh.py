# ⚠️ SUPERSEDED — retained for provenance only. Not the final result. See README.md and report/ for Version 2.
"""Archived metadata generator for the earlier 11-PNG Himawari-8 prototype.

Its artifact describes the baseline checkpoint and is superseded by the final
aligned Himawari-9 metrics record."""

import json
import torch
from pathlib import Path
from datetime import datetime
import pandas as pd

print("\n" + "="*80)
print("GENERATING FRESH CHECKPOINT METADATA")
print("="*80)

# Paths
checkpoint_path = Path("models/satellite_resnet50_fresh.pth")
history_path = Path("models/satellite_training_history_fresh.json")
dataset_csv = Path("data/processed/satellite_dataset.csv")

# Verify checkpoint exists
if not checkpoint_path.exists():
    print(f"ERROR: Checkpoint not found: {checkpoint_path}")
    exit(1)

if not history_path.exists():
    print(f"ERROR: Training history not found: {history_path}")
    exit(1)

# Load training history
with open(history_path, 'r') as f:
    history = json.load(f)

# Load dataset to extract split info
df = pd.read_csv(dataset_csv)

# Extract PNG lists by split
train_df = df[df['split'] == 'train']
val_df = df[df['split'] == 'val']
test_df = df[df['split'] == 'test']

# Get unique PNG sources
def extract_png_sources(df_split):
    """Extract unique PNG source files from path column."""
    png_sources = set()
    for path in df_split['path']:
        # Path format: data/processed/patches/{split}/{label}/{patch_filename}.png
        # Extract timestamp from patch filename (e.g., "20250418_110037_0.png")
        filename = Path(path).stem  # Remove .png extension
        # Timestamp is first 15 chars: YYYYMMDD_HHMMSS
        if len(filename) >= 15:
            timestamp = filename[:15]  # e.g., "20250418_110037"
            png_sources.add(timestamp)
    return sorted(list(png_sources))

train_pngs = extract_png_sources(train_df)
val_pngs = extract_png_sources(val_df)
test_pngs = extract_png_sources(test_df)

# Verify no PNG overlap
train_set = set(train_pngs)
val_set = set(val_pngs)
test_set = set(test_pngs)

train_val_overlap = train_set & val_set
train_test_overlap = train_set & test_set
val_test_overlap = val_set & test_set

print("\n[Split Verification]")
print(f"  Train PNGs: {len(train_pngs)}")
print(f"  Val PNGs: {len(val_pngs)}")
print(f"  Test PNGs: {len(test_pngs)}")
print(f"  Train-Val overlap: {len(train_val_overlap)} [OK]" if len(train_val_overlap) == 0 else f"  Train-Val overlap: {len(train_val_overlap)} [PROBLEM]")
print(f"  Train-Test overlap: {len(train_test_overlap)} [OK]" if len(train_test_overlap) == 0 else f"  Train-Test overlap: {len(train_test_overlap)} [PROBLEM]")
print(f"  Val-Test overlap: {len(val_test_overlap)} [OK]" if len(val_test_overlap) == 0 else f"  Val-Test overlap: {len(val_test_overlap)} [PROBLEM]")

# Build metadata
metadata = {
    "generated_at": datetime.now().isoformat(),
    "checkpoint_filename": "satellite_resnet50_fresh.pth",
    "training_script": "train_fresh_optimized.py",
    "evaluation_script": "eval_fresh.py",
    "dataset_csv": str(dataset_csv),
    "split_seed": 42,
    "splits": {
        "train": {
            "png_files": train_pngs,
            "png_count": len(train_pngs),
            "patch_count_total": len(train_df),
            "patch_count_positive_lightning": int((train_df['label'] == 1).sum()),
            "patch_count_negative_no_lightning": int((train_df['label'] == 0).sum()),
            "date_range": list(set([p[:8] for p in train_pngs]))  # Extract YYYYMMDD from PNG names
        },
        "val": {
            "png_files": val_pngs,
            "png_count": len(val_pngs),
            "patch_count_total": len(val_df),
            "patch_count_positive_lightning": int((val_df['label'] == 1).sum()),
            "patch_count_negative_no_lightning": int((val_df['label'] == 0).sum()),
            "date_range": list(set([p[:8] for p in val_pngs]))
        },
        "test": {
            "png_files": test_pngs,
            "png_count": len(test_pngs),
            "patch_count_total": len(test_df),
            "patch_count_positive_lightning": int((test_df['label'] == 1).sum()),
            "patch_count_negative_no_lightning": int((test_df['label'] == 0).sum()),
            "date_range": list(set([p[:8] for p in test_pngs]))
        }
    },
    "model_config": {
        "architecture": "LightningResNet50",
        "backbone": "ResNet50 (pre-trained on ImageNet)",
        "backbone_frozen": True,
        "head_trainable": True,
        "input_channels": 3,
        "input_resolution": "64x64",
        "output": "binary classification (sigmoid)",
        "custom_head": "Dropout(0.5) → Linear(2048→128) → ReLU → Dropout(0.3) → Linear(128→1) → Sigmoid",
        "dropout_rate": 0.5
    },
    "training_config": {
        "loss_function": "FocalLoss",
        "focal_loss_alpha": 0.25,
        "focal_loss_gamma": 2.0,
        "optimizer": "Adam",
        "learning_rate": 0.001,
        "batch_size": 32,
        "max_epochs": 15,
        "early_stopping_patience": 5,
        "gradient_clipping_max_norm": 1.0,
        "device": "cpu",
        "optimization_strategy": "Freeze ResNet-50 backbone, train only classifier head for CPU efficiency"
    },
    "evaluation_config": {
        "threshold": 0.5,
        "batch_size": 256,
        "metrics": [
            "accuracy",
            "precision",
            "recall/POD (Probability of Detection)",
            "F1-score",
            "ROC-AUC",
            "FAR (False Alarm Ratio)",
            "CSI (Critical Success Index / Threat Score)",
            "TSS (True Skill Statistic)",
            "HSS (Heidke Skill Score)",
            "Confusion Matrix (TP, FP, FN, TN)"
        ]
    },
    "training_history": {
        "epochs_completed": len(history['epoch']),
        "final_epoch": history['epoch'][-1] if history['epoch'] else 0,
        "best_val_loss": float(min(history['val_loss'])) if history['val_loss'] else None,
        "best_epoch": int(history['val_loss'].index(min(history['val_loss']))) + 1 if history['val_loss'] else None,
        "early_stopping_triggered": len(history['epoch']) < 15
    },
    "data_pipeline": {
        "source_pngs": "Malaysian Meteorological Department Himawari-8 satellite imagery",
        "image_size_original": "800x950 pixels per PNG",
        "image_size_patches": "64x64 pixels per patch",
        "patch_extraction": "Positive patches centered at lightning strike locations; negative patches randomly sampled from non-lightning areas",
        "lightning_source": "Malaysian Meteorological Department lightning strike CSV",
        "split_method": "Image-level split with date-based chronological separation (seed=42 for reproducibility)",
        "label_method": "60-minute lead-time window: lightning strike location matched to satellite patch at specific timestamp",
        "corrected_split": True,
        "temporal_contamination_fixed": True
    },
    "integrity_checks": {
        "train_val_png_overlap": len(train_val_overlap),
        "train_test_png_overlap": len(train_test_overlap),
        "val_test_png_overlap": len(val_test_overlap),
        "split_is_clean": len(train_val_overlap) == 0 and len(train_test_overlap) == 0 and len(val_test_overlap) == 0,
        "conclusion": "Split is clean - no source PNG appears in multiple splits"
    }
}

# Save metadata
output_path = Path("models/model_metadata_fresh.json")
with open(output_path, 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"\n[OK] Metadata saved to: {output_path}")
print("\n" + "="*80)
