"""
Comprehensive test set evaluation for fresh trained satellite checkpoint.
Reports all requested metrics on the corrected unseen test set.
Run this after train_fresh_optimized.py completes.
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import json
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, roc_curve
)
from tqdm import tqdm
from datetime import datetime

from src.model_arch import LightningResNet50
from src.himawari_data_loader import HimawariPatchDataset
from torch.utils.data import DataLoader

print("\n" + "="*80)
print("TEST SET EVALUATION - FRESH CHECKPOINT")
print("="*80)

# Device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")

# Paths
checkpoint_path = 'models/satellite_resnet50_fresh.pth'
dataset_csv = 'data/processed/satellite_dataset.csv'
output_path = 'models/test_evaluation_fresh.json'

# Load model
print(f"\n[1/4] Loading model...")
model = LightningResNet50(num_input_channels=3, num_classes=1, dropout_rate=0.5)

try:
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model = model.to(device)
    model.eval()
    print(f"[OK] Model loaded: {checkpoint_path}")
except Exception as e:
    print(f"[ERROR] Failed to load model: {e}")
    print(f"  Make sure train_fresh_optimized.py completed first")
    exit(1)

# Load test data
print(f"\n[2/4] Loading test data...")
df = pd.read_csv(dataset_csv)
test_df = df[df['split'] == 'test'].copy()
print(f"[OK] Test samples: {len(test_df):,}")
print(f"  Positive (lightning): {(test_df['label'] == 1).sum():,}")
print(f"  Negative (no lightning): {(test_df['label'] == 0).sum():,}")

# Create test dataset and loader
test_dataset = HimawariPatchDataset(
    dataset_csv=dataset_csv,
    split='test',
    augment=False
)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=0)
print(f"[OK] Test batches (batch_size=256): {len(test_loader)}")

# Run inference
print(f"\n[3/4] Running inference on test set...")
all_probs = []
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in tqdm(test_loader, desc='Inference'):
        images = images.to(device)
        outputs = model(images).squeeze()
        probs = torch.sigmoid(outputs).cpu().numpy()
        preds = (probs > 0.5).astype(int)
        
        all_probs.extend(probs)
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())

all_probs = np.array(all_probs)
all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

print(f"[OK] Inference complete: {len(all_labels):,} samples")

# Compute metrics
print(f"\n[4/4] Computing metrics...")

# Core metrics
accuracy = accuracy_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds, zero_division=0)
recall = recall_score(all_labels, all_preds, zero_division=0)  # POD = Recall
f1 = f1_score(all_labels, all_preds, zero_division=0)
roc_auc = roc_auc_score(all_labels, all_probs)

# Confusion matrix
tn, fp, fn, tp = confusion_matrix(all_labels, all_preds).ravel()

# Derived metrics
far = fp / (fp + tn) if (fp + tn) > 0 else 0  # False Alarm Ratio
csi = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0  # Critical Success Index / Threat Score
tss = (tp * tn - fp * fn) / ((tp + fn) * (fp + tn)) if ((tp + fn) * (fp + tn)) > 0 else 0  # True Skill Statistic
hss = (tp + tn - fp - fn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0  # Heidke Skill Score

# Print results
print(f"\n{'='*80}")
print("TEST SET METRICS (threshold=0.5)")
print(f"{'='*80}")

print(f"\n[ACCURACY & LOSS]")
print(f"  Accuracy:                {accuracy:.4f}")

print(f"\n[DETECTION PERFORMANCE]")
print(f"  Precision:               {precision:.4f}")
print(f"  Recall / POD:            {recall:.4f}  (Probability of Detection)")
print(f"  F1-Score:                {f1:.4f}")

print(f"\n[DISCRIMINATION]")
print(f"  ROC-AUC:                 {roc_auc:.4f}")

print(f"\n[WEATHER/VERIFICATION METRICS]")
print(f"  FAR:                     {far:.4f}  (False Alarm Ratio)")
print(f"  CSI / Threat Score:      {csi:.4f}  (Critical Success Index)")
print(f"  TSS:                     {tss:.4f}  (True Skill Statistic)")
print(f"  HSS:                     {hss:.4f}  (Heidke Skill Score)")

print(f"\n[CONFUSION MATRIX]")
print(f"  True Positives (TP):     {tp:,}    (correctly detected lightning)")
print(f"  False Positives (FP):    {fp:,}    (false alarms)")
print(f"  False Negatives (FN):    {fn:,}    (missed lightning)")
print(f"  True Negatives (TN):     {tn:,}    (correctly rejected non-events)")

print(f"\n[SAMPLE STATISTICS]")
print(f"  Total test samples:      {len(all_labels):,}")
print(f"  Correct predictions:     {(all_preds == all_labels).sum():,} / {len(all_labels):,}")
print(f"  Model predicted positive: {(all_preds == 1).sum():,}")
print(f"  Model predicted negative: {(all_preds == 0).sum():,}")

# Save results
results = {
    'timestamp': datetime.now().isoformat(),
    'checkpoint': checkpoint_path,
    'threshold': 0.5,
    'test_samples': len(all_labels),
    'test_samples_positive': int((all_labels == 1).sum()),
    'test_samples_negative': int((all_labels == 0).sum()),
    'metrics': {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall_pod': float(recall),
        'f1_score': float(f1),
        'roc_auc': float(roc_auc),
        'far': float(far),
        'csi_threat_score': float(csi),
        'tss': float(tss),
        'hss': float(hss),
    },
    'confusion_matrix': {
        'tp': int(tp),
        'fp': int(fp),
        'fn': int(fn),
        'tn': int(tn),
    }
}

with open(output_path, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n[OK] Results saved to: {output_path}")
print(f"{'='*80}\n")
