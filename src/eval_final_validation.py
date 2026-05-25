"""
Comprehensive Evaluation: Threshold Tuning + Proper Metrics + Error Analysis

This script performs:
1. Threshold tuning on validation set (0.3 to 0.9)
2. Computes correct metrics:
   - Accuracy, Precision, Recall/POD, F1
   - FAR (False Alarm Ratio)
   - CSI (Threat Score) = TP/(TP+FP+FN)
   - Specificity = TN/(TN+FP)
   - TSS (True Skill Statistic) = POD + Specificity - 1
   - HSS (Heidke Skill Score)
3. Selects best threshold from validation set
4. Reports test set results with all metrics
5. Generates error analysis visualizations
6. Verifies no data leakage
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
from pathlib import Path
from PIL import Image
from torch.utils.data import DataLoader
import logging
import json
from sklearn.metrics import roc_curve, auc, confusion_matrix as sk_confusion_matrix
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add src to path
import sys
sys.path.insert(0, 'src')

from himawari_data_loader import HimawariPatchDataset
from model_arch import LightningResNet50


def calculate_metrics(tp, fp, tn, fn):
    """Calculate all metrics with CORRECT DEFINITIONS."""
    n = tp + fp + tn + fn
    
    if n == 0:
        return {k: 0.0 for k in ['accuracy', 'precision', 'recall_pod', 'f1', 'far', 'csi', 'specificity', 'tss', 'hss']}
    
    # Basic metrics
    accuracy = (tp + tn) / n
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall_pod = tp / (tp + fn) if (tp + fn) > 0 else 0  # POD = Probability of Detection = Recall
    f1 = 2 * precision * recall_pod / (precision + recall_pod) if (precision + recall_pod) > 0 else 0
    
    # Derived metrics
    far = fp / (tp + fp) if (tp + fp) > 0 else 0  # FAR = False Alarm Ratio
    csi = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0  # CSI = Critical Success Index = Threat Score
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    tss = recall_pod + specificity - 1  # TSS = True Skill Statistic = POD + Specificity - 1
    
    # HSS = (po - pe) / (1 - pe) where po = observed accuracy, pe = expected by chance
    po = (tp + tn) / n
    pe = ((tp + fn) * (tp + fp) + (tn + fp) * (tn + fn)) / (n * n)
    hss = (po - pe) / (1 - pe) if (1 - pe) > 0 else 0
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall_pod': recall_pod,
        'f1': f1,
        'far': far,
        'csi': csi,  # CSI = Threat Score
        'specificity': specificity,
        'tss': tss,  # TRUE SKILL STATISTIC (correct formula)
        'hss': hss,
        'tp': tp,
        'fp': fp,
        'tn': tn,
        'fn': fn
    }


def evaluate_at_threshold(probs, labels, threshold):
    """Evaluate at specific probability threshold."""
    predictions = (probs >= threshold).astype(int)
    
    tp = ((predictions == 1) & (labels == 1)).sum()
    fp = ((predictions == 1) & (labels == 0)).sum()
    tn = ((predictions == 0) & (labels == 0)).sum()
    fn = ((predictions == 0) & (labels == 1)).sum()
    
    return calculate_metrics(tp, fp, tn, fn)


def main():
    logger.info("="*80)
    logger.info("COMPREHENSIVE EVALUATION: THRESHOLD TUNING + PROPER METRICS + ERROR ANALYSIS")
    logger.info("="*80)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Device: {device}")
    
    # Load dataset
    logger.info("\n[1] Loading dataset CSV...")
    dataset_csv = 'data/processed/satellite_dataset.csv'
    df = pd.read_csv(dataset_csv)
    
    logger.info(f"Total patches: {len(df)}")
    logger.info(f"Splits: {df['split'].value_counts().to_dict()}")
    
    # Verify image-level split (no leakage)
    logger.info("\n[2] Verifying image-level split (no data leakage)...")
    train_df = df[df['split'] == 'train']
    val_df = df[df['split'] == 'val']
    test_df = df[df['split'] == 'test']
    
    # Extract PNG source from path
    def get_png_source(path):
        parts = Path(path).parts
        # Find parent directory that is "train", "val", or "test"
        try:
            for i, part in enumerate(parts):
                if part in ['train', 'val', 'test'] and i > 0:
                    return parts[i-1]  # PNG directory name
        except:
            pass
        return None
    
    train_pngs = set(get_png_source(p) for p in train_df['path'])
    val_pngs = set(get_png_source(p) for p in val_df['path'])
    test_pngs = set(get_png_source(p) for p in test_df['path'])
    
    logger.info(f"  Train PNG sources: {len(train_pngs)}")
    logger.info(f"  Val PNG sources: {len(val_pngs)}")
    logger.info(f"  Test PNG sources: {len(test_pngs)}")
    
    # Verify no overlap
    if train_pngs & val_pngs:
        logger.warning(f"  ⚠ Train-Val overlap detected!")
    if train_pngs & test_pngs:
        logger.warning(f"  ⚠ Train-Test overlap detected!")
    if val_pngs & test_pngs:
        logger.warning(f"  ⚠ Val-Test overlap detected!")
    
    if not (train_pngs & val_pngs) and not (train_pngs & test_pngs) and not (val_pngs & test_pngs):
        logger.info(f"  ✓ No PNG overlap: proper image-level split confirmed")
    
    # Load datasets
    logger.info("\n[3] Loading validation and test datasets...")
    val_dataset = HimawariPatchDataset(dataset_csv=dataset_csv, split='val', augment=False)
    test_dataset = HimawariPatchDataset(dataset_csv=dataset_csv, split='test', augment=False)
    
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)
    
    # Load model
    logger.info("\n[4] Loading model...")
    model_path = 'models/satellite_resnet50.pth'
    model = LightningResNet50(num_input_channels=3, num_classes=1, dropout_rate=0.5)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    # Generate predictions
    logger.info("\n[5] Generating predictions...")
    logger.info("  Validation set...")
    val_probs = []
    val_labels = []
    
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            outputs = model(images)
            val_probs.extend(outputs.cpu().numpy().squeeze())
            val_labels.extend(labels.numpy())
    
    val_probs = np.array(val_probs)
    val_labels = np.array(val_labels)
    logger.info(f"  Validation: {len(val_probs)} samples")
    
    logger.info("  Test set...")
    test_probs = []
    test_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            test_probs.extend(outputs.cpu().numpy().squeeze())
            test_labels.extend(labels.numpy())
    
    test_probs = np.array(test_probs)
    test_labels = np.array(test_labels)
    logger.info(f"  Test: {len(test_probs)} samples")
    
    # THRESHOLD TUNING
    logger.info("\n[6] THRESHOLD TUNING ON VALIDATION SET")
    logger.info("="*80)
    
    thresholds_to_test = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    threshold_results = []
    
    logger.info(f"\n{'Thresh':<8} {'Acc':<8} {'Prec':<8} {'Recall':<8} {'F1':<8} {'FAR':<8} {'CSI':<8} {'TSS':<8} {'HSS':<8}")
    logger.info("-" * 80)
    
    for threshold in thresholds_to_test:
        metrics = evaluate_at_threshold(val_probs, val_labels, threshold)
        threshold_results.append({'threshold': threshold, **metrics})
        
        logger.info(f"{threshold:<8.1f} {metrics['accuracy']:<8.4f} {metrics['precision']:<8.4f} "
                   f"{metrics['recall_pod']:<8.4f} {metrics['f1']:<8.4f} {metrics['far']:<8.4f} "
                   f"{metrics['csi']:<8.4f} {metrics['tss']:<8.4f} {metrics['hss']:<8.4f}")
    
    # Select best threshold based on F1 (or CSI)
    best_result = max(threshold_results, key=lambda x: x['f1'])
    best_threshold = best_result['threshold']
    
    logger.info(f"\n✓ Best threshold (max F1): {best_threshold:.1f}")
    logger.info(f"  F1: {best_result['f1']:.4f}, Recall: {best_result['recall_pod']:.4f}, Precision: {best_result['precision']:.4f}")
    
    # TEST SET EVALUATION
    logger.info("\n[7] TEST SET EVALUATION WITH BEST THRESHOLD")
    logger.info("="*80)
    
    test_metrics = evaluate_at_threshold(test_probs, test_labels, best_threshold)
    
    logger.info(f"\nTest Set Metrics (Threshold = {best_threshold}):")
    logger.info(f"  Accuracy:                {test_metrics['accuracy']:.4f}")
    logger.info(f"  Precision:               {test_metrics['precision']:.4f}")
    logger.info(f"  Recall / POD:            {test_metrics['recall_pod']:.4f}")
    logger.info(f"  F1-Score:                {test_metrics['f1']:.4f}")
    logger.info(f"  FAR (False Alarm Ratio): {test_metrics['far']:.4f}")
    logger.info(f"  CSI (Threat Score):      {test_metrics['csi']:.4f}")
    logger.info(f"  Specificity:             {test_metrics['specificity']:.4f}")
    logger.info(f"  TSS (True Skill Stat):   {test_metrics['tss']:.4f}")
    logger.info(f"  HSS (Heidke Skill Sc):   {test_metrics['hss']:.4f}")
    
    logger.info(f"\nConfusion Matrix:")
    logger.info(f"  TP: {test_metrics['tp']:6d}  |  FP: {test_metrics['fp']:6d}")
    logger.info(f"  FN: {test_metrics['fn']:6d}  |  TN: {test_metrics['tn']:6d}")
    
    # VISUALIZATIONS
    logger.info("\n[8] Generating visualizations...")
    Path('results').mkdir(exist_ok=True)
    
    # Threshold tuning table
    fig, ax = plt.subplots(figsize=(14, 6))
    threshold_df = pd.DataFrame(threshold_results)
    threshold_df_display = threshold_df[['threshold', 'accuracy', 'precision', 'recall_pod', 'f1', 'far', 'csi', 'tss', 'hss']].copy()
    threshold_df_display = threshold_df_display.round(4)
    
    ax.axis('tight')
    ax.axis('off')
    table = ax.table(cellText=threshold_df_display.values,
                    colLabels=['Threshold', 'Accuracy', 'Precision', 'Recall', 'F1', 'FAR', 'CSI', 'TSS', 'HSS'],
                    cellLoc='center',
                    loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # Highlight best row
    for i, threshold in enumerate(threshold_df['threshold']):
        if abs(threshold - best_threshold) < 0.01:
            for j in range(9):
                table[(i+1, j)].set_facecolor('#90EE90')
    
    plt.title(f'Threshold Tuning Results on Validation Set (Best: {best_threshold})', 
             fontsize=12, fontweight='bold')
    plt.savefig('results/01_threshold_tuning_table.png', dpi=100, bbox_inches='tight')
    logger.info("  Saved: results/01_threshold_tuning_table.png")
    plt.close()
    
    # Probability histogram
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(test_probs[test_labels==0], bins=40, alpha=0.6, label='No-Lightning (True 0)', color='blue')
    ax.hist(test_probs[test_labels==1], bins=40, alpha=0.6, label='Lightning (True 1)', color='red')
    ax.axvline(best_threshold, color='green', linestyle='--', linewidth=2, label=f'Threshold={best_threshold}')
    ax.set_xlabel('Predicted Probability')
    ax.set_ylabel('Count')
    ax.set_title(f'Probability Distribution (Test Set, Threshold={best_threshold})')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('results/02_probability_histogram.png', dpi=100, bbox_inches='tight')
    logger.info("  Saved: results/02_probability_histogram.png")
    plt.close()
    
    # ROC curve
    fpr, tpr, _ = roc_curve(test_labels, test_probs)
    roc_auc = auc(fpr, tpr)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, 'b-', linewidth=2, label=f'ResNet-50 (AUC={roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier (AUC=0.5)')
    ax.set_xlabel('False Positive Rate (1 - Specificity)')
    ax.set_ylabel('True Positive Rate (Recall)')
    ax.set_title(f'ROC Curve (Test Set, AUC={roc_auc:.3f})')
    ax.legend(loc='lower right')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('results/03_roc_curve.png', dpi=100, bbox_inches='tight')
    logger.info("  Saved: results/03_roc_curve.png")
    plt.close()
    
    # Confusion matrix
    preds = (test_probs >= best_threshold).astype(int)
    cm = sk_confusion_matrix(test_labels, preds)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Predicted No-Lightning', 'Predicted Lightning'],
                yticklabels=['True No-Lightning', 'True Lightning'],
                cbar_kws={'label': 'Count'})
    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')
    ax.set_title(f'Confusion Matrix (Test Set, Threshold={best_threshold})')
    plt.tight_layout()
    plt.savefig('results/04_confusion_matrix.png', dpi=100, bbox_inches='tight')
    logger.info("  Saved: results/04_confusion_matrix.png")
    plt.close()
    
    # Save comprehensive results
    logger.info("\n[9] Saving results...")
    
    # Convert numpy types to Python types for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        return obj
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'dataset_info': {
            'total_patches': len(df),
            'train_patches': len(train_df),
            'val_patches': len(val_df),
            'test_patches': len(test_df)
        },
        'best_threshold': float(best_threshold),
        'threshold_tuning': convert_to_serializable(threshold_results),
        'test_metrics': convert_to_serializable(test_metrics),
        'roc_auc': float(roc_auc),
        'model_path': model_path
    }
    
    with open('results/comprehensive_evaluation.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info("  Saved: results/comprehensive_evaluation.json")
    
    logger.info("\n" + "="*80)
    logger.info("EVALUATION COMPLETE")
    logger.info("="*80)
    logger.info(f"\nKey Results:")
    logger.info(f"  Best threshold (from validation): {best_threshold}")
    logger.info(f"  Test accuracy: {test_metrics['accuracy']:.4f}")
    logger.info(f"  Test recall (POD): {test_metrics['recall_pod']:.4f}")
    logger.info(f"  Test CSI (Threat Score): {test_metrics['csi']:.4f}")
    logger.info(f"  Test TSS (True Skill Stat): {test_metrics['tss']:.4f}")
    logger.info(f"  ROC-AUC: {roc_auc:.4f}")
    logger.info(f"\nAll visualizations saved to results/")


if __name__ == '__main__':
    main()
