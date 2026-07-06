"""
Compare leaky (metadata) vs clean (lat/lon/time) lightning models.
This evaluation quantifies the impact of label leakage on model performance.
"""

import torch
import numpy as np
import json
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, average_precision_score
)
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

sys.path.insert(0, '.')

from lightning_data_loader import create_lightning_loaders
from lightning_model import LightningMetadataClassifier


def evaluate_model(
    model_path: str,
    hdf5_path: str,
    feature_mode: str,
    device: torch.device,
) -> dict:
    """Evaluate a single model and return metrics."""
    
    logger.info(f"\n{'='*70}")
    logger.info(f"Evaluating {feature_mode.upper()} model: {Path(model_path).name}")
    logger.info(f"{'='*70}")
    
    # Load data
    loaders = create_lightning_loaders(hdf5_path, batch_size=512, feature_mode=feature_mode)
    test_loader = loaders['test']
    
    # Load model
    input_size = 4 if feature_mode == 'metadata' else 5
    model = LightningMetadataClassifier(input_size=input_size, hidden_size=256, dropout=0.3)
    model.load_state_dict(torch.load(model_path, map_location=device))
    
    # Legacy models use nn.Sequential, load directly
    checkpoint = torch.load(model_path, map_location=device)
    if 'net.0.weight' in checkpoint:
        # Build nn.Sequential model
        model = torch.nn.Sequential(
            torch.nn.Linear(input_size, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1),
            torch.nn.Sigmoid(),
        )
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    logger.info(f"Model loaded: {model_path}")
    
    # Collect predictions
    predictions_list = []
    labels_list = []
    
    with torch.no_grad():
        for batch_idx, (features, labels) in enumerate(test_loader):
            features = features.to(device)
            predictions = model(features)
            
            predictions_list.append(predictions.cpu().numpy().flatten())
            labels_list.append(labels.numpy())
            
            if (batch_idx + 1) % 200 == 0:
                logger.info(f"  Processed {batch_idx + 1}/{len(test_loader)} batches")
    
    # Concatenate all predictions
    all_predictions = np.concatenate(predictions_list)
    all_labels = np.concatenate(labels_list)
    
    # Binary predictions at threshold 0.5
    binary_predictions = (all_predictions >= 0.5).astype(int)
    
    # Compute metrics
    accuracy = accuracy_score(all_labels, binary_predictions)
    precision = precision_score(all_labels, binary_predictions, zero_division=0)
    recall = recall_score(all_labels, binary_predictions, zero_division=0)
    f1 = f1_score(all_labels, binary_predictions, zero_division=0)
    
    # PR-AUC for minority class (no-strike, label=0)
    pr_auc = average_precision_score((all_labels == 0).astype(int), 1.0 - all_predictions)
    
    # ROC-AUC
    try:
        roc_auc = roc_auc_score(all_labels, all_predictions)
    except:
        roc_auc = 0.0
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, binary_predictions)
    tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
    
    # Meteorological metrics
    pod = tp / (tp + fn) if (tp + fn) > 0 else 0  # Probability of Detection
    far = fp / (fp + tp) if (fp + tp) > 0 else 0  # False Alarm Ratio
    
    # Class distribution
    n_neg = (all_labels == 0).sum()
    n_pos = (all_labels == 1).sum()
    
    # Log results
    logger.info(f"\nTest Set Metrics:")
    logger.info(f"  Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    logger.info(f"  Precision: {precision:.4f}")
    logger.info(f"  Recall:    {recall:.4f}")
    logger.info(f"  F1-Score:  {f1:.4f}")
    logger.info(f"  ROC-AUC:   {roc_auc:.4f}")
    logger.info(f"  PR-AUC (no-strike): {pr_auc:.4f}")
    logger.info(f"\nConfusion Matrix (threshold=0.5):")
    logger.info(f"  True Neg:   {tn:,}")
    logger.info(f"  False Pos:  {fp:,}")
    logger.info(f"  False Neg:  {fn:,}")
    logger.info(f"  True Pos:   {tp:,}")
    logger.info(f"\nMeteorological Metrics:")
    logger.info(f"  POD (Probability of Detection): {pod:.4f}")
    logger.info(f"  FAR (False Alarm Ratio):        {far:.4f}")
    logger.info(f"\nClass Distribution:")
    logger.info(f"  Negatives (no-strike):   {n_neg:,} ({n_neg/(n_neg+n_pos)*100:.2f}%)")
    logger.info(f"  Positives (strike):      {n_pos:,} ({n_pos/(n_neg+n_pos)*100:.2f}%)")
    
    return {
        'feature_mode': feature_mode,
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'roc_auc': float(roc_auc),
        'pr_auc_no_strike': float(pr_auc),
        'pod': float(pod),
        'far': float(far),
        'confusion_matrix': {
            'true_neg': int(tn),
            'false_pos': int(fp),
            'false_neg': int(fn),
            'true_pos': int(tp),
        },
        'class_distribution': {
            'negatives': int(n_neg),
            'positives': int(n_pos),
            'pct_negative': float(n_neg / (n_neg + n_pos)),
        },
        'test_samples': len(all_labels),
    }


def compare_models():
    """Evaluate both models and generate comparison."""
    
    logger.info("="*70)
    logger.info("LEAKAGE IMPACT ANALYSIS: METADATA vs CLEAN MODELS")
    logger.info("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}\n")
    
    hdf5_path = "data/processed/lightning_dataset.h5"
    
    # Evaluate both models
    metadata_metrics = evaluate_model(
        model_path="models/lightning_classifier_metadata_probe.pth",
        hdf5_path=hdf5_path,
        feature_mode='metadata',
        device=device,
    )
    
    clean_metrics = evaluate_model(
        model_path="models/lightning_classifier_clean_probe.pth",
        hdf5_path=hdf5_path,
        feature_mode='clean',
        device=device,
    )
    
    # Compare results
    logger.info(f"\n{'='*70}")
    logger.info("COMPARISON: IMPACT OF LEAKAGE")
    logger.info(f"{'='*70}")
    
    logger.info(f"\n{'Metric':<20} {'Metadata (Leaky)':<20} {'Clean':<20} {'Delta':<15}")
    logger.info("-" * 75)
    
    for key in ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc', 'pr_auc_no_strike', 'pod', 'far']:
        meta_val = metadata_metrics[key]
        clean_val = clean_metrics[key]
        delta = meta_val - clean_val
        logger.info(f"{key:<20} {meta_val:<20.4f} {clean_val:<20.4f} {delta:+.4f}")
    
    # Interpretation
    logger.info(f"\n{'='*70}")
    logger.info("INTERPRETATION")
    logger.info(f"{'='*70}")
    
    recall_drop = metadata_metrics['recall'] - clean_metrics['recall']
    logger.info(f"\nRecall drop (metadata → clean): {recall_drop:+.4f}")
    
    if recall_drop > 0.5:
        logger.info("⚠️  LARGE DROP: Metadata model relied heavily on leaky features (amplitude, strike_type)")
        logger.info("   This confirms that amplitude and strike_type are consequences of strike detection,")
        logger.info("   not independent predictors.")
    elif recall_drop > 0.1:
        logger.info("⚠️  MODERATE DROP: Some leakage detected, but location/time have partial signal")
    else:
        logger.info("✅ SMALL DROP: Location and time features alone are nearly as predictive")
    
    logger.info(f"\nClean model performance:")
    if clean_metrics['recall'] > 0.5:
        logger.info("✅ Clean model achieves >50% recall: Location/time have real predictive value")
    else:
        logger.info("❌ Clean model has low recall: Lightning prediction is inherently difficult with location/time only")
    
    # Save comparison to JSON
    comparison = {
        'timestamp': str(np.datetime64('now')),
        'metadata_probe': metadata_metrics,
        'clean_probe': clean_metrics,
        'interpretation': {
            'recall_drop': float(recall_drop),
            'is_leakage_severe': recall_drop > 0.5,
            'clean_model_viable': clean_metrics['recall'] > 0.5,
        }
    }
    
    output_path = Path("models/leakage_comparison.json")
    with open(output_path, 'w') as f:
        json.dump(comparison, f, indent=2)
    logger.info(f"\nComparison saved to: {output_path}")
    
    return metadata_metrics, clean_metrics, comparison


if __name__ == '__main__':
    meta, clean, comp = compare_models()
