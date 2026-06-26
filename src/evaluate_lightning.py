"""
Evaluate lightning detection model on test set.
Compute honest minority-class metrics and PR-AUC.
"""

import torch
import numpy as np
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, average_precision_score
)
import matplotlib.pyplot as plt
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add src to path
sys.path.insert(0, '.')


def evaluate_lightning_model(
    model_path: str = "../models/lightning_classifier.pth",
    hdf5_path: str = "../data/processed/lightning_dataset.h5",
    device_str: str = 'cpu',
    feature_mode: str = 'metadata',
):
    """Evaluate trained lightning detection model."""
    
    from lightning_data_loader import create_lightning_loaders
    from lightning_model import LightningMetadataClassifier
    
    logger.info("=" * 70)
    logger.info("EVALUATING LIGHTNING DETECTION MODEL")
    logger.info("=" * 70)
    
    device = torch.device(device_str)
    
    # Load data
    logger.info("\nLoading test data...")
    loaders = create_lightning_loaders(hdf5_path, batch_size=512, feature_mode=feature_mode)
    test_loader = loaders['test']
    logger.info(f"  Test batches: {len(test_loader)}")
    
    # Load model
    logger.info("\nLoading model...")
    input_size = 4 if feature_mode == 'metadata' else 5
    model = LightningMetadataClassifier(input_size=input_size, hidden_size=256, dropout=0.3)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    logger.info(f"  Model loaded from: {model_path}")
    
    # Evaluate
    logger.info("\nEvaluating on test set...")
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
    logger.info("\n" + "=" * 70)
    logger.info("TEST SET METRICS")
    logger.info("=" * 70)
    
    accuracy = accuracy_score(all_labels, binary_predictions)
    precision = precision_score(all_labels, binary_predictions, zero_division=0)
    recall = recall_score(all_labels, binary_predictions, zero_division=0)
    f1 = f1_score(all_labels, binary_predictions, zero_division=0)
    pr_auc = average_precision_score((all_labels == 0).astype(int), 1.0 - all_predictions)
    
    # ROC-AUC (need at least one positive and one negative)
    try:
        roc_auc = roc_auc_score(all_labels, all_predictions)
    except:
        roc_auc = 0.0
    
    logger.info(f"\nAccuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall:    {recall:.4f} {'✅ PASS' if recall >= 0.85 else '❌ FAIL'} (target: ≥0.85)")
    logger.info(f"F1-Score:  {f1:.4f}")
    logger.info(f"ROC-AUC:   {roc_auc:.4f}")
    logger.info(f"PR-AUC(no-strike): {pr_auc:.4f}")
    
    # Class distribution
    logger.info(f"\nPredicted class distribution:")
    logger.info(f"  Negative (0): {(binary_predictions == 0).sum():,} ({(binary_predictions == 0).sum()/len(binary_predictions)*100:.2f}%)")
    logger.info(f"  Positive (1): {(binary_predictions == 1).sum():,} ({(binary_predictions == 1).sum()/len(binary_predictions)*100:.2f}%)")
    
    logger.info(f"\nTrue class distribution:")
    logger.info(f"  Negative (0): {(all_labels == 0).sum():,} ({(all_labels == 0).sum()/len(all_labels)*100:.2f}%)")
    logger.info(f"  Positive (1): {(all_labels == 1).sum():,} ({(all_labels == 1).sum()/len(all_labels)*100:.2f}%)")
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, binary_predictions)
    logger.info(f"\nConfusion Matrix:")
    logger.info(f"  True Neg:  {cm[0, 0]:,}")
    logger.info(f"  False Pos: {cm[0, 1]:,}")
    logger.info(f"  False Neg: {cm[1, 0]:,}")
    logger.info(f"  True Pos:  {cm[1, 1]:,}")
    
    # Meteorological metrics
    pod = cm[1, 1] / (cm[1, 1] + cm[1, 0]) if (cm[1, 1] + cm[1, 0]) > 0 else 0  # Probability of Detection
    far = cm[0, 1] / (cm[0, 1] + cm[1, 1]) if (cm[0, 1] + cm[1, 1]) > 0 else 0  # False Alarm Ratio
    
    logger.info(f"\nMeteorological Metrics:")
    logger.info(f"  POD (Probability of Detection): {pod:.4f}")
    logger.info(f"  FAR (False Alarm Ratio):        {far:.4f}")
    
    logger.info("\n" + "=" * 70)
    if recall >= 0.85:
        logger.info("✅ MODEL MEETS RECALL REQUIREMENT (≥85%)")
    else:
        logger.info("⚠️  MODEL DOES NOT MEET RECALL REQUIREMENT (<85%)")
    logger.info("=" * 70)
    
    # Return metrics dict
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'pr_auc_no_strike': pr_auc,
        'pod': pod,
        'far': far,
        'test_samples': len(all_labels),
        'feature_mode': feature_mode,
    }
    
    return metrics, all_predictions, all_labels


if __name__ == "__main__":
    import os
    
    # Adjust paths if running from src directory
    cwd = os.getcwd()
    if os.path.basename(cwd) == 'src':
        model_path = "../models/lightning_classifier.pth"
        hdf5_path = "../data/processed/lightning_dataset.h5"
    else:
        model_path = "models/lightning_classifier.pth"
        hdf5_path = "data/processed/lightning_dataset.h5"
    
    try:
        metrics, predictions, labels = evaluate_lightning_model(
            model_path=model_path,
            hdf5_path=hdf5_path,
        )
    except FileNotFoundError as e:
        logger.error(f"Model not found: {e}")
        logger.error("Train the model first with: python src/train_lightning.py")
