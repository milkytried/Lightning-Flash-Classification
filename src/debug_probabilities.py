"""
Debug script to analyze probability distributions and test all thresholds.
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
import logging
import json
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

import sys
sys.path.insert(0, 'src')

from himawari_data_loader import HimawariPatchDataset
from model_arch import LightningResNet50


def calculate_metrics(tp, fp, tn, fn):
    """Calculate all metrics."""
    n = tp + fp + tn + fn
    if n == 0:
        return {k: 0.0 for k in ['accuracy', 'precision', 'recall_pod', 'f1', 'far', 'csi', 'specificity', 'tss', 'hss']}
    
    accuracy = (tp + tn) / n
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall_pod = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall_pod / (precision + recall_pod) if (precision + recall_pod) > 0 else 0
    far = fp / (tp + fp) if (tp + fp) > 0 else 0
    csi = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    tss = recall_pod + specificity - 1
    
    po = (tp + tn) / n
    pe = ((tp + fn) * (tp + fp) + (tn + fp) * (tn + fn)) / (n * n)
    hss = (po - pe) / (1 - pe) if (1 - pe) > 0 else 0
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall_pod': recall_pod,
        'f1': f1,
        'far': far,
        'csi': csi,
        'specificity': specificity,
        'tss': tss,
        'hss': hss,
        'tp': int(tp),
        'fp': int(fp),
        'tn': int(tn),
        'fn': int(fn),
        'predicted_positives': int(tp + fp),
        'predicted_negatives': int(tn + fn)
    }


def evaluate_at_threshold(probs, labels, threshold):
    """Evaluate at specific threshold."""
    predictions = (probs >= threshold).astype(int)
    
    tp = ((predictions == 1) & (labels == 1)).sum()
    fp = ((predictions == 1) & (labels == 0)).sum()
    tn = ((predictions == 0) & (labels == 0)).sum()
    fn = ((predictions == 0) & (labels == 1)).sum()
    
    return calculate_metrics(tp, fp, tn, fn)


def main():
    device = torch.device('cpu')
    
    # Load dataset info
    logger.info("[1] Loading dataset and model...")
    dataset_csv = 'data/processed/satellite_dataset.csv'
    
    df = pd.read_csv(dataset_csv)
    train_df = df[df['split'] == 'train']
    val_df = df[df['split'] == 'val']
    test_df = df[df['split'] == 'test']
    
    logger.info(f"Dataset splits:")
    logger.info(f"  Train: {len(train_df)} patches")
    logger.info(f"  Val:   {len(val_df)} patches")
    logger.info(f"  Test:  {len(test_df)} patches")
    
    # Load test dataset
    test_dataset = HimawariPatchDataset(dataset_csv=dataset_csv, split='test', augment=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)
    
    # Load model
    model_path = 'models/satellite_resnet50.pth'
    model = LightningResNet50(num_input_channels=3, num_classes=1, dropout_rate=0.5)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    # Generate test predictions
    logger.info("\n[2] Generating test predictions...")
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
    
    logger.info(f"Test probabilities: {len(test_probs)} samples")
    logger.info(f"  Min: {test_probs.min():.4f}")
    logger.info(f"  Max: {test_probs.max():.4f}")
    logger.info(f"  Mean: {test_probs.mean():.4f}")
    logger.info(f"  Std: {test_probs.std():.4f}")
    logger.info(f"  Median: {np.median(test_probs):.4f}")
    
    logger.info(f"\nTest labels: {len(test_labels)} samples")
    logger.info(f"  Positive (1): {(test_labels == 1).sum()}")
    logger.info(f"  Negative (0): {(test_labels == 0).sum()}")
    
    # Evaluate at all thresholds
    logger.info("\n[3] COMPREHENSIVE THRESHOLD TABLE (TEST SET)")
    logger.info("="*140)
    
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    results = []
    
    header = f"{'Thresh':<8} {'Pred+':<8} {'Pred-':<8} {'Acc':<8} {'Prec':<8} {'Recall':<8} {'F1':<8} {'FAR':<8} {'CSI':<8} {'TSS':<8} {'HSS':<8}"
    logger.info(header)
    logger.info("-"*140)
    
    for threshold in thresholds:
        metrics = evaluate_at_threshold(test_probs, test_labels, threshold)
        results.append({'threshold': threshold, **metrics})
        
        logger.info(f"{threshold:<8.1f} {metrics['predicted_positives']:<8d} {metrics['predicted_negatives']:<8d} "
                   f"{metrics['accuracy']:<8.4f} {metrics['precision']:<8.4f} {metrics['recall_pod']:<8.4f} "
                   f"{metrics['f1']:<8.4f} {metrics['far']:<8.4f} {metrics['csi']:<8.4f} "
                   f"{metrics['tss']:<8.4f} {metrics['hss']:<8.4f}")
    
    # Probability distributions by label
    logger.info("\n[4] PROBABILITY DISTRIBUTIONS BY LABEL")
    logger.info("-"*60)
    
    positive_probs = test_probs[test_labels == 1]
    negative_probs = test_probs[test_labels == 0]
    
    logger.info(f"Lightning (label=1) probabilities: {len(positive_probs)} samples")
    logger.info(f"  Min: {positive_probs.min():.4f}, Max: {positive_probs.max():.4f}")
    logger.info(f"  Mean: {positive_probs.mean():.4f}, Std: {positive_probs.std():.4f}")
    logger.info(f"  Median: {np.median(positive_probs):.4f}")
    
    logger.info(f"\nNo-Lightning (label=0) probabilities: {len(negative_probs)} samples")
    logger.info(f"  Min: {negative_probs.min():.4f}, Max: {negative_probs.max():.4f}")
    logger.info(f"  Mean: {negative_probs.mean():.4f}, Std: {negative_probs.std():.4f}")
    logger.info(f"  Median: {np.median(negative_probs):.4f}")
    
    # Check for probability reversal
    logger.info(f"\n[5] PROBABILITY CALIBRATION CHECK")
    logger.info("-"*60)
    
    # Ideally, positive samples should have higher probabilities
    mean_lightning_prob = positive_probs.mean()
    mean_no_lightning_prob = negative_probs.mean()
    
    logger.info(f"Mean probability for Lightning: {mean_lightning_prob:.4f}")
    logger.info(f"Mean probability for No-Lightning: {mean_no_lightning_prob:.4f}")
    
    if mean_lightning_prob > mean_no_lightning_prob:
        logger.info("✓ Probabilities are WELL-CALIBRATED (higher for lightning)")
    else:
        logger.warning("⚠ WARNING: Probabilities may be REVERSED (higher for no-lightning)")
    
    # Save results
    output_file = 'results/debug_threshold_table.json'
    Path('results').mkdir(exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\n✓ Results saved to {output_file}")


if __name__ == '__main__':
    main()
