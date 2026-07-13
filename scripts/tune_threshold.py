"""Reproduce threshold tuning for the earlier 11-PNG Himawari-8 baseline.

This script generated the validation-selected 0.55 threshold and the Table 5.2
baseline metrics (accuracy 0.8765, precision 0.8601, recall 0.8993, F1 0.8792,
ROC-AUC 0.9199). It is not part of the final aligned Himawari-9 workflow."""

import json
import torch
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tqdm import tqdm
from src.model_arch import LightningResNet50
from src.himawari_data_loader import create_himawari_loaders

def calculate_metrics(y_true, y_pred, threshold=0.5):
    """Calculate classification metrics at given threshold."""
    y_pred_binary = (y_pred > threshold).astype(int)
    
    tp = np.sum((y_pred_binary == 1) & (y_true == 1))
    fp = np.sum((y_pred_binary == 1) & (y_true == 0))
    tn = np.sum((y_pred_binary == 0) & (y_true == 0))
    fn = np.sum((y_pred_binary == 0) & (y_true == 1))
    
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # FAR = False Alarm Ratio = FP / (TP + FP)
    far = fp / (tp + fp) if (tp + fp) > 0 else 0
    
    # CSI = Critical Success Index = TP / (TP + FP + FN)
    csi = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0
    
    # TSS = True Skill Statistic = (TP/(TP+FN)) - (FP/(FP+TN))
    pod = tp / (tp + fn) if (tp + fn) > 0 else 0
    pofd = fp / (fp + tn) if (fp + tn) > 0 else 0
    tss = pod - pofd
    
    # HSS = Heidke Skill Score = 2(TP*TN - FP*FN) / ((TP+FN)*(FN+TN) + (TP+FP)*(FP+TN))
    n = tp + tn + fp + fn
    po = (tp + tn) / n if n > 0 else 0
    pe = ((tp + fn) * (tp + fp) + (tn + fp) * (tn + fn)) / (n * n) if n > 0 else 0
    hss = (po - pe) / (1 - pe) if (1 - pe) > 0 else 0
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'far': far,
        'csi': csi,
        'tss': tss,
        'hss': hss,
        'tp': int(tp),
        'fp': int(fp),
        'tn': int(tn),
        'fn': int(fn),
    }

def get_predictions(model_path, data_loader, desc='', device='cpu'):
    """Get predictions on dataset."""
    model = LightningResNet50()
    model = model.to(device)
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc=desc, disable=(desc == '')):
            images, labels = batch
            images = images.to(device)
            
            logits = model(images)
            probs = logits.cpu().numpy().flatten()
            
            all_preds.extend(probs)
            all_labels.extend(labels.numpy().flatten())
    
    return np.array(all_preds), np.array(all_labels)

def main():
    """Tune threshold on validation set."""
    device = 'cpu'
    model_path = 'models/satellite_resnet50_fresh.pth'
    
    print("=" * 100)
    print("THRESHOLD TUNING ON VALIDATION SET")
    print("=" * 100)
    
    # Load data loaders
    print("\n[1/4] Loading data...")
    loaders = create_himawari_loaders(
        dataset_csv='data/processed/satellite_dataset.csv',
        batch_size=256,
        num_workers=0
    )
    val_loader = loaders['val']
    test_loader = loaders['test']
    print("      OK - Data loaded")
    
    # Get validation predictions
    print("\n[2/4] Getting validation predictions...")
    val_preds, val_labels = get_predictions(model_path, val_loader, desc='Validation inference', device=device)
    print(f"      OK - {len(val_preds):,} validation samples")
    
    # Tune thresholds
    print("\n[3/4] Tuning thresholds on validation set...")
    print("-" * 120)
    print(f"{'Thresh':<10} {'Acc':<10} {'Prec':<10} {'Recall':<10} {'F1':<10} {'FAR':<10} {'CSI':<10} {'TSS':<10} {'HSS':<10}")
    print("-" * 120)
    
    best_threshold = 0.5
    best_f1 = 0
    results = {}
    
    thresholds = np.arange(0.10, 0.96, 0.05)
    
    for threshold in thresholds:
        metrics = calculate_metrics(val_labels, val_preds, threshold)
        results[float(threshold)] = metrics
        
        print(f"{threshold:<10.2f} {metrics['accuracy']:<10.4f} {metrics['precision']:<10.4f} {metrics['recall']:<10.4f} "
              f"{metrics['f1']:<10.4f} {metrics['far']:<10.4f} {metrics['csi']:<10.4f} {metrics['tss']:<10.4f} {metrics['hss']:<10.4f}")
        
        # Select threshold with best F1 (balances precision and recall)
        if metrics['f1'] > best_f1:
            best_f1 = metrics['f1']
            best_threshold = threshold
    
    print("-" * 120)
    print(f"\n✅ Best threshold on validation set: {best_threshold:.2f} (F1 = {best_f1:.4f})")
    
    # Get test predictions and apply best threshold
    print(f"\n[4/4] Applying threshold {best_threshold:.2f} to test set...")
    test_preds, test_labels = get_predictions(model_path, test_loader, desc='Test inference', device=device)
    test_metrics = calculate_metrics(test_labels, test_preds, best_threshold)
    
    # Calculate ROC-AUC (uses probabilities, not threshold-dependent)
    from sklearn.metrics import roc_auc_score
    roc_auc = roc_auc_score(test_labels, test_preds)
    test_metrics['roc_auc'] = roc_auc
    
    print("\n" + "="*100)
    print("FINAL TEST SET EVALUATION (with tuned threshold)")
    print("="*100)
    print(f"\nThreshold: {best_threshold:.2f}\n")
    print(f"Accuracy:     {test_metrics['accuracy']:.4f} ({test_metrics['accuracy']*100:.2f}%)")
    print(f"Precision:    {test_metrics['precision']:.4f} ({test_metrics['precision']*100:.2f}%)")
    print(f"Recall/POD:   {test_metrics['recall']:.4f} ({test_metrics['recall']*100:.2f}%)")
    print(f"F1-Score:     {test_metrics['f1']:.4f}")
    print(f"ROC-AUC:      {test_metrics['roc_auc']:.4f}")
    print(f"FAR:          {test_metrics['far']:.4f} ({test_metrics['far']*100:.2f}%)")
    print(f"CSI/Threat:   {test_metrics['csi']:.4f}")
    print(f"TSS:          {test_metrics['tss']:.4f}")
    print(f"HSS:          {test_metrics['hss']:.4f}")
    
    print(f"\nConfusion Matrix:")
    print(f"  TP (true lightning):        {test_metrics['tp']:>8,}")
    print(f"  FP (false alarms):          {test_metrics['fp']:>8,}")
    print(f"  TN (true non-lightning):    {test_metrics['tn']:>8,}")
    print(f"  FN (missed lightning):      {test_metrics['fn']:>8,}")
    
    # Save results
    output_data = {
        'best_threshold': float(best_threshold),
        'validation_results': {str(k): v for k, v in results.items()},
        'test_metrics': test_metrics,
        'test_metrics_summary': {
            'accuracy': float(test_metrics['accuracy']),
            'precision': float(test_metrics['precision']),
            'recall': float(test_metrics['recall']),
            'f1': float(test_metrics['f1']),
            'roc_auc': float(test_metrics['roc_auc']),
            'far': float(test_metrics['far']),
            'csi': float(test_metrics['csi']),
            'tss': float(test_metrics['tss']),
            'hss': float(test_metrics['hss']),
            'tp': test_metrics['tp'],
            'fp': test_metrics['fp'],
            'tn': test_metrics['tn'],
            'fn': test_metrics['fn'],
        }
    }
    
    with open('models/threshold_tuning_results.json', 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n✅ Results saved to models/threshold_tuning_results.json")
    print("=" * 100)
    
    return best_threshold, test_metrics

if __name__ == '__main__':
    best_threshold, test_metrics = main()
