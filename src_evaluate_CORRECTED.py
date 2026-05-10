"""
Evaluation module for computing metrics and error analysis.
"""

from typing import Dict, Tuple, Optional
import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, roc_curve, auc, classification_report
)
import matplotlib.pyplot as plt
import json
from pathlib import Path
import yaml


def compute_metrics(
    predictions: np.ndarray, 
    labels: np.ndarray, 
    threshold: float = 0.5
) -> Dict:
    """
    Compute classification metrics including standard and meteorological measures.
    
    Args:
        predictions (np.ndarray): Predicted probabilities [0, 1], shape (N,)
        labels (np.ndarray): Ground truth labels {0, 1}, shape (N,)
        threshold (float): Classification threshold for binary prediction
    
    Returns:
        Dict: Metrics dictionary with keys:
            - accuracy, precision, recall, f1_score, roc_auc
            - pod (Probability of Detection), far (False Alarm Ratio)
            - hss (Heidke Skill Score), tss (True Skill Statistic)
            - confusion_matrix: {tn, fp, fn, tp}
    
    Raises:
        ValueError: If shapes don't match or threshold invalid
    """
    if predictions.shape[0] != labels.shape[0]:
        raise ValueError(f"predictions {predictions.shape} and labels {labels.shape} shape mismatch")
    if not (0.0 <= threshold <= 1.0):
        raise ValueError(f"threshold must be in [0, 1], got {threshold}")
    
    preds_binary = (predictions > threshold).astype(int)
    
    # Compute confusion matrix safely
    try:
        cm = confusion_matrix(labels, preds_binary)
        # Handle cases where not all classes are present
        if cm.size == 1:
            # Only one class present
            if labels[0] == 0:
                tn, fp, fn, tp = cm[0, 0], 0, 0, 0
            else:
                tn, fp, fn, tp = 0, 0, 0, cm[0, 0]
        elif cm.shape == (1, 2):
            tn, fp = cm[0]
            fn, tp = 0, 0
        elif cm.shape == (2, 1):
            tn, fn = cm[0]
            fp, tp = 0, 0
        else:
            # Normal 2x2 case
            tn, fp, fn, tp = cm.ravel()
    except Exception as e:
        raise RuntimeError(f"Failed to compute confusion matrix: {str(e)}")
    
    # Compute metrics with safe division
    accuracy = accuracy_score(labels, preds_binary)
    precision = precision_score(labels, preds_binary, zero_division=0)
    recall = recall_score(labels, preds_binary, zero_division=0)
    f1 = f1_score(labels, preds_binary, zero_division=0)
    
    # ROC-AUC with single-class handling
    try:
        if len(np.unique(labels)) < 2:
            roc_auc = float('nan')
        else:
            roc_auc = roc_auc_score(labels, predictions)
    except:
        roc_auc = float('nan')
    
    # Meteorological metrics
    pod = recall  # Probability of Detection
    
    # False Alarm Ratio (FAR)
    if (tp + fp) > 0:
        far = fp / (tp + fp)
    else:
        far = 0.0
    
    # Heidke Skill Score (HSS)
    po = accuracy
    if len(labels) > 0:
        expected = (
            ((tp + fn) / len(labels)) * ((tp + fp) / len(labels)) +
            ((tn + fp) / len(labels)) * ((tn + fn) / len(labels))
        )
    else:
        expected = 0.0
    
    if (1 - expected) != 0:
        hss = (po - expected) / (1 - expected)
    else:
        hss = 0.0
    
    # True Skill Statistic (TSS) = POD - FAR
    tss = pod - far
    
    metrics = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),  # POD
        'f1_score': float(f1),
        'roc_auc': float(roc_auc),
        'pod': float(pod),
        'far': float(far),
        'hss': float(hss),
        'tss': float(tss),
        'confusion_matrix': {
            'tn': int(tn),
            'fp': int(fp),
            'fn': int(fn),
            'tp': int(tp),
        }
    }
    
    return metrics


def evaluate_model(
    model: torch.nn.Module, 
    test_loader, 
    device: torch.device,
    results_dir: str = 'results/'
) -> Tuple[Dict, np.ndarray, np.ndarray]:
    """
    Comprehensive evaluation on test set.
    
    Args:
        model (torch.nn.Module): Trained PyTorch model
        test_loader: Test DataLoader
        device (torch.device): Device to evaluate on (cuda or cpu)
        results_dir (str): Directory to save results
    
    Returns:
        Tuple[Dict, np.ndarray, np.ndarray]: (metrics dict, predictions, labels)
    
    Raises:
        RuntimeError: If evaluation fails
    """
    model.eval()
    all_preds = []
    all_labels = []
    
    try:
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                outputs = model(images).cpu().numpy()
                all_preds.append(outputs)
                all_labels.append(labels.numpy())
    except Exception as e:
        raise RuntimeError(f"Error during model evaluation: {str(e)}")
    
    preds = np.concatenate(all_preds, axis=0).flatten()
    labels = np.concatenate(all_labels, axis=0).flatten()
    
    # Compute metrics
    try:
        metrics = compute_metrics(preds, labels, threshold=0.5)
    except Exception as e:
        raise RuntimeError(f"Error computing metrics: {str(e)}")
    
    # Save metrics
    try:
        Path(results_dir).mkdir(parents=True, exist_ok=True)
        with open(f"{results_dir}/metrics.json", 'w') as f:
            json.dump(metrics, f, indent=2)
    except Exception as e:
        print(f"Warning: Failed to save metrics: {str(e)}")
    
    # Print results
    print("\n=== Evaluation Metrics ===")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1-Score:  {metrics['f1_score']:.4f}")
    print(f"ROC-AUC:   {metrics['roc_auc']:.4f}")
    print(f"\nMeteorological Metrics:")
    print(f"POD:       {metrics['pod']:.4f}")
    print(f"FAR:       {metrics['far']:.4f}")
    print(f"HSS:       {metrics['hss']:.4f}")
    print(f"TSS:       {metrics['tss']:.4f}")
    
    return metrics, preds, labels


def plot_roc_curve(
    predictions: np.ndarray, 
    labels: np.ndarray, 
    results_dir: str = 'results/'
) -> None:
    """
    Plot ROC curve and save to file.
    
    Args:
        predictions (np.ndarray): Predicted probabilities
        labels (np.ndarray): Ground truth labels
        results_dir (str): Directory to save plot
    
    Raises:
        ValueError: If shapes don't match or not enough data
    """
    if len(predictions) != len(labels):
        raise ValueError("predictions and labels must have same length")
    if len(np.unique(labels)) < 2:
        print("Warning: ROC curve requires both classes present")
        return
    
    try:
        fpr, tpr, thresholds = roc_curve(labels, predictions)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        
        Path(results_dir).mkdir(parents=True, exist_ok=True)
        plt.savefig(f"{results_dir}/roc_curve.png", dpi=150, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"Error plotting ROC curve: {str(e)}")


def plot_confusion_matrix(
    predictions: np.ndarray, 
    labels: np.ndarray, 
    results_dir: str = 'results/',
    threshold: float = 0.5
) -> None:
    """
    Plot confusion matrix and save to file.
    
    Args:
        predictions (np.ndarray): Predicted probabilities
        labels (np.ndarray): Ground truth labels
        results_dir (str): Directory to save plot
        threshold (float): Classification threshold
    
    Raises:
        ValueError: If inputs invalid
    """
    if len(predictions) != len(labels):
        raise ValueError("predictions and labels must have same length")
    
    try:
        preds_binary = (predictions > threshold).astype(int)
        cm = confusion_matrix(labels, preds_binary)
        
        plt.figure(figsize=(8, 6))
        plt.imshow(cm, interpolation='nearest', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.colorbar()
        tick_marks = np.arange(2)
        plt.xticks(tick_marks, ['No Lightning', 'Lightning'])
        plt.yticks(tick_marks, ['No Lightning', 'Lightning'])
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        
        # Add counts
        thresh = cm.max() / 2.0
        for i, j in np.ndindex(cm.shape):
            plt.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
        
        Path(results_dir).mkdir(parents=True, exist_ok=True)
        plt.savefig(f"{results_dir}/confusion_matrix.png", dpi=150, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"Error plotting confusion matrix: {str(e)}")


def error_analysis(
    predictions: np.ndarray, 
    labels: np.ndarray, 
    images: Optional[np.ndarray] = None,
    results_dir: str = 'results/',
    num_examples: int = 10
) -> None:
    """
    Identify and visualize false positives and false negatives.
    
    Args:
        predictions (np.ndarray): Predicted probabilities
        labels (np.ndarray): Ground truth labels
        images (Optional[np.ndarray]): Image tensors (optional, for visualization)
        results_dir (str): Directory to save results
        num_examples (int): Number of examples to visualize
    
    Raises:
        ValueError: If inputs invalid
    """
    if len(predictions) != len(labels):
        raise ValueError("predictions and labels must have same length")
    
    try:
        preds_binary = (predictions > 0.5).astype(int)
        
        # Find FP/FN
        tp = (preds_binary == 1) & (labels == 1)
        fp = (preds_binary == 1) & (labels == 0)
        tn = (preds_binary == 0) & (labels == 0)
        fn = (preds_binary == 0) & (labels == 1)
        
        print(f"\n=== Error Analysis ===")
        print(f"True Positives:  {tp.sum()}")
        print(f"True Negatives:  {tn.sum()}")
        print(f"False Positives: {fp.sum()}")
        print(f"False Negatives: {fn.sum()}")
        
        # Save to JSON
        Path(results_dir).mkdir(parents=True, exist_ok=True)
        error_summary = {
            'tp': int(tp.sum()),
            'tn': int(tn.sum()),
            'fp': int(fp.sum()),
            'fn': int(fn.sum()),
        }
        with open(f"{results_dir}/error_analysis.json", 'w') as f:
            json.dump(error_summary, f, indent=2)
    except Exception as e:
        print(f"Error in error analysis: {str(e)}")


if __name__ == '__main__':
    print("Evaluation module loaded. Use evaluate_model() to evaluate model.")
