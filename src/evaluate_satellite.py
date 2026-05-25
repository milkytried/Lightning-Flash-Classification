"""
Evaluate ResNet-50 satellite model and generate visualizations.

Computes metrics on test set and creates plots.
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
import logging
import json

from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, roc_curve, auc
)
from PIL import Image

from model_arch import LightningResNet50, FocalLoss
from himawari_data_loader import create_himawari_loaders

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SatelliteEvaluator:
    """Evaluator for ResNet-50 satellite model."""
    
    def __init__(self, model_path: str = 'models/satellite_resnet50.pth',
                 device: str = 'cpu'):
        """
        Initialize evaluator.
        
        Args:
            model_path: Path to trained model
            device: 'cpu' or 'cuda'
        """
        self.model_path = Path(model_path)
        self.device = device
        
        # Load model
        self.model = LightningResNet50(num_input_channels=3, num_classes=1, dropout_rate=0.5)
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model = self.model.to(device)
        self.model.eval()
        
        logger.info(f"SatelliteEvaluator initialized")
        logger.info(f"  Device: {device}")
        logger.info(f"  Model: {model_path}")
    
    def evaluate_split(self, dataloader, split_name: str = 'test'):
        """
        Evaluate on a split.
        
        Args:
            dataloader: DataLoader for split
            split_name: Name of split (for logging)
        
        Returns:
            (predictions, probabilities, labels) arrays
        """
        logger.info(f"Evaluating on {split_name} split...")
        
        all_preds = []
        all_probs = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in tqdm(dataloader, desc=split_name):
                images = images.to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                probs = torch.sigmoid(outputs).squeeze().cpu().numpy()
                
                # Handle single image (squeeze might remove batch dimension)
                if np.isscalar(probs):
                    probs = np.array([probs])
                
                preds = (probs > 0.5).astype(int)
                
                all_probs.append(probs)
                all_preds.append(preds)
                all_labels.append(labels.numpy())
        
        if not all_probs:
            raise ValueError(f"No predictions made for split '{split_name}' - dataloader was empty")
        
        all_probs = np.concatenate(all_probs)
        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
        
        return all_preds, all_probs, all_labels
    
    def compute_metrics(self, preds, probs, labels):
        """
        Compute evaluation metrics.
        
        Args:
            preds: Binary predictions (0/1)
            probs: Predicted probabilities
            labels: Ground truth labels
        
        Returns:
            Metrics dictionary
        """
        metrics = {
            'accuracy': accuracy_score(labels, preds),
            'precision': precision_score(labels, preds, zero_division=0),
            'recall': recall_score(labels, preds, zero_division=0),
            'f1': f1_score(labels, preds, zero_division=0),
            'roc_auc': roc_auc_score(labels, probs) if len(np.unique(labels)) > 1 else 0.0,
            'n_samples': len(labels),
            'n_positive': np.sum(labels),
            'n_negative': len(labels) - np.sum(labels),
            'n_tp': np.sum((preds == 1) & (labels == 1)),
            'n_fp': np.sum((preds == 1) & (labels == 0)),
            'n_tn': np.sum((preds == 0) & (labels == 0)),
            'n_fn': np.sum((preds == 0) & (labels == 1)),
        }
        
        # POD and FAR (from meteorology)
        metrics['pod'] = metrics['recall']  # Probability of Detection
        metrics['far'] = metrics['n_fp'] / max(1, (metrics['n_fp'] + metrics['n_tp']))  # False Alarm Rate
        
        return metrics
    
    def plot_confusion_matrix(self, preds, labels, split_name: str = 'test'):
        """Plot confusion matrix."""
        cm = confusion_matrix(labels, preds)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['No Lightning', 'Lightning'],
                   yticklabels=['No Lightning', 'Lightning'])
        plt.title(f'Confusion Matrix ({split_name})')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        output_path = Path('results/confusion_matrix.png')
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        logger.info(f"Confusion matrix saved: {output_path}")
        plt.close()
    
    def plot_roc_curve(self, probs, labels):
        """Plot ROC curve."""
        if len(np.unique(labels)) < 2:
            logger.warning("Cannot plot ROC: only one class in labels")
            return
        
        fpr, tpr, _ = roc_curve(labels, probs)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve (Test Set)')
        plt.legend(loc="lower right")
        
        output_path = Path('results/roc_curve.png')
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        logger.info(f"ROC curve saved: {output_path}")
        plt.close()
    
    def plot_metrics_summary(self, metrics_dict):
        """Plot metrics summary bar chart."""
        metrics_to_plot = {
            'Accuracy': metrics_dict['accuracy'],
            'Precision': metrics_dict['precision'],
            'Recall': metrics_dict['recall'],
            'F1-Score': metrics_dict['f1'],
            'ROC-AUC': metrics_dict['roc_auc'],
            'POD': metrics_dict['pod'],
            'FAR': metrics_dict['far']
        }
        
        # Plot showing if recall ≥ 0.85
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = ['green' if metrics_dict['recall'] >= 0.85 else 'red' if k == 'Recall' else 'steelblue' 
                 for k in metrics_to_plot.keys()]
        
        bars = ax.bar(metrics_to_plot.keys(), metrics_to_plot.values(), color=colors, alpha=0.7)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        ax.axhline(y=0.85, color='r', linestyle='--', linewidth=2, label='Target Recall (0.85)')
        ax.set_ylim([0, 1.05])
        ax.set_ylabel('Score')
        ax.set_title('Satellite Model Performance Metrics')
        ax.legend()
        plt.xticks(rotation=45, ha='right')
        
        output_path = Path('results/metrics_summary.png')
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        logger.info(f"Metrics summary saved: {output_path}")
        plt.close()
    
    def print_metrics_report(self, metrics_dict, split_name: str = 'Test'):
        """Print detailed metrics report."""
        logger.info("\n" + "=" * 70)
        logger.info(f"{split_name.upper()} SET EVALUATION REPORT")
        logger.info("=" * 70)
        
        logger.info(f"\nDataset Statistics:")
        logger.info(f"  Total samples:     {metrics_dict['n_samples']}")
        logger.info(f"  Positive (Lightning): {metrics_dict['n_positive']} ({100*metrics_dict['n_positive']/metrics_dict['n_samples']:.1f}%)")
        logger.info(f"  Negative (No Lightning): {metrics_dict['n_negative']} ({100*metrics_dict['n_negative']/metrics_dict['n_samples']:.1f}%)")
        
        logger.info(f"\nClassification Results:")
        logger.info(f"  True Positives:    {metrics_dict['n_tp']}")
        logger.info(f"  False Positives:   {metrics_dict['n_fp']}")
        logger.info(f"  True Negatives:    {metrics_dict['n_tn']}")
        logger.info(f"  False Negatives:   {metrics_dict['n_fn']}")
        
        logger.info(f"\nMetrics:")
        logger.info(f"  Accuracy:          {metrics_dict['accuracy']:.4f}")
        logger.info(f"  Precision:         {metrics_dict['precision']:.4f}")
        
        # Highlight recall (key metric for lightning detection)
        recall_status = "✅ PASS" if metrics_dict['recall'] >= 0.85 else "⚠️  BELOW TARGET"
        logger.info(f"  Recall (POD):      {metrics_dict['recall']:.4f} {recall_status}")
        
        logger.info(f"  F1-Score:          {metrics_dict['f1']:.4f}")
        logger.info(f"  ROC-AUC:           {metrics_dict['roc_auc']:.4f}")
        logger.info(f"  False Alarm Rate:  {metrics_dict['far']:.4f}")
        
        logger.info("=" * 70)
    
    def save_metrics_json(self, metrics_dict):
        """Save metrics to JSON."""
        output_path = Path('results/satellite_metrics.json')
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert numpy types for JSON serialization
        metrics_serializable = {
            k: float(v) if isinstance(v, (np.integer, np.floating)) else v
            for k, v in metrics_dict.items()
        }
        
        with open(output_path, 'w') as f:
            json.dump(metrics_serializable, f, indent=2)
        
        logger.info(f"Metrics saved: {output_path}")


def evaluate_satellite_model(dataset_csv: str = 'data/processed/satellite_dataset.csv',
                            model_path: str = 'models/satellite_resnet50.pth',
                            batch_size: int = 32):
    """
    Main evaluation function.
    
    Args:
        dataset_csv: Path to dataset CSV
        model_path: Path to trained model
        batch_size: Batch size
    """
    # Select device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")
    
    # Initialize evaluator
    evaluator = SatelliteEvaluator(model_path=model_path, device=device)
    
    # Load test data
    logger.info(f"Loading test data from {dataset_csv}...")
    loaders = create_himawari_loaders(dataset_csv, batch_size=batch_size)
    test_loader = loaders.get('test', None)
    
    # If test set is empty, use train set instead
    if test_loader is None or len(test_loader) == 0:
        logger.warning("Test set empty - using training set for evaluation instead")
        test_loader = loaders.get('train', None)
        eval_split = 'train'
    else:
        eval_split = 'test'
    
    if test_loader is None or len(test_loader) == 0:
        logger.error("No data available for evaluation")
        return
    
    # Evaluate
    logger.info(f"Evaluating on {eval_split} set...")
    preds, probs, labels = evaluator.evaluate_split(test_loader, split_name=eval_split)
    
    # Compute metrics
    metrics = evaluator.compute_metrics(preds, probs, labels)
    
    # Print report
    evaluator.print_metrics_report(metrics, split_name=eval_split.capitalize())
    
    # Generate visualizations
    logger.info("\nGenerating visualizations...")
    evaluator.plot_confusion_matrix(preds, labels, split_name='test')
    evaluator.plot_roc_curve(probs, labels)
    evaluator.plot_metrics_summary(metrics)
    
    # Save metrics
    evaluator.save_metrics_json(metrics)
    
    logger.info("\n" + "=" * 70)
    logger.info("EVALUATION COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Recall achieved: {metrics['recall']:.4f}")
    logger.info(f"Target recall:   0.8500")
    logger.info(f"Status:          {'✅ PASS' if metrics['recall'] >= 0.85 else '⚠️  BELOW TARGET'}")
    logger.info("\nResults saved to:")
    logger.info("  - results/confusion_matrix.png")
    logger.info("  - results/roc_curve.png")
    logger.info("  - results/metrics_summary.png")
    logger.info("  - results/satellite_metrics.json")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate satellite model on test set')
    parser.add_argument('--dataset', type=str, default='data/processed/satellite_dataset.csv',
                       help='Path to dataset CSV')
    parser.add_argument('--model', type=str, default='models/best_resnet50.pth',
                       help='Path to trained model')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    
    args = parser.parse_args()
    
    evaluate_satellite_model(
        dataset_csv=args.dataset,
        model_path=args.model,
        batch_size=args.batch_size
    )
