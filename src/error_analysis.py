"""
Error analysis: Generate visualizations of TP, FP, TN, FN samples with metadata.

Shows 20 samples from each category to understand model behavior and errors.
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from PIL import Image
import logging
import sys

sys.path.insert(0, 'src')

from himawari_data_loader import HimawariPatchDataset
from model_arch import LightningResNet50

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_error_analysis(best_threshold=0.5):
    """Generate error analysis with sample visualizations."""
    
    logger.info("="*70)
    logger.info("ERROR ANALYSIS: TP/FP/TN/FN SAMPLE VISUALIZATION")
    logger.info("="*70)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load model and data
    logger.info("\n[1] Loading model and test data...")
    model_path = 'models/satellite_resnet50.pth'
    model = LightningResNet50(num_input_channels=3, num_classes=1, dropout_rate=0.5)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    dataset_csv = 'data/processed/satellite_dataset.csv'
    df = pd.read_csv(dataset_csv)
    test_df = df[df['split'] == 'test'].reset_index(drop=True)
    
    # Generate predictions
    logger.info("Generating predictions on test set...")
    test_dataset = HimawariPatchDataset(dataset_csv=dataset_csv, split='test', augment=False)
    
    all_probs = []
    all_labels = []
    all_indices = []
    
    with torch.no_grad():
        for idx in range(len(test_dataset)):
            image, label = test_dataset[idx]
            image = image.unsqueeze(0).to(device)
            prob = model(image).cpu().item()
            all_probs.append(prob)
            all_labels.append(label.item())
            all_indices.append(idx)
    
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    all_indices = np.array(all_indices)
    
    # Categorize
    preds = (all_probs >= best_threshold).astype(int)
    
    tp_idx = np.where((preds == 1) & (all_labels == 1))[0]
    fp_idx = np.where((preds == 1) & (all_labels == 0))[0]
    tn_idx = np.where((preds == 0) & (all_labels == 0))[0]
    fn_idx = np.where((preds == 0) & (all_labels == 1))[0]
    
    logger.info(f"Sample counts:")
    logger.info(f"  TP: {len(tp_idx)}")
    logger.info(f"  FP: {len(fp_idx)}")
    logger.info(f"  TN: {len(tn_idx)}")
    logger.info(f"  FN: {len(fn_idx)}")
    
    # Function to create visualization for a category
    def visualize_category(category_idx, category_name, category_label, n_samples=20):
        """Create visualization for a prediction category."""
        
        if len(category_idx) == 0:
            logger.warning(f"No {category_name} samples found")
            return
        
        # Select random samples
        selected_idx = np.random.choice(category_idx, size=min(n_samples, len(category_idx)), replace=False)
        
        n_cols = 5
        n_rows = (len(selected_idx) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 3*n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        
        axes = axes.flatten()
        
        for plot_idx, sample_idx in enumerate(selected_idx):
            ax = axes[plot_idx]
            
            # Load image
            patch_path = test_df.iloc[sample_idx]['path']
            try:
                img = Image.open(patch_path).convert('RGB')
                img_array = np.array(img) / 255.0
            except Exception as e:
                logger.warning(f"Could not load {patch_path}: {e}")
                continue
            
            # Display image
            ax.imshow(img_array)
            
            # Get metadata
            true_label = int(all_labels[sample_idx])
            pred_prob = all_probs[sample_idx]
            
            # Color code
            if category_name == 'TP':
                color = 'green'
            elif category_name == 'FP':
                color = 'red'
            elif category_name == 'TN':
                color = 'blue'
            else:  # FN
                color = 'orange'
            
            # Add border
            for spine in ax.spines.values():
                spine.set_edgecolor(color)
                spine.set_linewidth(3)
            
            # Title with metadata
            label_str = "Lightning" if true_label == 1 else "No-Lightning"
            ax.set_title(f"{category_name}\nTrue: {label_str}\nProb: {pred_prob:.3f}", 
                        fontsize=9, color=color, fontweight='bold')
            ax.axis('off')
        
        # Hide unused subplots
        for idx in range(len(selected_idx), len(axes)):
            axes[idx].axis('off')
        
        plt.suptitle(f'{category_name} Samples (Threshold={best_threshold})', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        output_path = f'results/error_analysis_{category_name.lower()}_samples.png'
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        logger.info(f"  Saved: {output_path}")
        plt.close()
    
    # Generate visualizations
    logger.info("\n[2] Generating sample visualizations...")
    
    np.random.seed(42)
    visualize_category(tp_idx, 'TP', 1, n_samples=20)
    visualize_category(fp_idx, 'FP', 1, n_samples=20)
    visualize_category(tn_idx, 'TN', 0, n_samples=20)
    visualize_category(fn_idx, 'FN', 0, n_samples=20)
    
    # Save detailed statistics
    logger.info("\n[3] Saving detailed error statistics...")
    
    error_stats = {
        'threshold': best_threshold,
        'total_test_samples': len(all_labels),
        'categories': {
            'TP': {
                'count': len(tp_idx),
                'mean_prob': float(all_probs[tp_idx].mean()) if len(tp_idx) > 0 else 0,
                'std_prob': float(all_probs[tp_idx].std()) if len(tp_idx) > 0 else 0
            },
            'FP': {
                'count': len(fp_idx),
                'mean_prob': float(all_probs[fp_idx].mean()) if len(fp_idx) > 0 else 0,
                'std_prob': float(all_probs[fp_idx].std()) if len(fp_idx) > 0 else 0
            },
            'TN': {
                'count': len(tn_idx),
                'mean_prob': float(all_probs[tn_idx].mean()) if len(tn_idx) > 0 else 0,
                'std_prob': float(all_probs[tn_idx].std()) if len(tn_idx) > 0 else 0
            },
            'FN': {
                'count': len(fn_idx),
                'mean_prob': float(all_probs[fn_idx].mean()) if len(fn_idx) > 0 else 0,
                'std_prob': float(all_probs[fn_idx].std()) if len(fn_idx) > 0 else 0
            }
        }
    }
    
    import json
    with open('results/error_analysis_stats.json', 'w') as f:
        json.dump(error_stats, f, indent=2)
    
    logger.info("  Saved: results/error_analysis_stats.json")
    
    logger.info("\n" + "="*70)
    logger.info("ERROR ANALYSIS COMPLETE")
    logger.info("="*70)
    logger.info(f"Generated sample visualizations:")
    logger.info(f"  - 20 TP (True Positives) - green borders")
    logger.info(f"  - 20 FP (False Positives) - red borders")
    logger.info(f"  - 20 TN (True Negatives) - blue borders")
    logger.info(f"  - 20 FN (False Negatives) - orange borders")


if __name__ == '__main__':
    # Run with best threshold from comprehensive evaluation
    best_threshold = 0.5  # Default, will be updated after threshold tuning
    get_error_analysis(best_threshold=best_threshold)
