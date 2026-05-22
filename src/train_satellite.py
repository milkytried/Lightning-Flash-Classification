"""
Train ResNet-50 CNN on Himawari-8 satellite image patches.

Trains on 64×64 patches to predict lightning occurrence with ≥85% recall target.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
import logging
import json

from model_arch import LightningResNet50, FocalLoss
from himawari_data_loader import create_himawari_loaders

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SatelliteTrainer:
    """Trainer for ResNet-50 on satellite patches."""
    
    def __init__(self, model_path: str = 'models/satellite_resnet50.pth',
                 device: str = 'cpu', use_focal_loss: bool = True):
        """
        Initialize trainer.
        
        Args:
            model_path: Path to save/load model
            device: 'cpu' or 'cuda'
            use_focal_loss: Use Focal Loss for class imbalance
        """
        self.model_path = Path(model_path)
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.device = device
        self.use_focal_loss = use_focal_loss
        
        # Initialize model
        self.model = LightningResNet50(num_input_channels=3, num_classes=1, dropout_rate=0.5)
        self.model = self.model.to(device)
        
        # Loss function
        if use_focal_loss:
            self.criterion = FocalLoss(alpha=0.25, gamma=2.0, reduction='mean')
        else:
            self.criterion = nn.BCELoss()
        
        logger.info(f"SatelliteTrainer initialized")
        logger.info(f"  Device: {device}")
        logger.info(f"  Model: LightningResNet50")
        logger.info(f"  Loss: {'Focal Loss' if use_focal_loss else 'BCELoss'}")
    
    def train_epoch(self, train_loader, optimizer):
        """
        Train for one epoch.
        
        Args:
            train_loader: Training DataLoader
            optimizer: Optimizer
        
        Returns:
            Average loss for epoch
        """
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc='Training')
        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device).float()
            
            # Forward pass
            optimizer.zero_grad()
            outputs = self.model(images).squeeze()
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / max(1, num_batches)
        return avg_loss
    
    def val_epoch(self, val_loader):
        """
        Validate for one epoch.
        
        Args:
            val_loader: Validation DataLoader
        
        Returns:
            (avg_loss, metrics_dict) tuple
        """
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc='Validating')
            for images, labels in pbar:
                images = images.to(self.device)
                labels = labels.to(self.device).float()
                
                # Forward pass
                outputs = self.model(images).squeeze()
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                num_batches += 1
                
                # Collect predictions
                probs = torch.sigmoid(outputs).cpu().numpy()
                preds = (probs > 0.5).astype(int)
                
                all_probs.append(probs)
                all_preds.append(preds)
                all_labels.append(labels.cpu().numpy())
                
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # Compute metrics
        all_probs = np.concatenate(all_probs)
        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
        
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
        
        metrics = {
            'accuracy': accuracy_score(all_labels, all_preds),
            'precision': precision_score(all_labels, all_preds, zero_division=0),
            'recall': recall_score(all_labels, all_preds, zero_division=0),
            'f1': f1_score(all_labels, all_preds, zero_division=0),
            'roc_auc': roc_auc_score(all_labels, all_probs) if len(np.unique(all_labels)) > 1 else 0.0
        }
        
        avg_loss = total_loss / max(1, num_batches)
        
        return avg_loss, metrics
    
    def train(self, dataset_csv: str, num_epochs: int = 50, 
              batch_size: int = 32, lr: float = 0.001,
              early_stopping_patience: int = 10):
        """
        Full training loop.
        
        Args:
            dataset_csv: Path to satellite_dataset.csv
            num_epochs: Max epochs
            batch_size: Batch size
            lr: Learning rate
            early_stopping_patience: Patience for early stopping
        
        Returns:
            Training history dictionary
        """
        logger.info(f"Starting training")
        logger.info(f"  Dataset: {dataset_csv}")
        logger.info(f"  Epochs: {num_epochs}")
        logger.info(f"  Batch size: {batch_size}")
        logger.info(f"  Learning rate: {lr}")
        
        # Create data loaders
        loaders = create_himawari_loaders(dataset_csv, batch_size=batch_size)
        train_loader = loaders['train']
        val_loader = loaders['val']
        
        # Optimizer and scheduler
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=5, verbose=True)
        
        # Training history
        history = {
            'epoch': [],
            'train_loss': [],
            'val_loss': [],
            'val_accuracy': [],
            'val_recall': [],
            'val_precision': [],
            'val_f1': [],
            'val_roc_auc': []
        }
        
        best_val_loss = float('inf')
        patience_count = 0
        
        for epoch in range(num_epochs):
            logger.info(f"\nEpoch {epoch + 1}/{num_epochs}")
            
            # Train
            train_loss = self.train_epoch(train_loader, optimizer)
            
            # Validate
            val_loss, metrics = self.val_epoch(val_loader)
            
            # Log metrics
            logger.info(f"Train Loss: {train_loss:.4f}")
            logger.info(f"Val Loss:   {val_loss:.4f}")
            logger.info(f"Accuracy:   {metrics['accuracy']:.4f}")
            logger.info(f"Precision:  {metrics['precision']:.4f}")
            logger.info(f"Recall:     {metrics['recall']:.4f} {'✅ PASS' if metrics['recall'] >= 0.85 else ''}")
            logger.info(f"F1-Score:   {metrics['f1']:.4f}")
            logger.info(f"ROC-AUC:    {metrics['roc_auc']:.4f}")
            
            # Record history
            history['epoch'].append(epoch + 1)
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['val_accuracy'].append(metrics['accuracy'])
            history['val_recall'].append(metrics['recall'])
            history['val_precision'].append(metrics['precision'])
            history['val_f1'].append(metrics['f1'])
            history['val_roc_auc'].append(metrics['roc_auc'])
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_count = 0
                
                # Save best model
                self.save_model()
                logger.info(f"✓ Model saved (loss improved to {val_loss:.4f})")
            else:
                patience_count += 1
                logger.info(f"No improvement ({patience_count}/{early_stopping_patience})")
                
                if patience_count >= early_stopping_patience:
                    logger.info(f"Early stopping triggered")
                    break
            
            # Learning rate scheduling
            scheduler.step(val_loss)
        
        # Save history
        self.save_history(history)
        
        logger.info(f"\nTraining complete!")
        
        return history
    
    def save_model(self):
        """Save model weights."""
        torch.save(self.model.state_dict(), self.model_path)
        logger.info(f"Model saved to {self.model_path}")
    
    def load_model(self):
        """Load model weights."""
        self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        logger.info(f"Model loaded from {self.model_path}")
    
    def save_history(self, history):
        """Save training history to JSON."""
        history_path = self.model_path.parent / 'satellite_training_history.json'
        
        # Convert numpy types to Python types for JSON serialization
        history_serializable = {
            k: [float(v) for v in vals] if isinstance(vals[0], (np.floating, float)) else vals
            for k, vals in history.items()
        }
        
        with open(history_path, 'w') as f:
            json.dump(history_serializable, f, indent=2)
        
        logger.info(f"History saved to {history_path}")


def train_satellite_model(dataset_csv: str = 'data/processed/satellite_dataset.csv',
                         num_epochs: int = 50, batch_size: int = 32):
    """
    Main training function.
    
    Args:
        dataset_csv: Path to satellite dataset index
        num_epochs: Maximum epochs
        batch_size: Batch size
    """
    # Select device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")
    
    # Initialize trainer
    trainer = SatelliteTrainer(
        model_path='models/satellite_resnet50.pth',
        device=device,
        use_focal_loss=True
    )
    
    # Train
    history = trainer.train(
        dataset_csv=dataset_csv,
        num_epochs=num_epochs,
        batch_size=batch_size,
        lr=0.001,
        early_stopping_patience=10
    )
    
    logger.info("\n" + "=" * 60)
    logger.info("TRAINING SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total epochs trained: {len(history['epoch'])}")
    logger.info(f"Best val recall: {max(history['val_recall']):.4f}")
    logger.info(f"Best val accuracy: {max(history['val_accuracy']):.4f}")
    logger.info(f"Best val F1: {max(history['val_f1']):.4f}")
    logger.info(f"Model saved to: models/satellite_resnet50.pth")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Train ResNet-50 on satellite patches')
    parser.add_argument('--dataset', type=str, default='data/processed/satellite_dataset.csv',
                       help='Path to dataset CSV')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    
    args = parser.parse_args()
    
    train_satellite_model(
        dataset_csv=args.dataset,
        num_epochs=args.epochs,
        batch_size=args.batch_size
    )
