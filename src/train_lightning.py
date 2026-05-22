"""
Train lightning detection model on Met Department Malaysia data (2023-2026).
"""

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from pathlib import Path
import json
import logging
from datetime import datetime

import sys
sys.path.insert(0, '.')

from lightning_data_loader import create_lightning_loaders
from lightning_model import LightningMetadataClassifier, FocalLoss

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def train_epoch(model, loader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    
    for batch_idx, (features, labels) in enumerate(loader):
        features = features.to(device)
        labels = labels.to(device).unsqueeze(1)
        
        optimizer.zero_grad()
        predictions = model(features)
        loss = criterion(predictions, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        
        if batch_idx % 100 == 0:
            logger.info(f"  Batch {batch_idx}/{len(loader)}: loss={loss.item():.4f}")
    
    return total_loss / len(loader)


def val_epoch(model, loader, criterion, device):
    """Validation epoch."""
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for features, labels in loader:
            features = features.to(device)
            labels = labels.to(device).unsqueeze(1)
            
            predictions = model(features)
            loss = criterion(predictions, labels)
            total_loss += loss.item()
    
    return total_loss / len(loader)


def train_lightning_model(
    hdf5_path: str = "data/processed/lightning_dataset.h5",
    output_path: str = "models/lightning_classifier.pth",
    max_epochs: int = 50,
    batch_size: int = 512,
    learning_rate: float = 0.001,
):
    """Train lightning detection model on real data."""
    
    logger.info("=" * 70)
    logger.info("TRAINING LIGHTNING DETECTION MODEL")
    logger.info(f"Dataset: {hdf5_path}")
    logger.info(f"Batch size: {batch_size}, Learning rate: {learning_rate}")
    logger.info("=" * 70)
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")
    
    # Data
    logger.info("\nLoading data...")
    loaders = create_lightning_loaders(hdf5_path, batch_size=batch_size)
    train_loader = loaders['train']
    val_loader = loaders['val']
    test_loader = loaders['test']
    
    logger.info(f"  Train batches: {len(train_loader)}")
    logger.info(f"  Val batches: {len(val_loader)}")
    logger.info(f"  Test batches: {len(test_loader)}")
    
    # Model
    logger.info("\nInitializing model...")
    model = LightningMetadataClassifier(input_size=4, hidden_size=256, dropout=0.3)
    model.to(device)
    logger.info(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss & Optimizer
    criterion = FocalLoss(alpha=0.25, gamma=2.0)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    # Training loop
    logger.info("\nStarting training...")
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': []}
    
    for epoch in range(max_epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = val_epoch(model, val_loader, criterion, device)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
        logger.info(f"Epoch {epoch:3d}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            
            # Save best model
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), output_path)
            logger.info(f"  → Best model saved (val_loss={val_loss:.6f})")
        else:
            patience_counter += 1
        
        # Learning rate scheduler
        scheduler.step(val_loss)
        
        if patience_counter >= patience:
            logger.info(f"Early stopping at epoch {epoch}")
            break
    
    logger.info("=" * 70)
    logger.info("✅ Training complete!")
    logger.info(f"   Best val_loss: {best_val_loss:.6f}")
    logger.info(f"   Model saved to: {output_path}")
    logger.info("=" * 70)
    
    return model, history


if __name__ == "__main__":
    import os
    
    # Adjust paths if running from src directory
    cwd = os.getcwd()
    if os.path.basename(cwd) == 'src':
        hdf5_path = "../data/processed/lightning_dataset.h5"
        output_path = "../models/lightning_classifier.pth"
    else:
        hdf5_path = "data/processed/lightning_dataset.h5"
        output_path = "models/lightning_classifier.pth"
    
    model, history = train_lightning_model(
        hdf5_path=hdf5_path,
        output_path=output_path,
        max_epochs=50,
        batch_size=512,
        learning_rate=0.001,
    )
