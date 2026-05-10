"""
Training loop with validation, checkpointing, early stopping.
"""

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm
import yaml

from src.model_arch import LightningResNet50, FocalLoss
from src.data_loader import create_data_loaders


def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_epoch(model, train_loader, optimizer, criterion, device):
    """Single training epoch."""
    model.train()
    total_loss = 0
    
    for batch_idx, (images, labels) in enumerate(tqdm(train_loader, desc='Train')):
        images, labels = images.to(device), labels.to(device).unsqueeze(1)
        
        # Forward
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)


def validate(model, val_loader, criterion, device):
    """Validation epoch."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc='Val'):
            images, labels = images.to(device), labels.to(device).unsqueeze(1)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            all_preds.append(outputs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    
    avg_loss = total_loss / len(val_loader)
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    
    return avg_loss, all_preds, all_labels


def train_full(config_path='config.yaml'):
    """Full training pipeline."""
    
    # Set seed for reproducibility
    set_seed(42)
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create output directories
    Path(config['paths']['models_dir']).mkdir(parents=True, exist_ok=True)
    Path(config['paths']['results_dir']).mkdir(parents=True, exist_ok=True)
    Path(config['paths']['logs_dir']).mkdir(parents=True, exist_ok=True)
    
    # Setup
    device = torch.device(config['train']['device'] if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = LightningResNet50(
        num_input_channels=config['model']['num_input_channels'],
        dropout_rate=config['model']['dropout']
    ).to(device)
    
    criterion = FocalLoss(
        alpha=config['train']['loss_alpha'],
        gamma=config['train']['loss_gamma']
    )
    
    optimizer = Adam(model.parameters(), lr=config['train']['learning_rate'])
    
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=config['train']['lr_scheduler_factor'],
        patience=config['train']['lr_scheduler_patience'],
        verbose=True
    )
    
    # Data
    print("Loading data...")
    loaders = create_data_loaders(
        config['data']['processed_dataset'],
        batch_size=config['train']['batch_size']
    )
    
    # Training
    best_val_loss = float('inf')
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': []}
    
    print("Starting training...")
    for epoch in range(config['train']['max_epochs']):
        train_loss = train_epoch(model, loaders['train'], optimizer, criterion, device)
        val_loss, _, _ = validate(model, loaders['val'], criterion, device)
        
        scheduler.step(val_loss)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
        print(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(
                model.state_dict(),
                f"{config['paths']['models_dir']}/best_resnet50.pth"
            )
            print(f"  → Best model saved (val_loss={val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= config['train']['early_stopping_patience']:
                print(f"Early stopping at epoch {epoch}")
                break
    
    # Save history
    with open(f"{config['paths']['results_dir']}/training_history.json", 'w') as f:
        json.dump(history, f)
    
    print("Training complete!")
    print(f"Best model saved to: {config['paths']['models_dir']}/best_resnet50.pth")
    
    return model, history


if __name__ == '__main__':
    model, history = train_full('config.yaml')
    print("Training pipeline finished!")
