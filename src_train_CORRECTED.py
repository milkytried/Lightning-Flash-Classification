"""
Training loop with validation, checkpointing, early stopping.
"""

from typing import Dict, Tuple, Optional
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


def set_seed(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility across numpy, torch, and cuda.
    
    Args:
        seed (int): Random seed value
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_epoch(
    model: nn.Module, 
    train_loader, 
    optimizer, 
    criterion: nn.Module, 
    device: torch.device
) -> float:
    """
    Execute single training epoch.
    
    Args:
        model (nn.Module): Neural network model
        train_loader: DataLoader for training data
        optimizer: PyTorch optimizer
        criterion (nn.Module): Loss function
        device (torch.device): Device to train on (cuda or cpu)
    
    Returns:
        float: Average loss over epoch
    """
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    try:
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
            num_batches += 1
    except Exception as e:
        raise RuntimeError(f"Error during training epoch: {str(e)}")
    
    return total_loss / max(num_batches, 1)


def validate(
    model: nn.Module, 
    val_loader, 
    criterion: nn.Module, 
    device: torch.device
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Execute validation epoch.
    
    Args:
        model (nn.Module): Neural network model
        val_loader: DataLoader for validation data
        criterion (nn.Module): Loss function
        device (torch.device): Device to validate on
    
    Returns:
        Tuple[float, np.ndarray, np.ndarray]: (average_loss, predictions, labels)
    """
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    num_batches = 0
    
    try:
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc='Val'):
                images, labels = images.to(device), labels.to(device).unsqueeze(1)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                total_loss += loss.item()
                all_preds.append(outputs.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
                num_batches += 1
    except Exception as e:
        raise RuntimeError(f"Error during validation: {str(e)}")
    
    avg_loss = total_loss / max(num_batches, 1)
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    return avg_loss, all_preds, all_labels


def _validate_config(config: Dict) -> None:
    """
    Validate that all required config keys exist.
    
    Args:
        config (Dict): Configuration dictionary
    
    Raises:
        KeyError: If required keys are missing
        ValueError: If config values are invalid
    """
    required_keys = {
        'data': ['processed_dataset'],
        'model': ['num_input_channels', 'dropout'],
        'train': [
            'device', 'batch_size', 'max_epochs', 'learning_rate',
            'loss_alpha', 'loss_gamma', 'lr_scheduler_factor',
            'lr_scheduler_patience', 'early_stopping_patience'
        ],
        'paths': ['models_dir', 'results_dir', 'logs_dir']
    }
    
    for section, keys in required_keys.items():
        if section not in config:
            raise KeyError(f"Missing config section: '{section}'")
        for key in keys:
            if key not in config[section]:
                raise KeyError(f"Missing config key: '{section}.{key}'")
    
    # Validate value types/ranges
    if config['train']['batch_size'] <= 0:
        raise ValueError(f"batch_size must be positive, got {config['train']['batch_size']}")
    if config['train']['max_epochs'] <= 0:
        raise ValueError(f"max_epochs must be positive, got {config['train']['max_epochs']}")
    if config['train']['learning_rate'] <= 0:
        raise ValueError(f"learning_rate must be positive, got {config['train']['learning_rate']}")


def train_full(config_path: str = 'config.yaml') -> Tuple[nn.Module, Dict]:
    """
    Full training pipeline with validation and early stopping.
    
    Args:
        config_path (str): Path to configuration YAML file
    
    Returns:
        Tuple[nn.Module, Dict]: (trained model, training history)
    
    Raises:
        FileNotFoundError: If config file not found
        KeyError: If config is incomplete
        RuntimeError: If training fails
    """
    # Set seed for reproducibility
    set_seed(42)
    
    # Load and validate config
    try:
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        if config is None:
            raise ValueError(f"Config file is empty: {config_path}")
        
        _validate_config(config)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Could not read config: {str(e)}")
    except (KeyError, ValueError) as e:
        raise RuntimeError(f"Invalid config: {str(e)}")
    except yaml.YAMLError as e:
        raise RuntimeError(f"Failed to parse YAML config: {str(e)}")
    
    # Create output directories
    try:
        Path(config['paths']['models_dir']).mkdir(parents=True, exist_ok=True)
        Path(config['paths']['results_dir']).mkdir(parents=True, exist_ok=True)
        Path(config['paths']['logs_dir']).mkdir(parents=True, exist_ok=True)
    except Exception as e:
        raise RuntimeError(f"Failed to create output directories: {str(e)}")
    
    # Setup device
    device = torch.device(
        config['train']['device'] if torch.cuda.is_available() else 'cpu'
    )
    print(f"Using device: {device}")
    
    # Initialize model, loss, optimizer
    try:
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
    except Exception as e:
        raise RuntimeError(f"Failed to initialize model/optimizer: {str(e)}")
    
    # Load data
    try:
        print("Loading data...")
        loaders = create_data_loaders(
            config['data']['processed_dataset'],
            batch_size=config['train']['batch_size']
        )
    except Exception as e:
        raise RuntimeError(f"Failed to load data: {str(e)}")
    
    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': []}
    
    print("Starting training...")
    try:
        for epoch in range(config['train']['max_epochs']):
            train_loss = train_epoch(model, loaders['train'], optimizer, criterion, device)
            val_loss, _, _ = validate(model, loaders['val'], criterion, device)
            
            scheduler.step(val_loss)
            
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            
            print(f"Epoch {epoch+1}/{config['train']['max_epochs']}: "
                  f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                
                model_path = Path(config['paths']['models_dir']) / 'best_resnet50.pth'
                torch.save(model.state_dict(), str(model_path))
                print(f"  → Best model saved (val_loss={val_loss:.4f})")
            else:
                patience_counter += 1
                if patience_counter >= config['train']['early_stopping_patience']:
                    print(f"Early stopping at epoch {epoch+1} (patience exceeded)")
                    break
    except KeyboardInterrupt:
        print("Training interrupted by user")
    except Exception as e:
        raise RuntimeError(f"Error during training: {str(e)}")
    
    # Save history
    try:
        history_path = Path(config['paths']['results_dir']) / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
    except Exception as e:
        print(f"Warning: Failed to save training history: {str(e)}")
    
    print("\nTraining complete!")
    model_path = Path(config['paths']['models_dir']) / 'best_resnet50.pth'
    print(f"Best model saved to: {model_path}")
    
    return model, history


if __name__ == '__main__':
    try:
        model, history = train_full('config.yaml')
        print("Training pipeline finished!")
    except Exception as e:
        print(f"Training failed: {str(e)}")
        exit(1)
