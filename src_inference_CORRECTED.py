"""
Inference API for per-image predictions.
"""

from typing import Dict, List, Optional, Union, Any
import torch
import numpy as np
from pathlib import Path
import yaml

from src.model_arch import LightningResNet50


class LightningPredictor:
    """
    Inference API for making predictions on Himawari-8 satellite images.
    Supports single-image and batch inference with configurable device handling.
    """
    
    def __init__(
        self, 
        model_path: str, 
        config_path: str = 'config.yaml', 
        device: Optional[str] = None
    ) -> None:
        """
        Initialize predictor with model and configuration.
        
        Args:
            model_path (str): Path to saved model weights (.pth file)
            config_path (str): Path to config.yaml
            device (Optional[str]): 'cuda' or 'cpu' (auto-detected if None)
        
        Raises:
            FileNotFoundError: If model or config file not found
            RuntimeError: If model loading fails
        """
        # Load config
        try:
            config_path = Path(config_path)
            if not config_path.exists():
                raise FileNotFoundError(f"Config not found: {config_path}")
            
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
            
            if self.config is None:
                raise ValueError(f"Config is empty: {config_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to load config: {str(e)}")
        
        # Setup device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            try:
                self.device = torch.device(device)
            except RuntimeError as e:
                raise RuntimeError(f"Invalid device '{device}': {str(e)}")
        
        # Load model
        try:
            model_path = Path(model_path)
            if not model_path.exists():
                raise FileNotFoundError(f"Model not found: {model_path}")
            
            self.model = LightningResNet50(
                num_input_channels=self.config['model']['num_input_channels']
            )
            
            # Load weights with device mapping
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint)
            self.model.to(self.device).eval()
            
            print(f"Model loaded from {model_path}")
            print(f"Device: {self.device}")
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Could not load model: {str(e)}")
        except Exception as e:
            raise RuntimeError(f"Failed to load model weights: {str(e)}")
    
    def predict(
        self, 
        image_tensor: Union[np.ndarray, torch.Tensor],
        lead_time_window: tuple = (0, 60)
    ) -> Dict[str, Any]:
        """
        Predict lightning probability for a single image.
        
        Args:
            image_tensor (Union[np.ndarray, torch.Tensor]): 
                Image array, shape (C, H, W) or (H, W, C)
                Values should be in [0, 1] or [0, 255] range
            lead_time_window (tuple): Lead time in minutes (start, end)
        
        Returns:
            Dict[str, Any]: {
                'probability': float in [0, 1],
                'prediction': int in {0, 1},
                'confidence': float in [0.5, 1.0],
                'lead_time_window': tuple,
                'threshold': float
            }
        
        Raises:
            ValueError: If image format invalid
            RuntimeError: If prediction fails
        """
        try:
            # Convert to tensor if needed
            if isinstance(image_tensor, np.ndarray):
                # Handle (H, W, C) format
                if image_tensor.ndim == 3 and image_tensor.shape[-1] in [1, 3, 5]:
                    image_tensor = np.transpose(image_tensor, (2, 0, 1))
                elif image_tensor.ndim != 3:
                    raise ValueError(f"Expected 3D array, got shape {image_tensor.shape}")
                
                image_tensor = torch.from_numpy(image_tensor).float()
            elif not isinstance(image_tensor, torch.Tensor):
                raise ValueError(f"Expected np.ndarray or torch.Tensor, got {type(image_tensor)}")
            
            # Add batch dimension if needed
            if image_tensor.ndim == 3:
                image_tensor = image_tensor.unsqueeze(0)
            elif image_tensor.ndim != 4:
                raise ValueError(f"Expected 3D or 4D tensor, got {image_tensor.ndim}D")
            
            # Move to device
            image_tensor = image_tensor.to(self.device)
            
            # Predict
            with torch.no_grad():
                output = self.model(image_tensor)
                # Use squeeze(0) to remove batch dimension only, not other dims
                if output.shape[0] == 1:
                    prob = output.squeeze(0).item()
                else:
                    raise RuntimeError(f"Unexpected output shape: {output.shape}")
            
            # Ensure probability is valid
            prob = max(0.0, min(1.0, prob))
            
            prediction = 1 if prob > 0.5 else 0
            confidence = max(prob, 1 - prob)
            
            return {
                'probability': float(prob),
                'prediction': int(prediction),
                'confidence': float(confidence),
                'lead_time_window': lead_time_window,
                'threshold': 0.5
            }
        except Exception as e:
            raise RuntimeError(f"Prediction failed: {str(e)}")
    
    def predict_batch(
        self, 
        image_list: List[Union[np.ndarray, torch.Tensor]],
        batch_size: int = 16,
        lead_time_window: tuple = (0, 60)
    ) -> List[Dict[str, Any]]:
        """
        Batch inference for multiple images.
        
        Args:
            image_list (List[Union[np.ndarray, torch.Tensor]]): List of image tensors
            batch_size (int): Batch size for inference (RTX 3050: recommend ≤16)
            lead_time_window (tuple): Lead time in minutes
        
        Returns:
            List[Dict[str, Any]]: List of prediction dictionaries
        
        Raises:
            ValueError: If image_list is empty or invalid
            RuntimeError: If batch inference fails
        """
        if not image_list:
            raise ValueError("image_list cannot be empty")
        
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        
        results = []
        
        try:
            for i in range(0, len(image_list), batch_size):
                batch = image_list[i:i+batch_size]
                
                # Stack batch
                batch_tensors = []
                for img in batch:
                    if isinstance(img, np.ndarray):
                        # Handle (H, W, C) format
                        if img.ndim == 3 and img.shape[-1] in [1, 3, 5]:
                            img = np.transpose(img, (2, 0, 1))
                        batch_tensors.append(torch.from_numpy(img).float())
                    else:
                        batch_tensors.append(img.float())
                
                batch_tensor = torch.stack(batch_tensors).to(self.device)
                
                # Predict
                with torch.no_grad():
                    outputs = self.model(batch_tensor)
                    probs = outputs.squeeze(1).cpu().numpy()
                
                # Handle scalar case (single sample)
                if probs.ndim == 0:
                    probs = np.array([probs])
                
                # Format results
                for prob in probs:
                    prob = float(prob)
                    prob = max(0.0, min(1.0, prob))
                    
                    prediction = 1 if prob > 0.5 else 0
                    confidence = max(prob, 1 - prob)
                    
                    results.append({
                        'probability': prob,
                        'prediction': int(prediction),
                        'confidence': float(confidence),
                        'lead_time_window': lead_time_window,
                        'threshold': 0.5
                    })
        except Exception as e:
            raise RuntimeError(f"Batch prediction failed: {str(e)}")
        
        return results


if __name__ == '__main__':
    # Example usage
    try:
        predictor = LightningPredictor(
            'models/best_resnet50.pth',
            config_path='config.yaml'
        )
        
        # Test single prediction
        dummy_image = np.random.rand(3, 64, 64).astype(np.float32)
        result = predictor.predict(dummy_image)
        print(f"Single prediction result: {result}")
        
        # Test batch prediction
        dummy_batch = [np.random.rand(3, 64, 64).astype(np.float32) for _ in range(5)]
        batch_results = predictor.predict_batch(dummy_batch, batch_size=4)
        print(f"Batch results ({len(batch_results)} samples): {batch_results[0]}")
        
    except FileNotFoundError as e:
        print(f"Model not found: {e}")
        print("Train the model first using src/train.py")
    except Exception as e:
        print(f"Error: {e}")
