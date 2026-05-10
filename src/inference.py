"""
Inference API for per-image predictions.
"""

import torch
import numpy as np
from pathlib import Path
import yaml

from src.model_arch import LightningResNet50


class LightningPredictor:
    """
    Inference API for making predictions on Himawari-8 satellite images.
    """
    
    def __init__(self, model_path, config_path='config.yaml', device=None):
        """
        Args:
            model_path (str): Path to saved model weights (.pth file)
            config_path (str): Path to config.yaml
            device (str): 'cuda' or 'cpu' (auto-detected if None)
        """
        # Load config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Setup device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Load model
        self.model = LightningResNet50(
            num_input_channels=self.config['model']['num_input_channels']
        )
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device).eval()
        
        print(f"Model loaded from {model_path}")
        print(f"Device: {self.device}")
    
    def predict(self, image_tensor, lead_time_window=(0, 60)):
        """
        Predict lightning probability for a single image.
        
        Args:
            image_tensor (np.ndarray or torch.Tensor): 
                Image array, shape (C, H, W) or (H, W, C)
                If (H, W, C), will be transposed to (C, H, W)
            lead_time_window (tuple): Lead time in minutes
        
        Returns:
            dict: {
                'probability': float [0, 1],
                'prediction': int [0, 1],
                'confidence': float,
                'lead_time_window': tuple
            }
        """
        # Convert to tensor if needed
        if isinstance(image_tensor, np.ndarray):
            # Handle (H, W, C) format
            if image_tensor.shape[-1] in [1, 3, 5]:
                image_tensor = np.transpose(image_tensor, (2, 0, 1))
            image_tensor = torch.from_numpy(image_tensor).float()
        
        # Add batch dimension
        image_tensor = image_tensor.unsqueeze(0).to(self.device)
        
        # Predict
        with torch.no_grad():
            prob = self.model(image_tensor).squeeze().item()
        
        prediction = 1 if prob > 0.5 else 0
        confidence = max(prob, 1 - prob)
        
        return {
            'probability': float(prob),
            'prediction': int(prediction),
            'confidence': float(confidence),
            'lead_time_window': lead_time_window,
            'threshold': 0.5
        }
    
    def predict_batch(self, image_list, batch_size=16, lead_time_window=(0, 60)):
        """
        Batch inference for multiple images.
        
        Args:
            image_list (list): List of image tensors
            batch_size (int): Batch size for inference
            lead_time_window (tuple): Lead time in minutes
        
        Returns:
            list: List of prediction dictionaries
        """
        results = []
        
        for i in range(0, len(image_list), batch_size):
            batch = image_list[i:i+batch_size]
            
            # Stack batch
            batch_tensor = torch.stack([
                torch.from_numpy(img).float() if isinstance(img, np.ndarray) 
                else img.float()
                for img in batch
            ]).to(self.device)
            
            # Predict
            with torch.no_grad():
                probs = self.model(batch_tensor).squeeze().cpu().numpy()
            
            # Format results
            if probs.ndim == 0:  # Single sample
                probs = np.array([probs])
            
            for prob in probs:
                prediction = 1 if prob > 0.5 else 0
                confidence = max(prob, 1 - prob)
                
                results.append({
                    'probability': float(prob),
                    'prediction': int(prediction),
                    'confidence': float(confidence),
                    'lead_time_window': lead_time_window,
                    'threshold': 0.5
                })
        
        return results


if __name__ == '__main__':
    # Example usage
    try:
        predictor = LightningPredictor(
            'models/best_resnet50.pth',
            config_path='config.yaml'
        )
        
        # Test prediction
        dummy_image = np.random.rand(3, 64, 64)
        result = predictor.predict(dummy_image)
        print(f"Prediction result: {result}")
        
    except FileNotFoundError as e:
        print(f"Model not found: {e}")
        print("Train the model first using src/train.py")
