"""Minimal compatibility wrapper for the legacy inference API used by older tests."""

from __future__ import annotations

import numpy as np
import torch
import yaml
from pathlib import Path

from src.model_arch import LightningResNet50


class LightningPredictor:
    def __init__(self, model_path, config_path=None, device='cpu'):
        self.device = torch.device(device)
        self.model_path = Path(model_path)
        if config_path is None:
            config_path = Path('config.yaml')
        self.config_path = Path(config_path)
        if not self.config_path.exists():
            raise FileNotFoundError('Config not found')
        if not self.model_path.exists():
            raise FileNotFoundError('Model weights not found')
        with open(self.config_path, 'r', encoding='utf-8') as fh:
            self.config = yaml.safe_load(fh)
        self.model = LightningResNet50(num_input_channels=3, pretrained=False).to(self.device)
        state = torch.load(self.model_path, map_location=self.device)
        if isinstance(state, dict) and 'state_dict' in state:
            state = state['state_dict']
        self.model.load_state_dict(state)
        self.model.eval()
        self.threshold = 0.5

    def predict(self, image):
        arr = np.asarray(image)
        if arr.ndim == 2:
            arr = np.repeat(arr[:, :, None], 3, axis=2)
        if arr.ndim == 3 and arr.shape[-1] != 3:
            arr = np.transpose(arr, (1, 2, 0))
        tensor = torch.from_numpy(arr.astype(np.float32)).permute(2, 0, 1) / 255.0
        tensor = tensor.unsqueeze(0).to(self.device)
        with torch.no_grad():
            prob = float(self.model(tensor).cpu().squeeze().item())
        pred = int(prob >= self.threshold)
        return {'probability': prob, 'prediction': pred, 'confidence': prob, 'threshold': self.threshold}

    def predict_batch(self, images, batch_size=16):
        return [self.predict(img) for img in images]
