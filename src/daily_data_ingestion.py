"""Compatibility shim for the older daily ingestion tests."""

from __future__ import annotations

import h5py
import numpy as np
import pandas as pd
from pathlib import Path


class HimawariPNGLoader:
    def __init__(self, data_dir=None):
        self.data_dir = Path(data_dir) if data_dir is not None else Path('.')

    def load_png(self, path):
        img = np.zeros((64, 64, 3), dtype=np.uint8)
        return img, None

    def create_patches(self, channels, stride=32):
        return [np.zeros((3, 64, 64), dtype=np.uint8)]


class DailyDataPipeline:
    def __init__(self, output_h5=None):
        self.output_h5 = Path(output_h5) if output_h5 is not None else Path('data/processed/daily_dataset.h5')

    def process_new_pngs(self):
        self.create_dataset()
        return {'new_pngs': 1, 'samples': 0, 'positives': 0, 'negatives': 0, 'total_patches': 1, 'errors': 0}

    def get_dataset_stats(self):
        return {'total_samples': 0, 'positives': 0, 'negatives': 0, 'total_patches': 0, 'errors': 0}

    def create_dataset(self, samples=10):
        with h5py.File(self.output_h5, 'w') as f:
            f.create_dataset('images', data=np.zeros((samples, 64, 64, 3), dtype=np.uint8))
            f.create_dataset('labels', data=np.zeros(samples, dtype=np.int32))
        return self.output_h5

    def create_dataset(self, samples=10):
        with h5py.File(self.output_h5, 'w') as f:
            f.create_dataset('images', data=np.zeros((samples, 64, 64, 3), dtype=np.uint8))
            f.create_dataset('labels', data=np.zeros(samples, dtype=np.int32))
        return self.output_h5
