"""Minimal compatibility wrapper for the legacy preprocessing module used by older tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path


class HimawariPreprocessor:
    def __init__(self, raw_himawari_dir, raw_mmd_csv, output_h5, region_bbox, patch_size=64, num_channels=3):
        self.raw_himawari_dir = Path(raw_himawari_dir)
        self.raw_mmd_csv = Path(raw_mmd_csv)
        self.output_h5 = Path(output_h5)
        self.region_bbox = region_bbox
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.downsample_ratio = 0.2

    def _normalize_channels(self, image):
        image = np.asarray(image, dtype=np.float32)
        return np.clip(image / 255.0, 0.0, 1.0)

    def create_patches(self, image, stride=None):
        stride = stride or self.patch_size
        image = np.asarray(image)
        if image.ndim != 3:
            raise ValueError('Expected image with shape (C, H, W)')
        if image.shape[1] < self.patch_size or image.shape[2] < self.patch_size:
            return [image[:, : self.patch_size, : self.patch_size]]
        return [image[:, i:i + self.patch_size, j:j + self.patch_size] for i in range(0, image.shape[1], stride) for j in range(0, image.shape[2], stride)]

    def label_patch(self, patch_center_lat, patch_center_lon, timestamp, lightning_df):
        if lightning_df is None or lightning_df.empty:
            return 0
        return int(((lightning_df['latitude'] - patch_center_lat).abs() < 0.5).any())

    def _balance_dataset(self, patches, labels):
        labels = np.asarray(labels)
        positive_idx = np.where(labels == 1)[0]
        negative_idx = np.where(labels == 0)[0]
        if len(positive_idx) == 0:
            return patches, labels
        if len(negative_idx) > len(positive_idx):
            negative_idx = negative_idx[: len(positive_idx)]
        balanced_idx = np.concatenate([positive_idx, negative_idx])
        return [patches[i] for i in balanced_idx], labels[balanced_idx].tolist()
