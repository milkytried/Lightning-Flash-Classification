import importlib
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from src.lightning_data_loader import create_lightning_loaders
from src.lightning_model import LightningMetadataClassifier


@pytest.fixture(scope='module')
def hdf5_path():
    return Path('data/processed/lightning_dataset.h5')


def test_metadata_dataset_exists(hdf5_path):
    assert hdf5_path.exists(), 'Metadata dataset is missing; run python src/ingest_met_data.py first.'


def test_metadata_loader_shapes(hdf5_path):
    loaders = create_lightning_loaders(str(hdf5_path), batch_size=4)
    features, labels = next(iter(loaders['train']))
    assert features.shape == (4, 4)
    assert labels.shape == (4,)


def test_metadata_model_forward_pass():
    model = LightningMetadataClassifier(input_size=4, hidden_size=64, dropout=0.0)
    x = torch.randn(8, 4)
    out = model(x)
    assert out.shape == (8, 1)
    assert torch.all(out >= 0) and torch.all(out <= 1)
