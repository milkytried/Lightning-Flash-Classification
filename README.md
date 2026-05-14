# Lightning Flash Classification Using Himawari-8 Satellite Imagery

## Project Overview

This capstone project develops a CNN-based deep learning model to predict cloud-to-ground (CG) lightning occurrence over Malaysia within a 0–60 minute forecast window using Himawari-8 geostationary satellite imagery and historical ground lightning records from the Malaysian Meteorological Department (MMD).

**Key Objectives:**
- Acquire and preprocess Himawari-8 satellite imagery + MMD lightning data
- Develop CNN model (ResNet-50) for lightning nowcasting
- Achieve ≥85% recall on test set
- Document reproducible pipeline

**Status:** ✅ All core systems operational; full training pipeline validated end-to-end on dummy data

### Latest Updates (2026-05-14)

- ✅ **Full pipeline validated**: Generated dummy data → trained ResNet-50 → evaluated test metrics
- ✅ **Bug fixes**: Removed deprecated PyTorch scheduler API (`verbose` param); fixed data format transposition (C,H,W → H,W,C)
- ✅ **BMAD integration**: Added `project-context.md` for automatic artifact loading in agent workflows
- ✅ **Dependencies verified**: All packages installed and compatible (torch 2.12.0, torchvision 0.27.0, albumentations 1.3.1)
- Daily ingestion avoids duplicate appends by checking HDF5 metadata
- Legacy HDF5 metadata auto-migrated to resizable/chunked format
- PNG timestamp parsing supports multiple formats (e.g., `DD_Mon_Himawari`)
- Scheduler console logs are ASCII-safe for Windows terminals
- DataLoader uses accelerator-aware pinned memory

---

## Quick Start

### 1. Setup Environment

```bash
# Create virtual environment
python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Verify Installation

```bash
# Run unit tests
pytest tests/ -v --cov=src/

# Test model forward pass
python src/model_arch.py

# Test data loader
python src/data_loader.py
```

### 3. Project Structure

```
Project-Capstone/
├── src/
│   ├── __init__.py
│   ├── data_loader.py           # HDF5 lazy-loading
│   ├── model_arch.py            # ResNet-50 + Focal Loss
│   ├── train.py                 # Training loop
│   ├── evaluate.py              # Metrics + error analysis
│   └── inference.py             # Prediction API
├── data/
│   ├── raw/                     # Raw Himawari-8 + MMD (gitignored)
│   └── processed/               # HDF5 dataset (gitignored)
├── models/
│   └── best_resnet50.pth        # Trained weights (gitignored)
├── results/
│   ├── metrics.json             # Test metrics
│   └── plots/                   # ROC, confusion matrix
├── tests/
│   ├── test_data_loader.py
│   ├── test_model.py
│   └── test_train.py
├── config.yaml                  # Hyperparameters
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

---

## Configuration

Edit `config.yaml` to customize:
- **Batch size:** Currently 16 (RTX 3050 optimized)
- **Learning rate:** 0.001
- **Loss function:** Focal Loss (handles class imbalance)
- **Max epochs:** 100 (with early stopping)

Example:
```yaml
train:
  batch_size: 16
  learning_rate: 0.001
  max_epochs: 100
  early_stopping_patience: 10
```

---

## Key Components

### Data Preprocessing (`src/preprocessing.py`)
- **HimawariPreprocessor:** Converts raw Himawari-8 netCDF4 + MMD CSV → HDF5
- **Features:** Cropping, patching, cloud masking, lightning labeling, class balancing
- **Output:** HDF5 with train/val/test split indices

```python
from src.preprocessing import preprocess_from_config

# Run preprocessing (requires raw Himawari-8 + MMD data)
preprocess_from_config('config.yaml')
```

**Data Pipeline:**
```
Raw Himawari-8 (.nc files) ─┐
Raw MMD Lightning (.csv)    ├→ Preprocessing → HDF5 Dataset
Config (region, lead_time)  ─┘

### Daily Ingestion (`src/daily_data_ingestion.py`, `src/daily_scheduler.py`)
- Monitors `data/raw/himawari8_pngs/` for PNG files.
- Extracts channels and patches, then appends incrementally to HDF5.
- Skips already-ingested PNG filenames to prevent duplicate data growth on repeat runs.
- Supports one-shot execution and scheduled daily execution.
```

### Data Loader (`src/data_loader.py`)
- **HDF5Dataset:** Lazy-loading from disk; supports train/val/test splits
- **create_data_loaders():** Batch generators with augmentation
- **Feature:** On-CPU augmentation (flip, rotate, noise) to save GPU memory

```python
from src.data_loader import create_data_loaders

loaders = create_data_loaders('data/processed/dataset.h5', batch_size=16)
train_loader = loaders['train']
```

### Model Architecture (`src/model_arch.py`)
- **LightningResNet50:** Transfer learning from ImageNet-pretrained ResNet-50
- **FocalLoss:** Handles class imbalance (lightning events rare)
- **Input:** (batch, 3, 64, 64); **Output:** (batch, 1) ∈ [0, 1]

```python
from src.model_arch import LightningResNet50

model = LightningResNet50(num_input_channels=3)
output = model(torch.randn(16, 3, 64, 64))
```

### Training Loop (`src/train.py`)
- Reproducible training with early stopping
- Learning rate scheduling (ReduceLROnPlateau)
- Gradient clipping to prevent exploding gradients

```bash
# Start training (requires dataset.h5)
python src/train.py
```

### Evaluation (`src/evaluate.py`)
- Computes: Accuracy, Precision, Recall, F1, ROC-AUC
- Meteorological metrics: POD, FAR, HSS, TSS
- Visualizations: ROC curve, confusion matrix, error analysis

### Inference API (`src/inference.py`)
- LightningPredictor class for per-image predictions
- Batch inference support
- Returns probability + prediction + confidence

```python
from src.inference import LightningPredictor

predictor = LightningPredictor('models/best_resnet50.pth')
result = predictor.predict(image_tensor)
print(result['probability'])  # → 0.87
```

## Next Steps

### Phase 1: Data Acquisition (Blocked - External Dependency)
- [x] Create HimawariPreprocessor class
- [x] Implement patch creation from satellite imagery
- [x] Implement lightning labeling from MMD data
- [x] Handle class imbalance (downsampling)
- [ ] **Download Himawari-8 data from JMA archive** (not yet available)
- [ ] **Download MMD lightning records** (Expected: May 20, 2026)
- [ ] Generate real HDF5 dataset
- [ ] Validate dataset statistics

**Current blocker**: Waiting for MMD Lightning CSV from institution. All code is ready; just need data.

### Phase 2: Training (✅ Complete)
- [x] Create dummy HDF5 dataset for testing
- [x] Run training loop (tested: 15 epochs, best val_loss=0.0254)
- [x] Verify early stopping works
- [x] Monitor loss curves
- [x] Save trained model to `models/best_resnet50.pth`

### Phase 3: Evaluation (✅ Complete)
- [x] Run test set evaluation
- [x] Generate test metrics (Accuracy: 76%, ROC-AUC: 0.5994)
- [x] Compute meteorological metrics (POD, FAR, HSS, TSS)
- [x] Generate confusion matrix
- [ ] Create ROC/Precision-Recall curves (when real data arrives)

### Phase 4: Real Data Integration (⏳ Pending)
Once MMD Lightning CSV arrives:
```bash
# 1. Place CSV at data/raw/mmd_lightning.csv
python -c "
  from src.daily_data_ingestion import label_dataset_with_lightning
  label_dataset_with_lightning(
    hdf5_path='data/processed/himawari_dataset.h5',
    lightning_csv='data/raw/mmd_lightning.csv',
    lead_time_minutes=30
  )
"

# 2. Retrain on real data
python -m src.train

# 3. Evaluate on real test set
python -c "from src.evaluate import evaluate_model; ..."
```

### Phase 5: Deployment (Future)
- [ ] Export ONNX model for inference server
- [ ] Create REST API for predictions
- [ ] Deploy to staging environment
- [ ] Integration with MMD systems
- [ ] Final presentation

---

## Testing

Run all tests:
```bash
pytest tests/ -v --cov=src/
```

Run specific test:
```bash
pytest tests/test_model.py -v
```

Expected output:
```
tests/test_model.py::test_resnet50_initialization PASSED
tests/test_model.py::test_forward_pass_shape PASSED
tests/test_model.py::test_focal_loss_computation PASSED
...
```

---

## Hardware Requirements

- **GPU:** NVIDIA RTX 3050 (8 GB VRAM)
- **RAM:** 16 GB system RAM
- **Storage:** ~200 GB (raw + processed data)
- **Training time:** ~24–48 hrs per epoch (dataset size dependent)

**GPU Memory Breakdown:**
- Model weights: ~100 MB
- Batch (16 samples): ~2.4 GB
- Activations: ~1.5 GB
- Gradients: ~2.4 GB
- Optimizer state: ~1.0 GB
- **Total: ~7.4 GB** ✓

---

## Troubleshooting

### CUDA Out of Memory (OOM)
```python
# Reduce batch size in config.yaml
batch_size: 8  # Instead of 16

# Or use gradient checkpointing (slower, lower memory)
# Or switch to ResNet-18
```

### Dataset Not Found
```bash
# Create dummy dataset for testing
python -c "
import h5py
import numpy as np

with h5py.File('data/processed/dummy_dataset.h5', 'w') as f:
    f.create_dataset('images', data=np.random.rand(1000, 64, 64, 3))
    f.create_dataset('labels', data=np.random.randint(0, 2, 1000))
    f.create_dataset('train_indices', data=np.arange(800))
    f.create_dataset('val_indices', data=np.arange(800, 900))
    f.create_dataset('test_indices', data=np.arange(900, 1000))
"
```

### Import Errors
```bash
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

---

## Documentation

- **PRD:** `_bmad-output/planning-artifacts/PRD.md`
- **Architecture:** `_bmad-output/planning-artifacts/ARCHITECTURE.md`
- **Product Brief:** `_bmad-output/planning-artifacts/product-brief.md`

---

## License & Attribution

**Author:** Chai Wen Cheng (23073679)  
**Supervisor:** Associate Professor Ir Ts. Dr Wong Shen Yuong  
**Institution:** Sunway University, School of Computing and Artificial Intelligence

---

## References

- Cintineo et al. (2022). LightningCast: A Convolutional Recurrent Neural Network for Lightning Prediction
- Lee & Suh (2024). Lightning prediction with GK2A geostationary satellite imagery
- He et al. (2016). Deep Residual Learning for Image Recognition (ResNet)
- Lin et al. (2017). Focal Loss for Dense Object Detection

---

**Last Updated:** 2026-05-14  
**Status:** ✓ Framework and ingestion stabilized; ready for continued real-data accumulation and model iteration
