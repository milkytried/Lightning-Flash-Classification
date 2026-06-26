# Lightning Flash Classification Using Real Malaysian Meteorological Data

## Project Overview

This capstone project develops a metadata-based deep learning classifier to predict cloud-to-ground (CG) lightning occurrence over Malaysia using historical ground lightning records from the Malaysian Meteorological Department (MMD). The current metadata result is now treated as an honest probe rather than a deployment-ready success because the feature set still mixes strike consequences with the prediction target.

**Key Objectives:**
- ✅ Ingest and preprocess 4-year MMD lightning strike dataset (5.3M records)
- ✅ Develop MLP classifier with Focal Loss for lightning detection
- Document a reproducible pipeline
- Investigate label leakage in the metadata negative sampling
- Report satellite results at a threshold chosen on validation data

**Status:** Dual-model prototype with a metadata classifier and a satellite CNN. The metadata classifier is not interview-safe as a deployment claim because it uses strike-derived features (amplitude, strike_type) to predict strike occurrence, which is circular. A clean lat/lon/time-only variant is evaluated below. The satellite model is reported with tuned-threshold metrics.

**Leakage reasoning:** amplitude and strike_type are observed only when a strike exists. A model that uses them to predict whether a strike occurred is learning a consequence of the event, not an independent precursor. That is why the metadata probe can look strong while still being scientifically weak.

### Repository Hygiene

- This repository keeps only production code, tests, and user-facing docs.
- Internal review artifacts, draft corrected copies, generated data, model weights, and BMAD workspace artifacts are intentionally excluded from version control.

### Recent Milestones

**Metadata Classifier (2026-06-26 — Honest Evaluation):**
- Ingested 5.3M real lightning strikes from MMD CSV files (4-year dataset)
- Created 581 MB HDF5 dataset with 70/15/15 train/val/test splits
- **ISSUE IDENTIFIED:** Negatives are still synthetic (generated amplitude and generated strike-type mix), which remains separable from real strike records
- **LEAKAGE:** Using amplitude and strike_type to predict lightning is circular — these features only exist IF a strike was detected
- **RETRAINING:** Models under evaluation:
  1. Leaky model (amplitude + strike_type): Shows how ground-truth features give false confidence
  2. Clean model (lat/lon/time only): Honest baseline using only genuinely independent features
The detailed leakage explanation is summarized below in the performance notes and the metadata evaluation section.

**Satellite CNN (Himawari-8) (2026-06-05):**
- Fresh training from scratch using a corrected chronological split
- Layer freezing optimization reduced training time to 9.1 hours on CPU
- Test evaluation on 46,796 unseen satellite patches
- Reported ROC-AUC 0.9199 and a tuned operating point of 87.65% accuracy, 86.01% precision, 89.93% recall, and 0.8792 F1 at threshold 0.55

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
# Run the metadata and satellite smoke tests
python -m pytest tests -q

# Run a lightweight demo over the current test split
python demo_inference.py
```

### 3. Project Structure

```
Project-Capstone/
├── src/
│   ├── __init__.py
│   ├── ingest_met_data.py       # Ingest MMD lightning CSV records into HDF5
│   ├── lightning_data_loader.py # Lazy-loading for metadata features
│   ├── lightning_model.py       # MLP classifier + Focal Loss
│   ├── train_lightning.py       # Metadata training loop with early stopping
│   ├── evaluate_lightning.py    # Metadata test-set evaluation
│   ├── himawari_data_loader.py  # Satellite patch loader for the CNN
│   ├── train_satellite.py       # Satellite CNN training driver
│   └── model_arch.py            # ResNet-50-based satellite model
├── data/
│   ├── raw/                     # Raw MMD CSV + Himawari-8 PNGs (gitignored)
│   │   ├── himawari8_pngs/      # 4-year PNG structure (2023-2026)
│   │   └── mmd_lightning/       # Met Dept CSV files (309 files, 5.3M records)
│   └── processed/
│       └── lightning_dataset.h5 # Processed dataset (581 MB, gitignored)
├── models/
│   ├── lightning_classifier.pth        # Trained metadata MLP (0.2 MB) ✅
│   ├── satellite_resnet50_fresh.pth    # Himawari-8 CNN checkpoint (91 MB, gitignored)
│   ├── model_metadata_fresh.json       # Satellite CNN metadata (split verification)
│   ├── test_evaluation_fresh.json      # Satellite CNN test metrics
│   └── best_resnet50.pth               # Legacy ResNet-50 weights (gitignored)
├── results/
│   ├── training_history.json    # Training metrics
│   └── plots/                   # ROC, confusion matrix
├── tests/
│   ├── test_data_loader.py
│   ├── test_model.py
│   └── test_train.py
├── config.yaml                  # Hyperparameters
├── requirements.txt             # Python dependencies
├── SATELLITE_MODEL_FRESH_REPORT.md     # Satellite CNN comprehensive report
├── FRESH_TRAINING_STATUS.md            # Satellite CNN training status log
├── TRAINING_FAILURE_DIAGNOSIS.md       # Root cause analysis (reference)
├── README.md                    # This file
```

---

## Component 1: Lightning Metadata Classifier

Metadata-based deep learning model for lightning occurrence prediction using real strike records from the Malaysian Meteorological Department.

### Configuration (Lightning Metadata Model)

### Training Hyperparameters
- **Batch size:** 512
- **Learning rate:** 0.001 (Adam optimizer)
- **Loss function:** Focal Loss (α=0.25, γ=2.0) for class imbalance
- **Optimizer:** Adam with gradient clipping (norm=1.0)
- **LR Scheduler:** ReduceLROnPlateau (factor=0.5, patience=5)
- **Early stopping:** patience=10
- **Max epochs:** 50
- **Actual epochs trained:** ~10-15 (early stopped)
- **Training time:** ~1h 50m (CPU)

### Model Architecture
```
Input (4 features: lat_norm, lon_norm, amp_norm, strike_code)
  ↓
Linear(4 → 256)
  ↓ BatchNorm1d → ReLU → Dropout(0.5)
  ↓
Linear(256 → 128) → ReLU → Dropout(0.5)
  ↓
Linear(128 → 64) → ReLU → Dropout(0.5)
  ↓
Linear(64 → 1) → Sigmoid
  ↓
Output: Binary probability [0, 1]

Total parameters: 43,393
```

---

## Key Components (Current Implementation)

### Data Ingestion (`src/ingest_met_data.py`)
- **Purpose:** Parse 309 MMD CSV files with 5.3M lightning strike records
- **Functions:**
  - `scan_lightning_csvs(data_root)`: Finds CSV files, aggregates strike records
  - `create_labeled_dataset(strikes_df)`: Creates positive samples (all strikes) + negative samples (no-strike days)
- **Output:** 581 MB HDF5 with 70/15/15 train/val/test split

```bash
# Ingest MMD data (one-time)
python src/ingest_met_data.py
# Output: data/processed/lightning_dataset.h5 (5.38M samples)
```

### Data Loader (`src/lightning_data_loader.py`)
- **LightningMetadataDataset:** Lazy-loads metadata features from HDF5
- **Features:** Normalized latitude, longitude, amplitude, strike type code
- **Normalization:** Lat [0°N, 6°N], Lon [99.5°E, 104.5°E], Amp clipped [-1, 1]

```python
from src.lightning_data_loader import create_lightning_loaders

loaders = create_lightning_loaders('data/processed/lightning_dataset.h5', batch_size=512)
train_loader = loaders['train']  # 3.77M samples
val_loader = loaders['val']      # 807K samples
test_loader = loaders['test']    # 807K samples
```

### Model Architecture (`src/lightning_model.py`)
- **LightningMetadataClassifier:** 4-layer MLP with BatchNorm + Dropout
- **Input:** 4 metadata features (lat, lon, amplitude, strike_type)
- **Output:** Binary probability [0, 1] via sigmoid
- **Loss:** Focal Loss for extreme class imbalance (99.84% positive)

```python
from src.lightning_model import LightningMetadataClassifier, FocalLoss

model = LightningMetadataClassifier(input_size=4, hidden_size=256)
criterion = FocalLoss(alpha=0.25, gamma=2.0)
output = model(torch.randn(512, 4))
```

### Training Loop (`src/train_lightning.py`)
- **Orchestrator:** Adam optimizer + ReduceLROnPlateau scheduler + Early stopping
- **Duration:** ~1h 50m on CPU (512 batch, 3.77M train samples)
- **Convergence:** Stopped at epoch ~12 via early stopping
- **Loss trajectory:** 0.1122 → 0.0000 (rapid convergence)

```bash
# Train metadata-based model
python src/train_lightning.py
# Output: models/lightning_classifier.pth (0.2 MB)
```

### Evaluation (`src/evaluate_lightning.py`)
- **Full evaluation** on the metadata test split
- **Metrics:** Minority-class precision, recall, F1, PR-AUC, plus ROC-AUC/POD/FAR for reference
- **Note:** The metadata result should be interpreted cautiously because the feature set is still circular.

```bash
# Full metadata evaluation
python src/evaluate_lightning.py
```

### Full Evaluation (`src/evaluate_lightning.py`)
- **Full test set** evaluation on all 807K test samples
- **Metrics:** Minority-class precision, recall, F1, PR-AUC, ROC-AUC, POD, FAR
- **Note:** Slow on CPU; use `demo_inference.py` for a quick smoke test instead

```bash
# Full evaluation (CPU: ~10-15 min expected, may timeout)
python src/evaluate_lightning.py
```

---

## Project Milestones

### ✅ Phase 1: Data Ingestion (COMPLETE)
- [x] Ingest 309 MMD CSV files
- [x] Parse 5.3M lightning strike records
- [x] Create 581 MB HDF5 dataset
- [x] Generate 70/15/15 train/val/test split (3.77M/807K/807K)
- [x] Document the extreme class imbalance (99.84% positive)

**To ingest MMD data:**
```bash
# One-time ingestion (place CSV files in data/raw/mmd_lightning/)
python src/ingest_met_data.py
# Output: data/processed/lightning_dataset.h5 (581 MB)
```

### ✅ Phase 2: Model Training (COMPLETE)
- [x] Design metadata-based MLP classifier
- [x] Implement Focal Loss for class imbalance
- [x] Train on 3.77M samples
- [x] Converge in ~1h 50m
- [x] Early stop at epoch ~12

**To train model:**
```bash
python src/train_lightning.py
# Output: models/lightning_classifier.pth (0.2 MB)
```

### ✅ Phase 3: Evaluation (COMPLETE)
- [x] Report honest minority-class metrics for the metadata probe and clean probe
- [x] Validate metrics on the held-out test split
- [x] Generate evaluation report

**To evaluate:**
```bash
# Full eval (807K test samples, CPU: slow)
python src/evaluate_lightning.py
```

### ✅ Phase 4: Production (COMPLETE)
- [x] Commit all production code to GitHub
- [x] Document codebase
- [x] Verify no sensitive data leaked
- [x] Ready for deployment

**Repository Note:**
The metadata results above are retrain probes, not deployment claims. The clean lat/lon/time-only variant is the honest baseline; amplitude and strike_type are strike-derived fields and therefore circular for occurrence prediction.
create mode 100644 src/lightning_model.py
create mode 100644 src/train_lightning.py
```

---

## Component 2: Satellite CNN (Himawari-8)

Convolutional neural network for satellite-based lightning detection using Himawari-8 64×64 patch imagery.

### Architecture & Configuration (Satellite CNN)

**Model Design:**
- **Type:** LightningResNet50 (ResNet-50 backbone + custom head)
- **Input:** 64×64 RGB satellite patches (3 channels)
- **Output:** Binary classification (lightning vs. no lightning)
- **Total Parameters:** 23,770,433
- **Trainable Parameters:** 262,401 (1.1%) — backbone frozen
- **Training Strategy:** Layer freezing for CPU efficiency

**Training Configuration:**
- **Loss Function:** FocalLoss (α=0.25, γ=2.0)
- **Optimizer:** Adam (lr=0.001, gradient_clip=1.0)
- **Batch Size:** 32 (train), 256 (test)
- **Device:** CPU only
- **Max Epochs:** 15
- **Early Stopping:** patience=5
- **Duration:** 9.1 hours (vs. 62+ days for full fine-tuning)
- **Epochs Trained:** 13 (early stopped at plateau)

**Data Splits (Chronologically Separated):**
- **Train:** 395,952 patches from 6 PNGs (2025-04-18)
- **Validation:** 38,608 patches from 3 PNGs (2025-04-22)
- **Test:** 46,796 patches from 2 PNGs (2025-04-22, completely unseen)
- **Split Integrity:** ✅ Zero PNG overlap between any splits

### Test Evaluation Results (Satellite CNN)

**Classification Metrics (threshold tuned on validation set):**
| Metric | Value | Interpretation |
|--------|-------|---|
| Accuracy | 87.65% | Correct predictions at the tuned operating point |
| Precision | 86.01% | Fraction of predicted lightning patches that were correct |
| Recall/POD | 89.93% | Fraction of true lightning patches recovered |
| F1-Score | 0.8792 | Harmonic mean of precision and recall |
| **ROC-AUC** | **0.9199** | Ranking quality independent of threshold |

**Key Finding:** The model’s ranking quality is strong, but the default 0.5 threshold is too conservative; the tuned threshold of 0.55 gives a more balanced operating point.

### Satellite CNN Milestones (COMPLETE)

**Phase 1: Diagnosis**
- [x] Identified CPU bottleneck: ~10 sec/batch → 62+ days projected
- [x] Root cause: Full ResNet-50 fine-tuning on CPU
- [x] Solution: Layer freezing strategy (backbone frozen, head trainable)

**Phase 2: Implementation**
- [x] Implemented parameter freezing (23.7M → 262K trainable)
- [x] Verified gradient flow and parameter updates
- [x] Created optimized training script

**Phase 3: Training**
- [x] Fresh training from scratch on corrected split
- [x] Achieved 9.1-hour CPU training (160x speedup)
- [x] Early stopped at epoch 13 (no improvement)
- [x] Saved 91 MB checkpoint

**Phase 4: Evaluation & Reporting**
- [x] Test evaluation on 46,796 unseen patches
- [x] Computed all metrics: accuracy, precision, recall, F1, ROC-AUC, FAR, CSI, TSS, HSS
- [x] Generated comprehensive report with findings
- [x] Verified split integrity: Zero PNG leakage

**Reports & Documentation:**
- `SATELLITE_MODEL_FRESH_REPORT.md` — Comprehensive training report with metrics and recommendations
- `FRESH_TRAINING_STATUS.md` — Detailed training status and logs
- `TRAINING_FAILURE_DIAGNOSIS.md` — Root cause analysis (archived reference)

---

## Testing

Run all tests:
```bash
python -m pytest tests -q
```

Run a metadata smoke test:
```bash
python -m pytest tests/test_metadata_pipeline.py -q
```

---

## Hardware Requirements

### Current System (Metadata-Based MLP)
- **GPU:** Not required (runs on CPU)
- **CPU:** Any modern processor
- **RAM:** 8 GB minimum (16 GB recommended)
- **Storage:** ~1 GB (581 MB dataset + 0.2 MB model + overhead)
- **Training time:** ~1h 50m (CPU, batch_size=512)

### Optional: GPU Acceleration
- **GPU:** NVIDIA (any with >2 GB VRAM)
- **Expected speedup:** 10-20x faster training
- **Estimated GPU training time:** ~5-10 minutes
- **Memory needed:** ~500 MB GPU VRAM

### Previous System (ResNet-50, for reference)
- **GPU:** NVIDIA RTX 3050 (8 GB VRAM) required
- **RAM:** 16 GB system RAM
- **Training time:** ~24–48 hrs per epoch
- **Note:** Legacy satellite imagery model; currently replaced by metadata approach

---

## Troubleshooting

### Dataset Not Found (`lightning_dataset.h5`)
```bash
# Run data ingestion to create dataset
python src/ingest_met_data.py
# Requires MMD CSV files in: data/raw/mmd_lightning/
```

If the raw MMD files are unavailable, the repository can still be exercised with the shipped demo script once the processed HDF5 and model checkpoint exist. To obtain the full dataset, request the MMD CSV export or place the files under data/raw/mmd_lightning/ before running ingestion.

### Model File Not Found (`lightning_classifier.pth`)
```bash
# Run training to generate model
python src/train_lightning.py
# Output: models/lightning_classifier.pth (0.2 MB)
```

For a one-command smoke test once the data and weights are present:
```bash
python demo_inference.py
```

### Import Errors or Missing Dependencies
```bash
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

### Slow Evaluation on CPU
- Full test evaluation on 807K samples is slow on CPU
- **Workaround:** Use `demo_inference.py` for a quick smoke test once the processed dataset and model weights are present
- **Alternative:** Enable GPU acceleration if available

### PyTorch Compatibility Issues
```bash
# Verified working versions:
# PyTorch 2.12.0 (CPU)
# Torchvision 0.27.0
# NumPy 1.24.0+
# H5py 3.0.0+

# If errors occur, reinstall:
pip uninstall torch torchvision -y
pip install torch==2.12.0 torchvision==0.27.0 --index-url https://download.pytorch.org/whl/cpu
```

---

## Documentation

- **Project notes:** [Project Capstone](Project%20Capstone) (repository root)

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

---

## Performance Summary

| Metric | Result | Notes |
|--------|--------|--------|
| Metadata no-strike precision | 1.0000 | Retrained probe on the current synthetic-negative feature set |
| Metadata no-strike recall | 1.0000 | Same probe |
| Metadata no-strike F1 | 1.0000 | Same probe |
| Metadata no-strike PR-AUC | 1.0000 | Same probe |
| Clean lat/lon/time no-strike precision | 1.0000 | Much weaker, because the circular strike-derived features are removed |
| Clean lat/lon/time no-strike recall | 0.1087 | Same probe |
| Clean lat/lon/time no-strike F1 | 0.1961 | Same probe |
| Clean lat/lon/time no-strike PR-AUC | 0.2966 | Same probe |
| Satellite ROC-AUC | 0.9199 | Good ranking performance independent of threshold |
| Satellite accuracy at tuned threshold | 87.65% | Threshold 0.55 selected on validation data |
| Satellite precision | 86.01% | At the tuned threshold |
| Satellite recall | 89.93% | At the tuned threshold |
| Satellite F1 | 0.8792 | At the tuned threshold |

Metadata artifact files from the final pass:
- `results/metadata_honest_probe_metrics.json`
- `models/lightning_classifier_metadata_probe.pth`
- `models/lightning_classifier_clean_probe.pth`

---

**Last Updated:** 2026-06-26  
**Status:** The repository now documents the metadata leakage concern and reports the satellite model at its tuned operating point rather than at the default 0.5 threshold.
