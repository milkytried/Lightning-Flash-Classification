# Himawari-8 Satellite Lightning Detection Pipeline - Status Report

## Overview

Complete implementation of 5-phase CNN pipeline to detect lightning from Himawari-8 satellite images. Target: **≥85% recall on test set**.

## Implementation Status

### ✅ PHASE 1: Data Loading (COMPLETE - Commit 2a5d151)

**Files:**
- `src/himawari_png_loader.py` (580+ lines)
- `src/lightning_csv_parser.py` (300+ lines)
- `src/test_phase1.py` (50+ lines)

**Components:**
| Component | Status | Tests |
|-----------|--------|-------|
| PNG Loading | ✅ | ✓ Loads 19 PNGs, shape=(800,950,3) |
| Coordinate Mapping | ✅ | ✓ lat/lon ↔ pixel conversion validated |
| CSV Parsing | ✅ | ✓ Found 2993 CSV files, parsed 4-year dataset |
| Lightning Records | ✅ | ✓ Timestamp, lat/lon, amplitude extracted |

**Key Features:**
- Geographic bounds: [1.0-6.5]°N, [99.5-120.0]°E (Malaysia region)
- Coordinate validation: Boundary checking on all lat/lon pairs
- Flexible CSV parsing: Handles multiple column name variations
- Date-based filtering: Load records by date or time window

**Test Results:** ✅ All tests passing

---

### ✅ PHASE 2 & 3: Patch Extraction & Dataset (COMPLETE - Commit 6f039e2)

**Files:**
- `src/satellite_patch_extractor.py` (400+ lines)
- `src/satellite_dataset_builder.py` (300+ lines)
- `src/himawari_data_loader.py` (250+ lines)
- `src/test_phase2_3.py` (100+ lines)

**Components:**
| Component | Status | Purpose |
|-----------|--------|---------|
| Patch Extractor | ✅ | Extract 64×64 patches at lightning locations + negatives |
| Dataset Builder | ✅ | Index all patches with time-based train/val/test split |
| DataLoader | ✅ | PyTorch Dataset + DataLoader with augmentation |

**Key Features:**

**Patch Extraction:**
- Positive patches: Centered at lightning locations
- Negative patches: Random samples away from lightning (exclusion_radius=100 pixels)
- Ratio: 1 negative per positive (configurable)
- Boundary handling: Validates all patches within PNG bounds
- Output: `data/processed/patches/{split}/{label}/{patch_id}.png`

**Dataset Building:**
- Chronological iteration: Process PNGs in date order
- Time-based split: Prevents temporal leakage
  - Train: Earliest 70% of images
  - Val: Next 15% of images
  - Test: Most recent 15% of images
- Output index: `data/processed/satellite_dataset.csv`
  - Columns: path, label, x, y, lat, lon, split
  - Class distribution tracked per split

**Data Loading:**
- Custom PyTorch Dataset class
- Augmentation pipeline (train set only):
  - HorizontalFlip: 50% probability
  - VerticalFlip: 50% probability
  - Rotate: ±15° random angle
  - GaussNoise: μ=0, σ=0.05
  - Normalize: ImageNet mean/std
- Output: Batches of (images, labels) tensors
  - images: (batch_size, 3, 64, 64)
  - labels: (batch_size,) ∈ {0, 1}

**Test Results:** ✅ Ready for validation

---

### ✅ PHASE 4 & 5: Training & Evaluation (COMPLETE - Commit fd764a4)

**Files:**
- `src/train_satellite.py` (500+ lines)
- `src/evaluate_satellite.py` (450+ lines)

**Components:**
| Component | Status | Purpose |
|-----------|--------|---------|
| Training Script | ✅ | Epoch-based ResNet-50 training |
| Evaluation Script | ✅ | Metrics + visualizations |

**Training (`train_satellite.py`):**
- Architecture: ResNet-50 (pretrained ImageNet, fine-tuned)
  - Input: 3×64×64 patches
  - Output: Binary classification (0=no lightning, 1=lightning)
- Loss function: Focal Loss (α=0.25, γ=2.0) for class imbalance
- Optimizer: Adam (lr=0.001)
- Learning rate scheduling: ReduceLROnPlateau (factor=0.5, patience=5)
- Early stopping: patience=10 epochs (best model checkpointing)
- Gradient clipping: max_norm=1.0 (training stability)
- Training history saved: `models/satellite_training_history.json`

**Training Configuration:**
```python
train_satellite.py \
  --dataset data/processed/satellite_dataset.csv \
  --epochs 50 \
  --batch-size 32
```

**Evaluation (`evaluate_satellite.py`):**
- Metrics computed on test set:
  * Accuracy, Precision, Recall, F1-Score
  * ROC-AUC
  * POD (Probability of Detection) = Recall
  * FAR (False Alarm Rate)
  * Confusion matrix (TP, FP, TN, FN)

**Visualizations:**
| Plot | Purpose | Output |
|------|---------|--------|
| Confusion Matrix | Classification results | `results/confusion_matrix.png` |
| ROC Curve | Trade-off analysis | `results/roc_curve.png` |
| Metrics Summary | Performance overview | `results/metrics_summary.png` |
| Metrics JSON | Programmatic access | `results/satellite_metrics.json` |

**Evaluation Configuration:**
```python
evaluate_satellite.py \
  --dataset data/processed/satellite_dataset.csv \
  --model models/satellite_resnet50.pth \
  --batch-size 32
```

**Success Criteria:**
- ✅ Recall ≥ 0.85 (primary metric for lightning detection)
- Status indicator: Visualizations highlight if target achieved

---

## End-to-End Pipeline Execution

### Step 1: Build Dataset (ONE-TIME)
```bash
python src/satellite_dataset_builder.py
```
**Output:**
- `data/processed/patches/` - 64×64 patch images
- `data/processed/satellite_dataset.csv` - Index with 3 splits

**Expected Behavior:**
- Iterates all 19 Himawari-8 PNGs
- Extracts patches from lightning-containing images
- Generates negative samples
- Creates train/val/test index (70/15/15 time-based split)
- Total patches: ~10,000-50,000 (depends on lightning frequency)

### Step 2: Train Model
```bash
python src/train_satellite.py --epochs 50 --batch-size 32
```
**Output:**
- `models/satellite_resnet50.pth` - Best model checkpoint
- `models/satellite_training_history.json` - Training curve data

**Expected Behavior:**
- Trains for max 50 epochs
- Early stops when validation loss plateaus
- Logs metrics per epoch
- Saves best model when validation loss improves

### Step 3: Evaluate Model
```bash
python src/evaluate_satellite.py --model models/satellite_resnet50.pth
```
**Outputs:**
- `results/confusion_matrix.png`
- `results/roc_curve.png`
- `results/metrics_summary.png`
- `results/satellite_metrics.json`

**Expected Behavior:**
- Loads test set from dataset CSV
- Computes metrics on test set
- Generates 3 visualizations
- Prints detailed metrics report with recall status

---

## File Structure

```
Project Capstone/
├── src/
│   ├── himawari_png_loader.py          # Phase 1: Load PNGs
│   ├── lightning_csv_parser.py         # Phase 1: Parse lightning CSVs
│   ├── test_phase1.py                  # Phase 1: Validation tests
│   ├── satellite_patch_extractor.py    # Phase 2: Extract patches
│   ├── satellite_dataset_builder.py    # Phase 3: Build indexed dataset
│   ├── himawari_data_loader.py         # Phase 3: PyTorch DataLoader
│   ├── test_phase2_3.py                # Phase 2-3: Integration tests
│   ├── train_satellite.py              # Phase 4: Training loop
│   ├── evaluate_satellite.py           # Phase 5: Evaluation
│   ├── model_arch.py                   # ResNet-50 architecture (existing)
│   └── ...                             # Other files
├── data/
│   ├── raw/
│   │   ├── himawari8_pngs/             # 19 Himawari-8 PNG images
│   │   └── csv_lightning_data/         # 2993 Malaysian Met Dept CSVs
│   └── processed/
│       ├── patches/                    # 64×64 patch images (generated)
│       └── satellite_dataset.csv       # Index with splits (generated)
├── models/
│   ├── satellite_resnet50.pth          # Trained model (generated)
│   └── satellite_training_history.json # Training curve (generated)
├── results/
│   ├── confusion_matrix.png            # Evaluation visualization
│   ├── roc_curve.png                   # Evaluation visualization
│   ├── metrics_summary.png             # Evaluation visualization
│   └── satellite_metrics.json          # Metrics data
└── ...
```

---

## Key Technical Details

### Coordinate System
- PNG bounds: [1.0°, 6.5°]N × [99.5°, 120.0°]E
- Resolution: 950×800 pixels
- Mapping: Linear interpolation between lat/lon and pixel coordinates
- Validation: All coordinates checked within bounds before patch extraction

### Class Imbalance Handling
- **Problem:** Lightning occurs ~0.1-1% of time → extreme imbalance
- **Solution 1:** Focal Loss (α=0.25, γ=2.0)
  - Reduces weight of easy negatives
  - Focuses on hard negatives and false negatives
- **Solution 2:** Balanced sampling
  - 1 negative patch per positive patch
  - Controlled negative-to-positive ratio
- **Solution 3:** Recall-focused evaluation
  - Recall ≥ 0.85 ensures detection sensitivity

### Temporal Leakage Prevention
- **Risk:** Random train/val/test split → future dates in training
- **Solution:** Chronological split
  - Sort PNGs by date
  - Train on earliest 70% of dates
  - Validate on middle 15% of dates
  - Test on most recent 15% of dates
- **Benefit:** Realistic evaluation on future data

### Transfer Learning
- **Architecture:** ResNet-50 (pretrained on ImageNet)
- **Input:** 3×64×64 RGB patches from satellite images
- **Fine-tuning:** All layers trainable
- **Dropout:** 0.5 (regularization)
- **Initialization:** ImageNet pretrained weights

---

## Metrics Definitions

| Metric | Formula | Interpretation |
|--------|---------|-----------------|
| **Recall** | TP/(TP+FN) | % of lightning events detected |
| **Precision** | TP/(TP+FP) | % of detections that are true |
| **F1** | 2·Precision·Recall/(Precision+Recall) | Harmonic mean |
| **POD** | TP/(TP+FN) | Probability of Detection (= Recall) |
| **FAR** | FP/(TP+FP) | False Alarm Ratio |
| **ROC-AUC** | Area under curve | Classification threshold trade-off |
| **Accuracy** | (TP+TN)/Total | Overall correctness |

**For Lightning Detection:** Recall is primary metric (minimize missed lightning)

---

## Next Steps

### Immediate (Ready to Execute)
1. Build dataset from all 19 PNGs
2. Train ResNet-50 model
3. Evaluate and validate ≥85% recall
4. Generate visualizations

### Optional Enhancements
- Hyperparameter tuning (learning rate, batch size, epochs)
- Data augmentation improvements
- Ensemble methods (multiple models)
- Threshold optimization for operating point
- Spatial analysis (geographic distribution of errors)

---

## Success Criteria

| Criterion | Target | Status |
|-----------|--------|--------|
| Pipeline complete | All 5 phases | ✅ Complete |
| Test recall | ≥ 0.85 | ⏳ Awaiting execution |
| Model saved | `models/satellite_resnet50.pth` | ⏳ Generated after training |
| Visualizations | 3 plots + metrics JSON | ⏳ Generated after evaluation |
| No data leaks | Time-based split | ✅ Implemented |
| Clean code | Type hints, docstrings | ✅ Implemented |

---

## Development Notes

### Git Commits
- **2bb2331**: Audit + planning documents
- **2a5d151**: Phase 1 (PNG loader + CSV parser) ✅
- **6f039e2**: Phase 2 & 3 (Patches + Dataset + DataLoader) ✅
- **fd764a4**: Phase 4 & 5 (Training + Evaluation) ✅

### Dependencies
- PyTorch 2.12.0 (CPU)
- Torchvision 0.27.0 (ResNet-50)
- Albumentations 1.3.1 (Augmentation)
- scikit-learn (Metrics)
- pandas, numpy, PIL (Data)
- matplotlib, seaborn (Visualization)

### Known Limitations
- CPU-only environment (no CUDA)
- Limited data: ~4 years of satellite images (19 PNGs total)
- Unbalanced classes: Lightning rare event (~1% frequency)
- Small patch size: 64×64 pixels (fine-grained features)

---

## References

- **Capstone Proposal:** Himawari-8 satellite CNN for lightning detection
- **Data Sources:**
  - Himawari-8 PNGs: `data/raw/himawari8_pngs/` (19 files)
  - Lightning records: Malaysian Met Department CSVs (2993 files, 4 years)
- **Architecture:** Transfer learning with pretrained ResNet-50
- **Evaluation:** Metrics focused on recall for safety-critical application

---

*Last Updated: 2024*
*Pipeline Status: Complete and Ready for Execution*
