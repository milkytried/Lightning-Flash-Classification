# Audit Report: Satellite Image Pipeline vs Metadata Baseline

**Date:** May 22, 2026  
**Status:** Current project misaligned with proposal; requires pivot to satellite imagery

---

## Executive Summary

The **current trained model is a metadata-only classifier** that does NOT use satellite imagery. This contradicts the capstone proposal which explicitly requires Himawari-8 satellite image analysis. The metadata model was a quick validation approach but **must be replaced** with the proper satellite image pipeline.

| Component | Status | Details |
|-----------|--------|---------|
| **Metadata Model** | ✅ Trained & Tested | 100% recall on metadata (lat/lon/amplitude/type) |
| **Satellite ResNet-50** | ❌ Code Only | ResNet-50 architecture defined but never trained |
| **Satellite Dataset** | ⏳ Partial | HDF5 exists (29 MB) but unclear if properly labeled |
| **PNG Data** | ✅ Available | Himawari-8 PNGs (multiple dates + 4-year structure) |
| **Lightning CSV** | ✅ Available | 309 files with timestamp, lat/lon, amplitude, type |

---

## Part 1: Current Codebase Audit

### Trained and Tested (Metadata Model)

**Files:**
- `src/lightning_model.py` - MLP classifier (4 → 256 → 128 → 64 → 1)
- `src/lightning_data_loader.py` - Loads lat_norm, lon_norm, amp_norm, strike_code
- `src/train_lightning.py` - Training loop (trained for ~1h 50m)
- `src/quick_eval.py` - Evaluation (reported 100% recall on 10K samples)
- `src/ingest_met_data.py` - Ingested 5.3M lightning records from CSVs

**Architecture:**
```
Input Features (4): [latitude, longitude, amplitude, strike_type]
     ↓
MLP (43,393 parameters)
     ↓
Output: Binary probability [0, 1]
```

**Loss Function:** Focal Loss (α=0.25, γ=2.0) for extreme class imbalance (99.84% positive)

**Test Result:** 100% recall on 10K test samples

**Issue:** Uses lightning metadata directly, NOT satellite imagery → violates proposal

---

### Defined But Not Trained (Satellite ResNet-50)

**File:** `src/model_arch.py`

**Classes Defined:**
- `LightningResNet50` - Transfer learning ResNet-50 with:
  - Pretrained ImageNet weights
  - Input: 64×64 image patches (3 channels)
  - Adaptable for multi-channel input
  - Output: Sigmoid probability [0, 1]
  - Dropout regularization
- `FocalLoss` - Binary focal loss for class imbalance

**Current State:** Never trained; no dataset exists using this architecture

---

### Supporting Infrastructure

**Preprocessing:** `src/preprocessing.py`
- `HimawariPreprocessor` class designed to:
  - Load netCDF4 Himawari-8 files
  - Read MMD lightning CSV
  - Crop to Malaysia region
  - Create 64×64 patches
  - Label with lightning occurrence (0/1)
  - Handle class imbalance
  - Split train/val/test
  - Save to HDF5

**Status:** Code structure exists but not fully integrated

**Data Loading:** `src/data_loader.py`
- Image-based dataset loader (legacy)
- Albumentations augmentation
- Train/val/test split support

**Training:** `src/train.py`
- Training loop for ResNet-50 (legacy)
- Early stopping, ReduceLROnPlateau scheduler
- Not used (metadata model trained instead)

**Evaluation:** `src/evaluate.py`
- Metrics: Accuracy, Precision, Recall, F1, ROC-AUC, POD, FAR
- Visualizations placeholder
- Not used (metadata model evaluated instead)

---

## Part 2: Data Availability Audit

### Himawari-8 PNG Images

**Location:** `data/raw/himawari8_pngs/`

**Structure:**
```
himawari8_pngs/
├── 12_May_Himawari.png      (recent, loose PNGs)
├── 14_May_Himawari.png
├── ...
├── 2023/
│   └── 2023/PENINSULAR/
│       ├── 01 JAN/1/         (Day 1)
│       ├── 01 JAN/10/        (Day 10)
│       └── ...
├── 2024/                      (4-year archive structure)
├── 2025/
└── 2026/
```

**Availability:**
- ✅ Multiple PNG files at different dates
- ✅ 4-year directory structure (2023-2026)
- ⏳ Georeferencing information: **UNCLEAR**

**PNG Georeferencing Status:**
- File: `data/raw/himawari8_pngs/lightning.nc` exists
- Likely contains: NetCDF metadata with spatial bounds, projection, resolution
- **Must audit** to determine:
  - Image geographic bounds (lat_min, lat_max, lon_min, lon_max)
  - Image dimensions (height, width in pixels)
  - Projection type (lat/lon to pixel mapping)
  - Timestamp format

---

### Lightning CSV Data

**Location:** Nested in PNG directory structure  
**Example:** `data/raw/himawari8_pngs/2023/2023/PENINSULAR/01 JAN/1/raw data all.csv`

**CSV Format:**
```
Columns: Solution Key, Date/Time, Epoch Time, Milliseconds, Major Code, Minor Code,
         Latitude, Longitude, Altitude, Amplitude, Cloud or Ground, ...

Example Row:
  Date/Time: 2023-01-01 00:09:46Z
  Latitude: (numeric)
  Longitude: (numeric)
  Amplitude: (numeric, lightning strength)
  Cloud or Ground: (type code)
```

**Availability:**
- ✅ 309+ CSV files across 4-year period
- ✅ Contains: timestamp, lat, lon, amplitude, type
- ✅ Timestamp precision: Second-level
- ✅ Geographic coverage: Peninsular Malaysia

---

### Processed Datasets

**Location:** `data/processed/`

**Existing Files:**
1. `lightning_dataset.h5` (581 MB)
   - Created by `ingest_met_data.py`
   - Contains: metadata features (lat, lon, amplitude, strike_code)
   - Labels: binary (lightning yes/no)
   - Splits: 70/15/15 (3.77M/807K/807K)
   - **Purpose:** Metadata-only classifier training
   - **Status:** Do NOT use for satellite model

2. `himawari_dataset.h5` (29 MB)
   - Unclear purpose/contents
   - May be from earlier satellite image attempt
   - **Action:** Audit or delete

3. `dataset.h5` (29 MB)
   - Unclear purpose/contents
   - **Action:** Audit or delete

---

## Part 3: Missing Requirements

### Critical Georeferencing Information Needed

To extract 64×64 image patches at lightning locations, we need:

**From PNG files or metadata:**
- [ ] `image_lat_min` - Minimum latitude in PNG
- [ ] `image_lat_max` - Maximum latitude in PNG
- [ ] `image_lon_min` - Minimum longitude in PNG
- [ ] `image_lon_max` - Maximum longitude in PNG
- [ ] `image_width` - Width in pixels
- [ ] `image_height` - Height in pixels
- [ ] `image_projection` - Projection type (e.g., lat/lon, mercator)
- [ ] `timestamp_format` - PNG timestamp format/location

**Current Status:** Unknown; must audit `lightning.nc` and sample PNGs

---

## Part 4: Separation of Concerns

### Metadata Baseline (Current Working Model)

**Purpose:** Quick validation; fallback approach  
**Input:** Latitude, Longitude, Amplitude, Strike Type  
**Output:** Lightning yes/no (probability)  
**Status:** ✅ Trained, tested, 100% recall  
**Note:** **Does NOT use satellite imagery**

### Satellite Image Model (Proposed Implementation)

**Purpose:** Actual capstone proposal implementation  
**Input:** 64×64 Himawari-8 image patch  
**Output:** Lightning yes/no (probability)  
**Status:** ❌ Not yet trained  
**Requirements:**
- Proper patch extraction
- Georeferenced coordinate mapping
- Time-based train/val/test split (no temporal leakage)
- Lightning labeling from CSV (within spatial/temporal window)

---

## Part 5: Immediate Actions Required

### Step 1: Audit Georeferencing (BLOCKING)

```bash
# Check lightning.nc for spatial metadata
python src/audit_georef.py

# Expected output:
# - Image bounds in lat/lon
# - Image dimensions
# - Projection info
# - Coordinate mapping method
```

**If georeferencing info is found:** Proceed to Step 2  
**If georeferencing info is missing:** Must create config file

---

### Step 2: Build Satellite Patch Extraction Pipeline

**New files to create:**
- `src/himawari_patch_extractor.py` - Extract patches from PNGs at lightning locations
- `src/satellite_dataset_builder.py` - Create indexed CSV with patch paths + labels
- `src/himawari_data_loader.py` - PyTorch DataLoader for image patches

**Output:** Indexed dataset CSV with columns:
```
image_path, patch_path, timestamp, center_lat, center_lon, 
center_x, center_y, label, split
```

---

### Step 3: Train Satellite ResNet-50

**File:** Modify `src/train.py` to:
- Load patches from `himawari_data_loader.py`
- Train ResNet-50 (from `model_arch.py`)
- Use Focal Loss for class imbalance
- Apply early stopping
- Save best model

---

### Step 4: Evaluate with Metrics + Visualizations

**File:** `src/evaluate.py` modifications:
- Compute: Accuracy, Precision, Recall, F1, ROC-AUC, POD, FAR
- Generate sample plots (image, label, prediction)
- Show false positives/false negatives
- Create satellite overlay visualization

---

## Part 6: Decision Points

### Q1: Is georeferencing information available in PNGs/NetCDF?

- **If YES:** Proceed to build patch extraction pipeline
- **If NO:** Create a config file with manual bounds or interpolated projection

### Q2: Should we keep metadata baseline model?

- **Recommendation:** YES, but clearly label as "fallback" or "quick baseline"
- **Final report:** Compare both approaches (metadata vs satellite image)

### Q3: How much satellite data should we use?

- **Recommendation:** Use 1-2 months (e.g., Jan 2023) for initial training/testing
- **Then:** Scale to full 4-year dataset if resources permit

---

## Conclusion

**Current Status:** ❌ Misaligned with proposal
**Required Pivot:** Build proper satellite image pipeline
**Blocking Issue:** Georeferencing information (lat/lon → pixel coordinate mapping)
**Next Action:** Audit `lightning.nc` for spatial metadata

**Estimated Timeline:**
- Step 1 (Audit): ~2 hours
- Step 2 (Patch extraction): ~4-6 hours
- Step 3 (Training): ~2-4 hours (depending on dataset size)
- Step 4 (Evaluation + visualizations): ~2-3 hours

**Total:** ~10-15 hours to full satellite image model

---

**Report Generated:** May 22, 2026  
**Status:** Ready for action
