# Development Plan: Satellite Image Lightning Classifier

**Date:** May 22, 2026  
**Status:** Ready to implement  
**Blocking Issue:** ✅ RESOLVED (georeferencing available)

---

## Executive Summary

The project has **all data needed** to build a proper satellite image classifier:
- ✅ Himawari-8 PNG images (950×800 pixels, multiple dates, 4-year archive)
- ✅ Lightning CSV records (timestamp, lat/lon, amplitude, type)
- ✅ Georeferencing (NetCDF with complete lat/lon coordinate system)

**Estimated Timeline:** 10-15 hours  
**Deliverable:** CNN classifier achieving ≥85% recall on satellite image patches

---

## Part 1: Georeferencing System

### Coordinate Mapping

**NetCDF Grid:**
- Full globe coverage: lat[-90, +90], lon[-180, +180]
- Resolution: 0.05° per pixel (approximately 5.6 km at equator)
- Grid dimensions: 7200 × 3600 (lon × lat)

**Malaysia Region:**
- Lat: [1.02°N, 6.53°N] (indices: 1779→1669, decreasing because lat array is reversed)
- Lon: [99.97°E, 119.97°E] (indices: 5599→5999)

**PNG Characteristics:**
- Dimensions: 950×800 (width × height)
- Format: RGB, no embedded georeferencing
- **Likely:** Cropped/downsampled Malaysia region from NetCDF
- **Assumption:** PNG covers similar Malaysia region as NetCDF

**Lat/Lon → Pixel Mapping Formula:**

```python
# Assuming PNG crop bounds (to be empirically determined or assumed):
png_lat_min, png_lat_max = 1.0, 6.5  # degrees N
png_lon_min, png_lon_max = 99.5, 120.0  # degrees E

png_height, png_width = 800, 950  # pixels

# Convert lat/lon to pixel coordinates
def latlon_to_pixel(lat, lon):
    # Normalize to [0, 1]
    x_norm = (lon - png_lon_min) / (png_lon_max - png_lon_min)
    y_norm = (png_lat_max - lat) / (png_lat_max - png_lat_min)  # inverted Y
    
    # Convert to pixel coordinates
    x_pixel = int(x_norm * png_width)
    y_pixel = int(y_norm * png_height)
    
    return x_pixel, y_pixel
```

---

## Part 2: Phase 1 - Satellite Patch Extraction Pipeline

### 2.1 Himawari PNG Loader

**File:** `src/himawari_png_loader.py` (NEW)

**Purpose:** Load PNG images with timestamp and geographic bounds

**Key Functions:**

```python
class HimawariPNGLoader:
    """Load Himawari-8 PNG files with geographic metadata."""
    
    def __init__(self, png_dir, config):
        """
        Args:
            png_dir: Path to PNG directory (e.g., 'data/raw/himawari8_pngs/')
            config: Dict with lat/lon bounds of PNG crop
                   {
                       'png_lat_min': 1.0,
                       'png_lat_max': 6.5,
                       'png_lon_min': 99.5,
                       'png_lon_max': 120.0,
                       'png_width': 950,
                       'png_height': 800
                   }
        """
    
    def find_png_files(self):
        """Recursively find all PNG files, return list with (path, datetime)."""
    
    def load_png(self, filepath):
        """Load PNG as numpy array (800, 950, 3), return RGB image."""
    
    def latlon_to_pixel(self, lat, lon):
        """Convert lat/lon to (x, y) pixel coordinates."""

```

**Output:** Library function for loading PNGs and coordinate conversion

---

### 2.2 Lightning CSV Parser

**File:** `src/lightning_csv_parser.py` (NEW)

**Purpose:** Parse Met Department CSV files and extract lightning records

**Key Functions:**

```python
class LightningCSVParser:
    """Parse MMD lightning CSV files."""
    
    def __init__(self, csv_dir='data/raw/himawari8_pngs'):
        """
        Args:
            csv_dir: Root directory containing nested CSV files
        """
    
    def find_csvs(self):
        """Find all CSV files recursively."""
        # Returns list of paths
    
    def parse_csv(self, filepath):
        """
        Parse single CSV file.
        
        Returns:
            pd.DataFrame with columns:
            - timestamp (datetime)
            - latitude (float)
            - longitude (float)
            - amplitude (float)
            - strike_type (str, e.g., 'CG' or 'CC')
        """
    
    def load_all_lightning(self, start_date, end_date):
        """
        Load all lightning records in date range.
        
        Returns:
            pd.DataFrame, sorted by timestamp
        """

```

**Output:** Lightning DataFrame with proper timestamp/lat/lon parsing

---

### 2.3 Satellite Patch Extractor

**File:** `src/satellite_patch_extractor.py` (NEW)

**Purpose:** Extract 64×64 patches from PNGs at lightning locations and negative samples

**Key Functions:**

```python
class SatellitePatchExtractor:
    """Extract image patches from Himawari-8 PNGs with lightning labels."""
    
    def __init__(self, png_loader, output_dir='data/processed/patches'):
        """Initialize with PNG loader and output directory."""
    
    def extract_positive_patch(self, png_array, center_lat, center_lon, 
                               patch_size=64):
        """
        Extract 64×64 patch centered at lightning location.
        
        Args:
            png_array: (H, W, 3) numpy array
            center_lat, center_lon: Lightning coordinates
            patch_size: 64
        
        Returns:
            patch: (64, 64, 3) numpy array
            x, y: pixel coordinates of center
        """
        # Convert lat/lon to pixel
        x, y = png_loader.latlon_to_pixel(center_lat, center_lon)
        
        # Extract patch with bounds checking
        patch = png_array[
            max(0, y-32):min(H, y+32),
            max(0, x-32):min(W, x+32),
            :
        ]
        return patch, x, y
    
    def extract_negative_patches(self, png_array, png_datetime, 
                                 lightning_records, n_samples=10):
        """
        Extract random patches from areas with NO lightning.
        
        Args:
            png_array: (H, W, 3) numpy array
            png_datetime: datetime of PNG
            lightning_records: DataFrame with lightning events near this time
            n_samples: Number of negative patches to extract
        
        Returns:
            List of (patch, x, y) tuples
        """
        # Mask out regions near lightning
        # Sample random patch centers away from lightning
        # Return negative patches
    
    def save_patch(self, patch, output_path):
        """Save patch as PNG or HDF5."""

```

**Output:** Extracted 64×64 patches saved to disk with metadata

---

### 2.4 Dataset Index Builder

**File:** `src/satellite_dataset_builder.py` (NEW)

**Purpose:** Create indexed CSV mapping patches to labels and metadata

**Process:**

1. **Iterate over PNG files** in chronological order
2. **For each PNG:**
   - Find lightning events within ±60 minute window (configurable)
   - Extract positive patches (at lightning locations)
   - Extract negative patches (random areas without lightning)
   - Save all patches to disk
   - Log metadata to index

3. **Create index CSV** with columns:
```
patch_id, image_path, patch_path, timestamp, 
center_lat, center_lon, center_x, center_y, 
label, split, lead_time_minutes
```

4. **Split into train/val/test by time:**
   - Train: 70% of images (early dates)
   - Val: 15% of images (middle dates)
   - Test: 15% of images (recent dates)
   - **Avoid random splitting** to prevent temporal leakage

**Output:** `data/processed/satellite_dataset.csv` + patches on disk

---

## Part 3: Phase 2 - Dataset Loading

**File:** `src/himawari_data_loader.py` (NEW)

**Purpose:** PyTorch DataLoader for image patches

**Implementation:**

```python
class HimawariPatchDataset(torch.utils.data.Dataset):
    """
    Load 64×64 satellite patches from disk.
    
    Inputs: Image patch (3, 64, 64)
    Labels: Binary (0/1) - lightning within lead time window
    """
    
    def __init__(self, dataset_csv, split='train', augment=True):
        """
        Args:
            dataset_csv: Path to satellite_dataset.csv
            split: 'train', 'val', or 'test'
            augment: Apply augmentations to train set
        """
        # Load CSV, filter by split
        # Store list of (patch_path, label) tuples
    
    def __len__(self):
        return len(self.patches)
    
    def __getitem__(self, idx):
        patch_path, label = self.patches[idx]
        patch = load_image(patch_path)  # (64, 64, 3) → (3, 64, 64)
        
        if self.augment and self.split == 'train':
            # Random flip, rotation, noise
            patch = apply_augmentations(patch)
        
        return torch.from_numpy(patch).float(), torch.tensor(label, dtype=torch.long)


def create_himawari_loaders(dataset_csv, batch_size=32):
    """Return dict with 'train', 'val', 'test' DataLoaders."""
    ...
```

**Output:** Ready-to-use PyTorch DataLoaders

---

## Part 4: Phase 3 - Model Training

### 4.1 Training Script

**File:** Modify `src/train.py` (EXISTING) or create `src/train_satellite.py` (NEW)

**Changes Needed:**

```python
def train_satellite_model():
    """Train ResNet-50 on satellite image patches."""
    
    # Load data
    loaders = create_himawari_loaders('data/processed/satellite_dataset.csv', 
                                       batch_size=32)
    train_loader = loaders['train']
    val_loader = loaders['val']
    
    # Initialize model
    model = LightningResNet50(num_input_channels=3, num_classes=1)
    criterion = FocalLoss(alpha=0.25, gamma=2.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=5, verbose=True)
    
    # Training loop
    best_val_loss = float('inf')
    patience_count = 0
    max_patience = 10
    
    for epoch in range(50):
        # Train epoch
        train_loss = train_epoch(model, train_loader, optimizer, criterion)
        
        # Validate epoch
        val_loss, val_metrics = val_epoch(model, val_loader, criterion)
        
        # Log metrics
        print(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
              f"val_recall={val_metrics['recall']:.4f}")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_count = 0
            # Save best model
            torch.save(model.state_dict(), 'models/best_satellite_resnet50.pth')
        else:
            patience_count += 1
            if patience_count >= max_patience:
                print("Early stopping")
                break
        
        scheduler.step(val_loss)
    
    print(f"Training complete. Best model saved to models/best_satellite_resnet50.pth")
```

**Hyperparameters:**
- Batch size: 32
- Learning rate: 0.001 (Adam)
- Loss: Focal Loss (α=0.25, γ=2.0)
- Optimizer: Adam
- Early stopping: patience=10
- Max epochs: 50

**Output:** `models/best_satellite_resnet50.pth`

---

## Part 5: Phase 4 - Evaluation & Visualization

### 5.1 Metrics Computation

**File:** Modify `src/evaluate.py` (EXISTING)

**Metrics to Report:**

```python
def evaluate_satellite_model(model_path, dataset_csv):
    """Evaluate on test set."""
    
    model = LightningResNet50()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    test_loader = create_himawari_loaders(dataset_csv, split='test')
    
    # Collect predictions
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for patches, labels in test_loader:
            probs = model(patches).cpu().numpy()
            preds = (probs > 0.5).astype(int)
            
            all_probs.append(probs)
            all_preds.append(preds)
            all_labels.append(labels.numpy())
    
    all_probs = np.concatenate(all_probs)
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    
    # Compute metrics
    accuracy = sklearn.metrics.accuracy_score(all_labels, all_preds)
    precision = sklearn.metrics.precision_score(all_labels, all_preds)
    recall = sklearn.metrics.recall_score(all_labels, all_preds)
    f1 = sklearn.metrics.f1_score(all_labels, all_preds)
    roc_auc = sklearn.metrics.roc_auc_score(all_labels, all_probs)
    
    # Meteorological metrics
    pod = recall  # Probability of Detection
    far = 1 - precision  # False Alarm Ratio
    
    # Report
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f} {'✅ PASS' if recall >= 0.85 else '❌ FAIL'} (target ≥0.85)")
    print(f"F1-Score:  {f1:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}")
    print(f"POD:       {pod:.4f}")
    print(f"FAR:       {far:.4f}")
    
    # Confusion matrix
    cm = sklearn.metrics.confusion_matrix(all_labels, all_preds)
    plot_confusion_matrix(cm)
    
    # ROC curve
    fpr, tpr, _ = sklearn.metrics.roc_curve(all_labels, all_probs)
    plot_roc_curve(fpr, tpr, roc_auc)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'pod': pod,
        'far': far,
        'confusion_matrix': cm
    }
```

**Output:** Metrics report + visualizations

---

### 5.2 Visual Validation

**File:** `src/visualize_predictions.py` (NEW)

**Visualizations:**

1. **Sample Predictions:**
   - Image patch (64×64)
   - True label
   - Predicted probability
   - Grid of 3×3 examples

2. **False Positives:**
   - Patches predicted as lightning but no actual lightning
   - Analyze what features model learns incorrectly

3. **False Negatives:**
   - Patches with lightning but model missed
   - Identify difficult cases

4. **Satellite Overlay (if time permits):**
   - Full PNG image
   - Red markers for detected lightning
   - Green markers for predicted high-risk areas

**Output:** PNG plots saved to `results/`

---

## Part 6: Implementation Timeline

### Week 1:

**Day 1-2 (4 hrs):** Georeferencing & Coordinate Mapping
- Finalize PNG crop bounds
- Implement `latlon_to_pixel()` conversion
- Unit test with sample lightning records

**Day 3-4 (4 hrs):** Patch Extraction Pipeline
- Implement PNG loader
- Implement Lightning CSV parser
- Implement patch extractor (positive + negative)
- Test on 1 day of data

**Day 5 (3 hrs):** Dataset Index Creation
- Build complete index CSV
- Split into train/val/test by time
- Verify no temporal leakage

### Week 2:

**Day 6-7 (4 hrs):** PyTorch DataLoader
- Implement `HimawariPatchDataset`
- Implement data augmentation
- Test loader on full dataset

**Day 8-9 (4 hrs):** Model Training
- Run `train_satellite.py`
- Monitor loss curves
- Early stopping validation

**Day 10 (3 hrs):** Evaluation & Visualization
- Run test evaluation
- Generate all plots
- Write results report

---

## Part 7: File Structure After Completion

```
src/
├── himawari_png_loader.py         (NEW)
├── lightning_csv_parser.py         (NEW)
├── satellite_patch_extractor.py    (NEW)
├── satellite_dataset_builder.py    (NEW)
├── himawari_data_loader.py         (NEW)
├── train_satellite.py              (NEW or modify train.py)
├── visualize_predictions.py        (NEW)
├── model_arch.py                   (EXISTING - ResNet-50 already defined)
├── evaluate.py                     (MODIFY for satellite eval)
└── ...

data/
├── raw/
│   └── himawari8_pngs/
│       ├── *.png                   (PNGs)
│       ├── lightning.nc            (NetCDF georeferencing)
│       └── 2023-2026/              (Nested CSV files)
└── processed/
    ├── lightning_dataset.h5        (OLD - metadata model)
    ├── satellite_dataset.csv       (NEW - patch index)
    └── patches/                    (NEW - extracted 64×64 patches)
        ├── train/
        ├── val/
        └── test/

models/
├── best_satellite_resnet50.pth     (NEW - trained weights)
└── lightning_classifier.pth        (OLD - metadata model)

results/
├── confusion_matrix.png             (NEW)
├── roc_curve.png                    (NEW)
├── sample_predictions.png           (NEW)
├── false_positives.png              (NEW)
└── false_negatives.png              (NEW)
```

---

## Part 8: Success Criteria

- ✅ Extract 64×64 patches from Himawari-8 PNGs
- ✅ Label patches using lightning CSV (within ±60 min window)
- ✅ Train ResNet-50 classifier
- ✅ Achieve ≥85% recall on test set
- ✅ Generate evaluation metrics (accuracy, precision, recall, F1, ROC-AUC, POD, FAR)
- ✅ Create visualizations (confusion matrix, ROC curve, sample predictions)
- ✅ Clearly separate satellite model from metadata baseline
- ✅ Commit code to GitHub with no data leaks

---

## Part 9: Known Risks & Mitigations

| Risk | Mitigation |
|------|-----------|
| PNG bounds unknown | Assume consistent with NetCDF region; validate visually |
| Temporal leakage | Use time-based split, not random |
| Class imbalance | Use Focal Loss; report metrics separately |
| Patches out-of-bounds | Pad or skip; careful boundary handling |
| Training slow | Start with 1 month data; scale if time permits |
| Recall < 85% | May require architecture tuning (deeper network, more data) |

---

## Next Action

**→ Start Phase 1: PNG Loader + CSV Parser (Day 1-2)**

Ready to proceed? Run:
```bash
git checkout -b feature/satellite-image-pipeline
```

Then implement `src/himawari_png_loader.py` and `src/lightning_csv_parser.py`.

---

**Created:** May 22, 2026  
**Status:** ✅ Ready to implement
