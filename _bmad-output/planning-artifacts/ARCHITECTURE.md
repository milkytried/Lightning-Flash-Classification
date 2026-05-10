# Technical Architecture Document
## Lightning Flash Classification Using Himawari-8 Satellite Imagery

**Project Name:** Deep Learning for Lightning Flash Classification Using Geostationary Satellite Imagery  
**Architecture Version:** 1.0  
**Created:** 2026-05-10  
**Architect:** Winston (System Architect)  
**Hardware Target:** NVIDIA RTX 3050 (8 GB VRAM)  
**Optimization Priority:** Memory efficiency + reproducibility  

---

## 1. System Overview

### 1.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     LIGHTNING NOWCASTING SYSTEM                 │
└─────────────────────────────────────────────────────────────────┘

┌──────────────────┐         ┌──────────────────┐
│  Himawari-8 AHI  │         │  MMD Lightning   │
│  netCDF4 Archive │         │  Detection CSV   │
└────────┬─────────┘         └────────┬─────────┘
         │                            │
         │  (Raw ~700 GB)             │  (Raw ~1-10 MB)
         │                            │
         v                            v
    ┌────────────────────────────────────────┐
    │     OFFLINE PREPROCESSING PIPELINE     │
    │  (Runs once; stores result in HDF5)    │
    │                                        │
    │  • Crop to Malaysia (20°×20° box)     │
    │  • Reproject lightning → pixel coords  │
    │  • Cloud masking (T > 290 K)          │
    │  • Normalize [0, 1]                   │
    │  • Stack multi-channel tensors        │
    │  • Generate labels (0-60 min window)  │
    │  • Downsample negatives / balance     │
    └────────────┬───────────────────────────┘
                 │
                 │  (Processed ~150 GB HDF5)
                 v
         ┌──────────────────┐
         │   HDF5 Dataset   │
         │   (Compressed)   │
         └────────┬─────────┘
                  │
        ┌─────────┴──────────┐
        │                    │
        v                    v
    ┌──────────────┐    ┌──────────────┐
    │ Train Split  │    │ Val Split    │
    │ 2018-2019    │    │ 2019 holdout │
    └──────┬───────┘    └──────┬───────┘
           │                   │
           v                   v
    ┌─────────────────────────────────────┐
    │   TRAINING PIPELINE (GPU-Resident)  │
    │   • Batch loader (batch_size=16)    │
    │   • Data augmentation (on-the-fly)  │
    │   • Model forward pass (ResNet-50)  │
    │   • Loss computation (Focal Loss)   │
    │   • Backward pass + optimizer step  │
    │   • Validation loop                 │
    └─────────┬───────────────────────────┘
              │
              v
         ┌─────────────────┐
         │  Best Weights   │
         │ (Checkpointed)  │
         └────────┬────────┘
                  │
                  v
         ┌──────────────────────┐
         │  EVALUATION PIPELINE │
         │  • Test set (2020)   │
         │  • Metrics compute   │
         │  • Error analysis    │
         │  • Visualization     │
         └──────────────────────┘
                  │
                  v
         ┌─────────────────────┐
         │ Results + Report    │
         └─────────────────────┘
```

---

## 2. Component Architecture

### 2.1 System Components

```
src/
├── preprocessing.py          # Offline data pipeline
├── data_loader.py            # HDF5 batch generator
├── model_arch.py             # ResNet-50 + U-Net definitions
├── train.py                  # Training loop
├── evaluate.py               # Evaluation metrics
└── inference.py              # Inference API
```

#### **Component 1: Preprocessing Module** (`preprocessing.py`)
**Responsibility:** Convert raw Himawari-8 + MMD data into labeled, balanced dataset.

**Inputs:**
- Raw Himawari-8 netCDF4 files (~1 GB each)
- MMD lightning CSV (time, lat, lon)

**Outputs:**
- HDF5 dataset with:
  - `images`: shape (N, H, W, C) → (N, 64, 64, 3) or (N, 256, 256, 3)
  - `labels`: shape (N,) → binary [0, 1]
  - `metadata`: timestamps, lat/lon, split assignment

**Key Functions:**
```python
def crop_himawari_region(nc_file, bbox=(100, 120, -5, 15)):
    """Crop to Malaysia region; returns (H, W, C) tensor"""

def reproject_lightning_to_pixels(lat, lon, projection, src_bounds):
    """Map MMD strike (lat/lon) → Himawari-8 pixel (i, j)"""

def cloud_mask(ir_band, temp_threshold=290):
    """Mask clear sky / shallow clouds; returns binary mask"""

def normalize_bands(rgb_tensor):
    """Normalize to [0, 1]; returns (H, W, C)"""

def label_patch(image_time, lightning_events, lead_window=(0, 60)):
    """Assign label 1 if ≥1 strike in [t, t+lead_window] else 0"""

def balance_dataset(images, labels, downsample_ratio=0.2):
    """Downsample negative class; return balanced dataset"""

def create_hdf5_dataset(images, labels, metadata, output_path):
    """Write to HDF5 with compression (gzip level 4)"""
```

**Design Decision:** **Offline preprocessing** (run once)
- ✅ Saves training time (no I/O during training)
- ✅ Reproducible (same dataset every run)
- ✅ Enables parallel preprocessing (CPU-bound)
- ✅ Storage trade-off: ~150 GB HDF5 vs. 700 GB raw

---

#### **Component 2: Data Loader Module** (`data_loader.py`)
**Responsibility:** Efficient batch generation from HDF5 for GPU training.

**Inputs:**
- HDF5 dataset file
- Batch size (16 for RTX 3050)
- Augmentation config

**Outputs:**
- Batches of (images, labels) loaded into GPU memory

**Key Class:**
```python
class HDF5DataLoader:
    def __init__(self, hdf5_path, batch_size=16, augment=True):
        self.hdf5_path = hdf5_path
        self.batch_size = batch_size
        self.augment = augment
        # Pre-load dataset info (not images)
        with h5py.File(hdf5_path, 'r') as f:
            self.num_samples = f['images'].shape[0]
            self.image_shape = f['images'].shape[1:]
    
    def __getitem__(self, idx):
        """Lazy load batch from disk; apply augmentation"""
        with h5py.File(self.hdf5_path, 'r') as f:
            batch_images = f['images'][idx*batch_size:(idx+1)*batch_size]
            batch_labels = f['labels'][idx*batch_size:(idx+1)*batch_size]
        
        # Augment on CPU (flip, rotate, jitter)
        if self.augment:
            batch_images = self._augment(batch_images)
        
        # Convert to tensors
        return torch.from_numpy(batch_images), torch.from_numpy(batch_labels)
    
    def _augment(self, images):
        """Random flips, rotations, intensity jitter"""
        # Implementation: albumentations or torchvision.transforms
```

**Design Decision:** **Lazy loading from disk**
- ✅ Fits entire dataset without loading into RAM
- ✅ Augmentation on CPU (frees GPU for training)
- ✅ Handles class imbalance via sampling strategy

---

#### **Component 3: Model Architecture Module** (`model_arch.py`)
**Responsibility:** Define CNN architectures (ResNet-50 primary, U-Net optional).

**Primary Model: ResNet-50 Patch Classifier**

```python
class LightningResNet50(nn.Module):
    def __init__(self, num_input_channels=3, num_classes=1):
        super().__init__()
        
        # Load pretrained ResNet-50 (ImageNet)
        self.backbone = torchvision.models.resnet50(pretrained=True)
        
        # Adapt first conv for multi-channel input
        if num_input_channels != 3:
            original_conv1 = self.backbone.conv1
            self.backbone.conv1 = nn.Conv2d(
                num_input_channels, 64, kernel_size=7, stride=2, padding=3
            )
            # Initialize new layer with mean of original weights
            with torch.no_grad():
                self.backbone.conv1.weight[:, :3, :, :] = original_conv1.weight
                if num_input_channels > 3:
                    self.backbone.conv1.weight[:, 3:, :, :] = original_conv1.weight[:, 0:1, :, :]
        
        # Replace final FC layer for binary classification
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
            nn.Sigmoid()  # Binary output [0, 1]
        )
    
    def forward(self, x):
        return self.backbone(x)
```

**Input Shape:** (batch_size, 3-5 channels, 64, 64)  
**Output Shape:** (batch_size, 1) → probability ∈ [0, 1]  
**Expected Memory:** ~6 GB with batch_size=16 on RTX 3050

**Optional Model: U-Net Segmentation**

```python
class LightningUNet(nn.Module):
    """Pixel-level probability map (segmentation approach)"""
    def __init__(self, num_input_channels=3):
        super().__init__()
        # 4-layer encoder + 4-layer decoder with skip connections
        # Input: (batch, C, 256, 256)
        # Output: (batch, 1, 256, 256) → per-pixel probability
    
    def forward(self, x):
        # Encoder path
        # Decoder path with skip connections
        return output_map
```

**Input Shape:** (batch_size, 3-5 channels, 256, 256)  
**Output Shape:** (batch_size, 1, 256, 256) → spatial probability map  
**Memory Trade-off:** Higher memory cost; skip if OOM

---

#### **Component 4: Training Module** (`train.py`)
**Responsibility:** Training loop with checkpointing, early stopping, logging.

**Key Functions:**

```python
def train_epoch(model, train_loader, optimizer, loss_fn, device):
    """Single epoch training"""
    model.train()
    total_loss = 0
    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = loss_fn(outputs, labels.unsqueeze(1))
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)

def validate(model, val_loader, loss_fn, device):
    """Validation loop"""
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = loss_fn(outputs, labels.unsqueeze(1))
            total_loss += loss.item()
            
            all_preds.append(outputs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    
    return total_loss / len(val_loader), np.concatenate(all_preds), np.concatenate(all_labels)

def train_full_pipeline(config):
    """End-to-end training with early stopping"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize
    model = LightningResNet50(num_input_channels=config['channels']).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    loss_fn = FocalLoss(alpha=0.25, gamma=2.0)  # For class imbalance
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    # Training loop
    for epoch in range(config['max_epochs']):
        train_loss = train_epoch(model, train_loader, optimizer, loss_fn, device)
        val_loss, val_preds, val_labels = validate(model, val_loader, loss_fn, device)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= config['patience']:
                print(f"Early stopping at epoch {epoch}")
                break
        
        print(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
    
    return model
```

**Training Configuration:**
```yaml
max_epochs: 100
batch_size: 16  # RTX 3050 constraint
learning_rate: 1e-3
optimizer: Adam
loss: FocalLoss  # Handles class imbalance
scheduler: ReduceLROnPlateau  # Reduce LR if validation loss plateaus
early_stopping_patience: 10
gradient_clipping: 1.0  # Prevent exploding gradients
```

---

#### **Component 5: Evaluation Module** (`evaluate.py`)
**Responsibility:** Compute metrics and generate error analysis.

```python
def evaluate_model(model, test_loader, device):
    """Comprehensive evaluation on test set"""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images).cpu().numpy()
            all_preds.append((outputs > 0.5).astype(int))
            all_probs.append(outputs)
            all_labels.append(labels.numpy())
    
    preds = np.concatenate(all_preds)
    probs = np.concatenate(all_probs)
    labels = np.concatenate(all_labels)
    
    # Compute metrics
    metrics = {
        'accuracy': accuracy_score(labels, preds),
        'precision': precision_score(labels, preds),
        'recall': recall_score(labels, preds),  # POD
        'f1': f1_score(labels, preds),
        'roc_auc': roc_auc_score(labels, probs),
        'far': fp / (tp + fp),  # False alarm ratio
        'hss': hss_score(labels, preds),  # Heidke skill score
    }
    
    return metrics, preds, labels

def error_analysis(preds, labels, images):
    """Identify and visualize FP/FN"""
    tp = (preds == 1) & (labels == 1)
    fp = (preds == 1) & (labels == 0)
    tn = (preds == 0) & (labels == 0)
    fn = (preds == 0) & (labels == 1)
    
    # Sample FP/FN patches for visualization
    fp_indices = np.where(fp)[0][:10]
    fn_indices = np.where(fn)[0][:10]
    
    # Create visualizations (matplotlib)
    visualize_patches(images[fp_indices], 'False Positives')
    visualize_patches(images[fn_indices], 'False Negatives')
```

---

#### **Component 6: Inference Module** (`inference.py`)
**Responsibility:** API for per-image predictions (useful for deployment).

```python
class LightningPredictor:
    def __init__(self, model_path, device='cuda'):
        self.device = torch.device(device)
        self.model = LightningResNet50(num_input_channels=3)
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.to(device).eval()
    
    def predict(self, himawari_image_path, lead_time_window=(0, 60)):
        """
        Args:
            himawari_image_path: Path to netCDF4 file
            lead_time_window: Tuple (start_min, end_min)
        
        Returns:
            {
                'probability': float [0, 1],
                'prediction': int [0, 1],
                'confidence': float,
                'lead_time_window': tuple
            }
        """
        # Load and preprocess image
        img = load_himawari_image(himawari_image_path)
        img_normalized = normalize_bands(img)
        img_tensor = torch.from_numpy(img_normalized).unsqueeze(0).to(self.device)
        
        # Predict
        with torch.no_grad():
            prob = self.model(img_tensor).item()
        
        prediction = 1 if prob > 0.5 else 0
        confidence = max(prob, 1 - prob)
        
        return {
            'probability': prob,
            'prediction': prediction,
            'confidence': confidence,
            'lead_time_window': lead_time_window
        }
    
    def predict_batch(self, image_list, batch_size=16):
        """Batch inference for multiple images"""
        results = []
        for i in range(0, len(image_list), batch_size):
            batch = image_list[i:i+batch_size]
            # Load and predict batch
            results.extend([...])
        return results
```

---

## 3. Data Flow & Storage Strategy

### 3.1 Data Pipeline Stages

**Stage 1: Raw Data Acquisition**
```
Himawari-8 Archive (JMA/NOAA)  →  netCDF4 files (~700 GB)
MMD Lightning Detection System  →  CSV (~10 MB)
```

**Stage 2: Offline Preprocessing** (CPU-bound, parallelizable)
```
for each Himawari-8 netCDF4 file:
    1. Extract 3 bands (IR, WV, visible)
    2. Crop to Malaysia region (20°×20°)
    3. Reproject lightning strikes → pixel coordinates
    4. Apply cloud mask (T > 290 K)
    5. Normalize to [0, 1]
    6. Crop into 64×64 patches (sliding window or centered on strikes)
    7. Label: 1 if lightning in [t, t+60min], else 0

Output: HDF5 dataset (~150 GB, compressed)
```

**Stage 3: Dataset Splits** (Time-ordered, no shuffle across years)
```
Train:  2018-01-01 to 2019-06-30  (18 months)
Val:    2019-07-01 to 2019-12-31  (6 months)
Test:   2020-01-01 to 2020-12-31  (12 months)
```

**Stage 4: Class Balancing**
```
Original imbalance: Lightning:Non-lightning ≈ 1:20
Strategy:
  • Downsample negatives to 1:5 ratio (reduce FP in training)
  • Use Focal Loss (emphasizes hard examples)
  • Class weights in loss: weight_lightning=5, weight_no_lightning=1
```

### 3.2 Storage Breakdown

| Component | Format | Size | Location |
|-----------|--------|------|----------|
| **Raw Himawari-8** | netCDF4 | ~700 GB | NOAA/JMA archive (download on demand) |
| **Raw MMD** | CSV | ~10 MB | Institutional database |
| **Preprocessed Dataset** | HDF5 (gzip level 4) | ~150 GB | Local SSD / NAS |
| **Train Split** | HDF5 (in-memory indexed) | ~75 GB | Reference via HDF5 |
| **Val Split** | HDF5 (in-memory indexed) | ~30 GB | Reference via HDF5 |
| **Test Split** | HDF5 (in-memory indexed) | ~45 GB | Reference via HDF5 |
| **Model Weights** | PyTorch .pth | ~100 MB | Checkpointed during training |
| **TensorBoard Logs** | Event files | ~1-2 GB | Training monitoring |

---

## 4. Memory Budget for RTX 3050

### 4.1 GPU Memory Allocation (8 GB total)

```
┌─────────────────────────────────────────────┐
│  NVIDIA RTX 3050 Memory Budget (8 GB)       │
├─────────────────────────────────────────────┤
│  Model weights (ResNet-50):      ~100 MB   │
│  Batch (16 × 64×64×3 float32):  ~2.4 GB   │
│  Activations (forward pass):     ~1.5 GB   │
│  Gradients (backward pass):      ~2.4 GB   │
│  Optimizer state (Adam):         ~1.0 GB   │
│  ─────────────────────────────────────────  │
│  Total:                          ~7.4 GB   │
│  ─────────────────────────────────────────  │
│  Headroom (safety margin):       ~0.6 GB   │
└─────────────────────────────────────────────┘
```

**Batch Size Analysis:**

| Batch Size | Total GPU Memory | Feasible? | Notes |
|----------|------------------|-----------|-------|
| 32 | ~8.8 GB | ❌ No | Exceeds RTX 3050; OOM risk |
| 24 | ~7.9 GB | ⚠️ Tight | Risky; not recommended |
| **16** | **~7.4 GB** | ✅ Yes | **Recommended; safe margin** |
| 8 | ~5.2 GB | ✅ Yes | Very safe; slower training |

**Recommendation: Use batch_size=16** ✓

---

### 4.2 GPU Optimization Techniques

**Technique 1: Gradient Checkpointing**
```python
# Reduces memory by ~30% at cost of 20% slower training
from torch.utils.checkpoint import checkpoint

def forward_with_checkpoint(model, x):
    return checkpoint(model, x)
```

**Technique 2: Mixed Precision Training (FP16)**
```python
# Reduces memory by ~40%; modern GPUs handle gracefully
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
with autocast():
    loss = criterion(model(x), y)
scaler.scale(loss).backward()
```

**Technique 3: Gradient Accumulation**
```python
# Simulate larger batch without loading full batch into memory
accumulation_steps = 2
effective_batch_size = batch_size * accumulation_steps

for i, (x, y) in enumerate(loader):
    loss = criterion(model(x), y)
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

**If OOM Still Occurs: Fall Back to ResNet-18**
```python
# ResNet-18: ~1/4 parameters, ~1/3 memory
backbone = torchvision.models.resnet18(pretrained=True)
# ... rest of model definition
```

---

## 5. System Architecture Diagram

### 5.1 Deployment View

```
┌────────────────────────────────────────────────────────┐
│                 TRAINING PHASE                          │
├────────────────────────────────────────────────────────┤
│                                                          │
│  HDF5 Dataset           PyTorch Model              GPU  │
│  (150 GB on disk)       (ResNet-50)           (RTX 3050)│
│         │                                          ▲    │
│         │                                          │    │
│         ├─ Batch Loader (16 samples at a time)───┤    │
│         │  • Lazy load from HDF5                   │    │
│         │  • On-the-fly augmentation (CPU)        │    │
│         │  • Async transfer to GPU                │    │
│         │                                          │    │
│         └─ Training Loop                           │    │
│            • Forward pass (ResNet-50)              │    │
│            • Loss (Focal Loss)                     │    │
│            • Backward pass                        │    │
│            • Optimizer step (Adam)                │    │
│            • Validation every N steps            │    │
│            • Early stopping if no improvement    │    │
│                                                   │    │
│  Best Weights ◄─────────────────────────────────┘    │
│  (saved to disk)                                       │
│                                                        │
└────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────┐
│               EVALUATION & INFERENCE PHASE              │
├────────────────────────────────────────────────────────┤
│                                                         │
│  Test Set (2020)     Loaded Model      Inference API  │
│  (45 GB HDF5)        (best_weights)    (inference.py) │
│         │                    │               ▲        │
│         │                    │               │        │
│         └─ Batch Inference ──┴───────────────┘        │
│            • Load batch (16 at a time)                │
│            • Forward pass → predictions               │
│            • Compute metrics (recall, FAR, etc.)      │
│            • Error analysis (FP/FN visualization)    │
│            • Generate reports + plots                │
│                                                        │
│  Output: Metrics JSON, plots, error analysis         │
│                                                        │
└────────────────────────────────────────────────────────┘
```

### 5.2 Code Execution Flow

```
main.py
  ├─ config = load_config('config.yaml')
  │
  ├─ Phase 1: Preprocessing (One-time, offline)
  │   └─ python src/preprocessing.py
  │       • Load Himawari-8 netCDF4 files
  │       • Load MMD lightning CSV
  │       • Process all images → HDF5
  │       • Create train/val/test splits
  │       • Balance dataset
  │       → Output: data/processed/dataset.h5
  │
  ├─ Phase 2: Training
  │   └─ python src/train.py
  │       • Initialize model (ResNet-50)
  │       • Create data loaders (batch_size=16)
  │       • Training loop (100 epochs, early stopping)
  │       • Save best weights
  │       → Output: models/best_resnet50.pth
  │
  ├─ Phase 3: Evaluation
  │   └─ python src/evaluate.py
  │       • Load best model
  │       • Run test set through model
  │       • Compute metrics
  │       • Generate error analysis
  │       • Create visualizations
  │       → Output: results/metrics.json, plots/
  │
  └─ Phase 4: Inference (Optional)
      └─ python src/inference.py
          • Load trained model
          • Accept new Himawari-8 image
          • Predict lightning probability
          • Return prediction + confidence
```

---

## 6. Technology Stack Justification

| Component | Technology | Why |
|-----------|-----------|-----|
| **Language** | Python 3.9+ | ML ecosystem; PyTorch/TensorFlow native support |
| **DL Framework** | PyTorch | Better GPU memory control; easier debugging; good for research |
| **Data** | HDF5 + h5py | Efficient storage; fast random access; compression support |
| **Satellite Data** | xarray + netCDF4 | Standard for geophysical data; Himawari-8 native format |
| **Preprocessing** | NumPy + Cartopy + PyProj | Scientific computing; geospatial reprojection |
| **Monitoring** | TensorBoard | Real-time training curves; easy to spot overfitting |
| **Version Control** | Git | Reproducibility; code history; collaboration |
| **Environment** | Conda | Manages Python + system dependencies (CUDA, cuDNN) |

---

## 7. Scalability & Fallback Plans

### 7.1 If GPU Memory Exceeds 8 GB

**Option 1: Reduce Batch Size**
```yaml
batch_size: 16  →  8  (50% more training time)
```

**Option 2: Use Gradient Checkpointing**
```python
model = checkpoint_sequential(model, 4, x)  # ~30% memory savings
```

**Option 3: Mixed Precision (FP16)**
```python
from torch.cuda.amp import autocast
with autocast():  # ~40% memory savings
    loss = criterion(model(x), y)
```

**Option 4: Smaller Model**
```yaml
backbone: ResNet-50  →  ResNet-18  (1/4 parameters)
```

### 7.2 If Training is Too Slow

**Option 1: Reduce Patch Size**
```yaml
patch_size: 64×64  →  32×32  (8x fewer pixels)
# Trade-off: Loss of spatial context
```

**Option 2: Reduce Dataset Size**
```python
# Use subset of 2018-2019 data
# Train on 30% of dataset first; validate approach
```

**Option 3: Reduce Number of Bands**
```yaml
input_channels: 3  →  1  (IR only)
# Use highest-priority channel (IR brightness temp)
```

---

## 8. Reproducibility & Version Control

### 8.1 Reproducibility Requirements

```python
# Fix random seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)
```

### 8.2 Version Tracking

```
.gitignore:
├── data/                    # Data files (too large)
├── models/*.pth             # Model weights (versioned separately)
├── __pycache__/
├── *.log
└── .DS_Store

Committed to Git:
├── src/                     # All code
├── tests/                   # Unit tests
├── config.yaml              # Hyperparameters
├── requirements.txt         # Dependency versions
├── environment.yml          # Conda environment
└── README.md                # Setup instructions
```

---

## 9. Testing Strategy

### 9.1 Unit Tests

```python
# test_preprocessing.py
def test_reproject_lightning_to_pixels():
    lat, lon = 3.0, 101.5  # Kuala Lumpur
    pixel_x, pixel_y = reproject_lightning_to_pixels(lat, lon)
    assert 0 <= pixel_x < img_width
    assert 0 <= pixel_y < img_height

def test_normalize_bands():
    img = np.random.rand(64, 64, 3) * 1000  # Arbitrary scale
    normalized = normalize_bands(img)
    assert normalized.min() >= 0 and normalized.max() <= 1

# test_model.py
def test_model_forward_pass():
    model = LightningResNet50(num_input_channels=3)
    x = torch.randn(16, 3, 64, 64)
    y = model(x)
    assert y.shape == (16, 1)
    assert (y >= 0).all() and (y <= 1).all()
```

### 9.2 Integration Tests

```python
def test_end_to_end_training():
    """Smoke test: can we train for 1 epoch without errors?"""
    loader = HDF5DataLoader('dummy_dataset.h5', batch_size=4)
    model = LightningResNet50()
    loss = train_epoch(model, loader, optimizer, criterion, device='cpu')
    assert not np.isnan(loss)
    assert loss > 0
```

---

## 10. Deployment Roadmap

### 10.1 Capstone Deliverable (Academic)
```
├── Reproducible code (GitHub)
├── Trained model weights
├── Preprocessed dataset (or reproducible build scripts)
├── Methods paper + documentation
└── Inference API (inference.py)
```

### 10.2 Future MMD Integration (Post-Capstone)
```
├── Docker container (easy deployment)
├── Real-time data pipeline (Himawari-8 feed)
├── REST API (predictions on demand)
├── Web dashboard (visualization)
└── Monitoring + retraining pipeline
```

---

## 11. Sign-off & Next Steps

**Prepared by:** Winston (System Architect)  
**Review by:** Bryan Chai Wen Cheng  
**Status:** Ready for Implementation  

### Key Takeaways

1. ✅ **RTX 3050 Compatible:** Batch size 16; ~7.4 GB memory usage
2. ✅ **Offline Preprocessing:** One-time HDF5 generation; saves training time
3. ✅ **Reproducible:** Fixed seeds, versioned code, documented pipeline
4. ✅ **Scalable:** Fallback options if OOM (reduce batch, use ResNet-18, gradient checkpointing)
5. ✅ **Monitored:** TensorBoard logging; early stopping; validation tracking

### Next Phase: Implementation

**Recommended Next Agent:** Amelia 💻 (Senior Developer)

Amelia will:
- Implement preprocessing.py (data pipeline)
- Implement model_arch.py (ResNet-50 definition)
- Implement train.py (training loop)
- Write unit tests
- Create reproducible environment (requirements.txt)

---

**END OF ARCHITECTURE DOCUMENT**
