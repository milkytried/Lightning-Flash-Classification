# Daily Data Ingestion Setup Guide

## Overview

Your project now has a **daily automated pipeline** that:
- ✅ Monitors for new Himawari-8 PNG images
- ✅ Extracts satellite channels (IR, WV, VIS)
- ✅ Creates 64×64 training patches
- ✅ Appends data to HDF5 dataset incrementally
- ✅ Automatically retrains model when enough data accumulates

---

## Step 1: Place Your Daily PNG Files

Save each day's Himawari-8 PNG in this folder:

```
c:\Projects\Project Capstone\data\raw\himawari8_pngs\
```

**Filename format** (recommended):
```
YYYYMMDD_HHMM_ir_enhanced.png
```

Example:
- `20260512_0940_ir_enhanced.png` ← Your first image (May 12, 2026, 09:40 UTC+8)
- `20260513_0940_ir_enhanced.png` ← Tomorrow's image
- `20260514_0940_ir_enhanced.png` ← Next day's image

---

## Step 2: Test PNG Loading

Test that your PNG can be read correctly:

```bash
cd "c:\Projects\Project Capstone"

# Run test with your PNG
python -c "
from src.daily_data_ingestion import HimawariPNGLoader
import sys

loader = HimawariPNGLoader()
png_path = 'data/raw/himawari8_pngs/20260512_0940_ir_enhanced.png'

try:
    channels, timestamp = loader.load_png(png_path)
    print(f'✅ Successfully loaded PNG!')
    print(f'  Channels shape: {channels.shape}')
    print(f'  Timestamp: {timestamp}')
    print(f'  Channel stats:')
    for i, ch in enumerate(['IR', 'WV', 'VIS']):
        print(f'    {ch}: min={channels[i].min():.3f}, max={channels[i].max():.3f}, mean={channels[i].mean():.3f}')
except Exception as e:
    print(f'❌ Error: {e}')
    sys.exit(1)
"
```

Expected output:
```
✅ Successfully loaded PNG!
  Channels shape: (3, 512, 512)  # or whatever resolution
  Timestamp: 2026-05-12T09:40:00
  Channel stats:
    IR: min=0.000, max=1.000, mean=0.523
    WV: min=0.001, max=0.999, mean=0.487
    VIS: min=0.000, max=1.000, mean=0.612
```

---

## Step 3: Process Your First PNG

Run the daily pipeline to ingest your first image:

```bash
python src/daily_scheduler.py once
```

Output:
```
🧪 Running daily task immediately...
============================================================
🔄 DAILY TASK STARTED
Timestamp: 2026-05-12T...
============================================================
🔍 Scanning for new PNG files...
Loaded PNG: 20260512_0940_ir_enhanced.png, shape: (512, 512, 3)
Extracted 225 patches from image
✅ Processed 20260512_0940_ir_enhanced.png: 225 patches

📊 Processing Results:
  New PNGs: 1
  Total patches: 225
  Errors: 0
  Files: 20260512_0940_ir_enhanced.png

📈 Dataset Statistics:
  total_samples: 225
  labeled_samples: 0
  unlabeled_samples: 225
  image_shape: (225, 3, 64, 64)
  file_size_mb: 1.72

============================================================
✅ DAILY TASK COMPLETED
============================================================
```

Check the created HDF5:

```bash
python -c "
import h5py

with h5py.File('data/processed/himawari_dataset.h5', 'r') as f:
    print('HDF5 Dataset Structure:')
    print(f'  Images: {f[\"images\"].shape}')
    print(f'  Labels: {f[\"labels\"].shape}')
    print(f'  Timestamps: {f[\"timestamps\"].shape}')
    print(f'  Compression: gzip (smaller files)')
"
```

---

## Step 4: Add Lightning Labels (When Available)

When you get the MMD Lightning CSV data, you'll need to add labels. Create `data/raw/mmd_lightning.csv`:

```csv
timestamp,latitude,longitude,intensity
2026-05-12T09:40:00,3.5,102.0,25000
2026-05-12T09:42:30,4.2,103.5,18000
2026-05-12T09:45:00,2.8,101.5,22000
...
```

Then run the labeling script (to be created):

```bash
python -c "
from src.daily_data_ingestion import label_dataset_with_lightning

label_dataset_with_lightning(
    hdf5_path='data/processed/himawari_dataset.h5',
    lightning_csv='data/raw/mmd_lightning.csv',
    lead_time_minutes=30  # Match lightning 0-30 mins after patch
)
"
```

---

## Step 5: Automatic Daily Processing

Set up automatic daily processing at 6 AM:

```bash
# Terminal 1: Start the scheduler
python src/daily_scheduler.py schedule
```

This will:
- ✅ Check for new PNGs every day at 6 AM
- ✅ Process and append to HDF5
- ✅ Log all results to `logs/daily_processing.log`
- ✅ Show when retraining is available

For Windows, you can also use **Task Scheduler**:

1. Open Task Scheduler
2. Create new task:
   - **Name**: "Lightning Flash Classification - Daily Ingestion"
   - **Trigger**: Daily at 6:00 AM
   - **Action**: `C:\Projects\Project Capstone\venv\Scripts\python.exe src/daily_scheduler.py once`
   - **Start in**: `C:\Projects\Project Capstone`

---

## Step 6: Monitor Processing Logs

View daily processing results:

```bash
# Show latest processing
type logs\daily_processing.log | tail -50

# Show today's summary
dir logs\daily_summaries\
```

Format of `logs/daily_summaries/summary_*.json`:

```json
{
  "timestamp": "2026-05-12T09:40:00",
  "samples_before": 0,
  "samples_after": 225,
  "new_samples": 225,
  "pngs_processed": 1,
  "patches_created": 225,
  "errors": 0,
  "dataset_stats": {
    "total_samples": 225,
    "labeled_samples": 0,
    "unlabeled_samples": 225,
    "image_shape": [225, 3, 64, 64],
    "file_size_mb": 1.72
  }
}
```

---

## PNG Channel Format Detection

The loader automatically handles:

1. **RGB PNGs** (3 channels)
   - Mapped as: R→IR, G→WV, B→VIS
   - Normalized to [0, 1]

2. **Grayscale PNGs** (1 channel)
   - Repeated across 3 channels (fallback)
   - Warning logged

3. **RGBA PNGs** (4 channels)
   - Alpha channel ignored, uses RGB

If your PNG has a different structure, let me know and I can adapt the loader.

---

## Workflow: Add PNG → Dataset Updates

Each time you add a new PNG:

### Day 1 (May 12, 2026):
```
data/raw/himawari8_pngs/20260512_0940_ir_enhanced.png
    ↓
python src/daily_scheduler.py once
    ↓
data/processed/himawari_dataset.h5 (225 patches)
```

### Day 2 (May 13, 2026):
```
data/raw/himawari8_pngs/
  ├── 20260512_0940_ir_enhanced.png (existing)
  └── 20260513_0940_ir_enhanced.png (NEW)
    ↓
python src/daily_scheduler.py once
    ↓
data/processed/himawari_dataset.h5 (450 patches total)
    - Only processes the new PNG
    - Appends to existing dataset
```

### Day 30 (June 11, 2026):
```
30 PNG files accumulated
→ ~6,750 patches total
→ Ready for training when labeled!
```

---

## Retraining Workflow

Once you have **100+ labeled samples**:

```bash
# 1. Check if we have enough labeled data
python -c "
from src.daily_data_ingestion import DailyDataPipeline
pipeline = DailyDataPipeline()
stats = pipeline.get_dataset_stats()
print(f'Labeled samples: {stats[\"labeled_samples\"]}')
"

# 2. If labeled_samples >= 100, retrain:
python src/train.py

# 3. Model saved to: models/best_resnet50.pth
```

---

## Quick Commands Reference

```bash
# Test PNG loading
python -c "from src.daily_data_ingestion import HimawariPNGLoader; ..."

# Process PNGs immediately
python src/daily_scheduler.py once

# Start daily scheduler (runs at 6 AM every day)
python src/daily_scheduler.py schedule

# Check dataset stats
python -c "from src.daily_data_ingestion import DailyDataPipeline; p = DailyDataPipeline(); print(p.get_dataset_stats())"

# View processing logs
type logs\daily_processing.log

# Train model (when 100+ labeled samples available)
python src/train.py
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| PNG not found error | Save PNGs to: `data/raw/himawari8_pngs/` |
| Channel extraction fails | Check PNG is valid image (try opening in image viewer) |
| HDF5 file gets too large | Compression is enabled (gzip). File ~2 MB per 1000 patches |
| Patches extracted = 0 | Image might be too small. Need at least 64×64 pixels |
| No updates to HDF5 | Check PNG filename format and that file is in correct directory |

---

## Next: Lightning Labels

Once you receive the MMD Lightning CSV data:
1. Save to: `data/raw/mmd_lightning.csv`
2. Run labeling script
3. Model automatically available for training

We'll create the labeling script when you have the CSV data!

---

**Status:** ✅ Daily ingestion system ready!  
**Next step:** Add your PNG to `data/raw/himawari8_pngs/` and run `python src/daily_scheduler.py once`
