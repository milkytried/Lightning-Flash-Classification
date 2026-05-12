# Real Data Quick Start - May 2026

## Your Current Setup

📷 **What you have:**
- 1 Himawari-8 PNG image: `20260512_0940_ir_enhanced.png` (May 12, 09:40 UTC+8)
- Will add 1 new PNG every day automatically

🎯 **What this does:**
- Extracts 64×64 satellite patches (IR, WV, VIS channels)
- Builds HDF5 training dataset incrementally
- Ready for model training once lightning labels are added

---

## Day 1: TODAY (May 12, 2026)

### Step 1️⃣: Place Your PNG

Save the image to:
```
c:\Projects\Project Capstone\data\raw\himawari8_pngs\20260512_0940_ir_enhanced.png
```

### Step 2️⃣: Test the Pipeline

Run the test suite to verify everything works:

```bash
cd "c:\Projects\Project Capstone"
python -m pytest tests/test_daily_ingestion.py -v
```

Expected output:
```
test_png_loading PASSED
test_patch_creation PASSED
test_hdf5_creation PASSED
test_dataset_stats PASSED

✅ ALL TESTS PASSED!
```

### Step 3️⃣: Process Your First PNG

```bash
python src/daily_scheduler.py once
```

Output:
```
============================================================
🔄 DAILY TASK STARTED
🔍 Scanning for new PNG files...
✅ Processed 20260512_0940_ir_enhanced.png: 225 patches
📊 Dataset Statistics:
  total_samples: 225
  labeled_samples: 0 (waiting for lightning data)
✅ DAILY TASK COMPLETED
```

### Check What Was Created

```bash
python -c "
from src.daily_data_ingestion import DailyDataPipeline
p = DailyDataPipeline()
stats = p.get_dataset_stats()
print('Dataset Stats:')
for k, v in stats.items():
    print(f'  {k}: {v}')
"
```

---

## Day 2-30: Repeat Daily

### Add Tomorrow's PNG (May 13, 09:40 UTC+8)

```
data/raw/himawari8_pngs/
  ├── 20260512_0940_ir_enhanced.png
  └── 20260513_0940_ir_enhanced.png  ← NEW
```

### Auto Process at 6 AM (Optional)

Set up Windows Task Scheduler to run daily:

```powershell
# Create scheduled task
$action = New-ScheduledTaskAction -Execute "C:\Projects\Project Capstone\venv\Scripts\python.exe" -Argument "src\daily_scheduler.py once" -WorkingDirectory "C:\Projects\Project Capstone"
$trigger = New-ScheduledTaskTrigger -Daily -At "06:00"
Register-ScheduledTask -TaskName "Lightning Flash - Daily Ingestion" -Action $action -Trigger $trigger -RunLevel Highest
```

Or run manually each day:

```bash
python src/daily_scheduler.py once
```

### Expected Growth (30-Day Forecast)

```
May 12 (Day 1):  225 patches
May 13 (Day 2):  450 patches total
May 14 (Day 3):  675 patches total
...
June 10 (Day 30): 6,750 patches total ✅ Ready for training!
```

---

## When You Get Lightning Data

Once you receive the **MMD Lightning CSV** (e.g., May 20):

### Step 1: Save Lightning Data

```
data/raw/mmd_lightning.csv
```

Format:
```csv
timestamp,latitude,longitude,intensity
2026-05-12T09:40:00,3.5,102.0,25000
2026-05-12T09:42:30,4.2,103.5,18000
...
```

### Step 2: Label Your Dataset

```bash
python -c "
from src.daily_data_ingestion import label_dataset_with_lightning

label_dataset_with_lightning(
    hdf5_path='data/processed/himawari_dataset.h5',
    lightning_csv='data/raw/mmd_lightning.csv',
    lead_time_minutes=30
)
"
```

### Step 3: Check Labeled Samples

```bash
python -c "
from src.daily_data_ingestion import DailyDataPipeline
p = DailyDataPipeline()
stats = p.get_dataset_stats()
print(f'Labeled: {stats[\"labeled_samples\"]} / {stats[\"total_samples\"]}')
"
```

### Step 4: Train When Ready

Once you have ≥100 labeled samples:

```bash
python src/train.py
```

Model will train and save to: `models/best_resnet50.pth`

---

## Monitoring Your Progress

### View Processing Logs

```bash
# Real-time log
type logs\daily_processing.log | tail -20

# Daily summary files
dir logs\daily_summaries\
```

### Check Dataset Growth

```bash
python -c "
from src.daily_data_ingestion import DailyDataPipeline
p = DailyDataPipeline()
stats = p.get_dataset_stats()
print(f'Dataset size: {stats[\"total_samples\"]} samples')
print(f'File size: {stats[\"file_size_mb\"]:.1f} MB')
"
```

### View Sample Images (Optional)

```bash
python -c "
import h5py
import numpy as np
import matplotlib.pyplot as plt

with h5py.File('data/processed/himawari_dataset.h5', 'r') as f:
    # Get first sample
    sample = f['images'][0]  # (3, 64, 64)
    
    # Show IR channel
    plt.imshow(sample[0], cmap='gray')
    plt.title('IR Channel - First Sample')
    plt.colorbar()
    plt.savefig('sample_ir.png')
    print('✅ Saved to sample_ir.png')
"
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| **PNG not found** | Check filename format and save to `data/raw/himawari8_pngs/` |
| **No patches extracted** | PNG might be too small. Need ≥64×64 pixels |
| **HDF5 file too large** | Normal - uses gzip compression. ~2 MB per 1000 patches |
| **Test fails** | Run `pip install -r requirements.txt` again |

---

## Timeline to Training

| Date | Event | Status |
|------|-------|--------|
| May 12, 2026 | 1st PNG added | ✅ Today |
| May 12-19 | Accumulate 8 days of data | ⏳ Coming |
| May 20 | Get MMD Lightning data | ⏳ Expected |
| May 20-22 | Label dataset (~100+ samples) | ⏳ Expected |
| May 23 | **Start Training** 🎯 | ⏳ Target |
| May 25 | Model trained (2 days GPU time) | ⏳ Target |
| May 26 | Deploy and make predictions | ⏳ Target |

---

## Commands Cheat Sheet

```bash
# Test everything
python -m pytest tests/test_daily_ingestion.py -v

# Process PNGs now
python src/daily_scheduler.py once

# Start daily scheduler
python src/daily_scheduler.py schedule

# Check dataset
python -c "from src.daily_data_ingestion import DailyDataPipeline; print(DailyDataPipeline().get_dataset_stats())"

# Train model (when 100+ labeled samples)
python src/train.py

# View logs
type logs\daily_processing.log
```

---

## Questions?

Each morning (May 13+):
1. ✅ Add new PNG to `data/raw/himawari8_pngs/`
2. ✅ Run `python src/daily_scheduler.py once` (or automatic via Task Scheduler)
3. ✅ Check `logs/daily_processing.log` for status

---

**Status:** ✅ Pipeline ready for daily Himawari-8 data!  
**Next:** Add your PNG and run the test suite
