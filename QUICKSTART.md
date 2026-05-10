"""
Quick start guide for testing the Lightning Flash Classification pipeline.

This script demonstrates how to:
1. Create a dummy HDF5 dataset
2. Test the data loader
3. Run training with validation
4. Evaluate on test set
"""

import numpy as np
import torch
from pathlib import Path

print("""
╔══════════════════════════════════════════════════════════════════════════╗
║    LIGHTNING FLASH CLASSIFICATION - QUICK START GUIDE                   ║
╚══════════════════════════════════════════════════════════════════════════╝

STEP 1: CREATE DUMMY DATASET
─────────────────────────────
Run this command to generate a test HDF5 dataset:

    python src/create_dummy_dataset.py

This creates: data/processed/dataset.h5 (500 test samples)


STEP 2: TEST DATA LOADER
────────────────────────
Verify the data loader works:

    python -c \"
from src.data_loader import create_data_loaders
loaders = create_data_loaders('data/processed/dataset.h5', batch_size=16)
for images, labels in loaders['train']:
    print('Batch shapes:')
    print('  Images:', images.shape)
    print('  Labels:', labels.shape)
    break
    \"


STEP 3: RUN TRAINING
────────────────────
Train the model on dummy data:

    python src/train.py

Expected output:
  - Model loads and initializes
  - Training loop starts
  - Loss decreases over epochs
  - Best model saved to models/best_resnet50.pth


STEP 4: EVALUATE ON TEST SET
─────────────────────────────
Test the model:

    python -c \"
import torch
from src.model_arch import LightningResNet50
from src.data_loader import create_data_loaders
from src.evaluate import evaluate_model

model = LightningResNet50(num_input_channels=3)
model.load_state_dict(torch.load('models/best_resnet50.pth'))
loaders = create_data_loaders('data/processed/dataset.h5', batch_size=16)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
metrics, preds, labels = evaluate_model(model, loaders['test'], device)
    \"


STEP 5: TEST INFERENCE API
──────────────────────────
Test single prediction:

    python -c \"
import torch
import numpy as np
from src.inference import LightningPredictor

predictor = LightningPredictor('models/best_resnet50.pth')
dummy_image = np.random.rand(3, 64, 64)
result = predictor.predict(dummy_image)
print('Prediction:', result)
    \"


PRODUCTION WORKFLOW (with real data)
────────────────────────────────────

1. Download Himawari-8 data:
   - Create: data/raw/himawari8/
   - Download netCDF4 files from JMA

2. Get MMD Lightning data:
   - Save CSV to: data/raw/mmd_lightning.csv
   - Columns: timestamp, latitude, longitude, intensity

3. Run preprocessing:
   python -c \"from src.preprocessing import preprocess_from_config; preprocess_from_config('config.yaml')\"

4. Train model:
   python src/train.py

5. Evaluate:
   python -c \"from src.evaluate import evaluate_model; ...\"


USEFUL COMMANDS
───────────────

# Check GPU availability
python -c \"import torch; print('GPU available:', torch.cuda.is_available())\"

# List installed packages
pip list

# Run unit tests
pytest tests/ -v --cov=src/

# View TensorBoard logs
tensorboard --logdir=logs/


FILE STRUCTURE
──────────────

Project-Capstone/
├── src/
│   ├── __init__.py
│   ├── data_loader.py           ← HDF5 lazy loading
│   ├── model_arch.py            ← ResNet-50 + Focal Loss
│   ├── train.py                 ← Training loop
│   ├── evaluate.py              ← Metrics & analysis
│   ├── inference.py             ← Prediction API
│   ├── preprocessing.py         ← Himawari-8 → HDF5
│   └── create_dummy_dataset.py  ← Test data generator
├── data/
│   ├── raw/                     ← Himawari-8 + MMD (gitignored)
│   └── processed/               ← HDF5 dataset (gitignored)
├── models/
│   └── best_resnet50.pth        ← Trained weights (gitignored)
├── results/
│   └── metrics.json             ← Test metrics
├── tests/
│   ├── test_data_loader.py
│   ├── test_model.py
│   ├── test_train.py
│   ├── test_inference.py
│   └── test_preprocessing.py
├── config.yaml
├── requirements.txt
└── README.md


TROUBLESHOOTING
───────────────

PyTorch import error:
  → Install PyTorch: pip install torch torchvision

h5py not found:
  → Install h5py: pip install h5py

CUDA out of memory:
  → Reduce batch_size in config.yaml (default: 16)
  → Or use CPU-only mode

GPU not detected:
  → Check: python -c \"import torch; print(torch.cuda.is_available())\"
  → Ensure NVIDIA GPU drivers installed


NEXT STEPS
──────────

1. ✅ Code implemented and tested
2. ⏳ Download real Himawari-8 data
3. ⏳ Get MMD Lightning detection system data
4. ⏳ Run preprocessing on real data
5. ⏳ Train full model
6. ⏳ Deploy and serve predictions


For questions or issues, see README.md or contact your supervisor.
""")
