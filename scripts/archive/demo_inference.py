# ⚠️ SUPERSEDED — retained for provenance only. Not the final result. See README.md and report/ for Version 2.
import os
import torch
from pathlib import Path
from src.lightning_model import LightningMetadataClassifier
from src.lightning_data_loader import create_lightning_loaders


def main():
    hdf5_path = Path('data/processed/lightning_dataset.h5')
    model_path = Path('models/lightning_classifier.pth')
    if not hdf5_path.exists():
        raise FileNotFoundError('Missing data/processed/lightning_dataset.h5. Run python src/ingest_met_data.py first.')
    if not model_path.exists():
        raise FileNotFoundError('Missing models/lightning_classifier.pth. Run python src/train_lightning.py first.')

    loaders = create_lightning_loaders(str(hdf5_path), batch_size=8)
    test_loader = loaders['test']
    model = LightningMetadataClassifier(input_size=4, hidden_size=256, dropout=0.3)
    checkpoint = torch.load(model_path, map_location='cpu')
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict)
    model.eval()

    with torch.no_grad():
        features, labels = next(iter(test_loader))
        outputs = model(features).squeeze()
        probs = torch.sigmoid(outputs).cpu().numpy()
        preds = (probs >= 0.5).astype(int)

    print('Demo inference on 8 test samples')
    print('probabilities:', [round(float(p), 4) for p in probs[:8]])
    print('predictions:', preds[:8].tolist())
    print('labels:', labels[:8].int().tolist())


if __name__ == '__main__':
    main()
