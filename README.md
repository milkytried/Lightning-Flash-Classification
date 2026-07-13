# Lightning flash classification from Himawari-9 imagery

This repository contains the PyTorch implementation used for the final FYP experiment: a frozen-backbone ResNet-50 that classifies same-time cloud-to-ground lightning occurrence from aligned Himawari-9 infrared patches. It is a research prototype, not an operational warning system and not a forward-looking nowcaster.

The final dataset contains 41,168 balanced 64×64 patches from AHI bands 8, 13, and 15 over approximately 99–120°E and 5°S–15°N. MMD strikes were aligned to 10-minute Himawari-9 frames and split chronologically: 33,226 training patches, 3,324 validation patches, and 4,618 test patches, with no date overlap between splits.

## Final aligned-model performance

The ResNet-50 backbone was frozen (23,508,032 parameters) and only the 262,401-parameter classification head was trained. Training used focal loss (α = 0.25, γ = 2.0). The decision threshold of 0.51 was selected by maximising F1 on the validation split and then applied once to the held-out test split.

| Metric | Held-out test value |
|---|---:|
| Accuracy | 0.9095 |
| Precision | 0.8742 |
| Recall / POD | 0.9567 |
| F1 | 0.9136 |
| ROC-AUC | 0.9681 |
| FAR | 0.126 |
| CSI | 0.841 |
| TSS | 0.819 |
| HSS | 0.819 |

The committed evidence for these values is [results/satellite_frozen_cpu_clean_metrics.json](results/satellite_frozen_cpu_clean_metrics.json). The earlier 11-PNG Himawari-8 experiment (accuracy 0.8765, ROC-AUC 0.9199) is retained only as the baseline used in Table 5.2 of the final report; it is not the headline result.

## Metadata baseline (leakage demonstration)

The metadata MLP can appear perfect when `amplitude` and `strike_type` are supplied, but those fields exist only because a strike was recorded. Using them to predict strike occurrence is circular label leakage, so the apparent perfect score is a negative result rather than an achievement. [scripts/retrain_honest_artifacts.py](scripts/retrain_honest_artifacts.py) reproduces both the leakage-prone probe and the honest latitude/longitude/time-only comparator, subject to the gitignored MMD dataset being available locally.

## Project structure

The executable source tree is:

```text
src/
├── __init__.py
├── build_occurrence_dataset.py
├── build_satellite_dataset.py
├── compare_lightning_models.py
├── daily_data_ingestion.py
├── data_loader.py
├── evaluate_lightning.py
├── evaluate_occurrence_baselines.py
├── himawari_data_loader.py
├── inference.py
├── ingest_met_data.py
├── lightning_data_loader.py
├── lightning_model.py
├── model_arch.py
├── plot_results.py
├── preprocessing.py
├── train.py
├── train_lightning.py
├── train_satellite.py
└── validate_satellite_dataset.py
```

Supporting paths are `scripts/` for retained result-generation utilities, `scripts/archive/` and `docs/archive/` for the superseded prototype audit trail, `tests/` for the offline-safe test suite, and `report/README.md` for the report-package index.

## Quick start

Python 3.10–3.12 is supported. The pinned requirements install CPU builds of PyTorch by default.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pytest tests -q
```

Useful checks that do not retrain a model are:

```bash
python src/validate_satellite_dataset.py --manifest data/processed/satellite_dataset.csv
python src/evaluate_occurrence_baselines.py --help
python src/plot_results.py --help
```

The first command requires the generated, gitignored satellite manifest and patches. The other commands name files committed to this repository and can be inspected with `--help` on a fresh clone.

## Reproducing the results

Place the licensed MMD CSV hierarchy under `data/raw/mmd_lightning/`, then run the final workflow from the repository root. These commands preserve the final run’s same-time labelling, three-band inputs, 1:1 class sampling, chronological split policy, no-data rejection, frozen backbone, and validation-only threshold selection.

1. Build the aligned dataset from MMD strikes and the public NOAA archive:

```bash
python src/build_satellite_dataset.py \
  --lightning-root data/raw/mmd_lightning \
  --cache-root data/raw/himawari_hsd \
  --patch-root data/processed/himawari_mmd_patches \
  --output-csv data/processed/satellite_dataset.csv \
  --bands B08 B13 B15 --segments 5 6 \
  --max-frames 1000 --max-frames-per-day 2 \
  --negative-ratio 1.0 --max-positives-per-frame 50 \
  --negative-min-distance-km 30 --patch-size 64 \
  --max-patch-black-fraction 0.02 --patch-black-threshold 8 \
  --resolution-degrees 0.02 --nowcast-minutes 0 --seed 42
```

2. Validate split integrity, class balance, patch files, and the no-data threshold:

```bash
python src/validate_satellite_dataset.py \
  --manifest data/processed/satellite_dataset.csv \
  --black-threshold 8 --max-black-fraction 0.02 \
  --batch-size 32 --num-workers 0
```

3. Train, select the threshold on validation, and evaluate once on test. `src/train_satellite.py` performs those three stages in that order and writes both the checkpoint and metrics artifact:

```bash
python src/train_satellite.py \
  --config config.yaml \
  --dataset data/processed/satellite_dataset.csv \
  --epochs 50 --batch-size 32 --eval-batch-size 128 \
  --device cpu --freeze-backbone --patience 5 --seed 42 \
  --model-path models/satellite_resnet50_frozen_cpu_clean_best.pth \
  --results-json results/satellite_frozen_cpu_clean_metrics.json \
  --log-dir logs/satellite_frozen_cpu_clean
```

4. Regenerate Figures 5.1–5.8. The script reads the committed metrics schema, generates the example-input grid from the held-out manifest, and reloads the checkpoint for ROC and probability-distribution inference:

```bash
python src/plot_results.py \
  --metrics-json results/satellite_frozen_cpu_clean_metrics.json \
  --checkpoint models/satellite_resnet50_frozen_cpu_clean_best.pth \
  --dataset-csv data/processed/satellite_dataset.csv \
  --output-dir results/figures --batch-size 128
```

Generated outputs under `data/`, `models/`, `logs/`, and `results/figures/` are intentionally gitignored. The committed metrics JSON is the compact numerical record of the final run.

## Data and weights

Raw MMD CSVs are licensed research data and are not redistributable through this repository. Himawari Level-1b imagery is fetched anonymously from the public NOAA S3 archives by Satpy/s3fs; the 2023–2025 aligned run uses Himawari-9 because Himawari-8 moved to standby in December 2022. Downloaded imagery, derived patches, manifests, and model checkpoints are large generated artifacts and are gitignored.

## Troubleshooting

- If the builder reports missing satellite dependencies, install the full pinned environment with `pip install -r requirements.txt`.
- If the builder cannot find MMD records, confirm that files named `raw data all.csv` exist below the path passed to `--lightning-root`.
- If validation reports missing patches, rebuild or restore `data/processed/himawari_mmd_patches/`; the manifest alone is insufficient.
- If plotting reports a missing checkpoint or manifest, regenerate those gitignored artifacts with steps 1–3 above.
- For GPU training, install the matching PyTorch build from the official PyTorch index and pass `--device cuda`; the reported final run used CPU with a frozen backbone.

## Documentation, license, and attribution

The supplied FYP report is the narrative source of truth. Repository cleanup decisions and unverifiable local-artifact items are recorded in [CLEANUP_NOTES.md](CLEANUP_NOTES.md), while superseded prototype documents remain in [docs/archive/](docs/archive/README.md). The report package index is [report/README.md](report/README.md).

The code is released under the [MIT License](LICENSE). MMD data remain subject to their provider’s terms and are not covered by that code license. Himawari imagery should be attributed to the Japan Meteorological Agency and the NOAA public archive used for access.
