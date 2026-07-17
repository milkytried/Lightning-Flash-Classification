# Repository Artifact Classification

- Current Version 2 production/research pipeline: `configs/v2_full.yaml`, `configs/v2_training.yaml`, `src/v2_*`, `src/create_mmd_inventory.py`, `src/mmd_spatiotemporal_index.py`, `tests/test_v2_*`.
- Version 1 frozen experiment: `src/build_satellite_dataset.py`, `src/train_satellite.py`, `src/plot_results.py`, `docs/version1_figure_provenance.md`, retained V1 reports.
- Historical legacy code: older metadata and prototype scripts such as `train.py`, `train_lightning.py`, `lightning_model.py`, and archived documents.
- Generated artifacts: `data/processed/`, `models/`, `results/`, `logs/`, `*_v2_phase3.log`, figures and prediction CSVs.
- Licensed/private data: raw MMD files under `data/raw/`; do not commit or redistribute.
- Temporary/quarantined execution output: `results/v2/phase3/quarantine/`; retained locally for audit, ignored by Git.

No large destructive cleanup was performed in the final consolidation phase.
