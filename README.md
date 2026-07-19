# Lightning flash classification from Himawari-9 imagery

**Status:** Version 2 is final. Last updated: 2026-07-19.

This repository contains the final code and audit trail for **Version 2 — Frozen Corrected Scientific Experiment**, a final-year project on satellite-image classification of Malaysian Meteorological Department (MMD) cloud-to-ground lightning associations from Himawari-9 AHI infrared patches.

**Bounded claim:** the final system demonstrates meaningful image-based discrimination of MMD-recorded cloud-to-ground lightning associations from Himawari-9 image patches within a conservative empirical Peninsular Malaysia study region. It is **not** an operational warning system, not a real-time nowcaster, and not proof of physical lightning absence where MMD recorded no strike.

## Final Model

The final model is a compact CNN trained from scratch. It is **not** the earlier frozen ResNet-50 and it is **not** a metadata MLP.

| Item | Final Version 2 value |
|---|---|
| Selected run | `small_cnn_seed2026_bce_pos_weight_train_split_none` |
| Checkpoint | `models/v2/phase3/small_cnn_seed2026_bce_pos_weight_train_split_none_best.pth` |
| Checkpoint SHA-256 | `888696cb7f6d1543875795fca0deec2aaf5b0e54157692633b619e17f216ce1a` |
| Architecture | 3 convolution blocks, batch normalization, global pooling, dropout, 64-unit hidden layer, 1-logit output |
| Parameters | 102,017 total; all trainable |
| Input | 64x64 RGB PNG patch where channels encode Himawari-9 AHI B08/B13/B15 |
| Loss | `BCEWithLogitsLoss(pos_weight=...)`, with `pos_weight` calculated from the training split |
| Optimizer | AdamW, learning rate `1e-3`, weight decay `1e-4` |
| Scheduler | ReduceLROnPlateau on validation PR-AUC, factor `0.5`, patience `3` |
| Training | max 50 epochs, early stopping patience 7; selected checkpoint at epoch 25 of a 32-epoch run |
| Threshold | `0.8307269811630249`, selected on validation F1 and then frozen |
| Calibration | validation-only temperature scaling (`T = 1.0706590414047241`) reported separately; class decisions use the frozen threshold |

The small CNN was selected using validation performance before test unlock. It was stable across seeds and outperformed the frozen ResNet-50 candidate family and simple baselines on validation PR-AUC.

## Corrected Version 2 Data Pipeline

Version 2 replaced the earlier balanced prototype with a corrected sampling design:

- Himawari-9 AHI Level-1b imagery, bands B08/B13/B15, converted into 64x64 patches.
- Conservative empirical Peninsular Malaysia study mask rather than a broad Malaysia-wide claim.
- Positive labels from MMD-recorded cloud-to-ground strike associations.
- Negative labels require no MMD-recorded strike in the full crop neighbourhood under the frozen temporal exclusion window `[t-20m, t+30m)`.
- Active and zero-recorded frames are both included.
- Train/validation/test are date- and storm-disjoint chronological splits.
- No-data/scan-edge patches are rejected.
- A separate natural-prevalence test keeps the observed class imbalance instead of forcing 1:1 balancing.

Dataset composition from the frozen manifests:

| Split | Rows | Negatives | Positives | Positive base rate | Date range |
|---|---:|---:|---:|---:|---|
| Train | 9,204 | 6,035 | 3,169 | 0.3443 | 2023-01-01 to 2024-04-01 |
| Validation | 2,612 | 1,983 | 629 | 0.2408 | 2025-01-01 to 2025-02-28 |
| Controlled test | 2,745 | 1,987 | 758 | 0.2761 | 2025-03-01 to 2025-04-01 |
| Natural-prevalence test | 2,475 | 1,873 | 602 | 0.2432 | 2025-03-01 to 2025-04-01 |

Raw MMD data, downloaded Himawari files, derived patches, checkpoints, logs, and generated figures are intentionally not committed.

## Final Results

All metrics below are recomputed from the frozen prediction artifacts at the validation-selected threshold `0.8307269811630249`.

| Evaluation | Rows | Accuracy | ROC-AUC | PR-AUC | Precision | Recall / POD | FAR / FDR | Confusion matrix |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Controlled test | 2,745 | 0.9556 | 0.9835 | 0.9662 | 0.9286 | 0.9090 | 0.0714 | TN 1934, FP 53, FN 69, TP 689 |
| Natural-prevalence test | 2,475 | 0.9111 | 0.9482 | 0.8962 | 0.9226 | 0.6927 | 0.0774 | TN 1838, FP 35, FN 185, TP 417 |

The natural-prevalence result is deliberately more conservative. Precision remains high (`0.9226`) and false alarms are low (`35` FP out of `1,873` recorded negatives), but recall drops from `0.9090` on the controlled test to `0.6927` under natural prevalence. That drop is part of the scientific result and should not be hidden.

Metric naming is explicit: FAR here means `FP / (TP + FP)`, also called false discovery ratio. FPR means `FP / (FP + TN)`. They are not interchangeable.

## Version 1 Diagnostic Baseline

Version 1 is retained only as **Version 1 — Frozen Reproducible Diagnostic Experiment**. It used a frozen ResNet-50 and reproducibly achieved about `0.9095` accuracy and `0.9681` ROC-AUC on its cleaned balanced patch dataset. Subsequent audits found scientific limitations, including geographic/sampling shortcuts, active-frame-only selection, centre-only negative exclusion, no-data contamination in earlier negatives, and no natural-prevalence evaluation. Version 1 is therefore an audit baseline, not the final FYP result.

Older metadata experiments are also not headline results. Apparent perfect metadata scores are consistent with label leakage because fields such as amplitude and strike type only exist once a strike has been recorded. They are preserved, if at all, as historical experiments and are not used as FYP evidence.

## Reproducibility

The compact verification path assumes the local, gitignored V2 artifacts are present:

```powershell
# Recompute final metrics from saved prediction CSVs
..\.venv\Scripts\python.exe src\v2_final_repro_audit.py

# Run official inference on one 64x64 V2 patch
..\.venv\Scripts\python.exe src\v2_inference.py <path-to-patch.png>

# Regenerate report figures from the checkpoint and manifests
..\.venv\Scripts\python.exe figures\make_figures.py

# Run tests
..\.venv\Scripts\python.exe -m pytest
```

Key provenance records:

- `docs/version2_experiment_provenance.md`
- `docs/version2_model_card.md`
- `docs/version2_dataset_card.md`
- `docs/version2_figure_provenance.md`
- `report/V2_FINAL_REPRODUCIBILITY_AUDIT.md`
- `report/V2_TEST_LOCK_CHRONOLOGY_AUDIT.md`
- `report/FINAL_PROJECT_SUMMARY_V2.md`

Generated outputs under `data/`, `models/`, `results/v2/`, `figures/*.png`, `logs/`, and `*.log` are excluded from git. This keeps the repository lightweight while preserving hashes and provenance for local frozen artifacts.

## Repository Layout

```text
src/        Version 2 builders, training/evaluation utilities, inference, and audits
tests/      Offline-safe regression tests
docs/       Current V2 model/dataset/provenance cards and historical archive
report/     Final V2 reports and phase summaries
figures/    Regenerable report-figure script and local PNG outputs
```

Historical reports in `docs/archive/` document superseded experiments and must not be read as current reproduction instructions.

## License And Attribution

The code is released under the MIT License. MMD lightning data remain subject to their provider's terms and are not redistributed. Himawari imagery should be attributed to the Japan Meteorological Agency and the NOAA public archive used for access.
