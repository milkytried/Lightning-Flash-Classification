# Lightning flash classification from Himawari-9 imagery

This repository contains the code, audit records, and report support material for a final-year project on cloud-to-ground lightning association from Himawari-9 infrared image patches over a conservative Peninsular Malaysia study region.

The project has two clearly separated experiments:

- **Version 1 — Frozen Reproducible Diagnostic Experiment**: an aligned balanced-patch prototype that is computationally reproducible, but scientifically limited by sampling shortcuts discovered during audit.
- **Version 2 — Frozen Corrected Scientific Experiment**: the corrected, preregistered scientific experiment and the main result of this repository.

Version 2 is the headline experiment. Version 1 is retained for comparison, audit history, and final-report discussion only.

## Final Claim

Version 2 supports the following bounded conclusion:

> Version 2 demonstrates meaningful image-based discrimination of MMD-recorded cloud-to-ground lightning associations from Himawari-9 image patches within a conservative empirical Peninsular Malaysia study region.

It is **not** an operational warning system, not a real-time nowcaster, and not evidence of lightning absence outside the sampled design. Version 2 decisions must not be changed using Version 2 test outcomes.

## Version 2 Selected Model

The frozen selected model is:

- Run: `small_cnn_seed2026_bce_pos_weight_train_split_none`
- Checkpoint: `models/v2/phase3/small_cnn_seed2026_bce_pos_weight_train_split_none_best.pth`
- Checkpoint SHA-256: `888696cb7f6d1543875795fca0deec2aaf5b0e54157692633b619e17f216ce1a`
- Decision threshold: `0.8307269811630249`, selected on validation only
- Calibration: validation temperature scaling, `1.0706590414047241`
- Architecture: small CNN trained from scratch on Version 2 patches

The checkpoint, datasets, prediction CSVs, logs, and figures are generated artifacts and are intentionally gitignored.

## Headline Results

| Evaluation | Accuracy | ROC-AUC | PR-AUC | Precision | Recall / POD | FAR / FDR | Confusion matrix |
|---|---:|---:|---:|---:|---:|---:|---|
| Controlled balanced test | 0.9556 | 0.9835 | 0.9662 | 0.9286 | 0.9090 | 0.0714 | TN 1934, FP 53, FN 69, TP 689 |
| Natural-prevalence evaluation | 0.9111 | 0.9482 | 0.8962 | 0.9226 | 0.6927 | 0.0774 | TN 1838, FP 35, FN 185, TP 417 |

Metric naming is explicit: FAR here means `FP / (TP + FP)`, also called false discovery ratio. FPR means `FP / (FP + TN)`. The two are not interchangeable.

## Version 1 Versus Version 2

| Experiment | Status | Main result | Scientific interpretation |
|---|---|---|---|
| Version 1 — Frozen Reproducible Diagnostic Experiment | Frozen and retained | 90.95% accuracy, 0.9681 ROC-AUC on the frozen balanced patch dataset | Computationally authentic, but sampling shortcuts mean it is a diagnostic benchmark rather than operational lightning-detection evidence. |
| Version 2 — Frozen Corrected Scientific Experiment | Final scientific result | 95.56% controlled accuracy, 0.9835 controlled ROC-AUC; 91.11% natural accuracy, 0.9482 natural ROC-AUC | Meaningful bounded image-based discrimination within the corrected empirical design. |

Recommended wording for Version 1 is preserved:

> On the frozen Version 1 balanced patch dataset, the saved model reproducibly achieved 90.95% accuracy and 0.9681 ROC-AUC. Subsequent auditing identified geographical and sampling shortcuts, so these results are retained as a diagnostic benchmark rather than operational lightning-detection evidence.

## Model Comparison Evidence

The final Version 2 model was selected before test unlock using validation PR-AUC. The small CNN was stable across seeds and clearly exceeded tabular baselines:

| Model family | Validation PR-AUC evidence |
|---|---:|
| Small CNN, seeds 2024/2025/2026 | 0.9550 / 0.9579 / 0.9588 |
| ResNet-18, seeds 2024/2025/2026 | 0.8614 / 0.8611 / 0.8641 |
| B13-only random forest baseline | 0.8542 |
| Geography/time random forest baseline | 0.6498 PR-AUC, 0.8170 ROC-AUC |

The selected small CNN result is therefore not a single lucky seed and is not explained by geography/time metadata alone.

## Repository Map

Key source files:

- `src/build_v2_dataset.py` and related Version 2 utilities: corrected dataset construction.
- `src/v2_phase3_train.py`: preregistered Version 2 Phase 3 training, validation selection, test unlock, and final reporting.
- `src/v2_inference.py`: frozen Version 2 inference entry point for one patch.
- `src/v2_final_repro_audit.py`: independent reproducibility audit from saved prediction CSVs.
- `src/v2_final_figures.py`: final figure generation from frozen artifacts.
- `tests/`: offline-safe tests, including the Version 2 inference consistency test.

Key documentation:

- `docs/version2_experiment_provenance.md`: hashes, commits, commands, and final selected artifact provenance.
- `docs/version2_model_card.md`: model card for the selected Version 2 model.
- `docs/version2_dataset_card.md`: dataset card for the corrected Version 2 dataset.
- `docs/version2_figure_provenance.md`: final figure-generation provenance and figure hashes.
- `docs/repository_artifact_classification.md`: what is current, frozen, legacy, generated, or private.
- `report/FINAL_PROJECT_SUMMARY_V2.md`: final project summary.
- `report/V2_FINAL_REPRODUCIBILITY_AUDIT.md`: independent metric recomputation audit.
- `report/V2_TEST_LOCK_CHRONOLOGY_AUDIT.md`: audit of validation selection before test unlock.
- `report/FINAL_DEFENCE_QA_V2.md`: viva/defence question-and-answer preparation.

## Reproducibility

A fresh clone contains code, tests, and documentation. It does not contain licensed/private MMD lightning CSVs, downloaded Himawari files, derived patches, checkpoints, logs, or generated result figures.

With the required local data/artifacts available, the important non-training verification commands are:

```powershell
..\.venv\Scripts\python.exe src\v2_final_repro_audit.py
..\.venv\Scripts\python.exe src\v2_inference.py <path-to-64x64-v2-patch.png>
..\.venv\Scripts\python.exe -m pytest
```

The official inference path loads the frozen selected checkpoint, verifies its SHA-256 by default, applies the committed validation temperature and threshold, and reports both raw and calibrated probabilities.

## Data And Artifact Policy

Raw MMD CSVs are licensed research data and are not redistributed. Himawari imagery is public archive data, but downloaded HSD files and derived patches are large generated artifacts. The following are intentionally uncommitted:

- `data/`
- `models/`
- `results/v2/`
- `logs/`
- `*.log`

Committed reports include hashes and provenance for frozen local artifacts so the results can be audited without committing the large files themselves.

## License And Attribution

The code is released under the MIT License. MMD data remain subject to their provider's terms. Himawari imagery should be attributed to the Japan Meteorological Agency and the NOAA public archive used for access.
