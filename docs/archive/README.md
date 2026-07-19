> ⚠️ SUPERSEDED — retained for provenance only. Not the final result. See README.md and report/ for Version 2.

# Superseded experiment archive

These files preserve historical status notes for earlier prototypes and failed/intermediate training attempts. They are **not** current reproduction instructions and must not be cited as the final FYP result.

The current result is documented in the root `README.md`, `docs/version2_model_card.md`, `docs/version2_dataset_card.md`, and `report/FINAL_PROJECT_SUMMARY_V2.md`.

Current final label: **Version 2 — Frozen Corrected Scientific Experiment**.

Current final model: compact CNN trained from scratch on Himawari-9 B08/B13/B15 patches, 102,017 trainable parameters, selected run `small_cnn_seed2026_bce_pos_weight_train_split_none`.

Current final test results:

- Controlled test: accuracy 0.9556, ROC-AUC 0.9835, PR-AUC 0.9662.
- Natural-prevalence test: accuracy 0.9111, ROC-AUC 0.9482, PR-AUC 0.8962.

Archived files:

- `FINAL_AUDIT.md` — older Version 1/prototype audit material.
- `FINAL_PROJECT_SUMMARY.md` — superseded project summary.
- `FRESH_TRAINING_STATUS.md` — historical training status snapshot.
- `FYP_VIVA_SUMMARY.md` — superseded viva notes.
- `PANEL_QA_PREP.md` — superseded question-and-answer preparation.
- `SATELLITE_MODEL_FRESH_REPORT.md` — historical satellite model report.
- `TRAINING_FAILURE_DIAGNOSIS.md` — historical training-failure diagnosis.

Version 1 remains useful only as a reproducible diagnostic baseline. It is scientifically limited by geographic/sampling bias, active-frame-only selection, and no-data contamination discovered during audit.
