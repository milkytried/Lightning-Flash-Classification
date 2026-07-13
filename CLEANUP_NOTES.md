# Cleanup notes

This cleanup aligns the submission repository with Chapters 3–5 of `FYP_Full_Report.docx`. No model was retrained, and no model, data-pipeline, threshold-selection, or evaluation logic was changed.

## Stale claims corrected

- The root README promoted the earlier Himawari-8 experiment and its 11-PNG metrics as the final result. It was rewritten around the aligned Himawari-9 / MMD run and now contains one performance table with the report values only.
- The README’s metadata framing treated the apparent perfect metadata score as an achievement. It now identifies `amplitude` and `strike_type` as circular strike-derived inputs and presents the MLP only as a leakage demonstration/negative result.
- The README project tree omitted nine committed source modules and referenced nonexistent or gitignored documentation. The tree now matches `src/`, the `quick_eval.py` and `_bmad-output/` references are gone, and every reproduction/troubleshooting command names a committed producer or explicitly identified generated artifact.
- `config.yaml` described the aligned run as Himawari-8, called the three infrared channels RGB, and used a 100°E western bound. It now documents Himawari-9, B08/B13/B15, the report counts, and `region_bbox: [99.0, 120.0, -5.0, 15.0]`, matching `MALAYSIA_BOUNDS` in `src/build_satellite_dataset.py`.
- General package/loader descriptions implied that all satellite work used Himawari-8. They now identify Himawari-9 as the final aligned run and explicitly scope Himawari-8 to the historical prototype.
- `report/README.md` indexed old “final” prototype material, source snapshots, and PNG paths that are absent from a fresh clone. It now points to the final README and committed metrics record and marks the checkpoint, manifest, patches, and figures as gitignored generated artifacts.
- The old audit claimed `train_fresh_optimized.py` and local artifacts were available, while status notes also cited `monitor_fresh_training.py`. Those documents were archived with a header stating that their path assertions are historical and may refer to moved or gitignored files.
- The “final summary,” viva summary, and panel notes still called the 11-PNG Himawari-8 run the final FYP result. They were archived and explicitly superseded.
- `src/plot_results.py` generated only Figures 5.2–5.8 and defaulted to the pre-correction artifact names. It now generates the held-out example grid for Figure 5.1, uses the final clean-run defaults, and still reloads the checkpoint for ROC/probability inference. A local full run regenerated all eight figures and reproduced ROC-AUC 0.968127 from the checkpoint, matching the JSON.
- Unit tests instantiated pretrained ResNet-50 weights by default. Tests now pass `pretrained=False` so a fresh clone does not need a weight download. Data-dependent metadata/ingestion tests retain explicit skip guards.
- The CI comment still described skip guards as a handoff item. It now states the implemented offline/artifact policy.
- The README had a license heading without a license. An MIT `LICENSE` was added.
- The final numerical artifact was gitignored. `results/satellite_frozen_cpu_clean_metrics.json` is now the committed traceable record; checkpoints and generated figures remain ignored.

## Files moved

- `FRESH_TRAINING_STATUS.md` → `docs/archive/FRESH_TRAINING_STATUS.md` — earlier prototype training status.
- `SATELLITE_MODEL_FRESH_REPORT.md` → `docs/archive/SATELLITE_MODEL_FRESH_REPORT.md` — earlier prototype evaluation.
- `TRAINING_FAILURE_DIAGNOSIS.md` → `docs/archive/TRAINING_FAILURE_DIAGNOSIS.md` — failed training diagnosis preceding the prototype.
- `report/audit/FINAL_AUDIT.md` → `docs/archive/FINAL_AUDIT.md` — audit of the 11-PNG baseline.
- `report/final_summary/FINAL_PROJECT_SUMMARY.md` → `docs/archive/FINAL_PROJECT_SUMMARY.md` — superseded prototype summary.
- `report/final_summary/FYP_VIVA_SUMMARY.md` → `docs/archive/FYP_VIVA_SUMMARY.md` — superseded prototype viva notes.
- `report/qa_prep/PANEL_QA_PREP.md` → `docs/archive/PANEL_QA_PREP.md` — superseded prototype panel notes.
- `eval_test_fresh.py` → `scripts/eval_test_fresh.py` — retained producer for the earlier baseline evaluation artifact.
- `tune_threshold.py` → `scripts/tune_threshold.py` — retained producer for the Table 5.2 baseline threshold/metrics.
- `retrain_honest_artifacts.py` → `scripts/retrain_honest_artifacts.py` — retained metadata leakage/honest-comparator generator.
- `complete_fresh_training.py` → `scripts/archive/complete_fresh_training.py` — dead prototype orchestrator superseded by `src/train_satellite.py` and `src/plot_results.py`.
- `generate_metadata_fresh.py` → `scripts/archive/generate_metadata_fresh.py` — superseded prototype metadata generator.
- `demo_inference.py` → `scripts/archive/demo_inference.py` — metadata-MLP demo unrelated to the final satellite result.

No tracked source or audit file was silently deleted. Locally ignored logs, caches, datasets, checkpoints, and scratch files were left untouched because they are not part of a fresh clone.

## Verification and explicit limitations

Verified:

- The final full local suite passed with 63 tests passed and 1 artifact-dependent skip.
- All eight result figures were regenerated locally from `results/satellite_frozen_cpu_clean_metrics.json`, `models/satellite_resnet50_frozen_cpu_clean_best.pth`, and `data/processed/satellite_dataset.csv`. Checkpoint inference recomputed ROC-AUC as 0.968127, identical to the JSON value.
- The exact pinned requirements resolve for Python 3.10, 3.11, and 3.12 with the package indexes configured in `requirements.txt`. A pip Python 3.12 dry run also resolved every package, including `torch==2.12.1+cpu`, `torchvision==0.27.1+cpu`, Satpy, pyresample, s3fs, and boto3. No nonexistent pin was found.
- A clean local clone at commit `a97357d` successfully completed `pip install -r requirements.txt` under Python 3.12, then ran `pytest tests -q` with 58 tests passed and 6 expected artifact-dependent skips. No data or model weights were present.

Could not verify from a fresh clone alone:

- Raw MMD CSV contents, strike counts, and licensing constraints beyond the supplied report, because the MMD source data are non-redistributable and gitignored.
- Dataset reconstruction byte-for-byte without those MMD CSVs and the downloaded NOAA cache.
- Training reproduction or checkpoint byte identity without rerunning training; retraining was explicitly prohibited.
- Figure/metric checkpoint inference on another machine without the gitignored checkpoint and derived dataset. The workflow was verified against the local artifacts, but a fresh clone correctly lacks them.
- Whether the university mandates a license other than MIT; no such requirement was present in the repository or supplied brief, so the requested MIT default was used.

## Figure 5.1 selection correction

The original Figure 5.1 helper used the first four test-manifest rows per class. Because the manifest is grouped by source frame, this selected near-duplicate lightning patches from one 10-minute scene. The reporting code now deduplicates each class by the manifest's `frame_id`, uses `timestamp` to choose evenly spaced frames across the test period, and uses fixed seed 42 to select one reproducible patch within each chosen frame. It reuses the existing checkpoint inference probabilities for panel annotations and writes the exact paths, frame timestamps, probabilities, threshold outcomes, and seed to `results/figures/example_input_patches_selection.json`. An independent checkpoint reload and one-image inference pass over each of the eight selected paths verified the indexed probability mapping to within 1e-6 (maximum absolute error 6.26e-7). No dataset, model, threshold, or evaluation logic changed.

**Report action — replace or extend the Figure 5.1 caption:** “Patches are an unbiased seed-42 sample of distinct held-out test frames, not curated examples. One of the four lightning patches falls below the 0.51 decision threshold, consistent with the reported test recall of 0.957.”

**Report action — revise Section 5.8:** checkpoint inference on all 2,309 positive test patches gives a median probability of 0.730797, an interquartile range of 0.647693–0.796086, and only 0.012126 (1.21%) above 0.9. The positive-class distribution is therefore not concentrated near 1.0; the report should not describe its positive mode as being near one.

## Threshold-sensitivity analysis (analysis only)

No threshold, model, or reported evaluation metric was changed. The operating threshold remains 0.51, selected once by validation F1 and then applied to the held-out test set. Read-only saved-checkpoint inference was used to compute recall/POD, precision, FAR, and F1 from 0.30 to 0.80 in 0.01 steps; the complete test and validation tables are in results/threshold_sensitivity_test.json, and the new report figure is results/figures/threshold_sensitivity.png. FAR retains the project definition FP / (TP + FP).

**Report action - correct Section 5.8:** the two classes are well separated in rank order (test ROC-AUC 0.968127), but confidence is compressed rather than bimodal at zero and one. Positive test probabilities have median 0.730797, IQR 0.647693-0.796086, and only 1.21% exceed 0.9. Negative test probabilities have median 0.236193, IQR 0.127930-0.401550, and only 18.10% fall below 0.1. There is no positive mode near 1.0. The frozen ResNet-50 backbone is the likely cause of this compressed confidence because only the classification head could adapt to the Himawari patch domain; this is an interpretation, not a separately tested causal result.

At threshold 0.51, test recall/POD is 0.956691 and F1 is 0.913565, exactly reproducing the reported 0.9567 and 0.9136. Test recall falls to 0.854049 at 0.60, 0.743612 at 0.65, and 0.599394 at 0.70; on the requested 0.01 grid, 0.57 is the first threshold where test recall is at or below 0.90. Validation recall/F1 at 0.51 are 0.939832/0.885237; validation recall is 0.795427 at 0.60, 0.697353 at 0.65, and 0.569194 at 0.70, with its first recall-at-or-below-0.90 threshold at 0.55.

The 0.51 operating point is not a local F1 knife-edge: it is the maximum-F1 point on the 0.01 grid for both validation and test, and nearby validation F1 is 0.884464 at 0.50, 0.885237 at 0.51, 0.885208 at 0.52, and 0.884915 at 0.53. However, recall is threshold-sensitive once the cutoff is raised because both class distributions occupy compressed, overlapping probability ranges. This sensitivity analysis is descriptive and does not retune on the test set.

## Dataset independence and effective sampling units

The manifest contains 41,168 patches from 1,000 distinct Himawari-9 source frames across 264 UTC dates. By split, train contains 729 frames across 189 dates, validation 150 frames across 44 dates, and test 121 frames across 31 dates. All split totals reconcile to the overall counts, and no source frame crosses a split.

Across the full manifest, patches per frame have median 22, IQR 6-88, and maximum 100. The 20,584 positive patches come from all 1,000 source frames; positives per positive frame have median 11, IQR 3-44, and maximum 50. Per split, positives per frame are: train median 16, IQR 4-47, maximum 50; validation median 3, IQR 1-12.5, maximum 50; and test median 9, IQR 3-40, maximum 50.

Using haversine DBSCAN separately within each frame (20 km radius, min_samples=1) gives 2,754 spatial clusters overall: 2,227 train, 261 validation, and 266 test. This is a coarse proxy for spatially distinct convective events within each 10-minute frame, not a formal autocorrelation-adjusted effective sample size and not storm tracking across consecutive frames. The reproducible calculation and all split-level distributions are recorded in results/dataset_independence.json.

**Viva headline:** "The 20,584 positive patches derive from 1,000 distinct source frames across 264 dates, representing approximately 2,754 spatially distinct within-frame convective clusters at a 20 km DBSCAN scale."
