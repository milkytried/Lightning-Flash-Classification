# Version 1 Figure Provenance

Label: **Version 1 ? Frozen Reproducible Diagnostic Experiment**

This record preserves provenance for the ignored Version 1 figure PNGs. The PNGs remain out of Git; this file records the command, source artifact hashes, threshold, generation timestamps, and figure hashes needed to reproduce/audit them.

## Scientific Status

Computational conclusion: Version 1 metrics are authentic and exactly reproducible from the saved checkpoint at threshold `0.51`.

Scientific conclusion: Version 1 sampling allows strong shortcuts, so the metrics do not establish general or operational lightning-detection performance.

Recommended wording:

> On the frozen Version 1 balanced patch dataset, the saved model reproducibly achieved 90.95% accuracy and 0.9681 ROC-AUC. Subsequent auditing identified geographical and sampling shortcuts, so these results are retained as a diagnostic benchmark rather than operational lightning-detection evidence.

## Metric Naming

- FAR is `FP / (TP + FP)`, also called false discovery ratio.
- FPR is `FP / (FP + TN)`.
- FAR and FPR must not be used interchangeably.

## Figure Generation

Primary result-figure command:

```powershell
..\.venv\Scripts\python.exe src\plot_results.py --metrics-json results\satellite_frozen_cpu_clean_metrics.json --checkpoint models\satellite_resnet50_frozen_cpu_clean_best.pth --dataset-csv data\processed\satellite_dataset.csv --output-dir results\figures
```

Example-input figure generation: regenerated in the same verification session from clean held-out TEST patches only, selecting three positive and three negative `64x64` Himawari-9 infrared patches with black fraction <= `0.02`. The selected negative examples all had black fraction `0.000000`.

## Source Revisions

- Current provenance-record commit base: `dcc978deecd974428750a4e074755ca0873873ae`
- `src/plot_results.py` last-touch commit: `7188297b9102af98c82c87d0ca0e5887168712ad`
- `src/plot_results.py` SHA-256: `3176a202c1cb4f28ce487434d147c0f64ce64bac50f067ea1ee928122fa48464`

## Frozen Source Artifact Hashes

| Artifact | SHA-256 |
|---|---|
| `models/satellite_resnet50_frozen_cpu_clean_best.pth` | `b8607ae8234de256f1e6b17a72a0ffac9b4aca12ad364cb4ddc6e44acabe3d63` |
| `data/processed/satellite_dataset.csv` | `cfe36ad8084143d0593320b76206e8a16242767d71f135f7d6350895cd0a7148` |
| `results/satellite_frozen_cpu_clean_metrics.json` | `70d8483f21859002aa7b20b1b12ed2376c13ec6205254fbd3a3318d2aee3f8b1` |

## Threshold

- Decision threshold: `0.51`

## Figure Files

| Figure | Size bytes | Last write UTC | SHA-256 |
|---|---:|---|---|
| `baseline_comparison.png` | 101446 | 2026-07-13T07:22:32Z | `8a4fa9553716ec52a47f168bdcec6704539f7dee40e3968fda6669b6cc91798b` |
| `confusion_matrix.png` | 88420 | 2026-07-13T07:22:32Z | `ca60afda0378aebba7e957bad038dcbdd5915c2580ce075bd8907123db58d368` |
| `example_input_patches.png` | 176313 | 2026-07-13T07:23:00Z | `9d25c458673475584c058fb47f7502db0c8bba087243d14b8fc08c1d86a62287` |
| `meteorological_metrics.png` | 65021 | 2026-07-13T07:22:32Z | `afec5a1712016859044b4fef85c074b170c933fd0eaf1de5e2431e0cb3ec66b6` |
| `probability_histogram.png` | 92373 | 2026-07-13T07:23:01Z | `471a8b769c7a9128a29dfa55e28e64e41ad6f8be2af08db6151c72d1815d97f4` |
| `roc_curve.png` | 122408 | 2026-07-13T07:23:01Z | `1850420b58e35acc6e2a4b78eeb4295df38c64e40c3bf3a825a9cff48d16929b` |
| `threshold_sensitivity.png` | 149630 | 2026-07-13T07:23:25Z | `d16d4ef0599b5f27673ba2ca658fd923aafad8b084642085f8062bcb7483afd2` |
| `training_curves.png` | 153749 | 2026-07-13T07:22:31Z | `c03007e188659558912ee5020a6c3af772997469bcee2288aefd6d570bd7365c` |
| `validation_metrics.png` | 81404 | 2026-07-13T07:22:32Z | `575077ff574ae3d536efcab8673ba2583b5ed256cee0b8d94cc67688db277abf` |

## Verification Snapshot

- Accuracy: `0.9094846254`
- ROC-AUC: `0.9681270551`
- Confusion matrix: TN `1991`, FP `318`, FN `100`, TP `2209`
- Trainable parameters: `262401`
- Frozen parameters: `23508032`
- Dataset rows: `41168`
- Dataset status: balanced, date-disjoint, no missing patch paths, no black replacement patch leakage above the `0.02` threshold.

## Version 2 Boundary

Version 1 test results, predictions, and errors must not be used to alter Version 2 sampling rules, study mask, temporal tolerance, split boundaries, frame selection, thresholds, or negative matching. Version 2 decisions must use only Version 2 training and validation data until the Version 2 pipeline is frozen.
