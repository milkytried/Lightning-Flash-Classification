# V2 Final Reproducibility Audit

Verdict: **CONFIRMED**. All frozen prediction CSV metrics match the committed Phase 3 reports within numerical tolerance.

## Selected Model

| Split | Samples | Accuracy | ROC-AUC | PR-AUC | TN | FP | FN | TP | FAR/FDR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Controlled test | 2745 | 0.955556 | 0.983538 | 0.966210 | 1934 | 53 | 69 | 689 | 0.071429 |
| Natural prevalence | 2475 | 0.911111 | 0.948159 | 0.896222 | 1838 | 35 | 185 | 417 | 0.077434 |

## Integrity Checks

- Controlled-test manifest rows: `2745`; every prediction file maps one-to-one by path.
- Natural-prevalence manifest rows: `2475`; every prediction file maps one-to-one by path.
- No duplicate prediction path rows were found.
- Every threshold equals the frozen value in `report/V2_PHASE3_TEST_UNLOCK.json`.
- Sigmoid was applied exactly once: `sigmoid(logit)` reproduces saved probabilities.
- Calibration parameters were frozen in the unlock record; no test-time calibration fitting was performed.
- FAR is reported as false discovery ratio `FP / (TP + FP)`; FPR is `FP / (FP + TN)`.
