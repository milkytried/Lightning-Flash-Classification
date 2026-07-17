# Version 2 Model Card

**Permanent label:** Version 2 ? Frozen Corrected Scientific Experiment  
**Selected model:** `small_cnn_seed2026_bce_pos_weight_train_split_none`  
**Checkpoint:** `models\v2\phase3\small_cnn_seed2026_bce_pos_weight_train_split_none_best.pth`  
**Checkpoint SHA-256:** `888696cb7f6d1543875795fca0deec2aaf5b0e54157692633b619e17f216ce1a`

## Intended Use
Research classification of MMD-recorded cloud-to-ground lightning associations from Himawari-9 image patches within a conservative empirical Peninsular Malaysia study region.

## Out-of-Scope Uses
Operational warning, real-time deployment, proof of physical lightning absence, general Malaysia-wide claims, nowcasting, or use outside the empirical study region.

## Architecture and Inputs
- Architecture: compact three-block CNN with batch normalization, global pooling and raw-logit output.
- Input channels: B08, B13, B15 in that order, encoded as RGB patch channels.
- Input shape: 3 ? 64 ? 64.
- Preprocessing: uint8 patch scaled to [0,1], then small-CNN normalization mean/std `[0.5,0.5,0.5]`.
- Loss: `BCEWithLogitsLoss(pos_weight=...)`, with `pos_weight` calculated from the training split only.
- Frozen threshold: `0.830726981` selected on validation F1.
- Calibration: validation-only temperature scaling, temperature `1.070659041`. Main classifications use the frozen validation threshold on uncalibrated probabilities; calibrated probabilities are reported separately.

## Metrics
| Split | Accuracy | ROC-AUC | PR-AUC | Precision | Recall/POD | FPR | FAR/FDR | TN | FP | FN | TP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Controlled test | 0.955556 | 0.983538 | 0.966210 | 0.928571 | 0.908971 | 0.026673 | 0.071429 | 1934 | 53 | 69 | 689 |
| Natural prevalence | 0.911111 | 0.948159 | 0.896222 | 0.922566 | 0.692691 | 0.018687 | 0.077434 | 1838 | 35 | 185 | 417 |

## Subgroup Behaviour
Natural-prevalence performance is conservative: high precision and low FPR, but 185 of 602 recorded-positive patches were missed. Active-frame results are stronger than zero-recorded false-alarm analysis, and subgroup metrics are in `report/V2_PHASE3_NATURAL_PREVALENCE.json`.

## Limitations
Residual geographic/time predictability remains; MMD zero-recorded does not prove no lightning; storm groups are derived, not official; the study mask is empirical; raw MMD data are not redistributed; no physical mechanism has been proven.

## Inference
Use `python src/v2_inference.py <patch.png>`. The script verifies the checkpoint hash, uses shared V2 preprocessing, preserves B08/B13/B15 order, returns raw logit, probability, calibrated probability and frozen-threshold classification, and fails loudly for missing/corrupt/wrong-sized inputs.
