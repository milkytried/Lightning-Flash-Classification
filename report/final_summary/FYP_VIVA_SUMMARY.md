# FYP Viva Summary - Himawari-8 Satellite CNN

## 1-Minute Explanation

My final FYP result is a Himawari-8 satellite CNN that classifies lightning using image pixels only. The main contribution is a clean, chronological satellite experiment with 11 source PNGs, split into 6 training PNGs from 2025-04-18 and 5 validation/test PNGs from 2025-04-22. Those 11 PNGs produced 481,356 total 64x64 patches. The final satellite model achieved 87.65% accuracy, 86.01% precision, 89.93% recall, 0.8792 F1, and 0.9199 ROC-AUC at a tuned threshold of 0.55.

The metadata model is kept only as a baseline and leakage lesson learned. It is not a deployment-safe result because amplitude and strike_type are strike-derived features.

## 3-Minute Explanation

This project asks whether lightning can be detected from Himawari-8 satellite imagery instead of using strike-derived metadata. The final answer is yes, the satellite CNN is the main validated result. I used a chronological split so training data came from 2025-04-18 and validation/test data came from 2025-04-22. That prevents the model from seeing the same source PNG in multiple splits and reduces temporal leakage.

Each source PNG was converted into many 64x64 image patches. Across 11 source PNGs, the pipeline generated 481,356 patches in total. The model architecture is a ResNet-50 backbone with the backbone frozen and a custom classifier head trained on the satellite patches. This freeze-the-backbone strategy made CPU training feasible.

The final operating point was selected using validation data only. I tuned the threshold to 0.55, then reported test metrics on completely unseen data. At that threshold, the model achieved strong ranking performance and a balanced precision-recall trade-off. The metadata model is still included, but only as an honest baseline and a lesson about circular features. Amplitude and strike_type are not valid deployment inputs because they are consequences of a detected strike, not independent predictors.

## Problem Statement

Can a CNN trained only on Himawari-8 satellite pixels detect lightning reliably enough to be the main research contribution for the FYP?

## Objective

Build and evaluate a satellite-only lightning classification model, then compare it with a leakage-aware metadata baseline.

## Methodology

1. Collect Himawari-8 satellite PNGs and lightning labels.
2. Build 64x64 patches from the source PNGs.
3. Split data chronologically by source PNG, not randomly by patch.
4. Train a ResNet-50 based CNN with the backbone frozen.
5. Tune the decision threshold on validation data only.
6. Report final test metrics on unseen data.

## Dataset

- Source data: 11 Himawari-8 satellite PNGs
- Training PNGs: 6 files from 2025-04-18
- Validation PNGs: 3 files from 2025-04-22
- Test PNGs: 2 files from 2025-04-22
- Total patches: 481,356
- Patch size: 64x64
- Scope: Malaysia-only satellite imagery

## Model Architecture

- Backbone: ResNet-50 pre-trained on ImageNet
- Strategy: Frozen backbone, trainable classification head
- Input: 64x64 RGB patches
- Output: Binary lightning probability
- Loss: Focal Loss
- Optimizer: Adam

## Final Metrics

| Metric | Value |
|---|---:|
| Accuracy | 87.65% |
| Precision | 86.01% |
| Recall / POD | 89.93% |
| F1-score | 0.8792 |
| ROC-AUC | 0.9199 |
| FAR | 13.99% |
| CSI | 0.7845 |
| TSS | 0.7530 |
| HSS | 0.7530 |

Threshold used for the final report: 0.55.

## Limitations

- This is a capstone research prototype, not an operational warning system.
- The satellite dataset uses a limited number of dates.
- The geographic scope is Malaysia-only.
- The labels come from a single lightning provider.
- More testing is needed across seasons and years.

## Future Work

- Test on more Himawari-8 dates and more weather regimes.
- Evaluate cross-season and cross-year generalization.
- Try stronger augmentation and alternative CNN backbones.
- Compare against more honest tabular baselines.
- Add calibration analysis for deployment-style threshold selection.

## Likely Panel Questions and Safe Answers

| Question | Safe Answer |
|---|---|
| Why is the satellite CNN the main result? | Because it uses image pixels only and is not circular or strike-derived. |
| Why is the metadata model not the headline result? | It uses amplitude and strike_type, which are strike-derived and therefore not deployment-safe. |
| Why use a frozen ResNet-50 backbone? | To make CPU training feasible while keeping strong visual feature extraction. |
| Why tune the threshold on validation data? | To avoid leaking test information into the final metrics. |
| What is the main limitation? | The dataset is limited to Malaysia and a small set of satellite dates. |
| Can this be used operationally now? | No. It is a research prototype and needs broader validation first. |
