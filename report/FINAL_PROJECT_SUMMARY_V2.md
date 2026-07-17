# Final Project Summary ? Version 2

## Research Question
Can Himawari-9 infrared image patches discriminate MMD-recorded cloud-to-ground lightning associations within a conservative empirical Peninsular Malaysia study region?

## Version 1 Problem Discovery
Version 1 is retained as **Version 1 ? Frozen Reproducible Diagnostic Experiment**. It was computationally reproducible but scientifically limited by geographic and sampling shortcuts, including problematic negative sampling and no natural-prevalence evaluation.

## Version 2 Redesign
Version 2 is **Version 2 ? Frozen Corrected Scientific Experiment**. It uses a conservative study mask, full-crop negative exclusion, `[t?20m,t+30m)` temporal exclusion, active and zero-recorded frames, date/storm-disjoint splits and a natural-prevalence test.

## Model Comparison
Small CNN validation PR-AUC was consistently about 0.955?0.959 across seeds. Frozen ResNet-50 validation PR-AUC was about 0.861?0.864. The strongest simple image baseline, B13-minimum random forest, reached controlled PR-AUC 0.8542; the strongest geographic/time baseline reached PR-AUC 0.6498 and ROC-AUC 0.8170.

## Controlled Test
Selected model `small_cnn_seed2026_bce_pos_weight_train_split_none` achieved accuracy 0.955556, ROC-AUC 0.983538, PR-AUC 0.966210, TN 1934, FP 53, FN 69, TP 689, FAR/FDR 0.071429.

## Natural Prevalence
Prevalence was 0.243232. The selected model achieved accuracy 0.911111, ROC-AUC 0.948159, PR-AUC 0.896222, precision 0.922566, recall/POD 0.692691, FPR 0.018687, FAR/FDR 0.077434, TN 1838, FP 35, FN 185, TP 417.

Strength: only 35 false positives among 1,873 recorded negatives. Limitation: 185 of 602 recorded positives were missed.

## Scientific Contribution
The experiment provides a frozen, reproducible, test-locked demonstration that image patches contain discriminative information beyond geography/time and simple B13 summaries for recorded lightning-association classification.

## Limitations and Future Work
Residual geography/time predictability remains; zero-recorded does not mean no physical lightning; the study mask is empirical; MMD completeness is not uniform; no operational claims are supported. Future work should use official coverage metadata, independent seasons/regions, interpretability, and operationally realistic temporal inputs before any deployment claim.

## Exact Final Claim
Version 2 demonstrates meaningful image-based discrimination of MMD-recorded cloud-to-ground lightning associations from Himawari-9 image patches within a conservative empirical Peninsular Malaysia study region.
