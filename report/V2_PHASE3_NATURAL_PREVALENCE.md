# V2 Phase 3 Natural Prevalence

Validation-selected thresholds were frozen before this inference.

| run | accuracy | precision | recall_pod | far | fpr | roc_auc | pr_auc | hss | tss |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| small_cnn_seed42_bce_pos_weight_train_split_none | 0.9030 | 0.9289 | 0.6512 | 0.0711 | 0.0160 | 0.9446 | 0.8847 | 0.7069 | 0.6351 |
| small_cnn_seed1337_bce_pos_weight_train_split_none | 0.8970 | 0.9506 | 0.6080 | 0.0494 | 0.0101 | 0.9459 | 0.8948 | 0.6811 | 0.5978 |
| small_cnn_seed2026_bce_pos_weight_train_split_none | 0.9111 | 0.9226 | 0.6927 | 0.0774 | 0.0187 | 0.9482 | 0.8962 | 0.7362 | 0.6740 |
| frozen_resnet50_seed42_bce_unweighted_none | 0.8622 | 0.7736 | 0.6130 | 0.2264 | 0.0577 | 0.8880 | 0.7656 | 0.5974 | 0.5553 |
| frozen_resnet50_seed1337_bce_unweighted_none | 0.8687 | 0.7856 | 0.6329 | 0.2144 | 0.0555 | 0.8885 | 0.7754 | 0.6181 | 0.5774 |
| frozen_resnet50_seed2026_bce_unweighted_none | 0.8687 | 0.8259 | 0.5831 | 0.1741 | 0.0395 | 0.8934 | 0.7831 | 0.6038 | 0.5435 |
