# V2 Phase 3 Controlled Test

Validation-selected thresholds were frozen before this inference.

| run | accuracy | precision | recall_pod | far | fpr | roc_auc | pr_auc | hss | tss |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| small_cnn_seed42_bce_pos_weight_train_split_none | 0.9443 | 0.9219 | 0.8720 | 0.0781 | 0.0282 | 0.9803 | 0.9575 | 0.8582 | 0.8438 |
| small_cnn_seed1337_bce_pos_weight_train_split_none | 0.9399 | 0.9419 | 0.8338 | 0.0581 | 0.0196 | 0.9835 | 0.9638 | 0.8441 | 0.8141 |
| small_cnn_seed2026_bce_pos_weight_train_split_none | 0.9556 | 0.9286 | 0.9090 | 0.0714 | 0.0267 | 0.9835 | 0.9662 | 0.8881 | 0.8823 |
| frozen_resnet50_seed42_bce_unweighted_none | 0.8940 | 0.7997 | 0.8219 | 0.2003 | 0.0785 | 0.9457 | 0.8805 | 0.7371 | 0.7434 |
| frozen_resnet50_seed1337_bce_unweighted_none | 0.8987 | 0.8069 | 0.8325 | 0.1931 | 0.0760 | 0.9449 | 0.8863 | 0.7491 | 0.7565 |
| frozen_resnet50_seed2026_bce_unweighted_none | 0.8980 | 0.8292 | 0.7942 | 0.1708 | 0.0624 | 0.9478 | 0.8814 | 0.7415 | 0.7318 |
