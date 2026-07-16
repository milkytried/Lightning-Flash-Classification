# V2 Phase 3 Training

Controlled-test and natural-prevalence inference remain locked. All choices below were made from training/validation only.

| run | val_pr_auc | val_roc_auc | threshold | checkpoint |
| --- | --- | --- | --- | --- |
| small_cnn_seed42_bce_pos_weight_train_split_none | 0.9550 | 0.9792 | 0.8693 | models\v2\phase3\small_cnn_seed42_bce_pos_weight_train_split_none_best.pth |
| small_cnn_seed1337_bce_pos_weight_train_split_none | 0.9579 | 0.9808 | 0.9199 | models\v2\phase3\small_cnn_seed1337_bce_pos_weight_train_split_none_best.pth |
| small_cnn_seed2026_bce_pos_weight_train_split_none | 0.9588 | 0.9811 | 0.8307 | models\v2\phase3\small_cnn_seed2026_bce_pos_weight_train_split_none_best.pth |
| frozen_resnet50_seed42_bce_unweighted_none | 0.8614 | 0.9422 | 0.4921 | models\v2\phase3\frozen_resnet50_seed42_bce_unweighted_none_best.pth |
| frozen_resnet50_seed1337_bce_unweighted_none | 0.8611 | 0.9415 | 0.4223 | models\v2\phase3\frozen_resnet50_seed1337_bce_unweighted_none_best.pth |
| frozen_resnet50_seed2026_bce_unweighted_none | 0.8641 | 0.9422 | 0.5907 | models\v2\phase3\frozen_resnet50_seed2026_bce_unweighted_none_best.pth |
