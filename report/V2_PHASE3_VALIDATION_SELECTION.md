# V2 Phase 3 Validation Selection

Selection used validation predictions only. Controlled-test and natural-prevalence predictions remain locked.

## Selected Loss/Augmentation by Architecture

| architecture | loss | augmentation | mean_val_pr_auc | mean_val_loss |
| --- | --- | --- | --- | --- |
| small_cnn | bce_pos_weight_train_split | none | 0.9573 | 0.2347 |
| frozen_resnet50 | bce_unweighted | none | 0.8622 | 0.2693 |

## Six Primary Runs for Test Unlock

| run | architecture | seed | loss | augmentation | best_epoch | val_pr_auc | val_roc_auc | threshold | checkpoint_sha256 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| small_cnn_seed42_bce_pos_weight_train_split_none | small_cnn | 42 | bce_pos_weight_train_split | none | 9 | 0.9550 | 0.9792 | 0.8693 | 8d4f6f8b0a00ad7f366cf7a48a8a80bb176f018f39b2b8956fd74bdf461c50b4 |
| small_cnn_seed1337_bce_pos_weight_train_split_none | small_cnn | 1337 | bce_pos_weight_train_split | none | 21 | 0.9579 | 0.9808 | 0.9199 | a0c95144ab1c30060a975229cc53a7bf1adfe45362956394ce876661b2d7ba39 |
| small_cnn_seed2026_bce_pos_weight_train_split_none | small_cnn | 2026 | bce_pos_weight_train_split | none | 25 | 0.9588 | 0.9811 | 0.8307 | 888696cb7f6d1543875795fca0deec2aaf5b0e54157692633b619e17f216ce1a |
| frozen_resnet50_seed42_bce_unweighted_none | frozen_resnet50 | 42 | bce_unweighted | none | 6 | 0.8614 | 0.9422 | 0.4921 | 8a85ef9c6a0b322a787bf5e5fc20e9808b244170dad93b8412f3d3fdec01b82a |
| frozen_resnet50_seed1337_bce_unweighted_none | frozen_resnet50 | 1337 | bce_unweighted | none | 4 | 0.8611 | 0.9415 | 0.4223 | d1d27934103f984a974ee63096afc79fd0024973313b8bcbe8a087e1e449a6d3 |
| frozen_resnet50_seed2026_bce_unweighted_none | frozen_resnet50 | 2026 | bce_unweighted | none | 5 | 0.8641 | 0.9422 | 0.5907 | d6452c7b2ccd9281e95bea10b6ae2a71fa22cd37639ceffd95e9a716fb1be54c |
