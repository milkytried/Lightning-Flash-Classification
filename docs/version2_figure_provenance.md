# Version 2 Figure Provenance

Generated: `2026-07-17T02:44:51.314881+00:00`

Command: `python src/v2_final_figures.py --out results/v2/final_figures`

Source commit: `72d5b86895a280850f69bda2702d103a06da4638`

## Inputs

- `report/V2_PHASE3_FINAL_DECISION.json`: `451c5a243d0dcfb27567c23330c6fd0cf34686a48238eb3159062d2388a4fa4e`
- `report/V2_PHASE3_CONTROLLED_TEST.json`: `6d34b5dbd98b472f0962724df7b9167bd49a981d1f0572941b0059473b2f0c46`
- `report/V2_PHASE3_NATURAL_PREVALENCE.json`: `a10055ac6d38a8954e86cf9e4ec0812ca1cc4c82bf4511685063e5fcb5b7e483`
- `report/V2_PHASE3_TRAINING.json`: `1fe8c22f12c1fe893a4f9933925ecfd374b64917fc274c20f5152942ac64fae2`
- `report/V2_FULL_BASELINES.json`: `7892725ea30a09ab5db892c22116546268f7ec6e35b1f0ddaeb69ca816133f0a`
- `results/v2/phase3/controlled_test_predictions/small_cnn_seed2026_bce_pos_weight_train_split_none.csv`: `061266389fd27e8ddd6a5eb5f2eac1cbe1036082d128f1b4784755523b46f5b8`
- `results/v2/phase3/natural_prevalence_predictions/small_cnn_seed2026_bce_pos_weight_train_split_none.csv`: `2e2874697313c6ebe3731c2d1538dcf2a648119f2484e552ff4524f5c6580549`

## Output Figures

- `results\v2\final_figures\version1_vs_version2_comparison.png` (74072 bytes): `1b23b3826ac144720d4e9d55b85cfb00e27fe32f201a9a3566783ef2045c6dec` — Version 1 retained as diagnostic benchmark; Version 2 is the corrected scientific experiment.
- `results\v2\final_figures\validation_pr_auc_by_seed.png` (152731 bytes): `85ebbffb8cb06515bad6bd72f5f970eb5d0da7c6c72f53238019b90e113f21aa` — Validation-only seed stability for selected small CNN and frozen ResNet-50 configurations.
- `results\v2\final_figures\controlled_selected_confusion_matrix.png` (77473 bytes): `1e649c3538e759f2076f5073e2b3150d8dbee2c10eb8649f467f8c9c761e7d27` — Controlled test confusion matrix for the final selected small CNN.
- `results\v2\final_figures\natural_selected_confusion_matrix.png` (77699 bytes): `af5e8518bfaa292fbd34407d3b97fc5481feac87a69fa650759f249d323885d4` — Natural-prevalence test confusion matrix for the final selected small CNN.
- `results\v2\final_figures\controlled_roc_selected.png` (87727 bytes): `2df847aa2afaed5bc43e3b4d69131050e3bec0f0da36c65714a3b83ccaee2b6c` — ROC curve for selected model on controlled split.
- `results\v2\final_figures\controlled_precision_recall_selected.png` (79406 bytes): `c5dc2fb30a5b2de231126a2656ae88f5e28a8fdfca1760381617bda53d9e3769` — Precision-recall curve for selected model on controlled split.
- `results\v2\final_figures\natural_roc_selected.png` (87447 bytes): `f697021f5af00463ea9c1bb80239c4b0f2db3cdf8775857ead96499e93e4b5a6` — ROC curve for selected model on natural split.
- `results\v2\final_figures\natural_precision_recall_selected.png` (82008 bytes): `67b9d1d6058f10388b471b700835162429ab261e9e3b5d3a16678f603febb554` — Precision-recall curve for selected model on natural split.
- `results\v2\final_figures\controlled_probability_distribution_selected.png` (64976 bytes): `48dcc4b7de97becf2e856a75fb499c5513d009239142b45e16d1dd124e745b55` — Probability distributions by true label on controlled.
- `results\v2\final_figures\natural_probability_distribution_selected.png` (63203 bytes): `49d838307779d09b3cf7787f7f03ea372c7e808640af92433241299b31aee147` — Probability distributions by true label on natural.
- `results\v2\final_figures\baseline_comparison_pr_auc.png` (85595 bytes): `3981883d784d9f97769bf464be6e2fe8e1f1ea925f6fe99b30176ed508fd8f0e` — Small CNN materially exceeds geographic/time and B13-minimum baselines on controlled PR-AUC.
- `results\v2\final_figures\natural_frame_category_subgroup_accuracy.png` (69265 bytes): `47dbc49b5979f0da3382454c4a19e243cb2f0211408d26587809d0ba7f66844a` — Natural-prevalence subgroup accuracy by frame_category.
- `results\v2\final_figures\natural_date_subgroup_accuracy.png` (114958 bytes): `d6b7c9010ca008b91687ac226192ba028d0703dee7772c3a0f9c01f5d3aaac05` — Natural-prevalence subgroup accuracy by date.
- `results\v2\final_figures\natural_month_prediction_accuracy.png` (53307 bytes): `c1f16bf016fb224710d2f65b509b98018f5aa4d473467544f33041ca6834b839` — Natural-prevalence selected-model accuracy grouped by month.
- `results\v2\final_figures\natural_frame_category_prediction_accuracy.png` (72949 bytes): `3d82000f9e46cdbd67ddae29b8c944c2b612abe4b288beb59a75cbaec1ad8dc0` — Natural-prevalence selected-model accuracy grouped by frame_category.
- `results\v2\final_figures\example_tp_tn_fp_fn_patches.png` (71755 bytes): `f0b4a29c9706f8002d4612044e28a615582719f0430548ab806397ee7318550b` — Example controlled-test TP, TN, FP and FN patches for the selected model.
