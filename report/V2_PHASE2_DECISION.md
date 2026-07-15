# Version 2 Phase 2 Decision

## Proceed to neural-network training

| index | pass |
| --- | --- |
| zero_primary_contamination | True |
| sufficient_holdout_samples | True |
| zero_cross_split_overlap | True |
| zero_duplicate_crops | True |
| zero_invalid_patches | True |
| manifest_regeneration_deterministic | True |
| geographic_rf_auc_below_0_75 | True |
| geography_not_competitive_with_image | True |
| cluster_intervals_stable | True |
| no_dominant_cluster | True |
| natural_prevalence_created | True |
| all_tests_pass | True |

Geographic RF controlled-test ROC-AUC is 0.626; best image-derived ROC-AUC is 0.946. Natural prevalence is 24.32%.

No CNN or ResNet was trained. The exact recommended experiments and cost estimates are recorded in the JSON report.
