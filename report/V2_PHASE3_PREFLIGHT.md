# V2 Phase 3 Preflight Gate

Created: `2026-07-15T15:33:05Z`

Gate result: `PASS`

## Go/No-Go Conditions

| condition | pass |
| --- | --- |
| primary_contamination_zero | True |
| cross_split_overlaps_zero | True |
| invalid_patch_counts_zero | True |
| deterministic_manifest_regeneration_passed | True |
| validation_and_test_support_passed | True |
| natural_prevalence_set_exists | True |
| latlon_rf_roc_auc_below_0_75 | True |
| geography_at_least_0_05_below_best_image | True |
| clustered_intervals_no_severe_instability | True |
| all_phase2_tests_passed | True |

## Frozen Input Hashes

| item | sha256_or_value |
| --- | --- |
| configuration_hash | eebe246103c5543f1c2b4618d76f91fe678edf14d966b19842c1e02b5928b081 |
| v2_full_config_sha256 | e3ce2188b8a21098cd424013d7e8f5e87e03b42f7d91b48331b400f7b2361333 |
| mmd_inventory_hash | b8c8154dcc41bc6c196a24b80ea7a1def999d79a8d1929bffa24d90d0dd2a77c |
| frame_ledger_hash | 3d7545d868cace4aea0750d0f235a32e3375e3fa00cfa6eaf9f066b761781a1a |
| controlled_manifest_hash | a0beb139f3b654028938952af39c29755fef7259547b01ba9dd58898fbfb585a |
| natural_prevalence_manifest_hash | fc592b8c679ed0f81a242c0d37aeabf254090648f9fef07df49f343500dca6a2 |
| source_code_commit | c86df59b081d78498c968aec3cf74c0d4aaf0254 |

## Environment

| item | value |
| --- | --- |
| python | 3.14.4 |
| platform | Windows-11-10.0.26200-SP0 |
| torch | 2.12.0+cpu |
| torchvision | 0.27.0+cpu |
| cuda_available | False |
| cuda_build | None |
| device_count | 0 |

No Version 1, Phase 1A, Phase 1B, or Phase 2 artifacts were modified by this preflight check.
