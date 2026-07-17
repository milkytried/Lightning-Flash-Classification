# Version 2 Experiment Provenance

```json
{
  "label": "Version 2 ? Frozen Corrected Scientific Experiment",
  "created_at_utc": "2026-07-17T02:46:38.943236+00:00",
  "selected_model": "small_cnn_seed2026_bce_pos_weight_train_split_none",
  "source_commits": {
    "protocol": "c86df59b081d78498c968aec3cf74c0d4aaf0254",
    "validation_selection": "1758f98",
    "test_unlock": "94dc243",
    "final_evaluation": "b8062b8",
    "artifact_generator": "72d5b86",
    "current": "72d5b86895a280850f69bda2702d103a06da4638"
  },
  "execution_bug_fixes": {
    "21dfee8": "validation-selected six-run unlock/status mechanics only",
    "c4affc3": "interrupted-run resume mechanics only",
    "e5ae784": "Windows checkpoint replacement retry only",
    "400ce32": "full checkpoint save retry only",
    "6744c57": "final markdown writer dependency removal only"
  },
  "hashes": {
    "training_config_canonical_sha256": "f169a3348d48ab3fd4dda51d9efa9c098a1c058d2713b78e049825fce3fffae8",
    "controlled_manifest_sha256": "a0beb139f3b654028938952af39c29755fef7259547b01ba9dd58898fbfb585a",
    "natural_prevalence_manifest_sha256": "fc592b8c679ed0f81a242c0d37aeabf254090648f9fef07df49f343500dca6a2",
    "frame_ledger_sha256": "3d7545d868cace4aea0750d0f235a32e3375e3fa00cfa6eaf9f066b761781a1a",
    "checkpoint_sha256": "888696cb7f6d1543875795fca0deec2aaf5b0e54157692633b619e17f216ce1a",
    "prediction_files": {
      "results/v2/phase3/controlled_test_predictions/small_cnn_seed2026_bce_pos_weight_train_split_none.csv": "061266389fd27e8ddd6a5eb5f2eac1cbe1036082d128f1b4784755523b46f5b8",
      "results/v2/phase3/natural_prevalence_predictions/small_cnn_seed2026_bce_pos_weight_train_split_none.csv": "2e2874697313c6ebe3731c2d1538dcf2a648119f2484e552ff4524f5c6580549"
    },
    "reports": {
      "report/V2_PHASE3_PREFLIGHT.json": "8d55a60ea5543ac3349b37e8508f8da37fbab793076db8c5c6d5bad874244bdf",
      "report/V2_PHASE3_TRAINING.json": "1fe8c22f12c1fe893a4f9933925ecfd374b64917fc274c20f5152942ac64fae2",
      "report/V2_PHASE3_VALIDATION_SELECTION.json": "0b5a152e9ff08a60889f15233d3489b71506bd5cc67fbd429efdc766bf6aa7a0",
      "report/V2_PHASE3_TEST_UNLOCK.json": "7f17988a2607bd3f953a521983f8273d1b3bab8d889c707e977f47e246c8b461",
      "report/V2_PHASE3_CONTROLLED_TEST.json": "6d34b5dbd98b472f0962724df7b9167bd49a981d1f0572941b0059473b2f0c46",
      "report/V2_PHASE3_NATURAL_PREVALENCE.json": "a10055ac6d38a8954e86cf9e4ec0812ca1cc4c82bf4511685063e5fcb5b7e483",
      "report/V2_PHASE3_FINAL_COMPARISON.json": "6433cf8d91557ee34ff82a3b75b9c15fa00083b8e7d9cc743947be668d24235b",
      "report/V2_PHASE3_FINAL_DECISION.json": "451c5a243d0dcfb27567c23330c6fd0cf34686a48238eb3159062d2388a4fa4e",
      "report/V2_FINAL_REPRODUCIBILITY_AUDIT.json": "32cb5c252e4901e02da808f906dedabc2e5e8449319489434c6b5bd66d355f52"
    }
  },
  "threshold": 0.8307269811630249,
  "calibration": {
    "method": "validation_logits_temperature_scaling",
    "temperature": 1.0706590414047241
  },
  "seeds": [
    42,
    1337,
    2026
  ],
  "environment": {
    "device": "cpu",
    "parameter_counts": {
      "frozen": 0,
      "total": 102017,
      "trainable": 102017
    }
  },
  "runtime_seconds": 1038.4969580173492,
  "test_unlock_commit": "94dc243"
}
```
