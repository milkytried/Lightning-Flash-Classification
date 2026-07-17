from pathlib import Path

import pandas as pd
import pytest


def test_v2_selected_inference_matches_saved_predictions():
    required = [
        Path("report/V2_PHASE3_TEST_UNLOCK.json"),
        Path("models/v2/phase3/small_cnn_seed2026_bce_pos_weight_train_split_none_best.pth"),
        Path("results/v2/phase3/controlled_test_predictions/small_cnn_seed2026_bce_pos_weight_train_split_none.csv"),
        Path("results/v2/phase3/natural_prevalence_predictions/small_cnn_seed2026_bce_pos_weight_train_split_none.csv"),
    ]
    if not all(path.exists() for path in required):
        pytest.skip("Frozen V2 Phase 3 artifacts are not present in this environment")
    from src.v2_inference import infer_patch

    for prediction_csv in required[2:]:
        frame = pd.read_csv(prediction_csv)
        for idx in [0, len(frame) // 2, len(frame) - 1]:
            row = frame.iloc[idx]
            output = infer_patch(row["path"])
            assert output["classification"] == int(row["prediction"])
            assert abs(output["probability"] - float(row["probability"])) < 1e-6
            assert abs(output["logit"] - float(row["logit"])) < 1e-5
            assert output["run_name"] == "small_cnn_seed2026_bce_pos_weight_train_split_none"
