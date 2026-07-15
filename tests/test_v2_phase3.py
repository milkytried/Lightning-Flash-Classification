from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

from src.v2_phase3_common import SmallCNN, V2PatchDataset, create_v2_loader, run_inference


def test_v2_phase3_loader_raises_on_missing_patch(tmp_path):
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame([{"path": str(tmp_path / "missing.png"), "label": 1, "split": "val"}]).to_csv(manifest, index=False)
    dataset = V2PatchDataset(manifest, split="val", architecture="small_cnn", augment=False)
    try:
        dataset[0]
    except FileNotFoundError as exc:
        assert "Patch image does not exist" in str(exc)
    else:
        raise AssertionError("missing image should not be replaced by a black patch")


def test_v2_phase3_eval_and_standalone_preprocessing_match(tmp_path):
    image_path = tmp_path / "patch.png"
    array = np.arange(64 * 64 * 3, dtype=np.uint32).reshape(64, 64, 3) % 255
    Image.fromarray(array.astype(np.uint8), mode="RGB").save(image_path)
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame([
        {"path": str(image_path), "label": 1, "split": "test", "date": "2025-03-01", "frame_id": "f1", "storm_id": "s1"}
    ]).to_csv(manifest, index=False)

    dataset = V2PatchDataset(manifest, split="test", architecture="small_cnn", augment=False)
    standalone_tensor = dataset.load_tensor_for_path(image_path, augment=False)
    loader_tensor, _, _ = dataset[0]
    assert torch.equal(standalone_tensor, loader_tensor)

    torch.manual_seed(42)
    model = SmallCNN().eval()
    with torch.no_grad():
        standalone_prob = torch.sigmoid(model(standalone_tensor.unsqueeze(0))).item()
    loader = create_v2_loader(manifest, split="test", architecture="small_cnn", batch_size=1, augment=False, shuffle=False)
    inferred = run_inference(model, loader, torch.device("cpu"))
    assert inferred.probs.shape == (1,)
    assert abs(float(inferred.probs[0]) - standalone_prob) < 1e-7


def test_v2_phase3_models_emit_logits_not_probabilities():
    model = SmallCNN().eval()
    with torch.no_grad():
        logits = model(torch.randn(4, 3, 64, 64))
    assert logits.shape == (4,)
    assert torch.is_floating_point(logits)
