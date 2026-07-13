import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from PIL import Image
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import plot_results


def build_synthetic_manifest(tmp_path: Path, frames_per_class: int = 6) -> tuple[Path, np.ndarray]:
    rows = []
    test_index = 0
    for label in (1, 0):
        for frame_index in range(frames_per_class):
            timestamp = pd.Timestamp("2025-03-01T00:00:00Z") + pd.Timedelta(
                days=frame_index * 10,
                minutes=label,
            )
            frame_id = f"H09_{timestamp:%Y%m%d_%H%M}"
            for patch_index in range(3):
                path = tmp_path / f"{label}_{frame_index}_{patch_index}.png"
                Image.new(
                    "RGB",
                    (64, 64),
                    color=(40 + label * 120, 30 + frame_index * 20, 80 + patch_index * 20),
                ).save(path)
                rows.append(
                    {
                        "path": str(path),
                        "label": label,
                        "split": "test",
                        "timestamp": timestamp.isoformat(),
                        "frame_id": frame_id,
                    }
                )
                test_index += 1

    manifest = tmp_path / "manifest.csv"
    pd.DataFrame(rows).to_csv(manifest, index=False)
    probabilities = np.linspace(0.01, 0.99, num=test_index)
    return manifest, probabilities


def test_example_selection_uses_distinct_frames_and_is_seeded(tmp_path, monkeypatch):
    monkeypatch.setattr(plot_results, "verify_selected_probability_mapping", lambda *args, **kwargs: [])
    manifest, probabilities = build_synthetic_manifest(tmp_path)
    first_output = tmp_path / "figures_first"
    second_output = tmp_path / "figures_second"

    first = plot_results.plot_example_input_patches(
        manifest,
        first_output,
        probabilities,
        checkpoint_path=tmp_path / "unused_checkpoint.pth",
        seed=42,
    )
    second = plot_results.plot_example_input_patches(
        manifest,
        second_output,
        probabilities,
        checkpoint_path=tmp_path / "unused_checkpoint.pth",
        seed=42,
    )

    assert first == second
    assert len(first) == 8
    for label in (1, 0):
        class_selections = [item for item in first if item["label"] == label]
        frame_ids = [item["frame_id"] for item in class_selections]
        assert len(frame_ids) == 4
        assert len(set(frame_ids)) == 4
        timestamps = [pd.Timestamp(item["frame_timestamp"]) for item in class_selections]
        assert timestamps == sorted(timestamps)
        assert timestamps[-1] - timestamps[0] == pd.Timedelta(days=50)

    sidecar = json.loads(
        (first_output / "example_input_patches_selection.json").read_text(encoding="utf-8")
    )
    assert sidecar["seed"] == 42
    assert sidecar["decision_threshold"] == pytest.approx(0.51)
    assert sidecar["selections"] == first
    assert (first_output / "example_input_patches.png").stat().st_size > 0


def test_example_selection_warns_instead_of_duplicating_frames(tmp_path, monkeypatch):
    monkeypatch.setattr(plot_results, "verify_selected_probability_mapping", lambda *args, **kwargs: [])
    manifest, probabilities = build_synthetic_manifest(tmp_path, frames_per_class=2)

    with pytest.warns(RuntimeWarning, match="only 2 distinct source frames"):
        selections = plot_results.plot_example_input_patches(
            manifest,
            tmp_path / "figures_fallback",
            probabilities,
            checkpoint_path=tmp_path / "unused_checkpoint.pth",
            seed=42,
        )

    assert len(selections) == 4
    for label in (1, 0):
        frame_ids = [item["frame_id"] for item in selections if item["label"] == label]
        assert len(frame_ids) == len(set(frame_ids)) == 2


def test_positive_probability_summary():
    probabilities = np.array([0.1, 0.2, 0.4, 0.8, 0.95, 0.99])
    labels = np.array([0, 1, 1, 1, 1, 0])

    summary = plot_results.summarize_positive_probabilities(probabilities, labels)

    assert summary["median"] == pytest.approx(0.6)
    assert summary["q1"] == pytest.approx(0.35)
    assert summary["q3"] == pytest.approx(0.8375)
    assert summary["fraction_above_0_9"] == pytest.approx(0.25)