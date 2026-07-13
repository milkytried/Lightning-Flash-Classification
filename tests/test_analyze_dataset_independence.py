from pathlib import Path
import sys

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from analyze_dataset_independence import analyze_manifest


def test_independence_analysis_counts_frames_dates_and_clusters():
    manifest = pd.DataFrame(
        [
            {
                "label": 1,
                "split": "train",
                "timestamp": "2025-01-01T00:00:00Z",
                "frame_id": "f1",
                "lat": 0.0,
                "lon": 100.0,
            },
            {
                "label": 1,
                "split": "train",
                "timestamp": "2025-01-01T00:00:00Z",
                "frame_id": "f1",
                "lat": 0.0,
                "lon": 100.1,
            },
            {
                "label": 1,
                "split": "train",
                "timestamp": "2025-01-01T00:00:00Z",
                "frame_id": "f1",
                "lat": 0.0,
                "lon": 101.0,
            },
            {
                "label": 0,
                "split": "train",
                "timestamp": "2025-01-01T00:00:00Z",
                "frame_id": "f1",
                "lat": 2.0,
                "lon": 102.0,
            },
            {
                "label": 1,
                "split": "test",
                "timestamp": "2025-01-02T00:00:00Z",
                "frame_id": "f2",
                "lat": 1.0,
                "lon": 103.0,
            },
            {
                "label": 0,
                "split": "test",
                "timestamp": "2025-01-02T00:00:00Z",
                "frame_id": "f2",
                "lat": 3.0,
                "lon": 104.0,
            },
        ]
    )

    result = analyze_manifest(manifest, eps_km=20.0)

    assert result["overall"]["distinct_source_frames"] == 2
    assert result["overall"]["distinct_dates"] == 2
    assert result["overall"]["positive_distinct_frames"] == 2
    assert result["overall"]["positive_spatial_clusters"] == 3
    assert result["per_split"]["train"]["positive_spatial_clusters"] == 2
    assert result["overall"]["patches_per_frame"]["median"] == pytest.approx(3.0)
    assert result["overall"]["positives_per_positive_frame"]["max"] == 3
