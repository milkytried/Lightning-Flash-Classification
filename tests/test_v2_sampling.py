import pandas as pd

from src.mmd_spatiotemporal_index import MMDSpatiotemporalIndex


def synthetic_index():
    return MMDSpatiotemporalIndex(pd.DataFrame({
        "timestamp": pd.to_datetime(["2025-01-01T00:05:00Z", "2025-01-01T00:25:00Z"]),
        "lat": [3.0, 4.0], "lon": [101.0, 102.0], "solution_key": ["a", "b"],
    }))


def test_query_window_is_end_exclusive_and_utc():
    index = synthetic_index()
    result = index.query_window(pd.Timestamp("2025-01-01T00:10:00Z"), -10, 15)
    assert list(result.solution_key) == ["a"]
    assert str(result.timestamp.dt.tz) == "UTC"


def test_full_patch_and_margin_rejects_strike_inside_rectangle():
    index = synthetic_index()
    result = index.patch_query(pd.Timestamp("2025-01-01T00:00:00Z"), 3.0, 101.0, 64, 0.02, 10.0, -10, 20)
    assert result["clear"] is False
    assert result["inside_count"] == 1


def test_full_patch_accepts_distant_candidate():
    index = synthetic_index()
    result = index.patch_query(pd.Timestamp("2025-01-01T00:00:00Z"), 6.0, 104.0, 64, 0.02, 10.0, -10, 20)
    assert result["clear"] is True

