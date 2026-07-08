import numpy as np
import pandas as pd
import pytest

from src.build_satellite_dataset import (
    FrameSlot,
    assign_slots,
    build_hsd_key,
    chronological_split,
    floor_to_ahi_slot,
    read_mmd_ground_strikes,
    sample_negative_centres,
    select_frame_times,
    strikes_in_target_window,
    target_window,
    validate_manifest,
)


def test_read_mmd_ground_strikes_filters_ground_and_bounds(tmp_path):
    csv_dir = tmp_path / "2024" / "PENINSULAR" / "01 JAN" / "1"
    csv_dir.mkdir(parents=True)
    csv_path = csv_dir / "raw data all.csv"
    csv_path.write_text(
        "\n".join(
            [
                "Date/Time,Latitude,Longitude,Amplitude,Cloud or Ground",
                "2024-01-15 00:09:46Z,3.38,101.68,7.4,Ground",
                "2024-01-15 00:10:46Z,3.45,101.69,0,Cloud",
                "2024-01-15 00:11:46Z,40.0,101.69,8.1,Ground",
            ]
        ),
        encoding="utf-8",
    )

    strikes = read_mmd_ground_strikes(tmp_path)

    assert len(strikes) == 1
    assert strikes.iloc[0]["timestamp"] == pd.Timestamp("2024-01-15 00:09:46Z")
    assert strikes.iloc[0]["lat"] == pytest.approx(3.38)
    assert strikes.iloc[0]["lon"] == pytest.approx(101.68)


def test_floor_to_ahi_slot_same_frame_and_nowcast():
    same_frame = floor_to_ahi_slot(pd.Timestamp("2024-01-15T00:19:46Z"))
    nowcast_frame = floor_to_ahi_slot(
        pd.Timestamp("2024-01-15T00:19:46Z"),
        nowcast_minutes=60,
    )

    assert same_frame.timestamp == pd.Timestamp("2024-01-15T00:10:00Z")
    assert nowcast_frame.timestamp == pd.Timestamp("2024-01-14T23:20:00Z")


def test_frame_slot_selects_himawari9_for_2023_onward():
    slot = FrameSlot(pd.Timestamp("2024-01-15T00:00:00Z"))

    assert slot.satellite_id == "H09"
    assert slot.bucket == "noaa-himawari9"


def test_build_hsd_key_matches_noaa_layout():
    slot = FrameSlot(pd.Timestamp("2024-01-15T00:00:00Z"))

    key = build_hsd_key(slot, "B13", 8)

    assert key == (
        "AHI-L1b-FLDK/2024/01/15/0000/"
        "HS_H09_20240115_0000_B13_FLDK_R20_S0810.DAT.bz2"
    )


def test_assign_slots_and_chronological_split():
    strikes = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2024-12-31T23:59:00Z", "2025-03-01T00:01:00Z", "2025-07-01T00:01:00Z"],
                utc=True,
            )
        }
    )

    slotted = assign_slots(strikes)

    assert [chronological_split(ts) for ts in slotted["frame_time"]] == ["train", "val", "test"]


def test_nowcast_target_window_uses_only_future_strikes():
    frame_time = pd.Timestamp("2024-01-14T23:20:00Z")
    strikes = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2024-01-14T23:19:59Z",
                    "2024-01-15T00:10:00Z",
                    "2024-01-15T00:20:00Z",
                    "2024-01-15T00:20:01Z",
                ],
                utc=True,
            )
        }
    )

    start, end = target_window(frame_time, nowcast_minutes=60)
    window_strikes = strikes_in_target_window(strikes, frame_time, nowcast_minutes=60)

    assert start == frame_time
    assert end == pd.Timestamp("2024-01-15T00:20:00Z")
    assert list(window_strikes["timestamp"]) == [
        pd.Timestamp("2024-01-15T00:10:00Z"),
        pd.Timestamp("2024-01-15T00:20:00Z"),
    ]


def test_select_frame_times_spans_timeline_when_capped():
    strikes = pd.DataFrame(
        {
            "frame_time": pd.date_range(
                "2023-01-01T00:00:00Z",
                periods=10,
                freq="30D",
            )
        }
    )

    selected = select_frame_times(strikes, max_frames=3)

    assert selected[0] == strikes.iloc[0]["frame_time"]
    assert selected[-1] == strikes.iloc[-1]["frame_time"]


def test_validate_manifest_rejects_date_overlap():
    manifest = pd.DataFrame(
        {
            "path": ["a.png", "b.png"],
            "label": [1, 0],
            "split": ["train", "val"],
            "timestamp": ["2025-01-01T00:00:00Z", "2025-01-01T00:10:00Z"],
            "frame_id": ["H09_20250101_0000", "H09_20250101_0010"],
            "target_window_start": ["2025-01-01T00:00:00Z", "2025-01-01T00:10:00Z"],
            "target_window_end": ["2025-01-01T00:10:00Z", "2025-01-01T00:20:00Z"],
            "lat": [3.0, 4.0],
            "lon": [101.0, 102.0],
        }
    )

    with pytest.raises(ValueError, match="Date leakage"):
        validate_manifest(manifest)


def test_sample_negative_centres_avoids_strikes():
    strikes = pd.DataFrame({"lat": [3.0], "lon": [101.0]})

    centres = sample_negative_centres(
        strikes,
        image_shape=(1000, 1050),
        count=5,
        min_distance_km=100.0,
        rng=np.random.default_rng(7),
    )

    assert len(centres) == 5
    for _x, _y, lat, lon in centres:
        assert not (abs(lat - 3.0) < 0.5 and abs(lon - 101.0) < 0.5)
