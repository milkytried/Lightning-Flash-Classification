import numpy as np
import pandas as pd
import pytest

from src.build_satellite_dataset import (
    FrameSlot,
    assign_slots,
    build_hsd_key,
    cached_frame_available,
    chronological_split,
    ensure_finite_patch,
    existing_frame_complete,
    floor_to_ahi_slot,
    order_frame_times_by_split,
    read_mmd_ground_strikes,
    reconcile_manifest_splits,
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
            "frame_time": pd.to_datetime(
                [
                    "2023-01-01T00:00:00Z",
                    "2023-06-01T00:00:00Z",
                    "2024-01-01T00:00:00Z",
                    "2024-06-01T00:00:00Z",
                    "2024-12-01T00:00:00Z",
                    "2025-01-08T00:00:00Z",
                    "2025-02-01T00:00:00Z",
                    "2025-02-20T00:00:00Z",
                    "2025-03-02T00:00:00Z",
                    "2025-03-15T00:00:00Z",
                    "2025-04-01T00:00:00Z",
                    "2025-04-18T00:00:00Z",
                ],
                utc=True,
            )
        }
    )

    selected = select_frame_times(strikes, max_frames=6)
    selected_splits = [chronological_split(frame_time) for frame_time in selected]

    assert {"train", "val", "test"}.issubset(selected_splits)
    assert selected_splits.count("train") >= selected_splits.count("val")
    assert selected_splits.count("train") >= selected_splits.count("test")


def test_order_frame_times_by_split_preserves_chronology_within_split():
    frame_times = pd.to_datetime(
        [
            "2025-03-03T00:10:00Z",
            "2023-01-01T00:10:00Z",
            "2025-01-02T00:10:00Z",
            "2023-01-02T00:10:00Z",
            "2025-03-04T00:10:00Z",
            "2025-01-03T00:10:00Z",
        ],
        utc=True,
    )

    ordered = order_frame_times_by_split(frame_times)

    for split in {"train", "val", "test"}:
        split_times = [frame_time for frame_time in ordered if chronological_split(frame_time) == split]
        assert split_times == sorted(split_times)


def test_order_frame_times_by_split_prioritizes_preferred_within_split():
    frame_times = pd.to_datetime(
        [
            "2025-03-03T00:10:00Z",
            "2025-03-15T00:10:00Z",
            "2025-03-20T00:10:00Z",
        ],
        utc=True,
    )

    ordered = order_frame_times_by_split(frame_times, preferred_frame_times={frame_times[1]})

    assert ordered[0] == frame_times[1]


def test_cached_frame_available_requires_all_segments(tmp_path):
    frame_time = pd.Timestamp("2025-03-15T00:00:00Z")
    slot = floor_to_ahi_slot(frame_time)

    for band in ["B08", "B13", "B15"]:
        for segment in [5, 6]:
            path = tmp_path / slot.bucket / build_hsd_key(slot, band, segment)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"cached")

    assert cached_frame_available(frame_time, ["B08", "B13", "B15"], [5, 6], tmp_path)
    (tmp_path / slot.bucket / build_hsd_key(slot, "B15", 6)).unlink()
    assert not cached_frame_available(frame_time, ["B08", "B13", "B15"], [5, 6], tmp_path)


def test_reconcile_manifest_splits_keeps_rows_and_updates_stale_split():
    manifest = pd.DataFrame(
        {
            "timestamp": ["2025-03-15T00:00:00Z"],
            "split": ["val"],
            "frame_id": ["H09_20250315_0000"],
        }
    )

    reconciled = reconcile_manifest_splits(manifest)

    assert len(reconciled) == 1
    assert reconciled.loc[0, "split"] == "test"


def test_validate_manifest_rejects_split_mismatch():
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

    with pytest.raises(ValueError, match="disagree with chronological_split"):
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


def test_ensure_finite_patch_replaces_nan_and_inf():
    patch = np.array([[[np.nan, np.inf, -np.inf]]], dtype=np.float32)

    cleaned = ensure_finite_patch(patch, "test_patch")

    assert cleaned.dtype == np.uint8
    assert np.isfinite(cleaned).all()
    assert cleaned.tolist() == [[[0, 255, 0]]]


def test_existing_frame_complete_requires_manifest_rows_and_files(tmp_path):
    patch_path = tmp_path / "patch.png"
    patch_path.write_bytes(b"not-empty")
    manifest = pd.DataFrame(
        {
            "path": [str(patch_path)],
            "frame_id": ["H09_20250101_0000"],
            "bands": ["B08+B13+B15"],
            "segments": ["05+06"],
            "target_window_start": ["2025-01-01T00:00:00+00:00"],
            "target_window_end": ["2025-01-01T00:10:00+00:00"],
        }
    )

    assert existing_frame_complete(
        manifest,
        "H09_20250101_0000",
        ["B08", "B13", "B15"],
        [5, 6],
        pd.Timestamp("2025-01-01T00:00:00Z"),
        pd.Timestamp("2025-01-01T00:10:00Z"),
    )
    assert not existing_frame_complete(
        manifest,
        "H09_20250101_0010",
        ["B08", "B13", "B15"],
        [5, 6],
        pd.Timestamp("2025-01-01T00:10:00Z"),
        pd.Timestamp("2025-01-01T00:20:00Z"),
    )
