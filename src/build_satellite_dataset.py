"""
Build an aligned Himawari -> MMD lightning patch dataset.

The existing satellite training loader expects 64x64 PNG patches plus a CSV with
at least path,label,split. This builder starts from the MMD lightning records,
downloads the matching Himawari HSD files, crops Malaysia, and writes a
drop-in manifest for the existing loader.
"""

from __future__ import annotations

import argparse
import logging
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from PIL import Image

try:
    import s3fs
except ImportError:  # pragma: no cover - exercised by CLI dependency check
    s3fs = None

try:
    from pyresample.geometry import AreaDefinition
    from satpy import Scene
except ImportError:  # pragma: no cover - exercised by CLI dependency check
    AreaDefinition = None
    Scene = None


logger = logging.getLogger(__name__)

MALAYSIA_BOUNDS = (99.0, -5.0, 120.0, 15.0)  # lon_min, lat_min, lon_max, lat_max
DEFAULT_BANDS = ("B08", "B13", "B15")
DEFAULT_SEGMENTS = (5, 6)
PHYSICAL_RANGES_K = {
    "B08": (190.0, 260.0),
    "B13": (180.0, 330.0),
    "B15": (180.0, 330.0),
}


@dataclass(frozen=True)
class FrameSlot:
    """A 10-minute Himawari full-disk scan slot in UTC."""

    timestamp: pd.Timestamp

    @property
    def date(self) -> str:
        return self.timestamp.strftime("%Y%m%d")

    @property
    def yyyy_mm_dd(self) -> tuple[str, str, str]:
        return (
            self.timestamp.strftime("%Y"),
            self.timestamp.strftime("%m"),
            self.timestamp.strftime("%d"),
        )

    @property
    def hhmm(self) -> str:
        return self.timestamp.strftime("%H%M")

    @property
    def satellite_id(self) -> str:
        return "H09" if self.timestamp >= pd.Timestamp("2022-12-13", tz="UTC") else "H08"

    @property
    def bucket(self) -> str:
        return "noaa-himawari9" if self.satellite_id == "H09" else "noaa-himawari8"


def require_satellite_dependencies() -> None:
    missing = []
    if s3fs is None:
        missing.append("s3fs")
    if Scene is None:
        missing.append("satpy[ahi_hsd]")
    if AreaDefinition is None:
        missing.append("pyresample")
    if missing:
        raise RuntimeError(
            "Missing satellite build dependencies: "
            + ", ".join(missing)
            + ". Install with: pip install \"satpy[ahi_hsd]\" pyresample s3fs boto3 bottleneck"
        )


def read_mmd_ground_strikes(data_root: str | Path) -> pd.DataFrame:
    """Read all MMD CSV files and keep valid cloud-to-ground strikes."""

    csv_files = sorted(Path(data_root).rglob("raw data all.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No 'raw data all.csv' files found under {data_root}")

    frames = []
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
        except Exception as exc:
            logger.warning("Skipping unreadable CSV %s: %s", csv_file, exc)
            continue

        required = {"Date/Time", "Latitude", "Longitude", "Cloud or Ground"}
        if not required.issubset(df.columns):
            logger.warning("Skipping %s because required columns are missing", csv_file)
            continue

        df = df.copy()
        df["source_file"] = str(csv_file)
        frames.append(df)

    if not frames:
        raise ValueError(f"No usable lightning CSVs found under {data_root}")

    strikes = pd.concat(frames, ignore_index=True)
    strikes = strikes[strikes["Cloud or Ground"].astype(str).str.casefold() == "ground"].copy()
    strikes["timestamp"] = pd.to_datetime(strikes["Date/Time"], utc=True, errors="coerce")
    strikes["lat"] = pd.to_numeric(strikes["Latitude"], errors="coerce")
    strikes["lon"] = pd.to_numeric(strikes["Longitude"], errors="coerce")
    if "Amplitude" in strikes.columns:
        strikes["amplitude"] = pd.to_numeric(strikes["Amplitude"], errors="coerce")
    else:
        strikes["amplitude"] = np.nan

    lon_min, lat_min, lon_max, lat_max = MALAYSIA_BOUNDS
    strikes = strikes.dropna(subset=["timestamp", "lat", "lon"])
    strikes = strikes[
        strikes["lat"].between(lat_min, lat_max) & strikes["lon"].between(lon_min, lon_max)
    ].copy()

    logger.info("Loaded %d cloud-to-ground strikes from %d CSV files", len(strikes), len(csv_files))
    return strikes.reset_index(drop=True)


def floor_to_ahi_slot(timestamp: pd.Timestamp, nowcast_minutes: int = 0) -> FrameSlot:
    """Map a strike timestamp to the matching or preceding 10-minute AHI slot."""

    ts = pd.Timestamp(timestamp)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    ts = ts.tz_convert("UTC")
    if nowcast_minutes > 0:
        ts = ts - pd.Timedelta(minutes=nowcast_minutes)
        slot = ts.ceil("10min")
    else:
        slot = ts.floor("10min")
    return FrameSlot(slot)


def assign_slots(strikes: pd.DataFrame, nowcast_minutes: int = 0) -> pd.DataFrame:
    strikes = strikes.copy()
    strikes["frame_time"] = [
        floor_to_ahi_slot(ts, nowcast_minutes=nowcast_minutes).timestamp for ts in strikes["timestamp"]
    ]
    return strikes


def chronological_split(frame_time: pd.Timestamp) -> str:
    ts = pd.Timestamp(frame_time).tz_convert("UTC")
    date = ts.date()
    if date <= pd.Timestamp("2024-12-31").date():
        return "train"
    if date <= pd.Timestamp("2025-03-01").date():
        return "val"
    return "test"


def target_window(frame_time: pd.Timestamp, nowcast_minutes: int) -> tuple[pd.Timestamp, pd.Timestamp]:
    start = pd.Timestamp(frame_time).tz_convert("UTC")
    horizon = pd.Timedelta(minutes=nowcast_minutes if nowcast_minutes > 0 else 10)
    return start, start + horizon


def strikes_in_target_window(
    strikes: pd.DataFrame,
    frame_time: pd.Timestamp,
    nowcast_minutes: int,
) -> pd.DataFrame:
    start, end = target_window(frame_time, nowcast_minutes)
    if nowcast_minutes > 0:
        mask = (strikes["timestamp"] > start) & (strikes["timestamp"] <= end)
    else:
        mask = (strikes["timestamp"] >= start) & (strikes["timestamp"] < end)
    return strikes[mask].copy()


def build_hsd_key(slot: FrameSlot, band: str, segment: int) -> str:
    year, month, day = slot.yyyy_mm_dd
    resolution = "R20" if band in {"B05", "B06", "B07", "B08", "B09", "B10", "B11", "B12", "B13", "B14", "B15", "B16"} else "R10"
    return (
        f"AHI-L1b-FLDK/{year}/{month}/{day}/{slot.hhmm}/"
        f"HS_{slot.satellite_id}_{slot.date}_{slot.hhmm}_{band}_FLDK_{resolution}_S{segment:02d}10.DAT.bz2"
    )


def download_frame(
    slot: FrameSlot,
    bands: Sequence[str],
    segments: Sequence[int],
    cache_root: str | Path,
    overwrite: bool = False,
) -> list[Path]:
    """Download selected Himawari HSD band/segment files from public NOAA S3."""

    require_satellite_dependencies()
    fs = s3fs.S3FileSystem(anon=True)
    local_files: list[Path] = []
    cache_root = Path(cache_root)

    for band in bands:
        for segment in segments:
            key = build_hsd_key(slot, band, segment)
            remote = f"{slot.bucket}/{key}"
            local_path = cache_root / slot.bucket / key
            local_path.parent.mkdir(parents=True, exist_ok=True)
            if local_path.exists() and not overwrite:
                local_files.append(local_path)
                continue

            if not fs.exists(remote):
                raise FileNotFoundError(f"Missing Himawari object: s3://{remote}")

            logger.info("Downloading s3://%s", remote)
            fs.get(remote, str(local_path))
            local_files.append(local_path)

    return local_files


def malaysia_area(resolution_degrees: float = 0.02):
    """Regular lon/lat grid over Malaysia, about 2 km per pixel."""

    require_satellite_dependencies()
    lon_min, lat_min, lon_max, lat_max = MALAYSIA_BOUNDS
    width = int(math.ceil((lon_max - lon_min) / resolution_degrees))
    height = int(math.ceil((lat_max - lat_min) / resolution_degrees))
    return AreaDefinition(
        "malaysia_2km",
        "Malaysia lon/lat grid",
        "malaysia_2km",
        {"proj": "longlat", "datum": "WGS84"},
        width,
        height,
        (lon_min, lat_min, lon_max, lat_max),
    )


def load_resampled_frame(files: Sequence[Path], bands: Sequence[str], resolution_degrees: float = 0.02):
    """Load HSD files with Satpy and resample them to the Malaysia lon/lat area."""

    require_satellite_dependencies()
    scene = Scene(filenames=[str(path) for path in files], reader="ahi_hsd")
    scene.load(list(bands), calibration="brightness_temperature")
    return scene.resample(malaysia_area(resolution_degrees))


def band_to_uint8(values: np.ndarray, band: str) -> np.ndarray:
    """Scale brightness temperatures to uint8 with fixed physical ranges."""

    lo, hi = PHYSICAL_RANGES_K.get(band, (180.0, 330.0))
    arr = np.asarray(values, dtype=np.float32)
    arr = np.clip(arr, lo, hi)
    arr = (arr - lo) / (hi - lo)
    arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
    return np.round(arr * 255).astype(np.uint8)


def stack_frame_uint8(resampled_scene, bands: Sequence[str]) -> np.ndarray:
    channels = [band_to_uint8(resampled_scene[band].values, band) for band in bands]
    return np.stack(channels, axis=-1)


def lonlat_to_pixel(lat: float, lon: float, shape: tuple[int, int]) -> tuple[int, int]:
    lon_min, lat_min, lon_max, lat_max = MALAYSIA_BOUNDS
    height, width = shape
    x = int(round((lon - lon_min) / (lon_max - lon_min) * (width - 1)))
    y = int(round((lat_max - lat) / (lat_max - lat_min) * (height - 1)))
    return x, y


def crop_patch(image: np.ndarray, x: int, y: int, patch_size: int = 64) -> np.ndarray | None:
    half = patch_size // 2
    x0 = x - half
    y0 = y - half
    x1 = x0 + patch_size
    y1 = y0 + patch_size
    if x0 < 0 or y0 < 0 or x1 > image.shape[1] or y1 > image.shape[0]:
        return None
    return image[y0:y1, x0:x1, :]


def haversine_km(lat1: np.ndarray, lon1: np.ndarray, lat2: float, lon2: float) -> np.ndarray:
    radius_km = 6371.0
    p1 = np.radians(lat1)
    p2 = math.radians(lat2)
    dp = np.radians(lat2 - lat1)
    dl = np.radians(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * math.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * radius_km * np.arcsin(np.sqrt(a))


def sample_negative_centres(
    strikes: pd.DataFrame,
    image_shape: tuple[int, int],
    count: int,
    min_distance_km: float,
    rng: np.random.Generator,
    patch_size: int = 64,
    max_attempts: int = 10000,
) -> list[tuple[int, int, float, float]]:
    """Sample crop centres far enough away from all strikes in the same frame."""

    height, width = image_shape
    half = patch_size // 2
    lon_min, lat_min, lon_max, lat_max = MALAYSIA_BOUNDS
    strike_lats = strikes["lat"].to_numpy(dtype=float)
    strike_lons = strikes["lon"].to_numpy(dtype=float)
    centres: list[tuple[int, int, float, float]] = []

    attempts = 0
    while len(centres) < count and attempts < max_attempts:
        attempts += 1
        x = int(rng.integers(half, width - half))
        y = int(rng.integers(half, height - half))
        lon = lon_min + (x / (width - 1)) * (lon_max - lon_min)
        lat = lat_max - (y / (height - 1)) * (lat_max - lat_min)

        if len(strike_lats) and np.min(haversine_km(strike_lats, strike_lons, lat, lon)) < min_distance_km:
            continue
        centres.append((x, y, lat, lon))

    return centres


def select_frame_times(
    strikes: pd.DataFrame,
    max_frames: int | None = None,
    max_frames_per_day: int | None = None,
) -> list[pd.Timestamp]:
    frame_times = (
        strikes[["frame_time"]]
        .drop_duplicates()
        .assign(day=lambda df: df["frame_time"].dt.date)
        .sort_values("frame_time")
    )
    if max_frames_per_day is not None:
        frame_times = frame_times.groupby("day", group_keys=False).head(max_frames_per_day)
    if max_frames is not None and len(frame_times) > max_frames:
        indices = np.linspace(0, len(frame_times) - 1, num=max_frames, dtype=int)
        frame_times = frame_times.iloc[indices]
    return list(frame_times["frame_time"])


def write_patch(path: Path, patch: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(patch, mode="RGB").save(path)


def build_dataset(args: argparse.Namespace) -> pd.DataFrame:
    rng = np.random.default_rng(args.seed)
    strikes = read_mmd_ground_strikes(args.lightning_root)
    strikes = assign_slots(strikes, nowcast_minutes=args.nowcast_minutes)
    frame_times = select_frame_times(strikes, args.max_frames, args.max_frames_per_day)
    if not frame_times:
        raise ValueError("No frame times selected; loosen sampling caps or check the input data")

    rows = []
    output_root = Path(args.patch_root)
    for frame_time in frame_times:
        slot = FrameSlot(pd.Timestamp(frame_time).tz_convert("UTC"))
        frame_strikes = strikes[strikes["frame_time"] == frame_time].copy()
        window_strikes = strikes_in_target_window(strikes, slot.timestamp, args.nowcast_minutes)
        window_start, window_end = target_window(slot.timestamp, args.nowcast_minutes)
        split = chronological_split(slot.timestamp)
        if args.max_positives_per_frame is not None and len(frame_strikes) > args.max_positives_per_frame:
            frame_strikes = frame_strikes.sample(
                n=args.max_positives_per_frame,
                random_state=args.seed,
            ).sort_values("timestamp")

        files = download_frame(
            slot,
            bands=args.bands,
            segments=args.segments,
            cache_root=args.cache_root,
            overwrite=args.overwrite,
        )
        scene = load_resampled_frame(files, args.bands, resolution_degrees=args.resolution_degrees)
        frame_image = stack_frame_uint8(scene, args.bands)
        himawari_files = ";".join(str(path) for path in files)
        frame_id = f"{slot.satellite_id}_{slot.date}_{slot.hhmm}"

        positive_written = 0
        for idx, strike in frame_strikes.iterrows():
            x, y = lonlat_to_pixel(strike["lat"], strike["lon"], frame_image.shape[:2])
            patch = crop_patch(frame_image, x, y, patch_size=args.patch_size)
            if patch is None:
                continue
            patch_path = output_root / split / "positive" / f"{slot.date}_{slot.hhmm}_pos_{idx}.png"
            write_patch(patch_path, patch)
            rows.append(
                {
                    "path": str(patch_path),
                    "label": 1,
                    "split": split,
                    "timestamp": slot.timestamp.isoformat(),
                    "frame_id": frame_id,
                    "target_window_start": window_start.isoformat(),
                    "target_window_end": window_end.isoformat(),
                    "strike_timestamp": pd.Timestamp(strike["timestamp"]).isoformat(),
                    "strike_id": strike.get("Solution Key", idx),
                    "source_mmd_file": strike.get("source_file", ""),
                    "himawari_files": himawari_files,
                    "frame_strike_count": int(len(window_strikes)),
                    "lat": float(strike["lat"]),
                    "lon": float(strike["lon"]),
                    "x": x,
                    "y": y,
                    "amplitude": strike.get("amplitude", np.nan),
                    "satellite": slot.satellite_id,
                    "bands": "+".join(args.bands),
                    "segments": "+".join(f"{segment:02d}" for segment in args.segments),
                }
            )
            positive_written += 1

        negative_target = int(round(positive_written * args.negative_ratio))
        negatives = sample_negative_centres(
            window_strikes,
            frame_image.shape[:2],
            count=negative_target,
            min_distance_km=args.negative_min_distance_km,
            rng=rng,
            patch_size=args.patch_size,
        )
        for neg_idx, (x, y, lat, lon) in enumerate(negatives):
            patch = crop_patch(frame_image, x, y, patch_size=args.patch_size)
            if patch is None:
                continue
            patch_path = output_root / split / "negative" / f"{slot.date}_{slot.hhmm}_neg_{neg_idx}.png"
            write_patch(patch_path, patch)
            rows.append(
                {
                    "path": str(patch_path),
                    "label": 0,
                    "split": split,
                    "timestamp": slot.timestamp.isoformat(),
                    "frame_id": frame_id,
                    "target_window_start": window_start.isoformat(),
                    "target_window_end": window_end.isoformat(),
                    "strike_timestamp": "",
                    "strike_id": "",
                    "source_mmd_file": "",
                    "himawari_files": himawari_files,
                    "frame_strike_count": int(len(window_strikes)),
                    "lat": float(lat),
                    "lon": float(lon),
                    "x": x,
                    "y": y,
                    "amplitude": np.nan,
                    "satellite": slot.satellite_id,
                    "bands": "+".join(args.bands),
                    "segments": "+".join(f"{segment:02d}" for segment in args.segments),
                }
            )

        logger.info(
            "%s %s: wrote %d positives and %d negatives",
            slot.date,
            slot.hhmm,
            positive_written,
            len(negatives),
        )

    manifest = pd.DataFrame(rows)
    if manifest.empty:
        raise ValueError("No patches were written; check bounds, segments, and selected frames")

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    validate_manifest(manifest)
    if output_csv.exists() and not args.no_backup:
        backup_path = output_csv.with_suffix(
            f".backup_{pd.Timestamp.now('UTC').strftime('%Y%m%dT%H%M%SZ')}{output_csv.suffix}"
        )
        shutil.copy2(output_csv, backup_path)
        logger.info("Backed up existing manifest to %s", backup_path)
    manifest.to_csv(output_csv, index=False)
    logger.info("Wrote %d rows to %s", len(manifest), output_csv)
    return manifest


def validate_manifest(manifest: pd.DataFrame) -> None:
    required = {
        "path",
        "label",
        "split",
        "timestamp",
        "frame_id",
        "target_window_start",
        "target_window_end",
        "lat",
        "lon",
    }
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"Manifest missing required columns: {sorted(missing)}")

    duplicated_paths = manifest[manifest["path"].duplicated()]["path"].head(5).tolist()
    if duplicated_paths:
        raise ValueError(f"Duplicate patch paths in manifest: {duplicated_paths}")

    frame_splits = manifest.groupby("frame_id")["split"].nunique()
    mixed_split_frames = frame_splits[frame_splits > 1].index.tolist()
    if mixed_split_frames:
        raise ValueError(f"Himawari frame appears in multiple splits: {mixed_split_frames[:5]}")

    dates_by_split = {
        split: set(pd.to_datetime(group["timestamp"], utc=True).dt.date)
        for split, group in manifest.groupby("split")
    }
    splits = list(dates_by_split)
    for i, split_a in enumerate(splits):
        for split_b in splits[i + 1 :]:
            overlap = dates_by_split[split_a] & dates_by_split[split_b]
            if overlap:
                raise ValueError(f"Date leakage between {split_a} and {split_b}: {sorted(overlap)[:5]}")

    positives = manifest[manifest["label"] == 1].copy()
    if not positives.empty and "strike_timestamp" in positives.columns:
        frame_times = pd.to_datetime(positives["timestamp"], utc=True)
        strike_times = pd.to_datetime(positives["strike_timestamp"], utc=True)
        if (frame_times > strike_times).any():
            raise ValueError("Future satellite imagery detected: a positive row uses a frame after its strike")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lightning-root", default="data/raw/himawari8_pngs")
    parser.add_argument("--cache-root", default="data/raw/himawari_hsd")
    parser.add_argument("--patch-root", default="data/processed/himawari_mmd_patches")
    parser.add_argument("--output-csv", default="data/processed/satellite_dataset.csv")
    parser.add_argument("--bands", nargs="+", default=list(DEFAULT_BANDS))
    parser.add_argument("--segments", nargs="+", type=int, default=list(DEFAULT_SEGMENTS))
    parser.add_argument("--max-frames", type=int, default=20)
    parser.add_argument("--max-frames-per-day", type=int, default=2)
    parser.add_argument("--negative-ratio", type=float, default=1.0)
    parser.add_argument("--max-positives-per-frame", type=int, default=50)
    parser.add_argument("--negative-min-distance-km", type=float, default=30.0)
    parser.add_argument("--patch-size", type=int, default=64)
    parser.add_argument("--resolution-degrees", type=float, default=0.02)
    parser.add_argument("--nowcast-minutes", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-backup", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    build_dataset(args)


if __name__ == "__main__":
    main()
