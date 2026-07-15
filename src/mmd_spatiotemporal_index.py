"""Compact deterministic spatiotemporal index for MMD-recorded ground strikes."""

from __future__ import annotations

import logging
import math
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)
EARTH_RADIUS_KM = 6371.0088
REQUIRED_COLUMNS = ["Date/Time", "Latitude", "Longitude", "Cloud or Ground", "Solution Key"]


class MMDSpatiotemporalIndex:
    """Index compact NumPy strike arrays by ten-minute UTC bin."""

    def __init__(self, strikes: pd.DataFrame):
        required = {"timestamp", "lat", "lon"}
        missing = required.difference(strikes.columns)
        if missing:
            raise ValueError(f"Strike table missing {sorted(missing)}")
        rows = strikes.sort_values(["timestamp", "lat", "lon"], kind="mergesort").reset_index(drop=True)
        self.timestamps_ns = rows.timestamp.to_numpy(dtype="datetime64[ns]").astype("int64")
        self.latitudes = rows.lat.to_numpy(np.float32)
        self.longitudes = rows.lon.to_numpy(np.float32)
        self.solution_keys = rows.get("solution_key", pd.Series([""] * len(rows))).astype(str).to_numpy()
        bins = rows.timestamp.dt.floor("10min").to_numpy(dtype="datetime64[ns]").astype("int64")
        self.bin_indices: dict[int, np.ndarray] = {}
        for value in np.unique(bins):
            self.bin_indices[int(value)] = np.flatnonzero(bins == value)
        memory = self.timestamps_ns.nbytes + self.latitudes.nbytes + self.longitudes.nbytes
        LOGGER.info("Indexed %d strikes in %d bins; compact arrays %.1f MiB", len(rows), len(self.bin_indices), memory / 2**20)

    @classmethod
    def from_inventory(cls, inventory_csv: Path, bounds: dict[str, float]) -> "MMDSpatiotemporalIndex":
        started = time.perf_counter()
        inventory = pd.read_csv(inventory_csv)
        frames = []
        for path in inventory.loc[inventory.status.eq("valid"), "path"]:
            available = pd.read_csv(path, nrows=0).columns
            usecols = [name for name in REQUIRED_COLUMNS if name in available]
            frame = pd.read_csv(path, usecols=usecols)
            ground = frame["Cloud or Ground"].astype(str).str.casefold().eq("ground")
            timestamps = pd.to_datetime(frame["Date/Time"], utc=True, errors="coerce")
            lat = pd.to_numeric(frame["Latitude"], errors="coerce")
            lon = pd.to_numeric(frame["Longitude"], errors="coerce")
            keep = ground & timestamps.notna() & lat.between(bounds["latitude_min"], bounds["latitude_max"]) & lon.between(bounds["longitude_min"], bounds["longitude_max"])
            selected = pd.DataFrame({"timestamp": timestamps[keep], "lat": lat[keep], "lon": lon[keep]})
            selected["solution_key"] = frame.loc[keep, "Solution Key"].astype(str) if "Solution Key" in frame else ""
            frames.append(selected)
        result = cls(pd.concat(frames, ignore_index=True))
        LOGGER.info("Loaded and indexed MMD strikes in %.2f seconds", time.perf_counter() - started)
        return result

    def _indices(self, start: pd.Timestamp, end: pd.Timestamp) -> np.ndarray:
        start = pd.Timestamp(start).tz_convert("UTC").floor("10min")
        end = pd.Timestamp(end).tz_convert("UTC").ceil("10min")
        arrays = [self.bin_indices.get(int(ts.value)) for ts in pd.date_range(start, end, freq="10min", inclusive="left")]
        arrays = [item for item in arrays if item is not None]
        return np.concatenate(arrays) if arrays else np.empty(0, dtype=np.int64)

    def query_window(self, frame_time: pd.Timestamp, start_minutes: int, end_minutes: int) -> pd.DataFrame:
        frame = pd.Timestamp(frame_time).tz_convert("UTC")
        start, end = frame + pd.Timedelta(minutes=start_minutes), frame + pd.Timedelta(minutes=end_minutes)
        idx = self._indices(start, end)
        if len(idx):
            exact = (self.timestamps_ns[idx] >= start.value) & (self.timestamps_ns[idx] < end.value)
            idx = idx[exact]
        return pd.DataFrame({"timestamp": pd.to_datetime(self.timestamps_ns[idx], utc=True), "lat": self.latitudes[idx], "lon": self.longitudes[idx], "solution_key": self.solution_keys[idx]})

    @staticmethod
    def haversine_km(lat: np.ndarray, lon: np.ndarray, centre_lat: float, centre_lon: float) -> np.ndarray:
        p1, p2 = np.radians(lat), math.radians(centre_lat)
        dp, dl = np.radians(lat - centre_lat), np.radians(lon - centre_lon)
        a = np.sin(dp / 2) ** 2 + np.cos(p1) * math.cos(p2) * np.sin(dl / 2) ** 2
        return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))

    def patch_query(self, frame_time: pd.Timestamp, centre_lat: float, centre_lon: float, patch_size: int,
                    degrees_per_pixel: float, safety_margin_km: float, start_minutes: int, end_minutes: int) -> dict:
        strikes = self.query_window(frame_time, start_minutes, end_minutes)
        half_deg = patch_size * degrees_per_pixel / 2
        lat_margin = safety_margin_km / 111.32
        lon_margin = safety_margin_km / (111.32 * max(math.cos(math.radians(centre_lat)), 0.1))
        if strikes.empty:
            return {"clear": True, "inside_count": 0, "nearest_distance_km": None, "nearest_time_difference_minutes": None}
        inside = (strikes.lat.sub(centre_lat).abs() <= half_deg + lat_margin) & (strikes.lon.sub(centre_lon).abs() <= half_deg + lon_margin)
        distances = self.haversine_km(strikes.lat.to_numpy(), strikes.lon.to_numpy(), centre_lat, centre_lon)
        nearest = int(np.argmin(distances))
        delta = (strikes.iloc[nearest].timestamp - pd.Timestamp(frame_time).tz_convert("UTC")).total_seconds() / 60
        return {"clear": not bool(inside.any()), "inside_count": int(inside.sum()), "nearest_distance_km": float(distances[nearest]), "nearest_time_difference_minutes": float(delta)}



