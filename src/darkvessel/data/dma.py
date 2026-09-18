"""Ingestion for the Danish Maritime Authority's historical AIS archive.

The DMA publishes daily CSVs of every AIS message received by the Danish coastal network.
This reads that format and hands rows to `data.ais.clean` — nothing here decides what counts
as a usable report, that is `clean`'s job, so its counts stay the single source of truth.
"""

import geopandas as gpd
import pandas as pd
from shapely import Point

_COLUMNS = {
    "# Timestamp": "timestamp",
    "MMSI": "mmsi",
    "Latitude": "latitude",
    "Longitude": "longitude",
    "Length": "length_m",
    "SOG": "speed_knots",
}


def read_dma_csv(path: str, crs: str) -> gpd.GeoDataFrame:
    """Read one day of the DMA's published AIS export into the project's AIS shape."""
    raw = pd.read_csv(path)
    raw = raw.rename(columns=_COLUMNS)
    raw["timestamp"] = pd.to_datetime(raw["timestamp"], utc=True)
    raw["mmsi"] = raw["mmsi"].astype("string")
    if "speed_knots" in raw.columns:
        raw["speed_ms"] = raw["speed_knots"] * 0.514444

    points = gpd.GeoSeries(
        [Point(lon, lat) for lon, lat in zip(raw["longitude"], raw["latitude"])],
        crs="EPSG:4326",
    ).to_crs(crs)

    columns = [c for c in ("mmsi", "timestamp", "length_m", "speed_ms") if c in raw.columns]
    return gpd.GeoDataFrame(raw[columns], geometry=points, crs=crs)
