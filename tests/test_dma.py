"""Ingestion of the Danish Maritime Authority's published AIS CSV export."""

import geopandas as gpd
import pandas as pd
import pytest

from darkvessel.data.dma import read_dma_csv

_CRS = "EPSG:25832"


def _csv(tmp_path, text: str):
    path = tmp_path / "aisdk.csv"
    path.write_text(text)
    return str(path)


def test_published_column_names_are_mapped_to_the_project_shape(tmp_path):
    path = _csv(
        tmp_path,
        "# Timestamp,MMSI,Latitude,Longitude,Length,SOG\n"
        "2026-08-09 05:31:24,219000001,55.7,9.0,140,12.0\n",
    )
    result = read_dma_csv(path, _CRS)
    assert list(result.columns) == ["mmsi", "timestamp", "length_m", "speed_ms", "geometry"]
    assert result["mmsi"].dtype == "string"
    assert result["length_m"].iloc[0] == 140


def test_speed_over_ground_is_converted_from_knots_to_metres_per_second(tmp_path):
    path = _csv(
        tmp_path,
        "# Timestamp,MMSI,Latitude,Longitude,Length,SOG\n"
        "2026-08-09 05:31:24,219000001,55.7,9.0,140,10.0\n",
    )
    result = read_dma_csv(path, _CRS)
    assert result["speed_ms"].iloc[0] == pytest.approx(5.14444)


def test_positions_are_reprojected_into_the_requested_crs(tmp_path):
    """The DMA publishes lat/lon; everything downstream works in a projected CRS, so a
    round-trip back to EPSG:4326 must land on the published coordinates."""
    path = _csv(
        tmp_path,
        "# Timestamp,MMSI,Latitude,Longitude,Length,SOG\n"
        "2026-08-09 05:31:24,219000001,55.7,9.0,140,10.0\n",
    )
    result = read_dma_csv(path, _CRS)
    assert result.crs == _CRS
    back = gpd.GeoSeries(result.geometry, crs=_CRS).to_crs("EPSG:4326")
    assert back.iloc[0].x == pytest.approx(9.0, abs=1e-6)
    assert back.iloc[0].y == pytest.approx(55.7, abs=1e-6)


def test_timestamps_are_parsed_as_utc(tmp_path):
    path = _csv(
        tmp_path,
        "# Timestamp,MMSI,Latitude,Longitude,Length,SOG\n"
        "2026-08-09 05:31:24,219000001,55.7,9.0,140,10.0\n",
    )
    result = read_dma_csv(path, _CRS)
    assert result["timestamp"].iloc[0] == pd.Timestamp("2026-08-09T05:31:24Z")


def test_a_file_without_a_speed_column_simply_has_no_speed(tmp_path):
    """`clean` skips its speed rule when the column is absent — reading must not invent one."""
    path = _csv(
        tmp_path,
        "# Timestamp,MMSI,Latitude,Longitude,Length\n2026-08-09 05:31:24,219000001,55.7,9.0,140\n",
    )
    result = read_dma_csv(path, _CRS)
    assert "speed_ms" not in result.columns


def test_a_one_row_file_survives_numpy_making_size_one_array_conversion_an_error(
    tmp_path, monkeypatch
):
    """pyproj 3.6 converts a size-1 array to a float, which NumPy 1.25+ deprecates.

    When NumPy turns that into an error it will be a TypeError, which pyproj catches and
    retries on its array path. This simulates that future, so if pyproj ever stops catching
    it, this fails here instead of on a real one-row file.
    """
    import warnings

    def future_numpy(message, category, *args, **kwargs):
        if category is DeprecationWarning and "ndim > 0 to a scalar" in str(message):
            raise TypeError("only 0-dimensional arrays can be converted to Python scalars")

    monkeypatch.setattr(warnings, "showwarning", future_numpy)
    path = _csv(
        tmp_path,
        "# Timestamp,MMSI,Latitude,Longitude,Length,SOG\n"
        "2026-08-09 05:31:24,219000001,55.7,9.0,140,12.0\n",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("always")
        result = read_dma_csv(path, _CRS)
    back = gpd.GeoSeries(result.geometry, crs=_CRS).to_crs("EPSG:4326")
    assert back.iloc[0].x == pytest.approx(9.0, abs=1e-6)
    assert back.iloc[0].y == pytest.approx(55.7, abs=1e-6)
