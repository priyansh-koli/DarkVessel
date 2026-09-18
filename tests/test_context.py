"""Contextual columns: present and empty whether or not the credentialed stage ever ran.

A layer's schema must not depend on whether Earth Engine sampling happened — see
`context/gee_layers.py`.
"""

import geopandas as gpd
import pandas as pd
from shapely import Point

from darkvessel.context.gee_layers import without_context
from darkvessel.context.schema import CONTEXT_COLUMNS


def _detections(count: int = 2) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        {"status": ["dark"] * count},
        geometry=[Point(float(i), 0.0) for i in range(count)],
        crs="EPSG:25832",
    )


def test_every_declared_context_column_is_present():
    result = without_context(_detections())
    assert set(CONTEXT_COLUMNS) <= set(result.columns)


def test_the_declared_dtypes_are_honoured():
    result = without_context(_detections())
    for column, dtype in CONTEXT_COLUMNS.items():
        assert result[column].dtype == dtype


def test_every_value_is_absent_rather_than_a_placeholder_number():
    """A zero here would read as "measured, and it was zero" — distance to shore of 0 m."""
    result = without_context(_detections())
    for column in CONTEXT_COLUMNS:
        assert result[column].isna().all()


def test_an_empty_frame_still_gets_the_columns():
    empty = gpd.GeoDataFrame(
        {"status": pd.Series([], dtype="object")}, geometry=[], crs="EPSG:25832"
    )
    result = without_context(empty)
    assert set(CONTEXT_COLUMNS) <= set(result.columns)


def test_the_input_is_left_alone():
    detections = _detections()
    without_context(detections)
    assert not set(CONTEXT_COLUMNS) & set(detections.columns)
