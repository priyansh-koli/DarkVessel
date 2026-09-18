"""Earth Engine contextual sampling. Not yet built — see the README's Status section.

`without_context` is the stand-in used everywhere until it is: it fills the schema with
absent values so downstream code never has to ask whether context ran.
"""

import geopandas as gpd
import numpy as np
import pandas as pd

from darkvessel.context.schema import CONTEXT_COLUMNS


def without_context(detections: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    filled = detections.copy()
    for column, dtype in CONTEXT_COLUMNS.items():
        empty_value = pd.NA if dtype == "string" else np.nan
        filled[column] = pd.Series([empty_value] * len(filled), index=filled.index, dtype=dtype)
    return filled
