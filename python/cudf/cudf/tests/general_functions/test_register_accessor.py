# SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import pandas as pd
import pytest

import cudf
from cudf.testing import assert_eq


@cudf.api.extensions.register_dataframe_accessor("point")
@pd.api.extensions.register_dataframe_accessor("point")
class PointsAccessor:
    def __init__(self, obj):
        self._validate(obj)
        self._obj = obj

    @staticmethod
    def _validate(obj):
        cols = obj.columns
        if not all(vertex in cols for vertex in ["x", "y"]):
            raise AttributeError("Must have vertices 'x', 'y'.")

    @property
    def bounding_box(self):
        xs, ys = self._obj["x"], self._obj["y"]
        min_x, min_y, max_x, max_y = xs.min(), ys.min(), xs.max(), ys.max()

        return (min_x, min_y, max_x, max_y)


def test_dataframe_accessor():
    gdf = cudf.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
    pdf = gdf.to_pandas()

    assert_eq(gdf.point.bounding_box, pdf.point.bounding_box)


def test_dataframe_accessor_identity():
    """Test that different objects produce different accessor instances."""
    gdf1 = cudf.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
    gdf2 = gdf1.copy()

    assert gdf1.point is not gdf2.point


@pd.api.extensions.register_index_accessor("odd")
@pd.api.extensions.register_series_accessor("odd")
@cudf.api.extensions.register_index_accessor("odd")
@cudf.api.extensions.register_series_accessor("odd")
class OddRowAccessor:
    def __init__(self, obj):
        self._obj = obj

    def __getitem__(self, i):
        return self._obj[2 * i - 1]


@pytest.mark.parametrize("klass", [cudf.Index, cudf.Series])
def test_index_series_accessor(klass):
    obj = klass([1, 2, 3])
    pobj = obj.to_pandas()
    assert_eq(obj.odd[1], pobj.odd[1])


def test_accessor_space_separate():
    data = [1, 2, 3]
    gdf = cudf.DataFrame(data)
    gidx = cudf.Index(data)
    gs = cudf.Series(data)
    assert not id(gdf._accessors) == id(gidx._accessors)
    assert not id(gidx._accessors) == id(gs._accessors)
    assert not id(gdf._accessors) == id(gs._accessors)
