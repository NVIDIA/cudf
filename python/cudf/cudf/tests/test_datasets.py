# SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pandas as pd
import pytest

import cudf
from cudf.testing import assert_eq


def test_dataset_timeseries():
    gdf1 = cudf.datasets.timeseries(
        dtypes={"x": int, "y": float}, freq="120s", nulls_frequency=0.3, seed=1
    )
    gdf2 = cudf.datasets.timeseries(
        dtypes={"x": int, "y": float}, freq="120s", nulls_frequency=0.3, seed=1
    )

    assert_eq(gdf1, gdf2)

    assert gdf1["x"].head().dtype == int
    assert gdf1["y"].head().dtype == float
    assert gdf1.index.name == "timestamp"

    gdf = cudf.datasets.timeseries(
        "2000",
        "2010",
        freq="2h",
        dtypes={"value": float, "name": "category", "id": int},
        nulls_frequency=0.7,
        seed=1,
    )

    assert gdf["value"].head().dtype == float
    assert gdf["id"].head().dtype == int
    assert gdf["name"].head().dtype == "category"

    gdf = cudf.datasets.randomdata()
    assert gdf["id"].head().dtype == int
    assert gdf["x"].head().dtype == float
    assert gdf["y"].head().dtype == float
    assert len(gdf) == 10

    gdf = cudf.datasets.randomdata(
        nrows=20, dtypes={"id": int, "a": int, "b": float}
    )
    assert gdf["id"].head().dtype == int
    assert gdf["a"].head().dtype == int
    assert gdf["b"].head().dtype == float
    assert len(gdf) == 20


@pytest.mark.parametrize(
    "start, end",
    [
        ("2020-01-03", "2020-01-01"),
        (pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-01")),
    ],
)
def test_dataset_timeseries_empty_range(start, end):
    gdf = cudf.datasets.timeseries(
        start=start,
        end=end,
        freq="D",
        dtypes={"value": float},
        nulls_frequency=0.5,
        seed=1,
    )
    assert len(gdf) == 0
    assert list(gdf.columns) == ["value"]
    assert gdf.index.name == "timestamp"


@pytest.mark.parametrize("nulls_frequency", [0.5, 1])
def test_dataset_timeseries_exclusive_end_null_mask(nulls_frequency):
    gdf = cudf.datasets.timeseries(
        start="2020-01-01",
        end=pd.Timestamp("2020-01-03"),
        freq="D",
        dtypes={"value": float, "count": int},
        nulls_frequency=nulls_frequency,
        seed=1,
    )
    assert len(gdf) == 2
    assert gdf.index.name == "timestamp"
    if nulls_frequency == 1:
        assert gdf.isna().all().all()


def test_make_bool():
    n = 10
    state = np.random.RandomState(12)
    arr = cudf.datasets.make_bool(n, state)
    assert np.all(np.isin(arr, [True, False]))
    assert arr.size == n
    assert arr.dtype == bool
