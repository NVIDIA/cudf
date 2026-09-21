# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import cupy as cp
import pandas as pd
import pytest

import cudf


@pytest.mark.parametrize(
    "data,index,dtype",
    [
        ([None, 3, 4], None, "float64"),
        ([None, None], None, "float64"),
        ([1, 2, 3, 4], None, "int64"),
        ([], None, "object"),
        ([None, 3, 4], ["x", "y", "z"], "float64"),
        ([1], None, "int64"),
        ([None], None, "float64"),
        ([None, 5], ["a", "b"], "Int64"),
    ],
)
def test_series_first_last_valid_index(data, index, dtype):
    ps = pd.Series(data, index=index, dtype=dtype)
    gs = cudf.from_pandas(ps)

    assert gs.first_valid_index() == ps.first_valid_index()
    assert gs.last_valid_index() == ps.last_valid_index()


def test_series_first_last_valid_index_cupy_input():
    ps = pd.Series([float("nan"), 3.0, 4.0])
    gs = cudf.Series(cp.array([float("nan"), 3.0, 4.0]))

    assert gs.first_valid_index() == ps.first_valid_index()
    assert gs.last_valid_index() == ps.last_valid_index()


@pytest.mark.parametrize(
    "data,index",
    [
        ({"A": [None, None, 2], "B": [None, 3, 4]}, None),
        ({"A": [None, None, None], "B": [None, None, None]}, None),
        ({"A": [1, 2, 3], "B": [4, 5, 6]}, None),
        ({}, None),
        ({"A": [None, 1], "B": ["x", None]}, ["p", "q"]),
    ],
)
def test_dataframe_first_last_valid_index(data, index):
    pdf = pd.DataFrame(data, index=index)
    gdf = cudf.from_pandas(pdf)

    assert gdf.first_valid_index() == pdf.first_valid_index()
    assert gdf.last_valid_index() == pdf.last_valid_index()
