# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import cupy as cp
import numpy as np
import pandas as pd
import pytest

import cudf
from cudf.testing import assert_eq


@pytest.mark.parametrize(
    "data",
    [
        [1, 1, 2],
        [pd.Timestamp(1), pd.Timestamp(1), pd.Timestamp(2)],
    ],
)
def test_unique_series_obj(data):
    actual = cudf.unique(cudf.Series(data))
    expected = pd.unique(pd.Series(data))
    assert isinstance(expected, np.ndarray)
    assert isinstance(actual, cudf.Series)
    assert_eq(actual, pd.Series(expected, name=actual.name))


@pytest.mark.parametrize(
    "cudf_index,pandas_index",
    [
        (cudf.Index, pd.Index),
        (cudf.MultiIndex, pd.MultiIndex),
        (cudf.DatetimeIndex, pd.DatetimeIndex),
        (cudf.CategoricalIndex, pd.CategoricalIndex),
    ],
)
@pytest.mark.parametrize(
    "data",
    [
        [1, 1, 2],
        [pd.Timestamp(1), pd.Timestamp(1), pd.Timestamp(2)],
    ],
)
def test_unique_index_obj(cudf_index, pandas_index, data):
    if cudf_index == cudf.MultiIndex:
        idx = cudf_index.from_arrays([data, data])
    else:
        idx = cudf_index(data)
    actual = cudf.unique(idx)
    expected = pd.unique(idx.to_pandas())

    isinstance(expected, np.ndarray)
    assert isinstance(actual, cudf_index)

    if cudf_index == cudf.MultiIndex:
        expect = pandas_index.from_arrays(
            [
                [x[0] for x in expected],
                [x[1] for x in expected],
            ],
            names=actual.names,
        )
        assert_eq(actual, expect)
    else:
        assert_eq(actual, cudf_index(expected, name=actual.name))


def test_unique_cupy_ndarray():
    ser = pd.Series(pd.Series([1, 1, 2]))
    arr = np.asarray(ser)
    garr = cp.asarray(ser)

    expected = pd.unique(arr)
    actual = cudf.unique(garr)

    isinstance(expected, np.ndarray)
    isinstance(actual, cp.ndarray)
    assert_eq(actual, expected)


@pytest.mark.parametrize(
    "data",
    [
        ["abc", "def", "abc", "a", "def", None],
        [10, 20, 100, -10, 0, 1, None, 10, 100],
    ],
)
def test_category_dtype_unique(data):
    gs = cudf.Series(data, dtype="category")
    ps = gs.to_pandas()

    actual = cudf.unique(gs)
    expected = pd.unique(ps)

    assert isinstance(expected, pd.Categorical)
    assert isinstance(actual, cudf.Series)
    assert_eq(actual, pd.Series(expected))


def test_unique_fails_value_error():
    df = pd.DataFrame({"foo": [1, 2, 3]})
    with pytest.raises(
        ValueError,
        match=r"Must pass cudf.Series, cudf.Index, or cupy.ndarray object",
    ):
        cudf.unique(cudf.DataFrame(df))


def test_unique_fails_not_implemented_error():
    ser = cudf.Series(["foo", "foo"], dtype="category")
    with cudf.option_context("mode.pandas_compatible", True):
        with pytest.raises(
            NotImplementedError, match=r"cudf.Categorical is not implemented"
        ):
            cudf.unique(ser)
