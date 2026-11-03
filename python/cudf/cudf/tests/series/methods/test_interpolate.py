# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import pytest

import cudf
from cudf.testing import assert_eq


@pytest.mark.parametrize(
    "data",
    [
        [1.0, 2.0, 3.0],
        [1.0, None, 3.0],
        [None, 2.0, None, 4.0],
        [1.0, None, 3.0, None],
        [None, None, 3.0, 4.0],
        [1.0, 2.0, None, None],
        [0.1, 0.2, 0.3],
    ],
)
def test_interpolate_series(data):
    axis = 0
    method = "linear"
    gsr = cudf.Series(data)
    psr = gsr.to_pandas()

    expect = psr.interpolate(method=method, axis=axis)
    got = gsr.interpolate(method=method, axis=axis)

    assert_eq(expect, got)


def test_interpolate_series_unsorted_index():
    gsr = cudf.Series([2.0, None, 4.0, None, 2.0], index=[1, 2, 3, 2, 1])
    psr = gsr.to_pandas()

    expect = psr.interpolate(method="values")
    got = gsr.interpolate(method="values")

    assert_eq(expect, got)


@pytest.mark.parametrize(
    "data",
    [
        [1.0, 2.0, 3.0, 4.0],
        [None, 2.0, 3.0, 4.0],
        [1.0, 2.0, 3.0, None],
        [None, None, 3.0, 4.0],
        [1.0, 2.0, None, None],
        [1.0, None, 3.0, None],
        [None, 2.0, None, 4.0],
    ],
)
@pytest.mark.parametrize("index", [[0, 1, 2, 3], [0, 2, 4, 6], [0, 3, 4, 9]])
@pytest.mark.parametrize("method", ["index", "values"])
def test_interpolate_series_values_or_index(data, index, method):
    gsr = cudf.Series(data, index=index)
    psr = gsr.to_pandas()

    expect = psr.interpolate(method=method)
    got = gsr.interpolate(method=method)

    assert_eq(expect, got)


def test_interpolate_noop_new_column():
    ser = cudf.Series([1.0, 2.0, 3.0])
    result = ser.interpolate()
    assert ser._column is not result._column
