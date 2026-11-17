# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import decimal

import numpy as np
import pandas as pd
import pytest

import cudf
from cudf.testing import assert_eq


@pytest.mark.parametrize(
    "arr",
    [
        np.random.default_rng(seed=0).normal(-100, 100, 10),
        np.random.default_rng(seed=0).integers(-50, 50, 10),
        np.zeros(10),
        np.repeat([-0.6459412758761901], 10),
        np.repeat(np.nan, 10),
        np.array([1.123, 2.343, np.nan, 0.0]),
        np.arange(-100.5, 101.5, 1),
    ],
)
@pytest.mark.parametrize("decimals", [-3, -1, 0, 1, 12, np.int8(1)])
def test_series_round(arr, decimals, nan_as_null):
    pser = pd.Series(arr)
    ser = cudf.Series(arr, nan_as_null=nan_as_null)
    result = ser.round(decimals)
    expected = pser.round(decimals)

    assert_eq(result, expected)


def test_series_round_half_up():
    s = cudf.Series([0.0, 1.0, 1.2, 1.7, 0.5, 1.5, 2.5, None])
    expect = cudf.Series([0.0, 1.0, 1.0, 2.0, 1.0, 2.0, 3.0, None])
    got = s.round(how="half_up")
    assert_eq(expect, got)


@pytest.mark.parametrize(
    "series_data",
    [
        [1.0, None, np.nan, 4.0],
        [1.24430, None, np.nan, 4.423530],
        [1.24430, np.nan, 4.423530],
        [-1.24430, np.nan, -4.423530],
        np.repeat(np.nan, 100),
    ],
)
@pytest.mark.parametrize("decimal", [0, 1, 3])
def test_round_nan_as_null_false(series_data, decimal):
    series = cudf.Series(series_data, nan_as_null=False)
    pser = series.to_pandas()
    result = series.round(decimal)
    expected = pser.round(decimal)
    assert_eq(result, expected, atol=1e-10)


@pytest.mark.parametrize(
    "data, dtype, decimals, expected_half_up, expected_half_even",
    [
        (
            [1.234, 2.345, 3.456],
            cudf.Decimal32Dtype(precision=5, scale=3),
            2,
            [1.23, 2.35, 3.46],
            [1.23, 2.34, 3.46],
        ),
        (
            [1.234, 2.345, 3.456],
            cudf.Decimal32Dtype(precision=5, scale=3),
            0,
            [1.0, 2.0, 3.0],
            [1.0, 2.0, 3.0],
        ),
        (
            [1.234, 2.345, 3.456],
            cudf.Decimal32Dtype(precision=5, scale=3),
            3,
            [1.234, 2.345, 3.456],
            [1.234, 2.345, 3.456],
        ),
        (
            [1.234567, 2.345678, 3.456789],
            cudf.Decimal64Dtype(precision=10, scale=6),
            4,
            [1.2346, 2.3457, 3.4568],
            [1.2346, 2.3457, 3.4568],
        ),
        (
            [1.234567, 2.345678, 3.456789],
            cudf.Decimal64Dtype(precision=10, scale=6),
            2,
            [1.23, 2.35, 3.46],
            [1.23, 2.35, 3.46],
        ),
        (
            [1.234567, 2.345678, 3.456789],
            cudf.Decimal64Dtype(precision=10, scale=6),
            6,
            [1.234567, 2.345678, 3.456789],
            [1.234567, 2.345678, 3.456789],
        ),
    ],
)
def test_series_round_decimal(
    data, dtype, decimals, expected_half_up, expected_half_even
):
    ser = cudf.Series(data).astype(dtype)

    result_half_up = ser.round(decimals=decimals, how="half_up").astype(dtype)
    expected_ser_half_up = cudf.Series(expected_half_up).astype(dtype)
    assert_eq(result_half_up, expected_ser_half_up)

    result_half_even = ser.round(decimals=decimals, how="half_even").astype(
        dtype
    )
    expected_ser_half_even = cudf.Series(expected_half_even).astype(dtype)
    assert_eq(result_half_even, expected_ser_half_even)


@pytest.mark.parametrize(
    "data",
    [
        [1.2234242333234, 323432.3243423, np.nan],
        pytest.param(
            pd.Series([34224, 324324, 324342], dtype="datetime64[ns]"),
            marks=pytest.mark.filterwarnings(
                "ignore:obj.round has no effect:UserWarning"
            ),
        ),
        pd.Series([224.242, None, 2424.234324], dtype="category"),
        [
            decimal.Decimal("342.3243234234242"),
            decimal.Decimal("89.32432497687622"),
            None,
        ],
    ],
)
@pytest.mark.parametrize("digits", [0, 1, 7])
def test_series_round_builtin(data, digits):
    ps = pd.Series(data)
    gs = cudf.from_pandas(ps, nan_as_null=False)

    # TODO: Remove `to_frame` workaround
    # after following issue is fixed:
    # https://github.com/pandas-dev/pandas/issues/55114
    expected = round(ps.to_frame(), digits)[0]
    expected.name = None
    actual = round(gs, digits)

    assert_eq(expected, actual)
