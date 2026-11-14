# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import cupy as cp
import numpy as np
import pandas as pd
import pytest

import cudf
from cudf.core._compat import PANDAS_GE_31
from cudf.testing import assert_eq


@pytest.mark.parametrize("tz", ["Asia/Tokyo", "America/New_York"])
@pytest.mark.parametrize(
    "attribute",
    [
        "quarter",
        "day_of_year",
        "days_in_month",
        "is_leap_year",
        "is_month_start",
        "is_month_end",
        "is_year_end",
    ],
)
def test_timezone_calendar_uses_wall_time(tz, attribute):
    index = pd.DatetimeIndex(
        ["2016-01-01 01:00", "2016-03-31 23:00", "2016-12-31 23:00"], tz=tz
    )
    result = getattr(cudf.DatetimeIndex(index.array), attribute)
    result = (
        result.to_pandas()
        if hasattr(result, "to_pandas")
        else cp.asnumpy(result)
    )
    assert result.tolist() == getattr(index, attribute).tolist()


@pytest.mark.parametrize("klass", [cudf.Series, cudf.DataFrame])
def test_timedelta_sum_overflow(klass):
    half = np.iinfo(np.int64).max // 2
    values = np.array([half, half, half], dtype="m8[ns]")
    obj = klass(values)
    with pytest.raises(pd.errors.OutOfBoundsTimedelta, match="overflow"):
        obj.sum()


def test_timedelta_sum_large_values_cancel():
    maximum = np.iinfo(np.int64).max
    values = np.array([maximum, -maximum, 1], dtype="m8[ns]")
    assert cudf.Series(values).sum() == pd.Series(values).sum()


@pytest.mark.skipif(
    not PANDAS_GE_31, reason="New pandas 3.1 precision behavior"
)
@pytest.mark.parametrize("unit", ["s", "ms", "us", "ns"])
def test_total_seconds_scalar_precision(unit):
    values = np.array(
        [0, 2**53 + 1, -(2**53) - 1, np.iinfo(np.int64).min, 1, 0],
        dtype=f"m8[{unit}]",
    )
    expected = pd.Series(values).iloc[1:].dt.total_seconds()
    result = cudf.Series(values).iloc[1:].dt.total_seconds()
    assert_eq(expected, result, check_exact=True)


@pytest.mark.skipif(not PANDAS_GE_31, reason="New pandas 3.1 scalar option")
@pytest.mark.parametrize("python_scalars", [False, True])
@pytest.mark.parametrize("klass", [cudf.Series, cudf.Index, cudf.RangeIndex])
def test_searchsorted_python_scalar_option(klass, python_scalars):
    obj = klass(1, 4) if klass is cudf.RangeIndex else klass([1, 2, 3])
    with (
        cudf.option_context("mode.pandas_compatible", True),
        pd.option_context("future.python_scalars", python_scalars),
    ):
        result = obj.searchsorted(2)
    assert result == 1
    assert type(result) is (int if python_scalars else np.intp)


@pytest.mark.parametrize("unit", ["D", "h", "m", "s", "ms", "us", "ns"])
@pytest.mark.parametrize("dtype", ["int64", "float64", "Int64", "Float64"])
def test_numeric_to_datetime_resolution(unit, dtype):
    source = pd.Series([1, 2], dtype=dtype)
    expected = pd.to_datetime(source, unit=unit)
    result = cudf.to_datetime(cudf.from_pandas(source), unit=unit)
    assert_eq(expected, result)


@pytest.mark.parametrize("unit", ["D", "h", "m", "s", "ms", "us", "ns"])
@pytest.mark.parametrize("dtype", ["float64", "Float64"])
def test_fractional_to_datetime_resolution(unit, dtype):
    source = pd.Series([1.5, 2.5, np.nan], dtype=dtype)
    expected = pd.to_datetime(source, unit=unit)
    result = cudf.to_datetime(cudf.from_pandas(source), unit=unit)
    assert_eq(expected, result)


@pytest.mark.skipif(
    not PANDAS_GE_31, reason="New pandas 3.1 empty aggregation"
)
@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize(
    "aggs", [[], {}, pd.Index([], dtype=object), np.array([], dtype=object)]
)
def test_empty_frame_aggregation(aggs, axis):
    source = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    expected = source.agg(aggs, axis=axis)
    result = cudf.from_pandas(source).agg(aggs, axis=axis)
    assert_eq(expected, result)
