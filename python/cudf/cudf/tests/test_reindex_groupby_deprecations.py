# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import warnings

import pandas as pd
import pytest

import cudf
from cudf.core._compat import PANDAS_GE_31
from cudf.testing import assert_eq


def _call_with_warnings(call):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = call()
    return result, [(type(w.message), str(w.message)) for w in caught]


@pytest.mark.parametrize("series", [False, True])
@pytest.mark.parametrize("key", [0, -1, [0, 1], slice(None, 2)])
def test_groupby_nth_indexer_deprecation(series, key):
    frame = pd.DataFrame({"a": [1, 1, 2], "b": [1, 2, 3]})
    expected_group = frame.groupby("a")
    actual_group = cudf.from_pandas(frame).groupby("a")
    if series:
        expected_group = expected_group["b"]
        actual_group = actual_group["b"]
    expected, expected_warnings = _call_with_warnings(
        lambda: expected_group.nth[key]
    )
    actual, actual_warnings = _call_with_warnings(
        lambda: actual_group.nth[key]
    )
    assert actual_warnings == expected_warnings
    assert_eq(actual, expected)
    called, called_warnings = _call_with_warnings(
        lambda: actual_group.nth(key)
    )
    assert not called_warnings
    assert_eq(called, expected)


@pytest.mark.parametrize("method", ["cummin", "cummax"])
@pytest.mark.parametrize("series", [False, True])
@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"numeric_only": False},
        {"skipna": True},
        {"dtype": None},
        {"axis": 0},
        {"extra": 1},
    ],
)
def test_groupby_cumulative_extra_keyword_warning(method, series, kwargs):
    def calculate(module):
        frame = module.DataFrame({"a": [1, 1, 2], "b": [1.0, 2.0, 3.0]})
        obj = frame.b if series else frame[["b"]]
        return getattr(obj.groupby(frame.a), method)(**kwargs)

    expected, expected_warnings = _call_with_warnings(lambda: calculate(pd))
    actual, actual_warnings = _call_with_warnings(lambda: calculate(cudf))
    assert actual_warnings == expected_warnings
    assert_eq(actual, expected)


@pytest.mark.parametrize(
    "dtype,value",
    [
        ("datetime64[ns]", "2001-01-01"),
        ("datetime64[us]", "2001-01-01"),
        ("datetime64[us]", "3000-01-01"),
        ("timedelta64[us]", "1 day"),
    ],
)
@pytest.mark.parametrize("frame", [False, True])
@pytest.mark.parametrize("add_row", [False, True])
def test_native_temporal_string_reindex_warning(dtype, value, frame, add_row):
    def source(module):
        obj = module.Series([None], dtype=dtype)
        return obj.to_frame() if frame else obj

    index = [0, 1] if add_row else [0]
    _, expected_warnings = _call_with_warnings(
        lambda: source(pd).reindex(index, fill_value=value)
    )
    original = source(cudf)
    actual, actual_warnings = _call_with_warnings(
        lambda: original.reindex(index, fill_value=value)
    )
    assert actual_warnings == expected_warnings
    assert original.isna().all().all() if frame else original.isna().all()
    if add_row:
        actual_value = actual.iloc[-1, 0] if frame else actual.iloc[-1]
        expected_value = (
            pd.Timestamp(value)
            if dtype.startswith("datetime")
            else pd.Timedelta(value)
        )
        assert actual_value == expected_value


def test_native_resampler_preserves_tuple_column_selection():
    frame = cudf.DataFrame(
        {("a", "b"): [1, 2]}, index=cudf.date_range("2020-01-01", periods=2)
    )
    actual = frame.resample("2D")[("a", "b")].sum()
    expected = frame.to_pandas()[("a", "b")].resample("2D").sum()
    assert_eq(actual, expected, check_freq=False)


@pytest.mark.skipif(
    not PANDAS_GE_31, reason="pandas 3.1 duplicate-level warning"
)
@pytest.mark.parametrize("by", ["f", ["f"], ["f", "f"]])
def test_groupby_duplicate_index_level_name_warning(by):
    def calculate(module):
        index = module.MultiIndex.from_tuples(
            [(1, 1), (1, 2), (1, 3)], names=["f", "f"]
        )
        return (
            module.DataFrame({"a": [1, 1, 2], "b": [3, 4, 5]}, index=index)
            .groupby(by)
            .sum()
        )

    expected, expected_warnings = _call_with_warnings(lambda: calculate(pd))
    actual, actual_warnings = _call_with_warnings(lambda: calculate(cudf))
    assert actual_warnings == expected_warnings
    assert_eq(actual, expected)
