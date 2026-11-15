# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import warnings

import pandas as xpd
import pytest

import cudf
from cudf.core._compat import PANDAS_GE_31
from cudf.pandas import LOADED
from cudf.testing import assert_eq

if not LOADED:
    raise ImportError("These tests must be run with cudf.pandas loaded")

pd = xpd._fsproxy_slow


def _capture(call):
    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always")
        result = call()
    return result, [
        (record.category.__name__, str(record.message)) for record in records
    ]


@pytest.mark.parametrize(
    "method,kwargs,fast",
    [
        ("drop", {"index": 1, "inplace": False}, True),
        ("dropna", {"inplace": True}, True),
        ("sort_values", {"inplace": True}, False),
        ("rename", {"index": "renamed", "inplace": True}, False),
    ],
)
def test_inplace_warns_once_on_gpu_and_fallback(method, kwargs, fast):
    expected = pd.Series([3, 1, 2])
    actual = xpd.Series([3, 1, 2])
    expected_result, expected_warnings = _capture(
        lambda: getattr(expected, method)(**kwargs)
    )
    actual_result, actual_warnings = _capture(
        lambda: getattr(actual, method)(**kwargs)
    )
    assert expected_warnings == actual_warnings
    if fast:
        assert isinstance(actual._fsproxy_wrapped, cudf.Series)
    assert_eq(expected, actual._fsproxy_slow)
    assert_eq(
        expected_result,
        actual_result._fsproxy_slow
        if hasattr(actual_result, "_fsproxy_slow")
        else actual_result,
    )


@pytest.mark.parametrize(
    "flag",
    [
        "infer_objects",
        "convert_string",
        "convert_integer",
        "convert_boolean",
        "convert_floating",
    ],
)
def test_convert_dtypes_explicit_flag_warns_once(flag):
    expected = pd.Series([1, 2, 3])
    actual = xpd.Series([1, 2, 3])
    expected_result, expected_warnings = _capture(
        lambda: expected.convert_dtypes(**{flag: False})
    )
    actual_result, actual_warnings = _capture(
        lambda: actual.convert_dtypes(**{flag: False})
    )
    assert expected_warnings == actual_warnings
    assert isinstance(actual_result._fsproxy_wrapped, cudf.Series)
    assert_eq(expected_result, actual_result._fsproxy_slow)


@pytest.mark.parametrize(
    "alias", ["weekday", "dayofweek", "dayofyear", "daysinmonth"]
)
def test_datetime_alias_warns_once(alias):
    expected = pd.Series(pd.date_range("2024-02-27", periods=4))
    actual = xpd.Series(xpd.date_range("2024-02-27", periods=4))
    expected_result, expected_warnings = _capture(
        lambda: getattr(expected.dt, alias)
    )
    actual_result, actual_warnings = _capture(
        lambda: getattr(actual.dt, alias)
    )
    assert expected_warnings == actual_warnings
    assert isinstance(actual_result._fsproxy_wrapped, cudf.Series)
    # Some native calendar accessors use narrower integers than pandas.
    assert_eq(expected_result, actual_result._fsproxy_slow, check_dtype=False)


def test_numeric_only_and_describe_warn_once():
    expected = pd.Series([1.0, 2.0, 3.0])
    actual = xpd.Series([1.0, 2.0, 3.0])
    expected_result, expected_warnings = _capture(
        lambda: expected.sum(numeric_only=None)
    )
    actual_result, actual_warnings = _capture(
        lambda: actual.sum(numeric_only=None)
    )
    assert expected_warnings == actual_warnings
    assert_eq(expected_result, actual_result)
    expected_result, expected_warnings = _capture(
        lambda: expected.describe(include="all")
    )
    actual_result, actual_warnings = _capture(
        lambda: actual.describe(include="all")
    )
    assert expected_warnings == actual_warnings
    assert isinstance(actual_result._fsproxy_wrapped, cudf.Series)
    assert_eq(expected_result, actual_result._fsproxy_slow)


@pytest.mark.parametrize(
    "dtype,value", [("int64", 1.5), ("int8", 128), ("Int64", 1.5)]
)
def test_loc_expansion_warns_once_on_gpu(dtype, value):
    if not PANDAS_GE_31 and dtype == "int8":
        pytest.skip("pandas 3.1 widens expanded integer Series to int64")
    expected = pd.Series([1, 2], dtype=dtype)
    actual = xpd.Series([1, 2], dtype=dtype)

    def expand(obj):
        obj.loc[2] = value

    _, expected_warnings = _capture(lambda: expand(expected))
    _, actual_warnings = _capture(lambda: expand(actual))
    assert expected_warnings == actual_warnings
    assert isinstance(actual._fsproxy_wrapped, cudf.Series)
    assert_eq(expected, actual._fsproxy_slow)


@pytest.mark.skipif(
    not PANDAS_GE_31, reason="pandas 3.1 expansion deprecation"
)
def test_loc_boolean_promotion_uses_object_fallback():
    expected = pd.Series([True, False])
    actual = xpd.Series([True, False])

    def expand(obj):
        obj.loc[2] = 3

    _, expected_warnings = _capture(lambda: expand(expected))
    _, actual_warnings = _capture(lambda: expand(actual))
    assert expected_warnings == actual_warnings
    assert_eq(expected, actual._fsproxy_slow)


@pytest.mark.parametrize(
    "kwargs,fast",
    [
        ({"dayfirst": False, "yearfirst": False, "ambiguous": "raise"}, True),
        ({"dayfirst": True}, False),
    ],
)
def test_datetime_constructor_warns_once_per_explicit_flag(kwargs, fast):
    data = ["2024-01-01", "2024-01-02"]
    expected, expected_warnings = _capture(
        lambda: pd.DatetimeIndex(data, **kwargs)
    )
    actual, actual_warnings = _capture(
        lambda: xpd.DatetimeIndex(data, **kwargs)
    )
    assert expected_warnings == actual_warnings
    if fast:
        assert isinstance(actual._fsproxy_wrapped, cudf.DatetimeIndex)
    assert_eq(expected, actual._fsproxy_slow)
