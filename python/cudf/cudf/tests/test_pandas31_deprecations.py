# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import warnings

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

import cudf
from cudf.api.extensions import no_default
from cudf.core._compat import PANDAS_GE_31
from cudf.testing import assert_eq


def _capture(call):
    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always")
        result = call()
    return result, records


def _assert_warning_parity(expected, actual):
    assert [(x.category.__name__, str(x.message)) for x in actual] == [
        (x.category.__name__, str(x.message)) for x in expected
    ]
    assert all(x.filename == __file__ for x in actual)


_METHODS = [
    ("DataFrame", name)
    for name in (
        "drop",
        "rename",
        "reset_index",
        "set_index",
        "dropna",
        "drop_duplicates",
        "sort_values",
        "sort_index",
        "eval",
    )
] + [
    ("Series", name)
    for name in (
        "drop",
        "rename",
        "reset_index",
        "dropna",
        "drop_duplicates",
        "sort_values",
        "sort_index",
    )
]


@pytest.mark.parametrize(
    "kind,method,inplace",
    [
        (kind, method, inplace)
        for kind, method in _METHODS
        for inplace in (None, False, True)
        if not (
            inplace is True
            and (
                method == "sort_values"
                or (kind == "Series" and method == "rename")
            )
        )
    ],
)
def test_inplace_argument_deprecation(kind, method, inplace):
    data = {"a": [3, 1, 2], "b": [30, 10, 20]}
    if kind == "Series":
        data = data["a"]
    expected = getattr(pd, kind)(data, index=["a", "b", "c"])
    actual = cudf.from_pandas(expected)
    kwargs = {} if inplace is None else {"inplace": inplace}
    if method == "drop":
        kwargs["index"] = "b"
    elif method == "rename":
        kwargs["columns" if kind == "DataFrame" else "index"] = (
            {"a": "renamed"} if kind == "DataFrame" else "renamed"
        )
    elif method == "reset_index":
        kwargs["drop"] = True
    elif method == "set_index":
        kwargs["keys"] = "a"
    elif method == "sort_values" and kind == "DataFrame":
        kwargs["by"] = "a"
    elif method == "eval":
        kwargs["expr"] = "result = a + 1"
    expected_result, expected_warnings = _capture(
        lambda: getattr(expected, method)(**kwargs)
    )
    actual_result, actual_warnings = _capture(
        lambda: getattr(actual, method)(**kwargs)
    )
    _assert_warning_parity(expected_warnings, actual_warnings)
    assert_eq(expected, actual)
    if method != "rename" or inplace is not True:
        assert_eq(expected_result, actual_result)


@pytest.mark.parametrize("kind", ["Series", "DataFrame"])
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
@pytest.mark.parametrize("value", [False, True])
def test_convert_dtypes_explicit_flags(kind, flag, value):
    data = [1, 2, None] if kind == "Series" else {"a": [1, 2, None]}
    expected = getattr(pd, kind)(data)
    actual = cudf.from_pandas(expected)
    expected_result, expected_warnings = _capture(
        lambda: expected.convert_dtypes(**{flag: value})
    )
    actual_result, actual_warnings = _capture(
        lambda: actual.convert_dtypes(**{flag: value})
    )
    _assert_warning_parity(expected_warnings, actual_warnings)
    assert_eq(expected_result, actual_result)


@pytest.mark.parametrize("kind", ["Series", "DataFrame"])
def test_convert_dtypes_default_has_no_deprecation(kind):
    data = [1, 2, None] if kind == "Series" else {"a": [1, 2, None]}
    expected = getattr(pd, kind)(data)
    actual = cudf.from_pandas(expected)
    expected_result, expected_warnings = _capture(expected.convert_dtypes)
    actual_result, actual_warnings = _capture(actual.convert_dtypes)
    _assert_warning_parity(expected_warnings, actual_warnings)
    assert_eq(expected_result, actual_result)


@pytest.mark.parametrize("kind", ["Series", "DatetimeIndex"])
@pytest.mark.parametrize(
    "alias,canonical",
    [
        ("weekday", "day_of_week"),
        ("dayofweek", "day_of_week"),
        ("dayofyear", "day_of_year"),
        ("daysinmonth", "days_in_month"),
    ],
)
def test_datetime_alias_deprecation(kind, alias, canonical):
    expected = pd.date_range("2024-02-27", periods=4)
    actual = cudf.from_pandas(expected)
    if kind == "Series":
        expected, actual = pd.Series(expected).dt, cudf.Series(actual).dt
    _, expected_warnings = _capture(lambda: getattr(expected, alias))
    actual_result, actual_warnings = _capture(lambda: getattr(actual, alias))
    canonical_result, canonical_warnings = _capture(
        lambda: getattr(actual, canonical)
    )
    _assert_warning_parity(expected_warnings, actual_warnings)
    assert not canonical_warnings
    assert_eq(canonical_result, actual_result)


@pytest.mark.parametrize(
    "kwargs", [{"include": "all"}, {"exclude": ["int64"]}]
)
def test_series_describe_unused_dtype_filters(kwargs):
    expected = pd.Series([1.0, 2.0, 3.0])
    actual = cudf.from_pandas(expected)
    expected_result, expected_warnings = _capture(
        lambda: expected.describe(**kwargs)
    )
    actual_result, actual_warnings = _capture(
        lambda: actual.describe(**kwargs)
    )
    _assert_warning_parity(expected_warnings, actual_warnings)
    assert_eq(expected_result, actual_result)


@pytest.mark.parametrize("kind", ["Series", "DataFrame"])
@pytest.mark.parametrize(
    "method",
    [
        "mean",
        "min",
        "max",
        "median",
        "skew",
        "kurt",
        "std",
        "var",
        "sum",
        "prod",
    ],
)
@pytest.mark.parametrize("numeric_only", [None, 0, 1, np.bool_(True)])
def test_reduction_nonboolean_numeric_only(kind, method, numeric_only):
    data = [3.0, 1.0, 2.0, 4.0, 5.0]
    expected = getattr(pd, kind)(data if kind == "Series" else {"a": data})
    actual = cudf.from_pandas(expected)
    expected_result, expected_warnings = _capture(
        lambda: getattr(expected, method)(numeric_only=numeric_only)
    )
    actual_result, actual_warnings = _capture(
        lambda: getattr(actual, method)(numeric_only=numeric_only)
    )
    _assert_warning_parity(expected_warnings, actual_warnings)
    assert_eq(expected_result, actual_result)


@pytest.mark.parametrize(
    "dtype,value",
    [
        ("int64", 1.5),
        ("int8", 128),
        ("float32", 1e40),
        ("Int64", 1.5),
        (pd.ArrowDtype(pa.int64()), 1.5),
    ],
)
@pytest.mark.parametrize("kind", ["Series", "DataFrame"])
def test_loc_expansion_dtype_promotion_warning(dtype, value, kind):
    if not PANDAS_GE_31 and kind == "Series" and dtype == "int8":
        pytest.skip("pandas 3.1 widens expanded integer Series to int64")
    series = pd.Series([1, 2], dtype=dtype)
    expected = series if kind == "Series" else series.to_frame("a")
    actual = cudf.from_pandas(expected)

    def expand(obj):
        # pandas probes float32 casts before retaining the float64 result.
        with np.errstate(over="ignore"):
            obj.loc[2] = value

    _, expected_warnings = _capture(lambda: expand(expected))
    _, actual_warnings = _capture(lambda: expand(actual))
    _assert_warning_parity(expected_warnings, actual_warnings)
    assert_eq(expected, actual)


@pytest.mark.parametrize("value", [3, 3.0, None])
def test_loc_expansion_compatible_values_are_quiet(value):
    expected = pd.Series([1, 2])
    actual = cudf.from_pandas(expected)

    def expand(obj):
        obj.loc[2] = value

    _, expected_warnings = _capture(lambda: expand(expected))
    _, actual_warnings = _capture(lambda: expand(actual))
    _assert_warning_parity(expected_warnings, actual_warnings)
    assert_eq(expected, actual)


def test_index_expansion_does_not_warn_about_data_dtype():
    expected = pd.Series([1, 2])
    actual = cudf.from_pandas(expected)

    def expand(obj):
        obj.loc[1.5] = 3

    _, expected_warnings = _capture(lambda: expand(expected))
    _, actual_warnings = _capture(lambda: expand(actual))
    _assert_warning_parity(expected_warnings, actual_warnings)
    assert_eq(expected, actual)


@pytest.mark.skipif(
    not PANDAS_GE_31, reason="pandas 3.1 expansion deprecation"
)
def test_expansion_warning_as_error_preserves_frame():
    actual = cudf.DataFrame({"a": [1.0, 2.0], "b": [1, 2]})
    original = actual.copy(deep=True)
    with warnings.catch_warnings():
        warnings.simplefilter("error", pd.errors.Pandas4Warning)
        with pytest.raises(
            pd.errors.Pandas4Warning, match="incompatible dtype"
        ):
            actual.loc[2] = 1.5
    assert_eq(original, actual)


@pytest.mark.parametrize("dayfirst", [no_default, False])
@pytest.mark.parametrize("yearfirst", [no_default, False])
@pytest.mark.parametrize("ambiguous", [no_default, "raise"])
def test_datetime_constructor_deprecated_flags(dayfirst, yearfirst, ambiguous):
    kwargs = {
        name: value
        for name, value in (
            ("dayfirst", dayfirst),
            ("yearfirst", yearfirst),
            ("ambiguous", ambiguous),
        )
        if value is not no_default
    }
    data = ["2024-01-01", "2024-01-02"]
    expected, expected_warnings = _capture(
        lambda: pd.DatetimeIndex(data, **kwargs)
    )
    actual, actual_warnings = _capture(
        lambda: cudf.DatetimeIndex(data, **kwargs)
    )
    _assert_warning_parity(expected_warnings, actual_warnings)
    assert_eq(expected, actual)
