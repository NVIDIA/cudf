# SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
import numpy as np
import pandas as pd
import pytest

import cudf
from cudf.testing import assert_eq


def _compare_to_pandas(data, dtype, kwargs):
    gs = cudf.Series(data, dtype=dtype)
    ps = pd.Series(data, dtype=dtype)
    got = gs.convert_dtypes(**kwargs)
    expected = ps.convert_dtypes(**kwargs)
    assert_eq(got, expected, check_dtype=True)


@pytest.mark.parametrize(
    "data, dtype",
    [
        ([1, 2, 3], "int8"),
        ([1, 2, 3], "int64"),
        ([1.1, 2.2, 3.3], "float32"),
        ([1.0, 2.0, 3.0], "float32"),
        ([1.0, 2.0, 3.0], "float64"),
        (["a", "b", "c"], "str"),
        (["a", "b", "c"], "category"),
        (["2001-01-01", "2001-01-02", "2001-01-03"], "datetime64[ns]"),
    ],
)
def test_convert_dtypes(data, dtype):
    s = pd.Series(data, dtype=dtype)
    gs = cudf.Series(data, dtype=dtype)
    expect = s.convert_dtypes()

    # because we don't have distinct nullable types, we check that we
    # get the same result if we convert to nullable pandas types:
    nullable = dtype not in ("category", "datetime64[ns]")
    got = gs.convert_dtypes().to_pandas(nullable=nullable)
    assert_eq(expect, got)


def test_convert_integer_false_convert_floating_true():
    data = [1.000000000000000000000000001, 1]
    expected = pd.Series(data).convert_dtypes(
        convert_integer=False, convert_floating=True
    )
    result = (
        cudf.Series(data)
        .convert_dtypes(convert_integer=False, convert_floating=True)
        .to_pandas(nullable=True)
    )
    assert_eq(result, expected)


def test_convert_dtypes_integer_to_nullable(integer_types_as_str):
    _compare_to_pandas([1, 2, 3], integer_types_as_str, {})


def test_convert_dtypes_integer_convert_integer_false(integer_types_as_str):
    _compare_to_pandas(
        [1, 2, 3], integer_types_as_str, {"convert_integer": False}
    )


def test_convert_dtypes_float_noninteger(float_types_as_str):
    _compare_to_pandas([1.5, 2.5, 3.5], float_types_as_str, {})


def test_convert_dtypes_float_integer_like_to_int64(float_types_as_str):
    _compare_to_pandas([1.0, 2.0, 3.0], float_types_as_str, {})


def test_convert_dtypes_float_convert_integer_false(float_types_as_str):
    _compare_to_pandas(
        [1.0, 2.0, 3.0], float_types_as_str, {"convert_integer": False}
    )


def test_convert_dtypes_float_both_false(float_types_as_str):
    _compare_to_pandas(
        [1.0, 2.0, 3.0],
        float_types_as_str,
        {"convert_integer": False, "convert_floating": False},
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"convert_boolean": False},
        {
            "convert_boolean": False,
            "convert_integer": False,
            "convert_floating": False,
        },
    ],
)
def test_convert_dtypes_bool(kwargs):
    _compare_to_pandas([True, False, True], "bool", kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"convert_string": False, "infer_objects": True},
        {"convert_string": False, "infer_objects": False},
    ],
)
def test_convert_dtypes_object_strings(kwargs):
    _compare_to_pandas(["a", "b", "c"], "O", kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"convert_string": False},
        {"convert_string": False, "infer_objects": False},
    ],
)
def test_convert_dtypes_string_dtype(kwargs):
    _compare_to_pandas(["a", "b", "c"], "str", kwargs)


def test_convert_dtypes_already_nullable_unchanged(
    all_supported_pandas_nullable_extension_dtypes,
):
    scalar, dtype = all_supported_pandas_nullable_extension_dtypes
    _compare_to_pandas([scalar, scalar, scalar], dtype, {})


@pytest.mark.parametrize(
    "data, dtype",
    [
        (["a", "b", "c"], "category"),
        (["2001-01-01", "2001-01-02", "2001-01-03"], "datetime64[ns]"),
    ],
)
def test_convert_dtypes_non_nullable_kept(data, dtype):
    _compare_to_pandas(data, dtype, {})


@pytest.mark.parametrize(
    "values, dtype",
    [
        ([1, 2, 3], "int64"),
        ([1, 2, 3], "int32"),
        ([1.0, 2.0, 3.0], "float64"),
        ([1.5, 2.5, 3.5], "float64"),
        ([True, False, True], "bool"),
        (["a", "b", "c"], "str"),
    ],
)
def test_convert_dtypes_returns_copy(values, dtype):
    gs = cudf.Series(values, dtype=dtype)
    original = gs.copy(deep=True)
    result = gs.convert_dtypes()
    result[result.notna()] = pd.NA
    assert_eq(gs, original)


@pytest.mark.parametrize(
    "data, dtype",
    [
        ([1, 2, 3], "int64"),
        ([1, 2, 3], "int32"),
        ([1.5, 2.5, 3.5], "float64"),
        (["a", "b", "c"], "str"),
    ],
)
def test_convert_dtypes_pyarrow_backend(data, dtype):
    _compare_to_pandas(data, dtype, {"dtype_backend": "pyarrow"})


def test_convert_dtypes_pyarrow_all_null_to_pa_null():
    _compare_to_pandas([None, None], "O", {"dtype_backend": "pyarrow"})


def test_convert_dtypes_float_nan_as_null_converts_to_int():
    _compare_to_pandas([10.0, np.nan, 20.0], "float64", {})


def test_convert_dtypes_float_preserved_nan_blocks_int_conversion():
    with pd.option_context("future.distinguish_nan_and_na", True):
        gs = cudf.Series(
            [10.0, np.nan, 20.0], dtype="float64", nan_as_null=False
        )
        ps = pd.Series([10.0, np.nan, 20.0], dtype="float64")
        assert_eq(
            gs.convert_dtypes(),
            ps.convert_dtypes(),
            check_dtype=True,
        )


def test_convert_dtypes_float_with_null_to_int64():
    _compare_to_pandas([10.0, None, 20.0], "float64", {})


def test_convert_dtypes_float_nonint_values_with_null():
    _compare_to_pandas([10.5, None, 20.5], "float64", {})


def test_convert_dtypes_pandas_compatible_mode():
    with cudf.option_context("mode.pandas_compatible", True):
        _compare_to_pandas([1, 2, 3], "int32", {})


@pytest.mark.parametrize(
    "data, dtype",
    [
        ([1, 2, 3], "int32"),
        ([1, 2, 3], "int64"),
        ([1.0, 2.0, 3.0], "float64"),
        ([1.5, 2.5, 3.5], "float64"),
        ([True, False, True], "bool"),
        (["a", "b", "c"], "str"),
        (["a", "b", "c"], "O"),
    ],
)
def test_convert_dtypes_matches_pandas_all_param_combinations(
    data,
    dtype,
    infer_objects,
    convert_string,
    convert_integer,
    convert_boolean,
    convert_floating,
):
    kwargs = {
        "infer_objects": infer_objects,
        "convert_string": convert_string,
        "convert_integer": convert_integer,
        "convert_boolean": convert_boolean,
        "convert_floating": convert_floating,
    }
    _compare_to_pandas(data, dtype, kwargs)


def test_convert_dtypes_pyarrow_null():
    pytest.importorskip("pyarrow")
    data = [None, None]

    expected = pd.Series(data).convert_dtypes(dtype_backend="pyarrow")
    result = cudf.Series(data).convert_dtypes(dtype_backend="pyarrow")

    assert_eq(result.to_pandas(), expected)
