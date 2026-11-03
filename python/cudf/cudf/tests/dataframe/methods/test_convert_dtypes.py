# SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
import numpy as np
import pandas as pd
import pytest

import cudf
from cudf.testing import assert_eq


def _compare_to_pandas(column_specs, kwargs):
    gdf = cudf.DataFrame(
        {k: cudf.Series(v, dtype=d) for k, (v, d) in column_specs.items()}
    )
    pdf = pd.DataFrame(
        {k: pd.Series(v, dtype=d) for k, (v, d) in column_specs.items()}
    )
    got = gdf.convert_dtypes(**kwargs)
    expected = pdf.convert_dtypes(**kwargs)
    assert_eq(got, expected, check_dtype=True)


def test_convert_dtypes():
    data = {
        "a": [1, 2, 3],
        "b": [1, 2, 3],
        "c": [1.1, 2.2, 3.3],
        "d": [1.0, 2.0, 3.0],
        "e": [1.0, 2.0, 3.0],
        "f": ["a", "b", "c"],
        "g": ["a", "b", "c"],
        "h": ["2001-01-01", "2001-01-02", "2001-01-03"],
    }
    dtypes = [
        "int8",
        "int64",
        "float32",
        "float32",
        "float64",
        "str",
        "category",
        "datetime64[ns]",
    ]
    nullable_columns = list("abcdef")
    non_nullable_columns = list(set(data.keys()).difference(nullable_columns))

    df = pd.DataFrame(
        {
            k: pd.Series(v, dtype=d)
            for k, v, d in zip(data.keys(), data.values(), dtypes, strict=True)
        }
    )
    gdf = cudf.DataFrame(df)
    expect = df[nullable_columns].convert_dtypes()
    got = gdf[nullable_columns].convert_dtypes().to_pandas(nullable=True)
    assert_eq(expect, got)

    with pytest.raises(NotImplementedError):
        # category and datetime64[ns] are not nullable
        gdf[non_nullable_columns].convert_dtypes().to_pandas(nullable=True)


@pytest.mark.parametrize(
    "column_specs",
    [
        {
            "a": ([1, 2, 3], "int32"),
            "b": ([1, 2, 3], "int64"),
            "c": ([1, 2, 3], "uint16"),
        },
        {
            "x": ([1.0, 2.0, 3.0], "float64"),
            "y": ([1.5, 2.5, 3.5], "float64"),
        },
        {
            "s": (["a", "b", "c"], "str"),
            "o": (["a", "b", "c"], "O"),
        },
        {
            "i": ([1, 2, 3], "int32"),
            "f": ([1.5, 2.5, 3.5], "float64"),
            "s": (["a", "b", "c"], "str"),
        },
    ],
)
def test_convert_dtypes_dataframe_multi_column(column_specs):
    _compare_to_pandas(column_specs, {})


@pytest.mark.parametrize("dtype_backend", ["pyarrow", "numpy_nullable"])
def test_convert_dtypes_dataframe_backend(dtype_backend):
    column_specs = {
        "i": ([1, 2, 3], "int64"),
        "f": ([1.5, 2.5, 3.5], "float64"),
        "s": (["a", "b", "c"], "str"),
    }
    _compare_to_pandas(column_specs, {"dtype_backend": dtype_backend})


def test_convert_dtypes_dataframe_param_combinations(
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
    column_specs = {
        "i": ([1, 2, 3], "int32"),
        "f_int": ([1.0, 2.0, 3.0], "float64"),
        "f_nonint": ([1.5, 2.5, 3.5], "float64"),
        "s": (["a", "b", "c"], "str"),
    }
    _compare_to_pandas(column_specs, kwargs)


def test_convert_dtypes_dataframe_returns_copy():
    gdf = cudf.DataFrame(
        {
            "a": cudf.Series([1, 2, 3], dtype="int64"),
            "b": cudf.Series([1.5, 2.5, 3.5], dtype="float64"),
        }
    )
    original = gdf.copy(deep=True)
    result = gdf.convert_dtypes()
    result.loc[:, "a"] = pd.NA
    result.loc[:, "b"] = pd.NA
    assert_eq(gdf, original)


def test_convert_dtypes_dataframe_pyarrow_all_null_column():
    _compare_to_pandas(
        {"a": ([None, None], "O")}, {"dtype_backend": "pyarrow"}
    )


def test_convert_dtypes_dataframe_float_nan_as_null_to_int64():
    _compare_to_pandas({"a": ([10.0, np.nan, 20.0], "float64")}, {})


def test_convert_dtypes_pyarrow_null():
    pytest.importorskip("pyarrow")
    data = {"a": [None, None]}

    expected = pd.DataFrame(data).convert_dtypes(dtype_backend="pyarrow")
    result = cudf.DataFrame(data).convert_dtypes(dtype_backend="pyarrow")

    assert_eq(result.to_pandas(), expected)
