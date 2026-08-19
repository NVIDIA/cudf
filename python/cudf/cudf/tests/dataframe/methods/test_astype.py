# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pandas as pd
import pytest

import cudf
from cudf.testing import assert_eq
from cudf.testing._utils import assert_exceptions_equal


@pytest.mark.parametrize("copy", [True, False])
def test_df_series_dataframe_astype_copy(copy):
    gdf = cudf.DataFrame({"col1": [1, 2], "col2": [3, 4]})
    pdf = gdf.to_pandas()

    result = gdf.astype(dtype="float", copy=copy)
    with pytest.warns(pd.errors.Pandas4Warning):
        expected = pdf.astype(dtype="float", copy=copy)
    assert_eq(result, expected)
    assert_eq(gdf, pdf)

    gsr = cudf.Series([1, 2])
    psr = gsr.to_pandas()

    result = gsr.astype(dtype="float", copy=copy)
    with pytest.warns(pd.errors.Pandas4Warning):
        expected = psr.astype(dtype="float", copy=copy)
    assert_eq(result, expected)
    assert_eq(gsr, psr)

    gsr = cudf.Series([1, 2])
    psr = gsr.to_pandas()

    actual = gsr.astype(dtype="int64", copy=copy)
    with pytest.warns(pd.errors.Pandas4Warning):
        expected = psr.astype(dtype="int64", copy=copy)
    assert_eq(expected, actual)
    assert_eq(gsr, psr)
    actual[0] = 3
    expected[0] = 3
    assert_eq(gsr, psr)


@pytest.mark.parametrize("copy", [True, False])
def test_df_series_dataframe_astype_dtype_dict(copy):
    gdf = cudf.DataFrame({"col1": [1, 2], "col2": [3, 4]})
    pdf = gdf.to_pandas()

    result = gdf.astype(dtype={"col1": "float"}, copy=copy)
    with pytest.warns(pd.errors.Pandas4Warning):
        expected = pdf.astype(dtype={"col1": "float"}, copy=copy)
    assert_eq(result, expected)
    assert_eq(gdf, pdf)

    gsr = cudf.Series([1, 2])
    psr = gsr.to_pandas()

    result = gsr.astype(dtype={None: "float"}, copy=copy)
    with pytest.warns(pd.errors.Pandas4Warning):
        expected = psr.astype(dtype={None: "float"}, copy=copy)
    assert_eq(result, expected)
    assert_eq(gsr, psr)

    assert_exceptions_equal(
        lfunc=psr.astype,
        rfunc=gsr.astype,
        lfunc_args_and_kwargs=([], {"dtype": {"a": "float"}}),
        rfunc_args_and_kwargs=([], {"dtype": {"a": "float"}}),
    )

    gsr = cudf.Series([1, 2])
    psr = gsr.to_pandas()

    actual = gsr.astype({None: "int64"}, copy=copy)
    with pytest.warns(pd.errors.Pandas4Warning):
        expected = psr.astype({None: "int64"}, copy=copy)
    assert_eq(expected, actual)
    assert_eq(gsr, psr)

    actual[0] = 3
    expected[0] = 3
    assert_eq(gsr, psr)


def test_astype_dict():
    gdf = cudf.DataFrame({"a": [1, 2, 3], "b": ["1", "2", "3"]})
    pdf = gdf.to_pandas()

    assert_eq(pdf.astype({"a": "str"}), gdf.astype({"a": "str"}))
    assert_eq(
        pdf.astype({"a": "str", "b": np.int64}),
        gdf.astype({"a": "str", "b": np.int64}),
    )


@pytest.mark.parametrize("dtype", ["int64", "datetime64[ns]", "int8"])
def test_dataframe_astype_preserves_column_dtype(dtype):
    result = cudf.DataFrame([1], columns=cudf.Index([1], dtype=dtype))
    result = result.astype(np.int32).columns
    expected = pd.Index([1], dtype=dtype)
    assert_eq(result, expected)


def test_dataframe_astype_preserves_column_rangeindex():
    result = cudf.DataFrame([1], columns=range(1))
    result = result.astype(np.int32).columns
    expected = pd.RangeIndex(1)
    assert_eq(result, expected)


def test_df_astype_numeric_to_all(
    numeric_types_as_str, all_supported_types_as_str
):
    if "uint" in numeric_types_as_str:
        data = [1, 2, None, 4, 7]
    elif "int" in numeric_types_as_str:
        data = [1, 2, None, 4, -7]
    elif "float" in numeric_types_as_str:
        data = [1.0, 2.0, None, 4.0, np.nan, -7.0]

    gdf = cudf.DataFrame()

    gdf["foo"] = cudf.Series(data, dtype=numeric_types_as_str)
    gdf["bar"] = cudf.Series(data, dtype=numeric_types_as_str)

    insert_data = cudf.Series(data, dtype=numeric_types_as_str)

    expect = cudf.DataFrame()
    expect["foo"] = insert_data.astype(all_supported_types_as_str)
    expect["bar"] = insert_data.astype(all_supported_types_as_str)

    got = gdf.astype(all_supported_types_as_str)

    assert_eq(expect, got)


@pytest.mark.parametrize(
    "as_dtype",
    [
        "int32",
        "float32",
        "category",
        "datetime64[s]",
        "datetime64[ms]",
        "datetime64[us]",
        "datetime64[ns]",
    ],
)
def test_df_astype_string_to_other(as_dtype):
    if "datetime64" in as_dtype:
        # change None to "NaT" after this issue is fixed:
        # https://github.com/NVIDIA/cudf/issues/5117
        data = ["2001-01-01", "2002-02-02", "2000-01-05", None]
    elif as_dtype == "int32":
        data = [1, 2, 3]
    elif as_dtype == "category":
        data = ["1", "2", "3", None]
    elif "float" in as_dtype:
        data = [1.0, 2.0, 3.0, np.nan]

    insert_data = cudf.Series(pd.Series(data, dtype="str"))
    expect_data = cudf.Series(data, dtype=as_dtype)

    gdf = cudf.DataFrame()
    expect = cudf.DataFrame()

    gdf["foo"] = insert_data
    gdf["bar"] = insert_data

    expect["foo"] = expect_data
    expect["bar"] = expect_data

    got = gdf.astype(as_dtype)
    assert_eq(expect, got)


@pytest.mark.parametrize(
    "as_dtype",
    [
        "int64",
        "datetime64[s]",
        "datetime64[us]",
        "datetime64[ns]",
        "str",
        "category",
    ],
)
def test_df_astype_datetime_to_other(as_dtype):
    data = [
        "1991-11-20 00:00:00.000",
        "2004-12-04 00:00:00.000",
        "2016-09-13 00:00:00.000",
        None,
    ]

    gdf = cudf.DataFrame(
        {
            "foo": cudf.Series(data, dtype="datetime64[ms]"),
            "bar": cudf.Series(data, dtype="datetime64[ms]"),
        }
    )
    expect = cudf.DataFrame()

    if as_dtype == "int64":
        expect["foo"] = cudf.Series(
            [690595200000, 1102118400000, 1473724800000, None], dtype="int64"
        )
        expect["bar"] = cudf.Series(
            [690595200000, 1102118400000, 1473724800000, None], dtype="int64"
        )
    elif as_dtype == "str":
        expect["foo"] = cudf.Series(data, dtype="str")
        expect["bar"] = cudf.Series(data, dtype="str")
    elif as_dtype == "category":
        expect["foo"] = cudf.Series(gdf["foo"], dtype="category")
        expect["bar"] = cudf.Series(gdf["bar"], dtype="category")
    else:
        expect["foo"] = cudf.Series(data, dtype=as_dtype)
        expect["bar"] = cudf.Series(data, dtype=as_dtype)

    got = gdf.astype(as_dtype)

    assert_eq(expect, got)


@pytest.mark.parametrize(
    "as_dtype",
    [
        "int32",
        "float32",
        "category",
        "datetime64[s]",
        "datetime64[ms]",
        "datetime64[us]",
        "datetime64[ns]",
        "str",
    ],
)
def test_df_astype_categorical_to_other(as_dtype):
    if "datetime64" in as_dtype:
        data = ["2001-01-01", "2002-02-02", "2000-01-05", "2001-01-01"]
    else:
        data = [1, 2, 3, 1]
    psr = pd.Series(data, dtype="category")
    pdf = pd.DataFrame({"foo": psr, "bar": psr})
    gdf = cudf.DataFrame(pdf)
    assert_eq(pdf.astype(as_dtype), gdf.astype(as_dtype))


@pytest.mark.parametrize("ordered", [True, False])
def test_df_astype_to_categorical_ordered(ordered):
    psr = pd.Series([1, 2, 3, 1], dtype="category")
    pdf = pd.DataFrame({"foo": psr, "bar": psr})
    gdf = cudf.DataFrame(pdf)

    ordered_dtype_pd = pd.CategoricalDtype(
        categories=[1, 2, 3], ordered=ordered
    )
    ordered_dtype_gd = cudf.CategoricalDtype(
        categories=ordered_dtype_pd.categories,
        ordered=ordered_dtype_pd.ordered,
    )

    assert_eq(
        pdf.astype(ordered_dtype_pd).astype("int32"),
        gdf.astype(ordered_dtype_gd).astype("int32"),
    )


def test_empty_df_astype(all_supported_types_as_str):
    df = cudf.DataFrame()
    result = df.astype(dtype=all_supported_types_as_str)
    assert_eq(df, result)
    assert_eq(df.to_pandas().astype(dtype=all_supported_types_as_str), result)


@pytest.mark.parametrize("copy", [True, False])
def test_dataframe_astype_no_copy(copy):
    gdf = cudf.DataFrame({"a": [1, 2], "b": [3, 4]})
    result = gdf.astype("int64", copy=copy)
    assert_eq(result, gdf)
    # Under CoW, copy=False returns a shallow copy (distinct object) rather
    # than self; identity is not guaranteed regardless of copy parameter.
    assert result is not gdf
