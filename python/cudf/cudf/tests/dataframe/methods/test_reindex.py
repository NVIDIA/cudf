# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import pandas as pd
import pytest

import cudf
from cudf.testing import assert_eq
from cudf.testing._utils import assert_exceptions_equal


@pytest.mark.parametrize("copy", [True, False])
@pytest.mark.parametrize(
    "args,gd_kwargs",
    [
        ([], {}),
        ([[-3, 0, 3, 0, -2, 1, 3, 4, 6]], {}),
        ([[-3, 0, 3, 0, -2, 1, 3, 4, 6]], {}),
        ([[-3, 0, 3, 0, -2, 1, 3, 4, 6]], {"axis": 0}),
        ([["a", "b", "c", "d", "e"]], {"axis": 1}),
        ([], {"labels": [-3, 0, 3, 0, -2, 1, 3, 4, 6], "axis": 0}),
        ([], {"labels": ["a", "b", "c", "d", "e"], "axis": 1}),
        ([], {"labels": [-3, 0, 3, 0, -2, 1, 3, 4, 6], "axis": "index"}),
        ([], {"labels": ["a", "b", "c", "d", "e"], "axis": "columns"}),
        ([], {"index": [-3, 0, 3, 0, -2, 1, 3, 4, 6]}),
        ([], {"columns": ["a", "b", "c", "d", "e"]}),
        (
            [],
            {
                "index": [-3, 0, 3, 0, -2, 1, 3, 4, 6],
                "columns": ["a", "b", "c", "d", "e"],
            },
        ),
    ],
)
def test_dataframe_reindex(copy, args, gd_kwargs):
    reindex_data = cudf.datasets.randomdata(
        nrows=6,
        dtypes={
            "a": "category",
            "c": float,
            "d": str,
        },
    )
    pdf, gdf = reindex_data.to_pandas(), reindex_data
    assert_eq(
        pdf.reindex(*args, **gd_kwargs),
        gdf.reindex(*args, **gd_kwargs, copy=copy),
    )


@pytest.mark.parametrize("fill_value", [-1.0, 0.0, 1.5])
@pytest.mark.parametrize(
    "args,kwargs",
    [
        ([], {}),
        ([[-3, 0, 3, 0, -2, 1, 3, 4, 6]], {}),
        ([[-3, 0, 3, 0, -2, 1, 3, 4, 6]], {}),
        ([[-3, 0, 3, 0, -2, 1, 3, 4, 6]], {"axis": 0}),
        ([["a", "b", "c", "d", "e"]], {"axis": 1}),
        ([], {"labels": [-3, 0, 3, 0, -2, 1, 3, 4, 6], "axis": 0}),
        ([], {"labels": ["a", "b", "c", "d", "e"], "axis": 1}),
        ([], {"labels": [-3, 0, 3, 0, -2, 1, 3, 4, 6], "axis": "index"}),
        ([], {"labels": ["a", "b", "c", "d", "e"], "axis": "columns"}),
        ([], {"index": [-3, 0, 3, 0, -2, 1, 3, 4, 6]}),
        ([], {"columns": ["a", "b", "c", "d", "e"]}),
        (
            [],
            {
                "index": [-3, 0, 3, 0, -2, 1, 3, 4, 6],
                "columns": ["a", "b", "c", "d", "e"],
            },
        ),
    ],
)
def test_dataframe_reindex_fill_value(args, kwargs, fill_value):
    reindex_data_numeric = cudf.datasets.randomdata(
        nrows=6,
        dtypes={"a": float, "b": float, "c": float},
    )
    pdf, gdf = reindex_data_numeric.to_pandas(), reindex_data_numeric
    assert_eq(
        pdf.reindex(*args, **kwargs, fill_value=fill_value),
        gdf.reindex(*args, **kwargs, fill_value=fill_value),
    )


@pytest.mark.parametrize("copy", [True, False])
def test_dataframe_reindex_change_dtype(copy):
    index = pd.date_range("12/29/2009", periods=10, freq="D")
    columns = ["a", "b", "c", "d", "e"]
    gdf = cudf.datasets.randomdata(
        nrows=6, dtypes={"a": "category", "c": float, "d": str}
    )
    pdf = gdf.to_pandas()
    # Validate reindexes both labels and column names when
    # index=index_labels and columns=column_labels
    expected = pdf.reindex(index=index, columns=columns)
    result = gdf.reindex(index=index, columns=columns, copy=copy)
    assert_eq(result, expected, check_freq=False)


@pytest.mark.parametrize("copy", [True, False])
def test_series_categorical_reindex(copy):
    index = [-3, 0, 3, 0, -2, 1, 3, 4, 6]
    gdf = cudf.datasets.randomdata(nrows=6, dtypes={"a": "category"})
    pdf = gdf.to_pandas()
    assert_eq(pdf["a"].reindex(), gdf["a"].reindex(copy=copy))
    assert_eq(pdf["a"].reindex(index), gdf["a"].reindex(index, copy=copy))
    assert_eq(
        pdf["a"].reindex(index=index),
        gdf["a"].reindex(index=index, copy=copy),
    )


@pytest.mark.parametrize("copy", [True, False])
def test_series_float_reindex(copy):
    index = [-3, 0, 3, 0, -2, 1, 3, 4, 6]
    gdf = cudf.datasets.randomdata(nrows=6, dtypes={"c": float})
    pdf = gdf.to_pandas()
    assert_eq(pdf["c"].reindex(), gdf["c"].reindex(copy=copy))
    assert_eq(pdf["c"].reindex(index), gdf["c"].reindex(index, copy=copy))
    assert_eq(
        pdf["c"].reindex(index=index),
        gdf["c"].reindex(index=index, copy=copy),
    )


@pytest.mark.parametrize("copy", [True, False])
def test_series_string_reindex(copy):
    index = [-3, 0, 3, 0, -2, 1, 3, 4, 6]
    gdf = cudf.datasets.randomdata(nrows=6, dtypes={"d": str})
    pdf = gdf.to_pandas()
    assert_eq(pdf["d"].reindex(), gdf["d"].reindex(copy=copy))
    assert_eq(pdf["d"].reindex(index), gdf["d"].reindex(index, copy=copy))
    assert_eq(
        pdf["d"].reindex(index=index),
        gdf["d"].reindex(index=index, copy=copy),
    )


@pytest.mark.parametrize("names", [None, ["a", "b"]])
@pytest.mark.parametrize("klass", [cudf.MultiIndex, pd.MultiIndex])
def test_reindex_multiindex_col_to_multiindex(names, klass):
    idx = pd.Index(
        [("A", "one"), ("A", "two")],
        dtype="object",
    )
    df = pd.DataFrame([[1, 2]], columns=idx)
    gdf = cudf.from_pandas(df)
    midx = klass.from_tuples([("A", "one"), ("A", "three")], names=names)
    result = gdf.reindex(columns=midx)
    expected = gdf.to_pandas().reindex(
        columns=midx.to_pandas() if isinstance(midx, cudf.MultiIndex) else midx
    )
    assert_eq(result, expected)


@pytest.mark.parametrize("names", [None, ["a", "b"]])
@pytest.mark.parametrize("klass", [cudf.MultiIndex, pd.MultiIndex])
def test_reindex_tuple_col_to_multiindex(names, klass):
    idx = pd.Index(
        [("A", "one"), ("A", "two")], dtype="object", tupleize_cols=False
    )
    df = pd.DataFrame([[1, 2]], columns=idx)
    gdf = cudf.from_pandas(df)
    midx = klass.from_tuples([("A", "one"), ("A", "two")], names=names)
    result = gdf.reindex(columns=midx)
    expected = cudf.DataFrame([[1, 2]], columns=midx)
    assert_eq(result, expected)


@pytest.mark.parametrize("name", [None, "foo"])
@pytest.mark.parametrize("klass", [range, cudf.RangeIndex, pd.RangeIndex])
def test_reindex_columns_rangeindex_keeps_rangeindex(name, klass):
    new_columns = klass(3)
    exp_name = None
    if klass is not range:
        new_columns.name = name
        exp_name = name
    df = cudf.DataFrame([[1, 2]])
    result = df.reindex(columns=new_columns).columns
    expected = pd.RangeIndex(3, name=exp_name)
    assert_eq(result, expected)


def test_dataframe_duplicate_index_reindex():
    gdf = cudf.DataFrame({"a": [0, 1, 2, 3]}, index=[0, 0, 1, 1])
    pdf = gdf.to_pandas()

    assert_exceptions_equal(
        gdf.reindex,
        pdf.reindex,
        lfunc_args_and_kwargs=([10, 11, 12, 13], {}),
        rfunc_args_and_kwargs=([10, 11, 12, 13], {}),
    )


def test_dataframe_reindex_keep_colname():
    gdf = cudf.DataFrame([1], columns=cudf.Index([1], name="foo"))
    result = gdf.reindex(index=[0, 1])
    expected = cudf.DataFrame(
        [1, None], columns=cudf.Index([1], name="foo"), index=[0, 1]
    )
    assert_eq(result, expected)


@pytest.mark.parametrize(
    "index_data,name",
    [([10, 13], "a"), ([30, 40, 20], "b"), (["ef"], "c"), ([2, 3], "Z")],
)
def test_dataframe_reindex_with_index_names(index_data, name):
    gdf = cudf.DataFrame(
        {
            "a": [10, 12, 13],
            "b": [20, 30, 40],
            "c": cudf.Series(["ab", "cd", "ef"], dtype="category"),
        }
    )
    if name in gdf.columns:
        gdf = gdf.set_index(name)
    pdf = gdf.to_pandas()

    gidx = cudf.Index(index_data, name=name)
    actual = gdf.reindex(gidx)
    expected = pdf.reindex(gidx.to_pandas())

    assert_eq(actual, expected)

    actual = gdf.reindex(index_data)
    expected = pdf.reindex(index_data)

    assert_eq(actual, expected)


@pytest.mark.parametrize("labels", [[4, 5], [4.0, 5.0]])
def test_reindex_numeric_index_int_float(labels):
    # Reindexing a float-labelled index with int labels (or vice versa)
    # matches by numeric equality instead of bailing to an all-null frame.
    pdf = pd.DataFrame({"a": range(5)}, index=[1.5, 2.0, 3.0, 4.0, 5.0])
    gdf = cudf.from_pandas(pdf)
    assert_eq(pdf.reindex(labels), gdf.reindex(labels), check_index_type=False)
