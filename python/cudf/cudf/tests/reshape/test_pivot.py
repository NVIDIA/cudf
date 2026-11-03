# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import pandas as pd
import pytest

import cudf
from cudf.testing import assert_eq


@pytest.mark.parametrize(
    "index, column, data",
    [
        ([], [], []),
        ([0], [0], [0]),
        ([0, 0], [0, 1], [1, 2.0]),
        ([0, 1], [0, 0], [1, 2.0]),
        ([0, 1], [0, 1], [1, 2.0]),
        (["a", "a", "b", "b"], ["c", "d", "c", "d"], [1, 2, 3, 4]),
        (
            ["a", "a", "b", "b", "a"],
            ["c", "d", "c", "d", "e"],
            [1, 2, 3, 4, 5],
        ),
    ],
)
def test_pivot_simple(index, column, data):
    pdf = pd.DataFrame({"index": index, "column": column, "data": data})
    gdf = cudf.from_pandas(pdf)

    expect = pdf.pivot(columns="column", index="index")
    got = gdf.pivot(columns="column", index="index")

    check_index_and_columns = expect.shape != (0, 0)
    assert_eq(
        expect,
        got,
        check_dtype=False,
        check_index_type=check_index_and_columns,
        check_column_type=check_index_and_columns,
    )


def test_pivot_multi_values():
    # from Pandas docs:
    # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.pivot.html
    pdf = pd.DataFrame(
        {
            "foo": ["one", "one", "one", "two", "two", "two"],
            "bar": ["A", "B", "C", "A", "B", "C"],
            "baz": [1, 2, 3, 4, 5, 6],
            "zoo": ["x", "y", "z", "q", "w", "t"],
        }
    )
    gdf = cudf.from_pandas(pdf)
    assert_eq(
        pdf.pivot(index="foo", columns="bar", values=["baz", "zoo"]),
        gdf.pivot(index="foo", columns="bar", values=["baz", "zoo"]),
        check_dtype=False,
    )


@pytest.mark.parametrize(
    "values", ["z", "z123", ["z123"], ["z", "z123", "123z"]]
)
def test_pivot_values(values):
    data = [
        ["A", "a", 0, 0, 0],
        ["A", "b", 1, 1, 1],
        ["A", "c", 2, 2, 2],
        ["B", "a", 0, 0, 0],
        ["B", "b", 1, 1, 1],
        ["B", "c", 2, 2, 2],
        ["C", "a", 0, 0, 0],
        ["C", "b", 1, 1, 1],
        ["C", "c", 2, 2, 2],
    ]
    columns = ["x", "y", "z", "z123", "123z"]
    pdf = pd.DataFrame(data, columns=columns)
    cdf = cudf.DataFrame(data, columns=columns)
    expected = pd.pivot(pdf, index="x", columns="y", values=values)
    actual = cudf.pivot(cdf, index="x", columns="y", values=values)
    assert_eq(
        expected,
        actual,
        check_dtype=False,
    )


def test_pivot_duplicate_error():
    gdf = cudf.DataFrame(
        {"a": [0, 1, 2, 2], "b": [1, 2, 3, 3], "d": [1, 2, 3, 4]}
    )
    with pytest.raises(ValueError):
        gdf.pivot(index="a", columns="b")
    with pytest.raises(ValueError):
        gdf.pivot(index="b", columns="a")


@pytest.mark.parametrize("index", [["ix"], ["ix", "foo"]])
@pytest.mark.parametrize("columns", [["col"], ["col", "baz"]])
def test_pivot_list_like_index_columns(index, columns):
    data = {
        "bar": ["x", "y", "z", "w"],
        "col": ["a", "b", "a", "b"],
        "foo": [1, 2, 3, 4],
        "ix": [1, 1, 2, 2],
        "baz": [0, 0, 0, 0],
    }
    pd_df = pd.DataFrame(data)
    cudf_df = cudf.DataFrame(data)
    result = cudf_df.pivot(columns=columns, index=index)
    expected = pd_df.pivot(columns=columns, index=index)
    assert_eq(result, expected)
