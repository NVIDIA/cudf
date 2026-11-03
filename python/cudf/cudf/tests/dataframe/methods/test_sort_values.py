# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import string

import numpy as np
import pandas as pd
import pytest

import cudf
from cudf import DataFrame, option_context
from cudf.testing import assert_eq
from cudf.testing._utils import assert_exceptions_equal, expect_warning_if


def test_dataframe_sort_values(numeric_types_as_str):
    nelem = 25
    rng = np.random.default_rng(seed=0)
    df = DataFrame()
    df["a"] = aa = (100 * rng.random(nelem)).astype(numeric_types_as_str)
    df["b"] = bb = (100 * rng.random(nelem)).astype(numeric_types_as_str)
    sorted_df = df.sort_values(by="a")
    # Check
    sorted_index = np.argsort(aa, kind="mergesort")
    assert_eq(sorted_df.index.values, sorted_index)
    assert_eq(sorted_df["a"].values, aa[sorted_index])
    assert_eq(sorted_df["b"].values, bb[sorted_index])


def test_sort_values_nans_pandas_compat():
    data = {"a": [0, 0, 2, -1], "b": [1, 3, 2, None]}
    with option_context("mode.pandas_compatible", True):
        result = DataFrame(data).sort_values("b", na_position="first")
    expected = pd.DataFrame(data).sort_values("b", na_position="first")
    assert_eq(result, expected)


@pytest.mark.parametrize("index", ["a", "b", ["a", "b"]])
def test_dataframe_sort_values_ignore_index(index, ignore_index):
    if isinstance(index, list) and not ignore_index:
        pytest.skip(
            reason="Unstable sorting by pandas(numpy): https://github.com/pandas-dev/pandas/issues/57531"
        )

    gdf = DataFrame(
        {"a": [1, 3, 5, 2, 4], "b": [1, 1, 2, 2, 3], "c": [9, 7, 7, 7, 1]}
    )
    gdf = gdf.set_index(index)

    pdf = gdf.to_pandas()

    expect = pdf.sort_values(list(pdf.columns), ignore_index=ignore_index)
    got = gdf.sort_values(list(gdf.columns), ignore_index=ignore_index)

    assert_eq(expect, got)


@pytest.mark.parametrize(
    "sliceobj", [slice(1, None), slice(None, -1), slice(1, -1)]
)
def test_dataframe_sort_values_sliced(sliceobj):
    rng = np.random.default_rng(seed=0)
    df = pd.DataFrame({"a": rng.random(20)})

    expect = df[sliceobj]["a"].sort_values()
    gdf = DataFrame(df)
    got = gdf[sliceobj]["a"].sort_values()
    assert (got.to_pandas() == expect).all()


@pytest.mark.parametrize("num_rows", [0, 5])
@pytest.mark.parametrize("na_position", ["first", "last"])
def test_dataframe_multi_column(
    num_rows, numeric_and_temporal_types_as_str, ascending, na_position
):
    num_cols = 5
    rng = np.random.default_rng(seed=0)
    by = list(string.ascii_lowercase[:num_cols])
    pdf = pd.DataFrame()

    for i in range(5):
        colname = string.ascii_lowercase[i]
        data = rng.integers(0, 26, num_rows).astype(
            numeric_and_temporal_types_as_str
        )
        pdf[colname] = data

    gdf = DataFrame(pdf)

    got = gdf.sort_values(by, ascending=ascending, na_position=na_position)
    expect = pdf.sort_values(by, ascending=ascending, na_position=na_position)

    assert_eq(
        got[by].reset_index(drop=True), expect[by].reset_index(drop=True)
    )


@pytest.mark.parametrize("num_rows", [0, 3])
@pytest.mark.parametrize("nulls", ["some", "all"])
@pytest.mark.parametrize("na_position", ["first", "last"])
def test_dataframe_multi_column_nulls(
    num_rows, float_types_as_str, nulls, ascending, na_position
):
    num_cols = 2
    rng = np.random.default_rng(seed=0)
    by = list(string.ascii_lowercase[:num_cols])
    pdf = pd.DataFrame()

    for colname in string.ascii_lowercase[:3]:
        data = rng.integers(0, 26, num_rows).astype(float_types_as_str)
        if nulls == "some":
            idx = np.array([], dtype="int64")
            if num_rows > 0:
                idx = rng.choice(
                    num_rows, size=int(num_rows / 4), replace=False
                )
            data[idx] = np.nan
        elif nulls == "all":
            data[:] = np.nan
        pdf[colname] = data

    gdf = DataFrame(pdf)

    got = gdf.sort_values(by, ascending=ascending, na_position=na_position)
    expect = pdf.sort_values(by, ascending=ascending, na_position=na_position)

    assert_eq(
        got[by].reset_index(drop=True), expect[by].reset_index(drop=True)
    )


@pytest.mark.parametrize("ascending1", [True, False])
@pytest.mark.parametrize("ascending2", [True, False])
@pytest.mark.parametrize("na_position", ["first", "last"])
def test_dataframe_multi_column_nulls_multiple_ascending(
    ascending1, ascending2, na_position
):
    ascending = (ascending1, ascending2)
    pdf = pd.DataFrame(
        {"a": [3, 1, None, 2, 2, None, 1], "b": [1, 2, 3, 4, 5, 6, 7]}
    )
    gdf = DataFrame(pdf)
    expect = pdf.sort_values(
        by=["a", "b"], ascending=ascending, na_position=na_position
    )
    actual = gdf.sort_values(
        by=["a", "b"], ascending=ascending, na_position=na_position
    )

    assert_eq(actual, expect)


@pytest.mark.parametrize(
    "kind", ["quicksort", "mergesort", "heapsort", "stable"]
)
def test_dataframe_sort_values_kind(numeric_types_as_str, kind):
    nelem = 20
    rng = np.random.default_rng(seed=0)
    df = DataFrame()
    df["a"] = aa = (100 * rng.random(nelem)).astype(numeric_types_as_str)
    df["b"] = bb = (100 * rng.random(nelem)).astype(numeric_types_as_str)
    with expect_warning_if(kind != "quicksort", UserWarning):
        sorted_df = df.sort_values(by="a", kind=kind)
    # Check
    sorted_index = np.argsort(aa, kind="mergesort")
    assert_eq(sorted_df.index.values, sorted_index)
    assert_eq(sorted_df["a"].values, aa[sorted_index])
    assert_eq(sorted_df["b"].values, bb[sorted_index])


def test_sort_values_by_index_level():
    df = pd.DataFrame({"a": [1, 3, 2]}, index=pd.Index([1, 3, 2], name="b"))
    cudf_df = DataFrame(df)
    result = cudf_df.sort_values("b")
    expected = df.sort_values("b")
    assert_eq(result, expected)


def test_sort_values_by_ambiguous():
    df = pd.DataFrame({"a": [1, 3, 2]}, index=pd.Index([1, 3, 2], name="a"))
    cudf_df = DataFrame(df)

    assert_exceptions_equal(
        lfunc=df.sort_values,
        rfunc=cudf_df.sort_values,
        lfunc_args_and_kwargs=(["a"], {}),
        rfunc_args_and_kwargs=(["a"], {}),
    )


def test_sort_values_datetime():
    rng = np.random.default_rng(seed=0)
    df = pd.DataFrame(
        {
            "date": np.array(
                [
                    np.datetime64("2016-11-20"),
                    np.datetime64("2020-11-20"),
                    np.datetime64("2019-11-20"),
                    np.datetime64("1918-11-20"),
                    np.datetime64("2118-11-20"),
                ]
            ),
            "vals": rng.random(5),
        }
    )

    gdf = cudf.from_pandas(df)

    s_df = df.sort_values(by="date")
    s_gdf = gdf.sort_values(by="date")

    assert_eq(s_df, s_gdf)


def test_dataframe_loc_duplicate_index_scalar():
    pdf = pd.DataFrame({"a": [1, 2, 3, 4, 5]}, index=[1, 2, 1, 4, 2])
    gdf = cudf.DataFrame(pdf)

    pdf_sorted = pdf.sort_values(by=list(pdf.columns), axis=0)
    gdf_sorted = gdf.sort_values(by=list(gdf.columns), axis=0)

    assert_eq(pdf_sorted, gdf_sorted)


def test_dataframe_sort_values_multindex():
    df = cudf.DataFrame(
        {"a": [2, 1, 2, 1], "b": [1, 2, 1, 2], "c": [4, 3, 2, 1]}
    )
    df = df.set_index(["a", "b"])
    pdf = df.to_pandas()

    expected = pdf.sort_values(by="a", ascending=True)
    with cudf.option_context("mode.pandas_compatible", True):
        result = df.sort_values(by="a", ascending=True)

    assert_eq(expected, result)
