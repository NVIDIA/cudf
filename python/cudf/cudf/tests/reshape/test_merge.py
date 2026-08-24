# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import itertools
import string

import numpy as np
import pandas as pd
import pytest

import cudf
from cudf.core.dtypes import CategoricalDtype, Decimal64Dtype, Decimal128Dtype
from cudf.testing import assert_eq
from cudf.testing._utils import (
    NUMERIC_TYPES,
    assert_exceptions_equal,
    expect_warning_if,
)
from cudf.utils.dtypes import find_common_type


@pytest.fixture(
    params=(
        "left",
        "inner",
        "outer",
        "right",
        "leftanti",
        "leftsemi",
        "cross",
    )
)
def how(request):
    return request.param


@pytest.fixture(params=[False, True])
def sort(request):
    return request.param


def assert_join_results_equal(expect, got, how, **kwargs):
    if how == "right":
        got = got[expect.columns]

    if isinstance(expect, (pd.Series, cudf.Series)):
        return assert_eq(
            expect.sort_values().reset_index(drop=True),
            got.sort_values().reset_index(drop=True),
            **kwargs,
        )
    elif isinstance(expect, (pd.DataFrame, cudf.DataFrame)):
        if not len(
            expect.columns
        ):  # can't sort_values() on a df without columns
            return assert_eq(expect, got, **kwargs)

        assert_eq(
            expect.sort_values(expect.columns.to_list()).reset_index(
                drop=True
            ),
            got.sort_values(got.columns.to_list()).reset_index(drop=True),
            **kwargs,
        )
    elif isinstance(expect, (pd.Index, cudf.Index)):
        return assert_eq(expect.sort_values(), got.sort_values(), **kwargs)
    else:
        raise ValueError(f"Not a join result: {type(expect).__name__}")


@pytest.mark.parametrize("on", ["key1", ["key1", "key2"], None])
def test_dataframe_merge_on(on):
    rng = np.random.default_rng(seed=0)

    # Make cuDF
    df_left = cudf.DataFrame()
    nelem = 500
    df_left["key1"] = rng.integers(0, 40, nelem)
    df_left["key2"] = rng.integers(0, 50, nelem)
    df_left["left_val"] = np.arange(nelem)

    df_right = cudf.DataFrame()
    nelem = 500
    df_right["key1"] = rng.integers(0, 30, nelem)
    df_right["key2"] = rng.integers(0, 50, nelem)
    df_right["right_val"] = np.arange(nelem)

    # Make pandas DF
    pddf_left = df_left.to_pandas()
    pddf_right = df_right.to_pandas()

    # Expected result (from pandas)
    pddf_joined = pddf_left.merge(pddf_right, on=on, how="left")

    # Test (from cuDF; doesn't check for ordering)
    join_result = df_left.merge(df_right, on=on, how="left")
    join_result_cudf = cudf.merge(df_left, df_right, on=on, how="left")

    join_result["right_val"] = (
        join_result["right_val"].astype(np.float64).fillna(np.nan)
    )

    join_result_cudf["right_val"] = (
        join_result_cudf["right_val"].astype(np.float64).fillna(np.nan)
    )

    for col in list(pddf_joined.columns):
        if col.count("_y") > 0:
            join_result[col] = (
                join_result[col].astype(np.float64).fillna(np.nan)
            )
            join_result_cudf[col] = (
                join_result_cudf[col].astype(np.float64).fillna(np.nan)
            )

    # Test dataframe equality (ignore order of rows and columns)
    cdf_result = (
        join_result.to_pandas()
        .sort_values(list(pddf_joined.columns))
        .reset_index(drop=True)
    )

    pdf_result = pddf_joined.sort_values(
        list(pddf_joined.columns)
    ).reset_index(drop=True)

    assert_join_results_equal(cdf_result, pdf_result, how="left")

    merge_func_result_cdf = (
        join_result_cudf.to_pandas()
        .sort_values(list(pddf_joined.columns))
        .reset_index(drop=True)
    )

    assert_join_results_equal(merge_func_result_cdf, cdf_result, how="left")


def test_dataframe_merge_on_unknown_column():
    rng = np.random.default_rng(seed=0)

    # Make cuDF
    df_left = cudf.DataFrame()
    nelem = 500
    df_left["key1"] = rng.integers(0, 40, nelem)
    df_left["key2"] = rng.integers(0, 50, nelem)
    df_left["left_val"] = np.arange(nelem)

    df_right = cudf.DataFrame()
    nelem = 500
    df_right["key1"] = rng.integers(0, 30, nelem)
    df_right["key2"] = rng.integers(0, 50, nelem)
    df_right["right_val"] = np.arange(nelem)

    with pytest.raises(KeyError) as raises:
        df_left.merge(df_right, on="bad_key", how="left")
    raises.match("bad_key")


def test_dataframe_merge_no_common_column():
    rng = np.random.default_rng(seed=0)

    # Make cuDF
    df_left = cudf.DataFrame()
    nelem = 500
    df_left["key1"] = rng.integers(0, 40, nelem)
    df_left["key2"] = rng.integers(0, 50, nelem)
    df_left["left_val"] = np.arange(nelem)

    df_right = cudf.DataFrame()
    nelem = 500
    df_right["key3"] = rng.integers(0, 30, nelem)
    df_right["key4"] = rng.integers(0, 50, nelem)
    df_right["right_val"] = np.arange(nelem)

    with pytest.raises(ValueError) as raises:
        df_left.merge(df_right, how="left")
    raises.match("No common columns to perform merge on")


def test_dataframe_empty_merge():
    gdf1 = cudf.DataFrame({"a": [], "b": []})
    gdf2 = cudf.DataFrame({"a": [], "c": []})

    expect = cudf.DataFrame({"a": [], "b": [], "c": []})
    got = gdf1.merge(gdf2, how="left", on=["a"])

    assert_join_results_equal(expect, got, how="left")


def test_dataframe_merge_order():
    gdf1 = cudf.DataFrame()
    gdf2 = cudf.DataFrame()
    gdf1["id"] = [10, 11]
    gdf1["timestamp"] = [1, 2]
    gdf1["a"] = [3, 4]

    gdf2["id"] = [4, 5]
    gdf2["a"] = [7, 8]

    gdf = gdf1.merge(gdf2, how="left", on=["id", "a"])

    df1 = pd.DataFrame()
    df2 = pd.DataFrame()
    df1["id"] = [10, 11]
    df1["timestamp"] = [1, 2]
    df1["a"] = [3, 4]

    df2["id"] = [4, 5]
    df2["a"] = [7, 8]

    df = df1.merge(df2, how="left", on=["id", "a"])
    assert_join_results_equal(df, gdf, how="left")


@pytest.mark.parametrize(
    "pairs",
    [
        ("", ""),
        ("", "abc"),
        ("a", "a"),
    ],
)
def test_dataframe_pairs_of_triples(pairs, how):
    if how in {"leftsemi", "leftanti"}:
        pytest.skip(f"{how} not implemented in pandas")
    rng = np.random.default_rng(seed=0)

    pdf_left = pd.DataFrame()
    pdf_right = pd.DataFrame()
    for left_column in pairs[0]:
        pdf_left[left_column] = rng.integers(0, 10, 10)
    for right_column in pairs[1]:
        pdf_right[right_column] = rng.integers(0, 10, 10)
    gdf_left = cudf.from_pandas(pdf_left)
    gdf_right = cudf.from_pandas(pdf_right)
    if not set(pdf_left.columns).intersection(pdf_right.columns):
        with pytest.raises(
            pd.errors.MergeError,
            match="No common columns to perform merge on",
        ):
            pdf_left.merge(pdf_right)
        with pytest.raises(
            ValueError, match="No common columns to perform merge on"
        ):
            gdf_left.merge(gdf_right)
    elif not [value for value in pdf_left if value in pdf_right]:
        with pytest.raises(
            pd.errors.MergeError,
            match="No common columns to perform merge on",
        ):
            pdf_left.merge(pdf_right)
        with pytest.raises(
            ValueError, match="No common columns to perform merge on"
        ):
            gdf_left.merge(gdf_right)
    else:
        pdf_result = pdf_left.merge(pdf_right, how=how)
        gdf_result = gdf_left.merge(gdf_right, how=how)
        assert np.array_equal(gdf_result.columns, pdf_result.columns)
        for column in gdf_result:
            gdf_col_result_sorted = gdf_result[column].fillna(-1).sort_values()
            pd_col_result_sorted = pdf_result[column].fillna(-1).sort_values()
            assert np.array_equal(
                gdf_col_result_sorted.to_pandas().values,
                pd_col_result_sorted.values,
            )


def test_safe_merging_with_left_empty():
    rng = np.random.default_rng(seed=0)

    pairs = ("bcd", "b")
    pdf_left = pd.DataFrame()
    pdf_right = pd.DataFrame()
    for left_column in pairs[0]:
        pdf_left[left_column] = rng.integers(0, 10, 0)
    for right_column in pairs[1]:
        pdf_right[right_column] = rng.integers(0, 10, 5)
    gdf_left = cudf.from_pandas(pdf_left)
    gdf_right = cudf.from_pandas(pdf_right)

    pdf_result = pdf_left.merge(pdf_right)
    gdf_result = gdf_left.merge(gdf_right)
    # Simplify test because pandas does not consider empty Index and RangeIndex
    # to be equivalent. TODO: Allow empty Index objects to have equivalence.
    assert len(pdf_result) == len(gdf_result)


@pytest.mark.parametrize("left_empty", [True, False])
@pytest.mark.parametrize("right_empty", [True, False])
def test_empty_joins(how, left_empty, right_empty):
    if how in {"leftsemi", "leftanti"}:
        pytest.skip(f"{how} not implemented in pandas")

    pdf = pd.DataFrame({"x": [1, 2, 3]})

    if left_empty:
        left = pdf.head(0)
    else:
        left = pdf
    if right_empty:
        right = pdf.head(0)
    else:
        right = pdf

    gleft = cudf.from_pandas(left)
    gright = cudf.from_pandas(right)

    expected = left.merge(right, how=how)
    result = gleft.merge(gright, how=how)
    assert len(expected) == len(result)


def test_merge_left_index_zero():
    left = pd.DataFrame({"x": [1, 2, 3, 4, 5, 6]}, index=[0, 1, 2, 3, 4, 5])
    right = pd.DataFrame(
        {"y": [10, 20, 30, 6, 5, 4]}, index=[0, 1, 2, 3, 4, 6]
    )
    gleft = cudf.from_pandas(left)
    gright = cudf.from_pandas(right)
    pd_merge = left.merge(right, left_on="x", right_on="y")
    gd_merge = gleft.merge(gright, left_on="x", right_on="y")

    assert_join_results_equal(pd_merge, gd_merge, how="left")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"left_index": True, "right_on": "y"},
        {"right_index": True, "left_on": "x"},
        {"left_on": "x", "right_on": "y"},
        {"left_index": True, "right_index": True},
    ],
)
def test_merge_left_right_index_left_right_on_zero_kwargs(kwargs):
    left = pd.DataFrame({"x": [1, 2, 3, 4, 5, 6]}, index=[0, 1, 2, 3, 4, 5])
    right = pd.DataFrame(
        {"y": [10, 20, 30, 6, 5, 4]}, index=[0, 1, 2, 3, 4, 6]
    )
    gleft = cudf.from_pandas(left)
    gright = cudf.from_pandas(right)
    pd_merge = left.merge(right, **kwargs)
    gd_merge = gleft.merge(gright, **kwargs)
    assert_join_results_equal(pd_merge, gd_merge, how="left")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"left_index": True, "right_on": "y"},
        {"right_index": True, "left_on": "x"},
        {"left_on": "x", "right_on": "y"},
        {"left_index": True, "right_index": True},
    ],
)
def test_merge_left_right_index_left_right_on_kwargs(kwargs):
    left = pd.DataFrame({"x": [1, 2, 3, 4, 5, 6]}, index=[1, 2, 3, 4, 5, 6])
    right = pd.DataFrame(
        {"y": [10, 20, 30, 6, 5, 4]}, index=[1, 2, 3, 4, 5, 7]
    )
    gleft = cudf.from_pandas(left)
    gright = cudf.from_pandas(right)
    pd_merge = left.merge(right, **kwargs)
    gd_merge = gleft.merge(gright, **kwargs)
    assert_join_results_equal(pd_merge, gd_merge, how="left")


def test_indicator():
    gdf = cudf.DataFrame({"x": [1, 2, 1]})
    gdf.merge(gdf, indicator=False)

    with pytest.raises(NotImplementedError, match=r".*indicator=False.*"):
        gdf.merge(gdf, indicator=True)


def test_merge_suffixes():
    pdf = cudf.DataFrame({"x": [1, 2, 1]})
    gdf = cudf.DataFrame({"x": [1, 2, 1]})
    assert_join_results_equal(
        gdf.merge(gdf, suffixes=("left", "right")),
        pdf.merge(pdf, suffixes=("left", "right")),
        how="left",
    )

    assert_exceptions_equal(
        lfunc=pdf.merge,
        rfunc=gdf.merge,
        lfunc_args_and_kwargs=([pdf], {"lsuffix": "left", "rsuffix": "right"}),
        rfunc_args_and_kwargs=([gdf], {"lsuffix": "left", "rsuffix": "right"}),
    )


@pytest.mark.parametrize(
    "col1, col2, suffixes, expected_cols",
    [
        (0, 0, ("", "_dup"), ["0", "0_dup"]),
        (0, 0, (None, "_dup"), [0, "0_dup"]),
        ("a", "a", ("_x", None), ["a_x", "a"]),
        ("a", "a", (None, "_x"), ["a", "a_x"]),
        (0.0, 0.0, ("_x", None), ["0.0_x", 0.0]),
        (0, 0, ("_a", None), ["0_a", 0]),
    ],
)
def test_merge_suffix_none_and_non_string(col1, col2, suffixes, expected_cols):
    # A ``None`` suffix leaves that side's (possibly non-string) label
    # unchanged; a string suffix is appended (coercing to a string), matching
    # pandas.
    a = cudf.DataFrame({col1: [1, 2, 3]})
    b = cudf.DataFrame({col2: [4, 5, 6]})
    pa = a.to_pandas()
    pb = b.to_pandas()

    result = a.merge(b, left_index=True, right_index=True, suffixes=suffixes)
    expected = pa.merge(
        pb, left_index=True, right_index=True, suffixes=suffixes
    )
    assert result.columns.tolist() == expected_cols
    assert_eq(result, expected)


@pytest.mark.parametrize("suffixes", [{"_x", "_y"}, {"left": 0, "right": 0}])
def test_merge_suffixes_non_sequence_raises(suffixes):
    # pandas rejects sets/dicts passed as ``suffixes`` with a TypeError.
    a = cudf.DataFrame({"a": [1, 2, 3]})
    b = cudf.DataFrame({"b": [3, 4, 5]})
    with pytest.raises(TypeError, match="Passing 'suffixes' as a"):
        a.merge(b, left_index=True, right_index=True, suffixes=suffixes)


@pytest.mark.parametrize("how", ["left", "right", "inner", "outer"])
def test_merge_left_on_right_index_pandas_semantics(how):
    # For left_on + right_index, the result index is the mapped left index
    # (unmatched rows -> NaN, upcasting a numpy int index to float64 and
    # dropping the name), and the key column is coalesced from the right index.
    pl = pd.DataFrame({"a": [0, 1, 2], "key": [0, 1, 2]}, index=[10, 20, 30])
    pr = pd.DataFrame({"b": [0, 1, 2, 3, 4, 5]})
    gl = cudf.from_pandas(pl)
    gr = cudf.from_pandas(pr)
    expect = pl.merge(pr, left_on="key", right_index=True, how=how)
    got = gl.merge(gr, left_on="key", right_index=True, how=how)
    assert_eq(expect, got)


def test_merge_numeric_vs_string_key_raises():
    # pandas refuses to merge a numeric key against a string key.
    left = cudf.DataFrame({"A": [0, 1, 2]})
    right = cudf.DataFrame({"A": ["0", "1", "2"]})
    with pytest.raises(ValueError, match="You are trying to merge on"):
        left.merge(right, on="A")


def test_merge_unmatched_rows_upcast_int_to_float():
    # Unmatched rows introduce NaN, upcasting a numpy integer column to
    # float64 (matching pandas).
    left = cudf.DataFrame({"key": [1, 2, 3], "a": [1, 2, 3]})
    right = cudf.DataFrame({"key": [1, 2], "b": [4, 5]})
    got = left.merge(right, on="key", how="left")
    assert got["b"].dtype == np.dtype("float64")
    assert_eq(
        left.to_pandas().merge(right.to_pandas(), on="key", how="left"), got
    )


def test_merge_differently_named_keys_keep_own_dtype():
    # Each surviving key retains its own operand's dtype.
    left = cudf.DataFrame({"X": [1, 2, 3]})
    right = cudf.DataFrame({"Y": [1.0, 2.0, 3.0]})
    got = left.merge(right, left_on="X", right_on="Y")
    assert got["X"].dtype == np.dtype("int64")
    assert got["Y"].dtype == np.dtype("float64")


@pytest.mark.parametrize("suffixes", [("_dup", ""), ("", "_dup")])
def test_merge_suffix_duplicate_columns_raises(suffixes):
    # Suffixing that collides with a pre-existing column raises MergeError.
    df1 = cudf.DataFrame({"col1": [1], "col2": [2]})
    df2 = cudf.DataFrame({"col1": [1], "col2": [2], "col2_dup": [3]})
    with pytest.raises(pd.errors.MergeError, match="duplicate columns"):
        df1.merge(df2, on="col1", suffixes=suffixes)


def test_merge_left_on_right_on():
    left = pd.DataFrame({"xx": [1, 2, 3, 4, 5, 6]})
    right = pd.DataFrame({"xx": [10, 20, 30, 6, 5, 4]})

    gleft = cudf.from_pandas(left)
    gright = cudf.from_pandas(right)

    assert_join_results_equal(
        left.merge(right, on="xx"), gleft.merge(gright, on="xx"), how="left"
    )

    assert_join_results_equal(
        left.merge(right, left_on="xx", right_on="xx"),
        gleft.merge(gright, left_on="xx", right_on="xx"),
        how="left",
    )


def test_merge_on_index_retained():
    df = cudf.DataFrame()
    df["a"] = [1, 2, 3, 4, 5]
    df["b"] = ["a", "b", "c", "d", "e"]
    df.index = [5, 3, 4, 2, 1]

    df2 = cudf.DataFrame()
    df2["a2"] = [1, 2, 3, 4, 5]
    df2["res"] = ["a", "b", "c", "d", "e"]

    pdf = df.to_pandas()
    pdf2 = df2.to_pandas()

    gdm = df.merge(df2, left_index=True, right_index=True, how="left")
    pdm = pdf.merge(pdf2, left_index=True, right_index=True, how="left")
    gdm["a2"] = gdm["a2"].astype("float64")
    assert_eq(gdm.sort_index(), pdm.sort_index())


@pytest.mark.parametrize(
    "kwargs",
    [
        {"left_index": True, "right_on": "y"},
        {"right_index": True, "left_on": "x"},
        {"left_on": "x", "right_on": "y"},
    ],
)
def test_merge_left_right_index_left_right_on_kwargs2(kwargs):
    left = pd.DataFrame({"x": [1, 2, 3]}, index=[10, 20, 30])
    right = pd.DataFrame({"y": [10, 20, 30]}, index=[1, 2, 30])
    gleft = cudf.from_pandas(left)
    gright = cudf.from_pandas(right)
    gd_merge = gleft.merge(gright, **kwargs)
    pd_merge = left.merge(right, **kwargs)
    if pd_merge.empty:
        assert gd_merge.empty


@pytest.mark.parametrize("how", ["inner", "outer", "left"])
@pytest.mark.parametrize(
    "on",
    [
        "a",
        ["a", "b"],
        ["b", "a"],
        ["a", "aa", "b"],
        ["b", "a", "aa"],
    ],
)
def test_merge_sort(on, how):
    kwargs = {
        "sort": True,
        "how": how,
        "on": on,
    }
    a = [4, 6, 9, 5, 2, 4, 1, 8, 1]
    b = [9, 8, 7, 8, 3, 9, 7, 9, 2]
    aa = [8, 9, 2, 9, 3, 1, 2, 3, 4]
    left = pd.DataFrame({"a": a, "b": b, "aa": aa})
    right = left.copy(deep=True)

    left.index = [6, 5, 4, 7, 5, 5, 5, 4, 4]
    right.index = [5, 4, 1, 9, 4, 3, 5, 4, 4]

    gleft = cudf.from_pandas(left)
    gright = cudf.from_pandas(right)
    gd_merge = gleft.merge(gright, **kwargs)

    pd_merge = left.merge(right, **kwargs)
    # require the join keys themselves to be sorted correctly
    # the non-key columns will NOT match pandas ordering
    assert_join_results_equal(
        pd_merge[kwargs["on"]], gd_merge[kwargs["on"]], how="left"
    )
    pd_merge = pd_merge.drop(kwargs["on"], axis=1)
    gd_merge = gd_merge.drop(kwargs["on"], axis=1)
    if not pd_merge.empty:
        # check to make sure the non join key columns are the same
        pd_merge = pd_merge.sort_values(list(pd_merge.columns)).reset_index(
            drop=True
        )
        gd_merge = gd_merge.sort_values(list(gd_merge.columns)).reset_index(
            drop=True
        )

    assert_join_results_equal(pd_merge, gd_merge, how="left")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"left_on": ["a"], "left_index": False, "right_index": True},
        {"right_on": ["b"], "left_index": True, "right_index": False},
    ],
)
def test_merge_sort_on_indexes(kwargs):
    left_index = kwargs["left_index"]
    right_index = kwargs["right_index"]
    kwargs["sort"] = True
    a = [4, 6, 9, 5, 2, 4, 1, 8, 1]
    left = pd.DataFrame({"a": a})
    right = pd.DataFrame({"b": a})

    left.index = [6, 5, 4, 7, 5, 5, 5, 4, 4]
    right.index = [5, 4, 1, 9, 4, 3, 5, 4, 4]

    gleft = cudf.from_pandas(left)
    gright = cudf.from_pandas(right)
    gd_merge = gleft.merge(gright, **kwargs)

    if left_index and right_index:
        check_if_sorted = gd_merge[["a", "b"]].to_pandas()
        check_if_sorted.index.name = "index"
        definitely_sorted = check_if_sorted.sort_values(["index", "a", "b"])
        definitely_sorted.index.name = None
        assert_eq(gd_merge, definitely_sorted)
    elif left_index:
        assert gd_merge["b"].is_monotonic_increasing
    elif right_index:
        assert gd_merge["a"].is_monotonic_increasing


def test_join_with_different_names():
    left = pd.DataFrame({"a": [0, 1, 2.0, 3, 4, 5, 9]})
    right = pd.DataFrame({"b": [12, 5, 3, 9.0, 5], "c": [1, 2, 3, 4, 5.0]})
    gleft = cudf.from_pandas(left)
    gright = cudf.from_pandas(right)
    pd_merge = left.merge(right, how="outer", left_on=["a"], right_on=["b"])
    gd_merge = gleft.merge(gright, how="outer", left_on=["a"], right_on=["b"])
    assert_join_results_equal(pd_merge, gd_merge, how="outer")


def test_join_same_name_different_order():
    left = pd.DataFrame({"a": [0, 0], "b": [1, 2]})
    right = pd.DataFrame({"a": [1, 2], "b": [0, 0]})
    gleft = cudf.from_pandas(left)
    gright = cudf.from_pandas(right)
    pd_merge = left.merge(right, left_on=["a", "b"], right_on=["b", "a"])
    gd_merge = gleft.merge(gright, left_on=["a", "b"], right_on=["b", "a"])
    assert_join_results_equal(pd_merge, gd_merge, how="left")


def test_join_empty_table_dtype():
    left = pd.DataFrame({"a": []})
    right = pd.DataFrame({"b": [12, 5, 3, 9.0, 5], "c": [1, 2, 3, 4, 5.0]})
    gleft = cudf.from_pandas(left)
    gright = cudf.from_pandas(right)
    pd_merge = left.merge(right, how="left", left_on=["a"], right_on=["b"])
    gd_merge = gleft.merge(gright, how="left", left_on=["a"], right_on=["b"])
    assert_eq(pd_merge["a"].dtype, gd_merge["a"].dtype)


@pytest.mark.parametrize(
    "kwargs",
    [
        {
            "left_on": ["a", "b"],
            "right_on": ["a", "b"],
            "left_index": False,
            "right_index": False,
        },  # left and right on, no indices
        {
            "left_on": None,
            "right_on": None,
            "left_index": True,
            "right_index": True,
        },  # left_index and right_index, no on
        {
            "left_on": ["a", "b"],
            "right_on": None,
            "left_index": False,
            "right_index": True,
        },  # left on and right_index
        {
            "left_on": None,
            "right_on": ["a", "b"],
            "left_index": True,
            "right_index": False,
        },  # right_on and left_index
    ],
)
def test_merge_multi(kwargs):
    left = cudf.DataFrame(
        {
            "a": [1, 2, 3, 4, 3, 5, 6],
            "b": [1, 3, 5, 7, 5, 9, 0],
            "c": ["o", "p", "q", "r", "s", "t", "u"],
            "d": ["v", "w", "x", "y", "z", "1", "2"],
        }
    )
    right = cudf.DataFrame(
        {
            "a": [0, 9, 3, 4, 3, 7, 8],
            "b": [2, 4, 5, 7, 5, 6, 8],
            "c": ["a", "b", "c", "d", "e", "f", "g"],
            "d": ["j", "i", "j", "k", "l", "m", "n"],
        }
    )

    if (
        kwargs["left_on"] is not None
        and kwargs["right_on"] is not None
        and kwargs["left_index"] is False
        and kwargs["right_index"] is False
    ):
        left = left.set_index(["c", "d"])
        right = right.set_index(["c", "d"])
    elif (
        kwargs["left_on"] is None
        and kwargs["right_on"] is None
        and kwargs["left_index"] is True
        and kwargs["right_index"] is True
    ):
        left = left.set_index(["a", "b"])
        right = right.set_index(["a", "b"])
    elif kwargs["left_on"] is not None and kwargs["right_index"] is True:
        left = left.set_index(["c", "d"])
        right = right.set_index(["a", "b"])
    elif kwargs["right_on"] is not None and kwargs["left_index"] is True:
        left = left.set_index(["a", "b"])
        right = right.set_index(["c", "d"])

    gleft = left.to_pandas()
    gright = right.to_pandas()

    kwargs["sort"] = True
    expect = gleft.merge(gright, **kwargs)
    got = left.merge(right, **kwargs)

    assert_eq(expect.sort_index().index, got.sort_index().index)

    expect.index = range(len(expect))
    got.index = range(len(got))
    expect = expect.sort_values(list(expect.columns))
    got = got.sort_values(list(got.columns))
    expect.index = range(len(expect))
    got.index = range(len(got))

    assert_join_results_equal(expect, got, how="left")


@pytest.fixture
def integer_types_as_str2(integer_types_as_str):
    return integer_types_as_str


def test_typecast_on_join_int_to_int(
    integer_types_as_str, integer_types_as_str2
):
    other_data = ["a", "b", "c"]

    join_data_l = cudf.Series([1, 2, 3], dtype=integer_types_as_str)
    join_data_r = cudf.Series([1, 2, 4], dtype=integer_types_as_str2)

    gdf_l = cudf.DataFrame({"join_col": join_data_l, "B": other_data})
    gdf_r = cudf.DataFrame({"join_col": join_data_r, "B": other_data})

    exp_dtype = find_common_type(
        (np.dtype(integer_types_as_str), np.dtype(integer_types_as_str2))
    )

    exp_join_data = [1, 2]
    exp_other_data = ["a", "b"]
    exp_join_col = cudf.Series(exp_join_data, dtype=exp_dtype)

    expect = cudf.DataFrame(
        {
            "join_col": exp_join_col,
            "B_x": exp_other_data,
            "B_y": exp_other_data,
        }
    )

    got = gdf_l.merge(gdf_r, on="join_col", how="inner")

    assert_join_results_equal(expect, got, how="inner")


@pytest.fixture
def float_types_as_str2(float_types_as_str):
    return float_types_as_str


def test_typecast_on_join_float_to_float(
    float_types_as_str, float_types_as_str2
):
    other_data = ["a", "b", "c", "d", "e", "f"]

    join_data_l = cudf.Series([1, 2, 3, 0.9, 4.5, 6], dtype=float_types_as_str)
    join_data_r = cudf.Series(
        [1, 2, 3, 0.9, 4.5, 7], dtype=float_types_as_str2
    )

    gdf_l = cudf.DataFrame({"join_col": join_data_l, "B": other_data})
    gdf_r = cudf.DataFrame({"join_col": join_data_r, "B": other_data})

    exp_dtype = find_common_type(
        (np.dtype(float_types_as_str), np.dtype(float_types_as_str2))
    )

    if float_types_as_str != float_types_as_str2:
        exp_join_data = [1, 2, 3, 4.5]
        exp_other_data = ["a", "b", "c", "e"]
    else:
        exp_join_data = [1, 2, 3, 0.9, 4.5]
        exp_other_data = ["a", "b", "c", "d", "e"]

    exp_join_col = cudf.Series(exp_join_data, dtype=exp_dtype)

    expect = cudf.DataFrame(
        {
            "join_col": exp_join_col,
            "B_x": exp_other_data,
            "B_y": exp_other_data,
        }
    )

    got = gdf_l.merge(gdf_r, on="join_col", how="inner")

    assert_join_results_equal(expect, got, how="inner")


@pytest.fixture
def numeric_types_as_str2(numeric_types_as_str):
    return numeric_types_as_str


def test_typecast_on_join_mixed_int_float(
    numeric_types_as_str, numeric_types_as_str2
):
    if (
        ("int" in numeric_types_as_str or "long" in numeric_types_as_str)
        and ("int" in numeric_types_as_str2 or "long" in numeric_types_as_str2)
    ) or (
        "float" in numeric_types_as_str and "float" in numeric_types_as_str2
    ):
        pytest.skip("like types not tested in this function")

    other_data = ["a", "b", "c", "d", "e", "f"]

    join_data_l = cudf.Series(
        [1, 2, 3, 0.9, 4.5, 6], dtype=numeric_types_as_str
    )
    join_data_r = cudf.Series(
        [1, 2, 3, 0.9, 4.5, 7], dtype=numeric_types_as_str2
    )

    gdf_l = cudf.DataFrame({"join_col": join_data_l, "B": other_data})
    gdf_r = cudf.DataFrame({"join_col": join_data_r, "B": other_data})

    exp_dtype = find_common_type(
        (np.dtype(numeric_types_as_str), np.dtype(numeric_types_as_str2))
    )

    exp_join_data = [1, 2, 3]
    exp_other_data = ["a", "b", "c"]
    exp_join_col = cudf.Series(exp_join_data, dtype=exp_dtype)

    expect = cudf.DataFrame(
        {
            "join_col": exp_join_col,
            "B_x": exp_other_data,
            "B_y": exp_other_data,
        }
    )

    got = gdf_l.merge(gdf_r, on="join_col", how="inner")

    assert_join_results_equal(expect, got, how="inner")


def test_typecast_on_join_no_float_round():
    other_data = ["a", "b", "c", "d", "e"]

    join_data_l = cudf.Series([1, 2, 3, 4, 5], dtype="int8")
    join_data_r = cudf.Series([1, 2, 3, 4.01, 4.99], dtype="float32")

    gdf_l = cudf.DataFrame({"join_col": join_data_l, "B": other_data})
    gdf_r = cudf.DataFrame({"join_col": join_data_r, "B": other_data})

    exp_join_data = [1, 2, 3, 4, 5]
    exp_Bx = ["a", "b", "c", "d", "e"]
    exp_By = ["a", "b", "c", None, None]
    # A left join keeps the left key's dtype (int8), matching pandas.
    exp_join_col = cudf.Series(exp_join_data, dtype="int8")

    expect = cudf.DataFrame(
        {"join_col": exp_join_col, "B_x": exp_Bx, "B_y": exp_By}
    )

    got = gdf_l.merge(gdf_r, on="join_col", how="left")

    assert_join_results_equal(expect, got, how="left")


@pytest.mark.parametrize(
    "dtypes",
    [
        (np.dtype("int8"), np.dtype("int16")),
        (np.dtype("int16"), np.dtype("int32")),
        (np.dtype("int32"), np.dtype("int64")),
        (np.dtype("uint8"), np.dtype("uint16")),
        (np.dtype("uint16"), np.dtype("uint32")),
        (np.dtype("uint32"), np.dtype("uint64")),
        (np.dtype("float32"), np.dtype("float64")),
        (np.dtype("int32"), np.dtype("float32")),
        (np.dtype("uint32"), np.dtype("float32")),
    ],
)
def test_typecast_on_join_overflow_unsafe(dtypes):
    dtype_l, dtype_r = dtypes
    if dtype_l.kind in {"i", "u"}:
        dtype_l_max = np.iinfo(dtype_l).max
    elif dtype_l.kind == "f":
        dtype_l_max = np.finfo(dtype_r).max

    lhs = cudf.DataFrame({"a": [1, 2, 3, 4, 5]}, dtype=dtype_l)
    rhs = cudf.DataFrame({"a": [1, 2, 3, 4, dtype_l_max + 1]}, dtype=dtype_r)

    p_lhs = lhs.to_pandas()
    p_rhs = rhs.to_pandas()

    with expect_warning_if(
        (dtype_l.kind == "f" and dtype_r.kind in {"i", "u"})
        or (dtype_l.kind in {"i", "u"} and dtype_r.kind == "f"),
        UserWarning,
    ):
        expect = p_lhs.merge(p_rhs, on="a", how="left")
    got = lhs.merge(rhs, on="a", how="left")

    # The dtypes here won't match exactly because pandas does some unsafe
    # conversions (with a warning that we are catching above) that we don't
    # want to match.
    assert_join_results_equal(expect, got, how="left", check_dtype=False)


@pytest.mark.parametrize(
    "dtype",
    [
        Decimal64Dtype(5, 2),
        Decimal64Dtype(7, 5),
        Decimal64Dtype(12, 7),
        Decimal128Dtype(20, 5),
    ],
)
def test_decimal_typecast_inner(dtype):
    other_data = ["a", "b", "c", "d", "e"]

    join_data_l = cudf.Series(["1.6", "9.5", "7.2", "8.7", "2.3"]).astype(
        dtype
    )
    join_data_r = cudf.Series(["1.6", "9.5", "7.2", "4.5", "2.3"]).astype(
        dtype
    )

    gdf_l = cudf.DataFrame({"join_col": join_data_l, "B": other_data})
    gdf_r = cudf.DataFrame({"join_col": join_data_r, "B": other_data})

    exp_join_data = ["1.6", "9.5", "7.2", "2.3"]
    exp_other_data = ["a", "b", "c", "e"]

    exp_join_col = cudf.Series(exp_join_data).astype(dtype)

    expected = cudf.DataFrame(
        {
            "join_col": exp_join_col,
            "B_x": exp_other_data,
            "B_y": exp_other_data,
        }
    )

    got = gdf_l.merge(gdf_r, on="join_col", how="inner")

    assert_join_results_equal(expected, got, how="inner")
    assert_eq(dtype, got["join_col"].dtype)


@pytest.mark.parametrize(
    "dtype",
    [
        Decimal64Dtype(7, 3),
        Decimal64Dtype(9, 5),
        Decimal64Dtype(14, 10),
        Decimal128Dtype(21, 9),
    ],
)
def test_decimal_typecast_left(dtype):
    other_data = ["a", "b", "c", "d"]

    join_data_l = cudf.Series(["95.05", "384.26", "74.22", "1456.94"]).astype(
        dtype
    )
    join_data_r = cudf.Series(
        ["95.05", "62.4056", "74.22", "1456.9472"]
    ).astype(dtype)

    gdf_l = cudf.DataFrame({"join_col": join_data_l, "B": other_data})
    gdf_r = cudf.DataFrame({"join_col": join_data_r, "B": other_data})

    exp_join_data = ["95.05", "74.22", "384.26", "1456.94"]
    exp_other_data_x = ["a", "c", "b", "d"]
    exp_other_data_y = ["a", "c", None, None]

    exp_join_col = cudf.Series(exp_join_data).astype(dtype)

    expected = cudf.DataFrame(
        {
            "join_col": exp_join_col,
            "B_x": exp_other_data_x,
            "B_y": exp_other_data_y,
        }
    )

    got = gdf_l.merge(gdf_r, on="join_col", how="left")

    assert_join_results_equal(expected, got, how="left")
    assert_eq(dtype, got["join_col"].dtype)


@pytest.mark.parametrize(
    "dtype",
    [
        Decimal64Dtype(7, 3),
        Decimal64Dtype(10, 5),
        Decimal64Dtype(18, 9),
        Decimal128Dtype(22, 8),
    ],
)
def test_decimal_typecast_outer(dtype):
    other_data = ["a", "b", "c"]
    join_data_l = cudf.Series(["741.248", "1029.528", "3627.292"]).astype(
        dtype
    )
    join_data_r = cudf.Series(["9284.103", "1029.528", "948.637"]).astype(
        dtype
    )
    gdf_l = cudf.DataFrame({"join_col": join_data_l, "B": other_data})
    gdf_r = cudf.DataFrame({"join_col": join_data_r, "B": other_data})
    exp_join_data = ["9284.103", "948.637", "1029.528", "741.248", "3627.292"]
    exp_other_data_x = [None, None, "b", "a", "c"]
    exp_other_data_y = ["a", "c", "b", None, None]
    exp_join_col = cudf.Series(exp_join_data).astype(dtype)
    expected = cudf.DataFrame(
        {
            "join_col": exp_join_col,
            "B_x": exp_other_data_x,
            "B_y": exp_other_data_y,
        }
    )
    got = gdf_l.merge(gdf_r, on="join_col", how="outer")

    assert_join_results_equal(expected, got, how="outer")
    assert_eq(dtype, got["join_col"].dtype)


@pytest.mark.parametrize(
    "dtype_l",
    [Decimal64Dtype(7, 3), Decimal64Dtype(9, 5)],
)
@pytest.mark.parametrize(
    "dtype_r",
    [Decimal64Dtype(8, 3), Decimal64Dtype(11, 6)],
)
def test_mixed_decimal_typecast(dtype_l, dtype_r):
    other_data = ["a", "b", "c", "d"]

    join_data_l = cudf.Series(["95.05", "34.6", "74.22", "14.94"]).astype(
        dtype_r
    )
    join_data_r = cudf.Series(["95.05", "62.4056", "74.22", "1.42"]).astype(
        dtype_l
    )

    gdf_l = cudf.DataFrame({"join_col": join_data_l, "B": other_data})
    gdf_r = cudf.DataFrame({"join_col": join_data_r, "B": other_data})

    with pytest.raises(
        TypeError,
        match="Decimal columns can only be merged with decimal columns "
        "of the same precision and scale",
    ):
        gdf_l.merge(gdf_r, on="join_col", how="inner")


@pytest.fixture
def datetime_types_as_str2(datetime_types_as_str):
    return datetime_types_as_str


def test_typecast_on_join_dt_to_dt(
    datetime_types_as_str, datetime_types_as_str2
):
    other_data = ["a", "b", "c", "d", "e"]
    join_data_l = cudf.Series(
        ["1991-11-20", "1999-12-31", "2004-12-04", "2015-01-01", "2019-08-15"]
    ).astype(datetime_types_as_str)
    join_data_r = cudf.Series(
        ["1991-11-20", "1999-12-31", "2004-12-04", "2015-01-01", "2019-08-16"]
    ).astype(datetime_types_as_str2)

    gdf_l = cudf.DataFrame({"join_col": join_data_l, "B": other_data})
    gdf_r = cudf.DataFrame({"join_col": join_data_r, "B": other_data})

    exp_dtype = max(
        np.dtype(datetime_types_as_str), np.dtype(datetime_types_as_str2)
    )

    exp_join_data = ["1991-11-20", "1999-12-31", "2004-12-04", "2015-01-01"]
    exp_other_data = ["a", "b", "c", "d"]
    exp_join_col = cudf.Series(exp_join_data, dtype=exp_dtype)

    expect = cudf.DataFrame(
        {
            "join_col": exp_join_col,
            "B_x": exp_other_data,
            "B_y": exp_other_data,
        }
    )

    got = gdf_l.merge(gdf_r, on="join_col", how="inner")

    assert_join_results_equal(expect, got, how="inner")


@pytest.mark.parametrize("dtype_l", ["category", "str", "int32", "float32"])
@pytest.mark.parametrize("dtype_r", ["category", "str", "int32", "float32"])
def test_typecast_on_join_categorical(dtype_l, dtype_r):
    if not (dtype_l == "category" or dtype_r == "category"):
        pytest.skip("at least one side must be category for this set of tests")
    if dtype_l == "category" and dtype_r == "category":
        pytest.skip("Can't determine which categorical to use")

    other_data = ["a", "b", "c", "d", "e"]
    join_data_l = cudf.Series([1, 2, 3, 4, 5], dtype=dtype_l)
    join_data_r = cudf.Series([1, 2, 3, 4, 6], dtype=dtype_r)
    if dtype_l == "category":
        exp_dtype = join_data_l.dtype.categories.dtype
    elif dtype_r == "category":
        exp_dtype = join_data_r.dtype.categories.dtype

    gdf_l = cudf.DataFrame({"join_col": join_data_l, "B": other_data})
    gdf_r = cudf.DataFrame({"join_col": join_data_r, "B": other_data})

    exp_join_data = [1, 2, 3, 4]
    exp_other_data = ["a", "b", "c", "d"]
    exp_join_col = cudf.Series(exp_join_data, dtype=exp_dtype)

    expect = cudf.DataFrame(
        {
            "join_col": exp_join_col,
            "B_x": exp_other_data,
            "B_y": exp_other_data,
        }
    )

    got = gdf_l.merge(gdf_r, on="join_col", how="inner")
    assert_join_results_equal(expect, got, how="inner")


def make_categorical_dataframe(categories, ordered=False):
    dtype = CategoricalDtype(categories=categories, ordered=ordered)
    data = cudf.Series(categories).astype(dtype)
    return cudf.DataFrame({"key": data})


def test_categorical_typecast_inner():
    # Inner join casting rules for categoricals

    # Equal categories, equal ordering -> common categorical
    left = make_categorical_dataframe([1, 2, 3], ordered=False)
    right = make_categorical_dataframe([1, 2, 3], ordered=False)
    result = left.merge(right, how="inner", on="key")

    expect_dtype = CategoricalDtype(categories=[1, 2, 3], ordered=False)
    expect_data = cudf.Series([1, 2, 3], dtype=expect_dtype, name="key")

    assert_join_results_equal(
        expect_data, result["key"], how="inner", check_categorical=False
    )

    # Equal categories, unequal ordering -> error
    left = make_categorical_dataframe([1, 2, 3], ordered=False)
    right = make_categorical_dataframe([1, 2, 3], ordered=True)

    with pytest.raises(TypeError):
        result = left.merge(right, how="inner", on="key")

    # Unequal categories
    # Neither ordered -> decategorized to the common categories dtype
    # (matching pandas, which only keeps the result categorical when the
    # category sets match).
    left = make_categorical_dataframe([1, 2, 3], ordered=False)
    right = make_categorical_dataframe([2, 3, 4], ordered=False)

    result = left.merge(right, how="inner", on="key")

    expect_data = cudf.Series([2, 3], dtype="int64", name="key")
    assert_join_results_equal(expect_data, result["key"], how="inner")

    # One is ordered -> error
    left = make_categorical_dataframe([1, 2, 3], ordered=False)
    right = make_categorical_dataframe([2, 3, 4], ordered=True)

    with pytest.raises(TypeError):
        result = left.merge(right, how="inner", on="key")

    # Both are ordered -> error
    left = make_categorical_dataframe([1, 2, 3], ordered=True)
    right = make_categorical_dataframe([2, 3, 4], ordered=True)

    with pytest.raises(TypeError):
        result = left.merge(right, how="inner", on="key")


def test_categorical_typecast_left():
    # TODO: generalize to right or write another test
    # Left join casting rules for categoricals

    # equal categories, neither ordered -> common dtype
    left = make_categorical_dataframe([1, 2, 3], ordered=False)
    right = make_categorical_dataframe([1, 2, 3], ordered=False)

    result = left.merge(right, on="key", how="left")

    expect_dtype = CategoricalDtype(categories=[1, 2, 3], ordered=False)
    expect_data = cudf.Series([1, 2, 3], dtype=expect_dtype, name="key")

    assert_join_results_equal(expect_data, result["key"], how="left")

    # equal categories, unequal ordering -> error
    left = make_categorical_dataframe([1, 2, 3], ordered=True)
    right = make_categorical_dataframe([1, 2, 3], ordered=False)

    with pytest.raises(TypeError):
        result = left.merge(right, on="key", how="left")
    with pytest.raises(TypeError):
        result = right.merge(left, on="key", how="left")

    # unequal categories neither ordered -> decategorized to the common
    # categories dtype (matching pandas)
    left = make_categorical_dataframe([1, 2, 3], ordered=False)
    right = make_categorical_dataframe([2, 3, 4], ordered=False)

    result = left.merge(right, on="key", how="left")
    expect_data = cudf.Series([1, 2, 3], dtype="int64", name="key")

    assert_join_results_equal(expect_data, result["key"], how="left")

    # unequal categories, unequal ordering -> error
    left = make_categorical_dataframe([1, 2, 3], ordered=True)
    right = make_categorical_dataframe([2, 3, 4], ordered=False)

    with pytest.raises(TypeError):
        result = left.merge(right, on="key", how="left")

    # unequal categories, right ordered -> error
    left = make_categorical_dataframe([1, 2, 3], ordered=False)
    right = make_categorical_dataframe([2, 3, 4], ordered=True)

    with pytest.raises(TypeError):
        result = left.merge(right, on="key", how="left")

    # unequal categories, both ordered -> error
    left = make_categorical_dataframe([1, 2, 3], ordered=True)
    right = make_categorical_dataframe([2, 3, 4], ordered=True)

    with pytest.raises(TypeError):
        result = left.merge(right, on="key", how="left")


def test_categorical_typecast_outer():
    # Outer join casting rules for categoricals

    # equal categories, neither ordered -> common dtype
    left = make_categorical_dataframe([1, 2, 3], ordered=False)
    right = make_categorical_dataframe([1, 2, 3], ordered=False)
    result = left.merge(right, on="key", how="outer")

    expect_dtype = CategoricalDtype(categories=[1, 2, 3], ordered=False)
    expect_data = cudf.Series([1, 2, 3], dtype=expect_dtype, name="key")

    assert_join_results_equal(expect_data, result["key"], how="outer")

    # equal categories, both ordered -> common dtype
    left = make_categorical_dataframe([1, 2, 3], ordered=True)
    right = make_categorical_dataframe([1, 2, 3], ordered=True)
    result = left.merge(right, on="key", how="outer")

    expect_dtype = CategoricalDtype(categories=[1, 2, 3], ordered=True)
    expect_data = cudf.Series([1, 2, 3], dtype=expect_dtype, name="key")

    assert_join_results_equal(expect_data, result["key"], how="outer")

    # equal categories, one ordered -> error
    left = make_categorical_dataframe([1, 2, 3], ordered=False)
    right = make_categorical_dataframe([1, 2, 3], ordered=True)

    with pytest.raises(TypeError):
        result = left.merge(right, how="outer", on="key")
    with pytest.raises(TypeError):
        result = right.merge(left, how="outer", on="key")

    # unequal categories, neither ordered -> decategorized to the common
    # categories dtype (matching pandas)
    left = make_categorical_dataframe([1, 2, 3], ordered=False)
    right = make_categorical_dataframe([2, 3, 4], ordered=False)
    result = left.merge(right, on="key", how="outer")

    expect_data = cudf.Series([1, 2, 3, 4], dtype="int64", name="key")

    assert_join_results_equal(expect_data, result["key"], how="outer")

    # unequal categories, one ordered -> error
    left = make_categorical_dataframe([1, 2, 3], ordered=False)
    right = make_categorical_dataframe([2, 3, 4], ordered=True)

    with pytest.raises(TypeError):
        result = left.merge(right, how="outer", on="key")
    with pytest.raises(TypeError):
        result = right.merge(left, how="outer", on="key")

    # unequal categories, both ordered -> error
    left = make_categorical_dataframe([1, 2, 3], ordered=True)
    right = make_categorical_dataframe([2, 3, 4], ordered=True)
    with pytest.raises(TypeError):
        result = left.merge(right, how="outer", on="key")


@pytest.mark.parametrize("dtype", [*NUMERIC_TYPES, "str"])
def test_categorical_typecast_inner_one_cat(dtype):
    data = np.array([1, 2, 3], dtype=dtype)

    left = make_categorical_dataframe(data)
    right = left.astype(left["key"].dtype.categories.dtype)

    result = left.merge(right, on="key", how="inner")
    assert result["key"].dtype == left["key"].dtype.categories.dtype


@pytest.mark.parametrize("dtype", [*NUMERIC_TYPES, "str"])
def test_categorical_typecast_left_one_cat(dtype):
    data = np.array([1, 2, 3], dtype=dtype)

    left = make_categorical_dataframe(data)
    right = left.astype(left["key"].dtype.categories.dtype)

    result = left.merge(right, on="key", how="left")
    # pandas decategorizes when only one side is categorical.
    assert result["key"].dtype == left["key"].dtype.categories.dtype


@pytest.mark.parametrize("dtype", [*NUMERIC_TYPES, "str"])
def test_categorical_typecast_outer_one_cat(dtype):
    data = np.array([1, 2, 3], dtype=dtype)

    left = make_categorical_dataframe(data)
    right = left.astype(left["key"].dtype.categories.dtype)

    result = left.merge(right, on="key", how="outer")
    assert result["key"].dtype == left["key"].dtype.categories.dtype


def test_merge_index_on_opposite_how_column_reset_index():
    df = pd.DataFrame({"a": [1, 2, 3, 4, 5]}, index=[1, 3, 5, 7, 9])
    ser = pd.Series([1, 2], index=pd.Index([1, 2], name="a"), name="b")
    df_cudf = cudf.DataFrame(df)
    ser_cudf = cudf.Series(ser)

    expected = pd.merge(df, ser, on="a", how="left")
    result = cudf.merge(df_cudf, ser_cudf, on="a", how="left")
    assert_eq(result, expected)

    expected = pd.merge(ser, df, on="a", how="right")
    result = cudf.merge(ser_cudf, df_cudf, on="a", how="right")
    assert_eq(result, expected)


def test_merge_suffixes_duplicate_label_raises():
    data = {"a": [1, 2, 3, 4, 5], "b": [6, 6, 6, 6, 6]}
    df_cudf = cudf.DataFrame(data)
    df_pd = pd.DataFrame(data)
    result = df_cudf.merge(df_cudf, on=["a"], suffixes=("", "_right"))
    expected = df_pd.merge(df_pd, on=["a"], suffixes=("", "_right"))
    assert_eq(result, expected)

    # Suffixing collides the new ``b`` -> ``b_right`` with the pre-existing
    # ``b_right`` column, which pandas rejects with a ``MergeError``.
    with pytest.raises(pd.errors.MergeError, match="duplicate columns"):
        result.merge(df_cudf, on=["a"], suffixes=("", "_right"))


def test_merge_left_on_right_index_sort():
    ser = cudf.Series(range(10), name="left_ser")
    ser2 = cudf.Series(
        range(10), index=[4, 5, 6, 3, 2, 1, 8, 9, 0, 7], name="right_ser"
    )
    ser_pd = ser.to_pandas()
    ser2_pd = ser2.to_pandas()
    result = cudf.merge(
        ser, ser2, how="left", left_on="left_ser", right_index=True, sort=True
    )
    expected = pd.merge(
        ser_pd,
        ser2_pd,
        how="left",
        left_on="left_ser",
        right_index=True,
        sort=True,
    )
    assert_eq(result, expected)


def test_merge_renamed_index():
    df = cudf.DataFrame(
        {0: [1, 2, 3, 4, 5], 1: [1, 2, 3, 4, 5], "c": [1, 2, 3, 4, 5]}
    ).set_index([0, 1])
    df.index.names = ["a", "b"]  # doesn't actually change df._index._data

    expect = df.to_pandas().merge(
        df.to_pandas(), left_index=True, right_index=True
    )
    got = df.merge(df, left_index=True, right_index=True, how="inner")
    assert_join_results_equal(expect, got, how="inner")


def test_merge_redundant_params():
    lhs = cudf.DataFrame(
        {"a": [1, 2, 3], "c": [2, 3, 4]}, index=cudf.Index([0, 1, 2], name="c")
    )
    rhs = cudf.DataFrame(
        {"b": [1, 2, 3]}, index=cudf.Index([0, 1, 2], name="a")
    )
    with pytest.raises(ValueError):
        lhs.merge(rhs, on="a", left_index=True)
    with pytest.raises(ValueError):
        lhs.merge(rhs, left_on="a", left_index=True, right_index=True)
    with pytest.raises(ValueError):
        lhs.merge(rhs, right_on="a", left_index=True, right_index=True)
    with pytest.raises(ValueError):
        lhs.merge(rhs, left_on="c", right_on="b")


def test_merge_datetime_timedelta_error(temporal_types_as_str):
    df1 = cudf.DataFrame(
        {"a": cudf.Series([10, 20, 30], dtype=temporal_types_as_str)}
    )
    df2 = df1.astype("int")

    with pytest.raises(TypeError):
        df1.merge(df2)


def test_join_ordering_pandas_compat(request, sort, how):
    if how in ["leftanti", "leftsemi", "cross"]:
        pytest.skip(f"Test not applicable for {how}")
    left_key = [1, 3, 2, 1, 1, 2, 5, 1, 4, 5, 8, 12, 12312, 1] * 100
    left_val = range(len(left_key))
    left = cudf.DataFrame({"key": left_key, "val": left_val})
    right_key = [12312, 12312, 3, 2, 1, 1, 5, 7, 2] * 200
    right_val = list(
        itertools.islice(itertools.cycle(string.ascii_letters), len(right_key))
    )
    right = cudf.DataFrame({"key": right_key, "val": right_val})
    with cudf.option_context("mode.pandas_compatible", True):
        actual = left.merge(right, on="key", how=how, sort=sort)
    expect = left.to_pandas().merge(
        right.to_pandas(), on="key", how=how, sort=sort
    )
    assert_eq(expect, actual)


@pytest.mark.parametrize("on_index", [True, False])
@pytest.mark.parametrize("left_unique", [True, False])
@pytest.mark.parametrize("left_monotonic", [True, False])
@pytest.mark.parametrize("right_unique", [True, False])
@pytest.mark.parametrize("right_monotonic", [True, False])
def test_merge_combinations(
    request,
    how,
    sort,
    on_index,
    left_unique,
    left_monotonic,
    right_unique,
    right_monotonic,
):
    if how in ["leftanti", "leftsemi", "cross"]:
        pytest.skip(f"Test not applicable for {how}")
    request.applymarker(
        pytest.mark.xfail(
            condition=how == "outer"
            and on_index
            and left_unique
            and not left_monotonic
            and right_unique
            and not right_monotonic,
            reason="https://github.com/pandas-dev/pandas/issues/55992",
        )
    )
    left = [2, 3]
    if left_unique:
        left.append(4 if left_monotonic else 1)
    else:
        left.append(3 if left_monotonic else 2)

    right = [2, 3]
    if right_unique:
        right.append(4 if right_monotonic else 1)
    else:
        right.append(3 if right_monotonic else 2)

    left = cudf.DataFrame({"key": left})
    right = cudf.DataFrame({"key": right})

    if on_index:
        left = left.set_index("key")
        right = right.set_index("key")
        on_kwargs = {"left_index": True, "right_index": True}
    else:
        on_kwargs = {"on": "key"}

    with cudf.option_context("mode.pandas_compatible", True):
        result = cudf.merge(left, right, how=how, sort=sort, **on_kwargs)
    if on_index:
        left = left.reset_index()
        right = right.reset_index()

    if how in ["left", "right", "inner"]:
        if how in ["left", "inner"]:
            expected, other, other_unique = left, right, right_unique
        else:
            expected, other, other_unique = right, left, left_unique
        if how == "inner":
            keep_values = set(left["key"].to_numpy()).intersection(
                right["key"].to_numpy()
            )
            keep_mask = expected["key"].isin(keep_values)
            expected = expected[keep_mask]
        if sort:
            expected = expected.sort_values("key")
        if not other_unique:
            other_value_counts = other["key"].value_counts()
            repeats = other_value_counts.reindex(
                expected["key"].values, fill_value=1
            )
            repeats = repeats.astype(np.intp)
            expected = expected["key"].repeat(repeats.values)
            expected = expected.to_frame()
    elif how == "outer":
        if on_index and left_unique and left["key"].equals(right["key"]):
            expected = cudf.DataFrame({"key": left["key"]})
        else:
            left_counts = left["key"].value_counts()
            right_counts = right["key"].value_counts()
            expected_counts = left_counts.mul(right_counts, fill_value=1)
            expected_counts = expected_counts.astype(np.intp)
            expected = expected_counts.index.to_numpy().repeat(
                expected_counts.to_numpy()
            )
            expected = cudf.DataFrame({"key": expected})
            expected = expected.sort_values("key")

    if on_index:
        expected = expected.set_index("key")
    else:
        expected = expected.reset_index(drop=True)

    assert_eq(result, expected)


@pytest.mark.parametrize("param", ["c", 1, 3.4])
def test_merge_invalid_input(param):
    left = cudf.DataFrame({"a": [1, 2, 3]})

    with pytest.raises(TypeError):
        left.merge(param)
    with pytest.raises(TypeError):
        cudf.merge(left["a"], param)


@pytest.mark.parametrize("left_cols", [True, False])
@pytest.mark.parametrize("right_cols", [True, False])
def test_cross_merge_zero_column_operand(left_cols, right_cols):
    # A cross merge where one or both operands have no columns must still
    # produce ``len(left) * len(right)`` rows (matching pandas), rather than
    # dropping the row count of the column-less operand.
    ldata = {"x": [1, 2, 3]} if left_cols else {}
    rdata = {"y": [10, 20, 30, 40]} if right_cols else {}
    pleft = pd.DataFrame(ldata, index=range(3))
    pright = pd.DataFrame(rdata, index=range(4))
    gleft = cudf.DataFrame(ldata, index=range(3))
    gright = cudf.DataFrame(rdata, index=range(4))

    expected = pleft.merge(pright, how="cross")
    result = gleft.merge(gright, how="cross")
    assert_eq(result, expected)


def test_merge_natural_join_key_order_matches_left_frame():
    # A merge without ``on`` joins on the common columns in left-frame
    # column order, like pandas. Building the key list from an unordered
    # set made the sorted output of an outer merge (and hence positional
    # comparisons against pandas) vary with the process hash seed.
    pl = pd.DataFrame({"a": ["foo", "bar"], "b": [1, 2]})
    pr = pd.DataFrame({"a": ["foo", "baz"], "b": [3, 4]})
    gl = cudf.from_pandas(pl)
    gr = cudf.from_pandas(pr)

    expect = pl.merge(pr, how="outer", sort=True)
    got = gl.merge(gr, how="outer", sort=True)
    assert_eq(expect.reset_index(drop=True), got.reset_index(drop=True))


@pytest.mark.parametrize("how", ["inner", "left", "right", "outer"])
@pytest.mark.parametrize("sort", [True, False])
def test_merge_natural_join_left_frame_order_any_how(how, sort):
    # The inferred keys follow left-frame column order for every ``how``,
    # including "right", which cuDF implements by swapping the operands:
    # both the key column order and the multi-key sort priority must still
    # come from the original left frame.
    pl = pd.DataFrame({"v": [5, 6, 9], "b": [2, 1, 9], "a": [1, 2, 9]})
    pr = pd.DataFrame({"w": [7, 8], "a": [2, 1], "b": [1, 2]})
    gl = cudf.from_pandas(pl)
    gr = cudf.from_pandas(pr)

    expect = pl.merge(pr, how=how, sort=sort)
    got = gl.merge(gr, how=how, sort=sort)
    if not sort:
        # unsorted row order is not guaranteed to match pandas
        expect = expect.sort_values(list(expect.columns)).reset_index(
            drop=True
        )
        got = got.sort_values(list(got.columns)).reset_index(drop=True)
    assert_eq(expect.reset_index(drop=True), got.reset_index(drop=True))


@pytest.mark.parametrize("how", ["left", "inner"])
def test_merge_on_index_level_keeps_right_key_column(how):
    # When ``on`` names an index level on the left but a column on the
    # right, the right key column is not a duplicate of any left output
    # column and must be kept (pandas keeps the key as a column and
    # resets the index).
    pl = pd.DataFrame(
        {"outer": [1, 1, 2], "inner": [1, 2, 1], "v1": [0.1, 0.2, 0.3]}
    ).set_index(["outer", "inner"])
    pr = pd.DataFrame(
        {"outer": [1, 1, 2], "inner": [1, 2, 2], "v2": [10.0, 11.0, 12.0]}
    )
    gl = cudf.from_pandas(pl)
    gr = cudf.from_pandas(pr)

    expect = pl.merge(pr, on=["inner"], how=how)
    got = gl.merge(gr, on=["inner"], how=how)
    assert sorted(got.to_pandas().columns) == sorted(expect.columns)
    assert_eq(
        expect.sort_values(list(expect.columns)).reset_index(drop=True),
        got.sort_values(list(expect.columns))
        .reset_index(drop=True)
        .to_pandas()[expect.columns],
        check_dtype=False,
    )


def test_left_merge_empty_right_preserves_key_dtype():
    # The int64 key must not be promoted to float64 when the right side
    # is an empty float64 column, matching pandas.
    left = cudf.DataFrame({"key": [1], "value": [2]})
    right = cudf.DataFrame({"key": []})

    pleft = left.to_pandas()
    pright = right.to_pandas()

    assert_eq(
        cudf.merge(left, right, on="key", how="left"),
        pd.merge(pleft, pright, on="key", how="left"),
    )
