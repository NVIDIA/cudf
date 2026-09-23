# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import numpy as np
import pandas as pd
import pytest

import cudf
from cudf.testing import assert_eq, assert_groupby_results_equal


@pytest.mark.parametrize("index", [None, [1, 2, 3, 4]])
def test_groupby_cumcount(index):
    pdf = pd.DataFrame(
        {
            "a": [1, 1, 3, 4],
            "b": ["bob", "bob", "alice", "cooper"],
            "c": [1, 2, 3, 4],
        },
        index=index,
    )
    gdf = cudf.from_pandas(pdf)

    assert_groupby_results_equal(
        pdf.groupby("a").cumcount(),
        gdf.groupby("a").cumcount(),
        check_dtype=False,
    )

    assert_groupby_results_equal(
        pdf.groupby(["a", "b", "c"]).cumcount(),
        gdf.groupby(["a", "b", "c"]).cumcount(),
        check_dtype=False,
    )

    sr = pd.Series(range(len(pdf)), index=index)
    assert_groupby_results_equal(
        pdf.groupby(sr).cumcount(),
        gdf.groupby(sr).cumcount(),
        check_dtype=False,
    )


@pytest.mark.parametrize(
    "func", ["cummin", "cummax", "cumcount", "cumsum", "cumprod"]
)
def test_groupby_2keys_scan(func):
    nelem = 20
    pdf = pd.DataFrame(np.ones((nelem, 3)), columns=["x", "y", "val"])
    expect_df = pdf.groupby(["x", "y"], sort=True).agg(func)
    gdf = cudf.from_pandas(pdf)
    got_df = gdf.groupby(["x", "y"], sort=True).agg(func)
    # pd.groupby.cumcount returns a series.
    if isinstance(expect_df, pd.Series):
        expect_df = expect_df.to_frame("val")

    assert_groupby_results_equal(got_df, expect_df)

    expect_df = getattr(pdf.groupby(["x", "y"], sort=True), func)()
    got_df = getattr(gdf.groupby(["x", "y"], sort=True), func)()
    assert_groupby_results_equal(got_df, expect_df)

    expect_df = getattr(pdf.groupby(["x", "y"], sort=True)[["x"]], func)()
    got_df = getattr(gdf.groupby(["x", "y"], sort=True)[["x"]], func)()
    assert_groupby_results_equal(got_df, expect_df)

    expect_df = getattr(pdf.groupby(["x", "y"], sort=True)["y"], func)()
    got_df = getattr(gdf.groupby(["x", "y"], sort=True)["y"], func)()
    assert_groupby_results_equal(got_df, expect_df)


@pytest.mark.parametrize(
    "with_nan", [False, True], ids=["just-NA", "also-NaN"]
)
@pytest.mark.parametrize(
    "duplicate_index", [False, True], ids=["rangeindex", "dupindex"]
)
def test_groupby_scan_null_keys(with_nan, dropna, duplicate_index):
    key_col = [None, 1, 2, None, 3, None, 3, 1, None, 1]
    if with_nan:
        df = pd.DataFrame(
            {"key": pd.Series(key_col, dtype="float32"), "value": range(10)}
        )
    else:
        df = pd.DataFrame(
            {"key": pd.Series(key_col, dtype="Int32"), "value": range(10)}
        )

    if duplicate_index:
        # Non-default index with duplicates
        df.index = [1, 2, 3, 1, 3, 2, 4, 1, 6, 10]

    cdf = cudf.from_pandas(df)

    expect = df.groupby("key", dropna=dropna).cumsum()
    got = cdf.groupby("key", dropna=dropna).cumsum()
    assert_groupby_results_equal(expect, got)


@pytest.mark.parametrize("op", ["cumsum", "cumprod", "cummin", "cummax"])
def test_scan_int_null_pandas_compatible(op):
    data = {"a": [1, 2, None, 3], "b": ["x"] * 4}
    df_pd = pd.DataFrame(data)
    df_cudf = cudf.DataFrame(data)
    expected = getattr(df_pd.groupby("b")["a"], op)()
    with cudf.option_context("mode.pandas_compatible", True):
        result = getattr(df_cudf.groupby("b")["a"], op)()
    assert_eq(result, expected)


@pytest.mark.parametrize("op", ["cumsum", "cumprod"])
def test_groupby_cumscan_masked_dtype_preserved(op):
    # pandas preserves masked extension dtypes for groupby cum-scans
    # (Int16 stays Int16, GH#58811) while numpy ints promote to 64-bit
    pdf = pd.DataFrame({"a": [1, 1, 2], "b": [1, pd.NA, 2]}, dtype="Int16")
    gdf = cudf.DataFrame(pdf)

    expected = getattr(pdf.groupby("a")["b"], op)()
    result = getattr(gdf.groupby("a")["b"], op)()

    assert_eq(expected, result)


def test_groupby_cumsum_numpy_dtype_promotes():
    # numpy int8 promotes to int64 (pandas GH#37493)
    pdf = pd.DataFrame({"a": [1, 1], "b": [111, 111]}, dtype="int8")
    gdf = cudf.DataFrame(pdf)

    expected = pdf.groupby("a").cumsum()
    result = gdf.groupby("a").cumsum()

    assert_eq(expected, result)
