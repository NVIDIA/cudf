# SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from functools import partial

import numpy as np
import pandas as pd
import pytest

from dask import dataframe as dd

import cudf

import dask_cudf

param_nrows = [5, 10, 50, 100]


@pytest.mark.parametrize("left_nrows", param_nrows)
@pytest.mark.parametrize("right_nrows", param_nrows)
@pytest.mark.parametrize("left_nkeys", [4, 5])
@pytest.mark.parametrize("right_nkeys", [4, 5])
def test_join_inner(left_nrows, right_nrows, left_nkeys, right_nkeys):
    chunksize = 50

    rng = np.random.default_rng(seed=0)

    # cuDF
    left = cudf.DataFrame(
        {
            "x": rng.integers(0, left_nkeys, size=left_nrows),
            "a": np.arange(left_nrows),
        }
    )
    right = cudf.DataFrame(
        {
            "x": rng.integers(0, right_nkeys, size=right_nrows),
            "a": 1000 * np.arange(right_nrows),
        }
    )

    expect = left.set_index("x").join(
        right.set_index("x"), how="inner", sort=True, lsuffix="l", rsuffix="r"
    )
    expect = expect.to_pandas()

    # dask_cudf
    left = dask_cudf.from_cudf(left, chunksize=chunksize)
    right = dask_cudf.from_cudf(right, chunksize=chunksize)

    joined = left.set_index("x").join(
        right.set_index("x"), how="inner", lsuffix="l", rsuffix="r"
    )
    got = joined.compute().to_pandas()

    if len(got.columns):
        got = got.sort_values(list(got.columns))
        expect = expect.sort_values(list(expect.columns))

    # Check index
    np.testing.assert_array_equal(expect.index.values, got.index.values)

    # Check rows in each groups
    expect_rows = {}
    got_rows = {}

    def gather(df, grows):
        grows[df["x"].values[0]] = (set(df.al), set(df.ar))

    expect.reset_index().groupby("x")[["x", "al", "ar"]].apply(
        partial(gather, grows=expect_rows)
    )
    expect.reset_index().groupby("x")[["x", "al", "ar"]].apply(
        partial(gather, grows=got_rows)
    )

    assert got_rows == expect_rows


@pytest.mark.parametrize("left_nrows", param_nrows)
@pytest.mark.parametrize("right_nrows", param_nrows)
@pytest.mark.parametrize("left_nkeys", [4, 5])
@pytest.mark.parametrize("right_nkeys", [4, 5])
@pytest.mark.parametrize("how", ["left", "right"])
def test_join_left(left_nrows, right_nrows, left_nkeys, right_nkeys, how):
    chunksize = 50

    rng = np.random.default_rng(seed=0)

    # cuDF
    left = cudf.DataFrame(
        {
            "x": rng.integers(0, left_nkeys, size=left_nrows),
            "a": np.arange(left_nrows, dtype=np.float64),
        }
    )
    right = cudf.DataFrame(
        {
            "x": rng.integers(0, right_nkeys, size=right_nrows),
            "a": 1000 * np.arange(right_nrows, dtype=np.float64),
        }
    )

    expect = left.set_index("x").join(
        right.set_index("x"), how=how, sort=True, lsuffix="l", rsuffix="r"
    )
    expect = expect.to_pandas()

    # dask_cudf
    left = dask_cudf.from_cudf(left, chunksize=chunksize)
    right = dask_cudf.from_cudf(right, chunksize=chunksize)

    joined = left.set_index("x").join(
        right.set_index("x"), how=how, lsuffix="l", rsuffix="r"
    )
    got = joined.compute().to_pandas()

    if len(got.columns):
        got = got.sort_values(list(got.columns))
        expect = expect.sort_values(list(expect.columns))

    # Check index
    np.testing.assert_array_equal(expect.index.values, got.index.values)

    # Check rows in each groups
    expect_rows = {}
    got_rows = {}

    def gather(df, grows):
        cola = np.sort(np.asarray(df.al))
        colb = np.sort(np.asarray(df.ar))

        grows[df["x"].values[0]] = (cola, colb)

    expect.reset_index().groupby("x")[["x", "al", "ar"]].apply(
        partial(gather, grows=expect_rows)
    )

    expect.reset_index().groupby("x")[["x", "al", "ar"]].apply(
        partial(gather, grows=got_rows)
    )

    for k in expect_rows:
        np.testing.assert_array_equal(expect_rows[k][0], got_rows[k][0])
        np.testing.assert_array_equal(expect_rows[k][1], got_rows[k][1])


@pytest.mark.parametrize("left_nrows", param_nrows)
@pytest.mark.parametrize("right_nrows", param_nrows)
@pytest.mark.parametrize("left_nkeys", [4, 5])
@pytest.mark.parametrize("right_nkeys", [4, 5])
def test_merge_left(
    left_nrows, right_nrows, left_nkeys, right_nkeys, how="left"
):
    chunksize = 3

    rng = np.random.default_rng(seed=42)

    # cuDF
    left = cudf.DataFrame(
        {
            "x": rng.integers(0, left_nkeys, size=left_nrows),
            "y": rng.integers(0, left_nkeys, size=left_nrows),
            "a": np.arange(left_nrows, dtype=np.float64),
        }
    )
    right = cudf.DataFrame(
        {
            "x": rng.integers(0, right_nkeys, size=right_nrows),
            "y": rng.integers(0, right_nkeys, size=right_nrows),
            "a": 1000 * np.arange(right_nrows, dtype=np.float64),
        }
    )

    expect = left.merge(right, on=["x", "y"], how=how)

    def normalize(df):
        return (
            df.to_pandas()
            .sort_values(["x", "y", "a_x", "a_y"])
            .reset_index(drop=True)
        )

    # dask_cudf
    left = dask_cudf.from_cudf(left, chunksize=chunksize)
    right = dask_cudf.from_cudf(right, chunksize=chunksize)

    result = left.merge(right, on=["x", "y"], how=how).compute(
        scheduler="single-threaded"
    )

    dd.assert_eq(normalize(expect), normalize(result))


@pytest.mark.parametrize("left_nrows", [2, 5])
@pytest.mark.parametrize("right_nrows", [5, 10])
@pytest.mark.parametrize("left_nkeys", [4])
@pytest.mark.parametrize("right_nkeys", [4])
def test_merge_1col_left(
    left_nrows, right_nrows, left_nkeys, right_nkeys, how="left"
):
    chunksize = 3

    rng = np.random.default_rng(seed=0)

    # cuDF
    left = cudf.DataFrame(
        {
            "x": rng.integers(0, left_nkeys, size=left_nrows),
            "a": np.arange(left_nrows, dtype=np.float64),
        }
    )
    right = cudf.DataFrame(
        {
            "x": rng.integers(0, right_nkeys, size=right_nrows),
            "a": 1000 * np.arange(right_nrows, dtype=np.float64),
        }
    )

    expect = left.merge(right, on=["x"], how=how)
    expect = (
        expect.to_pandas()
        .sort_values(["x", "a_x", "a_y"])
        .reset_index(drop=True)
    )

    # dask_cudf
    left = dask_cudf.from_cudf(left, chunksize=chunksize)
    right = dask_cudf.from_cudf(right, chunksize=chunksize)

    joined = left.merge(right, on=["x"], how=how)

    got = joined.compute().to_pandas()

    got = got.sort_values(["x", "a_x", "a_y"]).reset_index(drop=True)

    dd.assert_eq(expect, got)


def test_merge_should_fail():
    # Expected failure cases described in #2694
    rng = np.random.default_rng(seed=0)
    df1 = pd.DataFrame(
        {
            "a": [1, 2, 3, 4, 5, 6] * 2,
            "b": rng.integers(0, 12, 12),
        }
    )
    df2 = pd.DataFrame(
        {
            "a": [7, 2, 3, 8, 5, 9] * 2,
            "c": rng.integers(0, 12, 12),
        }
    )

    left = dask_cudf.from_cudf(df1, 1).groupby("a").b.min().to_frame()
    right = dask_cudf.from_cudf(df2, 1).groupby("a").c.min().to_frame()

    with pytest.raises(KeyError):
        left.merge(right, how="left", on=["nonCol"])
    with pytest.raises(KeyError):
        left.merge(right, how="left", on=["b"])
    with pytest.raises(KeyError):
        left.merge(right, how="left", on=["c"])

    # Same column names
    df2["b"] = np.random.default_rng(seed=0).integers(0, 12, 12)
    right = dask_cudf.from_cudf(df2, 1).groupby("a").b.min().to_frame()

    with pytest.raises(KeyError):
        left.merge(right, how="left", on="NonCol")


@pytest.mark.parametrize("how", ["inner", "left"])
def test_indexed_join(how):
    p_left = pd.DataFrame({"x": np.arange(10)}, index=np.arange(10) * 2)
    p_right = pd.DataFrame({"y": 1}, index=np.arange(15))

    g_left = cudf.from_pandas(p_left)
    g_right = cudf.from_pandas(p_right)

    dg_left = dd.from_pandas(g_left, npartitions=4)
    dg_right = dd.from_pandas(g_right, npartitions=5)

    d = g_left.merge(g_right, left_index=True, right_index=True, how=how)
    dg = dg_left.merge(dg_right, left_index=True, right_index=True, how=how)

    # occasionally order is not correct (possibly do to hashing in the merge)
    d = d.sort_values("x")  # index is preserved
    dg = dg.sort_values(
        "x"
    )  # index is reset -- sort_values will slow test down

    dd.assert_eq(d, dg, check_index=False)


@pytest.mark.parametrize("how", ["left", "inner"])
def test_how(how):
    left = cudf.DataFrame(
        {"x": [1, 2, 3, 4, None], "y": [1.0, 2.0, 3.0, 4.0, 0.0]}
    )
    right = cudf.DataFrame({"x": [2, 3, None, 2], "y": [20, 30, 0, 20]})

    dleft = dd.from_pandas(left, npartitions=2)
    dright = dd.from_pandas(right, npartitions=3)

    expected = left.merge(right, how=how, on="x")
    result = dleft.merge(dright, how=how, on="x")

    dd.assert_eq(
        result.compute().to_pandas().sort_values("x"),
        expected.to_pandas().sort_values("x"),
        check_index=False,
    )


@pytest.mark.parametrize("daskify", [True, False])
def test_single_dataframe_merge(daskify):
    right = cudf.DataFrame({"x": [1, 2, 1, 2], "y": [1, 2, 3, 4]})
    left = cudf.DataFrame({"x": np.arange(100) % 10, "z": np.arange(100)})

    dleft = dd.from_pandas(left, npartitions=10)

    if daskify:
        dright = dd.from_pandas(right, npartitions=1)
    else:
        dright = right

    expected = left.merge(right, how="inner")
    result = dd.merge(dleft, dright, how="inner")
    assert len(result.dask) < 25

    dd.assert_eq(
        result.compute().to_pandas().sort_values(["z", "y"]),
        expected.to_pandas().sort_values(["z", "y"]),
        check_index=False,
    )


@pytest.mark.parametrize("how", ["inner", "left"])
@pytest.mark.parametrize("on", ["id_1", ["id_1"], ["id_1", "id_2"]])
def test_on(how, on):
    left = cudf.DataFrame(
        {"id_1": [1, 2, 3, 4, 5], "id_2": [1.0, 2.0, 3.0, 4.0, 0.0]}
    )
    right = cudf.DataFrame(
        {"id_1": [2, 3, None, 2], "id_2": [2.0, 3.0, 4.0, 20]}
    )

    dleft = dd.from_pandas(left, npartitions=2)
    dright = dd.from_pandas(right, npartitions=3)

    expected = left.merge(right, how=how, on=on)
    result = dleft.merge(dright, how=how, on=on)

    dd.assert_eq(
        result.compute().to_pandas().sort_values(on),
        expected.to_pandas().sort_values(on),
        check_index=False,
    )


def test_single_partition():
    left = cudf.DataFrame({"x": range(200), "y": range(200)})
    right = cudf.DataFrame({"x": range(100), "z": range(100)})

    dleft = dd.from_pandas(left, npartitions=1)
    dright = dd.from_pandas(right, npartitions=10)

    m = dleft.merge(dright, how="inner")
    assert len(m.dask) < len(dleft.dask) + len(dright.dask) * 3

    dleft = dd.from_pandas(left, npartitions=5)
    m2 = dleft.merge(right, how="inner")
    assert len(m2.dask) < len(dleft.dask) * 3
    assert len(m2) == 100


def test_issue_12773():
    # https://github.com/NVIDIA/cudf/issues/12773
    df1 = cudf.DataFrame({"a": ["a", "b"], "b": [1, 2]})
    df2 = cudf.DataFrame({"a": ["a", "c"], "b": [2, 3]})

    dleft = dd.from_pandas(df1, npartitions=2).set_index("a")
    dright = dd.from_pandas(df2, npartitions=2).set_index("a")

    expected = df1.set_index("a").merge(
        df2.set_index("a"), left_index=True, right_index=True, how="left"
    )
    result = dleft.merge(dright, left_index=True, right_index=True, how="left")
    dd.assert_eq(
        result.compute().to_pandas(),
        expected.to_pandas(),
        check_index=False,
    )


@pytest.mark.parametrize(
    "typ", [cudf.Decimal32Dtype, cudf.Decimal64Dtype, cudf.Decimal128Dtype]
)
def test_merge_on_decimal(typ):
    df = cudf.DataFrame({"a": [1], "b": [2]}, dtype=typ(1))
    ddf = dask_cudf.from_cudf(df, npartitions=1)
    result = ddf.merge(ddf, left_on="a", right_on="a")
    expected = df.merge(df, left_on="a", right_on="a")
    dd.assert_eq(result, expected)
