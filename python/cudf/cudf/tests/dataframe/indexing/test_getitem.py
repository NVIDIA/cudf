# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import operator
from itertools import combinations

import numpy as np
import pandas as pd
import pytest

import cudf
from cudf.testing import assert_eq
from cudf.testing._utils import assert_exceptions_equal


def test_struct_of_struct_loc():
    df = cudf.DataFrame({"col": [{"a": {"b": 1}}]})
    expect = cudf.Series([{"a": {"b": 1}}], name="col")
    assert_eq(expect, df["col"])


def test_dataframe_midx_cols_getitem():
    df = cudf.DataFrame(
        {
            "a": ["a", "b", "c"],
            "b": ["b", "", ""],
            "c": [10, 11, 12],
        }
    )
    df.columns = df.set_index(["a", "b"]).index
    pdf = df.to_pandas()

    expected = df["c"]
    actual = pdf["c"]
    assert_eq(expected, actual)
    df = cudf.DataFrame(
        [[1, 0], [0, 1]],
        columns=[
            ["foo", "foo"],
            ["location", "location"],
            ["x", "y"],
        ],
    )
    df = df.assign(bools=cudf.Series([True, False], dtype="bool"))
    assert_eq(df["bools"], df.to_pandas()["bools"])


def test_multicolumn_item():
    gdf = cudf.DataFrame({"x": range(10), "y": range(10), "z": range(10)})
    gdg = gdf.groupby(["x", "y"]).min()
    gdgT = gdg.T
    pdgT = gdgT.to_pandas()
    assert_eq(gdgT[(0, 0)], pdgT[(0, 0)])


def test_dataframe_column_name_indexing():
    df = cudf.DataFrame()
    data = np.asarray(range(10), dtype=np.int32)
    df["a"] = data
    df[1] = data
    np.testing.assert_equal(
        df["a"].to_numpy(), np.asarray(range(10), dtype=np.int32)
    )
    np.testing.assert_equal(
        df[1].to_numpy(), np.asarray(range(10), dtype=np.int32)
    )

    rng = np.random.default_rng(seed=0)
    pdf = pd.DataFrame()
    nelem = 10
    pdf = pd.DataFrame(
        {
            "key1": rng.integers(0, 5, nelem),
            "key2": rng.integers(0, 3, nelem),
            1: np.arange(1, 1 + nelem),
            2: rng.random(nelem),
        }
    )
    df = cudf.from_pandas(pdf)

    assert_eq(df[df.columns], df)
    assert_eq(df[df.columns[:1]], df[["key1"]])

    for i in range(1, len(pdf.columns) + 1):
        for idx in combinations(pdf.columns, i):
            assert pdf[list(idx)].equals(df[list(idx)].to_pandas())

    # test for only numeric columns
    df = pd.DataFrame()
    for i in range(0, 10):
        df[i] = range(nelem)
    gdf = cudf.DataFrame(df)
    assert_eq(gdf, df)

    assert_eq(gdf[gdf.columns], gdf)
    assert_eq(gdf[gdf.columns[:3]], gdf[[0, 1, 2]])


def test_dataframe_slicing():
    rng = np.random.default_rng(seed=0)
    df = cudf.DataFrame()
    size = 123
    df["a"] = ha = rng.integers(low=0, high=100, size=size).astype(np.int32)
    df["b"] = hb = rng.random(size).astype(np.float32)
    df["c"] = hc = rng.integers(low=0, high=100, size=size).astype(np.int64)
    df["d"] = hd = rng.random(size).astype(np.float64)

    # Row slice first 10
    first_10 = df[:10]
    assert len(first_10) == 10
    assert tuple(first_10.columns) == ("a", "b", "c", "d")
    np.testing.assert_equal(first_10["a"].to_numpy(), ha[:10])
    np.testing.assert_equal(first_10["b"].to_numpy(), hb[:10])
    np.testing.assert_equal(first_10["c"].to_numpy(), hc[:10])
    np.testing.assert_equal(first_10["d"].to_numpy(), hd[:10])
    del first_10

    # Row slice last 10
    last_10 = df[-10:]
    assert len(last_10) == 10
    assert tuple(last_10.columns) == ("a", "b", "c", "d")
    np.testing.assert_equal(last_10["a"].to_numpy(), ha[-10:])
    np.testing.assert_equal(last_10["b"].to_numpy(), hb[-10:])
    np.testing.assert_equal(last_10["c"].to_numpy(), hc[-10:])
    np.testing.assert_equal(last_10["d"].to_numpy(), hd[-10:])
    del last_10

    # Row slice [begin:end]
    begin = 7
    end = 121
    subrange = df[begin:end]
    assert len(subrange) == end - begin
    assert tuple(subrange.columns) == ("a", "b", "c", "d")
    np.testing.assert_equal(subrange["a"].to_numpy(), ha[begin:end])
    np.testing.assert_equal(subrange["b"].to_numpy(), hb[begin:end])
    np.testing.assert_equal(subrange["c"].to_numpy(), hc[begin:end])
    np.testing.assert_equal(subrange["d"].to_numpy(), hd[begin:end])
    del subrange


@pytest.mark.parametrize("slice_start", [None, 0, 1, 3, 10, -10])
@pytest.mark.parametrize("slice_end", [None, 0, 1, 30, 50, -1])
def test_dataframe_masked_slicing(slice_start, slice_end):
    nelem = 50
    rng = np.random.default_rng(seed=0)
    mask = rng.choice([True, False], size=nelem)
    gdf = cudf.DataFrame(
        {"a": list(range(nelem)), "b": list(range(nelem, 2 * nelem))}
    )
    gdf.loc[mask, "a"] = None
    gdf.loc[mask, "b"] = None

    def do_slice(x):
        return x[slice_start:slice_end]

    expect = do_slice(gdf.to_pandas())
    got = do_slice(gdf).to_pandas()

    assert_eq(expect, got, check_dtype=False)


@pytest.mark.parametrize("dtype", [int, float, str])
def test_empty_boolean_mask(dtype):
    gdf = cudf.DataFrame({"a": []}, dtype=dtype)
    pdf = gdf.to_pandas()

    compare_val = dtype(1)

    expected = pdf[pdf.a == compare_val]
    got = gdf[gdf.a == compare_val]
    assert_eq(expected, got)

    expected = pdf.a[pdf.a == compare_val]
    got = gdf.a[gdf.a == compare_val]
    assert_eq(expected, got)


def test_dataframe_apply_boolean_mask():
    pdf = pd.DataFrame(
        {
            "a": [0, 1, 2, 3],
            "b": [0.1, 0.2, None, 0.3],
            "c": ["a", None, "b", "c"],
        }
    )
    gdf = cudf.DataFrame(pdf)
    assert_eq(pdf[[True, False, True, False]], gdf[[True, False, True, False]])


@pytest.mark.parametrize(
    "mask_fn", [lambda x: x, lambda x: np.array(x), lambda x: pd.Series(x)]
)
def test_dataframe_boolean_mask(mask_fn):
    mask_base = [
        True,
        False,
        True,
        False,
        True,
        False,
        True,
        False,
        True,
        False,
    ]
    pdf = pd.DataFrame({"x": range(10), "y": range(10)})
    gdf = cudf.from_pandas(pdf)
    mask = mask_fn(mask_base)
    assert len(mask) == gdf.shape[0]
    pdf_masked = pdf[mask]
    gdf_masked = gdf[mask]
    assert pdf_masked.to_string().split() == gdf_masked.to_string().split()


def test_dataframe_getitem_bool_pyarrow_mask():
    pdf = pd.DataFrame({"x": range(4), "y": range(10, 14)})
    gdf = cudf.from_pandas(pdf)
    mask = cudf.Series([True, False, True, False], dtype="bool[pyarrow]")

    assert_eq(pdf[[True, False, True, False]], gdf[mask])


@pytest.mark.parametrize(
    "gdf_kwargs",
    [
        {"data": {"a": range(1000)}},
        {"data": {"a": range(1000), "b": range(1000)}},
        {
            "data": {
                "a": range(20),
                "b": range(20),
                "c": ["abc", "def", "xyz", "def", "pqr"] * 4,
            }
        },
        {"index": [1, 2, 3]},
        {"index": range(1000)},
        {"columns": ["a", "b", "c", "d"]},
        {"columns": ["a"], "index": range(1000)},
        {"columns": ["a", "col2", "...col n"], "index": range(1000)},
        {"index": cudf.Series(range(1000)).astype("str")},
        {
            "columns": ["a", "b", "c", "d"],
            "index": cudf.Series(range(1000)).astype("str"),
        },
    ],
)
@pytest.mark.parametrize(
    "slc",
    [
        slice(6, None),  # start but no stop, [6:]
        slice(None, None, 3),  # only step, [::3]
        slice(1, 10, 2),  # start, stop, step
        slice(3, -5, 2),  # negative stop
        slice(-2, -4),  # slice is empty
        slice(-10, -20, -1),  # reversed slice
        slice(None),  # slices everything, same as [:]
        slice(250, 500),
        slice(250, 251),
        slice(50),
        slice(1, 10),
        slice(10, 20),
        slice(15, 24),
        slice(6),
    ],
)
def test_dataframe_sliced(gdf_kwargs, slc):
    gdf = cudf.DataFrame(**gdf_kwargs)
    pdf = gdf.to_pandas()

    actual = gdf[slc]
    expected = pdf[slc]

    assert_eq(actual, expected)


def test_duplicate_labels_raises():
    df = cudf.DataFrame([[1, 2]], columns=["a", "b"])
    with pytest.raises(ValueError):
        df[["a", "a"]]
    with pytest.raises(ValueError):
        df.loc[:, ["a", "a"]]


def test_boolmask():
    pdf = pd.DataFrame({"x": range(10), "y": range(10)})
    gdf = cudf.DataFrame({"x": range(10), "y": range(10)})
    rng = np.random.default_rng(seed=0)
    boolmask = rng.choice([True, False], size=len(pdf))
    gdf = gdf[boolmask]
    pdf = pdf[boolmask]
    assert_eq(pdf, gdf)


@pytest.mark.parametrize(
    "mask_shape",
    [
        (2, "ab"),
        (2, "abc"),
        (3, "ab"),
        (3, "abc"),
        (3, "abcd"),
        (4, "abc"),
        (4, "abcd"),
    ],
)
def test_dataframe_boolmask(mask_shape):
    rng = np.random.default_rng(seed=0)
    pdf = pd.DataFrame({col: rng.integers(0, 10, 3) for col in "abc"})
    pdf_mask = pd.DataFrame(
        {col: rng.integers(0, 2, mask_shape[0]) > 0 for col in mask_shape[1]}
    )
    gdf = cudf.DataFrame(pdf)
    gdf_mask = cudf.DataFrame(pdf_mask)
    gdf = gdf[gdf_mask]
    pdf = pdf[pdf_mask]

    assert np.array_equal(gdf.columns, pdf.columns)
    for col in gdf.columns:
        assert np.array_equal(
            gdf[col].fillna(-1).to_pandas().values, pdf[col].fillna(-1).values
        )


@pytest.mark.parametrize(
    "box",
    [
        list,
        pytest.param(
            cudf.Series,
            marks=pytest.mark.xfail(
                reason="Pandas can't index a multiindex with a Series"
            ),
        ),
    ],
)
def test_dataframe_multiindex_boolmask(box):
    mask = box([True, False, True])
    gdf = cudf.DataFrame(
        {"w": [3, 2, 1], "x": [1, 2, 3], "y": [0, 1, 0], "z": [1, 1, 1]}
    )
    gdg = gdf.groupby(["w", "x"]).count()
    pdg = gdg.to_pandas()
    assert_eq(gdg[mask], pdg[mask])


@pytest.mark.parametrize(
    "arg", [slice(2, 8, 3), slice(1, 20, 4), slice(-2, -6, -2)]
)
def test_dataframe_strided_slice(arg):
    mul = pd.DataFrame(
        {
            "Index": [1, 2, 3, 4, 5, 6, 7, 8, 9],
            "AlphaIndex": ["a", "b", "c", "d", "e", "f", "g", "h", "i"],
        }
    )
    pdf = pd.DataFrame(
        {"Val": [10, 9, 8, 7, 6, 5, 4, 3, 2]},
        index=pd.MultiIndex.from_frame(mul),
    )
    gdf = cudf.DataFrame(pdf)

    expect = pdf[arg]
    got = gdf[arg]

    assert_eq(expect, got)


def test_getitem_slice_in_tuple_raises():
    # A tuple key in __getitem__ is a single (possibly MultiIndex) column
    # label; a slice inside it is invalid (use .loc for slicing) and
    # raises InvalidIndexError like pandas.
    mi = pd.MultiIndex.from_product([["x", "y"], [0, 1]])
    pdf = pd.DataFrame([[1, 1, 2, 2], [3, 3, 4, 4]], columns=mi)
    gdf = cudf.from_pandas(pdf)
    assert_exceptions_equal(
        lfunc=operator.getitem,
        rfunc=operator.getitem,
        lfunc_args_and_kwargs=([pdf, (slice(None), 0)],),
        rfunc_args_and_kwargs=([gdf, (slice(None), 0)],),
    )
