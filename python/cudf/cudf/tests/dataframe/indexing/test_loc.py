# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import operator
import re
import weakref
from contextlib import contextmanager

import cupy as cp
import numpy as np
import pandas as pd
import pytest

import cudf
from cudf.testing import assert_eq
from cudf.testing._utils import assert_exceptions_equal, expect_warning_if


def test_dataframe_midx_columns_loc():
    idx_1 = ["Hi", "Lo"]
    idx_2 = ["I", "II", "III"]
    idx = cudf.MultiIndex.from_product([idx_1, idx_2])

    data_rand = (
        np.random.default_rng(seed=0)
        .uniform(0, 1, 3 * len(idx))
        .reshape(3, -1)
    )
    df = cudf.DataFrame(data_rand, index=["A", "B", "C"], columns=idx)
    pdf = df.to_pandas()

    assert_eq(df.shape, pdf.shape)

    expected = pdf.loc[["A", "B"]]
    actual = df.loc[["A", "B"]]

    assert_eq(expected, actual)
    assert_eq(df, pdf)


@pytest.mark.parametrize("dtype1", ["int16", "float32"])
@pytest.mark.parametrize("dtype2", ["int16", "float32"])
def test_dataframe_loc_int_float(dtype1, dtype2):
    df = cudf.DataFrame(
        {"a": [10, 11, 12, 13, 14]},
        index=cudf.Index([1, 2, 3, 4, 5], dtype=dtype1),
    )
    pdf = df.to_pandas()

    gidx = cudf.Index([2, 3, 4], dtype=dtype2)
    pidx = gidx.to_pandas()

    actual = df.loc[gidx]
    expected = pdf.loc[pidx]

    assert_eq(actual, expected, check_index_type=True, check_dtype=True)


@pytest.mark.xfail(reason="Not yet properly supported.")
def test_multiindex_wildcard_selection_three_level_all():
    midx = cudf.MultiIndex.from_tuples(
        [(c1, c2, c3) for c1 in "abcd" for c2 in "abc" for c3 in "ab"]
    )
    df = cudf.DataFrame({f"{i}": [i] for i in range(24)})
    df.columns = midx

    expect = df.to_pandas().loc[:, (slice("a", "c"), slice("a", "b"), "b")]
    got = df.loc[:, (slice(None), "b")]
    assert_eq(expect, got)


def test_multiindex_wildcard_selection_all():
    midx = cudf.MultiIndex.from_tuples(
        [(c1, c2) for c1 in "abc" for c2 in "ab"]
    )
    df = cudf.DataFrame({f"{i}": [i] for i in range(6)})
    df.columns = midx
    expect = df.to_pandas().loc[:, (slice(None), "b")]
    got = df.loc[:, (slice(None), "b")]
    assert_eq(expect, got)


@pytest.mark.xfail(reason="Not yet properly supported.")
def test_multiindex_wildcard_selection_partial():
    midx = cudf.MultiIndex.from_tuples(
        [(c1, c2) for c1 in "abc" for c2 in "ab"]
    )
    df = cudf.DataFrame({f"{i}": [i] for i in range(6)})
    df.columns = midx
    expect = df.to_pandas().loc[:, (slice("a", "b"), "b")]
    got = df.loc[:, (slice("a", "b"), "b")]
    assert_eq(expect, got)


@pytest.mark.parametrize(
    "value",
    [
        "7",
        pytest.param(
            ["7", "8"],
            marks=pytest.mark.xfail(
                reason="https://github.com/rapidsai/cudf/issues/11298"
            ),
        ),
    ],
)
def test_loc_setitem_string_11298(value):
    df = pd.DataFrame({"a": ["a", "b", "c"]})
    cdf = cudf.from_pandas(df)

    df.loc[:1, "a"] = value

    cdf.loc[:1, "a"] = value

    assert_eq(df, cdf)


@pytest.mark.xfail(reason="https://github.com/rapidsai/cudf/issues/11944")
def test_loc_setitem_list_11944():
    df = pd.DataFrame(
        data={"a": ["yes", "no"], "b": [["l1", "l2"], ["c", "d"]]}
    )
    cdf = cudf.from_pandas(df)
    df.loc[df.a == "yes", "b"] = [["hello"]]
    cdf.loc[cdf.a == "yes", "b"] = [["hello"]]
    assert_eq(df, cdf)


@pytest.mark.xfail(reason="https://github.com/rapidsai/cudf/issues/12504")
def test_loc_setitem_extend_empty_12504():
    df = pd.DataFrame(columns=["a"])
    cdf = cudf.from_pandas(df)

    df.loc[0] = [1]

    cdf.loc[0] = [1]

    assert_eq(df, cdf)


def test_loc_setitem_extend_existing_12505():
    df = pd.DataFrame({"a": [0]})
    cdf = cudf.from_pandas(df)

    df.loc[1] = 1

    cdf.loc[1] = 1

    assert_eq(df, cdf)


def test_loc_setitem_list_arg_missing_raises():
    data = {"a": [0]}
    gdf = cudf.DataFrame(data)
    pdf = pd.DataFrame(data)

    cudf_msg = re.escape("[1] not in the index.")
    with pytest.raises(KeyError, match=cudf_msg):
        gdf.loc[[1]] = 1

    with pytest.raises(KeyError, match=cudf_msg):
        gdf.loc[[1], "a"] = 1

    with pytest.raises(KeyError):
        pdf.loc[[1]] = 1

    with pytest.raises(KeyError):
        pdf.loc[[1], "a"] = 1


@pytest.mark.xfail(reason="https://github.com/rapidsai/cudf/issues/12801")
def test_loc_setitem_add_column_partial_12801():
    df = pd.DataFrame({"a": [0, 1, 2]})
    cdf = cudf.from_pandas(df)

    df.loc[df.a < 2, "b"] = 1

    cdf.loc[cdf.a < 2, "b"] = 1

    assert_eq(df, cdf)


@contextmanager
def expect_pandas_performance_warning(idx):
    with expect_warning_if(
        (not isinstance(idx[0], tuple) and len(idx) > 2)
        or (isinstance(idx[0], tuple) and len(idx[0]) > 2),
        pd.errors.PerformanceWarning,
    ):
        yield


@pytest.mark.parametrize(
    "key_tuple",
    [
        # return 2 rows, 0 remaining keys = dataframe with entire index
        ("a", "store", "clouds", "fire"),
        (("a", "store", "clouds", "fire"), slice(None)),
        # return 2 rows, 1 remaining key = dataframe with n-k index columns
        ("a", "store", "storm"),
        (("a", "store", "storm"), slice(None)),
        # return 2 rows, 2 remaining keys = dataframe with n-k index columns
        ("a", "store"),
        (("a", "store"), slice(None)),
        # return 2 rows, n-1 remaining keys = dataframe with n-k index columns
        ("a",),
        "a",
        "b",
        "c",
        (("a",), slice(None)),
        # return 1 row, 0 remaining keys = dataframe with entire index
        ("a", "store", "storm", "smoke"),
        (("a", "store", "storm", "smoke"), slice(None)),
        # return 1 row and 1 remaining key = series
        ("c", "forest", "clear"),
        (("c", "forest", "clear"), slice(None)),
    ],
)
def test_multiindex_loc(key_tuple):
    rng = np.random.default_rng(seed=0)
    pdf = pd.DataFrame(rng.random(size=(7, 5)))
    pdfIndex = pd.MultiIndex(
        [
            ["a", "b", "c"],
            ["house", "store", "forest"],
            ["clouds", "clear", "storm"],
            ["fire", "smoke", "clear"],
            [
                np.datetime64("2001-01-01", "ns"),
                np.datetime64("2002-01-01", "ns"),
                np.datetime64("2003-01-01", "ns"),
            ],
        ],
        [
            [0, 0, 0, 0, 1, 1, 2],
            [1, 1, 1, 1, 0, 0, 2],
            [0, 0, 2, 2, 2, 0, 1],
            [0, 0, 0, 1, 2, 0, 1],
            [1, 0, 1, 2, 0, 0, 1],
        ],
    )
    pdfIndex.names = ["alpha", "location", "weather", "sign", "timestamp"]
    gdfIndex = cudf.from_pandas(pdfIndex)
    assert_eq(pdfIndex, gdfIndex)
    gdf = cudf.from_pandas(pdf)
    pdf.index = pdfIndex
    gdf.index = gdfIndex
    # The index is unsorted, which makes things slow but is fine for testing.
    with expect_pandas_performance_warning(key_tuple):
        expected = pdf.loc[key_tuple]
    got = gdf.loc[key_tuple].sort_index()
    assert_eq(expected.sort_index(), got)

    with cudf.option_context("mode.pandas_compatible", True):
        got = gdf.loc[key_tuple]
    assert_eq(expected, got)


@pytest.mark.parametrize("second_val", [[0, 1], [1, 0]])
def test_multiindex_compatible_ordering(second_val):
    indexer = (([1, 1], second_val), slice(None))
    df = pd.DataFrame(
        {"a": [1, 1, 2, 3], "b": [1, 0, 1, 1], "c": [1, 2, 3, 4]}
    ).set_index(["a", "b"])
    cdf = cudf.from_pandas(df)
    expect = df.loc[indexer]
    with cudf.option_context("mode.pandas_compatible", True):
        actual = cdf.loc[indexer]
    assert_eq(actual, expect)


@pytest.mark.parametrize(
    "arg",
    [
        slice(("a", "store"), ("b", "house")),
        slice(None, ("b", "house")),
        slice(("a", "store"), None),
        slice(None),
    ],
)
def test_multiindex_loc_slice(arg):
    rng = np.random.default_rng(seed=0)
    pdf = pd.DataFrame(rng.random(size=(7, 5)))
    pdfIndex = pd.MultiIndex(
        [
            ["a", "b", "c"],
            ["house", "store", "forest"],
            ["clouds", "clear", "storm"],
            ["fire", "smoke", "clear"],
            [
                np.datetime64("2001-01-01", "ns"),
                np.datetime64("2002-01-01", "ns"),
                np.datetime64("2003-01-01", "ns"),
            ],
        ],
        [
            [0, 0, 0, 0, 1, 1, 2],
            [1, 1, 1, 1, 0, 0, 2],
            [0, 0, 2, 2, 2, 0, 1],
            [0, 0, 0, 1, 2, 0, 1],
            [1, 0, 1, 2, 0, 0, 1],
        ],
    )
    pdfIndex.names = ["alpha", "location", "weather", "sign", "timestamp"]
    gdf = cudf.from_pandas(pdf)
    gdfIndex = cudf.from_pandas(pdfIndex)
    pdf = pdf.copy(deep=False)
    pdf.index = pdfIndex
    gdf.index = gdfIndex
    assert_eq(pdf.loc[arg], gdf.loc[arg])


def test_multiindex_loc_errors():
    rng = np.random.default_rng(seed=0)
    pdf = pd.DataFrame(rng.random(size=(7, 5)))
    pdfIndex = pd.MultiIndex(
        [
            ["a", "b", "c"],
            ["house", "store", "forest"],
            ["clouds", "clear", "storm"],
            ["fire", "smoke", "clear"],
            [
                np.datetime64("2001-01-01", "ns"),
                np.datetime64("2002-01-01", "ns"),
                np.datetime64("2003-01-01", "ns"),
            ],
        ],
        [
            [0, 0, 0, 0, 1, 1, 2],
            [1, 1, 1, 1, 0, 0, 2],
            [0, 0, 2, 2, 2, 0, 1],
            [0, 0, 0, 1, 2, 0, 1],
            [1, 0, 1, 2, 0, 0, 1],
        ],
    )
    pdfIndex.names = ["alpha", "location", "weather", "sign", "timestamp"]
    gdfIndex = cudf.from_pandas(pdfIndex)
    gdf = cudf.from_pandas(pdf)
    gdf.index = gdfIndex

    with pytest.raises(KeyError):
        gdf.loc[("a", "store", "clouds", "foo")]
    with pytest.raises(IndexError):
        gdf.loc[
            ("a", "store", "clouds", "fire", "x", "y")
        ]  # too many indexers
    with pytest.raises(IndexError):
        gdf.loc[slice(None, ("a", "store", "clouds", "fire", "x", "y"))]


def test_multiindex_loc_then_column():
    rng = np.random.default_rng(seed=0)
    pdf = pd.DataFrame(rng.random(size=(7, 5)))
    pdfIndex = pd.MultiIndex(
        [
            ["a", "b", "c"],
            ["house", "store", "forest"],
            ["clouds", "clear", "storm"],
            ["fire", "smoke", "clear"],
            [
                np.datetime64("2001-01-01", "ns"),
                np.datetime64("2002-01-01", "ns"),
                np.datetime64("2003-01-01", "ns"),
            ],
        ],
        [
            [0, 0, 0, 0, 1, 1, 2],
            [1, 1, 1, 1, 0, 0, 2],
            [0, 0, 2, 2, 2, 0, 1],
            [0, 0, 0, 1, 2, 0, 1],
            [1, 0, 1, 2, 0, 0, 1],
        ],
    )
    pdfIndex.names = ["alpha", "location", "weather", "sign", "timestamp"]
    gdf = cudf.from_pandas(pdf)
    gdfIndex = cudf.from_pandas(pdfIndex)
    assert_eq(pdfIndex, gdfIndex)
    pdf.index = pdfIndex
    gdf.index = gdfIndex
    # The index is unsorted, which makes things slow but is fine for testing.
    with pytest.warns(pd.errors.PerformanceWarning):
        expected = pdf.loc[("a", "store", "clouds", "fire"), :][0]
    got = gdf.loc[("a", "store", "clouds", "fire"), :][0]
    assert_eq(expected, got)


def test_multiindex_loc_rows_0():
    rng = np.random.default_rng(seed=0)
    pdf = pd.DataFrame(rng.random(size=(7, 5)))
    pdfIndex = pd.MultiIndex(
        [
            ["a", "b", "c"],
            ["house", "store", "forest"],
            ["clouds", "clear", "storm"],
            ["fire", "smoke", "clear"],
            [
                np.datetime64("2001-01-01", "ns"),
                np.datetime64("2002-01-01", "ns"),
                np.datetime64("2003-01-01", "ns"),
            ],
        ],
        [
            [0, 0, 0, 0, 1, 1, 2],
            [1, 1, 1, 1, 0, 0, 2],
            [0, 0, 2, 2, 2, 0, 1],
            [0, 0, 0, 1, 2, 0, 1],
            [1, 0, 1, 2, 0, 0, 1],
        ],
    )
    pdfIndex.names = ["alpha", "location", "weather", "sign", "timestamp"]
    gdfIndex = cudf.from_pandas(pdfIndex)
    gdf = cudf.from_pandas(pdf)
    pdf.index = pdfIndex
    gdf.index = gdfIndex

    assert_exceptions_equal(
        lfunc=pdf.loc.__getitem__,
        rfunc=gdf.loc.__getitem__,
        lfunc_args_and_kwargs=([(("d",), slice(None, None, None))],),
        rfunc_args_and_kwargs=([(("d",), slice(None, None, None))],),
    )


def test_multiindex_loc_rows_1_2_key():
    rng = np.random.default_rng(seed=0)
    pdf = pd.DataFrame(rng.random(size=(7, 5)))
    pdfIndex = pd.MultiIndex(
        [
            ["a", "b", "c"],
            ["house", "store", "forest"],
            ["clouds", "clear", "storm"],
            ["fire", "smoke", "clear"],
            [
                np.datetime64("2001-01-01", "ns"),
                np.datetime64("2002-01-01", "ns"),
                np.datetime64("2003-01-01", "ns"),
            ],
        ],
        [
            [0, 0, 0, 0, 1, 1, 2],
            [1, 1, 1, 1, 0, 0, 2],
            [0, 0, 2, 2, 2, 0, 1],
            [0, 0, 0, 1, 2, 0, 1],
            [1, 0, 1, 2, 0, 0, 1],
        ],
    )
    pdfIndex.names = ["alpha", "location", "weather", "sign", "timestamp"]
    gdfIndex = cudf.from_pandas(pdfIndex)
    gdf = cudf.from_pandas(pdf)
    pdf.index = pdfIndex
    gdf.index = gdfIndex
    assert_eq(pdf.loc[("c", "forest"), :], gdf.loc[("c", "forest"), :])


def test_multiindex_loc_rows_1_1_key():
    rng = np.random.default_rng(seed=0)
    pdf = pd.DataFrame(rng.random(size=(7, 5)))
    pdfIndex = pd.MultiIndex(
        [
            ["a", "b", "c"],
            ["house", "store", "forest"],
            ["clouds", "clear", "storm"],
            ["fire", "smoke", "clear"],
            [
                np.datetime64("2001-01-01", "ns"),
                np.datetime64("2002-01-01", "ns"),
                np.datetime64("2003-01-01", "ns"),
            ],
        ],
        [
            [0, 0, 0, 0, 1, 1, 2],
            [1, 1, 1, 1, 0, 0, 2],
            [0, 0, 2, 2, 2, 0, 1],
            [0, 0, 0, 1, 2, 0, 1],
            [1, 0, 1, 2, 0, 0, 1],
        ],
    )
    pdfIndex.names = ["alpha", "location", "weather", "sign", "timestamp"]
    gdfIndex = cudf.from_pandas(pdfIndex)
    gdf = cudf.from_pandas(pdf)
    pdf.index = pdfIndex
    gdf.index = gdfIndex
    assert_eq(pdf.loc[("c",), :], gdf.loc[("c",), :])


def test_multiindex_index_single_row():
    arrays = [["a", "a", "b", "b"], [1, 2, 3, 4]]
    tuples = list(zip(*arrays, strict=True))
    idx = cudf.MultiIndex.from_tuples(tuples)
    gdf = cudf.DataFrame({"first": range(4), "second": range(4)})
    gdf.index = idx
    pdf = gdf.to_pandas()
    assert_eq(pdf.loc[("b", 3)], gdf.loc[("b", 3)])


@pytest.mark.parametrize("idx_get", [(0, 0), (0, 1), (1, 0), (1, 1)])
@pytest.mark.parametrize("cols_get", [0, 1, [0, 1], [1, 0], [1], [0]])
def test_multiindex_loc_scalar(idx_get, cols_get):
    idx = cudf.MultiIndex.from_tuples([(0, 0), (0, 1), (1, 0), (1, 1)])
    df = cudf.DataFrame({0: range(4), 1: range(10, 50, 10)}, index=idx)
    pdf = df.to_pandas()

    actual = df.loc[idx_get, cols_get]
    expected = pdf.loc[idx_get, cols_get]

    assert_eq(actual, expected)


def test_multiindex_rows_with_wildcard():
    rng = np.random.default_rng(seed=0)
    pdf = pd.DataFrame(rng.random(size=(7, 5)))
    gdf = cudf.from_pandas(pdf)
    pdfIndex = pd.MultiIndex(
        [
            ["a", "b", "c"],
            ["house", "store", "forest"],
            ["clouds", "clear", "storm"],
            ["fire", "smoke", "clear"],
            [
                np.datetime64("2001-01-01", "ns"),
                np.datetime64("2002-01-01", "ns"),
                np.datetime64("2003-01-01", "ns"),
            ],
        ],
        [
            [0, 0, 0, 0, 1, 1, 2],
            [1, 1, 1, 1, 0, 0, 2],
            [0, 0, 2, 2, 2, 0, 1],
            [0, 0, 0, 1, 2, 0, 1],
            [1, 0, 1, 2, 0, 0, 1],
        ],
    )
    pdfIndex.names = ["alpha", "location", "weather", "sign", "timestamp"]
    gdfIndex = cudf.from_pandas(pdfIndex)
    pdf.index = pdfIndex
    gdf.index = gdfIndex
    # The index is unsorted, which makes things slow but is fine for testing.
    with pytest.warns(pd.errors.PerformanceWarning):
        assert_eq(
            pdf.loc[("a",), :].sort_index(), gdf.loc[("a",), :].sort_index()
        )
        assert_eq(
            pdf.loc[(("a"), ("store")), :].sort_index(),
            gdf.loc[(("a"), ("store")), :].sort_index(),
        )
        assert_eq(
            pdf.loc[(("a"), ("store"), ("storm")), :].sort_index(),
            gdf.loc[(("a"), ("store"), ("storm")), :].sort_index(),
        )
        assert_eq(
            pdf.loc[(("a"), ("store"), ("storm"), ("smoke")), :].sort_index(),
            gdf.loc[(("a"), ("store"), ("storm"), ("smoke")), :].sort_index(),
        )
        assert_eq(
            pdf.loc[(slice(None), "store"), :].sort_index(),
            gdf.loc[(slice(None), "store"), :].sort_index(),
        )
        assert_eq(
            pdf.loc[(slice(None), slice(None), "storm"), :].sort_index(),
            gdf.loc[(slice(None), slice(None), "storm"), :].sort_index(),
        )
        assert_eq(
            pdf.loc[
                (slice(None), slice(None), slice(None), "smoke"), :
            ].sort_index(),
            gdf.loc[
                (slice(None), slice(None), slice(None), "smoke"), :
            ].sort_index(),
        )


def test_multicolumn_loc():
    rng = np.random.default_rng(seed=0)
    pdf = pd.DataFrame(rng.random(size=(7, 5)))
    pdfIndex = pd.MultiIndex(
        [
            ["a", "b", "c"],
            ["house", "store", "forest"],
            ["clouds", "clear", "storm"],
            ["fire", "smoke", "clear"],
            [
                np.datetime64("2001-01-01", "ns"),
                np.datetime64("2002-01-01", "ns"),
                np.datetime64("2003-01-01", "ns"),
            ],
        ],
        [
            [0, 0, 0, 0, 1, 1, 2],
            [1, 1, 1, 1, 0, 0, 2],
            [0, 0, 2, 2, 2, 0, 1],
            [0, 0, 0, 1, 2, 0, 1],
            [1, 0, 1, 2, 0, 0, 1],
        ],
    )
    pdfIndex.names = ["alpha", "location", "weather", "sign", "timestamp"]
    pdf = pdf.T
    pdf.columns = pdfIndex
    gdf = cudf.from_pandas(pdf)
    assert_eq(pdf.loc[:, "a"], gdf.loc[:, "a"])
    assert_eq(pdf.loc[:, ("a", "store")], gdf.loc[:, ("a", "store")])
    assert_eq(pdf.loc[:, "a":"b"], gdf.loc[:, "a":"b"])
    assert_eq(pdf.loc[:, ["a", "b"]], gdf.loc[:, ["a", "b"]])


@pytest.mark.parametrize("scalar", [0, 100])
def test_dataframe_loc(scalar):
    size = 123
    rng = np.random.default_rng(seed=0)
    pdf = pd.DataFrame(
        {
            "a": rng.integers(low=0, high=100, size=size),
            "b": rng.random(size).astype(np.float32),
            "c": rng.random(size).astype(np.float64),
            "d": rng.random(size).astype(np.float64),
        }
    )
    pdf.index.name = "index"

    df = cudf.DataFrame(pdf)

    assert_eq(df.loc[:, ["a"]], pdf.loc[:, ["a"]])

    assert_eq(df.loc[:, "d"], pdf.loc[:, "d"])

    # Scalar label
    assert_eq(df.loc[scalar], pdf.loc[scalar])

    # Full slice
    assert_eq(df.loc[:, "c"], pdf.loc[:, "c"])


@pytest.mark.parametrize("step", [1, 5])
def test_dataframe_loc_slice(step):
    size = 123
    rng = np.random.default_rng(seed=0)
    pdf = pd.DataFrame(
        {
            "a": rng.integers(low=0, high=100, size=size),
            "b": rng.random(size).astype(np.float32),
            "c": rng.random(size).astype(np.float64),
            "d": rng.random(size).astype(np.float64),
        }
    )
    pdf.index.name = "index"

    df = cudf.DataFrame(pdf)
    begin = 110
    end = 122

    assert_eq(
        df.loc[begin:end:step, ["c", "d", "a"]],
        pdf.loc[begin:end:step, ["c", "d", "a"]],
    )

    assert_eq(df.loc[begin:end, ["c", "d"]], pdf.loc[begin:end, ["c", "d"]])

    # Slicing on columns:
    assert_eq(
        df.loc[begin:end:step, "a":"c"], pdf.loc[begin:end:step, "a":"c"]
    )

    # Slicing of size 1:
    assert_eq(df.loc[begin:begin, "a"], pdf.loc[begin:begin, "a"])

    # TODO: Pandas changes the dtype here when it shouldn't
    assert_eq(
        df.loc[begin, "a":"a"], pdf.loc[begin, "a":"a"], check_dtype=False
    )


def test_dataframe_loc_arraylike():
    size = 123
    rng = np.random.default_rng(seed=0)
    pdf = pd.DataFrame(
        {
            "a": rng.integers(low=0, high=100, size=size),
            "b": rng.random(size).astype(np.float32),
            "c": rng.random(size).astype(np.float64),
            "d": rng.random(size).astype(np.float64),
        }
    )
    pdf.index.name = "index"

    df = cudf.DataFrame(pdf)
    # Make int64 index
    offset = 50
    df2 = df[offset:]
    pdf2 = pdf[offset:]
    begin = 117
    end = 122
    assert_eq(
        df2.loc[begin:end, ["c", "d", "a"]],
        pdf2.loc[begin:end, ["c", "d", "a"]],
    )

    # loc with list like indexing
    assert_eq(df.loc[[0]], pdf.loc[[0]])
    # loc with column like indexing
    assert_eq(df.loc[cudf.Series([0])], pdf.loc[pd.Series([0])])
    assert_eq(df.loc[cudf.Series([0])._column], pdf.loc[pd.Series([0])])
    assert_eq(df.loc[np.array([0])], pdf.loc[np.array([0])])


@pytest.mark.parametrize(
    "mask",
    [[True, False, False, False, False], [True, False, True, False, True]],
)
@pytest.mark.parametrize("arg", ["a", slice("a", "a"), slice("a", "b")])
def test_dataframe_loc_mask(mask, arg):
    pdf = pd.DataFrame(
        {"a": ["a", "b", "c", "d", "e"], "b": ["f", "g", "h", "i", "j"]}
    )
    gdf = cudf.DataFrame(pdf)

    assert_eq(pdf.loc[mask, arg], gdf.loc[mask, arg])


def test_dataframe_loc_outbound():
    rng = np.random.default_rng(seed=0)
    size = 10
    ha = rng.integers(low=0, high=100, size=size).astype(np.int32)
    hb = rng.random(size).astype(np.float32)
    df = cudf.DataFrame({"a": ha, "b": hb})
    pdf = pd.DataFrame({"a": ha, "b": hb})
    assert_exceptions_equal(lambda: pdf.loc[11], lambda: df.loc[11])


@pytest.mark.parametrize(
    "key, value",
    [
        (("one", "a"), 5),
        ((slice(None), "a"), 5),
        ((slice(None), "a"), range(3)),
        ((slice(None), "a"), [3, 2, 1]),
        ((slice(None, "two"), "a"), range(2)),
        ((slice(None, "two"), "a"), [4, 5]),
        ((["one", "two"], "a"), 5),
        (("one", "c"), 5),
        ((["one", "two"], "c"), 5),
        ((slice(None), "c"), 5),
        ((slice(None), "c"), range(3)),
        ((slice(None), "c"), [3, 2, 1]),
        ((slice(None, "two"), "c"), range(2)),
        ((slice(None, "two"), "c"), [4, 5]),
    ],
)
def test_dataframe_setitem_loc(key, value):
    pdf = pd.DataFrame(
        {"a": [1, 2, 3], "b": ["c", "d", "e"]}, index=["one", "two", "three"]
    )
    gdf = cudf.from_pandas(pdf)
    pdf.loc[key] = value
    gdf.loc[key] = value
    assert_eq(pdf, gdf)


@pytest.mark.parametrize(
    "key, value",
    [
        (("one", "a"), 5),
        ((slice(None), "a"), range(3)),
        ((slice(None), "a"), [3, 2, 1]),
    ],
)
def test_dataframe_setitem_loc_empty_df(key, value):
    pdf, gdf = pd.DataFrame(), cudf.DataFrame()
    pdf.loc[key] = value
    gdf.loc[key] = value
    assert_eq(pdf, gdf, check_dtype=False)


def test_sliced_indexing():
    a = list(range(4, 4 + 150))
    b = list(range(0, 0 + 150))
    pdf = pd.DataFrame({"a": a, "b": b})
    gdf = cudf.DataFrame(pdf)
    pdf = pdf.set_index("a")
    gdf = gdf.set_index("a")
    pidx = pdf.index[:75]
    gidx = gdf.index[:75]

    assert_eq(pdf.loc[pidx], gdf.loc[gidx])


@pytest.mark.parametrize(
    "sli",
    [
        slice("2001", "2002"),
        slice("2002", "2001"),
        slice("2001", None),
    ],
)
@pytest.mark.parametrize("is_dataframe", [True, False])
def test_loc_datetime_index(sli, is_dataframe):
    # Preserve None as None (not NaT) so pandas treats it as an open bound.
    # pd.to_datetime(None) returns NaT, which pandas rejects as a slice
    # bound on a non-monotonic datetime index.
    sli = slice(
        pd.to_datetime(sli.start) if sli.start is not None else None,
        pd.to_datetime(sli.stop) if sli.stop is not None else None,
    )

    if is_dataframe is True:
        pd_data = pd.DataFrame(
            {"a": [1, 2, 3]},
            index=pd.Series(["2001", "2009", "2002"], dtype="datetime64[ns]"),
        )
    else:
        pd_data = pd.Series(
            [1, 2, 3],
            pd.Series(["2001", "2009", "2002"], dtype="datetime64[ns]"),
        )

    gd_data = cudf.from_pandas(pd_data)
    expect = pd_data.loc[sli]
    got = gd_data.loc[sli]
    assert_eq(expect, got)


@pytest.mark.parametrize(
    "sli",
    [
        slice("2001", "2009"),
        slice("2001", "2006"),
        slice(None, "2009"),
    ],
)
def test_loc_datetime_index_string_slice_non_monotonic(sli):
    pdf = pd.DataFrame(
        {"a": [1, 2, 3]},
        index=pd.Series(["2001", "2009", "2002"], dtype="datetime64[ns]"),
    )
    gdf = cudf.from_pandas(pdf)

    with pytest.raises(KeyError, match="non-monotonic DatetimeIndexes"):
        pdf.loc[sli]

    with pytest.raises(KeyError, match="non-monotonic DatetimeIndexes"):
        gdf.loc[sli]


@pytest.mark.parametrize(
    ("key, value"),
    [
        (
            ([0], ["x", "y"]),
            [10, 20],
        ),
        (
            ([0, 2], ["x", "y"]),
            [[10, 30], [20, 40]],
        ),
        (
            (0, ["x", "y"]),
            [10, 20],
        ),
        (
            ([0, 2], "x"),
            [10, 20],
        ),
    ],
)
def test_dataframe_loc_inplace_update(key, value):
    gdf = cudf.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
    pdf = gdf.to_pandas()

    actual = gdf.loc[key] = value
    expected = pdf.loc[key] = value

    assert_eq(expected, actual)


def test_dataframe_loc_inplace_update_string_index():
    gdf = cudf.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]}, index=list("abc"))
    pdf = gdf.to_pandas()

    actual = gdf.loc[["a"], ["x", "y"]] = [10, 20]
    expected = pdf.loc[["a"], ["x", "y"]] = [10, 20]

    assert_eq(expected, actual)


def test_dataframe_loc_iloc_inplace_update_with_RHS_dataframe():
    loc_key = ([0, 2], ["x", "y"])
    iloc_key = [0, 2]
    data = {"x": [10, 20], "y": [30, 40]}
    index = [0, 2]
    gdf = cudf.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
    pdf = gdf.to_pandas()

    actual = gdf.loc[loc_key] = cudf.DataFrame(data, index=cudf.Index(index))
    expected = pdf.loc[loc_key] = pd.DataFrame(data, index=pd.Index(index))
    assert_eq(expected, actual)

    actual = gdf.iloc[iloc_key] = cudf.DataFrame(data, index=cudf.Index(index))
    expected = pdf.iloc[iloc_key] = pd.DataFrame(data, index=pd.Index(index))
    assert_eq(expected, actual)


def test_dataframe_loc_inplace_update_with_invalid_RHS_df_columns():
    # Use float columns so both cudf and pandas can hold NaN when aligning
    # RHS columns that don't match the loc selection (int64 can't hold NaN).
    gdf = cudf.DataFrame({"x": [1.0, 2.0, 3.0], "y": [4.0, 5.0, 6.0]})
    pdf = gdf.to_pandas()

    actual = gdf.loc[[0, 2], ["x", "y"]] = cudf.DataFrame(
        {"b": [10, 20], "y": [30, 40]}, index=cudf.Index([0, 2])
    )
    expected = pdf.loc[[0, 2], ["x", "y"]] = pd.DataFrame(
        {"b": [10, 20], "y": [30, 40]}, index=pd.Index([0, 2])
    )

    assert_eq(expected, actual)


@pytest.mark.parametrize(
    ("key, value"),
    [
        (([0, 2], ["x", "y"]), [[10, 30, 50], [20, 40, 60]]),
        (([0], ["x", "y"]), [[10], [20]]),
    ],
)
def test_dataframe_loc_inplace_update_shape_mismatch(key, value):
    gdf = cudf.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
    with pytest.raises(ValueError, match="shape mismatch:"):
        gdf.loc[key] = value


def test_dataframe_loc_inplace_update_shape_mismatch_RHS_df():
    gdf = cudf.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
    with pytest.raises(ValueError, match="shape mismatch:"):
        gdf.loc[([0, 2], ["x", "y"])] = cudf.DataFrame(
            {"x": [10, 20]}, index=cudf.Index([0, 2])
        )


@pytest.mark.parametrize(
    "end, second_dim, is_error",
    [
        (40, 2, False),
        (50, 3, True),
        (30, 1, False),
    ],
)
@pytest.mark.parametrize("mod", [cp, np])
def test_dataframe_indexing_setitem_np_cp_array(
    end, second_dim, is_error, mod
):
    array = mod.arange(20, end).reshape(-1, second_dim)
    gdf = cudf.DataFrame({"a": range(10), "b": range(10)})
    pdf = gdf.to_pandas()
    if not is_error:
        gdf.loc[:, ["a", "b"]] = array
        pdf.loc[:, ["a", "b"]] = cp.asnumpy(array)

        assert_eq(gdf, pdf)
    else:
        assert_exceptions_equal(
            lfunc=pdf.loc.__setitem__,
            rfunc=gdf.loc.__setitem__,
            lfunc_args_and_kwargs=(
                [(slice(None, None, None), ["a", "b"]), cp.asnumpy(array)],
                {},
            ),
            rfunc_args_and_kwargs=(
                [(slice(None, None, None), ["a", "b"]), array],
                {},
            ),
        )


class TestLocIndexWithOrder:
    # https://github.com/rapidsai/cudf/issues/12833
    @pytest.fixture(params=["increasing", "decreasing", "neither"])
    def order(self, request):
        return request.param

    @pytest.fixture(params=[-1, 1], ids=["reverse", "forward"])
    def take_order(self, request):
        return request.param

    @pytest.fixture(params=["float", "int", "string", "range"])
    def dtype(self, request):
        return request.param

    @pytest.fixture
    def index(self, order, dtype):
        if dtype == "string":
            index = ["a", "h", "f", "z"]
        elif dtype == "int":
            index = [-1, 10, 7, 14]
        elif dtype == "float":
            index = [-1.5, 7.10, 2.4, 11.2]
        elif dtype == "range":
            if order == "increasing":
                return cudf.RangeIndex(2, 10, 3)
            elif order == "decreasing":
                return cudf.RangeIndex(10, 1, -3)
            else:
                return cudf.RangeIndex(10, 20, 3)
        else:
            raise ValueError(f"Unhandled index dtype {dtype}")
        if order == "decreasing":
            return sorted(index, reverse=True)
        elif order == "increasing":
            return sorted(index)
        elif order == "neither":
            return index
        else:
            raise ValueError(f"Unhandled index order {order}")

    @pytest.fixture
    def df(self, index):
        return cudf.DataFrame({"a": range(len(index))}, index=index)

    def test_loc_index_inindex_slice(self, df, take_order):
        pdf = df.to_pandas()
        lo = pdf.index[1]
        hi = pdf.index[-2]
        expect = pdf.loc[lo:hi:take_order]
        actual = df.loc[lo:hi:take_order]
        assert_eq(expect, actual)

    def test_loc_index_inindex_subset(self, df, take_order):
        pdf = df.to_pandas()
        vals = [pdf.index[0], pdf.index[2]][::take_order]
        expect = pdf.loc[vals]
        actual = df.loc[vals]
        assert_eq(expect, actual)

    def test_loc_index_notinindex_slice(self, df, order, dtype, take_order):
        pdf = df.to_pandas()
        lo = pdf.index[1]
        hi = pdf.index[-2]
        if isinstance(lo, str):
            lo = chr(ord(lo) - 1)
            hi = chr(ord(hi) + 1)
        else:
            lo -= 1
            hi += 1
        if order == "neither" and dtype != "range":
            with pytest.raises(KeyError):
                pdf.loc[lo:hi:take_order]
            with pytest.raises(KeyError):
                df.loc[lo:hi:take_order]
        else:
            expect = pdf.loc[lo:hi:take_order]
            actual = df.loc[lo:hi:take_order]
            assert_eq(expect, actual)


def test_loc_single_row_from_slice():
    # see https://github.com/rapidsai/cudf/issues/11930
    pdf = pd.DataFrame({"a": [10, 20, 30], "b": [1, 2, 3]}).set_index("a")
    df = cudf.from_pandas(pdf)
    assert_eq(pdf.loc[5:10], df.loc[5:10])


@pytest.mark.parametrize("indexer", ["loc", "iloc"])
def test_boolean_mask_columns(indexer):
    df = pd.DataFrame(np.zeros((3, 3)))
    cdf = cudf.from_pandas(df)
    mask = [True, False, True]
    expect = getattr(df, indexer)[:, mask]
    got = getattr(cdf, indexer)[:, mask]

    assert_eq(expect, got)


@pytest.mark.parametrize("indexer", ["loc", "iloc"])
@pytest.mark.parametrize(
    "mask",
    [[False, True], [False, False, True, True, True]],
    ids=["too-short", "too-long"],
)
def test_boolean_mask_columns_wrong_length(indexer, mask):
    df = pd.DataFrame(np.zeros((3, 3)))
    cdf = cudf.from_pandas(df)

    with pytest.raises(IndexError):
        getattr(df, indexer)[:, mask]
    with pytest.raises(IndexError):
        getattr(cdf, indexer)[:, mask]


@pytest.mark.parametrize("index_type", ["single", "slice"])
def test_loc_timestamp_issue_8585(index_type):
    rng = np.random.default_rng(seed=0)
    # https://github.com/rapidsai/cudf/issues/8585
    start = pd.Timestamp("2021-03-12 00:00")
    end = pd.Timestamp("2021-03-12 11:00")
    timestamps = pd.date_range(start, end, periods=12)
    value = rng.normal(size=12)
    df = pd.DataFrame(value, index=timestamps, columns=["value"])
    cdf = cudf.from_pandas(df)
    if index_type == "single":
        index = pd.Timestamp("2021-03-12 03:00")
    elif index_type == "slice":
        index = slice(start, end, None)
    else:
        raise ValueError("Invalid index type")
    expect = df.loc[index]
    actual = cdf.loc[index]
    assert_eq(expect, actual)


@pytest.mark.parametrize(
    "index_type",
    [
        "single",
        pytest.param(
            "slice",
            marks=pytest.mark.xfail(
                reason="https://github.com/rapidsai/cudf/issues/8585"
            ),
        ),
        pytest.param(
            "date_range",
            marks=pytest.mark.xfail(
                reason="https://github.com/rapidsai/cudf/issues/8585"
            ),
        ),
    ],
)
def test_loc_multiindex_timestamp_issue_8585(index_type):
    rng = np.random.default_rng(seed=0)
    # https://github.com/rapidsai/cudf/issues/8585
    start = pd.Timestamp("2021-03-12 00:00")
    end = pd.Timestamp("2021-03-12 03:00")
    timestamps = pd.date_range(start, end, periods=4)
    labels = ["A", "B", "C"]
    index = pd.MultiIndex.from_product(
        [timestamps, labels], names=["timestamp", "label"]
    )
    value = rng.normal(size=12)
    df = pd.DataFrame(value, index=index, columns=["value"])
    cdf = cudf.from_pandas(df)
    start = pd.Timestamp("2021-03-12 01:00")
    end = pd.Timestamp("2021-03-12 02:00")
    if index_type == "single":
        index = pd.Timestamp("2021-03-12 03:00")
    elif index_type == "slice":
        index = slice(start, end, None)
    elif index_type == "date_range":
        index = pd.date_range(start, end, periods=2)
    else:
        raise ValueError("Invalid index type")
    expect = df.loc[index]
    actual = cdf.loc[index]
    assert_eq(expect, actual)


@pytest.mark.parametrize(
    "indexer", [(..., 0), (0, ...)], ids=["row_ellipsis", "column_ellipsis"]
)
def test_loc_ellipsis_as_slice_issue_13268(indexer):
    # https://github.com/rapidsai/cudf/issues/13268
    df = pd.DataFrame(np.arange(4).reshape(2, 2))
    cdf = cudf.from_pandas(df)

    expect = df.loc[indexer]
    actual = cdf.loc[indexer]
    assert_eq(expect, actual)


@pytest.mark.xfail(
    reason="https://github.com/rapidsai/cudf/issues/13269 "
    "and https://github.com/rapidsai/cudf/issues/13273"
)
def test_loc_repeated_column_label_issue_13269():
    # https://github.com/rapidsai/cudf/issues/13269
    # https://github.com/rapidsai/cudf/issues/13273
    df = pd.DataFrame(np.arange(4).reshape(2, 2))
    cdf = cudf.from_pandas(df)

    expect = df.loc[:, [0, 1, 0]]
    actual = cdf.loc[:, [0, 1, 0]]
    assert_eq(expect, actual)


def test_loc_column_boolean_mask_issue_13270():
    # https://github.com/rapidsai/cudf/issues/13270
    df = pd.DataFrame(np.arange(4).reshape(2, 2))
    cdf = cudf.from_pandas(df)
    expect = df.loc[:, [True, True]]
    actual = cdf.loc[:, [True, True]]
    assert_eq(expect, actual)


def test_loc_unsorted_index_slice_lookup_keyerror_issue_12833():
    # https://github.com/rapidsai/cudf/issues/12833
    df = pd.DataFrame({"a": [1, 2, 3]}, index=[7, 0, 4])
    cdf = cudf.from_pandas(df)

    # Check that pandas don't change their mind
    with pytest.raises(KeyError):
        df.loc[1:5]

    with pytest.raises(KeyError):
        cdf.loc[1:5]


@pytest.mark.parametrize("index", [range(5), list(range(5))])
def test_loc_missing_label_keyerror_issue_13379(index):
    # https://github.com/rapidsai/cudf/issues/13379
    df = pd.DataFrame({"a": index}, index=index)
    cdf = cudf.from_pandas(df)
    # Check that pandas don't change their mind
    with pytest.raises(KeyError):
        df.loc[[0, 5]]

    with pytest.raises(KeyError):
        cdf.loc[[0, 5]]


@pytest.mark.parametrize("series", [True, False], ids=["Series", "DataFrame"])
def test_loc_repeated_label_ordering_issue_13658(series):
    # https://github.com/rapidsai/cudf/issues/13658
    values = range(2048)
    index = [1 for _ in values]
    if series:
        frame = cudf.Series(values, index=index)
    else:
        frame = cudf.DataFrame({"a": values}, index=index)
    expect = frame.to_pandas().loc[[1]]
    actual = frame.loc[[1]]
    assert_eq(actual, expect)


@pytest.mark.parametrize(
    "arg",
    [
        (2, ("one", "second")),
        (slice(None, None, None), ("two", "first")),
        (1, ("one", "first")),
        (slice(None, None, None), ("two", "second")),
        (slice(None, None, None), ("two", "first", "three")),
        (3, ("two", "first", "three")),
        (slice(None, None, None), ("two",)),
        (0, ("two",)),
    ],
)
def test_loc_dataframe_column_multiindex(arg):
    gdf = cudf.DataFrame(
        [list("abcd"), list("efgh"), list("ijkl"), list("mnop")],
        columns=cudf.MultiIndex.from_product(
            [["one", "two"], ["first", "second"], ["three"]]
        ),
    )
    pdf = gdf.to_pandas()

    assert_eq(gdf.loc[arg], pdf.loc[arg])


def test_loc_setitem_categorical_integer_not_position_based():
    gdf = cudf.DataFrame(range(3), index=cudf.CategoricalIndex([1, 2, 3]))
    pdf = gdf.to_pandas()
    gdf.loc[1] = 10
    pdf.loc[1] = 10
    assert_eq(gdf, pdf)


def test_scalar_loc_row_categoricalindex():
    df = cudf.DataFrame(
        range(4), index=cudf.CategoricalIndex(["a", "a", "b", "c"])
    )
    result = df.loc["a"]
    expected = df.to_pandas().loc["a"]
    assert_eq(result, expected)


@pytest.mark.parametrize("klass", [cudf.DataFrame, cudf.Series])
@pytest.mark.parametrize("indexer", ["iloc", "loc"])
def test_iloc_loc_no_circular_reference(klass, indexer):
    obj = klass([0])
    ref = weakref.ref(obj)
    getattr(obj, indexer)[0]
    del obj
    assert ref() is None


def test_loc_setitem_empty_dataframe():
    pdf = pd.DataFrame(index=["index_1", "index_2", "index_3"])
    gdf = cudf.from_pandas(pdf)
    pdf.loc[["index_1"], "new_col"] = "A"
    gdf.loc[["index_1"], "new_col"] = "A"

    assert_eq(pdf, gdf)


@pytest.mark.parametrize(
    "data",
    [
        [15, 14, 12, 10, 1],
        [1, 10, 12, 14, 15],
    ],
)
@pytest.mark.parametrize(
    "scalar",
    [
        1,
        10,
        15,
        14,
        0,
        2,
    ],
)
def test_loc_datetime_monotonic_with_ts(data, scalar):
    gdf = cudf.DataFrame(
        {"a": [1, 1, 1, 2, 2], "b": [1, 2, 3, 4, 5]},
        index=cudf.Index(data, dtype="datetime64[ns]"),
    )
    pdf = gdf.to_pandas()

    i = pd.Timestamp(scalar)

    actual = gdf.loc[i:]
    expected = pdf.loc[i:]

    assert_eq(actual, expected)

    actual = gdf.loc[:i]
    expected = pdf.loc[:i]

    assert_eq(actual, expected)


@pytest.mark.parametrize("scalar", [1, 0])
def test_loc_datetime_random_with_ts(scalar):
    gdf = cudf.DataFrame(
        {"a": [1, 1, 1, 2, 2], "b": [1, 2, 3, 4, 5]},
        index=cudf.Index([15, 14, 3, 10, 1], dtype="datetime64[ns]"),
    )
    pdf = gdf.to_pandas()

    i = pd.Timestamp(scalar)

    if i not in pdf.index:
        assert_exceptions_equal(
            lambda: pdf.loc[i:],
            lambda: gdf.loc[i:],
            lfunc_args_and_kwargs=([],),
            rfunc_args_and_kwargs=([],),
        )
        assert_exceptions_equal(
            lambda: pdf.loc[:i],
            lambda: gdf.loc[:i],
            lfunc_args_and_kwargs=([],),
            rfunc_args_and_kwargs=([],),
        )
    else:
        actual = gdf.loc[i:]
        expected = pdf.loc[i:]

        assert_eq(actual, expected)

        actual = gdf.loc[:i]
        expected = pdf.loc[:i]

        assert_eq(actual, expected)


@pytest.mark.parametrize("indexer", ["iloc", "loc"])
@pytest.mark.parametrize("dtype", ["category", "timedelta64[ns]"])
def test_loc_iloc_setitem_col_slice_non_cupy_types(indexer, dtype):
    df_pd = pd.DataFrame(range(2), dtype=dtype)
    df_cudf = cudf.DataFrame(df_pd)
    getattr(df_pd, indexer)[:, 0] = getattr(df_pd, indexer)[:, 0]
    getattr(df_cudf, indexer)[:, 0] = getattr(df_cudf, indexer)[:, 0]
    assert_eq(df_pd, df_cudf)


@pytest.mark.parametrize("indexer", ["iloc", "loc"])
@pytest.mark.parametrize(
    "column_slice",
    [
        slice(None),
        slice(0, 0),
        slice(0, 1),
        slice(1, 0),
        slice(0, 2, 2),
    ],
)
def test_slice_empty_columns(indexer, column_slice):
    df_pd = pd.DataFrame(index=[0, 1, 2])
    df_cudf = cudf.from_pandas(df_pd)
    result = getattr(df_cudf, indexer)[:, column_slice]
    expected = getattr(df_pd, indexer)[:, column_slice]
    assert_eq(result, expected)


# ---------------------------------------------------------------------------
# MultiIndex .loc behavior (pandas-compatibility fixes)
# ---------------------------------------------------------------------------


@pytest.fixture
def mi_df():
    # 2-level row MultiIndex, single-level columns.
    index = cudf.MultiIndex.from_product(
        [[1, 2], ["a", "b"]], names=["i", "j"]
    )
    pdf = pd.DataFrame(
        {"x": [10, 20, 30, 40], "y": [1, 2, 3, 4]}, index=index.to_pandas()
    )
    return cudf.from_pandas(pdf), pdf


def test_loc_scalar_column_returns_series(mi_df):
    gdf, pdf = mi_df
    assert_eq(gdf.loc[:, "x"], pdf.loc[:, "x"])
    assert isinstance(gdf.loc[:, "x"], cudf.Series)


def test_loc_full_column_tuple_returns_series():
    columns = pd.MultiIndex.from_tuples(
        [("a", "foo"), ("a", "bar"), ("b", "foo")]
    )
    pdf = pd.DataFrame(np.arange(12).reshape(4, 3), columns=columns)
    gdf = cudf.from_pandas(pdf)
    # full column tuple -> Series; partial top-level label -> DataFrame
    assert_eq(gdf.loc[:, ("a", "foo")], pdf.loc[:, ("a", "foo")])
    assert isinstance(gdf.loc[:, ("a", "foo")], cudf.Series)
    assert isinstance(gdf.loc[:, "a"], cudf.DataFrame)


def test_loc_list_row_scalar_col_returns_series():
    index = cudf.MultiIndex.from_arrays([[3], [4]], names=["c", "d"])
    pdf = pd.DataFrame({"a": 1.0, "b": 2}, index=index.to_pandas())
    gdf = cudf.from_pandas(pdf)
    # list-of-one-tuple row + scalar col -> length-1 Series (not a scalar)
    assert_eq(gdf.loc[[(3, 4)], "b"], pdf.loc[[(3, 4)], "b"])
    assert isinstance(gdf.loc[[(3, 4)], "b"], cudf.Series)
    # full scalar tuple row + scalar col -> a scalar
    assert gdf.loc[(3, 4), "b"] == pdf.loc[(3, 4), "b"]
    assert np.isscalar(gdf.loc[(3, 4), "b"]) or isinstance(
        gdf.loc[(3, 4), "b"], np.generic
    )


@pytest.mark.parametrize("indexer", [{1, 2}, ({1, 2}, ["a"])])
def test_loc_set_indexer_raises(mi_df, indexer):
    gdf, _ = mi_df
    with pytest.raises(TypeError, match="as an indexer is not supported"):
        gdf.loc[indexer]


def test_loc_missing_scalar_label_raises_keyerror(mi_df):
    gdf, pdf = mi_df
    # 5 is not a level-0 label -> KeyError in both cuDF modes (must not be
    # silently reinterpreted as a positional row).
    assert_exceptions_equal(
        lfunc=pdf.loc.__getitem__,
        rfunc=gdf.loc.__getitem__,
        lfunc_args_and_kwargs=([5],),
        rfunc_args_and_kwargs=([5],),
    )


def test_loc_scalar_row_drops_leading_level(mi_df):
    gdf, pdf = mi_df
    # df.loc[scalar, :] selects level-0 == scalar and drops that level.
    assert_eq(gdf.loc[1, :], pdf.loc[1, :])


@pytest.mark.parametrize(
    "row_indexer",
    [slice(None, None, -1), slice(None, None, 2), slice(2, 1, -1)],
)
def test_loc_strided_multiindex_slice_not_implemented(mi_df, row_indexer):
    gdf, _ = mi_df
    # Strided/reversed MultiIndex label slices are not supported on the
    # classic fast path (they would silently return the wrong rows).
    with pytest.raises(NotImplementedError):
        gdf.loc[row_indexer]


def test_loc_empty_multiindex_columns_preserve_dtype():
    # Selecting zero columns from a MultiIndex-columns frame keeps the
    # per-level column dtype (so inferred_type stays "string", not "empty").
    data = np.arange(20, dtype="float64").reshape(5, 4)
    gmi = cudf.MultiIndex.from_product([["foo", "bar"], ["alpha", "beta"]])
    gdf = cudf.DataFrame(data, index=range(5), columns=gmi).sort_index(
        level=0, axis=1
    )
    pdf = pd.DataFrame(
        data, index=range(5), columns=gmi.to_pandas()
    ).sort_index(level=0, axis=1)

    result = gdf.loc[:, ([], slice(None))]
    expected = pdf.loc[:, ([], slice(None))]
    assert result.shape == expected.shape
    assert (
        result.columns.get_level_values(0).inferred_type
        == expected.columns.get_level_values(0).inferred_type
    )


@pytest.mark.parametrize(
    "indexer,expected_values",
    [
        (([10, 20], [2, 3]), [1, 2, 4, 5]),
        (([10, 30], [1]), [0, 6]),
    ],
)
def test_loc_multi_list_cartesian_pandas_compatible(indexer, expected_values):
    index = pd.MultiIndex.from_product([[10, 20, 30], [1, 2, 3]])
    pdf = pd.DataFrame(
        np.arange(9, dtype="int64"), index=index, columns=["Data"]
    )
    gdf = cudf.from_pandas(pdf)
    with cudf.option_context("mode.pandas_compatible", True):
        result = gdf.loc[indexer, "Data"]
    expected = pdf.loc[indexer, "Data"]
    assert_eq(result, expected)
    assert list(result.to_pandas().values) == expected_values


def test_loc_duplicate_labels_no_cartesian_explosion_pandas_compatible():
    # GH#40978: a duplicate label in a level indexer must not multiply the
    # matched rows (the de-dup keeps the result at 5 rows, not 8).
    index = pd.MultiIndex.from_tuples([(1, 1), (1, 2), (2, 1), (2, 2), (1, 1)])
    pdf = pd.DataFrame({"v": range(5)}, index=index)
    gdf = cudf.from_pandas(pdf)
    indexer = ([1, 1, 2], slice(None))
    with cudf.option_context("mode.pandas_compatible", True):
        result = gdf.loc[indexer, :]
    assert_eq(result, pdf.loc[indexer, :])


def test_loc_missing_combination_raises_pandas_compatible():
    index = pd.MultiIndex.from_arrays(
        [["a", "a", "b"], ["1", "2", "2"]], names=["one", "two"]
    )
    pdf = pd.DataFrame({"v": [0, 1, 2]}, index=index)
    gdf = cudf.from_pandas(pdf)
    # "b" and "1" each exist in their level, but ("b", "1") matches no row.
    with cudf.option_context("mode.pandas_compatible", True):
        with pytest.raises(KeyError):
            gdf.loc[("b", "1"), :]


@pytest.mark.parametrize(
    "key", [{1}, {1: 1}, ({1}, "a"), (1, {"a"}), (({1}, 2), "a")]
)
def test_loc_getitem_dict_or_set_indexer_disallowed(key):
    # A set/dict (or a tuple nesting one, incl. nested MultiIndex keys) is
    # not a valid indexer.
    pdf = pd.DataFrame(
        [[1, 2], [3, 4]],
        columns=["a", "b"],
        index=pd.MultiIndex.from_tuples([(1, 2), (3, 4)]),
    )
    gdf = cudf.from_pandas(pdf)
    assert_exceptions_equal(
        lfunc=operator.getitem,
        rfunc=operator.getitem,
        lfunc_args_and_kwargs=([pdf.loc, key],),
        rfunc_args_and_kwargs=([gdf.loc, key],),
    )


def test_loc_getitem_boolean_series_misaligned():
    # A boolean Series indexer carrying a reordered index is aligned by
    # label rather than applied positionally.
    pdf = pd.DataFrame({"a": [1, 4, 2, 3], "b": [5, 6, 7, 8]})
    gdf = cudf.from_pandas(pdf)
    pmask = (pdf["a"] >= 3)[::-1]
    gmask = (gdf["a"] >= 3)[::-1]
    assert_eq(pdf.loc[pmask], gdf.loc[gmask])


def test_loc_getitem_list_of_tuples_multiindex():
    # A list of full-length tuples selects those exact MultiIndex rows;
    # a scalar column key downcasts the result to a Series.
    pdf = pd.DataFrame(
        {"c": [1, 2, 3, 4]},
        index=pd.MultiIndex.from_arrays(
            [[1, 2, 1, 2], [1, 2, 3, 4]], names=["i1", "i2"]
        ),
    )
    gdf = cudf.from_pandas(pdf)
    assert_eq(pdf.loc[[(1, 1)], "c"], gdf.loc[[(1, 1)], "c"])
    assert_eq(pdf.loc[[(1, 1)], ["c"]], gdf.loc[[(1, 1)], ["c"]])


def test_loc_one_level_multiindex_keeps_frame():
    # .loc on a single-level MultiIndex keeps the result a DataFrame with
    # the MultiIndex preserved (no downcast to a Series).
    pdf = pd.DataFrame(
        [[0], [1]],
        index=pd.MultiIndex.from_tuples([("a",), ("b",)], names=["first"]),
    )
    gdf = cudf.from_pandas(pdf)
    assert_eq(pdf.loc["a"], gdf.loc["a"])
