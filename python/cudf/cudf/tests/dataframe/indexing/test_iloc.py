# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import numpy as np
import pandas as pd
import pytest

import cudf
from cudf.testing import assert_eq


@pytest.mark.parametrize(
    "iloc_rows",
    [
        0,
        1,
        slice(None, 0),
        slice(None, 1),
        slice(0, 1),
        slice(1, 2),
        slice(0, 2),
        slice(0, None),
        slice(1, None),
    ],
)
@pytest.mark.parametrize(
    "iloc_columns",
    [
        0,
        1,
        slice(None, 0),
        slice(None, 1),
        slice(0, 1),
        slice(1, 2),
        slice(0, 2),
        slice(0, None),
        slice(1, None),
    ],
)
def test_multiindex_iloc(iloc_rows, iloc_columns):
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
    assert_eq(pdfIndex, gdfIndex)
    pdf.index = pdfIndex
    gdf.index = gdfIndex
    presult = pdf.iloc[iloc_rows, iloc_columns]
    gresult = gdf.iloc[iloc_rows, iloc_columns]
    if isinstance(gresult, cudf.DataFrame):
        assert_eq(
            presult, gresult, check_index_type=False, check_column_type=False
        )
    else:
        assert_eq(presult, gresult, check_index_type=False, check_dtype=False)


def test_multiindex_iloc_scalar():
    arrays = [["a", "a", "b", "b"], [1, 2, 3, 4]]
    tuples = list(zip(*arrays, strict=True))
    idx = cudf.MultiIndex.from_tuples(tuples)
    rng = np.random.default_rng(0)
    gdf = cudf.DataFrame({"first": rng.random(4), "second": rng.random(4)})
    gdf.index = idx

    pdf = gdf.to_pandas()
    assert_eq(pdf.iloc[3], gdf.iloc[3])


@pytest.mark.parametrize(
    "iloc_rows",
    [
        0,
        1,
        slice(None, 0),
        slice(None, 1),
        slice(0, 1),
        slice(1, 2),
        slice(0, 2),
        slice(0, None),
        slice(1, None),
    ],
)
@pytest.mark.parametrize(
    "iloc_columns",
    [
        0,
        1,
        slice(None, 0),
        slice(None, 1),
        slice(0, 1),
        slice(1, 2),
        slice(0, 2),
        slice(0, None),
        slice(1, None),
    ],
)
def test_multicolumn_iloc(iloc_rows, iloc_columns):
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
    assert_eq(pdfIndex, gdfIndex)
    pdf.index = pdfIndex
    gdf.index = gdfIndex
    pdf = pdf.T
    gdf = gdf.T
    presult = pdf.iloc[iloc_rows, iloc_columns]
    gresult = gdf.iloc[iloc_rows, iloc_columns]
    if hasattr(gresult, "name") and isinstance(gresult.name, tuple):
        name = gresult.name[len(gresult.name) - 1]
        if isinstance(name, str) and "cudf" in name:
            gresult.name = name
    if isinstance(presult, pd.DataFrame):
        assert_eq(
            presult, gresult, check_index_type=False, check_column_type=False
        )
    else:
        assert_eq(presult, gresult, check_index_type=False, check_dtype=False)


def test_multiindex_multicolumn_zero_row_slice():
    gdf = cudf.DataFrame(
        {"x": [1, 5, 3, 4, 1], "y": [1, 1, 2, 2, 5], "z": [1, 2, 3, 4, 5]}
    )
    pdf = gdf.to_pandas()
    gdg = gdf.groupby(["x", "y"]).agg({"z": ["count"]}).iloc[:0]
    pdg = pdf.groupby(["x", "y"]).agg({"z": ["count"]}).iloc[:0]
    assert_eq(pdg, gdg, check_dtype=False)


def test_dataframe_iloc():
    nelem = 20
    rng = np.random.default_rng(seed=0)
    data = {
        "a": rng.integers(low=0, high=100, size=nelem).astype(np.int32),
        "b": rng.random(nelem).astype(np.float32),
    }
    gdf = cudf.DataFrame(data)
    pdf = pd.DataFrame(data)

    assert_eq(gdf.iloc[-1:1], pdf.iloc[-1:1])
    assert_eq(gdf.iloc[nelem - 1 : -1], pdf.iloc[nelem - 1 : -1])
    assert_eq(gdf.iloc[0 : nelem - 1], pdf.iloc[0 : nelem - 1])
    assert_eq(gdf.iloc[0:nelem], pdf.iloc[0:nelem])
    assert_eq(gdf.iloc[1:1], pdf.iloc[1:1])
    assert_eq(gdf.iloc[1:2], pdf.iloc[1:2])
    assert_eq(gdf.iloc[nelem - 1 : nelem + 1], pdf.iloc[nelem - 1 : nelem + 1])
    assert_eq(gdf.iloc[nelem : nelem * 2], pdf.iloc[nelem : nelem * 2])

    assert_eq(gdf.iloc[-1 * nelem], pdf.iloc[-1 * nelem])
    assert_eq(gdf.iloc[-1], pdf.iloc[-1])
    assert_eq(gdf.iloc[0], pdf.iloc[0])
    assert_eq(gdf.iloc[1], pdf.iloc[1])
    assert_eq(gdf.iloc[nelem - 1], pdf.iloc[nelem - 1])

    # Repeat the above with iat[]
    assert_eq(gdf.iloc[-1:1], gdf.iat[-1:1])
    assert_eq(gdf.iloc[nelem - 1 : -1], gdf.iat[nelem - 1 : -1])
    assert_eq(gdf.iloc[0 : nelem - 1], gdf.iat[0 : nelem - 1])
    assert_eq(gdf.iloc[0:nelem], gdf.iat[0:nelem])
    assert_eq(gdf.iloc[1:1], gdf.iat[1:1])
    assert_eq(gdf.iloc[1:2], gdf.iat[1:2])
    assert_eq(gdf.iloc[nelem - 1 : nelem + 1], gdf.iat[nelem - 1 : nelem + 1])
    assert_eq(gdf.iloc[nelem : nelem * 2], gdf.iat[nelem : nelem * 2])

    assert_eq(gdf.iloc[-1 * nelem], gdf.iat[-1 * nelem])
    assert_eq(gdf.iloc[-1], gdf.iat[-1])
    assert_eq(gdf.iloc[0], gdf.iat[0])
    assert_eq(gdf.iloc[1], gdf.iat[1])
    assert_eq(gdf.iloc[nelem - 1], gdf.iat[nelem - 1])

    # iloc with list like indexing
    assert_eq(gdf.iloc[[0]], pdf.iloc[[0]])
    # iloc with column like indexing
    assert_eq(gdf.iloc[cudf.Series([0])], pdf.iloc[pd.Series([0])])
    assert_eq(gdf.iloc[cudf.Series([0])._column], pdf.iloc[pd.Series([0])])
    assert_eq(gdf.iloc[np.array([0])], pdf.loc[np.array([0])])


def test_dataframe_iloc_tuple():
    rng = np.random.default_rng(seed=0)
    nelem = 20
    data = {
        "a": rng.integers(low=0, high=100, size=nelem).astype(np.int32),
        "b": rng.random(nelem).astype(np.float32),
    }
    gdf = cudf.DataFrame(data)
    pdf = pd.DataFrame(data)

    assert_eq(gdf.iloc[1, [1]], pdf.iloc[1, [1]], check_dtype=False)
    assert_eq(gdf.iloc[:, -1], pdf.iloc[:, -1])


def test_dataframe_iloc_index_error():
    rng = np.random.default_rng(seed=0)
    nelem = 20
    data = {
        "a": rng.integers(low=0, high=100, size=nelem).astype(np.int32),
        "b": rng.random(nelem).astype(np.float32),
    }
    gdf = cudf.DataFrame(data)
    pdf = pd.DataFrame(data)

    with pytest.raises(IndexError):
        pdf.iloc[nelem * 2]
    with pytest.raises(IndexError):
        gdf.iloc[nelem * 2]


@pytest.mark.parametrize(
    "key, value",
    [
        ((0, 0), 5),
        ((slice(None), 0), 5),
        ((slice(None), 0), range(3)),
        ((slice(None, -1), 0), range(2)),
        (([0, 1], 0), 5),
    ],
)
def test_dataframe_setitem_iloc(key, value):
    pdf = pd.DataFrame(
        {"a": [1, 2, 3], "b": ["c", "d", "e"]}, index=["one", "two", "three"]
    )
    gdf = cudf.from_pandas(pdf)
    pdf.iloc[key] = value
    gdf.iloc[key] = value
    assert_eq(pdf, gdf)


@pytest.mark.parametrize(
    "key,value",
    [
        ((0, 0), 5.0),
        ((slice(None), 0), 5.0),
        ((slice(None), 0), np.arange(7, dtype="float64")),
    ],
)
def test_dataframe_setitem_iloc_multiindex(key, value):
    rng = np.random.default_rng(seed=0)
    pdf = pd.DataFrame(rng.random(size=(7, 5)))
    pdfIndex = pd.MultiIndex(
        [
            ["a", "b", "c"],
            ["house", "store", "forest"],
            ["clouds", "clear", "storm"],
            ["fire", "smoke", "clear"],
        ],
        [
            [0, 0, 0, 0, 1, 1, 2],
            [1, 1, 1, 1, 0, 0, 2],
            [0, 0, 2, 2, 2, 0, 1],
            [0, 0, 0, 1, 2, 0, 1],
        ],
    )
    pdfIndex.names = ["alpha", "location", "weather", "sign"]
    pdf.index = pdfIndex
    gdf = cudf.from_pandas(pdf)

    pdf.iloc[key] = value
    gdf.iloc[key] = value

    assert_eq(pdf, gdf)


def test_boolean_indexing_single_row():
    pdf = pd.DataFrame(
        {"a": [1, 2, 3], "b": ["c", "d", "e"]}, index=["one", "two", "three"]
    )
    gdf = cudf.from_pandas(pdf)
    assert_eq(
        pdf.loc[[True, False, False], :], gdf.loc[[True, False, False], :]
    )


@pytest.mark.parametrize("index", [["a"], ["a", "a"], ["a", "a", "b", "c"]])
def test_iloc_categorical_index(index):
    gdf = cudf.DataFrame({"data": range(len(index))}, index=index)
    gdf.index = gdf.index.astype("category")
    pdf = gdf.to_pandas()
    expect = pdf.iloc[:, 0]
    got = gdf.iloc[:, 0]
    assert_eq(expect, got)


@pytest.mark.parametrize(
    ("key, value"),
    [
        ([0], [10, 20]),
        ([0, 2], [[10, 30], [20, 40]]),
        (([0, 2], [0, 1]), [[10, 30], [20, 40]]),
        (([0, 2], 0), [10, 30]),
        ((0, [0, 1]), [20, 40]),
    ],
)
def test_dataframe_iloc_inplace_update(key, value):
    gdf = cudf.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
    pdf = gdf.to_pandas()

    actual = gdf.iloc[key] = value
    expected = pdf.iloc[key] = value

    assert_eq(expected, actual)


@pytest.mark.parametrize(
    ("key, value"),
    [
        ([0, 2], [[10, 30, 50], [20, 40, 60]]),
        ([0], [[10], [20]]),
    ],
)
def test_dataframe_iloc_inplace_update_shape_mismatch(key, value):
    gdf = cudf.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
    with pytest.raises(ValueError, match="shape mismatch:"):
        gdf.iloc[key] = value


def test_dataframe_iloc_inplace_update_shape_mismatch_RHS_df():
    gdf = cudf.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
    with pytest.raises(ValueError, match="shape mismatch:"):
        gdf.iloc[[0, 2]] = cudf.DataFrame(
            {"x": [10, 20]}, index=cudf.Index([0, 2])
        )


def test_iloc_single_row_with_nullable_column():
    # see https://github.com/NVIDIA/cudf/issues/11349
    pdf = pd.DataFrame({"a": [0, 1, 2, 3], "b": [0.1, 0.2, None, 0.4]})
    df = cudf.from_pandas(pdf)

    df.iloc[0]  # before the fix for #11349 this would segfault
    assert_eq(pdf.iloc[0], df.iloc[0])


@pytest.mark.xfail(
    raises=NotImplementedError,
    reason="cuDF needs to implement boolean mask along columns",
)
def test_boolean_mask_columns_iloc_series():
    df = pd.DataFrame(np.zeros((3, 3)))
    cdf = cudf.from_pandas(df)

    mask = pd.Series([True, False, True], dtype=bool)
    df.iloc[:, mask] = 1

    cdf.iloc[:, mask] = 1
    assert_eq(df, cdf)


def test_iloc_column_boolean_mask_issue_13265():
    # https://github.com/NVIDIA/cudf/issues/13265
    df = pd.DataFrame(np.arange(4).reshape(2, 2))
    cdf = cudf.from_pandas(df)
    expect = df.iloc[:, [True, True]]
    actual = cdf.iloc[:, [True, True]]
    assert_eq(expect, actual)


def test_iloc_repeated_column_label_issue_13266():
    # https://github.com/NVIDIA/cudf/issues/13266
    # https://github.com/NVIDIA/cudf/issues/13273
    df = pd.DataFrame(np.arange(4).reshape(2, 2))
    cdf = cudf.from_pandas(df)

    with pytest.raises(NotImplementedError):
        cdf.iloc[:, [0, 1, 0]]


@pytest.mark.parametrize(
    "indexer",
    [
        (..., 0),
        (0, ...),
    ],
    ids=["row_ellipsis", "column_ellipsis"],
)
def test_iloc_ellipsis_as_slice_issue_13267(indexer):
    # https://github.com/NVIDIA/cudf/issues/13267
    df = pd.DataFrame(np.arange(4).reshape(2, 2))
    cdf = cudf.from_pandas(df)

    expect = df.iloc[indexer]
    actual = cdf.iloc[indexer]
    assert_eq(expect, actual)


@pytest.mark.parametrize(
    "indexer",
    [
        0,
        (slice(None), 0),
        ([0, 2], 1),
        (slice(None), slice(None)),
        (slice(None), [1, 0]),
        (0, 0),
        (1, [1, 0]),
        ([1, 0], 0),
        ([1, 2], [0, 1]),
    ],
)
def test_iloc_multiindex_lookup_as_label_issue_13515(indexer):
    # https://github.com/NVIDIA/cudf/issues/13515
    df = pd.DataFrame(
        {"a": [1, 1, 3], "b": [2, 3, 4], "c": [1, 6, 7], "d": [1, 8, 9]}
    ).set_index(["a", "b"])
    cdf = cudf.from_pandas(df)

    expect = df.iloc[indexer]
    actual = cdf.iloc[indexer]
    assert_eq(expect, actual)


def test_iloc_loc_mixed_dtype():
    df = cudf.DataFrame({"a": ["a", "b"], "b": [0, 1]})
    with cudf.option_context("mode.pandas_compatible", True):
        with pytest.raises(TypeError):
            df.iloc[0]
        with pytest.raises(TypeError):
            df.loc[0]
    df = df.astype("str")
    pdf = df.to_pandas()

    assert_eq(df.iloc[0], pdf.iloc[0])
    assert_eq(df.loc[0], pdf.loc[0])


@pytest.mark.parametrize("typ", ["datetime64[ns]", "timedelta64[ns]"])
@pytest.mark.parametrize(
    "idx_method, row_key, col_key", [["iloc", 0, 0], ["loc", "a", "a"]]
)
def test_dataframe_iloc_scalar_datetimelike_return_pd_scalar(
    typ, idx_method, row_key, col_key
):
    obj = cudf.DataFrame(
        [1, 2, 3], index=list("abc"), columns=["a"], dtype=typ
    )
    with cudf.option_context("mode.pandas_compatible", True):
        result = getattr(obj, idx_method)[row_key, col_key]
    expected = getattr(obj.to_pandas(), idx_method)[row_key, col_key]
    assert result == expected


@pytest.mark.parametrize(
    "idx_method, row_key, col_key", [["iloc", 0, 0], ["loc", "a", "a"]]
)
def test_dataframe_iloc_scalar_interval_return_pd_scalar(
    idx_method, row_key, col_key
):
    iidx = cudf.IntervalIndex.from_breaks([1, 2, 3])
    obj = cudf.DataFrame({"a": iidx}, index=list("ab"))
    with cudf.option_context("mode.pandas_compatible", True):
        result = getattr(obj, idx_method)[row_key, col_key]
    expected = getattr(obj.to_pandas(), idx_method)[row_key, col_key]
    assert result == expected


def test_sliced_categorical_as_ordered():
    df = cudf.DataFrame({"a": list("caba"), "b": list(range(4))})
    df["a"] = df["a"].astype("category")
    df = df.iloc[:2]
    result = df["a"].cat.as_ordered()
    expected = cudf.Series(
        ["c", "a"],
        dtype=cudf.CategoricalDtype(list("abc"), ordered=True),
        name="a",
    )
    assert_eq(result, expected)


@pytest.mark.parametrize("obj", [cudf.Series, cudf.Index])
def test_iloc_columns_with_cudf_object(obj):
    data = {"a": [1, 2], "c": [0, 2], "d": ["c", "a"]}
    col_indexer = obj([0, 2])
    result = cudf.DataFrame(data).iloc[:, col_indexer]
    expected = pd.DataFrame(data).iloc[:, col_indexer.to_pandas()]
    assert_eq(result, expected)
