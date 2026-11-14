# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

import cudf
from cudf.testing import assert_eq


def test_datetime_dataframe():
    data = {
        "timearray": np.array(
            [0, 1, None, 2, 20, None, 897], dtype="datetime64[ms]"
        )
    }
    gdf = cudf.DataFrame(data)
    pdf = pd.DataFrame(data)

    assert_eq(pdf, gdf)

    assert_eq(pdf.dropna(), gdf.dropna())

    assert_eq(pdf.isnull(), gdf.isnull())

    data = np.array([0, 1, None, 2, 20, None, 897], dtype="datetime64[ms]")
    gs = cudf.Series(data)
    ps = pd.Series(data)

    assert_eq(ps, gs)

    assert_eq(ps.dropna(), gs.dropna())

    assert_eq(ps.isnull(), gs.isnull())


@pytest.mark.parametrize(
    "data",
    [
        {"a": [1, 2, None]},
        {"a": [1, 2, None], "b": [3, 4, 5]},
        {"a": [1, 2, None], "b": [3, 4, None]},
        {"a": [None, 1, 2], "b": [1, 2, None]},
        {"a": [None, 1, None], "b": [None, 2, None]},
        {"a": [None, None, 1], "b": [1, 2, None]},
        {"a": ["d", "e", "f"], "b": ["a", None, "c"]},
    ],
)
def test_dropna_dataframe(data, dropna_how, axis, inplace):
    pdf = pd.DataFrame(data)
    gdf = cudf.from_pandas(pdf)

    expected = pdf.dropna(axis=axis, how=dropna_how)
    actual = gdf.dropna(axis=axis, how=dropna_how, inplace=inplace)

    if inplace:
        assert actual is None
        actual = gdf

    assert_eq(expected, actual)


@pytest.mark.parametrize(
    "data",
    [
        {
            "a": pa.array([None, None, None], type=pa.float64()),
            "b": [1, 2, None],
        },
        {
            "a": pa.array([np.nan, np.nan, np.nan]),
            "b": [1, 2, None],
        },
        {"a": pa.array([None, None, None], type=pa.string())},
    ],
)
def test_dropna_with_all_nulls(dropna_how, data, axis):
    gdf = cudf.DataFrame(data)
    pdf = gdf.to_pandas()

    assert_eq(
        pdf.dropna(axis=axis, how=dropna_how),
        gdf.dropna(axis=axis, how=dropna_how),
        check_dtype=False,
    )


@pytest.mark.parametrize(
    "data,subset",
    [
        ({"a": [1, None], "b": [1, 2]}, ["a"]),
        ({"a": [1, None], "b": [1, 2]}, ["b"]),
        ({"a": [1, None], "b": [1, 2]}, []),
        ({"a": [1, 2], "b": [1, 2]}, ["b"]),
        ({"a": [1, 2, None], "b": [1, None, 2]}, ["a"]),
        ({"a": [1, 2, None], "b": [1, None, 2]}, ["b"]),
        ({"a": [1, 2, None], "b": [1, None, 2]}, ["a", "b"]),
    ],
)
def test_dropna_subset_rows(data, subset):
    pdf = pd.DataFrame(data)
    gdf = cudf.from_pandas(pdf)

    assert_eq(pdf.dropna(subset=subset), gdf.dropna(subset=subset))


@pytest.mark.parametrize(
    "data, subset",
    [
        ({"a": [1, None], "b": [1, 2]}, [0]),
        ({"a": [1, None], "b": [1, 2]}, [1]),
        ({"a": [1, None], "b": [1, 2]}, []),
        ({"a": [1, 2], "b": [1, 2]}, [0]),
        ({"a": [1, 2], "b": [None, 2], "c": [3, None]}, [0]),
        ({"a": [1, 2], "b": [None, 2], "c": [3, None]}, [1]),
        ({"a": [1, 2], "b": [None, 2], "c": [3, None]}, [0, 1]),
    ],
)
def test_dropna_subset_cols(data, subset):
    pdf = pd.DataFrame(data)
    gdf = cudf.from_pandas(pdf)

    assert_eq(
        pdf.dropna(axis=1, subset=subset), gdf.dropna(axis=1, subset=subset)
    )


# TODO: can't test with subset=[] below since Pandas
# returns empty DF when both subset=[] and thresh are specified.
@pytest.mark.parametrize("thresh", [0, 1, 2])
@pytest.mark.parametrize("subset", [None, ["a"], ["b"], ["a", "b"]])
def test_dropna_thresh(thresh, subset):
    pdf = pd.DataFrame({"a": [1, 2, None, None], "b": [1, 2, 3, None]})
    gdf = cudf.from_pandas(pdf)

    assert_eq(
        pdf.dropna(axis=0, thresh=thresh, subset=subset),
        gdf.dropna(axis=0, thresh=thresh, subset=subset),
    )


@pytest.mark.parametrize("thresh", [0, 1, 2])
@pytest.mark.parametrize("subset", [None, [0], [1], [0, 1]])
def test_dropna_thresh_cols(thresh, subset, inplace):
    pdf = pd.DataFrame(
        {"a": [1, 2], "b": [3, 4], "c": [5, None], "d": [np.nan, np.nan]}
    )
    gdf = cudf.from_pandas(pdf)

    expected = pdf.dropna(axis=1, thresh=thresh, subset=subset)
    actual = gdf.dropna(axis=1, thresh=thresh, subset=subset, inplace=inplace)

    if inplace:
        assert actual is None
        actual = gdf

    assert_eq(
        expected,
        actual,
    )


@pytest.mark.parametrize(
    "data",
    [
        {
            "key": [1, 2, 10],
            "val": pa.array([np.nan, 3.0, 1.0]),
            "abc": [np.nan, None, 1],
        },
        {
            "key": [None, 2, 1],
            "val": pa.array([3.0, None, 0.1]),
            "abc": [None, 1, None],
        },
    ],
)
def test_dropna_dataframe_np_nan(data, axis):
    gdf = cudf.DataFrame(data)
    pd_data = {
        key: value.to_pandas() if isinstance(value, cudf.Series) else value
        for key, value in data.items()
    }
    pdf = pd.DataFrame(pd_data)

    assert_eq(pdf.dropna(axis=axis), gdf.dropna(axis=axis), check_dtype=False)
