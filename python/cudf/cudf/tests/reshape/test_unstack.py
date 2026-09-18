# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import re

import pandas as pd
import pytest

import cudf
from cudf.testing import assert_eq


@pytest.mark.parametrize(
    "level",
    [
        0,
        1,
        2,
        "foo",
        "bar",
        "baz",
        [],
        pytest.param(
            [0, 1],
            marks=pytest.mark.xfail(
                reason="Categorical column indexes not supported"
            ),
        ),
        ["foo"],
        pytest.param(
            ["foo", "bar"],
            marks=pytest.mark.xfail(
                reason="Categorical column indexes not supported"
            ),
        ),
        pytest.param(
            [0, 1, 2],
            marks=pytest.mark.xfail(reason="Pandas behaviour unclear"),
        ),
        pytest.param(
            ["foo", "bar", "baz"],
            marks=pytest.mark.xfail(reason="Pandas behaviour unclear"),
        ),
    ],
)
def test_unstack_multiindex(level):
    pdf = pd.DataFrame(
        {
            "foo": ["one", "one", "one", "two", "two", "two"],
            "bar": pd.Categorical(["A", "B", "C", "A", "B", "C"]),
            "baz": [1, 2, 3, 4, 5, 6],
            "zoo": ["x", "y", "z", "q", "w", "t"],
        }
    ).set_index(["foo", "bar", "baz"])
    gdf = cudf.from_pandas(pdf)
    assert_eq(
        pdf.unstack(level=level),
        gdf.unstack(level=level),
        check_dtype=False,
    )


@pytest.mark.parametrize(
    "index",
    [
        pd.Index(range(0, 5), name=None),
        pd.Index(range(0, 5), name="row_index"),
        pd.CategoricalIndex(["d", "e", "f", "g", "h"]),
    ],
)
@pytest.mark.parametrize(
    "col_idx",
    [
        pd.Index(["a", "b"], name=None),
        pd.Index(["a", "b"], name="col_index"),
        pd.MultiIndex.from_tuples(
            [("c", 1), ("c", 2)], names=["col_index1", None]
        ),
    ],
)
def test_unstack_index(index, col_idx):
    data = {
        "A": [1.0, 2.0, 3.0, 4.0, 5.0],
        "B": [11.0, 12.0, 13.0, 14.0, 15.0],
    }
    pdf = pd.DataFrame(data)
    gdf = cudf.from_pandas(pdf)

    pdf.index = index
    pdf.columns = col_idx

    gdf.index = cudf.from_pandas(index)
    gdf.columns = cudf.from_pandas(col_idx)

    assert_eq(pdf.unstack(), gdf.unstack())


def test_unstack_index_invalid():
    gdf = cudf.DataFrame({"a": [1, 2, 3], "b": ["a", "b", "c"]})
    with pytest.raises(
        ValueError,
        match=re.escape(
            "Calling unstack() on single index dataframe with "
            "different column datatype is not supported."
        ),
    ):
        gdf.unstack()


@pytest.mark.parametrize("level", [-1, 0, 1, "foo", "bar"])
@pytest.mark.parametrize("name", [None, "quux"])
def test_series_unstack_multiindex(level, name):
    index = pd.MultiIndex.from_tuples(
        [
            ("one", "a"),
            ("one", "b"),
            ("two", "a"),
            ("two", "b"),
        ],
        names=["foo", "bar"],
    )
    ps = pd.Series([1, 2, 3, 4], index=index, name=name)
    gs = cudf.from_pandas(ps)
    assert_eq(ps.unstack(level=level), gs.unstack(level=level))


def test_series_unstack_index_invalid():
    gs = cudf.Series([1, 2, 3], index=["a", "b", "c"])
    with pytest.raises(
        ValueError,
        match=re.escape(
            "index must be a MultiIndex to unstack, "
            "<class 'cudf.core.index.Index'> was passed"
        ),
    ):
        gs.unstack()


def test_series_unstack_empty_level_is_a_noop():
    index = pd.MultiIndex.from_tuples(
        [("one", "a"), ("one", "b"), ("two", "a"), ("two", "b")]
    )
    ps = pd.Series([1, 2, 3, 4], index=index, name="v")
    gs = cudf.from_pandas(ps)
    assert_eq(ps.unstack(level=[]), gs.unstack(level=[]))
