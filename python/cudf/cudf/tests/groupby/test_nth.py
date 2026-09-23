# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import numpy as np
import pandas as pd
import pytest

import cudf
from cudf.testing import assert_groupby_results_equal


@pytest.mark.parametrize("n", [0, 2, 10])
@pytest.mark.parametrize("by", ["a", ["a", "b"], ["a", "c"]])
def test_groupby_nth(n, by):
    pdf = pd.DataFrame(
        {
            "a": [1, 1, 1, 2, 3],
            "b": [1, 2, 2, 2, 1],
            "c": [1, 2, None, 4, 5],
            "d": ["a", "b", "c", "d", "e"],
        }
    )
    gdf = cudf.from_pandas(pdf)

    expect = pdf.groupby(by).nth(n)
    got = gdf.groupby(by).nth(n)

    assert_groupby_results_equal(expect, got, check_dtype=False)


def test_groupby_consecutive_operations():
    df = cudf.DataFrame([[1, np.nan], [1, 4], [5, 6]], columns=["A", "B"])
    pdf = df.to_pandas()

    gg = df.groupby("A")
    pg = pdf.groupby("A")

    actual = gg.nth(-1)
    expected = pg.nth(-1)

    assert_groupby_results_equal(actual, expected, check_dtype=False)

    actual = gg.nth(0)
    expected = pg.nth(0)

    assert_groupby_results_equal(actual, expected, check_dtype=False)

    actual = gg.cumsum()
    expected = pg.cumsum()

    assert_groupby_results_equal(actual, expected, check_dtype=False)

    actual = gg.cumcount()
    expected = pg.cumcount()

    assert_groupby_results_equal(actual, expected, check_dtype=False)

    actual = gg.cumsum()
    expected = pg.cumsum()

    assert_groupby_results_equal(actual, expected, check_dtype=False)


@pytest.mark.parametrize(
    "arg",
    [
        0,
        -1,
        [0, 1],
        slice(None, 2),
        slice(1, None),
        slice(None, None, 2),
        slice(-2, None),
    ],
)
def test_nth_selector_indexing(arg):
    # GroupBy.nth mirrors pandas' GroupByNthSelector: it supports both the
    # call form gb.nth(n) and the index form gb.nth[n], acting as a
    # positional row filter that keeps the original index and row order.
    pdf = pd.DataFrame(
        {
            "a": [1, 1, 1, 2, 2, 3],
            "b": [10, 20, 30, 40, 50, 60],
        },
        index=[5, 4, 3, 2, 1, 0],
    )
    gdf = cudf.from_pandas(pdf)

    assert_groupby_results_equal(
        pdf.groupby("a").nth[arg],
        gdf.groupby("a").nth[arg],
    )
    if not isinstance(arg, slice):
        assert_groupby_results_equal(
            pdf.groupby("a").nth(arg),
            gdf.groupby("a").nth(arg),
        )


def test_nth_invalid_args():
    gb = cudf.DataFrame({"a": [1, 1, 2], "b": [1, 2, 3]}).groupby("a")
    with pytest.raises(TypeError, match="Invalid index"):
        gb.nth(3.14)
    with pytest.raises(ValueError, match="Invalid step"):
        gb.nth(slice(None, None, -1))
    with pytest.raises(NotImplementedError):
        gb.nth(0, dropna="any")
