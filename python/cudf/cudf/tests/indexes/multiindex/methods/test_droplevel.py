# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import itertools

import numpy as np
import pandas as pd
import pytest

import cudf
from cudf.testing import assert_eq


@pytest.mark.parametrize(
    "level",
    [
        [],
        "alpha",
        "location",
        "weather",
        0,
        1,
        [0, 1],
        -1,
        [-1, -2],
        [-1, "weather"],
    ],
)
def test_multiindex_droplevel_simple(level):
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
    assert_eq(pdfIndex.droplevel(level), gdfIndex.droplevel(level))


@pytest.mark.parametrize(
    "level",
    itertools.chain(
        *(
            itertools.combinations(
                ("alpha", "location", "weather", "sign", "timestamp"), r
            )
            for r in range(5)
        )
    ),
)
def test_multiindex_droplevel_name(level):
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
    level = list(level)
    gdfIndex = cudf.from_pandas(pdfIndex)
    assert_eq(pdfIndex.droplevel(level), gdfIndex.droplevel(level))


@pytest.mark.parametrize(
    "level",
    itertools.chain(*(itertools.combinations(range(5), r) for r in range(5))),
)
def test_multiindex_droplevel_index(level):
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
    level = list(level)
    gdfIndex = cudf.from_pandas(pdfIndex)
    assert_eq(pdfIndex.droplevel(level), gdfIndex.droplevel(level))


@pytest.mark.parametrize("level", [-3, 2])
def test_multiindex_droplevel_out_of_bounds(level):
    data = [(1, 2), (3, 4)]
    pidx = pd.MultiIndex.from_tuples(data, names=["l0", "l1"])
    gidx = cudf.MultiIndex.from_tuples(data, names=["l0", "l1"])
    with pytest.raises(IndexError, match="Too many levels"):
        pidx.droplevel(level)
    with pytest.raises(IndexError, match="Too many levels"):
        gidx.droplevel(level)


def test_multiindex_droplevel_single_level_none_names():
    data = [(1, 2), (3, 4)]
    pidx = pd.MultiIndex.from_tuples(data, names=[None, None])
    gidx = cudf.MultiIndex.from_tuples(data, names=[None, None])
    result = gidx.droplevel(0)
    expected = pidx.droplevel(0)
    assert_eq(result, expected)
