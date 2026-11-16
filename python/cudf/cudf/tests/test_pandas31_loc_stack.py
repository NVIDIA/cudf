# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections import namedtuple

import numpy as np
import pandas as pd
import pytest

import cudf
from cudf.core._compat import PANDAS_GE_31
from cudf.testing import assert_eq


@pytest.mark.skipif(not PANDAS_GE_31, reason="New pandas 3.1 row expansion")
@pytest.mark.parametrize("index", [None, [], [1, 2, 3], pd.RangeIndex(3)])
@pytest.mark.parametrize("two_axes", [False, True])
def test_loc_empty_columns_new_row(index, two_axes):
    expected = pd.DataFrame(index=index)
    actual = cudf.from_pandas(expected)
    key = (4, slice(None)) if two_axes else 4
    expected.loc[key] = 3
    actual.loc[key] = 3
    assert_eq(expected, actual)


@pytest.mark.parametrize("ncols", [0, 1, 2])
@pytest.mark.parametrize("named", [False, True])
@pytest.mark.parametrize(
    "rows", [("A", ["a", "b"]), (["A", "B"], "a"), (slice(None), "a")]
)
def test_loc_multiindex_scalar_level_with_list(rows, named, ncols):
    index = pd.MultiIndex.from_product(
        [["A", "B"], ["a", "b", "c"]], names=["first", "second"]
    )
    expected = pd.DataFrame(
        np.arange(6 * ncols).reshape(6, ncols), index=index
    )
    if named:
        rows = namedtuple("Indexer", index.names)(*rows)
    with cudf.option_context("mode.pandas_compatible", True):
        actual = cudf.from_pandas(expected)
        assert_eq(expected.loc[rows, :], actual.loc[rows, :])


@pytest.mark.skipif(not PANDAS_GE_31, reason="New pandas 3.1 stack validation")
@pytest.mark.parametrize("level", [[0, 0], ["l0", "l0"], [2, -1], [1, -2]])
def test_stack_duplicate_levels(level):
    columns = pd.MultiIndex.from_product(
        [["A", "B"], ["c", "d"], ["e", "f"]], names=["l0", "l1", "l2"]
    )
    expected = pd.DataFrame(np.arange(8).reshape(1, 8), columns=columns)
    actual = cudf.from_pandas(expected)
    for frame in (expected, actual):
        with pytest.raises(
            ValueError, match="level should not contain duplicate"
        ):
            frame.stack(level)


@pytest.mark.parametrize("level", [[0, 1], ["l0", "l2"], [1, -1]])
def test_stack_distinct_levels(level):
    columns = pd.MultiIndex.from_product(
        [["A", "B"], ["c", "d"], ["e", "f"]], names=["l0", "l1", "l2"]
    )
    expected = pd.DataFrame(np.arange(8).reshape(1, 8), columns=columns)
    actual = cudf.from_pandas(expected)
    assert_eq(expected.stack(level), actual.stack(level))
