# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pandas as pd
import pytest

import cudf
from cudf.testing import assert_eq


@pytest.mark.parametrize(
    "rng",
    [
        range(1, 20, 3),
        range(20, 35, 3),
        range(35, 77, 3),
        range(77, 110, 3),
    ],
)
@pytest.mark.parametrize("method", [None, "ffill", "bfill", "nearest"])
@pytest.mark.parametrize("tolerance", [None, 0, 1, 13, 20])
def test_get_indexer_rangeindex(rng, method, tolerance):
    key = list(rng)
    pi = pd.RangeIndex(3, 100, 4)
    gi = cudf.from_pandas(pi)

    expected = pi.get_indexer(
        key, method=method, tolerance=None if method is None else tolerance
    )
    got = gi.get_indexer(
        key, method=method, tolerance=None if method is None else tolerance
    )

    assert_eq(expected, got)

    with cudf.option_context("mode.pandas_compatible", True):
        got = gi.get_indexer(
            key, method=method, tolerance=None if method is None else tolerance
        )
    assert_eq(expected, got, check_dtype=True)


@pytest.mark.parametrize(
    "bounds, target, expected",
    [
        ((10, 13, 1), [-(2**63), 10, 11, 12, 13], [-1, 0, 1, 2, -1]),
        (
            (-(2**62) - 1, 2**63 - 1, 2**62),
            [2**62 - 1, -1, 5, -(2**62) - 1],
            [2, 1, -1, 0],
        ),
        (
            (2**62 - 1, -(2**62) - 2, -(2**62)),
            [-(2**63), -1, 2**62 - 1, -(2**62) - 1],
            [-1, 1, 0, 2],
        ),
        (
            (0, 3, 1),
            np.array([2**63, 0, 1, 5], dtype="uint64"),
            [-1, 0, 1, -1],
        ),
        (
            (-2, 3, 2),
            np.array([0, 2, 2**63], dtype="uint64"),
            [1, 2, -1],
        ),
        ((3, 3, 1), [3], [-1]),
        ((5, 100, 100), [5, 6, 99, 100], [0, -1, -1, -1]),
        ((5, -100, -200), [5, 6, 99, 100], [0, -1, -1, -1]),
        ((1, 10, 2), [1.0, 3.5, 9.0, np.nan], [0, -1, 4, -1]),
    ],
)
def test_get_indexer_extreme_and_sparse_ranges(bounds, target, expected):
    # Explicit positions also cover pandas 3.0's signed-offset overflow bug.
    index = cudf.RangeIndex(*bounds)
    actual = index.get_indexer(target)
    assert_eq(np.array(expected, dtype="int64"), actual)
