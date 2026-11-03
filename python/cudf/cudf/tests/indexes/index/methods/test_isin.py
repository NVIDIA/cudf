# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import re

import pandas as pd
import pytest

import cudf
from cudf.testing import assert_eq


@pytest.mark.parametrize(
    "index",
    [
        pd.Index([]),
        pd.Index(["a", "b", "c", "d", "e"]),
        pd.Index([0, None, 9]),
        pd.date_range("2019-01-01", periods=3),
    ],
)
@pytest.mark.parametrize(
    "values",
    [
        [],
        ["this", "is"],
        [0, 19, 13],
        ["2019-01-01 04:00:00", "2019-01-01 06:00:00", "2018-03-02 10:00:00"],
    ],
)
def test_isin_index(index, values):
    pidx = index
    gidx = cudf.Index(pidx)
    got = gidx.isin(values)
    expected = pidx.isin(values)

    assert_eq(got, expected)


@pytest.mark.parametrize(
    "idx, values",
    [
        (range(100, 1000, 10), [200, 600, 800]),
        ([None, "a", "3.2", "z", None, None], ["a", "z"]),
        (pd.Series(["a", "b", None], dtype="category"), [10, None]),
    ],
)
def test_index_isin_values(idx, values):
    gidx = cudf.Index(idx)
    pidx = gidx.to_pandas()

    actual = gidx.isin(values)
    expected = pidx.isin(values)

    assert_eq(expected, actual)


@pytest.mark.parametrize(
    "idx, scalar",
    [
        (range(0, -10, -2), -4),
        ([None, "a", "3.2", "z", None, None], "x"),
        (pd.Series(["a", "b", None], dtype="category"), 10),
    ],
)
def test_index_isin_scalar_values(idx, scalar):
    gidx = cudf.Index(idx)

    with pytest.raises(
        TypeError,
        match=re.escape(
            f"only list-like objects are allowed to be passed "
            f"to isin(), you passed a {type(scalar).__name__}"
        ),
    ):
        gidx.isin(scalar)
