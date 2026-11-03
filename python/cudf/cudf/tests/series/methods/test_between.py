# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pandas as pd
import pytest

import cudf
from cudf.testing import assert_eq


@pytest.fixture(params=["both", "neither", "left", "right"])
def inclusive(request):
    return request.param


@pytest.mark.parametrize(
    "data,left,right",
    [
        ([0, 1, 2, 3, 4, 5, 10], 0, 5),
        ([0, 1, 2, 3, 4, 5, 10], 10, 1),
        ([0, 1, 2, 3, 4, 5], [0, 10, 11] * 2, [1, 2, 5] * 2),
        (["a", "few", "set", "of", "strings", "xyz", "abc"], "banana", "few"),
        (["a", "few", "set", "of", "strings", "xyz", "abc"], "phone", "hello"),
        (
            ["a", "few", "set", "of", "strings", "xyz", "abc"],
            ["a", "hello", "rapids", "ai", "world", "chars", "strs"],
            ["yes", "no", "hi", "bye", "test", "pass", "fail"],
        ),
        ([0, 1, 2, np.nan, 4, np.nan, 10], 10, 1),
    ],
)
def test_series_between(data, left, right, inclusive):
    ps = pd.Series(data)
    gs = cudf.from_pandas(ps, nan_as_null=False)

    expected = ps.between(left, right, inclusive=inclusive)
    actual = gs.between(left, right, inclusive=inclusive)

    assert_eq(expected, actual)


@pytest.mark.parametrize(
    "data,dtype,left,right",
    [
        ([0, 1, 2, None, 4, 5, 10], "Int64", 0, 5),
        ([0, 1, 2, 3, None, 5, 10], "Int64", 10, 1),
        ([None, 1, 2, 3, 4, None], "Int64", [0, 10, 11] * 2, [1, 2, 5] * 2),
        (
            ["a", "few", "set", None, "strings", "xyz", "abc"],
            "string",
            ["a", "hello", "rapids", "ai", "world", "chars", "strs"],
            ["yes", "no", "hi", "bye", "test", "pass", "fail"],
        ),
    ],
)
def test_series_between_with_null(data, dtype, left, right, inclusive):
    gs = cudf.Series(data, dtype=dtype)
    ps = gs.to_pandas()

    expected = ps.between(left, right, inclusive=inclusive)
    actual = gs.between(left, right, inclusive=inclusive)

    assert_eq(expected, actual)
