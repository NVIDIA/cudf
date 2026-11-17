# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import pandas as pd
import pytest

import cudf
from cudf.testing import assert_eq


def test_series_describe_numeric(numeric_types_as_str):
    ps = pd.Series([0, 1, 2, 3, 1, 2, 3], dtype=numeric_types_as_str)
    gs = cudf.from_pandas(ps)
    actual = gs.describe()
    expected = ps.describe()

    assert_eq(expected, actual, check_dtype=True)


def test_series_describe_temporal(temporal_types_as_str):
    gs = cudf.Series([0, 1, 2, 3, 1, 2, 3], dtype=temporal_types_as_str)
    ps = gs.to_pandas()

    expected = ps.describe()
    actual = gs.describe()

    assert_eq(expected.astype("str"), actual)


@pytest.mark.parametrize(
    "ps",
    [
        pd.Series(["a", "b", "c", "d", "e", "a"]),
        pd.Series([True, False, True, True, False]),
        pd.Series([], dtype="str"),
        pd.Series(["a", "b", "c", "a"], dtype="category"),
        pd.Series(["d", "e", "f"], dtype="category"),
        pd.Series(pd.Categorical(["d", "e", "f"], categories=["f", "e", "d"])),
        pd.Series(
            pd.Categorical(
                ["d", "e", "f"], categories=["f", "e", "d"], ordered=True
            )
        ),
    ],
)
def test_series_describe_other_types(ps):
    gs = cudf.from_pandas(ps)

    expected = ps.describe()
    actual = gs.describe()

    if len(ps) == 0:
        assert_eq(expected.fillna("a").astype("str"), actual.fillna("a"))
    else:
        assert_eq(expected.astype("str"), actual)
