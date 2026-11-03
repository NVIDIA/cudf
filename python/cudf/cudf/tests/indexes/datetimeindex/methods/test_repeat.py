# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import pandas as pd

import cudf
from cudf.testing import assert_eq


def test_index_datetime_repeat():
    gidx = cudf.date_range("2021-01-01", periods=3, freq="D")
    pidx = gidx.to_pandas()

    actual = gidx.repeat(5)
    expected = pidx.repeat(5)

    assert_eq(actual, expected)

    actual = gidx.to_frame().repeat(5)

    assert_eq(actual.index, expected)


def test_repeat_matches_pandas_drops_freq():
    # Repeating values produces duplicates that no longer respect the
    # original spacing, so pandas drops freq — cudf must match.
    pidx = pd.date_range("2021-01-01", periods=3, freq="D")
    gidx = cudf.from_pandas(pidx)

    expected = pidx.repeat(2)
    actual = gidx.repeat(2)

    assert expected.freq is None
    assert actual.freq is None
    assert_eq(actual, expected)
