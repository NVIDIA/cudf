# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import pandas as pd

import cudf
from cudf.testing import assert_eq


def test_sort_values_descending_values_match_pandas_freq_dropped():
    # Pandas keeps freq as -1*Day on a descending sort; cudf is more
    # conservative and drops it. The values must still match pandas.
    pidx = pd.date_range("2001-01-01", periods=5, freq="D")
    gidx = cudf.from_pandas(pidx)

    expected = pidx.sort_values(ascending=False)
    actual = gidx.sort_values(ascending=False)

    assert actual.freq is None
    assert_eq(actual.values, expected.values)
    assert_eq(actual, expected)


def test_sort_values_after_take_matches_pandas():
    # take() then sort_values() must not surface a stale freq, and the
    # values must agree with pandas.
    pidx = pd.date_range("2001-01-01", periods=11, freq="D")
    gidx = cudf.from_pandas(pidx)

    perm = [4, 1, 3, 10, 9, 7, 8, 5, 0, 2, 6]
    expected = pidx.take(perm).sort_values()
    actual = gidx.take(perm).sort_values()

    assert_eq(actual.values, expected.values)
    assert_eq(actual, expected)
    # Pandas re-attaches Day freq here because the values are regular again;
    # cudf is conservative and leaves it None — no stale freq either way.
    assert actual.freq is None


def test_sort_values_to_pandas_round_trip_matches_pandas_values():
    # Regression for the to_pandas() ValueError on reversed values.
    pidx = pd.date_range("2001-01-01", periods=5, freq="D")
    gidx = cudf.from_pandas(pidx)

    expected = pidx.sort_values(ascending=False)
    actual = gidx.sort_values(ascending=False)

    assert_eq(actual.values, expected.values)
    assert_eq(actual, expected)
