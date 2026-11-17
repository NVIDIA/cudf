# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import pandas as pd
import pytest

import cudf
from cudf.testing import assert_eq


def test_take_shuffled_matches_pandas_drops_freq():
    # Pandas drops freq when take(...) shuffles values; cudf must match.
    pidx = pd.date_range("2001-01-01", periods=11, freq="D")
    gidx = cudf.from_pandas(pidx)

    perm = [4, 1, 3, 10, 9, 7, 8, 5, 0, 2, 6]
    expected = pidx.take(perm)
    actual = gidx.take(perm)

    assert expected.freq is None
    assert actual.freq is None
    assert_eq(actual, expected)


def test_take_to_pandas_after_shuffle_matches_pandas():
    # Regression: prior to the fix, this raised
    # "Inferred frequency None ... does not conform to passed frequency D".
    pidx = pd.date_range("2001-01-01", periods=11, freq="D")
    gidx = cudf.from_pandas(pidx)

    perm = [4, 1, 3, 10, 9, 7, 8, 5, 0, 2, 6]
    expected = pidx.take(perm)
    actual = gidx.take(perm).to_pandas()

    assert actual.freq == expected.freq  # both None
    assert_eq(actual, expected)


def test_take_preserving_order_values_match_pandas():
    # The gather map preserves order — values must match pandas.
    # (cudf is more conservative about freq here; we don't compare freq.)
    pidx = pd.date_range("2001-01-01", periods=5, freq="D")
    gidx = cudf.from_pandas(pidx)

    perm = [0, 1, 2, 3, 4]
    assert_eq(
        gidx.take(perm),
        pidx.take(perm),
    )
    assert_eq(
        gidx.take(perm).values,
        pidx.take(perm).values,
    )


def test_take_on_no_freq_index_matches_pandas():
    pidx = pd.DatetimeIndex(["2020-01-01", "2020-03-15", "2020-07-04"])
    gidx = cudf.from_pandas(pidx)

    perm = [2, 0, 1]
    expected = pidx.take(perm)
    actual = gidx.take(perm)

    assert expected.freq is None
    assert actual.freq is None
    assert_eq(actual, expected)


@pytest.mark.parametrize("freq", ["D", "h", "min"])
def test_take_shuffled_matches_pandas_for_various_freqs(freq):
    pidx = pd.date_range("2001-01-01", periods=6, freq=freq)
    gidx = cudf.from_pandas(pidx)

    perm = [2, 5, 0, 1]
    expected = pidx.take(perm)
    actual = gidx.take(perm)

    assert expected.freq is None
    assert actual.freq is None
    assert_eq(actual, expected)
