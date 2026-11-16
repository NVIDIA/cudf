# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pandas as xpd
import pytest

from cudf.core._compat import PANDAS_GE_31
from cudf.pandas import LOADED

if not LOADED:
    raise ImportError("These tests must be run with cudf.pandas loaded")

pd = xpd._fsproxy_slow


@pytest.mark.parametrize("method", ["pad", "backfill", "nearest"])
def test_get_indexer_nan_target_matches_pandas(method):
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    expected = pd.Index(values).get_indexer([np.nan], method=method)
    actual = xpd.Index(values).get_indexer([np.nan], method=method)
    np.testing.assert_array_equal(expected, actual)


@pytest.mark.skipif(
    not PANDAS_GE_31, reason="pandas 3.1 MultiIndex list-key precision"
)
def test_multiindex_get_locs_checks_float32_level_key_precision():
    arrays = [["a", "a", "b"], np.array([1.1, 2.2, 1.1], dtype="float32")]
    actual = xpd.MultiIndex.from_arrays(arrays)
    with pytest.raises(KeyError):
        actual.get_locs((slice(None), [1.1]))
    with pytest.raises(KeyError):
        xpd.Series(range(2), index=actual.levels[1]).loc[[1.1]]
    np.testing.assert_array_equal(actual.get_locs((slice(None), 1.1)), [0, 2])


@pytest.mark.skipif(
    not PANDAS_GE_31,
    reason="pandas 3.1 validates every MultiIndex tuple width",
)
def test_multiindex_isin_checks_short_tuple_after_valid_tuple():
    actual = xpd.MultiIndex.from_product(
        [["foo", "bar"], ["one", "two"], ["A", "B"]]
    )
    with pytest.raises(
        ValueError, match="an element of length 1 at position 1"
    ):
        actual.isin([("foo", "one", "B"), ("bar",)])
