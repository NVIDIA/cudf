# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pandas as xpd
import pytest

from cudf.core._compat import PANDAS_GE_31
from cudf.pandas import LOADED
from cudf.testing import assert_eq

if not LOADED:
    raise ImportError("These tests must be run with cudf.pandas loaded")

pd = xpd._fsproxy_slow


@pytest.mark.skipif(
    not PANDAS_GE_31, reason="pandas 3.1 empty dict resampling columns"
)
@pytest.mark.parametrize("func", [[], {}])
def test_series_resample_empty_aggregation_column_metadata(func):
    index = pd.date_range("2000", periods=3)
    expected = (
        pd.Series([1, 2, 3], index=index, name="a").resample("D").agg(func)
    )
    actual = (
        xpd.Series([1, 2, 3], index=index, name="a").resample("D").agg(func)
    )
    assert_eq(expected, actual._fsproxy_slow)


@pytest.mark.skipif(
    not PANDAS_GE_31, reason="pandas 3.1 missing MultiIndex merge keys"
)
@pytest.mark.parametrize("index_side", ["left", "right"])
def test_merge_matches_missing_multiindex_keys(index_side):
    def frames(module):
        data = {
            "key1": [1.0, np.nan, 3.0],
            "key2": ["a", "b", "c"],
            "val": [10, 20, 30],
        }
        index = module.MultiIndex.from_arrays(
            [data["key1"], data["key2"]], names=["key1", "key2"]
        )
        return module.DataFrame(data), module.DataFrame(
            {"data": [100, 200, 300]}, index=index
        )

    expected_columns, expected_index = frames(pd)
    actual_columns, actual_index = frames(xpd)
    if index_side == "right":
        args = {"left_on": ["key1", "key2"], "right_index": True}
        expected = expected_columns.merge(expected_index, **args)
        actual = actual_columns.merge(actual_index, **args)
    else:
        args = {"left_index": True, "right_on": ["key1", "key2"]}
        expected = expected_index.merge(expected_columns, **args)
        actual = actual_index.merge(actual_columns, **args)
    assert_eq(expected, actual._fsproxy_slow)
