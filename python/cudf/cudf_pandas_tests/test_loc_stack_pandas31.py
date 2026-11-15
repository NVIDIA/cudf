# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections import namedtuple

import numpy as np
import pandas as xpd
import pytest

from cudf.core._compat import PANDAS_GE_31
from cudf.pandas import LOADED
from cudf.testing import assert_eq

if not LOADED:
    raise ImportError("These tests must be run with cudf.pandas loaded")

pd = xpd._fsproxy_slow


@pytest.mark.skipif(not PANDAS_GE_31, reason="New pandas 3.1 row expansion")
@pytest.mark.parametrize("index", [None, [], [1, 2, 3], pd.RangeIndex(3)])
def test_loc_empty_columns_new_row_stays_on_gpu(index, monkeypatch):
    expected = pd.DataFrame(index=index)
    actual = xpd.DataFrame(index=index)
    expected.loc[4] = 3
    with monkeypatch.context() as context:
        context.setenv("CUDF_PANDAS_FAIL_ON_FALLBACK", "1")
        actual.loc[4] = 3
    assert_eq(expected, actual._fsproxy_slow)


@pytest.mark.parametrize("ncols", [0, 1])
def test_loc_namedtuple_multiindex_stays_on_gpu(ncols, monkeypatch):
    index = pd.MultiIndex.from_product(
        [["A", "B"], ["a", "b", "c"]], names=["first", "second"]
    )
    expected = pd.DataFrame(
        np.arange(6 * ncols).reshape(6, ncols), index=index
    )
    actual = xpd.DataFrame(expected)
    rows = namedtuple("Indexer", index.names)(first="A", second=["a", "b"])
    expected = expected.loc[rows, :]
    with monkeypatch.context() as context:
        context.setenv("CUDF_PANDAS_FAIL_ON_FALLBACK", "1")
        actual = actual.loc[rows, :]
    assert_eq(expected, actual._fsproxy_slow)
