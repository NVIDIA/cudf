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
    not PANDAS_GE_31, reason="pandas 3.1 empty-frame duplicated semantics"
)
def test_empty_columns_duplicated_keeps_row_index():
    source = pd.DataFrame(index=pd.Index(["a", "a"], name="rows"))
    actual = xpd.DataFrame(source)
    assert_eq(source.duplicated(), actual.duplicated()._fsproxy_slow)
    assert_eq(
        source[source.duplicated()], actual[actual.duplicated()]._fsproxy_slow
    )


def test_rename_preserves_nullable_column_metadata():
    source = pd.DataFrame(
        [[1, 2, 3]], columns=pd.Index([1.0, 2.0, 3.0], dtype="Float64")
    )
    expected = source.rename(columns={1.0: 9.0})
    actual = xpd.DataFrame(source).rename(columns={1.0: 9.0})
    pd.testing.assert_index_equal(
        expected.columns, actual._fsproxy_slow.columns, exact=True
    )


@pytest.mark.skipif(
    not PANDAS_GE_31,
    reason="pandas 3.1 rejects zero-dimensional timedelta input",
)
def test_timedeltaindex_rejects_zero_dimensional_input():
    with pytest.raises(
        ValueError, match="Cannot construct TimedeltaArray from 0-dim input"
    ):
        xpd.TimedeltaIndex(np.array(1))
