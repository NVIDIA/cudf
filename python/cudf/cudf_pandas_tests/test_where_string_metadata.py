# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from decimal import Decimal

import numpy as np
import pandas as xpd
import pyarrow as pa
import pytest

from cudf.core._compat import PANDAS_GE_31
from cudf.pandas import LOADED
from cudf.testing import assert_eq

if not LOADED:
    raise ImportError("These tests must be run with cudf.pandas loaded")

pd = xpd._fsproxy_slow


@pytest.mark.skipif(
    not PANDAS_GE_31, reason="pandas 3.1 listlike EA replacements"
)
@pytest.mark.parametrize("storage", ["python", "pyarrow"])
@pytest.mark.parametrize("na_value", [pd.NA, np.nan])
@pytest.mark.parametrize(
    "other", [["z", "y", "x"], ["z"], ("x", "y", "z"), {"zz": 1}]
)
def test_where_string_replacement_semantics(storage, na_value, other):
    dtype = pd.StringDtype(storage=storage, na_value=na_value)
    expected = pd.Series(["a", "b", "c"], dtype=dtype).where(
        [True, False, True], other
    )
    actual = xpd.Series(["a", "b", "c"], dtype=dtype).where(
        [True, False, True], other
    )
    assert_eq(expected, actual._fsproxy_slow)


@pytest.mark.skipif(
    not PANDAS_GE_31, reason="pandas 3.1 retains EA metadata on expansion"
)
@pytest.mark.parametrize(
    "dtype",
    [
        pd.StringDtype(storage="python"),
        pd.StringDtype(storage="python", na_value=np.nan),
        pd.ArrowDtype(pa.string()),
    ],
)
def test_loc_expansion_retains_string_metadata(dtype):
    expected = pd.Series(["a", "b", "c"], dtype=dtype)
    actual = xpd.Series(["a", "b", "c"], dtype=dtype)
    expected.loc[3] = "d"
    actual.loc[3] = "d"
    assert_eq(expected, actual._fsproxy_slow)
    expected_frame = pd.DataFrame({"a": expected.iloc[:3]})
    actual_frame = xpd.DataFrame({"a": xpd.Series(expected.iloc[:3])})
    expected_frame.loc[3] = ["d"]
    actual_frame.loc[3] = ["d"]
    assert_eq(expected_frame, actual_frame._fsproxy_slow)


@pytest.mark.skipif(
    not PANDAS_GE_31, reason="pandas 3.1 listlike EA replacements"
)
@pytest.mark.parametrize(
    "other", [[Decimal("1"), Decimal("2"), Decimal("3")], [Decimal("1")]]
)
def test_arrow_decimal_where_keeps_precision(other):
    dtype = pd.ArrowDtype(pa.decimal128(7, 3))
    expected = pd.Series(
        [Decimal("1"), None, Decimal("3")], dtype=dtype
    ).where([True, False, True], other)
    actual = xpd.Series([Decimal("1"), None, Decimal("3")], dtype=dtype).where(
        [True, False, True], other
    )
    assert_eq(expected, actual._fsproxy_slow)
