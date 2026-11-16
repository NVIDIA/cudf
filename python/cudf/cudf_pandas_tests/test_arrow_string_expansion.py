# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import re

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
    not PANDAS_GE_31, reason="pandas 3.1 Arrow string expansion"
)
@pytest.mark.parametrize("pa_type", [pa.string(), pa.large_string()])
@pytest.mark.parametrize("data", [[], [None, None]])
@pytest.mark.parametrize(
    "method", ["split", "rsplit", "partition", "rpartition"]
)
def test_arrow_expanded_string_degenerate_shapes(data, pa_type, method):
    dtype = pd.ArrowDtype(pa_type)
    expected = getattr(pd.Series(data, dtype=dtype).str, method)(
        "b", expand=True
    )
    actual = getattr(xpd.Series(data, dtype=dtype).str, method)(
        "b", expand=True
    )
    assert_eq(expected, actual._fsproxy_slow)


@pytest.mark.skipif(
    not PANDAS_GE_31, reason="pandas 3.1 Arrow string regex extraction"
)
@pytest.mark.parametrize("pattern", [r"([ab])(\d)", r"(?P<letter>[ab])(\d)"])
@pytest.mark.parametrize("flags", [0, re.IGNORECASE])
def test_arrow_extract_mixed_groups_and_flags(pattern, flags):
    dtype = pd.ArrowDtype(pa.string())
    expected = pd.Series(["A1", "b2", None, "c3"], dtype=dtype).str.extract(
        pattern, flags=flags
    )
    actual = xpd.Series(["A1", "b2", None, "c3"], dtype=dtype).str.extract(
        pattern, flags=flags
    )
    assert_eq(expected, actual._fsproxy_slow)
