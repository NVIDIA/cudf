# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import re

import numpy as np
import pandas as xpd
import pytest

from cudf.pandas import LOADED
from cudf.testing import assert_eq

if not LOADED:
    raise ImportError("These tests must be run with cudf.pandas loaded")

pd = xpd._fsproxy_slow


@pytest.mark.parametrize("storage", ["python", "pyarrow"])
@pytest.mark.parametrize("na_value", [pd.NA, np.nan])
@pytest.mark.parametrize(
    "pattern",
    [
        r"\u20AC",
        r"\U000020AC",
        r"\N{EURO SIGN}",
        r"[\u0e00-\u0e7f]",
        r"\\u20AC",
    ],
)
@pytest.mark.parametrize("compiled", [False, True])
def test_unicode_regex_escape_parity(storage, na_value, pattern, compiled):
    dtype = pd.StringDtype(storage=storage, na_value=na_value)
    values = ["a", "€", "ก", None, r"\u20AC"]
    expected = pd.Series(values, dtype=dtype)
    actual = xpd.Series(values, dtype=dtype)
    if compiled:
        pattern = re.compile(pattern)
    try:
        expected_result = expected.str.contains(pattern)
    except Exception as error:
        with pytest.raises(type(error), match=re.escape(str(error))):
            actual.str.contains(pattern)
    else:
        assert_eq(actual.str.contains(pattern)._fsproxy_slow, expected_result)
