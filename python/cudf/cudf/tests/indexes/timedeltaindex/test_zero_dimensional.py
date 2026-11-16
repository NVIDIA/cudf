# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import cupy as cp
import numpy as np
import pandas as pd
import pytest

import cudf
from cudf.core._compat import PANDAS_GE_31
from cudf.testing import assert_eq


@pytest.mark.skipif(
    not PANDAS_GE_31,
    reason="pandas 3.1 rejects zero-dimensional timedelta input",
)
@pytest.mark.parametrize("xp", [np, cp])
def test_timedeltaindex_rejects_zero_dimensional_array(xp):
    message = "Cannot construct TimedeltaArray from 0-dim input"
    with pytest.raises(ValueError, match=message):
        cudf.TimedeltaIndex(xp.array(1))


@pytest.mark.parametrize("xp", [np, cp])
def test_timedeltaindex_accepts_one_dimensional_array(xp):
    assert_eq(pd.TimedeltaIndex([1]), cudf.TimedeltaIndex(xp.array([1])))
