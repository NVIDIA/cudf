# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pandas as xpd
import pytest

from cudf.pandas import LOADED
from cudf.testing import assert_eq

if not LOADED:
    raise ImportError("These tests must be run with cudf.pandas loaded")

pd = xpd._fsproxy_slow


@pytest.mark.parametrize(
    "values,kwargs",
    [
        (np.array([float(2**63)]), {"unit": "ns"}),
        (np.array([np.iinfo(np.uint64).max], dtype="uint64"), {"unit": "ns"}),
        (["1677-09-21 00:12:43.145224192"], {}),
        (
            ["21/09/1677 00:12:43.145224192"],
            {"format": "%d/%m/%Y %H:%M:%S.%f"},
        ),
    ],
)
@pytest.mark.parametrize("box", ["list", "Series", "Index"])
def test_to_datetime_storage_boundaries_match_pandas(values, kwargs, box):
    expected_values = values if box == "list" else getattr(pd, box)(values)
    actual_values = values if box == "list" else getattr(xpd, box)(values)
    try:
        expected = pd.to_datetime(expected_values, **kwargs)
    except pd.errors.OutOfBoundsDatetime:
        with pytest.raises(pd.errors.OutOfBoundsDatetime):
            xpd.to_datetime(actual_values, **kwargs)
    else:
        actual = xpd.to_datetime(actual_values, **kwargs)
        assert_eq(expected, actual._fsproxy_slow)
