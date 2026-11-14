# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import numpy as np
import pandas as pd
import pytest

import cudf
from cudf.testing import assert_eq


def test_rangeindex_take_default_user_option(default_integer_bitwidth):
    # Test that RangeIndex is materialized into 32 bit index under user
    # configuration for take operation.
    idx = cudf.RangeIndex(0, 100)
    actual = idx.take([0, 3, 7, 62])
    expected = cudf.Index(
        [0, 3, 7, 62], dtype=f"int{default_integer_bitwidth}"
    )
    assert_eq(expected, actual)


@pytest.mark.parametrize("bounds", [(1, 5, 1), (10, -2, -3)])
@pytest.mark.parametrize(
    "indices",
    [
        np.array([]),
        np.array([2, 1], dtype="uint32"),
        [1, -4],
        [0, 0],
        [3, 0, 1],
    ],
)
def test_rangeindex_take_preserves_result_type(bounds, indices):
    expected = pd.RangeIndex(*bounds, name="positions").take(indices)
    actual = cudf.RangeIndex(*bounds, name="positions").take(indices)
    assert_eq(expected, actual, exact=True)
