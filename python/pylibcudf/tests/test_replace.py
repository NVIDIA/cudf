# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pyarrow as pa
import pytest

import pylibcudf as plc


@pytest.mark.parametrize("inplace", [False, True])
def test_normalize_nans_and_zeros(inplace):
    data = pa.array([-0.0, 1.0, -np.nan, 0.0], type=pa.float64())
    col = plc.Column.from_arrow(data)
    result = plc.replace.normalize_nans_and_zeros(col, inplace=inplace)
    if inplace:
        assert result is None
        result = col
    got = result.to_arrow().to_numpy()
    assert not np.signbit(got).any()
    np.testing.assert_array_equal(got, [0.0, 1.0, np.nan, 0.0])
