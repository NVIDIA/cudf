# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pandas as pd
import pyarrow as pa
import pytest

import cudf
from cudf.testing import assert_eq


@pytest.mark.parametrize(
    "dtype",
    [
        pd.Int32Dtype(),
        pd.Int64Dtype(),
        pd.Float32Dtype(),
        pd.Float64Dtype(),
        pd.ArrowDtype(pa.int64()),
        pd.ArrowDtype(pa.float64()),
    ],
)
@pytest.mark.parametrize("shape", ["empty", "all_null", "mixed"])
@pytest.mark.parametrize("decimals", [-1, 0, 1])
def test_round_nullable_metadata(dtype, shape, decimals):
    if shape == "empty":
        values = []
    elif shape == "all_null":
        values = [None, None]
    elif dtype.kind in "iu":
        values = [12, None, 28]
    else:
        values = [1.25, None, 2.75]
    expected = pd.Series(values, dtype=dtype).round(decimals)
    result = cudf.from_pandas(pd.Series(values, dtype=dtype)).round(decimals)

    assert result.dtype == expected.dtype
    assert_eq(expected, result)
