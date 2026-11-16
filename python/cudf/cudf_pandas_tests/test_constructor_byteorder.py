# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pandas as xpd
import pytest

from cudf.pandas import LOADED
from cudf.pandas.module_accelerator import disable_module_accelerator
from cudf.testing import assert_eq

if not LOADED:
    raise ImportError("These tests must be run with cudf.pandas loaded")

pd = xpd._fsproxy_slow


@pytest.mark.parametrize("box", ["dict", "matrix"])
@pytest.mark.parametrize("dtype", [None, "float64", ">f8"])
@pytest.mark.parametrize("input_dtype", [">f8", ">i4"])
def test_host_array_byteorder_survives_column_selection(
    box, dtype, input_dtype
):
    values = np.array([1, 256], dtype=input_dtype)
    data = {"a": values} if box == "dict" else values[:, None]
    with disable_module_accelerator():
        expected = pd.DataFrame(data, dtype=dtype)
    actual = xpd.DataFrame(data, dtype=dtype)
    selected = actual[actual.columns]
    with disable_module_accelerator():
        assert_eq(selected._fsproxy_slow, expected[expected.columns])
    if dtype == "float64":
        assert actual.as_gpu_object().dtypes.iloc[0] == np.dtype("float64")
