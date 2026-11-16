# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pandas as pd
import pytest

import cudf
from cudf.testing import assert_eq


@pytest.mark.parametrize("box", ["dict", "matrix"])
@pytest.mark.parametrize("dtype", [None, "float64"])
@pytest.mark.parametrize("compatible", [False, True])
def test_non_native_host_array_constructor(box, dtype, compatible):
    values = np.array([1.0, 256.0], dtype=">f8")
    data = {"a": values} if box == "dict" else values[:, None]
    with cudf.option_context("mode.pandas_compatible", compatible):
        if compatible and dtype is None:
            with pytest.raises(NotImplementedError, match="byte-order"):
                cudf.DataFrame(data)
        else:
            actual = cudf.DataFrame(data, dtype=dtype)
            native_data = (
                {"a": values.astype("float64")}
                if box == "dict"
                else values.astype("float64")[:, None]
            )
            assert_eq(actual, pd.DataFrame(native_data, dtype=dtype))
