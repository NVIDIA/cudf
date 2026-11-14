# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import re

import numpy as np
import pandas as pd
import pytest

from cudf.pandas.fast_slow_proxy import _slow_arg
from cudf.pandas.module_accelerator import disable_module_accelerator


@pytest.mark.parametrize("ufunc", [np.add, np.maximum])
@pytest.mark.parametrize("method", ["accumulate", "reduce"])
def test_sparse_ufunc_protocol(ufunc, method):
    data = [0.0, 1.0, 0.0, 3.0]
    try:
        with disable_module_accelerator():
            expected = getattr(ufunc, method)(
                pd.arrays.SparseArray(data, fill_value=0.0)
            )
    except TypeError as error:
        # pandas 3.0 rejects accumulate on sparse arrays.
        with pytest.raises(TypeError, match=re.escape(str(error))):
            getattr(ufunc, method)(pd.arrays.SparseArray(data, fill_value=0.0))
        return

    result = getattr(ufunc, method)(
        pd.arrays.SparseArray(data, fill_value=0.0)
    )
    result = _slow_arg(result)

    with disable_module_accelerator():
        assert type(result) is type(expected)
        if isinstance(expected, pd.arrays.SparseArray):
            pd.testing.assert_extension_array_equal(result, expected)
        else:
            np.testing.assert_equal(result, expected)


@pytest.mark.parametrize("kind", ["M8", "m8"])
@pytest.mark.parametrize("unit", ["s", "ms", "us", "ns"])
def test_sparse_datetimelike_array_protocol(kind, unit):
    with disable_module_accelerator():
        values = np.array([1, 2, 3], dtype="i8").astype(f"{kind}[{unit}]")
        values[1] = "NaT"
        expected = np.asarray(pd.arrays.SparseArray(values), dtype=object)

    result = np.asarray(pd.arrays.SparseArray(values), dtype=object)

    with disable_module_accelerator():
        pd._testing.assert_numpy_array_equal(_slow_arg(result), expected)
