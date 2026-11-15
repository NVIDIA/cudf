# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pandas as pd
import pytest

from cudf.core._compat import PANDAS_GE_31
from cudf.pandas._wrappers.numpy import ndarray
from cudf.pandas.fast_slow_proxy import _slow_arg
from cudf.pandas.module_accelerator import disable_module_accelerator


@pytest.mark.parametrize("operation", ["setitem", "fill", "ufunc_out"])
def test_readonly_numpy_array_rejects_mutation(operation):
    data = np.arange(3)
    data.flags.writeable = False
    result = ndarray._fsproxy_wrap(data, None)

    with pytest.raises(ValueError, match="read-only"):
        if operation == "setitem":
            result[0] = 99
        elif operation == "fill":
            result.fill(99)
        else:
            np.add(result, 1, out=result)

    np.testing.assert_array_equal(data, np.arange(3))


def test_readonly_numpy_copy_is_writable():
    data = np.arange(3)
    data.flags.writeable = False
    result = ndarray._fsproxy_wrap(data, None).copy()
    result[0] = 99
    np.testing.assert_array_equal(_slow_arg(result), [99, 1, 2])
    np.testing.assert_array_equal(data, np.arange(3))


@pytest.mark.skipif(
    not PANDAS_GE_31, reason="Index arrays are read-only in pandas 3.1"
)
def test_sparse_index_astype_readonly():
    index = pd.Index(pd.arrays.SparseArray(np.array([1, 2, 3]), fill_value=0))
    expected = _slow_arg(index).copy(deep=True)
    result = index.array.astype(np.dtype("int64"), copy=False)
    assert not result.flags.writeable
    with pytest.raises(ValueError, match="read-only"):
        result[0] = 99
    with disable_module_accelerator():
        pd.testing.assert_index_equal(_slow_arg(index), expected)
