# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pandas as xpd
import pandas._testing as tm
import pytest

pd = xpd._fsproxy_slow


@pytest.fixture(params=["complex128", "float64", "object"])
def data(request):
    if request.param == "object":
        result = np.empty(3, dtype=object)
        for position in range(3):
            result[position] = (position,)
        return result
    return np.arange(1, 4, dtype=request.param)


def test_numpy_extension_array_protocol(data):
    expected = pd.arrays.NumpyExtensionArray(data)
    result = xpd.arrays.NumpyExtensionArray(data)
    assert isinstance(result, pd.api.extensions.ExtensionArray)
    assert result.dtype.numpy_dtype == expected.dtype.numpy_dtype
    assert np.asarray(result).shape == data.shape
    np.testing.assert_array_equal(np.asarray(result), np.asarray(expected))


def test_numpy_extension_series_constructor(data):
    expected = pd.Series(pd.arrays.NumpyExtensionArray(data))
    result = xpd.Series(xpd.arrays.NumpyExtensionArray(data))
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("dtype", ["complex128", "float64"])
@pytest.mark.parametrize("method", ["__call__", "reduce", "accumulate"])
def test_numpy_extension_ufunc(dtype, method):
    data = np.arange(1, 4, dtype=dtype)
    expected_array = pd.arrays.NumpyExtensionArray(data)
    result_array = xpd.arrays.NumpyExtensionArray(data)
    function = getattr(np.add, method)
    if method == "__call__":
        expected = function(data, expected_array)
        result = function(data, result_array)
    else:
        expected = function(expected_array)
        result = function(result_array)
    if isinstance(expected, pd.api.extensions.ExtensionArray):
        assert isinstance(result, pd.api.extensions.ExtensionArray)
    np.testing.assert_array_equal(np.asarray(result), np.asarray(expected))


def test_numpy_extension_readonly_array():
    data = np.arange(3)
    data.flags.writeable = False
    expected = np.asarray(pd.arrays.NumpyExtensionArray(data))
    result = np.asarray(xpd.arrays.NumpyExtensionArray(data))
    assert result.flags.writeable == expected.flags.writeable
    np.testing.assert_array_equal(result, expected)
