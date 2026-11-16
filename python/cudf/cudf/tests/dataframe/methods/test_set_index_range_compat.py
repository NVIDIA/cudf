# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

import cudf
from cudf.testing import assert_eq

_VALUES = [[0, 1, 2], [5, 7, 9], [5, 3, 1], [0, 2, 7], [3, 3, 3], [5], []]


@pytest.mark.parametrize("values", _VALUES)
@pytest.mark.parametrize(
    "dtype",
    ["int32", "int64", "uint8", "float32", "Int64", pd.ArrowDtype(pa.int32())],
)
@pytest.mark.parametrize("explicit_index", [False, True])
def test_set_index_range_in_compatible_mode(values, dtype, explicit_index):
    expected = pd.DataFrame(
        {"key": pd.Series(values, dtype=dtype), "value": range(len(values))}
    )
    actual = cudf.from_pandas(expected)
    expected_key = (
        pd.Index(expected["key"], name="key") if explicit_index else "key"
    )
    actual_key = cudf.from_pandas(expected_key) if explicit_index else "key"
    with cudf.option_context("mode.pandas_compatible", True):
        result = actual.set_index(actual_key)
    expected = expected.set_index(expected_key)
    assert_eq(expected, result)
    pd.testing.assert_index_equal(
        expected.index, result.index.to_pandas(), exact=True
    )


@pytest.mark.parametrize("values", _VALUES)
def test_set_index_keeps_native_integer_dtype(values):
    frame = cudf.DataFrame({"key": cudf.Series(values, dtype="int32")})
    with cudf.option_context("mode.pandas_compatible", False):
        result = frame.set_index("key")
    assert result.index.dtype == np.dtype("int32")
    assert result.index.to_pandas().tolist() == values


def test_set_index_append_keeps_multiindex():
    expected = pd.DataFrame(
        {"key": np.arange(3, dtype="int32"), "value": [4, 5, 6]}
    )
    actual = cudf.from_pandas(expected)
    with cudf.option_context("mode.pandas_compatible", True):
        result = actual.set_index("key", append=True)
    assert isinstance(result.index, cudf.MultiIndex)
    assert_eq(expected.set_index("key", append=True), result)


@pytest.mark.parametrize("values", [[], [5], [5, 7, 9]])
def test_set_index_preserves_explicit_range(values):
    expected = pd.DataFrame({"value": values})
    index = pd.RangeIndex(5, 5 + 2 * len(values), 2, name="key")
    with cudf.option_context("mode.pandas_compatible", True):
        result = cudf.from_pandas(expected).set_index(cudf.from_pandas(index))
    pd.testing.assert_index_equal(
        expected.set_index(index).index, result.index.to_pandas(), exact=True
    )


@pytest.mark.parametrize(
    "values",
    [
        [
            np.iinfo("int64").max - 2,
            np.iinfo("int64").max - 1,
            np.iinfo("int64").max,
        ],
        [
            np.iinfo("int64").min + 2,
            np.iinfo("int64").min + 1,
            np.iinfo("int64").min,
        ],
        [np.iinfo("int64").min, np.iinfo("int64").max],
        [np.iinfo("int64").max, np.iinfo("int64").min],
        [0, np.iinfo("int64").max, -2, np.iinfo("int64").max - 2, -4],
    ],
)
def test_set_index_range_extreme_values(values):
    frame = cudf.DataFrame({"key": cudf.Series(values, dtype="int64")})
    with cudf.option_context("mode.pandas_compatible", True):
        result = frame.set_index("key")
    assert result.index.dtype == np.dtype("int64")
    assert result.index.to_pandas().tolist() == values
