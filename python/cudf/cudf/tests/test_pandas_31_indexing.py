# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pandas as pd
import pytest

import cudf
from cudf.testing import assert_eq


@pytest.mark.parametrize(
    "indexer",
    [
        slice(None, None, -1),
        slice(None, None, -2),
        slice(3, None, -1),
        slice(-1, 0, -2),
        slice(1, 4, -1),
    ],
)
@pytest.mark.parametrize("scalar", [False, True])
@pytest.mark.parametrize("dtype", ["int64", "datetime64[ns]"])
def test_series_iloc_reverse_assignment(indexer, scalar, dtype):
    values = np.arange(5, dtype="int64").astype(dtype)
    expected = values.copy()
    size = len(expected[indexer])
    replacement = np.arange(10, 10 + size, dtype="int64").astype(dtype)
    if scalar:
        replacement = np.array(10, dtype="int64").astype(dtype)[()]
    expected[indexer] = replacement
    actual = cudf.Series(values)
    actual.iloc[indexer] = replacement
    assert_eq(pd.Series(expected), actual)


@pytest.mark.parametrize("dtype", ["int64", "float64", "datetime64[ns]"])
@pytest.mark.parametrize(
    "rows", [[2, 0], slice(1, 4), range(2), [True, False, True, False]]
)
@pytest.mark.parametrize("box", [np.asarray, cudf.Series, cudf.Index])
def test_frame_iloc_single_column_vector_assignment(dtype, rows, box):
    values = np.arange(12, dtype="int64").reshape(4, 3).astype(dtype)
    expected = values.copy()
    replacement = np.arange(100, 100 + len(expected[rows]), dtype="int64")
    replacement = replacement.astype(dtype)
    expected[rows, 1] = replacement
    actual = cudf.DataFrame(values)
    actual.iloc[rows, [1]] = box(replacement)
    assert_eq(pd.DataFrame(expected), actual)


@pytest.mark.parametrize("rows", [np.array(0), np.array([[0], [1]])])
def test_frame_iloc_rejects_non_1d_rows_with_list_columns(rows):
    actual = cudf.DataFrame(np.zeros((4, 3)))
    with pytest.raises(ValueError, match="Cross index must be 1 dimensional"):
        actual.iloc[rows, [1]] = [1.0, 2.0]
    assert_eq(pd.DataFrame(np.zeros((4, 3))), actual)


@pytest.mark.parametrize("dtype", ["int64", "datetime64[ns]"])
@pytest.mark.parametrize("rows", [[2, 0], slice(None, None, -1)])
@pytest.mark.parametrize(
    "columns",
    [slice(None, None, -1), slice(None, None, -2), slice(1, None, -1)],
)
def test_frame_iloc_reverse_columns_assignment(dtype, rows, columns):
    values = np.arange(12, dtype="int64").reshape(4, 3).astype(dtype)
    expected = values.copy()
    width = expected[:, columns].shape[1]
    replacement = np.arange(100, 100 + width, dtype="int64").astype(dtype)
    expected[rows, columns] = replacement
    actual = cudf.DataFrame(values)
    actual.iloc[rows, columns] = replacement
    assert_eq(pd.DataFrame(expected), actual)
