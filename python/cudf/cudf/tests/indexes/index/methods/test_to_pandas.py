# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import datetime

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

import cudf
from cudf.testing import assert_eq


@pytest.mark.parametrize(
    "data,expected_dtype",
    [
        ([10, 11, 12], pd.Int64Dtype()),
        ([0.1, 10.2, 12.3], pd.Float64Dtype()),
        (["abc", None, "def"], pd.StringDtype(na_value=np.nan)),
    ],
)
def test_index_to_pandas_nullable(data, expected_dtype):
    gi = cudf.Index(data)
    pi = gi.to_pandas(nullable=True)
    expected = pd.Index(data, dtype=expected_dtype)

    assert_eq(pi, expected)


@pytest.mark.parametrize(
    "data",
    [
        range(1),
        np.array([1, 2], dtype="datetime64[ns]"),
        np.array([1, 2], dtype="timedelta64[ns]"),
    ],
)
def test_index_to_pandas_nullable_notimplemented(data):
    idx = cudf.Index(data)
    with pytest.raises(NotImplementedError):
        idx.to_pandas(nullable=True)


@pytest.mark.parametrize(
    "scalar",
    [
        1,
        1.0,
        "a",
        datetime.datetime(2020, 1, 1),
        datetime.timedelta(1),
        pd.Interval(1, 2),
    ],
)
def test_index_to_pandas_arrow_type_nullable_raises(scalar):
    data = [scalar, None]
    idx = cudf.Index(data)
    with pytest.raises(ValueError):
        idx.to_pandas(nullable=True, arrow_type=True)


@pytest.mark.parametrize(
    "scalar",
    [
        1,
        1.0,
        "a",
        datetime.datetime(2020, 1, 1),
        datetime.timedelta(1),
    ],
)
def test_index_to_pandas_arrow_type(scalar):
    pa_array = pa.array([scalar, None])
    idx = cudf.Index(pa_array)
    result = idx.to_pandas(arrow_type=True)
    expected = pd.Index(pd.arrays.ArrowExtensionArray(pa_array))
    pd.testing.assert_index_equal(result, expected)
