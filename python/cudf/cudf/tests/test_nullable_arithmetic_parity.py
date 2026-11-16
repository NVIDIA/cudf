# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import operator

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

import cudf
from cudf.core._compat import PANDAS_GE_31
from cudf.testing import assert_eq


@pytest.mark.parametrize(
    "op",
    [
        operator.eq,
        operator.ne,
        operator.lt,
        operator.le,
        operator.gt,
        operator.ge,
    ],
)
@pytest.mark.parametrize("left_dtype", ["int64", "float64"])
@pytest.mark.parametrize(
    "dtype",
    [
        "Int8",
        "UInt64",
        "Float32",
        pd.ArrowDtype(pa.int32()),
        pd.ArrowDtype(pa.float64()),
    ],
)
def test_numpy_series_comparison_with_nullable_series(op, left_dtype, dtype):
    left = pd.Series([1, 2, 3], dtype=left_dtype)
    if left_dtype == "float64":
        left.iloc[0] = np.nan
    right = pd.Series([1, 2, None], dtype=dtype)
    expected = op(left, right)
    with cudf.option_context("mode.pandas_compatible", True):
        actual = op(cudf.from_pandas(left), cudf.from_pandas(right))
    assert_eq(actual, expected)


@pytest.mark.parametrize(
    "dtype", ["bool", "boolean", pd.ArrowDtype(pa.bool_())]
)
@pytest.mark.parametrize(
    "data", [[[True, None], [False, None]], [[None, None], [None, None]]]
)
def test_boolean_frame_constructor_preserves_nullable_mask(dtype, data):
    expected = pd.DataFrame(data, dtype=dtype)
    with cudf.option_context("mode.pandas_compatible", True):
        actual = cudf.DataFrame(data, dtype=dtype)
    assert_eq(actual, expected)


@pytest.mark.parametrize("source_dtype", ["str", "string"])
@pytest.mark.parametrize("dtype", ["boolean", pd.ArrowDtype(pa.bool_())])
@pytest.mark.parametrize("start,stop", [(0, 6), (1, 5), (2, 6)])
def test_nullable_boolean_cast_preserves_sliced_string_mask(
    source_dtype, dtype, start, stop
):
    with cudf.option_context("mode.pandas_compatible", True):
        actual = (
            cudf.Series(["", None, "x", "", None, "y"], dtype=source_dtype)
            .iloc[start:stop]
            .astype(dtype)
        )
    expected = pd.Series(
        [False, None, True, False, None, True], dtype=dtype
    ).iloc[start:stop]
    assert_eq(actual, expected)


@pytest.mark.skipif(
    not PANDAS_GE_31, reason="pandas 3.1 adds Series fill_value broadcasting"
)
@pytest.mark.parametrize("op", ["add", "sub", "mul"])
@pytest.mark.parametrize("mismatched", [False, True])
def test_frame_series_fill_value_preserves_both_missing(op, mismatched):
    frame = pd.DataFrame({"a": [1.0, np.nan], "b": [np.nan, 4.0]})
    other = pd.Series(
        [np.nan, 2.0], index=["a", "c"] if mismatched else ["a", "b"]
    )
    expected = getattr(frame, op)(other, fill_value=5)
    with cudf.option_context("mode.pandas_compatible", True):
        actual = getattr(cudf.from_pandas(frame), op)(
            cudf.from_pandas(other), fill_value=5
        )
    assert_eq(actual, expected)
