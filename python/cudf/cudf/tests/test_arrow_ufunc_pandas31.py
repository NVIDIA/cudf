# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

import cudf
from cudf.core._compat import PANDAS_GE_31
from cudf.testing import assert_eq


@pytest.mark.parametrize("ufunc", [np.isnan, np.isfinite, np.sqrt])
def test_dataframe_ufunc_keeps_each_column_mask(ufunc):
    expected_source = pd.DataFrame(
        {"a": [1, None, 3], "b": [None, 2, 3]}, dtype="Int64"
    )
    with cudf.option_context("mode.pandas_compatible", True):
        actual = ufunc(cudf.from_pandas(expected_source))
    expected = ufunc(expected_source)
    if isinstance(expected, tuple):
        for left, right in zip(expected, actual, strict=True):
            assert_eq(left, right)
    else:
        assert_eq(expected, actual)


@pytest.mark.skipif(
    not PANDAS_GE_31, reason="pandas 3.1 Arrow ufunc semantics"
)
@pytest.mark.parametrize("ufunc", [np.isnan, np.isfinite, np.sqrt, np.modf])
@pytest.mark.parametrize("physical_dtype", [pa.float32(), pa.float64()])
def test_arrow_ufunc_distinguishes_nan_and_na(ufunc, physical_dtype):
    with (
        pd.option_context("future.distinguish_nan_and_na", True),
        cudf.option_context("mode.pandas_compatible", True),
    ):
        expected_source = pd.DataFrame(
            {"a": [1.0, np.nan, None], "b": [None, 4.0, np.nan]},
            dtype=pd.ArrowDtype(physical_dtype),
        )
        if ufunc is np.modf:
            expected_source = expected_source["a"]
        actual_source = cudf.from_pandas(expected_source, nan_as_null=False)
        actual = ufunc(actual_source)
        expected = ufunc(expected_source)
        if isinstance(expected, tuple):
            for left, right in zip(expected, actual, strict=True):
                assert_eq(left, right)
        else:
            assert_eq(expected, actual)


@pytest.mark.skipif(
    not PANDAS_GE_31, reason="pandas 3.1 Arrow NaN/NA semantics"
)
@pytest.mark.parametrize("fill_value", [0.0, np.nan])
def test_arrow_fillna_and_missing_mask_distinguish_nan(fill_value):
    with (
        pd.option_context("future.distinguish_nan_and_na", True),
        cudf.option_context("mode.pandas_compatible", True),
    ):
        expected = pd.Series([None, 1.0, np.nan], dtype="double[pyarrow]")
        actual = cudf.from_pandas(expected, nan_as_null=False)
        assert_eq(expected.isna(), actual.isna())
        assert_eq(expected.notna(), actual.notna())
        assert_eq(expected.fillna(fill_value), actual.fillna(fill_value))
