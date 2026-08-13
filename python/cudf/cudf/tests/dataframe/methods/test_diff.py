# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pandas as pd
import pytest

import cudf
from cudf.testing import assert_eq


@pytest.mark.parametrize(
    "data",
    [
        [1, 0, 4, -10, 6],
        np.array([1.123, 2.343, 5.890, 0.0]),
        [True, False, True, False, False],
        {"a": [1.123, 2.343, np.nan, np.nan], "b": [None, 3, 9.08, None]},
    ],
)
@pytest.mark.parametrize("periods", (-5, -1, 0, 1, 5))
def test_diff_numeric_dtypes(data, periods):
    gdf = cudf.DataFrame(data)
    # Use arrow_type=True because that is the only case where we can guarantee that the
    # same NaN/null representation will be used. Without it, pandas will fall back to
    # object dtype for calling shift (called internally by diff) on boolean data.
    pdf = gdf.to_pandas(arrow_type=True)

    actual = gdf.diff(periods=periods, axis=0)
    expected = pdf.diff(periods=periods, axis=0)

    assert_eq(
        expected,
        actual.to_pandas(arrow_type=True),
    )


@pytest.mark.parametrize(
    ("precision", "scale"),
    [(5, 2), (8, 5)],
)
@pytest.mark.parametrize(
    "dtype",
    [cudf.Decimal32Dtype, cudf.Decimal64Dtype],
)
def test_diff_decimal_dtypes(precision, scale, dtype):
    gdf = cudf.DataFrame(
        np.random.default_rng(seed=42).uniform(10.5, 75.5, (10, 6)),
        dtype=dtype(precision=precision, scale=scale),
    )
    pdf = gdf.to_pandas(arrow_type=True)

    actual = gdf.diff()
    expected = pdf.diff()

    assert_eq(
        expected,
        actual,
        check_dtype=False,
    )


def test_diff_invalid_axis():
    gdf = cudf.DataFrame(np.array([1.123, 2.343, 5.890, 0.0]))
    with pytest.raises(
        NotImplementedError, match=r"Only axis=0 is supported."
    ):
        gdf.diff(periods=1, axis=1)


@pytest.mark.parametrize(
    "data",
    [
        {
            "int_col": [1, 2, 3, 4, 5],
            "float_col": [1.0, 2.0, 3.0, 4.0, 5.0],
            "string_col": ["a", "b", "c", "d", "e"],
        },
        ["a", "b", "c", "d", "e"],
    ],
)
def test_diff_unsupported_dtypes(data):
    gdf = cudf.DataFrame(data)
    with pytest.raises(
        TypeError,
        match=r"unsupported operand type\(s\)",
    ):
        gdf.diff()


def test_diff_many_dtypes():
    pdf = pd.DataFrame(
        {
            "dates": pd.date_range("2020-01-01", "2020-01-06", freq="D"),
            "bools": pd.array(
                [True, True, True, False, True, True], dtype="boolean"
            ),
            "floats": pd.array(
                [1.0, 2.0, 3.5, np.nan, 5.0, -1.7], dtype="Float64"
            ),
            "ints": pd.array([1, 2, 3, 3, 4, 5], dtype="Int64"),
            "nans_nulls": pd.array(
                [np.nan, None, None, np.nan, np.nan, None], dtype="Float64"
            ),
        }
    )
    gdf = cudf.from_pandas(pdf)
    assert_eq(pdf.diff(), gdf.diff())
    assert_eq(pdf.diff(periods=2), gdf.diff(periods=2))
