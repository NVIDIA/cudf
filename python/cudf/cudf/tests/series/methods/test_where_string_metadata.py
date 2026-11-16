# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from decimal import Decimal

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

import cudf
from cudf.core._compat import PANDAS_GE_31
from cudf.testing import assert_eq


@pytest.mark.skipif(
    not PANDAS_GE_31, reason="pandas 3.1 listlike EA replacements"
)
@pytest.mark.parametrize(
    "dtype",
    [
        pd.StringDtype(storage=storage, na_value=na_value)
        for storage in ["python", "pyarrow"]
        for na_value in [pd.NA, np.nan]
    ]
    + [pd.ArrowDtype(pa.string()), pd.ArrowDtype(pa.large_string())],
)
@pytest.mark.parametrize("other", [["c", "d", "e", "f"], ["c"], "c", None])
def test_where_preserves_string_dtype(dtype, other):
    source = pd.Series(["prefix", "a", None, "b", "c"], dtype=dtype)
    expected = source.iloc[1:].where([True, False, True, False], other)
    actual = (
        cudf.from_pandas(source)
        .iloc[1:]
        .where([True, False, True, False], other)
    )
    assert_eq(expected, actual)
    assert actual.dtype == dtype


@pytest.mark.skipif(
    not PANDAS_GE_31, reason="pandas 3.1 listlike EA replacements"
)
@pytest.mark.parametrize(
    "other",
    [
        [Decimal("3.0"), Decimal("4.0"), Decimal("5.0")],
        [Decimal("3.0")],
        Decimal("3.0"),
        None,
    ],
)
def test_where_preserves_arrow_decimal_precision(other):
    dtype = pd.ArrowDtype(pa.decimal128(7, 3))
    source = pd.Series([Decimal("1"), None, Decimal("2")], dtype=dtype)
    expected = source.where([True, False, False], other)
    actual = cudf.from_pandas(source).where([True, False, False], other)
    assert_eq(expected, actual)
    assert actual.dtype == dtype


@pytest.mark.skipif(
    not PANDAS_GE_31, reason="pandas 3.1 rejects unordered EA replacements"
)
@pytest.mark.parametrize(
    "dtype",
    [
        "Int64",
        "Float64",
        pd.ArrowDtype(pa.int64()),
        pd.ArrowDtype(pa.float64()),
    ],
)
@pytest.mark.parametrize("cond", [[True, False, True], [True, True, True]])
def test_where_rejects_unordered_numeric_ea_replacements(dtype, cond):
    source = pd.Series([1, 2, 3], dtype=dtype)
    if all(cond):
        assert_eq(
            source.where(cond, {7, 8, 9}),
            cudf.from_pandas(source).where(cond, {7, 8, 9}),
        )
    else:
        with pytest.raises(TypeError):
            source.where(cond, {7, 8, 9})
        with pytest.raises(TypeError):
            cudf.from_pandas(source).where(cond, {7, 8, 9})


@pytest.mark.skipif(
    not PANDAS_GE_31,
    reason="pandas 3.1 retains string EA metadata on expansion",
)
@pytest.mark.parametrize("storage", ["python", "pyarrow"])
@pytest.mark.parametrize("na_value", [pd.NA, np.nan])
def test_loc_expansion_preserves_string_dtype(storage, na_value):
    source = pd.Series(
        ["a", None, "b"],
        dtype=pd.StringDtype(storage=storage, na_value=na_value),
    )
    actual = cudf.from_pandas(source)
    source.loc[3] = "c"
    actual.loc[3] = "c"
    assert_eq(source, actual)
