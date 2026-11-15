# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pandas as pd
import pytest

import cudf
from cudf.core._compat import PANDAS_GE_31
from cudf.testing import assert_eq


@pytest.mark.skipif(
    not PANDAS_GE_31, reason="pandas 3.1 compound column selectors"
)
@pytest.mark.parametrize(
    "key",
    [
        (["A", "B"], "t1"),
        ("A", ["t1", "t2"]),
        (["A", "C"], slice(None)),
        (["A", "B"], ["t1"]),
        (slice(None), ["t1"]),
        (np.array(["A", "B"]), "t1"),
    ],
)
def test_multiindex_getitem_per_level_lists(key):
    columns = pd.MultiIndex.from_product(
        [["A", "B", "C"], ["t1", "t2"]], names=["group", "kind"]
    )
    expected = pd.DataFrame(np.arange(18).reshape(3, 6), columns=columns)
    actual = cudf.from_pandas(expected)
    assert_eq(actual[key], expected[key])


@pytest.mark.skipif(not PANDAS_GE_31, reason="pandas 3.1 datetime std dtype")
@pytest.mark.parametrize("unit", ["s", "ms", "us", "ns"])
@pytest.mark.parametrize("skipna", [False, True])
@pytest.mark.parametrize("ddof", [0, 1])
def test_datetime_axis1_std_returns_timedelta(unit, skipna, ddof):
    values = np.arange(9).reshape(3, 3).astype(f"datetime64[{unit}]")
    values[1, 1] = np.datetime64("NaT")
    expected = pd.DataFrame(values)
    actual = cudf.from_pandas(expected)
    assert_eq(
        actual.std(axis=1, skipna=skipna, ddof=ddof),
        expected.std(axis=1, skipna=skipna, ddof=ddof),
    )


@pytest.mark.skipif(
    not PANDAS_GE_31, reason="pandas 3.1 constructor reindex metadata"
)
def test_constructor_multiindex_to_flat_keeps_column_index():
    expected = pd.DataFrame(
        np.arange(9.0).reshape(3, 3),
        columns=[[2, 2, 4], [6, 8, 10]],
        index=[[4, 4, 8], [8, 10, 12]],
    )
    actual = cudf.from_pandas(expected)
    assert_eq(
        cudf.DataFrame(actual.iloc[:2], index=[8, 10]),
        pd.DataFrame(expected.iloc[:2], index=[8, 10]),
    )


@pytest.mark.parametrize("dtype", [">i4", ">i8", ">f8"])
def test_native_non_native_byteorder_input_values(dtype):
    values = np.array([0, 256], dtype=dtype)
    with cudf.option_context("mode.pandas_compatible", False):
        actual = cudf.Series(values)
    assert_eq(actual, pd.Series(values.astype(values.dtype.newbyteorder("="))))
