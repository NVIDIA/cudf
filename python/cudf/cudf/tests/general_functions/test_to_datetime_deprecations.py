# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import warnings

import numpy as np
import pandas as pd
import pytest

import cudf
from cudf.core._compat import PANDAS_GE_31
from cudf.testing import assert_eq


@pytest.mark.parametrize(
    "dtype, container",
    [
        (dtype, container)
        for dtype in ["int32", "int64", "uint64", "float32", "float64"]
        for container in ["scalar", "list", "array", "series", "index"]
        if container != "list" or not dtype.startswith("float")
    ],
)
def test_numeric_format_deprecation(dtype, container):
    values = np.array([2020, 2021], dtype=dtype)
    if container == "scalar":
        values = values[0]
    elif container == "list":
        values = values.tolist()
    elif container == "series":
        values = pd.Series(values, name="dates")
    elif container == "index":
        values = pd.Index(values, name="dates")
    actual_values = (
        cudf.from_pandas(values)
        if isinstance(values, (pd.Series, pd.Index))
        else values
    )

    with warnings.catch_warnings(record=True) as expected_warnings:
        warnings.simplefilter("always")
        expected = pd.to_datetime(values, format="%Y")
    with warnings.catch_warnings(record=True) as actual_warnings:
        warnings.simplefilter("always")
        actual = cudf.to_datetime(actual_values, format="%Y")

    assert [warning.category for warning in actual_warnings] == [
        warning.category for warning in expected_warnings
    ]
    assert [str(warning.message) for warning in actual_warnings] == [
        str(warning.message) for warning in expected_warnings
    ]
    assert len(actual_warnings) == int(PANDAS_GE_31)
    assert_eq(actual, expected)


@pytest.mark.parametrize("values", [[], [None], [np.nan], ["2020", None]])
def test_format_without_numeric_values_is_quiet(values):
    with warnings.catch_warnings(record=True) as actual_warnings:
        warnings.simplefilter("always")
        actual = cudf.to_datetime(values, format="%Y")
    assert not actual_warnings
    assert_eq(actual, pd.to_datetime(values, format="%Y"))


@pytest.mark.skipif(
    not PANDAS_GE_31, reason="Numeric format parsing deprecated in 3.1"
)
def test_numeric_format_warning_is_not_coerced_to_nat():
    with warnings.catch_warnings():
        warnings.simplefilter("error", pd.errors.Pandas4Warning)
        with pytest.raises(
            pd.errors.Pandas4Warning, match="Parsing integer or float"
        ):
            cudf.to_datetime(2020, format="%Y", errors="coerce")
