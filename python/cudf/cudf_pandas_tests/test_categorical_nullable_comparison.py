# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import operator
import re

import pandas as xpd
import pytest

from cudf.pandas import LOADED
from cudf.testing import assert_eq

if not LOADED:
    raise ImportError("These tests must be run with cudf.pandas loaded")

pd = xpd._fsproxy_slow


@pytest.mark.parametrize(
    "categories_dtype", ["Int64", "Float64", "int64[pyarrow]", "boolean"]
)
@pytest.mark.parametrize("numpy_dtype", ["int64", "float64", "bool"])
@pytest.mark.parametrize("has_nulls", [False, True])
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("operation", [operator.eq, operator.ne])
def test_categorical_nullable_categories_numpy_comparison(
    categories_dtype, numpy_dtype, has_nulls, reverse, operation
):
    categories = pd.Index(
        [False, True] if categories_dtype == "boolean" else [1, 2],
        dtype=categories_dtype,
    )
    categorical = pd.Series(
        pd.Categorical.from_codes(
            [0, 1, -1 if has_nulls else 0], categories=categories
        )
    )
    numeric = pd.Series(
        [False, True, True] if numpy_dtype == "bool" else [1, 2, 3],
        dtype=numpy_dtype,
    )
    left, right = (categorical, numeric) if reverse else (numeric, categorical)
    actual_left, actual_right = xpd.Series(left), xpd.Series(right)
    try:
        expected = operation(left, right)
    except Exception as error:
        with pytest.raises(type(error), match=re.escape(str(error))):
            operation(actual_left, actual_right)
    else:
        actual = operation(actual_left, actual_right)
        assert_eq(actual._fsproxy_slow, expected)
