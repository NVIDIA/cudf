# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from decimal import Decimal

import pandas as xpd
import pytest

import cudf
from cudf.pandas import LOADED
from cudf.pandas.module_accelerator import disable_module_accelerator
from cudf.testing import assert_eq

if not LOADED:
    raise ImportError("These tests must be run with cudf.pandas loaded")

pd = xpd._fsproxy_slow


@pytest.mark.parametrize(
    "text", ["NaN", "-NaN", "NaN42", "Infinity", "-Infinity"]
)
@pytest.mark.parametrize(
    "box", ["series", "pandas_series", "frame", "dict", "index"]
)
def test_nonfinite_decimal_object_values_are_preserved(text, box):
    values = [Decimal(text), Decimal("1.25")]

    def construct(module):
        if box == "series":
            return module.Series(values, dtype=object)
        if box == "pandas_series":
            return module.Series(pd.Series(values, dtype=object))
        if box == "frame":
            return module.DataFrame(
                [[value] for value in values], columns=["a"]
            )
        if box == "dict":
            return module.DataFrame({"a": values})
        return module.Index(values, dtype=object)

    with disable_module_accelerator():
        expected = construct(pd)
    actual = construct(xpd)
    selected = actual["a"] if box in {"frame", "dict"} else actual
    expected_selected = expected["a"] if box in {"frame", "dict"} else expected
    assert [type(value) for value in selected.tolist()] == [
        type(value) for value in expected_selected.tolist()
    ]
    assert [str(value) for value in selected.tolist()] == [
        str(value) for value in expected_selected.tolist()
    ]
    assert_eq(actual._fsproxy_slow, expected)


def test_decimal_nan_combine_first_preserves_object_scalar():
    def combine(module):
        frame = module.DataFrame(
            [[Decimal("NaN"), Decimal("NaN")]], columns=["a", "b"]
        )
        other = module.DataFrame(
            [[module.Interval(0, 1), module.Interval(2, 3, closed="left")]],
            columns=["b", "c"],
        )
        return frame.combine_first(other)

    with disable_module_accelerator():
        expected = combine(pd)
    actual = combine(xpd)
    assert isinstance(actual.iloc[0, 0], Decimal)
    assert actual.iloc[0, 0].is_nan()
    assert_eq(actual._fsproxy_slow, expected)


def test_finite_decimal_objects_keep_gpu_storage():
    values = [Decimal("1.25"), Decimal("2.50")]
    actual = xpd.Series(values)
    assert isinstance(actual._fsproxy_wrapped, cudf.Series)
    assert_eq(actual._fsproxy_slow, pd.Series(values))
