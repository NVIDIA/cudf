# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import operator

import numpy as np
import pandas as pd
import pytest

import cudf
from cudf.testing import assert_eq


@pytest.mark.parametrize(
    "op, expected, expected_kind",
    [
        (lambda idx: 2**idx, [2, 4, 8, 16], "int"),
        (lambda idx: idx**2, [1, 4, 9, 16], "int"),
        (lambda idx: idx / 2, [0.5, 1, 1.5, 2], "float"),
        (lambda idx: 2 / idx, [2, 1, 2 / 3, 0.5], "float"),
        (lambda idx: idx % 3, [1, 2, 0, 1], "int"),
        (lambda idx: 3 % idx, [0, 1, 0, 3], "int"),
    ],
)
def test_rangeindex_binops_user_option(
    op, expected, expected_kind, default_integer_bitwidth
):
    # Test that RangeIndex is materialized into 32 bit index under user
    # configuration for binary operation.
    idx = cudf.RangeIndex(1, 5)
    actual = op(idx)
    expected = cudf.Index(
        expected, dtype=f"{expected_kind}{default_integer_bitwidth}"
    )
    assert_eq(
        expected,
        actual,
    )


@pytest.mark.parametrize(
    "op", [operator.add, operator.sub, operator.mul, operator.truediv]
)
def test_rangeindex_binop_diff_names_none(op):
    idx1 = cudf.RangeIndex(10, 13, name="foo")
    idx2 = cudf.RangeIndex(13, 16, name="bar")
    result = op(idx1, idx2)
    expected = op(idx1.to_pandas(), idx2.to_pandas())
    assert_eq(result, expected)
    assert result.name is None


@pytest.mark.parametrize(
    "rng",
    [range(0, 10, 2), range(5, 20, 3), range(0, 0, 1), range(-10, 10, 4)],
)
@pytest.mark.parametrize(
    "other",
    # Unsigned numpy scalars are deliberately not covered: pandas computes
    # with the raw operand, so e.g. ``RangeIndex(0, 10, 2) - np.uint8(4)``
    # wraps around (or raises OverflowError); cudf converts to a Python int
    # and keeps exact arithmetic instead.
    [2, -3, np.int32(3), np.int64(5), True, np.array(4)],
)
@pytest.mark.parametrize(
    "op",
    [operator.add, operator.sub, operator.mul],
    ids=lambda op: op.__name__,
)
@pytest.mark.parametrize("reflected", [False, True])
def test_rangeindex_shift_rescale_ops_preserve_range(
    rng, other, op, reflected
):
    # Integer (and bool, as 0/1) shifts and nonzero rescales return a
    # RangeIndex with the same contents as pandas; reflected variants too.
    gidx = cudf.RangeIndex(rng.start, rng.stop, rng.step, name="x")
    pidx = pd.RangeIndex(rng.start, rng.stop, rng.step, name="x")
    if reflected:
        result, expected = op(other, gidx), op(other, pidx)
    else:
        result, expected = op(gidx, other), op(pidx, other)
    assert isinstance(result, cudf.RangeIndex)
    assert isinstance(expected, pd.RangeIndex)
    assert_eq(result, expected)
    assert result.name == expected.name


def test_rangeindex_mul_timedelta_not_rescaled():
    # np.timedelta64 subclasses np.integer but must not hit the int rescale
    # fastpath (which would silently multiply by the raw tick count).
    # Classic cudf does not support int64 * timedelta64, so this raises;
    # under cudf.pandas the raise triggers the pandas fallback, which
    # returns a TimedeltaIndex.
    gidx = cudf.RangeIndex(5)
    with pytest.raises(TypeError):
        gidx * np.timedelta64(1, "D")
    with pytest.raises(TypeError):
        np.timedelta64(1, "D") * gidx


@pytest.mark.parametrize("other", [0, False])
def test_rangeindex_mul_zero_materializes(other):
    # Rescaling by 0 cannot be represented as a range; pandas returns an
    # int64 index of zeros.
    gidx = cudf.RangeIndex(0, 10, 2, name="x")
    pidx = pd.RangeIndex(0, 10, 2, name="x")
    result = gidx * other
    expected = pidx * other
    assert not isinstance(result, cudf.RangeIndex)
    assert not isinstance(expected, pd.RangeIndex)
    assert_eq(result, expected)
