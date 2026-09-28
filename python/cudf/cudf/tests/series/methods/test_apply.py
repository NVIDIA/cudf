# SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import operator

import numpy as np
import pytest

import cudf
from cudf.core.udf.utils import precompiled
from cudf.testing import assert_eq


def run_masked_udf_series(func, data, args=(), nullable=True, **kwargs):
    gsr = data
    psr = data.to_pandas(nullable=nullable)

    expect = psr.apply(func, args=args)
    obtain = gsr.apply(func, args=args).to_pandas(nullable=nullable)
    if "check_dtype" in kwargs and not kwargs.get("check_dtype", True):
        obtain = obtain.astype(expect.dtype, errors="ignore")
    assert_eq(expect, obtain, **kwargs)


@pytest.mark.parametrize(
    "data",
    [
        np.array(
            [0, 1, -1, 0, np.iinfo("int64").min, np.iinfo("int64").max],
            dtype="int64",
        ),
        np.array([0, 0, 1, np.iinfo("uint64").max], dtype="uint64"),
        np.array(
            [
                0,
                0.0,
                -1.0,
                1.5,
                -1.5,
                np.finfo("float64").min,
                np.finfo("float64").max,
                np.nan,
                np.inf,
                -np.inf,
            ],
            dtype="float64",
        ),
        [False, True, False, cudf.NA],
    ],
)
def test_masked_udf_abs(data):
    data = cudf.Series(data)
    data[0] = cudf.NA

    def func(x):
        return abs(x)

    run_masked_udf_series(func, data, check_dtype=False)


@pytest.mark.parametrize(
    "data", [[1.0, 0.0, 1.5], [1, 0, 2], [True, False, True]]
)
@pytest.mark.parametrize("operator", [float, int, bool])
def test_masked_udf_casting(operator, data):
    data = cudf.Series(data)

    def func(x):
        return operator(x)

    run_masked_udf_series(func, data, check_dtype=False)


def test_masked_udf_caching():
    # Make sure similar functions that differ
    # by simple things like constants actually
    # recompile

    data = cudf.Series([1, 2, 3])

    expect = data**2
    got = data.apply(lambda x: x**2)
    assert_eq(expect, got, check_dtype=False)

    # update the constant value being used and make sure
    # it does not result in a cache hit

    expect = data**3
    got = data.apply(lambda x: x**3)
    assert_eq(expect, got, check_dtype=False)

    # make sure we get a hit when reapplying
    def f(x):
        return x + 1

    precompiled.clear()
    assert precompiled.currsize == 0
    data.apply(f)

    assert precompiled.currsize == 1
    data.apply(f)

    assert precompiled.currsize == 1

    # validate that changing the type of a scalar arg
    # results in a miss
    precompiled.clear()

    def f(x, c):
        return x + c

    data.apply(f, args=(1,))
    assert precompiled.currsize == 1

    data.apply(f, args=(1.5,))
    assert precompiled.currsize == 2


@pytest.mark.parametrize(
    "data",
    [
        [1, cudf.NA, 3],
        [0.5, 2.0, cudf.NA, cudf.NA, 5.0],
        [True, False, cudf.NA],
    ],
)
def test_masked_udf_scalar_args_binops_multiple_series(
    request, data, binary_op
):
    data = cudf.Series(data)
    request.applymarker(
        pytest.mark.xfail(
            binary_op
            in [
                operator.eq,
                operator.ne,
                operator.lt,
                operator.le,
                operator.gt,
                operator.ge,
            ]
            and data.dtype.kind != "b",
            reason="https://github.com/pandas-dev/pandas/issues/57390",
        )
    )

    def func(data, c, k):
        x = binary_op(data, c)
        y = binary_op(x, k)
        return y

    run_masked_udf_series(func, data, args=(1, 2), check_dtype=False)


@pytest.mark.parametrize(
    "data",
    [
        [1, cudf.NA, 3],
        [0.5, 2.0, cudf.NA, cudf.NA, 5.0],
        [True, False, cudf.NA],
    ],
)
def test_mask_udf_scalar_args_binops_series(data):
    data = cudf.Series(data)

    def func(x, c):
        return x + c

    run_masked_udf_series(func, data, args=(1,), check_dtype=False)


@pytest.mark.parametrize(
    "data,name",
    [([1, 2, 3], None), ([1, cudf.NA, 3], None), ([1, 2, 3], "test_name")],
)
def test_series_apply_basic(data, name):
    data = cudf.Series(data, name=name)

    def func(x):
        return x + 1

    run_masked_udf_series(func, data, check_dtype=False)


@pytest.mark.xfail(
    reason="https://github.com/pandas-dev/pandas/issues/57390",
)
def test_series_apply_null_conditional():
    def func(x):
        if x is cudf.NA:
            return 42
        else:
            return x - 1

    data = cudf.Series([1, cudf.NA, 3])

    run_masked_udf_series(func, data)


def test_series_arith_masked_vs_masked(arithmetic_op):
    def func(x):
        return arithmetic_op(x, x)

    data = cudf.Series([1, cudf.NA, 3])
    run_masked_udf_series(func, data, check_dtype=False)


@pytest.mark.xfail(
    reason="https://github.com/pandas-dev/pandas/issues/57390",
)
def test_series_compare_masked_vs_masked(comparison_op):
    """
    In the series case, only one other MaskedType to compare with
    - itself
    """

    def func(x):
        return comparison_op(x, x)

    data = cudf.Series([1, cudf.NA, 3])
    run_masked_udf_series(func, data, check_dtype=False)


@pytest.mark.parametrize("constant", [1, 1.5, cudf.NA])
def test_series_arith_masked_vs_constant(request, arithmetic_op, constant):
    def func(x):
        return arithmetic_op(x, constant)

    # Just a single column -> result will be all NA
    data = cudf.Series([1, 2, cudf.NA])
    # in pandas, 1**NA == 1. In cudf, 1**NA == NA.
    request.applymarker(
        pytest.mark.xfail(
            condition=(
                constant is cudf.NA
                and arithmetic_op in {operator.pow, operator.ipow}
            ),
            reason="https://github.com/NVIDIA/cudf/issues/7478",
        )
    )
    run_masked_udf_series(func, data, check_dtype=False)


@pytest.mark.parametrize("constant", [1, 1.5, cudf.NA])
def test_series_arith_masked_vs_constant_reflected(
    request, arithmetic_op, constant
):
    def func(x):
        return arithmetic_op(constant, x)

    # Just a single column -> result will be all NA
    data = cudf.Series([1, 2, cudf.NA])
    # Using in {1} since bool(NA == 1) raises a TypeError since NA is
    # neither truthy nor falsy
    # in pandas, 1**NA == 1. In cudf, 1**NA == NA.
    request.applymarker(
        pytest.mark.xfail(
            condition=(
                constant in {1}
                and arithmetic_op in {operator.pow, operator.ipow}
            ),
            reason="https://github.com/NVIDIA/cudf/issues/7478",
        )
    )
    run_masked_udf_series(func, data, check_dtype=False)


@pytest.mark.xfail(
    reason="https://github.com/pandas-dev/pandas/issues/57390",
)
def test_series_masked_is_null_conditional():
    def func(x):
        if x is cudf.NA:
            return 42
        else:
            return x

    data = cudf.Series([1, cudf.NA, 3, cudf.NA])

    run_masked_udf_series(func, data, check_dtype=False)
