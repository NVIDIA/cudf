# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import datetime
import operator
from functools import reduce

import cupy as cp
import numpy as np
import pandas as pd
import pytest

import cudf
from cudf.testing import assert_eq
from cudf.testing._utils import (
    gen_rand_ufunc_input,
    set_random_null_mask_inplace,
)


@pytest.mark.parametrize("has_nulls", [True, False])
@pytest.mark.parametrize("indexed", [True, False])
def test_ufunc_series(request, numpy_ufunc, has_nulls, indexed):
    # Note: This test assumes that all ufuncs are unary or binary.
    request.applymarker(
        pytest.mark.xfail(
            condition=(
                indexed
                and numpy_ufunc
                in {
                    np.greater,
                    np.greater_equal,
                    np.less,
                    np.less_equal,
                    np.not_equal,
                    np.equal,
                }
            ),
            reason="Comparison operators do not support misaligned indexes.",
        )
    )
    request.applymarker(
        pytest.mark.xfail(
            condition=numpy_ufunc == np.matmul and has_nulls,
            reason="Can't call cupy on column with nulls",
        )
    )

    N = 100
    # Avoid zeros in either array to skip division by 0 errors. Also limit the
    # scale to avoid issues with overflow, etc. We use ints because some
    # operations (like bitwise ops) are not defined for floats.
    rng = np.random.default_rng(0)
    pandas_args = args = [
        cudf.Series(
            gen_rand_ufunc_input(numpy_ufunc, rng, N),
            index=rng.choice(range(N), N, False) if indexed else None,
        )
        for _ in range(numpy_ufunc.nin)
    ]

    if has_nulls:
        # Converting nullable integer cudf.Series to pandas will produce a
        # float pd.Series, so instead we replace nulls with an arbitrary
        # integer value, precompute the mask, and then reapply it afterwards.
        for arg in args:
            set_random_null_mask_inplace(arg)
        pandas_args = [arg.fillna(0) for arg in args]

        # Note: Different indexes must be aligned before the mask is computed.
        # This requires using an internal function (_align_indices), and that
        # is unlikely to change for the foreseeable future.
        aligned = (
            cudf.core.series._align_indices(args, allow_non_unique=True)
            if indexed and numpy_ufunc.nin == 2
            else args
        )
        mask = reduce(operator.or_, (a.isna() for a in aligned)).to_pandas()
        if numpy_ufunc in (np.power, np.float_power):
            # pandas honors 1 ** x == 1 and x ** 0 == 1 even when x is
            # missing, and cudf matches, so those positions are valid in the
            # result. The 0-filled pandas args already compute 1 there
            # (1 ** 0 and 0 ** 0), so just unmask them.
            base, exponent = aligned
            identity = ((base == 1).fillna(False) & exponent.isna()) | (
                (exponent == 0).fillna(False) & base.isna()
            )
            mask &= ~identity.to_pandas()

    got = numpy_ufunc(*args)

    expect = numpy_ufunc(*(arg.to_pandas() for arg in pandas_args))

    if numpy_ufunc.nout > 1:
        for g, e in zip(got, expect, strict=True):
            if has_nulls:
                e[mask] = np.nan
            assert_eq(g, e, check_exact=False)
    else:
        if has_nulls:
            if numpy_ufunc in (
                np.isfinite,
                np.isinf,
                np.isnan,
                np.logical_and,
                np.logical_not,
                np.logical_or,
                np.logical_xor,
                np.signbit,
                np.equal,
                np.greater,
                np.greater_equal,
                np.less,
                np.less_equal,
                np.not_equal,
            ):
                # cuDF .to_pandas for bools with nulls represents missing as None,
                # should this be np.nan?
                expect = expect.astype(object).mask(mask, None)
            else:
                expect[mask] = np.nan
            assert_eq(got, expect, check_exact=False)


@pytest.mark.parametrize(
    "ufunc", [np.add, np.greater, np.greater_equal, np.logical_and]
)
@pytest.mark.parametrize("has_nulls", [True, False])
@pytest.mark.parametrize("indexed", [True, False])
@pytest.mark.parametrize("reflect", [True, False])
def test_binary_ufunc_series_array(
    request, ufunc, has_nulls, indexed, reflect
):
    fname = ufunc.__name__
    request.applymarker(
        pytest.mark.xfail(
            condition=reflect and has_nulls,
            reason=(
                "When cupy is the left operand there is no way for us to "
                "avoid calling its binary operators, which cannot handle "
                "cudf objects that contain nulls."
            ),
        )
    )
    # The way cudf casts nans in arrays to nulls during binops with cudf
    # objects is currently incompatible with pandas.
    request.applymarker(
        pytest.mark.xfail(
            condition=(
                fname in {"greater", "greater_equal", "logical_and"}
                and has_nulls
            ),
            reason=(
                "cudf and pandas incompatible casting nans to nulls in binops"
            ),
        )
    )
    N = 100
    # Avoid zeros in either array to skip division by 0 errors. Also limit the
    # scale to avoid issues with overflow, etc. We use ints because some
    # operations (like bitwise ops) are not defined for floats.
    rng = np.random.default_rng(0)
    args = [
        cudf.Series(
            rng.random(N),
            index=rng.choice(range(N), N, False) if indexed else None,
        )
        for _ in range(ufunc.nin)
    ]

    if has_nulls:
        # Converting nullable integer cudf.Series to pandas will produce a
        # float pd.Series, so instead we replace nulls with an arbitrary
        # integer value, precompute the mask, and then reapply it afterwards.
        for arg in args:
            set_random_null_mask_inplace(arg)

        # Cupy doesn't support nulls, so we fill with nans before converting.
        args[1] = args[1].fillna(cp.nan)
        mask = args[0].isna().to_pandas()

    arg1 = args[1].to_cupy()

    if reflect:
        got = ufunc(arg1, args[0])
        expect = ufunc(args[1].to_numpy(), args[0].to_pandas())
    else:
        got = ufunc(args[0], arg1)
        expect = ufunc(args[0].to_pandas(), args[1].to_numpy())

    if ufunc.nout > 1:
        for g, e in zip(got, expect, strict=True):
            if has_nulls:
                e[mask] = np.nan
            if reflect:
                assert (cp.asnumpy(g) == e).all()
            else:
                assert_eq(g, e, check_exact=False)
    else:
        if has_nulls:
            expect[mask] = np.nan
        if reflect:
            assert (cp.asnumpy(got) == expect).all()
        else:
            assert_eq(got, expect, check_exact=False)


def test_ufunc_cudf_series_error_with_out_kwarg():
    cudf_s1 = cudf.Series(data=[-1, 2, 3, 0])
    cudf_s2 = cudf.Series(data=[-1, 2, 3, 0])
    cudf_s3 = cudf.Series(data=[0, 0, 0, 0])
    with pytest.raises(TypeError):
        np.add(x1=cudf_s1, x2=cudf_s2, out=cudf_s3)


@pytest.mark.parametrize(
    "ufunc", [np.cos, np.sin, np.exp, np.log, np.sqrt, np.sign, np.abs]
)
@pytest.mark.parametrize(
    "input_dtype",
    [
        "Float32",
        "Float64",
        "Int8",
        "Int16",
        "Int32",
        "Int64",
        "UInt8",
        "UInt16",
        "UInt32",
        "UInt64",
    ],
)
def test_unary_ufunc_preserves_pandas_nullable_dtype(
    request, ufunc, input_dtype
):
    # Match pandas behavior: a unary ufunc on a Series with a pandas-nullable
    # dtype should return a Series whose dtype is the corresponding
    # pandas-nullable dtype (Float64 for transcendental ops on integers, same
    # dtype family for sign/abs, etc.).
    request.applymarker(
        pytest.mark.xfail(
            condition=(
                input_dtype in {"Int8", "UInt8"}
                and ufunc in {np.cos, np.sin, np.exp, np.log, np.sqrt}
            ),
            reason=(
                "cupy promotes 8-bit ints to float16 for transcendental ops, "
                "which cudf does not support."
            ),
        )
    )
    psr = pd.Series([1, 2, 3, pd.NA], dtype=input_dtype)
    gsr = cudf.from_pandas(psr)

    with np.errstate(invalid="ignore"):
        expected = ufunc(psr)
        got = ufunc(gsr)

    assert expected.dtype == got.dtype
    assert_eq(expected, got, check_dtype=True)


@pytest.mark.parametrize("ufunc", [np.add, np.subtract, np.multiply])
@pytest.mark.parametrize(
    "left_dtype, right_dtype",
    [
        ("Float64", "Float64"),
        ("Int64", "Int64"),
        ("Float64", "Int64"),
        ("Int32", "Float32"),
    ],
)
def test_binary_ufunc_preserves_pandas_nullable_dtype(
    ufunc, left_dtype, right_dtype
):
    pa = pd.Series([1, 2, 3, pd.NA], dtype=left_dtype)
    pb = pd.Series([4, 5, pd.NA, 7], dtype=right_dtype)
    ga = cudf.from_pandas(pa)
    gb = cudf.from_pandas(pb)

    expected = ufunc(pa, pb)
    got = ufunc(ga, gb)

    assert expected.dtype == got.dtype
    assert_eq(expected, got, check_dtype=True)


@pytest.mark.parametrize("ufunc", [np.cos, np.sqrt, np.sign])
def test_unary_ufunc_plain_numpy_dtype_unchanged(ufunc):
    # Sanity check: plain numpy dtypes still produce plain numpy dtype outputs
    # (no accidental upcast to pandas-nullable).
    psr = pd.Series([1.0, 2.0, 3.0])
    gsr = cudf.from_pandas(psr)

    with np.errstate(invalid="ignore"):
        expected = ufunc(psr)
        got = ufunc(gsr)

    assert expected.dtype == got.dtype
    assert_eq(expected, got, check_dtype=True)


@pytest.mark.parametrize(
    "data",
    [
        [1000000, 200000, 3000000],
        [1000000, 200000, None],
        [],
        [None],
        [None, None, None, None, None],
        [12, 12, 22, 343, 4353534, 435342],
        np.array([10, 20, 30, None, 100]),
        cp.asarray([10, 20, 30, 100]),
        [1000000, 200000, 3000000],
        [1000000, 200000, None],
        [1],
        [12, 11, 232, 223432411, 2343241, 234324, 23234],
        [12, 11, 2.32, 2234.32411, 2343.241, 23432.4, 23234],
        [1.321, 1132.324, 23223231.11, 233.41, 0.2434, 332, 323],
        [12, 11, 2.32, 2234.32411, 2343.241, 23432.4, 23234],
    ],
)
@pytest.mark.parametrize(
    "scalar",
    [
        datetime.timedelta(days=768),
        datetime.timedelta(seconds=768),
        datetime.timedelta(microseconds=7),
        np.timedelta64("nat"),
        np.timedelta64(1, "s"),
        np.timedelta64(1, "ms"),
        np.timedelta64(1, "us"),
        np.timedelta64(1, "ns"),
    ],
)
@pytest.mark.parametrize("op", [np.add, np.subtract])
def test_datetime_series_ops_with_scalars_misc(
    data, scalar, datetime_types_as_str, op
):
    gsr = cudf.Series(data=data, dtype=datetime_types_as_str)
    psr = gsr.to_pandas()

    expect = op(psr, scalar)
    got = op(gsr, scalar)

    assert_eq(expect, got)
