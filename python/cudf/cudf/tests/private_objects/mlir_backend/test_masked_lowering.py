# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import operator

import cupy as cp
import numpy as np
import pytest

pytest.importorskip("numba_cuda_mlir")

from numba_cuda_mlir import (
    cuda,
    types,
)

import cudf.core.udf.mlir_backend.masked_lowering
import cudf.core.udf.mlir_backend.masked_typing  # noqa: F401
from cudf.core.missing import NA
from cudf.core.udf.api import Masked
from cudf.core.udf.utils import DEPRECATED_SM_REGEX

from .utils import MLIRNumbaCudaConfig

pytestmark = [
    pytest.mark.filterwarnings(f"ignore:{DEPRECATED_SM_REGEX}:UserWarning"),
    pytest.mark.filterwarnings(
        "ignore:Linking LTOIR with optimization_level:"
        "numba_cuda_mlir.numba_cuda.core.errors.NumbaWarning"
    ),
]


def _launch(kernel, *args):
    # Launch over a 1x1 grid
    with MLIRNumbaCudaConfig():
        kernel[1, 1](*args)
    cuda.synchronize()


_DTYPE_SAMPLES = [
    (types.int8, np.int8, np.int8(7)),
    (types.int16, np.int16, np.int16(-300)),
    (types.int32, np.int32, np.int32(123_456)),
    (types.int64, np.int64, np.int64(-9_999_999)),
    (types.uint8, np.uint8, np.uint8(255)),
    (types.uint16, np.uint16, np.uint16(60_000)),
    (types.uint32, np.uint32, np.uint32(4_000_000_000)),
    (types.uint64, np.uint64, np.uint64(9_223_372_036_854_775_808)),
    (types.float32, np.float32, np.float32(3.5)),
    (types.float64, np.float64, np.float64(-1.25)),
    (types.boolean, np.bool_, np.bool_(True)),
]


@pytest.mark.parametrize("valid", [True, False])
@pytest.mark.parametrize("nb_ty,np_dt,sample", _DTYPE_SAMPLES)
def test_masked_constructor_and_accessors_literal_validity(
    nb_ty, np_dt, sample, valid
):
    @cuda.jit(types.void(nb_ty[::1], types.boolean[::1], nb_ty[::1]))
    def k(out_value, out_valid, v):
        m = Masked(v[0], valid)
        out_value[0] = m.value
        out_valid[0] = m.valid

    in_v = cp.array([sample], dtype=np_dt)
    out_value = cp.zeros(1, dtype=np_dt)
    out_valid = cp.zeros(1, dtype=np.bool_)
    _launch(k, out_value, out_valid, in_v)
    assert bool(out_valid.get()[0]) is valid
    if valid:
        assert out_value.get()[0] == sample


@pytest.mark.parametrize("valid", [True, False])
def test_masked_constructor_and_accessors_runtime_validity(valid):
    @cuda.jit(
        types.void(
            types.int64[::1],
            types.boolean[::1],
            types.int64[::1],
            types.boolean[::1],
        )
    )
    def k(out_value, out_valid, v, valid_in):
        m = Masked(v[0], valid_in[0])
        out_value[0] = m.value
        out_valid[0] = m.valid

    in_v = cp.array([42], dtype=np.int64)
    in_valid = cp.array([valid], dtype=np.bool_)
    out_value = cp.zeros(1, dtype=np.int64)
    out_valid = cp.zeros(1, dtype=np.bool_)
    _launch(k, out_value, out_valid, in_v, in_valid)
    assert bool(out_valid.get()[0]) is valid
    if valid:
        assert int(out_value.get()[0]) == 42


def test_cast_na_to_masked_via_branch_unification():
    """``return cudf.NA`` branch unifies with a Masked branch.

    The taken-NA branch should produce a Masked with valid=False;
    the taken-value branch should produce valid=True. Exercises
    ``cast(na_type -> MaskedType)`` and ``cast(scalar -> MaskedType)``
    in the same kernel via branch unification.
    """
    from cudf.core.missing import NA

    @cuda.jit(device=True)
    def fn(x, take_na):
        if take_na:
            return NA
        return x

    @cuda.jit(
        types.void(
            types.int64[::1],
            types.boolean[::1],
            types.int64[::1],
            types.boolean[::1],
        )
    )
    def k(out_value, out_valid, v, take_na):
        m = fn(v[0], take_na[0])
        out_value[0] = m.value
        out_valid[0] = m.valid

    # Path 1: NA branch taken.
    in_v = cp.array([42], dtype=np.int64)
    in_take_na = cp.array([True], dtype=np.bool_)
    out_value = cp.zeros(1, dtype=np.int64)
    out_valid = cp.zeros(1, dtype=np.bool_)
    _launch(k, out_value, out_valid, in_v, in_take_na)
    # value is undef so we don't assert on it; valid must be False.
    assert bool(out_valid.get()[0]) is False

    # Path 2: scalar branch taken -> Masked(42, True).
    in_take_na = cp.array([False], dtype=np.bool_)
    out_value = cp.zeros(1, dtype=np.int64)
    out_valid = cp.zeros(1, dtype=np.bool_)
    _launch(k, out_value, out_valid, in_v, in_take_na)
    assert int(out_value.get()[0]) == 42
    assert bool(out_valid.get()[0]) is True


def test_cast_masked_to_masked_promotes_value_type():
    """Branches with different Masked widths unify on the wider type;
    the narrow branch's value gets promoted, validity is preserved.
    """

    @cuda.jit(device=True)
    def fn(x_int, x_float, take_int):
        # Branches return Masked(int32) and Masked(float64) respectively.
        # Numba unifies to Masked(float64); the int32 branch goes through
        # cast(MaskedType -> MaskedType).
        if take_int:
            return Masked(x_int, True)
        return Masked(x_float, False)

    @cuda.jit(
        types.void(
            types.float64[::1],
            types.boolean[::1],
            types.int32[::1],
            types.float64[::1],
            types.boolean[::1],
        )
    )
    def k(out_value, out_valid, x_int, x_float, take_int):
        m = fn(x_int[0], x_float[0], take_int[0])
        out_value[0] = m.value
        out_valid[0] = m.valid

    in_int = cp.array([7], dtype=np.int32)
    in_float = cp.array([2.5], dtype=np.float64)

    # int32 branch: value promoted to float64; valid kept (True).
    take_int = cp.array([True], dtype=np.bool_)
    out_value = cp.zeros(1, dtype=np.float64)
    out_valid = cp.zeros(1, dtype=np.bool_)
    _launch(k, out_value, out_valid, in_int, in_float, take_int)
    assert float(out_value.get()[0]) == 7.0
    assert bool(out_valid.get()[0]) is True

    # float64 branch: value passes through; valid kept (False).
    take_int = cp.array([False], dtype=np.bool_)
    out_value = cp.zeros(1, dtype=np.float64)
    out_valid = cp.zeros(1, dtype=np.bool_)
    _launch(k, out_value, out_valid, in_int, in_float, take_int)
    assert float(out_value.get()[0]) == 2.5
    assert bool(out_valid.get()[0]) is False


@pytest.mark.parametrize("valid_in", [True, False])
def test_masked_is_na(valid_in):
    """``m is NA`` and ``NA is m`` (both orders) return ``not m.valid``."""
    from cudf.core.missing import NA

    @cuda.jit(
        types.void(types.boolean[::1], types.int64[::1], types.boolean[::1])
    )
    def k(out, v, valid):
        m = Masked(v[0], valid[0])
        out[0] = m is NA
        out[1] = NA is m

    in_v = cp.array([42], dtype=np.int64)
    in_valid = cp.array([valid_in], dtype=np.bool_)
    out = cp.zeros(2, dtype=np.bool_)
    _launch(k, out, in_v, in_valid)
    result = out.get()
    assert bool(result[0]) is (not valid_in)
    assert bool(result[1]) is (not valid_in)


@pytest.mark.parametrize("valid_in", [True, False])
def test_masked_is_not_na(valid_in):
    """``m is not NA`` and ``NA is not m`` (both orders) return ``m.valid``."""
    from cudf.core.missing import NA

    @cuda.jit(
        types.void(types.boolean[::1], types.int64[::1], types.boolean[::1])
    )
    def k(out, v, valid):
        m = Masked(v[0], valid[0])
        out[0] = m is not NA
        out[1] = NA is not m

    in_v = cp.array([42], dtype=np.int64)
    in_valid = cp.array([valid_in], dtype=np.bool_)
    out = cp.zeros(2, dtype=np.bool_)
    _launch(k, out, in_v, in_valid)
    result = out.get()
    assert bool(result[0]) is valid_in
    assert bool(result[1]) is valid_in


_ARITH = [operator.add, operator.sub, operator.mul]


@pytest.mark.parametrize("op", _ARITH)
def test_masked_masked_arith_value(op):
    """``Masked(a) <op> Masked(b)`` computes ``op(a, b)`` in the value field."""

    @cuda.jit(
        types.void(
            types.int64[::1],
            types.int64[::1],
            types.boolean[::1],
            types.int64[::1],
            types.boolean[::1],
        )
    )
    def k(out, a, av, b, bv):
        m = op(Masked(a[0], av[0]), Masked(b[0], bv[0]))
        out[0] = m.value

    a, b = 12, 5
    in_a = cp.array([a], dtype=np.int64)
    in_b = cp.array([b], dtype=np.int64)
    true_ = cp.array([True], dtype=np.bool_)
    out = cp.zeros(1, dtype=np.int64)
    _launch(k, out, in_a, true_, in_b, true_)
    assert int(out.get()[0]) == op(a, b)


@pytest.mark.parametrize(
    "av,bv,expected",
    [
        (True, True, True),
        (True, False, False),
        (False, True, False),
        (False, False, False),
    ],
)
def test_masked_masked_validity_is_anded(av, bv, expected):
    """``Masked op Masked`` validity is the AND of the operand validities."""

    @cuda.jit(
        types.void(
            types.boolean[::1],
            types.int64[::1],
            types.boolean[::1],
            types.int64[::1],
            types.boolean[::1],
        )
    )
    def k(out_valid, a, a_valid, b, b_valid):
        m = Masked(a[0], a_valid[0]) + Masked(b[0], b_valid[0])
        out_valid[0] = m.valid

    in_a = cp.array([1], dtype=np.int64)
    in_b = cp.array([2], dtype=np.int64)
    out_valid = cp.zeros(1, dtype=np.bool_)
    _launch(
        k,
        out_valid,
        in_a,
        cp.array([av], dtype=np.bool_),
        in_b,
        cp.array([bv], dtype=np.bool_),
    )
    assert bool(out_valid.get()[0]) is expected


_CMP = [
    operator.lt,
    operator.le,
    operator.gt,
    operator.ge,
    operator.eq,
    operator.ne,
]


@pytest.mark.parametrize("op", _CMP)
@pytest.mark.parametrize("a,b", [(3, 5), (5, 5), (8, 5)])
def test_masked_masked_comparison(op, a, b):
    """Comparison of two Masked values yields a Masked(boolean)."""

    @cuda.jit(
        types.void(
            types.boolean[::1],
            types.int64[::1],
            types.boolean[::1],
            types.int64[::1],
            types.boolean[::1],
        )
    )
    def k(out, x, xv, y, yv):
        m = op(Masked(x[0], xv[0]), Masked(y[0], yv[0]))
        out[0] = m.value

    true_ = cp.array([True], dtype=np.bool_)
    out = cp.zeros(1, dtype=np.bool_)
    _launch(
        k,
        out,
        cp.array([a], dtype=np.int64),
        true_,
        cp.array([b], dtype=np.int64),
        true_,
    )
    assert bool(out.get()[0]) == op(a, b)


@pytest.mark.parametrize("op", _ARITH)
def test_masked_scalar_arith(op):
    """``Masked(a) <op> literal`` carries the Masked operand's validity."""

    @cuda.jit(
        types.void(
            types.int64[::1],
            types.boolean[::1],
            types.int64[::1],
            types.boolean[::1],
        )
    )
    def k(out_v, out_valid, a, av):
        m = op(Masked(a[0], av[0]), 4)
        out_v[0] = m.value
        out_valid[0] = m.valid

    a = 10
    out_v = cp.zeros(1, dtype=np.int64)
    out_valid = cp.zeros(1, dtype=np.bool_)
    _launch(
        k,
        out_v,
        out_valid,
        cp.array([a], dtype=np.int64),
        cp.array([False], dtype=np.bool_),
    )
    assert int(out_v.get()[0]) == op(a, 4)
    # validity is carried from the (invalid) Masked operand
    assert bool(out_valid.get()[0]) is False


@pytest.mark.parametrize("op", _ARITH)
def test_scalar_masked_arith(op):
    """``literal <op> Masked(a)`` puts the scalar on the left."""

    @cuda.jit(
        types.void(
            types.int64[::1],
            types.boolean[::1],
            types.int64[::1],
            types.boolean[::1],
        )
    )
    def k(out_v, out_valid, a, av):
        m = op(100, Masked(a[0], av[0]))
        out_v[0] = m.value
        out_valid[0] = m.valid

    a = 30
    out_v = cp.zeros(1, dtype=np.int64)
    out_valid = cp.zeros(1, dtype=np.bool_)
    _launch(
        k,
        out_v,
        out_valid,
        cp.array([a], dtype=np.int64),
        cp.array([True], dtype=np.bool_),
    )
    assert int(out_v.get()[0]) == op(100, a)
    assert bool(out_valid.get()[0]) is True


def test_masked_scalar_comparison_against_literal():
    """``Masked(a) < literal`` -- the scalar literal must not be confused
    with the masked operand (regression guard for ``row['a'] < 1``).
    """

    @cuda.jit(
        types.void(types.boolean[::1], types.int64[::1], types.boolean[::1])
    )
    def k(out, a, av):
        m = Masked(a[0], av[0]) < 7
        out[0] = m.value

    true_ = cp.array([True], dtype=np.bool_)
    out = cp.zeros(1, dtype=np.bool_)
    _launch(k, out, cp.array([3], dtype=np.int64), true_)
    assert bool(out.get()[0]) is True
    out = cp.zeros(1, dtype=np.bool_)
    _launch(k, out, cp.array([9], dtype=np.int64), true_)
    assert bool(out.get()[0]) is False


@pytest.mark.parametrize("na_first", [True, False])
def test_masked_binary_with_na_is_invalid(na_first):
    """``Masked <op> NA`` (and ``NA <op> Masked``) produce an invalid result."""
    if na_first:

        @cuda.jit(
            types.void(
                types.boolean[::1], types.int64[::1], types.boolean[::1]
            )
        )
        def k(out_valid, a, av):
            m = NA + Masked(a[0], av[0])
            out_valid[0] = m.valid
    else:

        @cuda.jit(
            types.void(
                types.boolean[::1], types.int64[::1], types.boolean[::1]
            )
        )
        def k(out_valid, a, av):
            m = Masked(a[0], av[0]) + NA
            out_valid[0] = m.valid

    out_valid = cp.ones(1, dtype=np.bool_)
    _launch(
        k,
        out_valid,
        cp.array([5], dtype=np.int64),
        cp.array([True], dtype=np.bool_),  # valid operand; NA still poisons
    )
    assert bool(out_valid.get()[0]) is False
