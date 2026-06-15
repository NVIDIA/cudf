# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
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


@pytest.mark.parametrize(
    "op,ref", [(operator.neg, lambda x: -x), (operator.pos, lambda x: +x)]
)
def test_masked_unary_sign(op, ref):
    """TODO: write docstring."""

    @cuda.jit(
        types.void(
            types.int64[::1],
            types.boolean[::1],
            types.int64[::1],
            types.boolean[::1],
        )
    )
    def k(out_v, out_valid, a, av):
        m = op(Masked(a[0], av[0]))
        out_v[0] = m.value
        out_valid[0] = m.valid

    out_v = cp.zeros(1, dtype=np.int64)
    out_valid = cp.zeros(1, dtype=np.bool_)
    _launch(
        k,
        out_v,
        out_valid,
        cp.array([7], dtype=np.int64),
        cp.array([False], dtype=np.bool_),
    )
    assert int(out_v.get()[0]) == ref(7)
    # validity carried from the operand
    assert bool(out_valid.get()[0]) is False


@pytest.mark.parametrize("x", [5, 0, -6, 255])
def test_masked_invert(x):
    """TODO: write docstring."""

    @cuda.jit(types.void(types.int64[::1], types.int64[::1], types.boolean[::1]))
    def k(out, a, av):
        out[0] = (~Masked(a[0], av[0])).value

    out = cp.zeros(1, dtype=np.int64)
    _launch(k, out, cp.array([x], dtype=np.int64), cp.array([True], dtype=np.bool_))
    assert int(out.get()[0]) == ~x


@pytest.mark.parametrize(
    "fn,ref",
    [
        (math.sin, math.sin),
        (math.cos, math.cos),
        (math.sqrt, math.sqrt),
        (math.exp, math.exp),
    ],
)
def test_masked_unary_math(fn, ref):
    """TODO: write docstring."""

    @cuda.jit(types.void(types.float64[::1], types.float64[::1], types.boolean[::1]))
    def k(out, a, av):
        out[0] = fn(Masked(a[0], av[0])).value

    out = cp.zeros(1, dtype=np.float64)
    _launch(
        k, out, cp.array([1.5], dtype=np.float64), cp.array([True], dtype=np.bool_)
    )
    np.testing.assert_allclose(float(out.get()[0]), ref(1.5), rtol=1e-12)


@pytest.mark.parametrize("x", [-9, 0, 12])
def test_masked_abs(x):
    """TODO: write docstring."""

    @cuda.jit(
        types.void(
            types.int64[::1],
            types.boolean[::1],
            types.int64[::1],
            types.boolean[::1],
        )
    )
    def k(out_v, out_valid, a, av):
        m = abs(Masked(a[0], av[0]))
        out_v[0] = m.value
        out_valid[0] = m.valid

    out_v = cp.zeros(1, dtype=np.int64)
    out_valid = cp.zeros(1, dtype=np.bool_)
    _launch(
        k,
        out_v,
        out_valid,
        cp.array([x], dtype=np.int64),
        cp.array([True], dtype=np.bool_),
    )
    assert int(out_v.get()[0]) == abs(x)
    assert bool(out_valid.get()[0]) is True


@pytest.mark.parametrize(
    "value,valid,expected",
    [
        (5, True, True),   # valid & truthy
        (0, True, False),  # valid & falsy
        (5, False, False),  # invalid -> False regardless of payload
        (0, False, False),
    ],
)
def test_masked_bool_truth(value, valid, expected):
    """TODO: write docstring."""

    @cuda.jit(types.void(types.boolean[::1], types.int64[::1], types.boolean[::1]))
    def k(out, a, av):
        out[0] = bool(Masked(a[0], av[0]))

    out = cp.zeros(1, dtype=np.bool_)
    _launch(
        k,
        out,
        cp.array([value], dtype=np.int64),
        cp.array([valid], dtype=np.bool_),
    )
    assert bool(out.get()[0]) is expected


def test_masked_bool_in_if_condition():
    """TODO: write docstring."""

    @cuda.jit(types.void(types.int64[::1], types.int64[::1], types.boolean[::1]))
    def k(out, a, av):
        m = Masked(a[0], av[0])
        if m:
            out[0] = 1
        else:
            out[0] = 0

    out = cp.zeros(1, dtype=np.int64)
    _launch(k, out, cp.array([5], dtype=np.int64), cp.array([True], dtype=np.bool_))
    assert int(out.get()[0]) == 1
    _launch(k, out, cp.array([5], dtype=np.int64), cp.array([False], dtype=np.bool_))
    assert int(out.get()[0]) == 0


def test_masked_float_cast():
    """TODO: write docstring."""

    @cuda.jit(
        types.void(
            types.float64[::1],
            types.boolean[::1],
            types.int64[::1],
            types.boolean[::1],
        )
    )
    def k(out_v, out_valid, a, av):
        m = float(Masked(a[0], av[0]))
        out_v[0] = m.value
        out_valid[0] = m.valid

    out_v = cp.zeros(1, dtype=np.float64)
    out_valid = cp.zeros(1, dtype=np.bool_)
    _launch(
        k,
        out_v,
        out_valid,
        cp.array([3], dtype=np.int64),
        cp.array([True], dtype=np.bool_),
    )
    assert float(out_v.get()[0]) == 3.0
    assert bool(out_valid.get()[0]) is True


def test_masked_int_cast():
    """TODO: write docstring."""

    @cuda.jit(
        types.void(
            types.int64[::1],
            types.boolean[::1],
            types.float64[::1],
            types.boolean[::1],
        )
    )
    def k(out_v, out_valid, a, av):
        m = int(Masked(a[0], av[0]))
        out_v[0] = m.value
        out_valid[0] = m.valid

    out_v = cp.zeros(1, dtype=np.int64)
    out_valid = cp.zeros(1, dtype=np.bool_)
    _launch(
        k,
        out_v,
        out_valid,
        cp.array([3.9], dtype=np.float64),
        cp.array([False], dtype=np.bool_),
    )
    assert int(out_v.get()[0]) == 3
    assert bool(out_valid.get()[0]) is False


#
# cupy rejects datetime64/timedelta64 dtypes directly, so device arrays are
# allocated as int64 and viewed as the temporal dtype (the same trick
# ``cudf.core.udf.utils._return_arr_from_dtype`` uses).

_DT = types.NPDatetime("ns")
_TD = types.NPTimedelta("ns")


def _dt_in(values):
    return cp.asarray(np.array(values, dtype="int64")).view("datetime64[ns]")


def _td_in(values):
    return cp.asarray(np.array(values, dtype="int64")).view("timedelta64[ns]")


def _dt_out():
    return cp.zeros(1, dtype=np.int64).view("datetime64[ns]")


def _td_out():
    return cp.zeros(1, dtype=np.int64).view("timedelta64[ns]")


def _bool(v):
    return cp.array([v], dtype=np.bool_)


def test_masked_datetime_minus_datetime_is_timedelta():
    """TODO: write docstring."""

    @cuda.jit(
        types.void(
            _TD[::1], types.boolean[::1],
            _DT[::1], types.boolean[::1],
            _DT[::1], types.boolean[::1],
        )
    )
    def k(out_v, out_valid, a, av, b, bv):
        m = Masked(a[0], av[0]) - Masked(b[0], bv[0])
        out_v[0] = m.value
        out_valid[0] = m.valid

    out_v = _td_out()
    out_valid = cp.zeros(1, dtype=np.bool_)
    _launch(k, out_v, out_valid, _dt_in([1000]), _bool(True),
            _dt_in([400]), _bool(True))
    assert int(out_v.get().view("int64")[0]) == 600
    assert bool(out_valid.get()[0]) is True


def test_masked_datetime_plus_timedelta_is_datetime():
    """TODO: write docstring."""

    @cuda.jit(
        types.void(
            _DT[::1],
            _DT[::1], types.boolean[::1],
            _TD[::1], types.boolean[::1],
        )
    )
    def k(out_v, a, av, t, tv):
        out_v[0] = (Masked(a[0], av[0]) + Masked(t[0], tv[0])).value

    out_v = _dt_out()
    _launch(k, out_v, _dt_in([1000]), _bool(True), _td_in([250]), _bool(True))
    assert int(out_v.get().view("int64")[0]) == 1250


def test_masked_timedelta_plus_timedelta_is_timedelta():
    """TODO: write docstring."""

    @cuda.jit(
        types.void(
            _TD[::1],
            _TD[::1], types.boolean[::1],
            _TD[::1], types.boolean[::1],
        )
    )
    def k(out_v, a, av, b, bv):
        out_v[0] = (Masked(a[0], av[0]) + Masked(b[0], bv[0])).value

    out_v = _td_out()
    _launch(k, out_v, _td_in([300]), _bool(True), _td_in([120]), _bool(True))
    assert int(out_v.get().view("int64")[0]) == 420


@pytest.mark.parametrize(
    "op,ref", [(operator.lt, lambda a, b: a < b), (operator.gt, lambda a, b: a > b)]
)
def test_masked_datetime_comparison(op, ref):
    """TODO: write docstring."""

    @cuda.jit(
        types.void(
            types.boolean[::1],
            _DT[::1], types.boolean[::1],
            _DT[::1], types.boolean[::1],
        )
    )
    def k(out, a, av, b, bv):
        out[0] = op(Masked(a[0], av[0]), Masked(b[0], bv[0])).value

    out = cp.zeros(1, dtype=np.bool_)
    _launch(k, out, _dt_in([400]), _bool(True), _dt_in([1000]), _bool(True))
    assert bool(out.get()[0]) == ref(400, 1000)


def test_masked_datetime_arith_validity_propagates():
    """TODO: write docstring."""

    @cuda.jit(
        types.void(
            _TD[::1], types.boolean[::1],
            _DT[::1], types.boolean[::1],
            _DT[::1], types.boolean[::1],
        )
    )
    def k(out_v, out_valid, a, av, b, bv):
        m = Masked(a[0], av[0]) - Masked(b[0], bv[0])
        out_v[0] = m.value
        out_valid[0] = m.valid

    out_v = _td_out()
    out_valid = cp.ones(1, dtype=np.bool_)
    _launch(k, out_v, out_valid, _dt_in([1000]), _bool(True),
            _dt_in([400]), _bool(False))
    assert bool(out_valid.get()[0]) is False


# --- membership: value in (...) -------------------------------------------


@pytest.mark.parametrize("x,expected", [(1, True), (3, True), (5, True), (4, False), (0, False)])
def test_masked_in_literal_tuple_int(x, expected):
    """``Masked(int) in (1, 3, 5)`` (literal tuple) -> Masked(boolean)."""

    @cuda.jit(types.void(types.boolean[::1], types.int64[::1], types.boolean[::1]))
    def k(out, a, av):
        out[0] = (Masked(a[0], av[0]) in (1, 3, 5)).value

    out = cp.zeros(1, dtype=np.bool_)
    _launch(k, out, cp.array([x], dtype=np.int64), cp.array([True], dtype=np.bool_))
    assert bool(out.get()[0]) is expected


@pytest.mark.parametrize("x,expected", [(1.5, True), (2.5, True), (9.9, False)])
def test_masked_in_literal_tuple_float(x, expected):
    """Literal-tuple membership works for float payloads too."""

    @cuda.jit(types.void(types.boolean[::1], types.float64[::1], types.boolean[::1]))
    def k(out, a, av):
        out[0] = (Masked(a[0], av[0]) in (1.5, 2.5)).value

    out = cp.zeros(1, dtype=np.bool_)
    _launch(k, out, cp.array([x], dtype=np.float64), cp.array([True], dtype=np.bool_))
    assert bool(out.get()[0]) is expected


@pytest.mark.parametrize("x,expected", [(2, True), (7, True), (9, True), (8, False)])
def test_masked_in_unittuple_int(x, expected):
    """``Masked(int) in (a, b, c)`` (runtime homogeneous UniTuple)."""

    @cuda.jit(
        types.void(
            types.boolean[::1], types.int64[::1], types.boolean[::1],
            types.int64[::1], types.int64[::1], types.int64[::1],
        )
    )
    def k(out, a, av, p, q, r):
        out[0] = (Masked(a[0], av[0]) in (p[0], q[0], r[0])).value

    out = cp.zeros(1, dtype=np.bool_)
    _launch(
        k, out,
        cp.array([x], dtype=np.int64), cp.array([True], dtype=np.bool_),
        cp.array([2], dtype=np.int64),
        cp.array([7], dtype=np.int64),
        cp.array([9], dtype=np.int64),
    )
    assert bool(out.get()[0]) is expected


def test_masked_in_tuple_invalid_propagates():
    """An invalid Masked operand makes the membership result invalid."""

    @cuda.jit(
        types.void(
            types.boolean[::1], types.boolean[::1],
            types.int64[::1], types.boolean[::1],
        )
    )
    def k(out_v, out_valid, a, av):
        m = Masked(a[0], av[0]) in (1, 3, 5)
        out_v[0] = m.value
        out_valid[0] = m.valid

    out_v = cp.zeros(1, dtype=np.bool_)
    out_valid = cp.ones(1, dtype=np.bool_)
    _launch(
        k, out_v, out_valid,
        cp.array([3], dtype=np.int64),  # would be a hit, but operand invalid
        cp.array([False], dtype=np.bool_),
    )
    assert bool(out_valid.get()[0]) is False
