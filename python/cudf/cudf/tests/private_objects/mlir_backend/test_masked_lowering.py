# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

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
