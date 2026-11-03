# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Self-contained device tests for the MLIR GroupBy JIT cooperative reductions.

Each test launches a single cooperative block over a device array, constructs a
``CooperativeArray`` and invokes a reduction directly (bypassing the
``GroupBy.apply`` wiring, which is not part of this PR), then verifies the
scalar result on the host. This mirrors the per-group reduction the JIT GroupBy
backend performs for one group.
"""
from __future__ import annotations

import cupy as cp
import numpy as np
import pytest
from numba_cuda_mlir import cuda, types

import cudf.core.udf.mlir_backend.cooperative_lowering  # noqa: F401 registers lowerings + link hook
import cudf.core.udf.mlir_backend.cooperative_typing  # noqa: F401 registers typing
from cudf.core.udf.mlir_backend.cooperative_typing import CooperativeArray
from cudf.core.udf.utils import UDFError

from .utils import MLIRNumbaCudaConfig

# 16-byte reserved slot + two 1024xf64 reduction scratch regions (matches the
# JIT GroupBy kernel launch configuration).
_SHM = 16 + 2 * 1024 * 8
_TPB = 128

# Block sizes exercised by the parametrized tests. Includes non-power-of-two
# sizes (96, 100) because the JIT GroupBy launcher rounds the block size up to a
# multiple of 32, so groups of 33-255 rows routinely launch 96, 160, ... threads
# (bdice review on #25018). 100 additionally proves the tree reduction is
# correct for sizes that are not even multiples of the warp size.
_TPBS = [32, 96, 100, 128, 256]

_NB = {
    np.dtype("int32"): types.int32,
    np.dtype("int64"): types.int64,
    np.dtype("float32"): types.float32,
    np.dtype("float64"): types.float64,
}


def _nb1(dtype):
    return _NB[np.dtype(dtype)][::1]


def _reduce(arr, method, out_dtype, tpb=_TPB):
    """Run ``CooperativeArray(arr).<method>()`` over one block; return the scalar."""
    out = cp.zeros(1, dtype=out_dtype)
    g = {"cuda": cuda, "CooperativeArray": CooperativeArray}
    exec(
        f"def _k(o, a):\n"
        f"    r = CooperativeArray(a, a.size).{method}()\n"
        f"    if cuda.grid(1) == 0:\n        o[0] = r\n",
        g,
    )
    k = cuda.jit(types.void(_nb1(out_dtype), _nb1(arr.dtype)))(g["_k"])
    with MLIRNumbaCudaConfig():
        k[1, tpb, 0, _SHM](out, arr)
    cuda.synchronize()
    return out.get()[0]


def _reduce_idx(arr, index, method, tpb=_TPB):
    """Run ``CooperativeArray(arr, size, index).<method>()``; return the int64 index."""
    out = cp.zeros(1, dtype=np.int64)
    g = {"cuda": cuda, "CooperativeArray": CooperativeArray}
    exec(
        f"def _k(o, a, idx):\n"
        f"    r = CooperativeArray(a, a.size, idx).{method}()\n"
        f"    if cuda.grid(1) == 0:\n        o[0] = r\n",
        g,
    )
    k = cuda.jit(types.void(types.int64[::1], _nb1(arr.dtype), types.int64[::1]))(
        g["_k"]
    )
    with MLIRNumbaCudaConfig():
        k[1, tpb, 0, _SHM](out, arr, index)
    cuda.synchronize()
    return out.get()[0]


def _corr(a, b, tpb=_TPB):
    """Run ``CooperativeArray(a).corr(CooperativeArray(b))``; return the scalar."""
    out = cp.zeros(1, dtype=np.float64)
    g = {"cuda": cuda, "CooperativeArray": CooperativeArray}
    exec(
        "def _k(o, a, b):\n"
        "    r = CooperativeArray(a, a.size).corr(CooperativeArray(b, b.size))\n"
        "    if cuda.grid(1) == 0:\n        o[0] = r\n",
        g,
    )
    k = cuda.jit(types.void(types.float64[::1], _nb1(a.dtype), _nb1(b.dtype)))(g["_k"])
    with MLIRNumbaCudaConfig():
        k[1, tpb, 0, _SHM](out, a, b)
    cuda.synchronize()
    return out.get()[0]


_VALS = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3]


@pytest.mark.parametrize("tpb", _TPBS)
@pytest.mark.parametrize("dtype", ["int32", "int64", "float32", "float64"])
def test_min(dtype, tpb):
    arr = cp.asarray(np.array(_VALS, dtype=dtype))
    assert _reduce(arr, "min", dtype, tpb=tpb) == np.array(_VALS, dtype=dtype).min()


@pytest.mark.parametrize("tpb", _TPBS)
@pytest.mark.parametrize("dtype", ["int32", "int64", "float32", "float64"])
def test_max(dtype, tpb):
    arr = cp.asarray(np.array(_VALS, dtype=dtype))
    assert _reduce(arr, "max", dtype, tpb=tpb) == np.array(_VALS, dtype=dtype).max()


@pytest.mark.parametrize("tpb", _TPBS)
@pytest.mark.parametrize("dtype", ["int32", "int64", "float32", "float64"])
def test_sum(dtype, tpb):
    arr = cp.asarray(np.array(_VALS, dtype=dtype))
    # Integer sums are promoted to int64 (overflow-safe), matching pandas.
    out_dtype = "int64" if np.dtype(dtype).kind in "iu" else dtype
    got = _reduce(arr, "sum", out_dtype, tpb=tpb)
    assert got == np.array(_VALS, dtype=dtype).sum()


@pytest.mark.parametrize("tpb", _TPBS)
@pytest.mark.parametrize("method", ["mean", "var", "std"])
@pytest.mark.parametrize("dtype", ["int32", "int64", "float32", "float64"])
def test_mean_var_std(method, dtype, tpb):
    # float32 stays float32; everything else -> float64 (matches pandas).
    out_dtype = "float32" if dtype == "float32" else "float64"
    arr = cp.asarray(np.array(_VALS, dtype=dtype))
    got = _reduce(arr, method, out_dtype, tpb=tpb)
    ref = getattr(np.array(_VALS, dtype=np.float64), method)(
        **({"ddof": 1} if method in ("var", "std") else {})
    )
    np.testing.assert_allclose(got, ref, rtol=1e-5)


@pytest.mark.parametrize("tpb", _TPBS)
@pytest.mark.parametrize("method", ["min", "max", "sum", "mean", "var", "std"])
def test_input_longer_than_block(method, tpb):
    # size >> blockDim: each thread strides over many elements before the tree
    # reduction, which only the parametrized block sizes exercise.
    rng = np.random.default_rng(0)
    vals = rng.standard_normal(1000).astype("float64")
    arr = cp.asarray(vals)
    got = _reduce(arr, method, "float64", tpb=tpb)
    ref = getattr(vals, method)(
        **({"ddof": 1} if method in ("var", "std") else {})
    )
    np.testing.assert_allclose(got, ref, rtol=1e-6, atol=1e-9)


@pytest.mark.parametrize("method", ["var", "std"])
def test_var_std_large_magnitude(method):
    # Catastrophic-cancellation guard: the one-pass sum(x^2) - sum(x)^2/n form
    # loses all precision for large-magnitude inputs; the two-pass form does
    # not (bdice review on #25018).
    vals = (1e8 + np.arange(10)).astype("float64")
    arr = cp.asarray(vals)
    got = _reduce(arr, method, "float64")
    ref = getattr(vals, method)(ddof=1)
    np.testing.assert_allclose(got, ref, rtol=1e-6)


@pytest.mark.parametrize("tpb", _TPBS)
@pytest.mark.parametrize("method", ["min", "max", "sum"])
def test_nan_skipping(method, tpb):
    # min/max/sum skip NaN (pandas skipna=True); all-NaN sum is 0.0.
    vals = np.array([1.0, np.nan, 3.0, np.nan, 2.0], dtype="float64")
    arr = cp.asarray(vals)
    got = _reduce(arr, method, "float64", tpb=tpb)
    ref = {"sum": np.nansum, "min": np.nanmin, "max": np.nanmax}[method](vals)
    assert got == ref


@pytest.mark.parametrize("tpb", _TPBS)
@pytest.mark.parametrize("method", ["idxmin", "idxmax"])
def test_idx(method, tpb):
    vals = np.array(_VALS, dtype="int64")
    index = np.array([10, 11, 12, 13, 14, 15, 16, 17, 18, 19], dtype="int64")
    arr = cp.asarray(vals)
    idx = cp.asarray(index)
    got = _reduce_idx(arr, idx, method, tpb=tpb)
    # smallest-index tie-break (pandas returns the first occurrence)
    pos = int(vals.argmin() if method == "idxmin" else vals.argmax())
    assert got == index[pos]


@pytest.mark.parametrize("method", ["idxmin", "idxmax"])
def test_idx_all_nan(method):
    # An all-NaN group falls back to the first index label, matching the legacy
    # BlockIdxMax/BlockIdxMin (bdice review on #25018).
    vals = np.array([np.nan, np.nan, np.nan], dtype="float64")
    index = np.array([10, 11, 12], dtype="int64")
    got = _reduce_idx(cp.asarray(vals), cp.asarray(index), method)
    assert got == index[0]


@pytest.mark.parametrize("tpb", _TPBS)
@pytest.mark.parametrize("dtype", ["int32", "int64"])
def test_corr(dtype, tpb):
    a = cp.asarray(np.array([1, 2, 3, 4, 5], dtype=dtype))
    b = cp.asarray(np.array([2, 4, 6, 8, 10], dtype=dtype))
    np.testing.assert_allclose(_corr(a, b, tpb=tpb), 1.0, rtol=1e-6)


def test_corr_float_unsupported():
    # Correlation of floating types is not supported (#13839); rejected in typing.
    a = cp.asarray(np.array([1.0, 2.0, 3.0], dtype="float64"))
    g = {"cuda": cuda, "CooperativeArray": CooperativeArray}
    exec(
        "def _k(o, a):\n"
        "    r = CooperativeArray(a, a.size).corr(CooperativeArray(a, a.size))\n"
        "    if cuda.grid(1) == 0:\n        o[0] = r\n",
        g,
    )
    with pytest.raises(UDFError, match="corr"):
        cuda.jit(types.void(types.float64[::1], types.float64[::1]))(g["_k"])


def test_corr_dtype_mismatch_rejected():
    # Mismatched integer dtypes are rejected in typing; otherwise lowering would
    # read one buffer at the wrong element width (bdice review on #25018).
    a = cp.asarray(np.array([1, 2, 3], dtype="int32"))
    b = cp.asarray(np.array([1, 2, 3], dtype="int64"))
    g = {"cuda": cuda, "CooperativeArray": CooperativeArray}
    exec(
        "def _k(o, a, b):\n"
        "    r = CooperativeArray(a, a.size).corr(CooperativeArray(b, b.size))\n"
        "    if cuda.grid(1) == 0:\n        o[0] = r\n",
        g,
    )
    with pytest.raises(UDFError, match="corr"):
        cuda.jit(
            types.void(types.float64[::1], types.int32[::1], types.int64[::1])
        )(g["_k"])


def test_operator_on_group_rejected():
    # Operators applied directly to a grouped column are not supported.
    a = cp.asarray(np.array([1, 2, 3], dtype="int64"))
    g = {"cuda": cuda, "CooperativeArray": CooperativeArray}
    exec(
        "def _k(o, a):\n"
        "    r = CooperativeArray(a, a.size) + CooperativeArray(a, a.size)\n"
        "    if cuda.grid(1) == 0:\n        o[0] = r.size\n",
        g,
    )
    with pytest.raises(UDFError, match="is not supported"):
        cuda.jit(types.void(types.int64[::1], types.int64[::1]))(g["_k"])
