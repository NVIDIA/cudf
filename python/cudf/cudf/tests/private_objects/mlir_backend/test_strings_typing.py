# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Python-surface tests for the MLIR backend strings typing module.

These exercise the host-side surface added by
``cudf.core.udf.mlir_backend.strings_typing`` without compiling a kernel:

* ``MLIRStringType`` identity / ABI size / dtype / repr / return_as.
* ``typeof()`` of the wrapper used to marshal ``mlir_string`` arrays.
* ``MLIRStringArgHandler.prepare_args`` for the bare ``CPointer`` case and
  the ``Tuple(CPointer, mask)`` case used for nullable string columns.

Kernel-level (lowering) tests live in ``test_strings_lowering.py``.
"""

from __future__ import annotations

import numpy as np
from numba_cuda_mlir import types
from numba_cuda_mlir.numba_cuda.typing.typeof import typeof

from cudf.core.udf.mlir_backend.strings_typing import (
    ManagedStrArrayWrapper,
    MLIRStringArgHandler,
    MLIRStringType,
    mlir_string,
    mlir_string_arg_handler,
)


class _FakeBuffer:
    """Minimal stand-in for a Buffer-like object exposing ``ptr``."""

    def __init__(self, ptr):
        self.ptr = ptr


def test_extensionty_size_constant():
    # 8 (meminfo) + 8 (data) + 8 (nbytes)
    assert MLIRStringType._extensionty_size == 24


def test_mlir_string_repr_and_return_as():
    assert repr(mlir_string) == "mlir_string"
    assert mlir_string.return_as is mlir_string


def test_mlir_string_np_dtype_object():
    assert MLIRStringType.np_dtype == np.dtype("O")


def test_mlir_string_is_not_internal():
    # Passed to kernels as a CPointer, never as an Array/memref.
    assert mlir_string.is_internal is False


def test_typeof_managed_str_array_wrapper():
    """``typeof(ManagedStrArrayWrapper(...))`` -> ``CPointer(mlir_string)``."""
    wrapper = ManagedStrArrayWrapper(_FakeBuffer(0xDEAD_BEEF))
    ty = typeof(wrapper)
    assert isinstance(ty, types.CPointer)
    assert ty.dtype is mlir_string


def test_arg_handler_cpointer_mlir_string():
    """Bare ``CPointer(mlir_string)`` is rewritten to ``(uint64, ptr)``."""
    handler = MLIRStringArgHandler()
    wrapper = ManagedStrArrayWrapper(_FakeBuffer(0xCAFE_F00D))
    ty = types.CPointer(mlir_string)
    new_ty, new_val = handler.prepare_args(ty, wrapper)
    assert new_ty is types.uint64
    assert new_val == 0xCAFE_F00D


def test_arg_handler_tuple_with_mask():
    """``Tuple(CPointer(mlir_string), mask)`` -> tuple with the ptr extracted.

    This is the shape produced for nullable string columns: the string-view
    pointer alongside its validity mask array.
    """
    handler = MLIRStringArgHandler()
    wrapper = ManagedStrArrayWrapper(_FakeBuffer(0xAA))
    mask_value = "<opaque-mask-array>"
    tuple_ty = types.Tuple((types.CPointer(mlir_string), types.boolean[::1]))
    new_ty, new_val = handler.prepare_args(tuple_ty, (wrapper, mask_value))
    assert new_ty is tuple_ty
    assert new_val == (0xAA, mask_value)


def test_arg_handler_passthrough_for_non_string_types():
    """Types unrelated to ``mlir_string`` pass through unchanged."""
    handler = MLIRStringArgHandler()
    plain_ty = types.float64[::1]
    sentinel = object()
    new_ty, new_val = handler.prepare_args(plain_ty, sentinel)
    assert new_ty is plain_ty
    assert new_val is sentinel


def test_module_exposes_arg_handler_singleton():
    assert isinstance(mlir_string_arg_handler, MLIRStringArgHandler)
