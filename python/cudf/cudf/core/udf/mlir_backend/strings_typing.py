# SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``mlir_string`` scalar string type and its kernel-launch marshalling."""

from __future__ import annotations

import numpy as np
from numba_cuda_mlir import types
from numba_cuda_mlir.numba_cuda.extending import typeof_impl


class MLIRStringType(types.Type):
    """NRT-managed owned string: ``{ptr meminfo, ptr data, i64 nbytes}``.

    ``data`` points at a raw UTF-8 byte buffer and ``nbytes`` is its byte
    length. ``meminfo`` is the NRT ``MemInfo`` pointer used for lifetime
    management of *owned* (allocated) strings; read-only strings that borrow
    an existing buffer (e.g. a column element consumed by ``len``) carry a
    null ``meminfo`` and are never freed.
    """

    # ABI size: 8 (meminfo) + 8 (data) + 8 (nbytes) = 24 bytes. Must match the
    # device struct layout the marshalling helpers and string lowerings use.
    _extensionty_size = 24
    np_dtype: np.dtype[np.object_] = np.dtype("object")

    def __init__(self) -> None:
        super().__init__(name="mlir_string")

    @property
    def is_internal(self) -> bool:
        """Whether numba can represent this type as a plain numpy array.

        Returns
        -------
        bool
            Always ``False``: ``mlir_string`` is an extension type that must be
            passed to kernels as a ``CPointer`` (not an Array/memref).
        """
        return False

    @property
    def return_as(self) -> MLIRStringType:
        """The type used when a value of this type is returned from a UDF.

        Returns
        -------
        MLIRStringType
            ``self``.
        """
        return self


mlir_string = MLIRStringType()


class ManagedStrArrayWrapper:
    """Host-side wrapper whose ``typeof`` is ``CPointer(mlir_string)``.

    Wraps a device buffer holding an array of ``mlir_string`` structs so that
    numba-cuda-mlir's dispatch types it as a pointer to ``mlir_string`` rather
    than inferring a raw ``uint8`` array from ``__cuda_array_interface__``.

    Parameters
    ----------
    buffer : rmm.DeviceBuffer or cudf.core.buffer.Buffer
        Device buffer holding a contiguous array of ``mlir_string`` structs;
        must expose a ``.ptr`` device address.
    """

    def __init__(self, buffer) -> None:
        self._buffer = buffer

    @property
    def ptr(self):
        """Device address of the wrapped buffer.

        Returns
        -------
        int
            The device pointer of the underlying buffer.
        """
        return self._buffer.ptr


@typeof_impl.register(ManagedStrArrayWrapper)
def _typeof_mlir_str_array_wrapper(val, c) -> types.CPointer:
    return types.CPointer(mlir_string)


class MLIRStringArgHandler:
    """Kernel-launch extension that marshals ``mlir_string`` pointers.

    A ``CPointer(mlir_string)`` argument is passed to the kernel as the raw
    device address (``uint64``). The tuple form carries an accompanying
    validity mask for nullable string columns.
    """

    def prepare_args(self, ty: types.Type, val, **kwargs) -> tuple:
        """Marshal a ``mlir_string`` pointer argument for a kernel launch.

        Parameters
        ----------
        ty : types.Type
            The numba type of the argument.
        val : object
            The host-side argument value (a wrapper exposing ``.ptr``, or a
            tuple whose first element is one).
        **kwargs
            Additional launch context (unused).

        Returns
        -------
        tuple
            ``(type, value)`` for the kernel: a bare ``CPointer(mlir_string)``
            becomes ``(uint64, device_ptr)``; a ``(pointer, mask)`` tuple keeps
            its type with the pointer replaced by its device address; other
            arguments pass through unchanged.
        """
        if isinstance(ty, types.CPointer) and isinstance(
            ty.dtype, MLIRStringType
        ):
            return types.uint64, val.ptr
        if isinstance(ty, types.Tuple) and len(ty) >= 1:
            first_ty = ty[0]
            if isinstance(first_ty, types.CPointer) and isinstance(
                first_ty.dtype, MLIRStringType
            ):
                ptr_val = val[0].ptr
                if len(ty) == 2:
                    return ty, (ptr_val, val[1])
                return ty, (ptr_val,)
        return ty, val


mlir_string_arg_handler = MLIRStringArgHandler()
