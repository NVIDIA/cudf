# SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import math

import numpy as np
from numba_cuda_mlir import cuda, types
from numba_cuda_mlir.cuda.experimental import consteval
from numba_cuda_mlir.extending import lowering_registry, typing_registry
from numba_cuda_mlir.numba_cuda.np import numpy_support
from numba_cuda_mlir.numba_cuda.typing.templates import AbstractTemplate
from numba_cuda_mlir.typing import signature as nb_signature

from cudf.api.types import is_string_dtype as _is_string_dtype
from cudf.core.udf.api import Masked, pack_return
from cudf.core.udf.mlir_backend.masked_typing import MaskedType
from cudf.core.udf.mlir_backend.strings_typing import mlir_string
from cudf.core.udf.mlir_backend.udf_kernel_base import ApplyKernelBase
from cudf.core.udf.utils import (
    JIT_SUPPORTED_TYPES,
    Row,
    _all_dtypes_from_frame,
    _get_extensionty_size,
    _mask_get,
    _supported_cols_from_frame,
)


def _row_slot():
    """Device-only: return a pointer to a single row-sized slot. Do not call from Python."""
    raise NotImplementedError("_row_slot is only for JIT")


_row_slot_cases = []


class RowSlotTemplate(AbstractTemplate):
    key = _row_slot

    def generic(self, args, kws):
        if len(args) != 0 or kws:
            return None
        for row_type in _row_slot_cases:
            return nb_signature(row_type)
        return None


def _lower_row_slot_impl(builder, target, row_type):
    """Allocate record_size bytes and store the pointer in target (Record = ptr to bytes)."""
    record_size = row_type.size
    i8 = builder.get_mlir_type(types.uint8)
    ptr = builder.alloca(i8, count=record_size)
    builder.store_var(target, ptr)


def _lower_row_slot(builder, target, args, kwargs):
    """Lowering for ``_row_slot()``.

    The row ``Record`` type is the target's numba type, so we derive the slot
    size from it. Registered once at import time (released numba-cuda-mlir
    snapshots the lowering registry into the target context at setup, so
    per-kernel late registration would never be installed).
    """
    row_type = builder.get_numba_type(target.name)
    _lower_row_slot_impl(builder, target, row_type)


def register_row_slot(row_type):
    """Record this kernel's row_type for _row_slot() typing (lowering is
    registered once at import via _register_row_slot).
    """
    global _row_slot_cases
    _row_slot_cases = [row_type]


def _register_row_slot():
    typing_registry.register_global(_row_slot, types.Function(RowSlotTemplate))
    lowering_registry.lower(_row_slot)(_lower_row_slot)


# Backwards-compatible alias (typing-only registration entry point).
_register_row_slot_typing = _register_row_slot


def _get_frame_row_type(dtype, poison_fields=frozenset()):
    """
    Get the Numba type of a row in a frame. Models each column and its mask as
    a MaskedType and models the row as a dictionary like data structure
    containing these MaskedTypes. Large parts of this function are copied with
    comments from the Numba internals and slightly modified to account for
    validity bools to be present in the final struct. See
    numba.np.numpy_support.from_struct_dtype for details.

    Columns whose names are in ``poison_fields`` have a dtype that the JIT
    engine does not support (e.g. decimal, categorical, list). They are modeled
    as ``Poison`` so a UDF that never touches them still compiles, while any use
    of such a column raises a ``TypingError`` (see ``_get_udf_return_type``).
    """

    # Create the numpy structured type corresponding to the numpy dtype.

    fields = []
    offset = 0

    sizes = [
        _get_extensionty_size(mlir_string)
        if val[0] == np.dtype("O")
        else val[0].itemsize
        for val in dtype.fields.values()
    ]

    for i, (name, info) in enumerate(dtype.fields.items()):
        # *info* consists of the element dtype, its offset from the beginning
        # of the record, and an optional "title" containing metadata.
        # We ignore the offset in info because its value assumes no masking;
        # instead, we compute the correct offset based on the masked type.
        elemdtype = info[0]
        title = info[2] if len(info) == 3 else None

        if name in poison_fields:
            ty = types.Poison(mlir_string)
        elif elemdtype == np.dtype("O"):
            ty = mlir_string
        else:
            ty = numpy_support.from_dtype(elemdtype)
        infos = {
            "type": MaskedType(ty),
            "offset": offset,
            "title": title,
        }
        fields.append((name, infos))

        # increment offset by itemsize plus one byte for validity
        itemsize = (
            _get_extensionty_size(mlir_string)
            if elemdtype == np.dtype("O")
            else elemdtype.itemsize
        )
        offset += itemsize + 1

        # Align the next member of the struct to be a multiple of the
        # memory access size, per PTX ISA 7.4/5.4.5
        if i < len(sizes) - 1:
            next_itemsize = sizes[i + 1]
            offset = int(math.ceil(offset / next_itemsize) * next_itemsize)

    # Numba requires that structures are aligned for the CUDA target
    _is_aligned_struct = True
    return Row(fields, offset, _is_aligned_struct)


class DataFrameApplyKernel(ApplyKernelBase):
    """
    Class representing a kernel that computes the result of
    a DataFrame.apply operation. Expects that the user passed
    a function that operates on an input row of the dataframe,
    for example

    def f(row):
        return row['x'] + row['y']
    """

    @property
    def kernel_type(self):
        return "dataframe_apply"

    def _get_frame_type(self):
        poison_fields = {
            name
            for name, col in self.frame._column_labels_and_values
            if str(col.dtype) not in JIT_SUPPORTED_TYPES
            and not _is_string_dtype(col.dtype)
        }
        return _get_frame_row_type(
            np.dtype(list(_all_dtypes_from_frame(self.frame).items())),
            poison_fields,
        )

    def _make_kernel(self):
        """Build the DataFrame.apply kernel by metaprogramming (``consteval``).

        ``consteval(range(num_columns))`` unrolls the per-column loop and
        ``consteval(column_names[idx])`` folds each column name to a literal
        Record key; ``*consteval(extra)`` splats the UDF's extra arguments. No
        string templates or ``exec`` are involved.
        """
        row_type = self._get_frame_type()
        register_row_slot(row_type)
        column_names = tuple(_supported_cols_from_frame(self.frame).keys())
        num_columns = len(column_names)
        device_func = self.device_func

        def _kernel(retval, size, columns, extra):
            i = cuda.grid(1)
            ret_data_arr, ret_mask_arr = retval
            if i < size:
                row = _row_slot()
                for idx in consteval(range(num_columns)):
                    d, m, off = columns[idx]
                    if m.size == 0:
                        valid = True
                    else:
                        valid = _mask_get(m, i + off)
                    row[consteval(column_names[idx])] = Masked(
                        d[i + off], valid
                    )
                ret = device_func(row, *consteval(extra))
                ret_masked = pack_return(ret)
                ret_data_arr[i] = ret_masked.value
                ret_mask_arr[i] = ret_masked.valid

        return _kernel


_register_row_slot()
