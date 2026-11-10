# SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import math

import numpy as np
from numba_cuda_mlir import cuda, types
from numba_cuda_mlir.extending import lowering_registry, typing_registry
from numba_cuda_mlir.numba_cuda.np import numpy_support
from numba_cuda_mlir.numba_cuda.typing.templates import AbstractTemplate
from numba_cuda_mlir.typing import signature as nb_signature

from cudf.core.udf.api import Masked, pack_return
from cudf.core.udf.mlir_backend.masked_typing import MaskedType
from cudf.core.udf.mlir_backend.strings_typing import mlir_string
from cudf.core.udf.mlir_backend.templates import row_kernel_template
from cudf.core.udf.mlir_backend.udf_kernel_base import ApplyKernelBase
from cudf.core.udf.templates import (
    masked_input_initializer_template,
    row_initializer_template,
    unmasked_input_initializer_template,
)
from cudf.core.udf.utils import (
    Row,
    _all_dtypes_from_frame,
    _get_extensionty_size,
    _supported_cols_from_frame,
)


def _row_slot():
    """Device-only: return a pointer to a single row-sized slot. Do not call from Python."""
    raise NotImplementedError("_row_slot is only for JIT")


_row_slot_cases: list[Row] = []


class RowSlotTemplate(AbstractTemplate):
    key = _row_slot

    def generic(self, args, kws):
        if len(args) != 0 or kws:
            return None
        for row_type in _row_slot_cases:
            return nb_signature(row_type)
        return None


def _lower_row_slot(builder, target, args, kwargs):
    """Lowering for ``_row_slot()``: allocate record_size bytes and store the
    pointer in target (Record = ptr to bytes).

    The row ``Record`` type is the target's numba type, so we derive the slot
    size from it. Registered once at import time (released numba-cuda-mlir
    snapshots the lowering registry into the target context at setup, so
    per-kernel late registration would never be installed).
    """
    row_type = builder.get_numba_type(target.name)
    i8 = builder.get_mlir_type(types.uint8)
    ptr = builder.alloca(i8, count=row_type.size)
    builder.store_var(target, ptr)


def register_row_slot(row_type):
    """Record this kernel's row_type for _row_slot() typing (lowering is
    registered once at import via _register_row_slot).
    """
    global _row_slot_cases
    _row_slot_cases = [row_type]


def _register_row_slot():
    typing_registry.register_global(_row_slot, types.Function(RowSlotTemplate))
    lowering_registry.lower(_row_slot)(_lower_row_slot)


def _get_frame_row_type(dtype):
    """
    Get the Numba type of a row in a frame. Models each column and its mask as
    a MaskedType and models the row as a dictionary like data structure
    containing these MaskedTypes. Large parts of this function are copied with
    comments from the Numba internals and slightly modified to account for
    validity bools to be present in the final struct. See
    numba.np.numpy_support.from_struct_dtype for details.
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

        ty = (
            mlir_string
            if elemdtype == np.dtype("O")
            else numpy_support.from_dtype(elemdtype)
        )
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

    # Numba requires that structures are aligned for the CUDA target.
    return Row(fields, offset, aligned=True)


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
        return _get_frame_row_type(
            np.dtype(list(_all_dtypes_from_frame(self.frame).items()))
        )

    def _get_kernel_string(self):
        frame = _supported_cols_from_frame(self.frame)

        input_columns = ", ".join(
            [f"input_col_{i}" for i in range(len(frame))]
        )
        input_offsets = ", ".join([f"offset_{i}" for i in range(len(frame))])
        extra_args = ", ".join(
            [f"extra_arg_{i}" for i in range(len(self.args))]
        )

        initializers = []
        row_initializers = []
        for i, (colname, col) in enumerate(frame.items()):
            idx = str(i)
            template = (
                masked_input_initializer_template
                if col.mask is not None
                else unmasked_input_initializer_template
            )
            initializers.append(template.format(idx=idx))
            row_initializers.append(row_initializer_template.format(idx=idx))

        return row_kernel_template.format(
            input_columns=input_columns,
            input_offsets=input_offsets,
            extra_args=extra_args,
            masked_input_initializers="\n".join(initializers),
            row_initializers="\n".join(row_initializers),
        )

    def _get_kernel_string_exec_context(self):
        row_type = self._get_frame_type()
        register_row_slot(row_type)
        col_names = tuple(_supported_cols_from_frame(self.frame).keys())
        return {
            "cuda": cuda,
            "Masked": Masked,
            "pack_return": pack_return,
            "row_type": row_type,
            "_row_slot": _row_slot,
            "_col_names": col_names,
        }


_register_row_slot()
