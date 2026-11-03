# SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
import math
from functools import cache

import numpy as np
from numba import cuda
from numba.np import numpy_support

from cudf.core.udf.api import Masked, pack_return
from cudf.core.udf.masked_typing import MaskedType
from cudf.core.udf.strings_typing import string_view
from cudf.core.udf.templates import (
    masked_input_initializer_template,
    row_initializer_template,
    row_kernel_template,
    unmasked_input_initializer_template,
)
from cudf.core.udf.udf_kernel_base import ApplyKernelBase
from cudf.core.udf.utils import (
    Row,
    _all_dtypes_from_frame,
    _get_extensionty_size,
    _mask_get,
    _supported_cols_from_frame,
)


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
        _get_extensionty_size(string_view)
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
            # columns of dtype string start life as string_view
            string_view
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
            _get_extensionty_size(string_view)
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
        return _get_frame_row_type(
            np.dtype(list(_all_dtypes_from_frame(self.frame).items()))
        )

    def _get_kernel_string(self):
        row_type = self._get_frame_type()

        # Create argument list for kernel
        frame = _supported_cols_from_frame(self.frame)

        input_columns = self._format_arg_list("input_col", len(frame))
        input_offsets = self._format_arg_list("offset", len(frame))
        extra_args = self._format_arg_list("extra_arg", len(self.args))

        # Generate the initializers for each device function argument
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
            numba_rectype=row_type,
        )

    @cache
    def _get_kernel_string_exec_context(self):
        # This is the global execution context that will be used
        # to compile the kernel. It contains the function being
        # compiled and the cuda module.

        row_type = self._get_frame_type()
        col_names = tuple(_supported_cols_from_frame(self.frame).keys())
        return {
            "cuda": cuda,
            "Masked": Masked,
            "_mask_get": _mask_get,
            "pack_return": pack_return,
            "row_type": row_type,
            "_col_names": col_names,
        }
