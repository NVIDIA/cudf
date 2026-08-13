# SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


from functools import cache

import cupy as cp
import numpy as np
from numba import cuda, types
from numba.core.errors import TypingError as CoreTypingError
from numba.cuda.core.errors import TypingError as CudaTypingError
from numba.cuda.cudadrv.devices import get_context
from numba.np import numpy_support

from cudf.core.column import as_column, column_empty
from cudf.core.column.column import ColumnBase
from cudf.core.udf.groupby_typing import (
    SUPPORTED_GROUPBY_NUMPY_TYPES,
    Group,
    GroupByJITDataFrame,
    GroupType,
)
from cudf.core.udf.templates import (
    group_initializer_template,
    groupby_apply_kernel_template,
)
from cudf.core.udf.udf_kernel_base import ApplyKernelBase
from cudf.core.udf.utils import (
    UDFError,
    _all_dtypes_from_frame,
    _get_extensionty_size,
    _supported_cols_from_frame,
)
from cudf.utils._numba import _CUDFNumbaConfig
from cudf.utils.performance_tracking import _performance_tracking


def _get_frame_groupby_type(dtype, index_dtype):
    """
    Get the Numba type corresponding to a row of grouped data. Models the
    column as a Record-like data structure containing GroupTypes. See
    numba.np.numpy_support.from_struct_dtype for details.

    Parameters
    ----------
    level : np.dtype
        A numpy structured array dtype associating field names
        to scalar dtypes
    index_dtype : np.dtype
        A numpy scalar dtype associated with the index of the
        incoming grouped data
    """
    # Create the numpy structured type corresponding to the numpy dtype.
    fields = []
    offset = 0

    sizes = [val[0].itemsize for val in dtype.fields.values()]
    for i, (name, info) in enumerate(dtype.fields.items()):
        elemdtype = info[0]
        title = info[2] if len(info) == 3 else None
        ty = numpy_support.from_dtype(elemdtype)
        indexty = numpy_support.from_dtype(index_dtype)
        groupty = GroupType(ty, indexty)
        infos = {
            "type": groupty,
            "offset": offset,
            "title": title,
        }
        fields.append((name, infos))
        offset += _get_extensionty_size(groupty)

        # Align the next member of the struct to be a multiple of the
        # memory access size, per PTX ISA 7.4/5.4.5
        if i < len(sizes) - 1:
            alignment = offset % 8
            if alignment != 0:
                offset += 8 - alignment

    # Numba requires that structures are aligned for the CUDA target
    _is_aligned_struct = True
    return GroupByJITDataFrame(fields, offset, _is_aligned_struct)


@_performance_tracking
def jit_groupby_apply(offsets, grouped_values, function, *args):
    """
    Main entrypoint for JIT Groupby.apply via Numba.

    Parameters
    ----------
    offsets : list
        A list of integers denoting the indices of the group
        boundaries in grouped_values
    grouped_values : DataFrame
        A DataFrame representing the source data
        sorted by group keys
    function : callable
        The user-defined function to execute
    """

    kr = GroupByApplyKernel(grouped_values, function, args)
    kernel, return_type = kr.get_kernel()
    offsets = as_column(offsets)
    ngroups = len(offsets) - 1

    output = column_empty(ngroups, dtype=return_type)
    output = output.set_mask(None, 0)
    launch_args = [
        offsets,
        output,
        grouped_values.index,
    ]
    launch_args += list(
        _supported_cols_from_frame(
            grouped_values, supported_types=SUPPORTED_GROUPBY_NUMPY_TYPES
        ).values()
    )
    launch_args += list(args)

    # The JIT kernel does not consult null masks, and numba-cuda's argument
    # marshalling rejects ``__cuda_array_interface__`` dicts that include a
    # ``mask`` entry. Strip masks from any nullable column args so the kernel
    # sees only the underlying data buffer. The user is responsible for
    # ensuring that null values do not affect correctness when explicitly
    # requesting ``engine="jit"``.
    launch_args = [
        arg.set_mask(None, 0)
        if isinstance(arg, ColumnBase) and arg.nullable
        else arg
        for arg in launch_args
    ]

    max_group_size = cp.diff(offsets).max()

    if max_group_size >= 256:
        blocklim = 256
    else:
        blocklim = ((max_group_size + 32 - 1) // 32) * 32

    if kernel.specialized:
        specialized = kernel
    else:
        specialized = kernel.specialize(*launch_args)

    # Ask the driver to give a good config
    ctx = get_context()
    # Dispatcher is specialized, so there's only one definition - get
    # it so we can get the cufunc from the code library
    (kern_def,) = specialized.overloads.values()
    _grid, tpb = ctx.get_max_potential_block_size(
        func=kern_def._codelibrary.get_cufunc(),
        b2d_func=0,
        memsize=0,
        blocksizelimit=int(blocklim),
    )

    # Launch kernel
    with _CUDFNumbaConfig():
        specialized[ngroups, tpb](*launch_args)

    return output


def _can_be_jitted(frame, func, args):
    """
    Determine if this UDF is supported through the JIT engine
    by attempting to compile just the function to PTX using the
    target set of types
    """
    if not hasattr(func, "__code__"):
        # Numba requires bytecode to be present to proceed.
        # See https://github.com/numba/numba/issues/4587
        return False

    if any(col.has_nulls() for col in frame._columns):
        return False
    kr = GroupByApplyKernel(frame, func, args)
    try:
        kr._get_udf_return_type()
        return True
    except (UDFError, CoreTypingError, CudaTypingError):
        return False


class GroupByApplyKernel(ApplyKernelBase):
    """
    Class representing a kernel that computes the result of
    a GroupBy.apply operation. Expects that the user passed
    a function that operates on a single group of the data,
    for example

    def f(group):
        return group['x'].sum() + group['y'].sum()
    """

    @property
    def kernel_type(self):
        return "groupby_apply"

    def _get_frame_type(self):
        return _get_frame_groupby_type(
            np.dtype(list(_all_dtypes_from_frame(self.frame).items())),
            self.frame.index.dtype,
        )

    def _get_kernel_string(self):
        # Create argument list for kernel
        frame = _supported_cols_from_frame(
            self.frame, supported_types=SUPPORTED_GROUPBY_NUMPY_TYPES
        )
        input_columns = self._format_arg_list("input_col", len(frame))
        extra_args = self._format_arg_list("extra_arg", len(self.args))

        # Generate the initializers for each device function argument
        initializers = []
        for i, colname in enumerate(frame.keys()):
            initializers.append(group_initializer_template.format(idx=i))

        return groupby_apply_kernel_template.format(
            input_columns=input_columns,
            extra_args=extra_args,
            group_initializers="\n".join(initializers),
        )

    @cache
    def _get_kernel_string_exec_context(self):
        dataframe_group_type = self._get_frame_type()
        col_names = tuple(
            _supported_cols_from_frame(
                self.frame, supported_types=SUPPORTED_GROUPBY_NUMPY_TYPES
            ).keys()
        )
        global_exec_context = {
            "cuda": cuda,
            "Group": Group,
            "dataframe_group_type": dataframe_group_type,
            "types": types,
            "_col_names": col_names,
        }
        return global_exec_context

    def _construct_signature(self, return_type):
        return None
