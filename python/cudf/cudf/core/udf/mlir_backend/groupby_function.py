# SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


from functools import cache

import cupy as cp
import numpy as np
from numba_cuda_mlir import cuda, types
from numba_cuda_mlir.numba_cuda.core.errors import (
    TypingError as CoreTypingError,
    TypingError as CudaTypingError,
)
from numba_cuda_mlir.numba_cuda.np import numpy_support

from cudf.core.column import as_column, column_empty
from cudf.core.udf.mlir_backend.cooperative_device_funcs import (
    cooperative_reductions_linkable,
)
from cudf.core.udf.mlir_backend.cooperative_types import CooperativeArrayType
from cudf.core.udf.mlir_backend.cooperative_typing import CooperativeArray
from cudf.core.udf.mlir_backend.groupby_lowering import (
    register_group_slot_lowering,
)
from cudf.core.udf.mlir_backend.groupby_typing import (
    SUPPORTED_GROUPBY_NUMPY_TYPES,
    GroupByJITDataFrame,
    _group_slot,
    register_group_slot,
)
from cudf.core.udf.mlir_backend.templates import groupby_apply_kernel_template
from cudf.core.udf.mlir_backend.udf_kernel_base import ApplyKernelBase
from cudf.core.udf.templates import group_initializer_template
from cudf.core.udf.utils import (
    UDFError,
    _all_dtypes_from_frame,
    _get_extensionty_size,
    _supported_cols_from_frame,
    set_malloc_heap_size,
)
from cudf.utils._numba import _CUDFNumbaConfig
from cudf.utils.performance_tracking import _performance_tracking


def _get_frame_groupby_type(dtype, index_dtype):
    """
    Get the Numba type corresponding to a row of grouped data. Models the
    column as a Record-like data structure containing CooperativeArrayTypes.

    Parameters
    ----------
    dtype : np.dtype
        A numpy structured array dtype associating field names
        to scalar dtypes
    index_dtype : np.dtype
        A numpy scalar dtype associated with the index of the
        incoming grouped data
    """
    fields = []
    offset = 0

    sizes = [val[0].itemsize for val in dtype.fields.values()]
    for i, (name, info) in enumerate(dtype.fields.items()):
        elemdtype = info[0]
        title = info[2] if len(info) == 3 else None
        if elemdtype in SUPPORTED_GROUPBY_NUMPY_TYPES:
            ty = numpy_support.from_dtype(elemdtype)
        else:
            # Model unsupported dtypes (e.g. uint8, strings) as Poison so a UDF
            # that never touches the column still compiles, while any use of it
            # raises a UDFError (see CooperativeArrayAttr).
            try:
                inner = numpy_support.from_dtype(elemdtype)
            except Exception:
                from cudf.core.udf.mlir_backend.strings_typing import (
                    mlir_string,
                )

                inner = mlir_string
            ty = types.Poison(inner)
        coopty = CooperativeArrayType(ty)
        infos = {
            "type": coopty,
            "offset": offset,
            "title": title,
        }
        fields.append((name, infos))
        offset += _get_extensionty_size(coopty)

        if i < len(sizes) - 1:
            alignment = offset % 8
            if alignment != 0:
                offset += 8 - alignment

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

    set_malloc_heap_size()

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

    max_group_size = cp.diff(offsets).max()

    if max_group_size >= 256:
        blocklim = 256
    else:
        blocklim = ((max_group_size + 32 - 1) // 32) * 32

    if kernel.specialized:
        specialized = kernel
    else:
        specialized = kernel.specialize(*launch_args)

    # Block size (tpb) from kernel attributes; grid is ngroups (numba_cuda_mlir kernels).
    (kern_def,) = specialized.overloads.values()
    attrs = kern_def._codelibrary.get_kernel_attributes()
    tpb = min(int(blocklim), attrs["max_threads_per_block"])

    # Dynamic shared memory: 16 bytes broadcast slot + reduction scratch.
    # Two-pass reductions (std/var/corr) use a second 1024*8 scratch region
    # at offset (16 + 1024*8), so size for two regions to avoid an
    # out-of-bounds shared-memory write (illegal address) for those ops.
    sharedmem = 16 + 2 * 1024 * 8

    # Launch kernel
    with _CUDFNumbaConfig():
        specialized[ngroups, tpb, 0, sharedmem](*launch_args)

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

    def _extra_link(self):
        # Link the cooperative reduction PTX shim (_coop_sum_*, etc.) that the
        # per-group reductions call.
        return [cooperative_reductions_linkable()]

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
        register_group_slot(dataframe_group_type)
        register_group_slot_lowering(dataframe_group_type)
        col_names = tuple(
            _supported_cols_from_frame(
                self.frame, supported_types=SUPPORTED_GROUPBY_NUMPY_TYPES
            ).keys()
        )
        global_exec_context = {
            "cuda": cuda,
            "CooperativeArray": CooperativeArray,
            "dataframe_group_type": dataframe_group_type,
            "types": types,
            "_group_slot": _group_slot,
            "_col_names": col_names,
        }
        return global_exec_context

    def _construct_signature(self, return_type):
        return None
