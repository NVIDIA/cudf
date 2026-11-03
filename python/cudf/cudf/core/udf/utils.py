# SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import functools
import os
import warnings
from pickle import dumps
from typing import TYPE_CHECKING

import cachetools
import cupy as cp
import llvmlite.binding as ll
import numpy as np
from cuda.bindings import runtime
from numba import cuda, typeof
from numba.core.datamodel import models
from numba.core.extending import register_model
from numba.cuda.descriptor import cuda_target
from numba.np import numpy_support
from numba.types import CPointer, Record, Tuple, int64, void

import rmm

from cudf._lib import strings_udf
from cudf.core.buffer import as_buffer
from cudf.core.dtype.validators import is_dtype_obj_string
from cudf.core.udf.masked_typing import MaskedType
from cudf.core.udf.nrt_utils import nrt_enabled
from cudf.core.udf.strings_typing import (
    NRT_decref,
    managed_udf_string,
    str_view_arg_handler,
    string_view,
)
from cudf.utils.dtypes import (
    BOOL_TYPES,
    DATETIME_TYPES,
    NUMERIC_TYPES,
    SIZE_TYPE_DTYPE,
    STRING_TYPES,
    TIMEDELTA_TYPES,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    import pylibcudf as plc

    from cudf.core.buffer.buffer import Buffer
    from cudf.core.indexed_frame import IndexedFrame


# Maximum size of a string column is 2 GiB
_STRINGS_UDF_DEFAULT_HEAP_SIZE = int(
    os.environ.get("STRINGS_UDF_HEAP_SIZE", str(2**31))
)
_HEAP_SIZE = 0

JIT_SUPPORTED_TYPES = (
    NUMERIC_TYPES
    | BOOL_TYPES
    | DATETIME_TYPES
    | TIMEDELTA_TYPES
    | STRING_TYPES
)
LIBCUDF_BITMASK_TYPE = numpy_support.from_dtype(SIZE_TYPE_DTYPE)
MASK_BITSIZE = SIZE_TYPE_DTYPE.itemsize * 8

precompiled: cachetools.LRUCache = cachetools.LRUCache(maxsize=32)

# This cache is keyed on the (signature, code, closure variables) of UDFs, so
# it can hit for distinct functions that are similar. The lru_cache wrapping
# compile_udf misses for these similar functions, but doesn't need to serialize
# closure variables to check for a hit.
_udf_code_cache: cachetools.LRUCache = cachetools.LRUCache(maxsize=32)


UDF_SHIM_FILE = os.path.join(
    os.path.dirname(strings_udf.__file__), "..", "core", "udf", "shim.fatbin"
)

DEPRECATED_SM_REGEX = "Architectures prior to '<compute/sm>_75' are deprecated"


def _all_dtypes_from_frame(frame, supported_types=JIT_SUPPORTED_TYPES):
    return {
        colname: dtype
        if str(dtype) in supported_types and not is_dtype_obj_string(dtype)
        else np.dtype("O")
        for colname, dtype in frame._dtypes
    }


def _supported_cols_from_frame(frame, supported_types=JIT_SUPPORTED_TYPES):
    return {
        colname: col
        for colname, col in frame._column_labels_and_values
        if str(col.dtype) in supported_types
    }


def _masked_array_type_from_col(col):
    """
    Return a type representing a tuple of arrays,
    the first element an array of the numba type
    corresponding to `dtype`, and the second an
    array of bools representing a mask.
    """

    if is_dtype_obj_string(col.dtype):
        col_type = CPointer(string_view)
    else:
        nb_scalar_ty = numpy_support.from_dtype(col.dtype)
        col_type = nb_scalar_ty[::1]

    if col.mask is None:
        return col_type
    else:
        return Tuple((col_type, LIBCUDF_BITMASK_TYPE[::1]))


class Row(Record):
    # Numba's Record type provides a convenient abstraction for representing a
    # row, in that it provides a mapping from strings (column / field names) to
    # types. However, it cannot be used directly since it assumes that all its
    # fields can be converted to NumPy types by Numba's internal conversion
    # mechanism (`numba.np_support.as_dtype). This is not the case for cuDF
    # extension types that might be the column types (e.g. masked types, string
    # types or group types).
    #
    # We use this type for type inference and type checking, but not in code
    # generation. For this use case, it is sufficient to provide a dtype for a
    # row that corresponds to any Python object.
    @property
    def dtype(self):
        return np.dtype("object")


register_model(Row)(models.RecordModel)


@cuda.jit(device=True)
def _mask_get(mask, pos):
    """Return the validity of mask[pos] as a word."""
    return (mask[pos // MASK_BITSIZE] >> (pos % MASK_BITSIZE)) & 1


def make_cache_key(udf, sig):
    """
    Build a cache key for a user defined function. Used to avoid
    recompiling the same function for the same set of types
    """
    codebytes = udf.__code__.co_code
    constants = udf.__code__.co_consts
    names = udf.__code__.co_names

    if udf.__closure__ is not None:
        cvars = tuple(x.cell_contents for x in udf.__closure__)
        cvarbytes = dumps(cvars)
    else:
        cvarbytes = b""

    return names, constants, codebytes, cvarbytes, sig


def compile_udf(udf, type_signature):
    """Compile ``udf`` with `numba`

    Compile a python callable function ``udf`` with
    `numba.cuda.compile_ptx_for_current_device(device=True)` using
    ``type_signature`` into CUDA PTX together with the generated output type.

    The output is expected to be passed to the PTX parser in `libcudf`
    to generate a CUDA device function to be inlined into CUDA kernels,
    compiled at runtime and launched.

    Parameters
    ----------
    udf:
      a python callable function

    type_signature:
      a tuple that specifies types of each of the input parameters of ``udf``.
      The types should be one in `numba.types` and could be converted from
      numpy types with `numba.numpy_support.from_dtype(...)`.

    Returns
    -------
    ptx_code:
      The compiled CUDA PTX

    output_type:
      An numpy type

    """
    key = make_cache_key(udf, type_signature)
    res = _udf_code_cache.get(key)
    if res:
        return res

    # We haven't compiled a function like this before, so need to fall back to
    # compilation with Numba
    ptx_code, return_type = cuda.compile_ptx_for_current_device(
        udf, type_signature, device=True
    )
    if not isinstance(return_type, MaskedType):
        output_type = numpy_support.as_dtype(return_type).type
    else:
        output_type = return_type

    # Populate the cache for this function
    res = (ptx_code, output_type)
    _udf_code_cache[key] = res

    return res


def _generate_cache_key(frame, func: Callable, args, suffix="__APPLY_UDF"):
    """Create a cache key that uniquely identifies a compilation.

    A new compilation is needed any time any of the following things change:
    - The UDF itself as defined in python by the user
    - The types of the columns utilized by the UDF
    - The existence of the input columns masks
    """
    scalar_argtypes = tuple(typeof(arg) for arg in args)
    return (
        make_cache_key(func, tuple(_all_dtypes_from_frame(frame).values())),
        *(col.mask is None for col in frame._columns),
        *frame._column_names,
        scalar_argtypes,
        suffix,
    )


def _get_input_args_from_frame(fr: IndexedFrame) -> list:
    args: list[Buffer | tuple[Buffer, Buffer]] = []
    offsets = []
    for col in _supported_cols_from_frame(fr).values():
        if is_dtype_obj_string(col.dtype):
            data = column_to_string_view_array_init_heap(col.plc_column)
        else:
            data = col.data
        if col.mask is not None:
            # argument is a tuple of data, mask
            args.append((data, col.mask))
        else:
            # argument is just the data pointer
            args.append(data)
        offsets.append(col.offset)

    return args + offsets


def _return_arr_from_dtype(dtype, size):
    if dtype == np.dtype("object"):
        return rmm.DeviceBuffer(
            size=size * _get_extensionty_size(managed_udf_string)
        )
    if dtype.kind in {"M", "m"}:
        # cupy>=14 rejects cp.empty() for datetime64
        # or timedelta64 as unsupported dtypes.
        # See https://github.com/cupy/cupy/pull/9711
        return cp.empty(size, dtype=np.int64).view(dtype)
    return cp.empty(size, dtype=dtype)


@functools.cache
def _make_free_string_kernel():
    with nrt_enabled():
        with warnings.catch_warnings():
            warnings.simplefilter("default")
            warnings.filterwarnings(
                "ignore",
                message=DEPRECATED_SM_REGEX,
                category=UserWarning,
                module=r"^numba\.cuda(\.|$)",
            )

            @cuda.jit(
                void(CPointer(managed_udf_string), int64),
                link=[UDF_SHIM_FILE],
                extensions=[str_view_arg_handler],
            )
            def free_managed_udf_string_array(ary, size):
                gid = cuda.grid(1)
                if gid < size:
                    NRT_decref(ary[gid])

    return free_managed_udf_string_array


# The only supported data layout in NVVM.
# See: https://docs.nvidia.com/cuda/nvvm-ir-spec/index.html?#data-layout
_nvvm_data_layout = (
    "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-"
    "i128:128:128-f32:32:32-f64:64:64-v16:16:16-v32:32:32-"
    "v64:64:64-v128:128:128-n16:32:64"
)


def _get_extensionty_size(ty):
    """
    Return the size of an extension type in bytes
    """
    target_data = ll.create_target_data(_nvvm_data_layout)
    llty = cuda_target.target_context.data_model_manager[ty].get_value_type()
    return llty.get_abi_size(target_data)


def initfunc(f):
    """
    Decorator for initialization functions that should
    be run exactly once.
    """

    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        if wrapper.initialized:
            return
        wrapper.initialized = True
        return f(*args, **kwargs)

    wrapper.initialized = False
    return wrapper


@initfunc
def set_malloc_heap_size(size=None):
    """
    Heap size control for strings_udf, size in bytes.
    """
    global _HEAP_SIZE
    if size is None:
        size = _STRINGS_UDF_DEFAULT_HEAP_SIZE
    if size != _HEAP_SIZE:
        (ret,) = runtime.cudaDeviceSetLimit(
            runtime.cudaLimit.cudaLimitMallocHeapSize, size
        )
        if ret.value != 0:
            raise RuntimeError("Unable to set cudaMalloc heap size")

        _HEAP_SIZE = size


def column_to_string_view_array_init_heap(col: plc.Column) -> Buffer:
    # lazily allocate heap only when a string needs to be returned
    return as_buffer(strings_udf.column_to_string_view_array(col))


class UDFError(RuntimeError):
    pass
