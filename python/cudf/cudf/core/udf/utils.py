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
from numba_cuda_mlir import cuda, models
from numba_cuda_mlir.models import register_model
from numba_cuda_mlir.numba_cuda import types as nb_types
from numba_cuda_mlir.numba_cuda.descriptor import cuda_target
from numba_cuda_mlir.numba_cuda.np import numpy_support
from numba_cuda_mlir.numba_cuda.typing.typeof import typeof
from numba_cuda_mlir.types import CPointer, Record, Tuple, int64, void

import rmm

from cudf.api.types import is_string_dtype as _is_string_dtype
from cudf.core.udf.nrt_utils import nrt_enabled
from cudf.core.udf.strings_typing import (
    ManagedStrArrayWrapper,
    MLIRStringType,
    NRT_decref,
    mlir_string,
    mlir_string_arg_handler,
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


SUPPORTED_MASKED_TYPES = (
    nb_types.Number,
    nb_types.Boolean,
    nb_types.NPDatetime,
    nb_types.NPTimedelta,
    MLIRStringType,
)

_units = ("ns", "us", "ms", "s")
_supported_masked_types = (
    nb_types.integer_domain
    | nb_types.real_domain
    | {nb_types.NPDatetime(u) for u in _units}
    | {nb_types.NPTimedelta(u) for u in _units}
    | {nb_types.boolean}
    | {mlir_string}
)

# Maximum size of a string column is 2 GiB
_STRINGS_UDF_DEFAULT_HEAP_SIZE = int(
    os.environ.get("STRINGS_UDF_HEAP_SIZE", 2**31)
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


DEPRECATED_SM_REGEX = "Architectures prior to '<compute/sm>_75' are deprecated"


def _all_dtypes_from_frame(frame, supported_types=JIT_SUPPORTED_TYPES):
    # String columns are modeled as object in the row struct (row_function
    # maps object -> mlir_string). Current-main string dtypes (e.g.
    # StringDtype(na_value=nan)) are not np.dtype-constructible, so normalize
    # them to object even though they are "supported".
    return {
        colname: dtype
        if (str(dtype) in supported_types and not _is_string_dtype(dtype))
        else np.dtype("O")
        for colname, dtype in frame._dtypes
    }


def _supported_dtypes_from_frame(frame, supported_types=JIT_SUPPORTED_TYPES):
    return {
        colname: dtype
        for colname, dtype in frame._dtypes
        if str(dtype) in supported_types
    }


def _supported_cols_from_frame(frame, supported_types=JIT_SUPPORTED_TYPES):
    return {
        colname: col
        for colname, col in frame._column_labels_and_values
        if str(col.dtype) in supported_types
    }


def _masked_array_type_from_col(col):
    """
    Return kernel arg type per column: Array for unmasked, Tuple(data, mask) for masked.
    Matches original numba extension: unmasked -> single array, masked -> unpacked (d, m).
    """
    if _is_string_dtype(col.dtype):
        col_type = CPointer(mlir_string)
    else:
        nb_scalar_ty = numpy_support.from_dtype(col.dtype)
        col_type = nb_scalar_ty[::1]

    if col.mask is None:
        return col_type
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

# Also register with numba's native model system for the numba-cuda backend.
try:
    from numba_cuda_mlir.numba_cuda.datamodel import models as numba_models
    from numba_cuda_mlir.numba_cuda.extending import (
        register_model as numba_register_model,
    )

    numba_register_model(Row)(numba_models.RecordModel)
except (ImportError, AttributeError):
    pass


def _mask_get_impl(mask, pos):
    """Return the validity of mask[pos] as a bool (raw impl for JIT)."""
    return bool((mask[pos // MASK_BITSIZE] >> (pos % MASK_BITSIZE)) & 1)


# numba_cuda_mlir-jitted device function for kernel exec context
_mask_get = cuda.jit(device=True)(_mask_get_impl)


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
    from cudf.core.udf.masked_typing import MaskedType

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


def _buffer_as_dtyped_view(data: "Buffer", n: int, dtype: np.dtype):
    """View a Buffer as a cupy array with shape (n,) and dtype so the kernel indexes by element.

    Buffer's cuda_array_interface uses shape=(size,) and typestr='|u1' (bytes), so passing it
    directly makes d_0[i] load the i-th byte instead of the i-th element. This view gives the
    kernel the correct dtype and shape so d_0[i + offset] is the (i+offset)-th element.
    """
    with data.access(mode="read"):
        ptr = data.ptr
    size_bytes = data.size
    memptr = cp.cuda.MemoryPointer(
        cp.cuda.UnownedMemory(ptr, size_bytes, None), 0
    )
    return cp.ndarray((n,), dtype=dtype, memptr=memptr)


def _get_input_args_from_frame(fr: IndexedFrame) -> list:
    args: list[Buffer | tuple[Buffer, Buffer]] = []
    offsets = []
    for col in _supported_cols_from_frame(fr).values():
        if _is_string_dtype(col.dtype):
            data = ManagedStrArrayWrapper(
                column_to_mlir_string_array_init_heap(col.plc_column)
            )
        else:
            # View buffer with column dtype/shape so kernel sees d_0[i] as i-th element, not i-th byte
            data = _buffer_as_dtyped_view(col.data, len(col), col.dtype)
        if col.mask is not None:
            # View mask buffer with LIBCUDF_BITMASK_TYPE so kernel gets expected dtype/shape
            mask_itemsize = np.dtype(SIZE_TYPE_DTYPE).itemsize
            mask_n = col.mask.size // mask_itemsize
            mask_view = _buffer_as_dtyped_view(
                col.mask, mask_n, SIZE_TYPE_DTYPE
            )
            args.append((data, mask_view))
        else:
            args.append(data)
        offsets.append(col.offset)

    return args + offsets


def _output_args_for_udf_kernel(ans_col, ans_mask, n):
    """Build output_args for kernel launch. Mask as writable array for numba_cuda_mlir kernel."""
    with ans_mask.data.access(mode="write"):
        ptr = ans_mask.data.ptr
    # NB: expose the byte-per-row validity mask to the kernel as int8 rather
    # than bool. Released numba-cuda-mlir's CUDA-array marshaller computes
    # ``itemsize = dtype.bitwidth`` unconditionally, and its ``Boolean`` type
    # has no ``bitwidth`` (latent upstream bug). int8 has identical byte
    # layout, so the kernel writes 0/1 into the same buffer cuDF reads as the
    # bool mask.
    ans_mask_arr = cp.ndarray(
        (n,),
        dtype=np.int8,
        memptr=cp.cuda.MemoryPointer(cp.cuda.UnownedMemory(ptr, n, None), 0),
    )
    if isinstance(ans_col, rmm.DeviceBuffer):
        ans_col = ManagedStrArrayWrapper(ans_col)
    return [(ans_col, ans_mask_arr), n]


def _return_arr_from_dtype(dtype, size):
    if _is_string_dtype(dtype):
        return rmm.DeviceBuffer(size=size * _get_extensionty_size(mlir_string))
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
                void(CPointer(mlir_string), int64),
                extensions=[mlir_string_arg_handler],
            )
            def free_mlir_string_array(ary, size):
                gid = cuda.grid(1)
                if gid < size:
                    NRT_decref(ary[gid])

    return free_mlir_string_array


# The only supported data layout in NVVM.
# See: https://docs.nvidia.com/cuda/nvvm-ir-spec/index.html?#data-layout
_nvvm_data_layout = (
    "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-"
    "i128:128:128-f32:32:32-f64:64:64-v16:16:16-v32:32:32-"
    "v64:64:64-v128:128:128-n16:32:64"
)


def _get_extensionty_size(ty):
    """
    Return the size of an extension type in bytes.
    MLIR-path types (MLIRStringType, GroupType) define
    _extensionty_size and are handled here without using the Numba/CUDA
    data model.
    """
    size = getattr(type(ty), "_extensionty_size", None)
    if size is not None:
        return size
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


def column_to_mlir_string_array_init_heap(col: plc.Column) -> Buffer:
    set_malloc_heap_size()
    return _column_to_mlir_string_array(col)


def _column_to_mlir_string_array(plc_col: plc.Column) -> rmm.DeviceBuffer:
    """Build a device array of mlir_string from a pylibcudf string column.

    Each element is {meminfo=null, data=chars+offset[i], nbytes=offset[i+1]-offset[i]}.
    Pure Python + cupy, no C++/Cython.
    """
    import pylibcudf as plc

    n = plc_col.size()
    if n == 0:
        return rmm.DeviceBuffer(size=0)

    offsets_data = plc_col.child(0).data()
    chars_data = plc_col.data()
    offsets_ptr = offsets_data.ptr
    chars_ptr = chars_data.ptr if chars_data is not None else 0

    offset_type_id = plc_col.child(0).type().id()
    if offset_type_id == plc.TypeId.INT64:
        offset_dtype = cp.int64
        offset_itemsize = 8
    else:
        offset_dtype = cp.int32
        offset_itemsize = 4

    offsets_cp = cp.ndarray(
        (n + 1,),
        dtype=offset_dtype,
        memptr=cp.cuda.MemoryPointer(
            cp.cuda.UnownedMemory(
                offsets_ptr, (n + 1) * offset_itemsize, None
            ),
            0,
        ),
    )

    starts = offsets_cp[:-1].astype(cp.int64)
    lengths = (offsets_cp[1:] - offsets_cp[:-1]).astype(cp.int64)
    data_ptrs = starts + chars_ptr

    out = cp.zeros((n, 3), dtype=cp.int64)
    out[:, 1] = data_ptrs
    out[:, 2] = lengths

    return rmm.DeviceBuffer(ptr=out.data.ptr, size=out.nbytes)


def _mlir_string_array_to_column(buf: rmm.DeviceBuffer, n: int) -> plc.Column:
    """Build a pylibcudf string column from a device array of mlir_string.

    Reads {meminfo, data, nbytes} triples, computes offsets via cupy prefix
    sum, scatters chars into a contiguous buffer, and assembles a plc.Column.
    Mask is applied separately by the caller.
    Pure Python + cupy + one scatter kernel, no C++/Cython.
    """
    import pylibcudf as plc
    from pylibcudf.gpumemoryview import gpumemoryview

    if n == 0:
        offsets_cp = cp.zeros(1, dtype=cp.int32)
        offsets_gmv = gpumemoryview(offsets_cp)
        offsets_col = plc.Column(
            plc.DataType(plc.TypeId.INT32),
            1,
            offsets_gmv,
            None,
            0,
            0,
            [],
        )
        chars_buf = rmm.DeviceBuffer(size=0)
        chars_gmv = gpumemoryview(chars_buf)
        return plc.Column(
            plc.DataType(plc.TypeId.STRING),
            0,
            chars_gmv,
            None,
            0,
            0,
            [offsets_col],
        )

    raw = cp.ndarray(
        (n, 3),
        dtype=cp.int64,
        memptr=cp.cuda.MemoryPointer(
            cp.cuda.UnownedMemory(buf.ptr, buf.size, None), 0
        ),
    )
    data_ptrs = cp.ascontiguousarray(raw[:, 1])
    lengths = cp.ascontiguousarray(raw[:, 2]).astype(cp.int32)

    offsets_cp = cp.zeros(n + 1, dtype=cp.int32)
    cp.cumsum(lengths, out=offsets_cp[1:])
    total_chars = int(offsets_cp[n])

    chars_dev = rmm.DeviceBuffer(size=max(total_chars, 1))
    chars_cp = cp.ndarray(
        (max(total_chars, 1),),
        dtype=cp.uint8,
        memptr=cp.cuda.MemoryPointer(
            cp.cuda.UnownedMemory(chars_dev.ptr, max(total_chars, 1), None), 0
        ),
    )

    _scatter_chars_kernel(data_ptrs, lengths, offsets_cp, chars_cp, n)

    offsets_gmv = gpumemoryview(offsets_cp)
    offsets_col = plc.Column(
        plc.DataType(plc.TypeId.INT32),
        n + 1,
        offsets_gmv,
        None,
        0,
        0,
        [],
    )
    chars_gmv = gpumemoryview(chars_dev)

    return plc.Column(
        plc.DataType(plc.TypeId.STRING),
        n,
        chars_gmv,
        None,
        0,
        0,
        [offsets_col],
    )


def _scatter_chars_kernel(data_ptrs, lengths, offsets, chars_out, n):
    """Copy each string's bytes into the contiguous chars buffer.

    Uses a simple cupy raw kernel: thread i copies lengths[i] bytes from
    data_ptrs[i] to chars_out + offsets[i].
    """
    if n == 0:
        return
    kernel = cp.RawKernel(
        r"""
    extern "C" __global__
    void scatter_chars(
        const long long* data_ptrs,
        const int* lengths,
        const int* offsets,
        unsigned char* chars_out,
        long long n
    ) {
        long long gid = blockIdx.x * blockDim.x + threadIdx.x;
        if (gid < n) {
            const unsigned char* src = (const unsigned char*)data_ptrs[gid];
            int len = lengths[gid];
            int off = offsets[gid];
            for (int j = 0; j < len; j++) {
                chars_out[off + j] = src[j];
            }
        }
    }
    """,
        "scatter_chars",
    )
    block = 256
    grid = (n + block - 1) // block
    kernel((grid,), (block,), (data_ptrs, lengths, offsets, chars_out, n))


class UDFError(RuntimeError):
    pass
