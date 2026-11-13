# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compiled device functions and borrowed device pointers for row transforms."""

from dataclasses import dataclass
from enum import IntEnum
from functools import lru_cache

from pylibcudf.types import TypeId


class LTOBinaryType(IntEnum):
    """Binary format of a device module to link into a JIT kernel."""

    #: LTO-IR binary.
    LTO_IR = 0
    #: FATBIN binary.
    FATBIN = 1
    #: PTX source.
    PTX = 2


@dataclass(frozen=True)
class UserData:
    """A borrowed device pointer, passed as a raw ``void*`` UDF argument.

    Keep the allocation alive until work on the transform's stream completes.
    ``owner`` can retain the object owning that allocation. Zero is a valid pointer.
    """

    ptr: int = 0
    owner: object = None


@dataclass(frozen=True)
class TransformUDF:
    """An explicitly named transform function and the modules defining it.

    The CUDF ABI is ``cudf::errc symbol(Out* out, Inputs... inputs)``. The
    output pointer precedes all inputs. When indexed, a logical row index
    precedes the output pointer. Indices use ``cudf::size_type`` and are never
    optional. The function returns a signed 32-bit status matching ``cudf::errc``.

    Null-aware calls wrap column and output values in ``cuda::std::optional``;
    :class:`UserData` inputs remain raw ``void*`` values. Modules may export
    multiple symbols and reference symbols in other supplied modules.
    This descriptor retains the module bytes. CUDA C++ source is not accepted.

    Parameters
    ----------
    symbol : str
        Exact exported device symbol, expressible as a CUDA identifier.
    fragments : tuple[bytes, ...]
        Compiled device modules defining the function and its dependencies,
        in link order. Each module must contain nonempty bytes.
    binary_type : LTOBinaryType, default LTOBinaryType.LTO_IR
        Binary format of all supplied modules. PTX bytes need not include
        a null terminator.
    is_null_aware : bool, default False
        Whether column and output values use ``cuda::std::optional``.
    indexed : bool, default False
        Whether a logical row index precedes all other arguments.

    Raises
    ------
    TypeError
        If ``binary_type`` is not an :class:`LTOBinaryType`, or ``fragments``
        is empty or contains values other than nonempty ``bytes`` objects.
    """

    symbol: str
    fragments: tuple[bytes, ...]
    binary_type: LTOBinaryType = LTOBinaryType.LTO_IR
    is_null_aware: bool = False
    indexed: bool = False

    def __post_init__(self):
        if not isinstance(self.binary_type, LTOBinaryType):
            raise TypeError("binary_type must be an LTOBinaryType")
        if not self.fragments or any(
            not isinstance(fragment, bytes) or not fragment
            for fragment in self.fragments
        ):
            raise TypeError(
                "fragments must contain nonempty LTO IR, fatbin, or PTX bytes"
            )


@lru_cache(maxsize=128)
def _compile_transform(function, input_ids, output_id, _compute_capability):
    # Numba is optional and needed only for implicit callable compilation.
    from numba import cuda, types
    from numba.cuda.dispatcher import CUDADispatcher

    if hasattr(function, "py_func") and not isinstance(
        function, CUDADispatcher
    ):
        raise TypeError(
            "Implicit compilation requires a Numba CUDA dispatcher with the "
            "status-returning device ABI"
        )
    if hasattr(function, "targetoptions") and not function.targetoptions.get(
        "device", False
    ):
        raise TypeError(
            "transform_udf must be a CUDA device function (device=True)"
        )
    mapping = {
        TypeId.INT8: types.int8,
        TypeId.INT16: types.int16,
        TypeId.INT32: types.int32,
        TypeId.INT64: types.int64,
        TypeId.UINT8: types.uint8,
        TypeId.UINT16: types.uint16,
        TypeId.UINT32: types.uint32,
        TypeId.UINT64: types.uint64,
        TypeId.FLOAT32: types.float32,
        TypeId.FLOAT64: types.float64,
        TypeId.BOOL8: types.boolean,
        TypeId.TIMESTAMP_DAYS: types.int32,
        TypeId.TIMESTAMP_SECONDS: types.int64,
        TypeId.TIMESTAMP_MILLISECONDS: types.int64,
        TypeId.TIMESTAMP_MICROSECONDS: types.int64,
        TypeId.TIMESTAMP_NANOSECONDS: types.int64,
        TypeId.DURATION_DAYS: types.int32,
        TypeId.DURATION_SECONDS: types.int64,
        TypeId.DURATION_MILLISECONDS: types.int64,
        TypeId.DURATION_MICROSECONDS: types.int64,
        TypeId.DURATION_NANOSECONDS: types.int64,
    }
    try:
        inputs = tuple(
            types.voidptr if t is None else mapping[t] for t in input_ids
        )
        output = mapping[output_id]
    except KeyError as error:
        raise TypeError(
            "This column type cannot be compiled as a transform UDF"
        ) from error
    dispatcher = (
        function
        if isinstance(function, CUDADispatcher)
        else cuda.jit(device=True, lto=True)(function)
    )
    # Numba's device ABI returns the compiler-generated status separately from
    # the Python return value, which is written through the leading output pointer.
    result = dispatcher.compile_device(inputs, return_type=output)
    if result.signature.return_type != output:
        raise TypeError(
            "The compiled CUDA UDF return type must match output_type"
        )
    return TransformUDF(
        result.fndesc.llvm_func_name,
        (bytes(result.library.get_ltoir(cc=_compute_capability)),),
    )


def _as_transform_udf(function, input_ids, output_id, is_null_aware):
    if isinstance(function, TransformUDF):
        if is_null_aware and not function.is_null_aware:
            raise ValueError(
                "Null-aware transforms require a null-aware TransformUDF"
            )
        return function
    if is_null_aware:
        raise NotImplementedError(
            "Null-aware Python transforms require compiled CUDF ABI modules"
        )
    if not callable(function):
        raise TypeError("transform_udf must be a callable or TransformUDF")

    from numba import cuda

    # LTO IR can be reused across contexts, but compilation depends on the target architecture.
    return _compile_transform(
        function,
        tuple(input_ids),
        output_id,
        tuple(cuda.get_current_device().compute_capability),
    )
