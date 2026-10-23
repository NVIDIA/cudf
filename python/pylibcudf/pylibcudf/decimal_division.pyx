# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move
from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.column.column_view cimport column_view
from pylibcudf.libcudf.fixed_point.math_ops cimport (
    decimal_intermediate,
    divide_decimal as cpp_divide_decimal,
)

from pylibcudf.libcudf.fixed_point.math_ops import \
    decimal_intermediate as DecimalIntermediate  # no-cython-lint

from pylibcudf.libcudf.round cimport rounding_method
from pylibcudf.libcudf.scalar.scalar cimport scalar

from rmm.pylibrmm.stream cimport Stream
from rmm.pylibrmm.memory_resource cimport DeviceMemoryResource

from .column cimport Column
from .scalar cimport Scalar
from .traits cimport is_fixed_point
from .types cimport DataType
from .utils cimport _get_stream, _get_memory_resource
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pylibcudf.typing import CudaStreamLike
from cuda.bindings.cyruntime cimport cudaStream_t

__all__ = [
    "DecimalIntermediate",
    "divide_decimal",
    "divide_decimal_column_scalar",
    "divide_decimal_scalar_column",
]


cdef void _check_decimal(object operand, str side) except *:
    if operand is None:
        raise TypeError(f"{side} operand must not be None")
    typ = operand.type()
    if not is_fixed_point(typ):
        raise TypeError(
            f"{side} operand must be a decimal type, got {typ.id()!r}"
        )


cdef void _check_output_type(DataType output_type) except *:
    if output_type is None:
        raise TypeError("output_type must not be None")
    if not is_fixed_point(output_type):
        raise TypeError(
            f"output_type must be a decimal type, got {output_type.id()!r}"
        )


cpdef Column divide_decimal(
    Column lhs,
    Column rhs,
    DataType output_type,
    rounding_method round_method = rounding_method.HALF_UP,
    decimal_intermediate intermediate = decimal_intermediate.STANDARD,
    object stream: CudaStreamLike | None = None,
    DeviceMemoryResource mr=None
):
    """Divides two decimal columns element-wise into a column of ``output_type``.

    For details, see :cpp:func:`divide_decimal`.

    Parameters
    ----------
    lhs : Column
        The dividend column. Must be a decimal type.
    rhs : Column
        The divisor column. Must be a decimal type, of any width and scale.
    output_type : DataType
        The decimal type and scale of the result. Results with more digits
        than this type can hold are null.
    round_method : RoundingMethod, optional
        How to round the quotient to the scale of ``output_type``.
        Can be one of { RoundingMethod.HALF_UP, RoundingMethod.HALF_EVEN }
        (default RoundingMethod.HALF_UP)
    intermediate : DecimalIntermediate, optional
        The width of the intermediate value used to rescale the dividend.
        With DecimalIntermediate.STANDARD, a result is also null if the
        rescaled dividend overflows the widest of the input and output types.
        DecimalIntermediate.WIDE uses 256 bits, so a result is null only for
        a zero divisor or a quotient that does not fit ``output_type``.
        (default DecimalIntermediate.STANDARD)
    stream : Stream | None
        CUDA stream on which to perform the operation.
    mr : DeviceMemoryResource | None
        Device memory resource used to allocate the returned column's device memory.

    Returns
    -------
    Column
        The quotients, with type ``output_type``.

    Raises
    ------
    TypeError
        If either operand or ``output_type`` is None or is not a decimal type.
    """
    _check_decimal(lhs, "Left")
    _check_decimal(rhs, "Right")
    _check_output_type(output_type)

    cdef unique_ptr[column] c_result
    cdef Stream _stream = _get_stream(stream)
    cdef cudaStream_t _cs = _stream.view().get()
    mr = _get_memory_resource(mr)

    cdef column_view c_lhs = lhs.view()
    cdef column_view c_rhs = rhs.view()
    with nogil:
        c_result = cpp_divide_decimal(
            c_lhs,
            c_rhs,
            output_type.c_obj,
            round_method,
            intermediate,
            _cs,
            mr.get_mr()
        )

    return Column.from_libcudf(move(c_result), _stream, mr)


cpdef Column divide_decimal_column_scalar(
    Column lhs,
    Scalar rhs,
    DataType output_type,
    rounding_method round_method = rounding_method.HALF_UP,
    decimal_intermediate intermediate = decimal_intermediate.STANDARD,
    object stream: CudaStreamLike | None = None,
    DeviceMemoryResource mr=None
):
    """Divides a decimal column by a decimal scalar into a column of ``output_type``.

    For details, see :cpp:func:`divide_decimal`.

    Parameters
    ----------
    lhs : Column
        The dividend column. Must be a decimal type.
    rhs : Scalar
        The divisor. Must be a decimal type, of any width and scale.
    output_type : DataType
        The decimal type and scale of the result. Results with more digits
        than this type can hold are null.
    round_method : RoundingMethod, optional
        How to round the quotient to the scale of ``output_type``.
        Can be one of { RoundingMethod.HALF_UP, RoundingMethod.HALF_EVEN }
        (default RoundingMethod.HALF_UP)
    intermediate : DecimalIntermediate, optional
        The width of the intermediate value used to rescale the dividend.
        With DecimalIntermediate.STANDARD, a result is also null if the
        rescaled dividend overflows the widest of the input and output types.
        DecimalIntermediate.WIDE uses 256 bits, so a result is null only for
        a zero divisor or a quotient that does not fit ``output_type``.
        (default DecimalIntermediate.STANDARD)
    stream : Stream | None
        CUDA stream on which to perform the operation.
    mr : DeviceMemoryResource | None
        Device memory resource used to allocate the returned column's device memory.

    Returns
    -------
    Column
        The quotients, with type ``output_type``.

    Raises
    ------
    TypeError
        If either operand or ``output_type`` is None or is not a decimal type.
    """
    _check_decimal(lhs, "Left")
    _check_decimal(rhs, "Right")
    _check_output_type(output_type)

    cdef unique_ptr[column] c_result
    cdef Stream _stream = _get_stream(stream)
    cdef cudaStream_t _cs = _stream.view().get()
    mr = _get_memory_resource(mr)

    cdef column_view c_lhs = lhs.view()
    cdef const scalar* c_rhs = rhs.get()
    with nogil:
        c_result = cpp_divide_decimal(
            c_lhs,
            c_rhs[0],
            output_type.c_obj,
            round_method,
            intermediate,
            _cs,
            mr.get_mr()
        )

    return Column.from_libcudf(move(c_result), _stream, mr)


cpdef Column divide_decimal_scalar_column(
    Scalar lhs,
    Column rhs,
    DataType output_type,
    rounding_method round_method = rounding_method.HALF_UP,
    decimal_intermediate intermediate = decimal_intermediate.STANDARD,
    object stream: CudaStreamLike | None = None,
    DeviceMemoryResource mr=None
):
    """Divides a decimal scalar by a decimal column into a column of ``output_type``.

    For details, see :cpp:func:`divide_decimal`.

    Parameters
    ----------
    lhs : Scalar
        The dividend. Must be a decimal type.
    rhs : Column
        The divisor column. Must be a decimal type, of any width and scale.
    output_type : DataType
        The decimal type and scale of the result. Results with more digits
        than this type can hold are null.
    round_method : RoundingMethod, optional
        How to round the quotient to the scale of ``output_type``.
        Can be one of { RoundingMethod.HALF_UP, RoundingMethod.HALF_EVEN }
        (default RoundingMethod.HALF_UP)
    intermediate : DecimalIntermediate, optional
        The width of the intermediate value used to rescale the dividend.
        With DecimalIntermediate.STANDARD, a result is also null if the
        rescaled dividend overflows the widest of the input and output types.
        DecimalIntermediate.WIDE uses 256 bits, so a result is null only for
        a zero divisor or a quotient that does not fit ``output_type``.
        (default DecimalIntermediate.STANDARD)
    stream : Stream | None
        CUDA stream on which to perform the operation.
    mr : DeviceMemoryResource | None
        Device memory resource used to allocate the returned column's device memory.

    Returns
    -------
    Column
        The quotients, with type ``output_type``.

    Raises
    ------
    TypeError
        If either operand or ``output_type`` is None or is not a decimal type.
    """
    _check_decimal(lhs, "Left")
    _check_decimal(rhs, "Right")
    _check_output_type(output_type)

    cdef unique_ptr[column] c_result
    cdef Stream _stream = _get_stream(stream)
    cdef cudaStream_t _cs = _stream.view().get()
    mr = _get_memory_resource(mr)

    cdef const scalar* c_lhs = lhs.get()
    cdef column_view c_rhs = rhs.view()
    with nogil:
        c_result = cpp_divide_decimal(
            c_lhs[0],
            c_rhs,
            output_type.c_obj,
            round_method,
            intermediate,
            _cs,
            mr.get_mr()
        )

    return Column.from_libcudf(move(c_result), _stream, mr)
