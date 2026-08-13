# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from cython.operator import dereference

from libcpp cimport bool
from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move
from pylibcudf.libcudf cimport binaryop as cpp_binaryop
from pylibcudf.libcudf.binaryop cimport binary_operator
from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.column.column_view cimport column_view

from pylibcudf.libcudf.binaryop import \
    binary_operator as BinaryOperator  # no-cython-lint

from rmm.pylibrmm.memory_resource cimport DeviceMemoryResource
from rmm.pylibrmm.stream cimport Stream

from .column cimport Column
from .scalar cimport Scalar
from .types cimport DataType
from .utils cimport _get_stream, _get_memory_resource
from cuda.bindings.cyruntime cimport cudaStream_t

__all__ = ["BinaryOperator", "binary_operation", "is_supported_operation"]

cpdef Column binary_operation(
    LeftBinaryOperand lhs,
    RightBinaryOperand rhs,
    binary_operator op,
    DataType output_type,
    object stream=None,
    DeviceMemoryResource mr=None,
):
    """Perform a binary operation between a column and another column or scalar.

    ``lhs`` and ``rhs`` may be a
    :py:class:`~pylibcudf.column.Column` or a
    :py:class:`~pylibcudf.scalar.Scalar`, but at least one must be a
    :py:class:`~pylibcudf.column.Column`.

    For details, see :cpp:func:`binary_operation`.

    Parameters
    ----------
    lhs : Union[Column, Scalar]
        The left hand side argument.
    rhs : Union[Column, Scalar]
        The right hand side argument.
    op : BinaryOperator
        The operation to perform.
    output_type : DataType
        The data type to use for the output.
    stream : Stream | None
        CUDA stream on which to perform the operation.
    mr : DeviceMemoryResource | None
        Device memory resource used to allocate the returned column's device memory.

    Returns
    -------
    pylibcudf.Column
        The result of the binary operation
    """
    cdef unique_ptr[column] result
    cdef Stream _stream = _get_stream(stream)
    cdef cudaStream_t _cs = _stream.view().value()
    cdef column_view c_lhs_column
    cdef column_view c_rhs_column

    mr = _get_memory_resource(mr)

    if LeftBinaryOperand is Column and RightBinaryOperand is Column:
        c_lhs_column = lhs.view()
        c_rhs_column = rhs.view()
        with nogil:
            result = cpp_binaryop.binary_operation(
                c_lhs_column,
                c_rhs_column,
                op,
                output_type.c_obj,
                _cs,
                mr.get_mr()
            )
    elif LeftBinaryOperand is Column and RightBinaryOperand is Scalar:
        c_lhs_column = lhs.view()
        with nogil:
            result = cpp_binaryop.binary_operation(
                c_lhs_column,
                dereference(rhs.c_obj),
                op,
                output_type.c_obj,
                _cs,
                mr.get_mr()
            )
    elif LeftBinaryOperand is Scalar and RightBinaryOperand is Column:
        c_rhs_column = rhs.view()
        with nogil:
            result = cpp_binaryop.binary_operation(
                dereference(lhs.c_obj),
                c_rhs_column,
                op,
                output_type.c_obj,
                _cs,
                mr.get_mr()
            )
    else:
        raise ValueError(f"Invalid arguments {lhs} and {rhs}")

    return Column.from_libcudf(move(result), _stream, mr)


cpdef bool is_supported_operation(
    DataType out,
    DataType lhs,
    DataType rhs,
    binary_operator op
):
    """Check if an operation is supported for the given data types.

    For details, see :cpp:func:`is_supported_operation`.

    Parameters
    ----------
    out : DataType
        The output data type.
    lhs : DataType
        The left hand side data type.
    rhs : DataType
        The right hand side data type.
    op : BinaryOperator
        The operation to check.

    Returns
    -------
    bool
        True if the operation is supported, False otherwise
    """

    return cpp_binaryop.is_supported_operation(
        out.c_obj,
        lhs.c_obj,
        rhs.c_obj,
        op
    )

BinaryOperator.__str__ = BinaryOperator.__repr__
