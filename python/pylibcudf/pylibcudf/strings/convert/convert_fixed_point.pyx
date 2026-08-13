# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move
from pylibcudf.column cimport Column
from pylibcudf.libcudf.column.column cimport column, column_view
from pylibcudf.libcudf.strings.convert cimport (
    convert_fixed_point as cpp_fixed_point,
)
from pylibcudf.types cimport DataType, type_id
from pylibcudf.utils cimport _get_stream, _get_memory_resource
from rmm.pylibrmm.memory_resource cimport DeviceMemoryResource
from rmm.pylibrmm.stream cimport Stream
from cuda.bindings.cyruntime cimport cudaStream_t

__all__ = ["from_fixed_point", "is_fixed_point", "to_fixed_point"]


cpdef Column to_fixed_point(
    Column input, DataType output_type, object stream=None, DeviceMemoryResource mr=None
):
    """
    Returns a new fixed-point column parsing decimal values from the
    provided strings column.

    For details, see :cpp:func:`to_fixed_point`

    Parameters
    ----------
    input : Column
        Strings instance for this operation.

    output_type : DataType
        Type of fixed-point column to return including the scale value.

    stream : Stream | None
        CUDA stream on which to perform the operation.

    Returns
    -------
    Column
        New column of output_type.
    """
    cdef unique_ptr[column] c_result
    cdef Stream _stream = _get_stream(stream)
    cdef cudaStream_t _cs = _stream.view().value()
    mr = _get_memory_resource(mr)

    cdef column_view c_input = input.view()
    with nogil:
        c_result = cpp_fixed_point.to_fixed_point(
            c_input,
            output_type.c_obj,
            _cs,
            mr.get_mr()
        )

    return Column.from_libcudf(move(c_result), _stream, mr)

cpdef Column from_fixed_point(
    Column input, object stream=None, DeviceMemoryResource mr=None
):
    """
    Returns a new strings column converting the fixed-point values
    into a strings column.

    For details, see :cpp:func:`from_fixed_point`

    Parameters
    ----------
    input : Column
        Fixed-point column to convert.

    stream : Stream | None
        CUDA stream on which to perform the operation.

    Returns
    -------
    Column
        New strings column.
    """
    cdef unique_ptr[column] c_result
    cdef Stream _stream = _get_stream(stream)
    cdef cudaStream_t _cs = _stream.view().value()
    mr = _get_memory_resource(mr)

    cdef column_view c_input = input.view()
    with nogil:
        c_result = cpp_fixed_point.from_fixed_point(
            c_input, _cs, mr.get_mr()
        )

    return Column.from_libcudf(move(c_result), _stream, mr)

cpdef Column is_fixed_point(
    Column input,
    DataType decimal_type=None,
    object stream=None,
    DeviceMemoryResource mr=None,
):
    """
    Returns a boolean column identifying strings in which all
    characters are valid for conversion to fixed-point.

    For details, see :cpp:func:`is_fixed_point`

    Parameters
    ----------
    input : Column
        Strings instance for this operation.

    decimal_type : DataType
        Fixed-point type (with scale) used only for checking overflow.
        Defaults to Decimal64

    stream : Stream | None
        CUDA stream on which to perform the operation.

    Returns
    -------
    Column
        New column of boolean results for each string.
    """
    cdef unique_ptr[column] c_result
    cdef Stream _stream = _get_stream(stream)
    cdef cudaStream_t _cs = _stream.view().value()
    mr = _get_memory_resource(mr)

    if decimal_type is None:
        decimal_type = DataType(type_id.DECIMAL64)

    cdef column_view c_input = input.view()
    with nogil:
        c_result = cpp_fixed_point.is_fixed_point(
            c_input,
            decimal_type.c_obj,
            _cs,
            mr.get_mr()
        )

    return Column.from_libcudf(move(c_result), _stream, mr)
