# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move
from pylibcudf.column cimport Column
from pylibcudf.libcudf.column.column cimport column, column_view
from pylibcudf.libcudf.strings.convert cimport convert_ipv4 as cpp_convert_ipv4
from pylibcudf.utils cimport _get_stream, _get_memory_resource
from rmm.pylibrmm.memory_resource cimport DeviceMemoryResource
from rmm.pylibrmm.stream cimport Stream
from cuda.bindings.cyruntime cimport cudaStream_t

__all__ = ["integers_to_ipv4", "ipv4_to_integers", "is_ipv4"]

cpdef Column ipv4_to_integers(
    Column input, object stream=None, DeviceMemoryResource mr=None
):
    """
    Converts IPv4 addresses into integers.

    For details, see cpp:func:`cudf::strings::ipv4_to_integers`

    Parameters
    ----------
    input : Column
        Strings instance for this operation

    stream : Stream | None
        CUDA stream on which to perform the operation.

    Returns
    -------
    Column
        New uint32 column converted from strings.
    """
    cdef unique_ptr[column] c_result
    cdef Stream _stream = _get_stream(stream)
    cdef cudaStream_t _cs = _stream.view().value()
    mr = _get_memory_resource(mr)

    cdef column_view c_input = input.view()
    with nogil:
        c_result = cpp_convert_ipv4.ipv4_to_integers(
            c_input, _cs, mr.get_mr()
        )

    return Column.from_libcudf(move(c_result), _stream, mr)


cpdef Column integers_to_ipv4(
    Column integers, object stream=None, DeviceMemoryResource mr=None
):
    """
    Converts integers into IPv4 addresses as strings.

    For details, see cpp:func:`cudf::strings::integers_to_ipv4`

    Parameters
    ----------
    integers : Column
        Integer (uint32) column to convert.

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

    cdef column_view c_integers = integers.view()
    with nogil:
        c_result = cpp_convert_ipv4.integers_to_ipv4(
            c_integers, _cs, mr.get_mr()
        )

    return Column.from_libcudf(move(c_result), _stream, mr)


cpdef Column is_ipv4(Column input, object stream=None, DeviceMemoryResource mr=None):
    """
    Returns a boolean column identifying strings in which all
    characters are valid for conversion to integers from IPv4 format.

    For details, see cpp:func:`cudf::strings::is_ipv4`

    Parameters
    ----------
    input : Column
        Strings instance for this operation.

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

    cdef column_view c_input = input.view()
    with nogil:
        c_result = cpp_convert_ipv4.is_ipv4(c_input, _cs, mr.get_mr())

    return Column.from_libcudf(move(c_result), _stream, mr)
