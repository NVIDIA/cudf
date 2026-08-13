# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move
from pylibcudf.column cimport Column
from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.column.column_view cimport column_view
from pylibcudf.libcudf.strings cimport attributes as cpp_attributes
from pylibcudf.utils cimport _get_stream, _get_memory_resource
from rmm.pylibrmm.memory_resource cimport DeviceMemoryResource
from rmm.pylibrmm.stream cimport Stream
from cuda.bindings.cyruntime cimport cudaStream_t

__all__ = ["code_points", "count_bytes", "count_characters"]

cpdef Column count_characters(
    Column source_strings, object stream=None, DeviceMemoryResource mr=None
):
    """
    Returns a column containing character lengths of each string
    in the given column.

    Parameters
    ----------
    source_strings : Column
        Column of strings.
    stream : Stream | None
        CUDA stream on which to perform the operation.

    Returns
    -------
    Column
        New column with lengths for each string
    """
    cdef unique_ptr[column] c_result
    cdef Stream _stream = _get_stream(stream)
    cdef cudaStream_t _cs = _stream.view().value()
    mr = _get_memory_resource(mr)
    cdef column_view c_source_strings = source_strings.view()
    with nogil:
        c_result = cpp_attributes.count_characters(
            c_source_strings, _cs, mr.get_mr()
        )

    return Column.from_libcudf(move(c_result), _stream, mr)


cpdef Column count_bytes(
    Column source_strings, object stream=None, DeviceMemoryResource mr=None
):
    """
    Returns a column containing byte lengths of each string
    in the given column.

    Parameters
    ----------
    source_strings : Column
        Column of strings.
    stream : Stream | None
        CUDA stream on which to perform the operation.

    Returns
    -------
    Column
        New column with the number of bytes for each string
    """
    cdef unique_ptr[column] c_result
    cdef Stream _stream = _get_stream(stream)
    cdef cudaStream_t _cs = _stream.view().value()
    mr = _get_memory_resource(mr)
    cdef column_view c_source_strings = source_strings.view()
    with nogil:
        c_result = cpp_attributes.count_bytes(
            c_source_strings, _cs, mr.get_mr()
        )

    return Column.from_libcudf(move(c_result), _stream, mr)


cpdef Column code_points(
    Column source_strings, object stream=None, DeviceMemoryResource mr=None
):
    """
    Creates a numeric column with code point values (integers)
    for each character of each string.

    Parameters
    ----------
    source_strings : Column
        Column of strings.
    stream : Stream | None
        CUDA stream on which to perform the operation.

    Returns
    -------
    Column
        New column with code point integer values for each character
    """
    cdef unique_ptr[column] c_result
    cdef Stream _stream = _get_stream(stream)
    cdef cudaStream_t _cs = _stream.view().value()
    mr = _get_memory_resource(mr)
    cdef column_view c_source_strings = source_strings.view()
    with nogil:
        c_result = cpp_attributes.code_points(
            c_source_strings, _cs, mr.get_mr()
        )

    return Column.from_libcudf(move(c_result), _stream, mr)
