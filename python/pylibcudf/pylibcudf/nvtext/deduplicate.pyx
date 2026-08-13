# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from cython.operator import dereference

from libcpp.memory cimport unique_ptr, make_unique
from libcpp.utility cimport move
from pylibcudf.column cimport Column
from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.column.column_view cimport column_view
from pylibcudf.libcudf.nvtext.deduplicate cimport (
    build_suffix_array as cpp_build_suffix_array,
    suffix_array_type as cpp_suffix_array_type,
    resolve_duplicates as cpp_resolve_duplicates,
    resolve_duplicates_pair as cpp_resolve_duplicates_pair,
)
from pylibcudf.libcudf.types cimport size_type
from pylibcudf.utils cimport _get_stream, _get_memory_resource
from rmm.pylibrmm.memory_resource cimport DeviceMemoryResource
from rmm.librmm.device_buffer cimport device_buffer
from rmm.pylibrmm.stream cimport Stream
from cuda.bindings.cyruntime cimport cudaStream_t

__all__ = [
    "build_suffix_array",
    "resolve_duplicates",
    "resolve_duplicates_pair",
]

cdef Column _column_from_suffix_array(
    cpp_suffix_array_type suffix_array, Stream stream, DeviceMemoryResource mr
):
    # helper to convert a suffix array to a Column
    return Column.from_libcudf(
        move(
            make_unique[column](
                move(dereference(suffix_array.get())),
                device_buffer(),
                0
            )
        ), stream, mr
    )


cpdef Column build_suffix_array(
    Column input, size_type min_width, object stream=None, DeviceMemoryResource mr=None
):
    """
    Builds a suffix array for the input strings column.
    A suffix array is the indices of the sorted set of substrings
    of the input column as: [ input[0:], input[1:], ... input[bytes-1:] ]
    where bytes is the total number of bytes in input.
    The returned array represent the sorted strings such that
    result[i] = input[result[i]:]

    For details, see :cpp:func:`build_suffix_array`

    Parameters
    ----------
    input : Column
        Strings column of text
    stream : Stream | None
        CUDA stream on which to perform the operation.

    Returns
    -------
    Column
        New column of suffix array
    """
    cdef cpp_suffix_array_type c_result
    cdef Stream _stream = _get_stream(stream)
    cdef cudaStream_t _cs = _stream.view().value()
    mr = _get_memory_resource(mr)

    cdef column_view c_input = input.view()
    with nogil:
        c_result = cpp_build_suffix_array(
            c_input, min_width, _cs, mr.get_mr()
        )

    return _column_from_suffix_array(move(c_result), _stream, mr)


cpdef Column resolve_duplicates(
    Column input,
    Column indices,
    size_type min_width,
    object stream=None,
    DeviceMemoryResource mr=None,
):
    """
    Returns duplicate strings found in the input column
    with min_width minimum number of bytes.
    The indices are expected to be the suffix array previously created
    for input. Otherwise, the results are undefined.

    For details, see :cpp:func:`resolve_duplicates`

    Parameters
    ----------
    input : Column
        Strings column of text
    indices : Column
        Suffix array from :cpp:func:`build_suffix_array`
    min_width : size_type
        Minimum width of bytes to detect duplicates
    stream : Stream | None
        CUDA stream on which to perform the operation.

    Returns
    -------
    Column
        New column of duplicate strings
    """
    cdef unique_ptr[column] c_result
    cdef Stream _stream = _get_stream(stream)
    cdef cudaStream_t _cs = _stream.view().value()
    mr = _get_memory_resource(mr)

    cdef column_view c_input = input.view()
    cdef column_view c_indices = indices.view()
    with nogil:
        c_result = cpp_resolve_duplicates(
            c_input, c_indices, min_width, _cs, mr.get_mr()
        )

    return Column.from_libcudf(move(c_result), _stream, mr)


cpdef Column resolve_duplicates_pair(
    Column input1,
    Column indices1,
    Column input2,
    Column indices2,
    size_type min_width,
    object stream=None,
    DeviceMemoryResource mr=None,
):
    """
    Returns duplicate strings in input1 found in input2
    with min_width minimum number of bytes.
    The indices are expected to be the suffix array previously created
    for the inputs. Otherwise, the results are undefined.

    For details, see :cpp:func:`resolve_duplicates_pair`

    Parameters
    ----------
    input1 : Column
        Strings column of text
    indices1 : Column
        Suffix array from :cpp:func:`build_suffix_array` for input1
    input2 : Column
        Strings column of text
    indices2 : Column
        Suffix array from :cpp:func:`build_suffix_array` for input2
    min_width : size_type
        Minimum width of bytes to detect duplicates
    stream : Stream | None
        CUDA stream on which to perform the operation.

    Returns
    -------
    Column
        New column of duplicate strings

    """
    cdef unique_ptr[column] c_result
    cdef Stream _stream = _get_stream(stream)
    cdef cudaStream_t _cs = _stream.view().value()
    mr = _get_memory_resource(mr)

    cdef column_view c_input1 = input1.view()
    cdef column_view c_indices1 = indices1.view()
    cdef column_view c_input2 = input2.view()
    cdef column_view c_indices2 = indices2.view()
    with nogil:
        c_result = cpp_resolve_duplicates_pair(
            c_input1,
            c_indices1,
            c_input2,
            c_indices2,
            min_width,
            _cs,
            mr.get_mr(),
        )

    return Column.from_libcudf(move(c_result), _stream, mr)
