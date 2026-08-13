# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move
from pylibcudf.column cimport Column
from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.column.column_view cimport column_view
from pylibcudf.libcudf.strings cimport findall as cpp_findall
from pylibcudf.strings.regex_program cimport RegexProgram
from pylibcudf.utils cimport _get_stream, _get_memory_resource
from rmm.pylibrmm.memory_resource cimport DeviceMemoryResource
from rmm.pylibrmm.stream cimport Stream
from cuda.bindings.cyruntime cimport cudaStream_t

__all__ = ["findall", "find_re"]

cpdef Column findall(
    Column input, RegexProgram pattern, object stream=None, DeviceMemoryResource mr=None
):
    """
    Returns a lists column of strings for each matching occurrence using
    the regex_program pattern within each string.

    For details, see :cpp:func:`findall`.

    Parameters
    ----------
    input : Column
        Strings instance for this operation
    pattern : RegexProgram
        Regex pattern
    stream : Stream | None
        CUDA stream on which to perform the operation.

    Returns
    -------
    Column
        New lists column of strings
    """
    cdef unique_ptr[column] c_result
    cdef Stream _stream = _get_stream(stream)
    cdef cudaStream_t _cs = _stream.view().value()
    mr = _get_memory_resource(mr)
    cdef column_view c_input = input.view()
    with nogil:
        c_result = cpp_findall.findall(
            c_input,
            pattern.c_obj.get()[0],
            _cs,
            mr.get_mr()
        )

    return Column.from_libcudf(move(c_result), _stream, mr)


cpdef Column find_re(
    Column input, RegexProgram pattern, object stream=None, DeviceMemoryResource mr=None
):
    """
    Returns character positions where the pattern first matches
    the elements in input strings.

    For details, see :cpp:func:`find_re`

    Parameters
    ----------
    input : Column
        Strings instance for this operation
    pattern : RegexProgram
        Regex pattern
    stream : Stream | None
        CUDA stream on which to perform the operation.

    Returns
    -------
    Column
        New column of integers
    """
    cdef unique_ptr[column] c_result
    cdef Stream _stream = _get_stream(stream)
    cdef cudaStream_t _cs = _stream.view().value()
    mr = _get_memory_resource(mr)
    cdef column_view c_input = input.view()
    with nogil:
        c_result = cpp_findall.find_re(
            c_input,
            pattern.c_obj.get()[0],
            _cs,
            mr.get_mr()
        )

    return Column.from_libcudf(move(c_result), _stream, mr)
