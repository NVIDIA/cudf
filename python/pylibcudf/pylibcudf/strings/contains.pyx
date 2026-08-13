# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move
from libcpp.string cimport string

from pylibcudf.column cimport Column
from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.column.column_view cimport column_view
from pylibcudf.libcudf.strings cimport contains as cpp_contains
from pylibcudf.strings.regex_program cimport RegexProgram
from pylibcudf.utils cimport _get_stream, _get_memory_resource
from rmm.pylibrmm.memory_resource cimport DeviceMemoryResource
from rmm.pylibrmm.stream cimport Stream
from cuda.bindings.cyruntime cimport cudaStream_t

__all__ = ["contains_re", "count_re", "like", "matches_re"]

cpdef Column contains_re(
    Column input,
    RegexProgram prog,
    object stream=None,
    DeviceMemoryResource mr=None,
):
    """Returns a boolean column identifying rows which match the given
    regex_program object.

    For details, see :cpp:func:`contains_re`.

    Parameters
    ----------
    input : Column
        The input strings
    prog : RegexProgram
        Regex program instance

    Returns
    -------
    pylibcudf.Column
        New column of boolean results for each string
    """

    cdef unique_ptr[column] result
    cdef Stream _stream = _get_stream(stream)
    cdef cudaStream_t _cs = _stream.view().value()
    if _stream is None:
        _stream = _get_stream(None)
    mr = _get_memory_resource(mr)
    cdef column_view c_input = input.view()
    with nogil:
        result = cpp_contains.contains_re(
            c_input,
            prog.c_obj.get()[0],
            _cs,
            mr.get_mr()
        )

    return Column.from_libcudf(move(result), _stream, mr)


cpdef Column count_re(
    Column input,
    RegexProgram prog,
    object stream=None,
    DeviceMemoryResource mr=None,
):
    """Returns the number of times the given regex_program's pattern
    matches in each string.

    For details, see :cpp:func:`count_re`.

    Parameters
    ----------
    input : Column
        The input strings
    prog : RegexProgram
        Regex program instance

    Returns
    -------
    pylibcudf.Column
        New column of match counts for each string
    """

    cdef unique_ptr[column] result
    cdef Stream _stream = _get_stream(stream)
    cdef cudaStream_t _cs = _stream.view().value()
    mr = _get_memory_resource(mr)
    cdef column_view c_input = input.view()
    with nogil:
        result = cpp_contains.count_re(
            c_input,
            prog.c_obj.get()[0],
            _cs,
            mr.get_mr()
        )

    return Column.from_libcudf(move(result), _stream, mr)


cpdef Column matches_re(
    Column input,
    RegexProgram prog,
    object stream=None,
    DeviceMemoryResource mr=None,
):
    """Returns a boolean column identifying rows which
    matching the given regex_program object but only at
    the beginning the string.

    For details, see :cpp:func:`matches_re`.

    Parameters
    ----------
    input : Column
        The input strings
    prog : RegexProgram
        Regex program instance

    Returns
    -------
    pylibcudf.Column
        New column of boolean results for each string
    """

    cdef unique_ptr[column] result
    cdef Stream _stream = _get_stream(stream)
    cdef cudaStream_t _cs = _stream.view().value()
    mr = _get_memory_resource(mr)
    cdef column_view c_input = input.view()
    with nogil:
        result = cpp_contains.matches_re(
            c_input,
            prog.c_obj.get()[0],
            _cs,
            mr.get_mr()
        )

    return Column.from_libcudf(move(result), _stream, mr)


cpdef Column like(
    Column input,
    str pattern,
    str escape_character=None,
    object stream=None,
    DeviceMemoryResource mr=None,
):
    """
    Returns a boolean column identifying rows which
    match the given like pattern.

    For details, see :cpp:func:`like`.

    Parameters
    ----------
    input : Column
        The input strings
    pattern : str
        Like pattern to match within each string
    escape_character : str
        Optional character specifies the escape prefix.
        Default is no escape character.

    Returns
    -------
    pylibcudf.Column
        New column of boolean results for each string
    """
    cdef unique_ptr[column] result
    cdef Stream _stream = _get_stream(stream)
    cdef cudaStream_t _cs = _stream.view().value()
    mr = _get_memory_resource(mr)
    cdef column_view c_input

    if escape_character is None:
        escape_character = ""

    cdef string c_escape_character = escape_character.encode()
    cdef string c_pattern = pattern.encode()

    c_input = input.view()
    with nogil:
        result = cpp_contains.like(
            c_input,
            c_pattern,
            c_escape_character,
            _cs,
            mr.get_mr()
        )
    _stream.synchronize()

    return Column.from_libcudf(move(result), _stream, mr)
