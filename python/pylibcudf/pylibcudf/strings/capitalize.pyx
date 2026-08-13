# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move
from pylibcudf.column cimport Column
from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.column.column_view cimport column_view
from pylibcudf.libcudf.scalar.scalar cimport string_scalar
from pylibcudf.libcudf.scalar.scalar_factories cimport (
    make_string_scalar as cpp_make_string_scalar,
)
from pylibcudf.libcudf.strings cimport capitalize as cpp_capitalize
from pylibcudf.scalar cimport Scalar
from pylibcudf.strings.char_types cimport string_character_types
from pylibcudf.utils cimport _get_stream, _get_memory_resource
from rmm.pylibrmm.memory_resource cimport DeviceMemoryResource
from rmm.pylibrmm.stream cimport Stream

from cython.operator import dereference
from cuda.bindings.cyruntime cimport cudaStream_t

__all__ = ["capitalize", "is_title", "title"]

cpdef Column capitalize(
    Column input,
    Scalar delimiters=None,
    object stream=None,
    DeviceMemoryResource mr=None,
    # TODO: default scalar values
    # https://github.com/rapidsai/cudf/issues/15505
):
    """Returns a column of capitalized strings.

    For details, see :cpp:func:`capitalize`.

    Parameters
    ----------
    input : Column
        String column
    delimiters : Scalar, default None
        Characters for identifying words to capitalize

    Returns
    -------
    pylibcudf.Column
        Column of strings capitalized from the input column
    """
    cdef unique_ptr[column] c_result
    cdef Stream _stream = _get_stream(stream)
    cdef cudaStream_t _cs = _stream.view().value()
    mr = _get_memory_resource(mr)
    cdef column_view c_input

    if delimiters is None:
        delimiters = Scalar.from_libcudf(
            cpp_make_string_scalar("".encode(), _stream.view().value(), mr.get_mr())
        )

    cdef const string_scalar* cpp_delimiters = <const string_scalar*>(
        delimiters.c_obj.get()
    )

    c_input = input.view()
    with nogil:
        c_result = cpp_capitalize.capitalize(
            c_input,
            dereference(cpp_delimiters),
            _cs,
            mr.get_mr()
        )

    return Column.from_libcudf(move(c_result), _stream, mr)


cpdef Column title(
    Column input,
    string_character_types sequence_type=string_character_types.ALPHA,
    object stream=None,
    DeviceMemoryResource mr=None,
):
    """Modifies first character of each word to upper-case and lower-cases
    the rest.

    For details, see :cpp:func:`title`.

    Parameters
    ----------
    input : Column
        String column
    sequence_type : string_character_types, default string_character_types.ALPHA
        The character type that is used when identifying words

    Returns
    -------
    pylibcudf.Column
        Column of titled strings
    """
    cdef unique_ptr[column] c_result
    cdef Stream _stream = _get_stream(stream)
    cdef cudaStream_t _cs = _stream.view().value()
    mr = _get_memory_resource(mr)
    cdef column_view c_input = input.view()
    with nogil:
        c_result = cpp_capitalize.title(
            c_input, sequence_type, _cs, mr.get_mr()
        )

    return Column.from_libcudf(move(c_result), _stream, mr)


cpdef Column is_title(Column input, object stream=None, DeviceMemoryResource mr=None):
    """Checks if the strings in the input column are title formatted.

    For details, see :cpp:func:`is_title`.

    Parameters
    ----------
    input : Column
        String column

    Returns
    -------
    pylibcudf.Column
        Column of type BOOL8
    """
    cdef unique_ptr[column] c_result
    cdef Stream _stream = _get_stream(stream)
    cdef cudaStream_t _cs = _stream.view().value()
    mr = _get_memory_resource(mr)
    cdef column_view c_input = input.view()
    with nogil:
        c_result = cpp_capitalize.is_title(c_input, _cs, mr.get_mr())

    return Column.from_libcudf(move(c_result), _stream, mr)
