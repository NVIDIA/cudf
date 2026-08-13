# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from libcpp.memory cimport unique_ptr
from libcpp.pair cimport pair
from libcpp.utility cimport move
from libcpp.vector cimport vector
from pylibcudf.column cimport Column
from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.column.column_view cimport column_view
from pylibcudf.libcudf.scalar.scalar cimport string_scalar
from pylibcudf.libcudf.strings cimport translate as cpp_translate
from pylibcudf.libcudf.types cimport char_utf8
from pylibcudf.scalar cimport Scalar
from pylibcudf.utils cimport _get_stream, _get_memory_resource
from rmm.pylibrmm.memory_resource cimport DeviceMemoryResource
from rmm.pylibrmm.stream cimport Stream

from cython.operator import dereference
from cuda.bindings.cyruntime cimport cudaStream_t
from pylibcudf.libcudf.strings.translate import \
    filter_type as FilterType  # no-cython-lint

__all__ = ["FilterType", "filter_characters", "translate"]

cdef vector[pair[char_utf8, char_utf8]] _table_to_c_table(dict table):
    """
    Convert str.maketrans table to cudf compatible table.
    """
    cdef int table_size = len(table)
    cdef vector[pair[char_utf8, char_utf8]] c_table

    c_table.reserve(table_size)
    for key, value in table.items():
        if isinstance(value, int):
            value = chr(value)
        if isinstance(value, str):
            value = int.from_bytes(value.encode(), byteorder='big')
        if isinstance(key, int):
            key = chr(key)
        if isinstance(key, str):
            key = int.from_bytes(key.encode(), byteorder='big')
        c_table.push_back((key, value))

    return c_table


cpdef Column translate(
    Column input, dict chars_table, object stream=None, DeviceMemoryResource mr=None
):
    """
    Translates individual characters within each string.

    For details, see :cpp:func:`translate`.

    Parameters
    ----------
    input : Column
        Strings instance for this operation

    chars_table : dict
        Table of UTF-8 character mappings
    stream : Stream | None
        CUDA stream on which to perform the operation.

    Returns
    -------
    Column
        New column with padded strings.
    """
    cdef unique_ptr[column] c_result
    cdef vector[pair[char_utf8, char_utf8]] c_chars_table = _table_to_c_table(
        chars_table
    )
    cdef Stream _stream = _get_stream(stream)
    cdef cudaStream_t _cs = _stream.view().value()
    mr = _get_memory_resource(mr)
    cdef column_view c_input = input.view()
    with nogil:
        c_result = cpp_translate.translate(
            c_input,
            c_chars_table,
            _cs,
            mr.get_mr()
        )
    return Column.from_libcudf(move(c_result), _stream, mr)


cpdef Column filter_characters(
    Column input,
    dict characters_to_filter,
    filter_type keep_characters,
    Scalar replacement,
    object stream=None,
    DeviceMemoryResource mr=None,
):
    """
    Removes ranges of characters from each string in a strings column.

    For details, see :cpp:func:`filter_characters`.

    Parameters
    ----------
    input : Column
        Strings instance for this operation

    characters_to_filter : dict
        Table of character ranges to filter on

    keep_characters : FilterType
        If true, the `characters_to_filter` are retained
        and all other characters are removed.

    replacement : Scalar
        Replacement string for each character removed.
    stream : Stream | None
        CUDA stream on which to perform the operation.

    Returns
    -------
    Column
        New column with filtered strings.
    """
    cdef unique_ptr[column] c_result
    cdef vector[pair[char_utf8, char_utf8]] c_characters_to_filter = _table_to_c_table(
        characters_to_filter
    )
    cdef const string_scalar* c_replacement = <const string_scalar*>(
        replacement.c_obj.get()
    )
    cdef Stream _stream = _get_stream(stream)
    cdef cudaStream_t _cs = _stream.view().value()
    mr = _get_memory_resource(mr)
    cdef column_view c_input = input.view()
    with nogil:
        c_result = cpp_translate.filter_characters(
            c_input,
            c_characters_to_filter,
            keep_characters,
            dereference(c_replacement),
            _cs,
            mr.get_mr()
        )
    return Column.from_libcudf(move(c_result), _stream, mr)

FilterType.__str__ = FilterType.__repr__
