# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move
from pylibcudf.column cimport Column
from pylibcudf.libcudf.column.column cimport column, column_view
from pylibcudf.libcudf.scalar.scalar cimport string_scalar
from pylibcudf.libcudf.strings.split cimport split as cpp_split
from pylibcudf.libcudf.table.table cimport table
from pylibcudf.libcudf.types cimport size_type
from pylibcudf.scalar cimport Scalar
from pylibcudf.strings.regex_program cimport RegexProgram
from pylibcudf.table cimport Table
from pylibcudf.utils cimport _get_stream, _get_memory_resource
from rmm.pylibrmm.memory_resource cimport DeviceMemoryResource
from rmm.pylibrmm.stream cimport Stream

from cython.operator import dereference
from cuda.bindings.cyruntime cimport cudaStream_t

__all__ = [
    "rsplit",
    "rsplit_re",
    "rsplit_record",
    "rsplit_record_re",
    "split",
    "split_re",
    "split_record",
    "split_record_re",
]

cpdef Table split(
    Column strings_column,
    Scalar delimiter,
    size_type maxsplit,
    object stream=None,
    DeviceMemoryResource mr=None,
):
    """
    Returns a list of columns by splitting each string using the
    specified delimiter.

    For details, see :cpp:func:`split`.

    Parameters
    ----------
    strings_column : Column
        Strings instance for this operation

    delimiter : Scalar
        UTF-8 encoded string indicating the split points in each string.

    maxsplit : int
        Maximum number of splits to perform. -1 indicates all possible
        splits on each string.

    stream : Stream | None
        CUDA stream on which to perform the operation.

    Returns
    -------
    Table
        New table of strings columns
    """
    cdef unique_ptr[table] c_result
    cdef const string_scalar* c_delimiter = <const string_scalar*>(
        delimiter.c_obj.get()
    )
    cdef Stream _stream = _get_stream(stream)
    cdef cudaStream_t _cs = _stream.view().value()
    mr = _get_memory_resource(mr)

    cdef column_view c_strings_column = strings_column.view()
    with nogil:
        c_result = cpp_split.split(
            c_strings_column,
            dereference(c_delimiter),
            maxsplit,
            _cs,
            mr.get_mr()
        )

    return Table.from_libcudf(move(c_result), _stream, mr)


cpdef Table rsplit(
    Column strings_column,
    Scalar delimiter,
    size_type maxsplit,
    object stream=None,
    DeviceMemoryResource mr=None,
):
    """
    Returns a list of columns by splitting each string using the
    specified delimiter starting from the end of each string.

    For details, see :cpp:func:`rsplit`.

    Parameters
    ----------
    strings_column : Column
        Strings instance for this operation

    delimiter : Scalar
        UTF-8 encoded string indicating the split points in each string.

    maxsplit : int
        Maximum number of splits to perform. -1 indicates all possible
        splits on each string.

    stream : Stream | None
        CUDA stream on which to perform the operation.

    Returns
    -------
    Table
        New table of strings columns.
    """
    cdef unique_ptr[table] c_result
    cdef const string_scalar* c_delimiter = <const string_scalar*>(
        delimiter.c_obj.get()
    )
    cdef Stream _stream = _get_stream(stream)
    cdef cudaStream_t _cs = _stream.view().value()
    mr = _get_memory_resource(mr)

    cdef column_view c_strings_column = strings_column.view()
    with nogil:
        c_result = cpp_split.rsplit(
            c_strings_column,
            dereference(c_delimiter),
            maxsplit,
            _cs,
            mr.get_mr()
        )

    return Table.from_libcudf(move(c_result), _stream, mr)

cpdef Column split_record(
    Column strings,
    Scalar delimiter,
    size_type maxsplit,
    object stream=None,
    DeviceMemoryResource mr=None,
):
    """
    Splits individual strings elements into a list of strings.

    For details, see :cpp:func:`split_record`.

    Parameters
    ----------
    strings : Column
        A column of string elements to be split.

    delimiter : Scalar
        The string to identify split points in each string.

    maxsplit : int
        Maximum number of splits to perform. -1 indicates all possible
        splits on each string.

    Returns
    -------
    Column
        Lists column of strings.
    """
    cdef unique_ptr[column] c_result
    cdef const string_scalar* c_delimiter = <const string_scalar*>(
        delimiter.c_obj.get()
    )
    cdef Stream _stream = _get_stream(stream)
    cdef cudaStream_t _cs = _stream.view().value()
    mr = _get_memory_resource(mr)

    cdef column_view c_strings = strings.view()
    with nogil:
        c_result = cpp_split.split_record(
            c_strings,
            dereference(c_delimiter),
            maxsplit,
            _cs,
            mr.get_mr()
        )

    return Column.from_libcudf(move(c_result), _stream, mr)


cpdef Column rsplit_record(
    Column strings,
    Scalar delimiter,
    size_type maxsplit,
    object stream=None,
    DeviceMemoryResource mr=None,
):
    """
    Splits individual strings elements into a list of strings starting
    from the end of each string.

    For details, see :cpp:func:`rsplit_record`.

    Parameters
    ----------
    strings : Column
        A column of string elements to be split.

    delimiter : Scalar
        The string to identify split points in each string.

    maxsplit : int
        Maximum number of splits to perform. -1 indicates all possible
        splits on each string.

    Returns
    -------
    Column
        Lists column of strings.
    """
    cdef unique_ptr[column] c_result
    cdef const string_scalar* c_delimiter = <const string_scalar*>(
        delimiter.c_obj.get()
    )
    cdef Stream _stream = _get_stream(stream)
    cdef cudaStream_t _cs = _stream.view().value()
    mr = _get_memory_resource(mr)

    cdef column_view c_strings = strings.view()
    with nogil:
        c_result = cpp_split.rsplit_record(
            c_strings,
            dereference(c_delimiter),
            maxsplit,
            _cs,
            mr.get_mr()
        )

    return Column.from_libcudf(move(c_result), _stream, mr)


cpdef Table split_re(
    Column input,
    RegexProgram prog,
    size_type maxsplit,
    object stream=None,
    DeviceMemoryResource mr=None,
):
    """
    Splits strings elements into a table of strings columns
    using a regex_program's pattern to delimit each string.

    For details, see :cpp:func:`split_re`.

    Parameters
    ----------
    input : Column
        A column of string elements to be split.

    prog : RegexProgram
        Regex program instance.

    maxsplit : int
        Maximum number of splits to perform. -1 indicates all possible
        splits on each string.

    Returns
    -------
    Table
        A table of columns of strings.
    """
    cdef unique_ptr[table] c_result
    cdef Stream _stream = _get_stream(stream)
    cdef cudaStream_t _cs = _stream.view().value()
    mr = _get_memory_resource(mr)

    cdef column_view c_input = input.view()
    with nogil:
        c_result = cpp_split.split_re(
            c_input,
            prog.c_obj.get()[0],
            maxsplit,
            _cs,
            mr.get_mr()
        )

    return Table.from_libcudf(move(c_result), _stream, mr)

cpdef Table rsplit_re(
    Column input,
    RegexProgram prog,
    size_type maxsplit,
    object stream=None,
    DeviceMemoryResource mr=None,
):
    """
    Splits strings elements into a table of strings columns
    using a regex_program's pattern to delimit each string starting from
    the end of the string.

    For details, see :cpp:func:`rsplit_re`.

    Parameters
    ----------
    input : Column
        A column of string elements to be split.

    prog : RegexProgram
        Regex program instance.

    maxsplit : int
        Maximum number of splits to perform. -1 indicates all possible
        splits on each string.

    Returns
    -------
    Table
        A table of columns of strings.
    """
    cdef unique_ptr[table] c_result
    cdef Stream _stream = _get_stream(stream)
    cdef cudaStream_t _cs = _stream.view().value()
    mr = _get_memory_resource(mr)

    cdef column_view c_input = input.view()
    with nogil:
        c_result = cpp_split.rsplit_re(
            c_input,
            prog.c_obj.get()[0],
            maxsplit,
            _cs,
            mr.get_mr()
        )

    return Table.from_libcudf(move(c_result), _stream, mr)

cpdef Column split_record_re(
    Column input,
    RegexProgram prog,
    size_type maxsplit,
    object stream=None,
    DeviceMemoryResource mr=None,
):
    """
    Splits strings elements into a list column of strings using the given
    regex_program to delimit each string.

    For details, see :cpp:func:`split_record_re`.

    Parameters
    ----------
    input : Column
        A column of string elements to be split.

    prog : RegexProgram
        Regex program instance.

    maxsplit : int
        Maximum number of splits to perform. -1 indicates all possible
        splits on each string.

    Returns
    -------
    Column
        Lists column of strings.
    """
    cdef unique_ptr[column] c_result
    cdef Stream _stream = _get_stream(stream)
    cdef cudaStream_t _cs = _stream.view().value()
    mr = _get_memory_resource(mr)

    cdef column_view c_input = input.view()
    with nogil:
        c_result = cpp_split.split_record_re(
            c_input,
            prog.c_obj.get()[0],
            maxsplit,
            _cs,
            mr.get_mr()
        )

    return Column.from_libcudf(move(c_result), _stream, mr)

cpdef Column rsplit_record_re(
    Column input, RegexProgram prog, size_type maxsplit, object stream=None,
    DeviceMemoryResource mr=None,
):
    """
    Splits strings elements into a list column of strings using the given
    regex_program to delimit each string starting from the end of the string.

    For details, see :cpp:func:`rsplit_record_re`.

    Parameters
    ----------
    input : Column
        A column of string elements to be split.

    prog : RegexProgram
        Regex program instance.

    maxsplit : int
        Maximum number of splits to perform. -1 indicates all possible
        splits on each string.

    Returns
    -------
    Column
        Lists column of strings.
    """
    cdef unique_ptr[column] c_result
    cdef Stream _stream = _get_stream(stream)
    cdef cudaStream_t _cs = _stream.view().value()
    mr = _get_memory_resource(mr)

    cdef column_view c_input = input.view()
    with nogil:
        c_result = cpp_split.rsplit_record_re(
            c_input,
            prog.c_obj.get()[0],
            maxsplit,
            _cs,
            mr.get_mr()
        )

    return Column.from_libcudf(move(c_result), _stream, mr)


cpdef Column split_part(
    Column input, Scalar delimiter, size_type index, object stream=None,
    DeviceMemoryResource mr=None,
):
    cdef unique_ptr[column] c_result
    cdef const string_scalar* c_delimiter = <const string_scalar*>(
        delimiter.c_obj.get()
    )
    cdef Stream _stream = _get_stream(stream)
    cdef cudaStream_t _cs = _stream.view().value()
    mr = _get_memory_resource(mr)

    cdef column_view c_input = input.view()
    with nogil:
        c_result = cpp_split.split_part(
            c_input,
            dereference(c_delimiter),
            index,
            _cs,
            mr.get_mr()
        )

    return Column.from_libcudf(move(c_result), _stream, mr)
