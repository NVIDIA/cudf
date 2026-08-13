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
from pylibcudf.libcudf.strings cimport combine as cpp_combine
from pylibcudf.scalar cimport Scalar
from pylibcudf.table cimport Table
from pylibcudf.libcudf.table.table_view cimport table_view
from pylibcudf.utils cimport _get_stream, _get_memory_resource
from rmm.pylibrmm.memory_resource cimport DeviceMemoryResource
from rmm.pylibrmm.stream cimport Stream

from cython.operator import dereference
from cuda.bindings.cyruntime cimport cudaStream_t
from pylibcudf.libcudf.strings.combine import \
    output_if_empty_list as OutputIfEmptyList  # no-cython-lint
from pylibcudf.libcudf.strings.combine import \
    separator_on_nulls as SeparatorOnNulls  # no-cython-lint

__all__ = [
    "OutputIfEmptyList",
    "SeparatorOnNulls",
    "concatenate",
    "join_list_elements",
    "join_strings",
]

cpdef Column concatenate(
    Table strings_columns,
    ColumnOrScalar separator,
    Scalar narep=None,
    Scalar col_narep=None,
    separator_on_nulls separate_nulls=separator_on_nulls.YES,
    object stream=None,
    DeviceMemoryResource mr=None,
):
    """
    Concatenate all columns in the table horizontally into one new string
    delimited by an optional separator string.

    Parameters
    ----------
    strings_columns : Table
        Strings for this operation

    separator : Column or Scalar
        Separator(s) for a given row

    narep : Scalar
        String to replace a null separator for a given row.

    col_narep : Scalar
        String that should be used in place of any null strings found in any column.
        An exception is raised when separator is a Scalar.

    separate_nulls : SeparatorOnNulls
        If YES, then the separator is included for null rows.

    Returns
    -------
    Column
        New column with concatenated results
    """
    cdef unique_ptr[column] c_result
    cdef const string_scalar* c_col_narep
    cdef const string_scalar* c_separator
    cdef Stream _stream = _get_stream(stream)
    cdef cudaStream_t _cs = _stream.view().value()
    mr = _get_memory_resource(mr)
    cdef table_view c_strings_columns
    cdef column_view c_separator_view

    if narep is None:
        narep = Scalar.from_libcudf(
            cpp_make_string_scalar("".encode(), _stream.view().value(), mr.get_mr())
        )
    cdef const string_scalar* c_narep = <const string_scalar*>(
        narep.c_obj.get()
    )

    if ColumnOrScalar is Column:
        if col_narep is None:
            col_narep = Scalar.from_libcudf(
                cpp_make_string_scalar("".encode(), _stream.view().value(), mr.get_mr())
            )
        c_col_narep = <const string_scalar*>(
            col_narep.c_obj.get()
        )
        c_strings_columns = strings_columns.view()
        c_separator_view = separator.view()
        with nogil:
            c_result = move(
                cpp_combine.concatenate(
                    c_strings_columns,
                    c_separator_view,
                    dereference(c_narep),
                    dereference(c_col_narep),
                    separate_nulls,
                    _cs,
                    mr.get_mr()
                )
            )
    elif ColumnOrScalar is Scalar:
        if col_narep is not None:
            raise ValueError(
                "col_narep cannot be specified when separator is a Scalar"
            )
        c_separator = <const string_scalar*>(separator.c_obj.get())
        c_strings_columns = strings_columns.view()
        with nogil:
            c_result = move(
                cpp_combine.concatenate(
                    c_strings_columns,
                    dereference(c_separator),
                    dereference(c_narep),
                    separate_nulls,
                    _cs,
                    mr.get_mr()
                )
            )
    else:
        raise ValueError("separator must be a Column or a Scalar")
    return Column.from_libcudf(move(c_result), _stream, mr)


cpdef Column join_strings(
    Column input,
    Scalar separator,
    Scalar narep,
    object stream=None,
    DeviceMemoryResource mr=None,
):
    """
    Concatenates all strings in the column into one new string delimited
    by an optional separator string.

    Parameters
    ----------
    input : Column
        List of strings columns to concatenate

    separator : Scalar
        Strings column that provides the separator for a given row

    narep : Scalar
        String to replace any null strings found.

    Returns
    -------
    Column
        New column containing one string
    """
    cdef unique_ptr[column] c_result
    cdef Stream _stream = _get_stream(stream)
    cdef cudaStream_t _cs = _stream.view().value()
    mr = _get_memory_resource(mr)
    cdef column_view c_input
    cdef const string_scalar* c_separator = <const string_scalar*>(
        separator.c_obj.get()
    )
    cdef const string_scalar* c_narep = <const string_scalar*>(
        narep.c_obj.get()
    )
    c_input = input.view()
    with nogil:
        c_result = move(
            cpp_combine.join_strings(
                c_input,
                dereference(c_separator),
                dereference(c_narep),
                _cs,
                mr.get_mr()
            )
        )

    return Column.from_libcudf(move(c_result), _stream, mr)


cpdef Column join_list_elements(
    Column lists_strings_column,
    ColumnOrScalar separator,
    Scalar separator_narep,
    Scalar string_narep,
    separator_on_nulls separate_nulls,
    output_if_empty_list empty_list_policy,
    object stream=None,
    DeviceMemoryResource mr=None,
):
    """
    Given a lists column of strings (each row is a list of strings),
    concatenates the strings within each row and returns a single strings
    column result.

    Parameters
    ----------
    lists_strings_column : Column
        Column containing lists of strings to concatenate

    separator : Column or Scalar
        String(s) that should inserted between each string from each row.

    separator_narep : Scalar
        String that should be used to replace a null separator.

    string_narep : Scalar
        String to replace null strings in any non-null list row.
        Ignored if separator is a Scalar.

    separate_nulls : SeparatorOnNulls
        If YES, then the separator is included for null rows
        if `narep` is valid

    empty_list_policy : OutputIfEmptyList
        If set to EMPTY_STRING, any input row that is an empty
        list will result in an empty string. Otherwise, it will
        result in a null.


    Returns
    -------
    Column
        New strings column with concatenated results
    """
    cdef unique_ptr[column] c_result
    cdef Stream _stream = _get_stream(stream)
    cdef cudaStream_t _cs = _stream.view().value()
    mr = _get_memory_resource(mr)
    cdef column_view c_lists_strings_column
    cdef column_view c_separator_view
    cdef const string_scalar* c_separator_narep = <const string_scalar*>(
        separator_narep.c_obj.get()
    )
    cdef const string_scalar* c_string_narep = <const string_scalar*>(
        string_narep.c_obj.get()
    )
    cdef const string_scalar* c_separator

    if ColumnOrScalar is Column:
        c_lists_strings_column = lists_strings_column.view()
        c_separator_view = separator.view()
        with nogil:
            c_result = move(
                cpp_combine.join_list_elements(
                    c_lists_strings_column,
                    c_separator_view,
                    dereference(c_separator_narep),
                    dereference(c_string_narep),
                    separate_nulls,
                    empty_list_policy,
                    _cs,
                    mr.get_mr()
                )
            )
    elif ColumnOrScalar is Scalar:
        c_separator = <const string_scalar*>(separator.c_obj.get())
        c_lists_strings_column = lists_strings_column.view()
        with nogil:
            c_result = move(
                cpp_combine.join_list_elements(
                    c_lists_strings_column,
                    dereference(c_separator),
                    dereference(c_separator_narep),
                    separate_nulls,
                    empty_list_policy,
                    _cs,
                    mr.get_mr()
                )
            )
    else:
        raise ValueError("separator must be a Column or a Scalar")
    return Column.from_libcudf(move(c_result), _stream, mr)

OutputIfEmptyList.__str__ = OutputIfEmptyList.__repr__
SeparatorOnNulls.__str__ = SeparatorOnNulls.__repr__
