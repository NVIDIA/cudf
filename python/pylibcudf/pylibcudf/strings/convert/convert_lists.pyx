# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move
from pylibcudf.column cimport Column
from pylibcudf.column_factories cimport make_empty_column
from pylibcudf.libcudf.column.column cimport column, column_view
from pylibcudf.libcudf.scalar.scalar cimport string_scalar
from pylibcudf.libcudf.scalar.scalar_factories cimport (
    make_string_scalar as cpp_make_string_scalar,
)
from pylibcudf.libcudf.strings.convert cimport (
    convert_lists as cpp_convert_lists,
)
from pylibcudf.scalar cimport Scalar
from pylibcudf.types cimport type_id
from pylibcudf.utils cimport _get_stream, _get_memory_resource
from rmm.pylibrmm.memory_resource cimport DeviceMemoryResource
from rmm.pylibrmm.stream cimport Stream

from cython.operator import dereference
from cuda.bindings.cyruntime cimport cudaStream_t

__all__ = ["format_list_column"]

cpdef Column format_list_column(
    Column input,
    Scalar na_rep=None,
    Column separators=None,
    object stream=None,
    DeviceMemoryResource mr=None,
):
    """
    Convert a list column of strings into a formatted strings column.

    For details, see :cpp:func`cudf::strings::format_list_column`

    Parameters
    ----------
    input : Column
        Lists column to format

    na_rep : Scalar
        Replacement string for null elements.
        Default, empty string

    separators : Column
        Strings to use for enclosing list components and separating elements.
        Default, ``,``, ``[``, ``]``

    stream : Stream | None
        CUDA stream on which to perform the operation.

    Returns
    -------
    Column
        New strings column
    """
    cdef unique_ptr[column] c_result

    cdef Stream _stream = _get_stream(stream)
    cdef cudaStream_t _cs = _stream.view().value()
    mr = _get_memory_resource(mr)

    if na_rep is None:
        na_rep = Scalar.from_libcudf(
            cpp_make_string_scalar("".encode(), _stream.view().value(), mr.get_mr())
        )

    cdef const string_scalar* c_na_rep = <const string_scalar*>(
        na_rep.c_obj.get()
    )

    if separators is None:
        separators = make_empty_column(type_id.STRING)

    cdef column_view c_input = input.view()
    cdef column_view c_separators = separators.view()
    with nogil:
        c_result = cpp_convert_lists.format_list_column(
            c_input,
            dereference(c_na_rep),
            c_separators,
            _cs,
            mr.get_mr()
        )

    return Column.from_libcudf(move(c_result), _stream, mr)
