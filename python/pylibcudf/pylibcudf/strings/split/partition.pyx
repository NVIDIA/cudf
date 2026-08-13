# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move
from pylibcudf.column cimport Column
from pylibcudf.libcudf.column.column cimport column_view
from pylibcudf.libcudf.scalar.scalar cimport string_scalar
from pylibcudf.libcudf.scalar.scalar_factories cimport (
    make_string_scalar as cpp_make_string_scalar,
)
from pylibcudf.libcudf.strings.split cimport partition as cpp_partition
from pylibcudf.libcudf.table.table cimport table
from pylibcudf.scalar cimport Scalar
from pylibcudf.table cimport Table
from pylibcudf.utils cimport _get_stream, _get_memory_resource
from rmm.pylibrmm.memory_resource cimport DeviceMemoryResource
from rmm.pylibrmm.stream cimport Stream

from cython.operator import dereference
from cuda.bindings.cyruntime cimport cudaStream_t

__all__ = ["partition", "rpartition"]

cpdef Table partition(
    Column input,
    Scalar delimiter=None,
    object stream=None,
    DeviceMemoryResource mr=None,
):
    """
    Returns a set of 3 columns by splitting each string using the
    specified delimiter.

    For details, see :cpp:func:`partition`.

    Parameters
    ----------
    input : Column
        Strings instance for this operation

    delimiter : Scalar
        UTF-8 encoded string indicating where to split each string.

    Returns
    -------
    Table
        New table of strings columns
    """
    cdef unique_ptr[table] c_result

    cdef Stream _stream = _get_stream(stream)
    cdef cudaStream_t _cs = _stream.view().value()
    mr = _get_memory_resource(mr)

    if delimiter is None:
        delimiter = Scalar.from_libcudf(
            cpp_make_string_scalar("".encode(), _stream.view().value(), mr.get_mr())
        )

    cdef const string_scalar* c_delimiter = <const string_scalar*>(
        delimiter.c_obj.get()
    )

    cdef column_view c_input = input.view()
    with nogil:
        c_result = cpp_partition.partition(
            c_input,
            dereference(c_delimiter),
            _cs,
            mr.get_mr()
        )

    return Table.from_libcudf(move(c_result), _stream, mr)

cpdef Table rpartition(
    Column input,
    Scalar delimiter=None,
    object stream=None,
    DeviceMemoryResource mr=None,
):
    """
    Returns a set of 3 columns by splitting each string using the
    specified delimiter starting from the end of each string.

    For details, see :cpp:func:`rpartition`.

    Parameters
    ----------
    input : Column
        Strings instance for this operation

    delimiter : Scalar
        UTF-8 encoded string indicating where to split each string.

    Returns
    -------
    Table
       New strings columns
    """
    cdef unique_ptr[table] c_result

    cdef Stream _stream = _get_stream(stream)
    cdef cudaStream_t _cs = _stream.view().value()
    mr = _get_memory_resource(mr)

    if delimiter is None:
        delimiter = Scalar.from_libcudf(
            cpp_make_string_scalar("".encode(), _stream.view().value(), mr.get_mr())
        )

    cdef const string_scalar* c_delimiter = <const string_scalar*>(
        delimiter.c_obj.get()
    )

    cdef column_view c_input = input.view()
    with nogil:
        c_result = cpp_partition.rpartition(
            c_input,
            dereference(c_delimiter),
            _cs,
            mr.get_mr()
        )

    return Table.from_libcudf(move(c_result), _stream, mr)
