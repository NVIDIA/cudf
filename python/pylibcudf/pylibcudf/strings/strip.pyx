# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from cython.operator cimport dereference
from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move
from pylibcudf.column cimport Column
from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.column.column_view cimport column_view
from pylibcudf.libcudf.scalar.scalar cimport string_scalar
from pylibcudf.libcudf.scalar.scalar_factories cimport (
    make_string_scalar as cpp_make_string_scalar,
)
from pylibcudf.libcudf.strings cimport strip as cpp_strip
from pylibcudf.scalar cimport Scalar
from pylibcudf.strings.side_type cimport side_type
from pylibcudf.utils cimport _get_stream, _get_memory_resource
from rmm.pylibrmm.memory_resource cimport DeviceMemoryResource
from rmm.pylibrmm.stream cimport Stream
from cuda.bindings.cyruntime cimport cudaStream_t

__all__ = ["strip"]

cpdef Column strip(
    Column input,
    side_type side=side_type.BOTH,
    Scalar to_strip=None,
    object stream=None,
    DeviceMemoryResource mr=None,
):
    """Removes the specified characters from the beginning
    or end (or both) of each string.

    For details, see :cpp:func:`strip`.

    Parameters
    ----------
    input : Column
        Strings column for this operation
    side : SideType, default SideType.BOTH
        Indicates characters are to be stripped from the beginning,
        end, or both of each string; Default is both
    to_strip : Scalar
        UTF-8 encoded characters to strip from each string;
        Default is empty string which indicates strip whitespace characters

    Returns
    -------
    pylibcudf.Column
        New strings column.
    """
    cdef Stream _stream = _get_stream(stream)
    cdef cudaStream_t _cs = _stream.view().value()
    mr = _get_memory_resource(mr)
    cdef column_view c_input

    if to_strip is None:
        to_strip = Scalar.from_libcudf(
            cpp_make_string_scalar("".encode(), _stream.view().value(), mr.get_mr())
        )

    cdef unique_ptr[column] c_result
    cdef string_scalar* cpp_to_strip
    cpp_to_strip = <string_scalar *>(to_strip.c_obj.get())

    c_input = input.view()
    with nogil:
        c_result = cpp_strip.strip(
            c_input,
            side,
            dereference(cpp_to_strip),
            _cs,
            mr.get_mr()
        )

    return Column.from_libcudf(move(c_result), _stream, mr)
