# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move
from pylibcudf.column cimport Column
from pylibcudf.libcudf.column.column cimport column, column_view
from pylibcudf.libcudf.scalar.scalar cimport string_scalar
from pylibcudf.libcudf.strings.convert cimport (
    convert_booleans as cpp_convert_booleans,
)
from pylibcudf.scalar cimport Scalar
from pylibcudf.utils cimport _get_stream, _get_memory_resource

from cython.operator import dereference
from rmm.pylibrmm.memory_resource cimport DeviceMemoryResource
from rmm.pylibrmm.stream cimport Stream
from cuda.bindings.cyruntime cimport cudaStream_t

__all__ = ["from_booleans", "to_booleans"]

cpdef Column to_booleans(
    Column input, Scalar true_string, object stream=None, DeviceMemoryResource mr=None
):
    """
    Returns a new bool column by parsing boolean values from the strings
    in the provided strings column.

    For details, see :cpp:func:`to_booleans`.

    Parameters
    ----------
    input :  Column
        Strings instance for this operation

    true_string : Scalar
        String to expect for true. Non-matching strings are false

    stream : Stream | None
        CUDA stream on which to perform the operation.

    Returns
    -------
    Column
        New bool column converted from strings.
    """
    cdef unique_ptr[column] c_result
    cdef const string_scalar* c_true_string = <const string_scalar*>(
        true_string.c_obj.get()
    )
    cdef Stream _stream = _get_stream(stream)
    cdef cudaStream_t _cs = _stream.view().value()
    mr = _get_memory_resource(mr)

    cdef column_view c_input = input.view()
    with nogil:
        c_result = cpp_convert_booleans.to_booleans(
            c_input,
            dereference(c_true_string),
            _cs,
            mr.get_mr()
        )

    return Column.from_libcudf(move(c_result), _stream, mr)

cpdef Column from_booleans(
    Column booleans,
    Scalar true_string,
    Scalar false_string,
    object stream=None,
    DeviceMemoryResource mr=None,
):
    """
    Returns a new strings column converting the boolean values from the
    provided column into strings.

    For details, see :cpp:func:`from_booleans`.

    Parameters
    ----------
    booleans :  Column
        Boolean column to convert.

    true_string : Scalar
        String to use for true in the output column.

    false_string : Scalar
        String to use for false in the output column.

    stream : Stream | None
        CUDA stream on which to perform the operation.

    Returns
    -------
    Column
        New strings column.
    """
    cdef unique_ptr[column] c_result
    cdef const string_scalar* c_true_string = <const string_scalar*>(
        true_string.c_obj.get()
    )
    cdef const string_scalar* c_false_string = <const string_scalar*>(
        false_string.c_obj.get()
    )
    cdef Stream _stream = _get_stream(stream)
    cdef cudaStream_t _cs = _stream.view().value()
    mr = _get_memory_resource(mr)

    cdef column_view c_booleans = booleans.view()
    with nogil:
        c_result = cpp_convert_booleans.from_booleans(
            c_booleans,
            dereference(c_true_string),
            dereference(c_false_string),
            _cs,
            mr.get_mr()
        )

    return Column.from_libcudf(move(c_result), _stream, mr)
