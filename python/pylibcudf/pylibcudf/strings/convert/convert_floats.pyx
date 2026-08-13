# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move
from pylibcudf.column cimport Column
from pylibcudf.libcudf.column.column cimport column, column_view
from pylibcudf.libcudf.strings.convert cimport (
    convert_floats as cpp_convert_floats,
)
from pylibcudf.types cimport DataType
from pylibcudf.utils cimport _get_stream, _get_memory_resource
from rmm.pylibrmm.memory_resource cimport DeviceMemoryResource
from rmm.pylibrmm.stream cimport Stream
from cuda.bindings.cyruntime cimport cudaStream_t

__all__ = ["from_floats", "is_float", "to_floats"]

cpdef Column to_floats(
    Column strings,
    DataType output_type,
    object stream=None,
    DeviceMemoryResource mr=None,
):
    """
    Returns a new numeric column by parsing float values from each string
    in the provided strings column.

    For details, see cpp:func:`cudf::strings::to_floats`

    Parameters
    ----------
    strings : Column
        Strings instance for this operation.

    output_type : DataType
        Type of float numeric column to return.

    stream : Stream | None
        CUDA stream on which to perform the operation.

    Returns
    -------
    Column
        New column with floats converted from strings.
    """
    cdef unique_ptr[column] c_result
    cdef Stream _stream = _get_stream(stream)
    cdef cudaStream_t _cs = _stream.view().value()
    mr = _get_memory_resource(mr)

    cdef column_view c_strings = strings.view()
    with nogil:
        c_result = cpp_convert_floats.to_floats(
            c_strings,
            output_type.c_obj,
            _cs,
            mr.get_mr()
        )

    return Column.from_libcudf(move(c_result), _stream, mr)


cpdef Column from_floats(
    Column floats, object stream=None, DeviceMemoryResource mr=None
):
    """
    Returns a new strings column converting the float values from the
    provided column into strings.

    For details, see cpp:func:`cudf::strings::from_floats`

    Parameters
    ----------
    floats : Column
        Numeric column to convert.

    stream : Stream | None
        CUDA stream on which to perform the operation.

    Returns
    -------
    Column
        New strings column with floats as strings.
    """
    cdef unique_ptr[column] c_result
    cdef Stream _stream = _get_stream(stream)
    cdef cudaStream_t _cs = _stream.view().value()
    mr = _get_memory_resource(mr)

    cdef column_view c_floats = floats.view()
    with nogil:
        c_result = cpp_convert_floats.from_floats(
            c_floats, _cs, mr.get_mr()
        )

    return Column.from_libcudf(move(c_result), _stream, mr)


cpdef Column is_float(Column input, object stream=None, DeviceMemoryResource mr=None):
    """
    Returns a boolean column identifying strings in which all
    characters are valid for conversion to floats.

    For details, see cpp:func:`cudf::strings::is_float`

    Parameters
    ----------
    input : Column
        Strings instance for this operation.

    stream : Stream | None
        CUDA stream on which to perform the operation.

    Returns
    -------
    Column
        New column of boolean results for each string.
    """
    cdef unique_ptr[column] c_result
    cdef Stream _stream = _get_stream(stream)
    cdef cudaStream_t _cs = _stream.view().value()
    mr = _get_memory_resource(mr)

    cdef column_view c_input = input.view()
    with nogil:
        c_result = cpp_convert_floats.is_float(
            c_input, _cs, mr.get_mr()
        )

    return Column.from_libcudf(move(c_result), _stream, mr)
