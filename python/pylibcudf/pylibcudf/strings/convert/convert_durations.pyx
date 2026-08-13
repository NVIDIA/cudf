# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from libcpp.memory cimport unique_ptr
from libcpp.string cimport string
from libcpp.utility cimport move
from pylibcudf.column cimport Column
from pylibcudf.libcudf.column.column cimport column, column_view
from pylibcudf.libcudf.strings.convert cimport (
    convert_durations as cpp_convert_durations,
)
from pylibcudf.utils cimport _get_stream, _get_memory_resource
from rmm.pylibrmm.memory_resource cimport DeviceMemoryResource
from rmm.pylibrmm.stream cimport Stream

from pylibcudf.types import DataType
from cuda.bindings.cyruntime cimport cudaStream_t

__all__ = ["from_durations", "to_durations"]

cpdef Column to_durations(
    Column input,
    DataType duration_type,
    str format,
    object stream=None,
    DeviceMemoryResource mr=None
):
    """
    Returns a new duration column converting a strings column into
    durations using the provided format pattern.

    For details, see cpp:func:`cudf::strings::to_durations`

    Parameters
    ----------
    input : Column
        Strings instance for this operation.

    duration_type : DataType
        The duration type used for creating the output column.

    format : str
        String specifying the duration format in strings.

    stream : Stream | None
        CUDA stream on which to perform the operation.

    Returns
    -------
    Column
        New duration column.
    """
    cdef unique_ptr[column] c_result
    cdef string c_format = format.encode()
    cdef Stream _stream = _get_stream(stream)
    cdef cudaStream_t _cs = _stream.view().value()
    mr = _get_memory_resource(mr)

    cdef column_view c_input = input.view()
    with nogil:
        c_result = cpp_convert_durations.to_durations(
            c_input,
            duration_type.c_obj,
            c_format,
            _cs,
            mr.get_mr()
        )

    return Column.from_libcudf(move(c_result), _stream, mr)

cpdef Column from_durations(
    Column durations,
    str format=None,
    object stream=None,
    DeviceMemoryResource mr=None
):
    """
    Returns a new strings column converting a duration column into
    strings using the provided format pattern.

    For details, see cpp:func:`cudf::strings::from_durations`

    Parameters
    ----------
    durations : Column
        Duration values to convert.

    format : str
        The string specifying output format.
        Default format is "%D days %H:%M:%S".

    stream : Stream | None
        CUDA stream on which to perform the operation.

    Returns
    -------
    Column
        New strings column with formatted durations.
    """
    cdef unique_ptr[column] c_result
    cdef Stream _stream = _get_stream(stream)
    cdef cudaStream_t _cs = _stream.view().value()
    mr = _get_memory_resource(mr)

    if format is None:
        format = "%D days %H:%M:%S"
    cdef string c_format = format.encode()

    cdef column_view c_durations = durations.view()
    with nogil:
        c_result = cpp_convert_durations.from_durations(
            c_durations,
            c_format,
            _cs,
            mr.get_mr()
        )

    return Column.from_libcudf(move(c_result), _stream, mr)
