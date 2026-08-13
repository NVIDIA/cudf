# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from libcpp.memory cimport unique_ptr
from libcpp.string cimport string
from libcpp.utility cimport move
from pylibcudf.column cimport Column
from pylibcudf.libcudf.column.column cimport column, column_view
from pylibcudf.libcudf.strings.convert cimport (
    convert_datetime as cpp_convert_datetime,
)
from pylibcudf.utils cimport _get_stream, _get_memory_resource
from rmm.pylibrmm.memory_resource cimport DeviceMemoryResource
from rmm.pylibrmm.stream cimport Stream

from pylibcudf.types import DataType
from cuda.bindings.cyruntime cimport cudaStream_t

__all__ = ["from_timestamps", "is_timestamp", "to_timestamps"]

cpdef Column to_timestamps(
    Column input,
    DataType timestamp_type,
    str format,
    object stream=None,
    DeviceMemoryResource mr=None,
):
    """
    Returns a new timestamp column converting a strings column into
    timestamps using the provided format pattern.

    For details, see cpp:`cudf::strings::to_timestamps`.

    Parameters
    ----------
    input : Column
        Strings instance for this operation.

    timestamp_type : DataType
        The timestamp type used for creating the output column.

    format : str
        String specifying the timestamp format in strings.

    stream : Stream | None
        CUDA stream on which to perform the operation.

    Returns
    -------
    Column
        New datetime column
    """
    cdef unique_ptr[column] c_result
    cdef string c_format = format.encode()
    cdef Stream _stream = _get_stream(stream)
    cdef cudaStream_t _cs = _stream.view().value()
    mr = _get_memory_resource(mr)
    cdef column_view c_input = input.view()
    with nogil:
        c_result = cpp_convert_datetime.to_timestamps(
            c_input,
            timestamp_type.c_obj,
            c_format,
            _cs,
            mr.get_mr()
        )

    return Column.from_libcudf(move(c_result), _stream, mr)

cpdef Column from_timestamps(
    Column timestamps,
    str format,
    Column input_strings_names,
    object stream=None,
    DeviceMemoryResource mr=None,
):
    """
    Returns a new strings column converting a timestamp column into
    strings using the provided format pattern.

    For details, see cpp:`cudf::strings::from_timestamps`.

    Parameters
    ----------
    timestamps : Column
        Timestamp values to convert

    format : str
        The string specifying output format.

    input_strings_names : Column
        The string names to use for weekdays ("%a", "%A") and months ("%b", "%B").

    stream : Stream | None
        CUDA stream on which to perform the operation.

    Returns
    -------
    Column
        New strings column with formatted timestamps.
    """
    cdef unique_ptr[column] c_result
    cdef string c_format = format.encode()
    cdef Stream _stream = _get_stream(stream)
    cdef cudaStream_t _cs = _stream.view().value()
    mr = _get_memory_resource(mr)
    cdef column_view c_timestamps = timestamps.view()
    cdef column_view c_input_strings_names = input_strings_names.view()
    with nogil:
        c_result = cpp_convert_datetime.from_timestamps(
            c_timestamps,
            c_format,
            c_input_strings_names,
            _cs,
            mr.get_mr()
        )

    return Column.from_libcudf(move(c_result), _stream, mr)

cpdef Column is_timestamp(
    Column input,
    str format,
    object stream=None,
    DeviceMemoryResource mr=None,
):
    """
    Verifies the given strings column can be parsed to timestamps
    using the provided format pattern.

    For details, see cpp:`cudf::strings::is_timestamp`.

    Parameters
    ----------
    input : Column
        Strings instance for this operation.

    format : str
        String specifying the timestamp format in strings.

    stream : Stream | None
        CUDA stream on which to perform the operation.

    Returns
    -------
    Column
        New bool column.
    """
    cdef unique_ptr[column] c_result
    cdef string c_format = format.encode()
    cdef Stream _stream = _get_stream(stream)
    cdef cudaStream_t _cs = _stream.view().value()
    mr = _get_memory_resource(mr)
    cdef column_view c_input = input.view()
    with nogil:
        c_result = cpp_convert_datetime.is_timestamp(
            c_input,
            c_format,
            _cs,
            mr.get_mr()
        )

    return Column.from_libcudf(move(c_result), _stream, mr)
