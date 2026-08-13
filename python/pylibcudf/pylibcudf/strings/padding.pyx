# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move
from pylibcudf.column cimport Column
from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.column.column_view cimport column_view
from pylibcudf.libcudf.strings cimport padding as cpp_padding
from pylibcudf.libcudf.strings.side_type cimport side_type
from pylibcudf.libcudf.types cimport size_type
from pylibcudf.utils cimport _get_stream, _get_memory_resource
from rmm.pylibrmm.memory_resource cimport DeviceMemoryResource
from rmm.pylibrmm.stream cimport Stream
from cuda.bindings.cyruntime cimport cudaStream_t

__all__ = ["pad", "zfill", "zfill_by_widths"]

cpdef Column pad(
    Column input,
    size_type width,
    side_type side,
    str fill_char,
    object stream=None,
    DeviceMemoryResource mr=None,
):
    """
    Add padding to each string using a provided character.

    For details, see :cpp:func:`pad`.

    Parameters
    ----------
    input : Column
        Strings instance for this operation
    width : int
        The minimum number of characters for each string.
    side : SideType
        Where to place the padding characters.
    fill_char : str
        Single UTF-8 character to use for padding
    stream : Stream | None
        CUDA stream on which to perform the operation.

    Returns
    -------
    Column
        New column with padded strings.
    """
    cdef unique_ptr[column] c_result
    cdef string c_fill_char = fill_char.encode("utf-8")
    cdef Stream _stream = _get_stream(stream)
    cdef cudaStream_t _cs = _stream.view().value()
    mr = _get_memory_resource(mr)
    cdef column_view c_input = input.view()
    with nogil:
        c_result = cpp_padding.pad(
            c_input,
            width,
            side,
            c_fill_char,
            _cs,
            mr.get_mr()
        )

    return Column.from_libcudf(move(c_result), _stream, mr)

cpdef Column zfill(
    Column input, size_type width, object stream=None, DeviceMemoryResource mr=None
):
    """
    Add '0' as padding to the left of each string.

    For details, see :cpp:func:`zfill`.

    Parameters
    ----------
    input : Column
        Strings instance for this operation
    width : int
        The minimum number of characters for each string.
    stream : Stream | None
        CUDA stream on which to perform the operation.

    Returns
    -------
    Column
        New column of strings.
    """
    cdef unique_ptr[column] c_result
    cdef Stream _stream = _get_stream(stream)
    cdef cudaStream_t _cs = _stream.view().value()
    mr = _get_memory_resource(mr)
    cdef column_view c_input = input.view()
    with nogil:
        c_result = cpp_padding.zfill(
            c_input,
            width,
            _cs,
            mr.get_mr()
        )

    return Column.from_libcudf(move(c_result), _stream, mr)

cpdef Column zfill_by_widths(
    Column input, Column widths, object stream=None, DeviceMemoryResource mr=None
):
    """
    Add '0' as padding to the left of each string.

    For details, see :cpp:func:`zfill_by_widths`.

    Parameters
    ----------
    input : Column
        Strings instance for this operation
    widths : Column
        The minimum number of characters for each string.
    stream : Stream | None
        CUDA stream on which to perform the operation.

    Returns
    -------
    Column
        New column of strings.
    """
    cdef unique_ptr[column] c_result
    cdef Stream _stream = _get_stream(stream)
    cdef cudaStream_t _cs = _stream.view().value()
    mr = _get_memory_resource(mr)
    cdef column_view c_input = input.view()
    cdef column_view c_widths = widths.view()
    with nogil:
        c_result = cpp_padding.zfill_by_widths(
            c_input,
            c_widths,
            _cs,
            mr.get_mr()
        )

    return Column.from_libcudf(move(c_result), _stream, mr)
