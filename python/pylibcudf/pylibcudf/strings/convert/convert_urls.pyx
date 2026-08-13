# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move
from pylibcudf.column cimport Column
from pylibcudf.libcudf.column.column cimport column, column_view
from pylibcudf.libcudf.strings.convert cimport convert_urls as cpp_convert_urls
from pylibcudf.utils cimport _get_stream, _get_memory_resource

from rmm.pylibrmm.memory_resource cimport DeviceMemoryResource
from rmm.pylibrmm.stream cimport Stream
from cuda.bindings.cyruntime cimport cudaStream_t

__all__ = ["url_decode", "url_encode"]

cpdef Column url_encode(Column input, object stream=None, DeviceMemoryResource mr=None):
    """
    Encodes each string using URL encoding.

    For details, see :cpp:func:`url_encode`

    Parameters
    ----------
    input : Column
        Strings instance for this operation.

    stream : Stream | None
        CUDA stream on which to perform the operation.

    Returns
    -------
    Column
        New strings column.
    """
    cdef unique_ptr[column] c_result
    cdef Stream _stream = _get_stream(stream)
    cdef cudaStream_t _cs = _stream.view().value()
    mr = _get_memory_resource(mr)

    cdef column_view c_input = input.view()
    with nogil:
        c_result = cpp_convert_urls.url_encode(
            c_input, _cs, mr.get_mr()
        )

    return Column.from_libcudf(move(c_result), _stream, mr)


cpdef Column url_decode(Column input, object stream=None, DeviceMemoryResource mr=None):
    """
    Decodes each string using URL encoding.

    For details, see :cpp:func:`url_decode`

    Parameters
    ----------
    input : Column
        Strings instance for this operation.

    stream : Stream | None
        CUDA stream on which to perform the operation.

    Returns
    -------
    Column
        New strings column.
    """
    cdef unique_ptr[column] c_result
    cdef Stream _stream = _get_stream(stream)
    cdef cudaStream_t _cs = _stream.view().value()
    mr = _get_memory_resource(mr)

    cdef column_view c_input = input.view()
    with nogil:
        c_result = cpp_convert_urls.url_decode(
            c_input, _cs, mr.get_mr()
        )

    return Column.from_libcudf(move(c_result), _stream, mr)
