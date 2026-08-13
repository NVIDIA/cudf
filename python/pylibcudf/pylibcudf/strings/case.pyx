# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move
from pylibcudf.column cimport Column
from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.column.column_view cimport column_view
from pylibcudf.libcudf.strings cimport case as cpp_case
from pylibcudf.utils cimport _get_stream, _get_memory_resource
from rmm.pylibrmm.memory_resource cimport DeviceMemoryResource
from rmm.pylibrmm.stream cimport Stream
from cuda.bindings.cyruntime cimport cudaStream_t

__all__ = ["swapcase", "to_lower", "to_upper"]

cpdef Column to_lower(Column input, object stream=None, DeviceMemoryResource mr=None):
    """Returns a column of lowercased strings.

    For details, see :cpp:func:`to_lower`.

    Parameters
    ----------
    input : Column
        String column
    stream : Stream | None
        CUDA stream on which to perform the operation.
    mr : DeviceMemoryResource | None
        Device memory resource used to allocate the returned column's device memory.

    Returns
    -------
    pylibcudf.Column
        Column of strings lowercased from the input column
    """
    cdef unique_ptr[column] c_result
    cdef Stream _stream = _get_stream(stream)
    cdef cudaStream_t _cs = _stream.view().value()
    mr = _get_memory_resource(mr)
    cdef column_view c_input = input.view()
    with nogil:
        c_result = cpp_case.to_lower(c_input, _cs, mr.get_mr())

    return Column.from_libcudf(move(c_result), _stream, mr)

cpdef Column to_upper(Column input, object stream=None, DeviceMemoryResource mr=None):
    """Returns a column of uppercased strings.

    For details, see :cpp:func:`to_upper`.

    Parameters
    ----------
    input : Column
        String column
    stream : Stream | None
        CUDA stream on which to perform the operation.
    mr : DeviceMemoryResource | None
        Device memory resource used to allocate the returned column's device memory.

    Returns
    -------
    pylibcudf.Column
        Column of strings uppercased from the input column
    """
    cdef unique_ptr[column] c_result
    cdef Stream _stream = _get_stream(stream)
    cdef cudaStream_t _cs = _stream.view().value()
    mr = _get_memory_resource(mr)
    cdef column_view c_input = input.view()
    with nogil:
        c_result = cpp_case.to_upper(c_input, _cs, mr.get_mr())

    return Column.from_libcudf(move(c_result), _stream, mr)

cpdef Column swapcase(Column input, object stream=None, DeviceMemoryResource mr=None):
    """Returns a column of strings where the lowercase characters
    are converted to uppercase and the uppercase characters
    are converted to lowercase.

    For details, see :cpp:func:`swapcase`.

    Parameters
    ----------
    input : Column
        String column
    stream : Stream | None
        CUDA stream on which to perform the operation.
    mr : DeviceMemoryResource | None
        Device memory resource used to allocate the returned column's device memory.

    Returns
    -------
    pylibcudf.Column
        Column of strings
    """
    cdef unique_ptr[column] c_result
    cdef Stream _stream = _get_stream(stream)
    cdef cudaStream_t _cs = _stream.view().value()
    mr = _get_memory_resource(mr)
    cdef column_view c_input = input.view()
    with nogil:
        c_result = cpp_case.swapcase(c_input, _cs, mr.get_mr())

    return Column.from_libcudf(move(c_result), _stream, mr)
