# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move
from pylibcudf.column cimport Column
from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.column.column_view cimport column_view
from pylibcudf.libcudf.strings cimport repeat as cpp_repeat
from pylibcudf.libcudf.types cimport size_type

from rmm.pylibrmm.memory_resource cimport DeviceMemoryResource
from rmm.pylibrmm.stream cimport Stream

from ..utils cimport _get_stream, _get_memory_resource
from cuda.bindings.cyruntime cimport cudaStream_t

__all__ = ["repeat_strings"]

cpdef Column repeat_strings(
    Column input,
    ColumnorSizeType repeat_times,
    object stream=None,
    DeviceMemoryResource mr=None,
):
    """
    Repeat each string in the given strings column by the numbers
    of times given in another numeric column.

    For details, see :cpp:func:`repeat`.

    Parameters
    ----------
    input : Column
        The column containing strings to repeat.
    repeat_times : Column or int
        Number(s) of times that the corresponding input strings
        for each row are repeated.
    stream : Stream | None
        CUDA stream on which to perform the operation.
    mr : DeviceMemoryResource | None
        Device memory resource used to allocate the returned column's device memory.

    Returns
    -------
    Column
        New column containing the repeated strings.
    """
    cdef unique_ptr[column] c_result
    cdef Stream _stream = _get_stream(stream)
    cdef cudaStream_t _cs = _stream.view().value()
    mr = _get_memory_resource(mr)
    cdef column_view c_input
    cdef column_view c_repeat_times

    if ColumnorSizeType is Column:
        c_input = input.view()
        c_repeat_times = repeat_times.view()
        with nogil:
            c_result = cpp_repeat.repeat_strings(
                c_input,
                c_repeat_times,
                _cs,
                mr.get_mr()
            )
    elif ColumnorSizeType is size_type:
        c_input = input.view()
        with nogil:
            c_result = cpp_repeat.repeat_strings(
                c_input,
                repeat_times,
                _cs,
                mr.get_mr()
            )
    else:
        raise ValueError("repeat_times must be size_type or integer")

    return Column.from_libcudf(move(c_result), _stream, mr)
