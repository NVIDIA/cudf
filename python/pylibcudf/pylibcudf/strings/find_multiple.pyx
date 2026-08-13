# SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move
from pylibcudf.column cimport Column
from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.column.column_view cimport column_view
from pylibcudf.libcudf.strings cimport find_multiple as cpp_find_multiple
from pylibcudf.libcudf.table.table cimport table
from pylibcudf.table cimport Table
from pylibcudf.utils cimport _get_stream, _get_memory_resource
from rmm.pylibrmm.memory_resource cimport DeviceMemoryResource
from rmm.pylibrmm.stream cimport Stream
from cuda.bindings.cyruntime cimport cudaStream_t

__all__ = ["find_multiple", "contains_multiple"]

cpdef Column find_multiple(
    Column input,
    Column targets,
    object stream=None,
    DeviceMemoryResource mr=None,
):
    """
    Returns a lists column with character position values where each
    of the target strings are found in each string.

    For details, see :cpp:func:`find_multiple`.

    Parameters
    ----------
    input : Column
        Strings instance for this operation
    targets : Column
        Strings to search for in each string
    stream : Stream | None
        CUDA stream on which to perform the operation.

    Returns
    -------
    Column
        Lists column with character position values
    """
    cdef unique_ptr[column] c_result
    cdef Stream _stream = _get_stream(stream)
    cdef cudaStream_t _cs = _stream.view().value()
    mr = _get_memory_resource(mr)
    cdef column_view c_input = input.view()
    cdef column_view c_targets = targets.view()
    with nogil:
        c_result = cpp_find_multiple.find_multiple(
            c_input,
            c_targets,
            _cs,
            mr.get_mr()
        )

    return Column.from_libcudf(move(c_result), _stream, mr)


cpdef Table contains_multiple(
    Column input,
    Column targets,
    object stream=None,
    DeviceMemoryResource mr=None,
):
    """
    Returns a table of boolean values where each column indicates
    whether the corresponding target is found at that row.

    For details, see :cpp:func:`contains_multiple`.

    Parameters
    ----------
    input : Column
        Strings instance for this operation
    targets : Column
        Strings to search for in each string
    stream : Stream | None
        CUDA stream on which to perform the operation.

    Returns
    -------
    Table
        Columns of booleans
    """
    cdef unique_ptr[table] c_result
    cdef Stream _stream = _get_stream(stream)
    cdef cudaStream_t _cs = _stream.view().value()
    mr = _get_memory_resource(mr)
    cdef column_view c_input = input.view()
    cdef column_view c_targets = targets.view()
    with nogil:
        c_result = cpp_find_multiple.contains_multiple(
            c_input,
            c_targets,
            _cs,
            mr.get_mr()
        )

    return Table.from_libcudf(move(c_result), _stream, mr)
