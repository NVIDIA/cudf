# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move
from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.column.column_view cimport column_view
from pylibcudf.libcudf.nvtext.edit_distance cimport (
    edit_distance as cpp_edit_distance,
)

from rmm.pylibrmm.memory_resource cimport DeviceMemoryResource
from rmm.pylibrmm.stream cimport Stream

from ..column cimport Column
from ..utils cimport _get_stream, _get_memory_resource
from cuda.bindings.cyruntime cimport cudaStream_t

__all__ = ["edit_distance"]

cpdef Column edit_distance(
    Column input,
    Column targets,
    object stream=None,
    DeviceMemoryResource mr=None,
):
    """
    Returns the edit distance between individual strings in two strings columns

    For details, see :cpp:func:`edit_distance`

    Parameters
    ----------
    input : Column
        Input strings
    targets : Column
        Strings to compute edit distance against
    stream : Stream | None
        CUDA stream on which to perform the operation.

    Returns
    -------
    Column
        New column of edit distance values
    """
    cdef column_view c_strings = input.view()
    cdef column_view c_targets = targets.view()
    cdef unique_ptr[column] c_result
    cdef Stream _stream = _get_stream(stream)
    cdef cudaStream_t _cs = _stream.view().value()
    mr = _get_memory_resource(mr)

    with nogil:
        c_result = cpp_edit_distance(c_strings, c_targets, _cs, mr.get_mr())

    return Column.from_libcudf(move(c_result), _stream, mr)
