# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move
from pylibcudf.libcudf cimport labeling as cpp_labeling
from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.column.column_view cimport column_view
from pylibcudf.libcudf.labeling cimport inclusive

from pylibcudf.libcudf.labeling import inclusive as Inclusive  # no-cython-lint

from rmm.pylibrmm.stream cimport Stream
from rmm.pylibrmm.memory_resource cimport DeviceMemoryResource

from .column cimport Column
from .utils cimport _get_stream, _get_memory_resource
from cuda.bindings.cyruntime cimport cudaStream_t

__all__ = ["Inclusive", "label_bins"]

cpdef Column label_bins(
    Column input,
    Column left_edges,
    inclusive left_inclusive,
    Column right_edges,
    inclusive right_inclusive,
    object stream=None,
    DeviceMemoryResource mr=None
):
    """Labels elements based on membership in the specified bins.

    For details see :cpp:func:`label_bins`.

    Parameters
    ----------
    input : Column
        Column of input elements to label according to the specified bins.
    left_edges : Column
        Column of the left edge of each bin.
    left_inclusive : Inclusive
        Whether or not the left edge is inclusive.
    right_edges : Column
        Column of the right edge of each bin.
    right_inclusive : Inclusive
        Whether or not the right edge is inclusive.
    stream : Stream | None
        CUDA stream on which to perform the operation.
    mr : DeviceMemoryResource | None
        Device memory resource used to allocate the returned column's device memory.

    Returns
    -------
    Column
        Column of integer labels of the elements in `input`
        according to the specified bins.
    """
    cdef unique_ptr[column] c_result
    cdef Stream _stream = _get_stream(stream)
    cdef cudaStream_t _cs = _stream.view().value()
    mr = _get_memory_resource(mr)

    cdef column_view c_input = input.view()
    cdef column_view c_left_edges = left_edges.view()
    cdef column_view c_right_edges = right_edges.view()
    with nogil:
        c_result = cpp_labeling.label_bins(
            c_input,
            c_left_edges,
            left_inclusive,
            c_right_edges,
            right_inclusive,
            _cs,
            mr.get_mr()
        )

    return Column.from_libcudf(move(c_result), _stream, mr)

Inclusive.__str__ = Inclusive.__repr__
