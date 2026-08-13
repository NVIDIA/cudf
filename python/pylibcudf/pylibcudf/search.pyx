# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move
from libcpp.vector cimport vector
from pylibcudf.libcudf cimport search as cpp_search
from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.column.column_view cimport column_view
from pylibcudf.libcudf.table.table_view cimport table_view
from pylibcudf.libcudf.types cimport null_order, order
from rmm.pylibrmm.stream cimport Stream
from rmm.pylibrmm.memory_resource cimport DeviceMemoryResource

from .column cimport Column
from .table cimport Table
from .utils cimport _get_stream, _get_memory_resource
from cuda.bindings.cyruntime cimport cudaStream_t

__all__ = ["contains", "lower_bound", "upper_bound"]

cpdef Column lower_bound(
    Table haystack,
    Table needles,
    list column_order,
    list null_precedence,
    object stream=None,
    DeviceMemoryResource mr=None,
):
    """Find smallest indices in haystack where needles may be inserted to retain order.

    For details, see :cpp:func:`lower_bound`.

    Parameters
    ----------
    haystack : Table
        The search space.
    needles : Table
        The values for which to find insertion points.
    column_order : List[ColumnOrder]
        Whether each column should be sorted in ascending or descending order.
    null_precedence : List[NullOrder]
        Whether nulls should come before or after non-nulls.
    stream : Stream | None
        CUDA stream on which to perform the operation.
    mr : DeviceMemoryResource | None
        Device memory resource used to allocate the returned column's device memory.

    Returns
    -------
    Column
        The insertion points
    """
    cdef unique_ptr[column] c_result
    cdef vector[order] c_orders = column_order
    cdef vector[null_order] c_null_precedence = null_precedence

    cdef Stream _stream = _get_stream(stream)
    cdef cudaStream_t _cs = _stream.view().value()
    mr = _get_memory_resource(mr)

    cdef table_view c_haystack = haystack.view()
    cdef table_view c_needles = needles.view()
    with nogil:
        c_result = cpp_search.lower_bound(
            c_haystack,
            c_needles,
            c_orders,
            c_null_precedence,
            _cs,
            mr.get_mr()
        )
    return Column.from_libcudf(move(c_result), _stream, mr)


cpdef Column upper_bound(
    Table haystack,
    Table needles,
    list column_order,
    list null_precedence,
    object stream=None,
    DeviceMemoryResource mr=None,
):
    """Find largest indices in haystack where needles may be inserted to retain order.

    For details, see :cpp:func:`upper_bound`.

    Parameters
    ----------
    haystack : Table
        The search space.
    needles : Table
        The values for which to find insertion points.
    column_order : List[ColumnOrder]
        Whether each column should be sorted in ascending or descending order.
    null_precedence : List[NullOrder]
        Whether nulls should come before or after non-nulls.
    stream : Stream | None
        CUDA stream on which to perform the operation.
    mr : DeviceMemoryResource | None
        Device memory resource used to allocate the returned column's device memory.

    Returns
    -------
    Column
        The insertion points
    """
    cdef unique_ptr[column] c_result
    cdef vector[order] c_orders = column_order
    cdef vector[null_order] c_null_precedence = null_precedence

    cdef Stream _stream = _get_stream(stream)
    cdef cudaStream_t _cs = _stream.view().value()
    mr = _get_memory_resource(mr)

    cdef table_view c_haystack = haystack.view()
    cdef table_view c_needles = needles.view()
    with nogil:
        c_result = cpp_search.upper_bound(
            c_haystack,
            c_needles,
            c_orders,
            c_null_precedence,
            _cs,
            mr.get_mr()
        )
    return Column.from_libcudf(move(c_result), _stream, mr)


cpdef Column contains(
    Column haystack, Column needles, object stream=None, DeviceMemoryResource mr=None
):
    """Check whether needles are present in haystack.

    For details, see :cpp:func:`contains`.

    Parameters
    ----------
    haystack : Column
        The search space.
    needles : Column
        The values for which to search.
    stream : Stream | None
        CUDA stream on which to perform the operation.
    mr : DeviceMemoryResource | None
        Device memory resource used to allocate the returned column's device memory.

    Returns
    -------
    Column
        Boolean indicator for each needle.
    """
    cdef unique_ptr[column] c_result

    cdef Stream _stream = _get_stream(stream)
    cdef cudaStream_t _cs = _stream.view().value()
    mr = _get_memory_resource(mr)

    cdef column_view c_haystack = haystack.view()
    cdef column_view c_needles = needles.view()
    with nogil:
        c_result = cpp_search.contains(
            c_haystack,
            c_needles,
            _cs,
            mr.get_mr()
        )
    return Column.from_libcudf(move(c_result), _stream, mr)
