# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move
from libcpp.vector cimport vector
from pylibcudf.libcudf cimport sorting as cpp_sorting
from pylibcudf.libcudf.aggregation cimport rank_method
from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.column.column_view cimport column_view
from pylibcudf.libcudf.table.table cimport table
from pylibcudf.libcudf.table.table_view cimport table_view
from pylibcudf.libcudf.types cimport null_order, null_policy, order, size_type
from rmm.pylibrmm.memory_resource cimport DeviceMemoryResource
from rmm.pylibrmm.stream cimport Stream

from .column cimport Column
from .table cimport Table
from .utils cimport _get_stream, _get_memory_resource
from cuda.bindings.cyruntime cimport cudaStream_t

__all__ = [
    "is_sorted",
    "rank",
    "segmented_sort_by_key",
    "sort",
    "sort_by_key",
    "sorted_order",
    "stable_segmented_sort_by_key",
    "stable_sort",
    "stable_sort_by_key",
    "stable_sorted_order",
    "top_k",
    "top_k_order",
]

cpdef Column sorted_order(
    Table source_table,
    list column_order,
    list null_precedence,
    object stream=None,
    DeviceMemoryResource mr=None,
):
    """Computes the row indices required to sort the table.

    For details, see :cpp:func:`sorted_order`.

    Parameters
    ----------
    source_table : Table
        The table to sort.
    column_order : List[ColumnOrder]
        Whether each column should be sorted in ascending or descending order.
    null_precedence : List[NullOrder]
        Whether nulls should come before or after non-nulls.

    Returns
    -------
    Column
        The row indices required to sort the table.
    """
    cdef unique_ptr[column] c_result
    cdef vector[order] c_orders = column_order
    cdef vector[null_order] c_null_precedence = null_precedence

    cdef Stream _stream = _get_stream(stream)
    cdef cudaStream_t _cs = _stream.view().value()
    mr = _get_memory_resource(mr)

    cdef table_view c_source_table = source_table.view()
    with nogil:
        c_result = cpp_sorting.sorted_order(
            c_source_table,
            c_orders,
            c_null_precedence,
            _cs,
            mr.get_mr()
        )
    return Column.from_libcudf(move(c_result), _stream, mr)


cpdef Column stable_sorted_order(
    Table source_table,
    list column_order,
    list null_precedence,
    object stream=None,
    DeviceMemoryResource mr=None,
):
    """Computes the row indices required to sort the table,
    preserving order of equal elements.

    For details, see :cpp:func:`stable_sorted_order`.

    Parameters
    ----------
    source_table : Table
        The table to sort.
    column_order : List[ColumnOrder]
        Whether each column should be sorted in ascending or descending order.
    null_precedence : List[NullOrder]
        Whether nulls should come before or after non-nulls.

    Returns
    -------
    Column
        The row indices required to sort the table.
    """
    cdef unique_ptr[column] c_result
    cdef vector[order] c_orders = column_order
    cdef vector[null_order] c_null_precedence = null_precedence

    cdef Stream _stream = _get_stream(stream)
    cdef cudaStream_t _cs = _stream.view().value()
    mr = _get_memory_resource(mr)

    cdef table_view c_source_table = source_table.view()
    with nogil:
        c_result = cpp_sorting.stable_sorted_order(
            c_source_table,
            c_orders,
            c_null_precedence,
            _cs,
            mr.get_mr()
        )
    return Column.from_libcudf(move(c_result), _stream, mr)


cpdef Column rank(
    Column input_view,
    rank_method method,
    order column_order,
    null_policy null_handling,
    null_order null_precedence,
    bool percentage,
    object stream=None,
    DeviceMemoryResource mr=None,
):
    """Computes the rank of each element in the column.

    For details, see :cpp:func:`rank`.

    Parameters
    ----------
    input_view : Column
        The column to rank.
    method : rank_method
        The method to use for ranking ties.
    column_order : order
        Whether the column should be sorted in ascending or descending order.
    null_handling : null_policy
        Whether or not nulls should be included in the ranking.
    null_precedence : null_order
        Whether nulls should come before or after non-nulls.
    percentage : bool
        Whether to return the rank as a percentage.

    Returns
    -------
    Column
        The rank of each element in the column.
    """
    cdef unique_ptr[column] c_result

    cdef Stream _stream = _get_stream(stream)
    cdef cudaStream_t _cs = _stream.view().value()
    mr = _get_memory_resource(mr)

    cdef column_view c_input_view = input_view.view()
    with nogil:
        c_result = cpp_sorting.rank(
            c_input_view,
            method,
            column_order,
            null_handling,
            null_precedence,
            percentage,
            _cs,
            mr.get_mr()
        )
    return Column.from_libcudf(move(c_result), _stream, mr)


cpdef bool is_sorted(
    Table tbl, list column_order, list null_precedence, object stream=None
):
    """Checks if the table is sorted.

    For details, see :cpp:func:`is_sorted`.

    Parameters
    ----------
    tbl : Table
        The table to check.
    column_order : List[ColumnOrder]
        Whether each column is expected to be sorted in ascending or descending order.
    null_precedence : List[NullOrder]
        Whether nulls are expected before or after non-nulls.

    Returns
    -------
    bool
        Whether the table is sorted.
    """
    cdef bool c_result
    cdef vector[order] c_orders = column_order
    cdef vector[null_order] c_null_precedence = null_precedence

    cdef Stream _stream = _get_stream(stream)
    cdef cudaStream_t _cs = _stream.view().value()
    cdef table_view c_tbl = tbl.view()

    with nogil:
        c_result = cpp_sorting.is_sorted(
            c_tbl,
            c_orders,
            c_null_precedence,
            _cs
        )
    return c_result


cpdef Table segmented_sort_by_key(
    Table values,
    Table keys,
    Column segment_offsets,
    list column_order,
    list null_precedence,
    object stream=None,
    DeviceMemoryResource mr=None,
):
    """Sorts the table by key, within segments.

    For details, see :cpp:func:`segmented_sort_by_key`.

    Parameters
    ----------
    values : Table
        The table to sort.
    keys : Table
        The table to sort by.
    segment_offsets : Column
        The offsets of the segments.
    column_order : List[ColumnOrder]
        Whether each column should be sorted in ascending or descending order.
    null_precedence : List[NullOrder]
        Whether nulls should come before or after non-nulls.

    Returns
    -------
    Table
        The sorted table.
    """
    cdef unique_ptr[table] c_result
    cdef vector[order] c_orders = column_order
    cdef vector[null_order] c_null_precedence = null_precedence

    cdef Stream _stream = _get_stream(stream)
    cdef cudaStream_t _cs = _stream.view().value()
    mr = _get_memory_resource(mr)

    cdef table_view c_values = values.view()
    cdef table_view c_keys = keys.view()
    cdef column_view c_segment_offsets = segment_offsets.view()
    with nogil:
        c_result = cpp_sorting.segmented_sort_by_key(
            c_values,
            c_keys,
            c_segment_offsets,
            c_orders,
            c_null_precedence,
            _cs,
            mr.get_mr()
        )
    return Table.from_libcudf(move(c_result), _stream, mr)


cpdef Table stable_segmented_sort_by_key(
    Table values,
    Table keys,
    Column segment_offsets,
    list column_order,
    list null_precedence,
    object stream=None,
    DeviceMemoryResource mr=None,
):
    """Sorts the table by key preserving order of equal elements,
    within segments.

    For details, see :cpp:func:`stable_segmented_sort_by_key`.

    Parameters
    ----------
    values : Table
        The table to sort.
    keys : Table
        The table to sort by.
    segment_offsets : Column
        The offsets of the segments.
    column_order : List[ColumnOrder]
        Whether each column should be sorted in ascending or descending order.
    null_precedence : List[NullOrder]
        Whether nulls should come before or after non-nulls.

    Returns
    -------
    Table
        The sorted table.
    """
    cdef unique_ptr[table] c_result
    cdef vector[order] c_orders = column_order
    cdef vector[null_order] c_null_precedence = null_precedence

    cdef Stream _stream = _get_stream(stream)
    cdef cudaStream_t _cs = _stream.view().value()
    mr = _get_memory_resource(mr)

    cdef table_view c_values = values.view()
    cdef table_view c_keys = keys.view()
    cdef column_view c_segment_offsets = segment_offsets.view()
    with nogil:
        c_result = cpp_sorting.stable_segmented_sort_by_key(
            c_values,
            c_keys,
            c_segment_offsets,
            c_orders,
            c_null_precedence,
            _cs,
            mr.get_mr()
        )
    return Table.from_libcudf(move(c_result), _stream, mr)


cpdef Table sort_by_key(
    Table values,
    Table keys,
    list column_order,
    list null_precedence,
    object stream=None,
    DeviceMemoryResource mr=None,
):
    """Sorts the table by key.

    For details, see :cpp:func:`sort_by_key`.

    Parameters
    ----------
    values : Table
        The table to sort.
    keys : Table
        The table to sort by.
    column_order : List[ColumnOrder]
        Whether each column should be sorted in ascending or descending order.
    null_precedence : List[NullOrder]
        Whether nulls should come before or after non-nulls.

    Returns
    -------
    Table
        The sorted table.
    """
    cdef unique_ptr[table] c_result
    cdef vector[order] c_orders = column_order
    cdef vector[null_order] c_null_precedence = null_precedence

    cdef Stream _stream = _get_stream(stream)
    cdef cudaStream_t _cs = _stream.view().value()
    mr = _get_memory_resource(mr)

    cdef table_view c_values = values.view()
    cdef table_view c_keys = keys.view()
    with nogil:
        c_result = cpp_sorting.sort_by_key(
            c_values,
            c_keys,
            c_orders,
            c_null_precedence,
            _cs,
            mr.get_mr()
        )
    return Table.from_libcudf(move(c_result), _stream, mr)


cpdef Table stable_sort_by_key(
    Table values,
    Table keys,
    list column_order,
    list null_precedence,
    object stream=None,
    DeviceMemoryResource mr=None,
):
    """Sorts the table by key preserving order of equal elements.

    For details, see :cpp:func:`stable_sort_by_key`.

    Parameters
    ----------
    values : Table
        The table to sort.
    keys : Table
        The table to sort by.
    column_order : List[ColumnOrder]
        Whether each column should be sorted in ascending or descending order.
    null_precedence : List[NullOrder]
        Whether nulls should come before or after non-nulls.

    Returns
    -------
    Table
        The sorted table.
    """
    cdef unique_ptr[table] c_result
    cdef vector[order] c_orders = column_order
    cdef vector[null_order] c_null_precedence = null_precedence

    cdef Stream _stream = _get_stream(stream)
    cdef cudaStream_t _cs = _stream.view().value()
    mr = _get_memory_resource(mr)

    cdef table_view c_values = values.view()
    cdef table_view c_keys = keys.view()
    with nogil:
        c_result = cpp_sorting.stable_sort_by_key(
            c_values,
            c_keys,
            c_orders,
            c_null_precedence,
            _cs,
            mr.get_mr()
        )
    return Table.from_libcudf(move(c_result), _stream, mr)


cpdef Table sort(
    Table source_table,
    list column_order,
    list null_precedence,
    object stream=None,
    DeviceMemoryResource mr=None,
):
    """Sorts the table.

    For details, see :cpp:func:`sort`.

    Parameters
    ----------
    source_table : Table
        The table to sort.
    column_order : List[ColumnOrder]
        Whether each column should be sorted in ascending or descending order.
    null_precedence : List[NullOrder]
        Whether nulls should come before or after non-nulls.

    Returns
    -------
    Table
        The sorted table.
    """
    cdef unique_ptr[table] c_result
    cdef vector[order] c_orders = column_order
    cdef vector[null_order] c_null_precedence = null_precedence

    cdef Stream _stream = _get_stream(stream)
    cdef cudaStream_t _cs = _stream.view().value()
    mr = _get_memory_resource(mr)

    cdef table_view c_source_table = source_table.view()
    with nogil:
        c_result = cpp_sorting.sort(
            c_source_table,
            c_orders,
            c_null_precedence,
            _cs,
            mr.get_mr()
        )
    return Table.from_libcudf(move(c_result), _stream, mr)


cpdef Table stable_sort(
    Table source_table,
    list column_order,
    list null_precedence,
    object stream=None,
    DeviceMemoryResource mr=None,
):
    """Sorts the table preserving order of equal elements.

    For details, see :cpp:func:`stable_sort`.

    Parameters
    ----------
    source_table : Table
        The table to sort.
    column_order : List[ColumnOrder]
        Whether each column should be sorted in ascending or descending order.
    null_precedence : List[NullOrder]
        Whether nulls should come before or after non-nulls.

    Returns
    -------
    Table
        The sorted table.
    """
    cdef unique_ptr[table] c_result
    cdef vector[order] c_orders = column_order
    cdef vector[null_order] c_null_precedence = null_precedence

    cdef Stream _stream = _get_stream(stream)
    cdef cudaStream_t _cs = _stream.view().value()
    mr = _get_memory_resource(mr)

    cdef table_view c_source_table = source_table.view()
    with nogil:
        c_result = cpp_sorting.stable_sort(
            c_source_table,
            c_orders,
            c_null_precedence,
            _cs,
            mr.get_mr()
        )
    return Table.from_libcudf(move(c_result), _stream, mr)


cpdef Column top_k(
    Column col,
    size_type k,
    order sort_order = order.DESCENDING,
    object stream=None,
    DeviceMemoryResource mr=None,
):
    """
    Computes the top-k values of a column.

    For details, see :cpp:func:`top_k`.

    Parameters
    ----------
    col : Column
        The input column.
    k : int
        The number of top values to retrieve.
    sort_order : Order, default DESCENDING
        The desired order of the top values. If ASCENDING, the smallest `k` values
        are returned. If DESCENDING, the largest `k` values are returned.

    Returns
    -------
    Column
        A column of the top ``k`` elements from the input.
    """
    cdef unique_ptr[column] c_result

    cdef Stream _stream = _get_stream(stream)
    cdef cudaStream_t _cs = _stream.view().value()
    mr = _get_memory_resource(mr)

    cdef column_view c_col = col.view()
    with nogil:
        c_result = cpp_sorting.top_k(
            c_col,
            k,
            sort_order,
            _cs,
            mr.get_mr()
        )
    return Column.from_libcudf(move(c_result), _stream, mr)


cpdef Column top_k_order(
    Column col,
    size_type k,
    order sort_order = order.DESCENDING,
    object stream=None,
    DeviceMemoryResource mr=None,
):
    """
    Computes the indices of the top-k values of a column.

    This returns the row indices of the top-k elements.

    For details, see :cpp:func:`top_k_order`.

    Parameters
    ----------
    col : Column
        The input column.
    k : int
        The number of top values to retrieve.
    sort_order : Order, default DESCENDING
        The desired order of the top values. If ASCENDING, the indices of the smallest
        `k` values are returned. If DESCENDING, the indices of the largest `k` values
        are returned.

    Returns
    -------
    Column
        A column of the indices of the top ``k`` elements.
    """
    cdef unique_ptr[column] c_result

    cdef Stream _stream = _get_stream(stream)
    cdef cudaStream_t _cs = _stream.view().value()
    mr = _get_memory_resource(mr)

    cdef column_view c_col = col.view()
    with nogil:
        c_result = cpp_sorting.top_k_order(
            c_col,
            k,
            sort_order,
            _cs,
            mr.get_mr()
        )
    return Column.from_libcudf(move(c_result), _stream, mr)
