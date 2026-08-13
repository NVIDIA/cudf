# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from cython.operator cimport dereference
from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move
from libcpp.vector cimport vector
from pylibcudf.libcudf cimport stream_compaction as cpp_stream_compaction
from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.column.column_view cimport column_view
from pylibcudf.libcudf.table.table_view cimport table_view
from pylibcudf.libcudf.stream_compaction cimport duplicate_keep_option
from pylibcudf.libcudf.table.table cimport table
from pylibcudf.libcudf.types cimport (
    nan_equality,
    null_equality,
    size_type,
)

from pylibcudf.libcudf.stream_compaction import \
    duplicate_keep_option as DuplicateKeepOption  # no-cython-lint, isort:skip
from rmm.pylibrmm.memory_resource cimport DeviceMemoryResource
from rmm.pylibrmm.stream cimport Stream

from .column cimport Column
from .expressions cimport Expression
from .table cimport Table
from .utils cimport _get_stream, _get_memory_resource
from cuda.bindings.cyruntime cimport cudaStream_t

__all__ = [
    "DuplicateKeepOption",
    "apply_boolean_mask",
    "apply_deletion_mask",
    "distinct",
    "distinct_indices",
    "drop_nans",
    "drop_nulls",
    "filter",
    "stable_distinct",
    "unique",
]

cpdef Table drop_nulls(
    Table source_table,
    list keys,
    size_type keep_threshold,
    object stream=None,
    DeviceMemoryResource mr=None,
):
    """Filters out rows from the input table based on the presence of nulls.

    For details, see :cpp:func:`drop_nulls`.

    Parameters
    ----------
    source_table : Table
        The input table to filter.
    keys : List[size_type]
        The list of column indexes to consider for null filtering.
    keep_threshold : size_type
        The minimum number of non-nulls required to keep a row.

    Returns
    -------
    Table
        A new table with rows removed based on the null count.
    """
    cdef unique_ptr[table] c_result
    cdef vector[size_type] c_keys = keys

    cdef Stream _stream = _get_stream(stream)
    cdef cudaStream_t _cs = _stream.view().value()
    mr = _get_memory_resource(mr)

    cdef table_view c_source_table = source_table.view()
    with nogil:
        c_result = cpp_stream_compaction.drop_nulls(
            c_source_table, c_keys, keep_threshold, _cs, mr.get_mr()
        )
    return Table.from_libcudf(move(c_result), _stream, mr)


cpdef Table drop_nans(
    Table source_table,
    list keys,
    size_type keep_threshold,
    object stream=None,
    DeviceMemoryResource mr=None,
):
    """Filters out rows from the input table based on the presence of NaNs.

    For details, see :cpp:func:`drop_nans`.

    Parameters
    ----------
    source_table : Table
        The input table to filter.
    keys : List[size_type]
        The list of column indexes to consider for NaN filtering.
    keep_threshold : size_type
        The minimum number of non-NaNs required to keep a row.

    Returns
    -------
    Table
        A new table with rows removed based on NaNs.
    """
    cdef unique_ptr[table] c_result
    cdef vector[size_type] c_keys = keys

    cdef Stream _stream = _get_stream(stream)
    cdef cudaStream_t _cs = _stream.view().value()
    mr = _get_memory_resource(mr)

    cdef table_view c_source_table = source_table.view()
    with nogil:
        c_result = cpp_stream_compaction.drop_nans(
            c_source_table, c_keys, keep_threshold, _cs, mr.get_mr()
        )
    return Table.from_libcudf(move(c_result), _stream, mr)


cpdef Table apply_boolean_mask(
    Table source_table,
    Column boolean_mask,
    object stream=None,
    DeviceMemoryResource mr=None,
):
    """Filters out rows from the input table based on a boolean mask.

    For details, see :cpp:func:`apply_boolean_mask`.

    Parameters
    ----------
    source_table : Table
        The input table to filter.
    boolean_mask : Column
        The boolean mask to apply to the input table.

    Returns
    -------
    Table
        A new table with rows removed based on the boolean mask.
    """
    cdef unique_ptr[table] c_result

    cdef Stream _stream = _get_stream(stream)
    cdef cudaStream_t _cs = _stream.view().value()
    mr = _get_memory_resource(mr)

    cdef table_view c_source_table = source_table.view()
    cdef column_view c_boolean_mask = boolean_mask.view()
    with nogil:
        c_result = cpp_stream_compaction.apply_boolean_mask(
            c_source_table, c_boolean_mask, _cs, mr.get_mr()
        )
    return Table.from_libcudf(move(c_result), _stream, mr)


cpdef Table apply_deletion_mask(
    Table source_table,
    Column deletion_mask,
    object stream=None,
    DeviceMemoryResource mr=None,
):
    """Filters out rows from the input table using a deletion mask.

    For details, see :cpp:func:`apply_deletion_mask`.

    Parameters
    ----------
    source_table : Table
        The input table to filter.
    deletion_mask : Column
        A boolean column used as a deletion mask.

    Returns
    -------
    Table
        Table with rows removed where deletion_mask is true.
    """
    cdef unique_ptr[table] c_result

    cdef Stream _stream = _get_stream(stream)
    cdef cudaStream_t _cs = _stream.view().value()
    mr = _get_memory_resource(mr)

    cdef table_view c_source_table = source_table.view()
    cdef column_view c_deletion_mask = deletion_mask.view()
    with nogil:
        c_result = cpp_stream_compaction.apply_deletion_mask(
            c_source_table, c_deletion_mask, _cs, mr.get_mr()
        )
    return Table.from_libcudf(move(c_result), _stream, mr)


cpdef Table unique(
    Table input,
    list keys,
    duplicate_keep_option keep,
    null_equality nulls_equal,
    object stream=None,
    DeviceMemoryResource mr=None,
):
    """Filter duplicate consecutive rows from the input table.

    For details, see :cpp:func:`unique`.

    Parameters
    ----------
    input : Table
        The input table to filter
    keys : list[int]
        The list of column indexes to consider for filtering.
    keep : duplicate_keep_option
        The option to specify which rows to keep in the case of duplicates.
    nulls_equal : null_equality
        The option to specify how nulls are handled in the comparison.

    Returns
    -------
    Table
        New Table with unique rows from each sequence of equivalent rows
        as specified by keep. In the same order as the input table.

    Notes
    -----
    If the input columns to be filtered on are sorted, then
    unique can produce the same result as stable_distinct, but faster.
    """
    cdef unique_ptr[table] c_result
    cdef vector[size_type] c_keys = keys

    cdef Stream _stream = _get_stream(stream)
    cdef cudaStream_t _cs = _stream.view().value()
    mr = _get_memory_resource(mr)

    cdef table_view c_input = input.view()
    with nogil:
        c_result = cpp_stream_compaction.unique(
            c_input, c_keys, keep, nulls_equal, _cs, mr.get_mr()
        )
    return Table.from_libcudf(move(c_result), _stream, mr)


cpdef Table distinct(
    Table input,
    list keys,
    duplicate_keep_option keep,
    null_equality nulls_equal,
    nan_equality nans_equal,
    object stream=None,
    DeviceMemoryResource mr=None,
):
    """Get the distinct rows from the input table.

    For details, see :cpp:func:`distinct`.

    Parameters
    ----------
    input : Table
        The input table to filter.
    keys : list
        The list of column indexes to consider for distinct filtering.
    keep : duplicate_keep_option
        The option to specify which rows to keep in the case of duplicates.
    nulls_equal : null_equality
        The option to specify how nulls are handled in the comparison.
    nans_equal : nan_equality
        The option to specify how NaNs are handled in the comparison.

    Returns
    -------
    Table
        A new table with distinct rows from the input table. The
        output will not necessarily be in the same order as the input.
    """
    cdef unique_ptr[table] c_result
    cdef vector[size_type] c_keys = keys

    cdef Stream _stream = _get_stream(stream)
    cdef cudaStream_t _cs = _stream.view().value()
    mr = _get_memory_resource(mr)

    cdef table_view c_input = input.view()
    with nogil:
        c_result = cpp_stream_compaction.distinct(
            c_input, c_keys, keep, nulls_equal, nans_equal, _cs,
            mr.get_mr()
        )
    return Table.from_libcudf(move(c_result), _stream, mr)


cpdef Column distinct_indices(
    Table input,
    duplicate_keep_option keep,
    null_equality nulls_equal,
    nan_equality nans_equal,
    object stream=None,
    DeviceMemoryResource mr=None,
):
    """Get the indices of the distinct rows from the input table.

    For details, see :cpp:func:`distinct_indices`.

    Parameters
    ----------
    input : Table
        The input table to filter.
    keep : duplicate_keep_option
        The option to specify which rows to keep in the case of duplicates.
    nulls_equal : null_equality
        The option to specify how nulls are handled in the comparison.
    nans_equal : nan_equality
        The option to specify how NaNs are handled in the comparison.

    Returns
    -------
    Column
        A new column with the indices of the distinct rows from the input table.
    """
    cdef unique_ptr[column] c_result

    cdef Stream _stream = _get_stream(stream)
    cdef cudaStream_t _cs = _stream.view().value()
    mr = _get_memory_resource(mr)

    cdef table_view c_input = input.view()
    with nogil:
        c_result = cpp_stream_compaction.distinct_indices(
            c_input, keep, nulls_equal, nans_equal, _cs, mr.get_mr()
        )
    return Column.from_libcudf(move(c_result), _stream, mr)


cpdef Table stable_distinct(
    Table input,
    list keys,
    duplicate_keep_option keep,
    null_equality nulls_equal,
    nan_equality nans_equal,
    object stream=None,
    DeviceMemoryResource mr=None,
):
    """Get the distinct rows from the input table, preserving input order.

    For details, see :cpp:func:`stable_distinct`.

    Parameters
    ----------
    input : Table
        The input table to filter.
    keys : list
        The list of column indexes to consider for distinct filtering.
    keep : duplicate_keep_option
        The option to specify which rows to keep in the case of duplicates.
    nulls_equal : null_equality
        The option to specify how nulls are handled in the comparison.
    nans_equal : nan_equality
        The option to specify how NaNs are handled in the comparison.

    Returns
    -------
    Table
        A new table with distinct rows from the input table, preserving
        the input table order.
    """
    cdef unique_ptr[table] c_result
    cdef vector[size_type] c_keys = keys

    cdef Stream _stream = _get_stream(stream)
    cdef cudaStream_t _cs = _stream.view().value()
    mr = _get_memory_resource(mr)

    cdef table_view c_input = input.view()
    with nogil:
        c_result = cpp_stream_compaction.stable_distinct(
            c_input, c_keys, keep, nulls_equal, nans_equal, _cs,
            mr.get_mr()
        )
    return Table.from_libcudf(move(c_result), _stream, mr)


cpdef Table filter(
    Table predicate_table,
    Expression predicate_expr,
    Table filter_table,
    object stream=None,
    DeviceMemoryResource mr=None,
):
    """Filters a table using a predicate expression.

    For details, see :cpp:func:`filter`.

    Parameters
    ----------
    predicate_table : Table
        The table used for predicate expression evaluation.
    predicate_expr : Expression
        The predicate filter expression.
    filter_table : Table
        The table to be filtered.

    Returns
    -------
    Table
        The filtered table.
    """
    cdef unique_ptr[table] c_result

    cdef Stream _stream = _get_stream(stream)
    cdef cudaStream_t _cs = _stream.view().value()
    mr = _get_memory_resource(mr)

    cdef table_view c_predicate_table = predicate_table.view()
    cdef table_view c_filter_table = filter_table.view()
    with nogil:
        c_result = cpp_stream_compaction.filter(
            c_predicate_table,
            dereference(predicate_expr.c_obj.get()),
            c_filter_table,
            _cs,
            mr.get_mr()
        )
    return Table.from_libcudf(move(c_result), _stream, mr)

DuplicateKeepOption.__str__ = DuplicateKeepOption.__repr__
