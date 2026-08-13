# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from cython.operator cimport dereference
from libcpp.functional cimport reference_wrapper
from libcpp cimport bool
from libcpp.memory cimport unique_ptr
from libcpp.optional cimport optional, nullopt
from libcpp.utility cimport move, pair
from pylibcudf.libcudf cimport distinct_count as cpp_distinct_count
from pylibcudf.libcudf cimport unique_count as cpp_unique_count
from pylibcudf.libcudf.aggregation cimport reduce_aggregation, scan_aggregation
from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.column.column_view cimport column_view
from pylibcudf.libcudf.table.table_view cimport table_view
from pylibcudf.libcudf.reduce cimport (
    reduce as cpp_reduce,
    scan as cpp_scan,
    minmax as cpp_minmax,
    scan_type,
    constscalar,
    is_valid_aggregation as cpp_is_valid_aggregation,
)
from pylibcudf.libcudf.scalar.scalar cimport scalar
from pylibcudf.libcudf.types cimport nan_policy, null_policy, size_type
from rmm.pylibrmm.stream cimport Stream
from rmm.pylibrmm.memory_resource cimport DeviceMemoryResource

from .aggregation cimport Aggregation
from .column cimport Column
from .scalar cimport Scalar
from .types cimport DataType
from .utils cimport _get_stream, _get_memory_resource

from pylibcudf.libcudf.reduce import scan_type as ScanType  # no-cython-lint
from cuda.bindings.cyruntime cimport cudaStream_t

__all__ = [
    "ScanType",
    "distinct_count",
    "is_valid_reduce_aggregation",
    "minmax",
    "reduce",
    "scan",
    "unique_count",
]

cpdef Scalar reduce(
    Column col,
    Aggregation agg,
    DataType data_type,
    Scalar init=None,
    object stream=None,
    DeviceMemoryResource mr=None,
):
    """Perform a reduction on a column

    For details, see ``cudf::reduce`` documentation.

    Parameters
    ----------
    col : Column
        The column to perform the reduction on.
    agg : Aggregation
        The aggregation to perform.
    data_type : DataType
        The data type of the result.
    init : Scalar | None
        The initial value for the reduction.
    stream : Stream | None
        CUDA stream on which to perform the operation.
    mr : DeviceMemoryResource | None
        Device memory resource used to allocate the returned scalar's device memory.

    Returns
    -------
    Scalar
        The result of the reduction.
    """
    cdef unique_ptr[scalar] result
    cdef const reduce_aggregation *c_agg = agg.view_underlying_as_reduce()
    cdef optional[reference_wrapper[constscalar]] c_init
    cdef const scalar* c_init_ptr

    cdef Stream _stream = _get_stream(stream)
    cdef cudaStream_t _cs = _stream.view().value()
    mr = _get_memory_resource(mr)

    if init is not None:
        c_init_ptr = init.get()
        c_init = optional[reference_wrapper[constscalar]](
            reference_wrapper[constscalar](dereference(c_init_ptr))
        )
    else:
        c_init = nullopt

    cdef column_view c_col = col.view()
    with nogil:
        result = cpp_reduce(
            c_col,
            dereference(c_agg),
            data_type.c_obj,
            c_init,
            _cs,
            mr.get_mr()
        )
    return Scalar.from_libcudf(move(result))


cpdef Column scan(
    Column col,
    Aggregation agg,
    scan_type inclusive,
    object stream=None,
    DeviceMemoryResource mr=None,
):
    """Perform a scan on a column

    For details, see ``cudf::scan`` documentation.

    Parameters
    ----------
    col : Column
        The column to perform the scan on.
    agg : Aggregation
        The aggregation to perform.
    inclusive : scan_type
        The type of scan to perform.
    stream : Stream | None
        CUDA stream on which to perform the operation.
    mr : DeviceMemoryResource | None
        Device memory resource used to allocate the returned column's device memory.

    Returns
    -------
    Column
        The result of the scan.
    """
    cdef unique_ptr[column] result
    cdef const scan_aggregation *c_agg = agg.view_underlying_as_scan()

    cdef Stream _stream = _get_stream(stream)
    cdef cudaStream_t _cs = _stream.view().value()
    mr = _get_memory_resource(mr)

    cdef column_view c_col = col.view()
    with nogil:
        result = cpp_scan(
            c_col,
            dereference(c_agg),
            inclusive,
            null_policy.EXCLUDE,
            _cs,
            mr.get_mr()
        )
    return Column.from_libcudf(move(result), _stream, mr)


cpdef tuple minmax(Column col, object stream=None, DeviceMemoryResource mr=None):
    """Compute the minimum and maximum of a column

    For details, see ``cudf::minmax`` documentation.

    Parameters
    ----------
    col : Column
        The column to compute the minimum and maximum of.
    stream : Stream | None
        CUDA stream on which to perform the operation.
    mr : DeviceMemoryResource | None
        Device memory resource used to allocate the returned scalars' device memory.

    Returns
    -------
    tuple
        A tuple of two Scalars, the first being the minimum and the second
        being the maximum.
    """
    cdef pair[unique_ptr[scalar], unique_ptr[scalar]] result
    cdef Scalar min_scalar
    cdef Scalar max_scalar

    cdef Stream _stream = _get_stream(stream)
    cdef cudaStream_t _cs = _stream.view().value()
    mr = _get_memory_resource(mr)

    cdef column_view c_col = col.view()
    with nogil:
        result = cpp_minmax(c_col, _cs, mr.get_mr())

    min_scalar = Scalar.from_libcudf(move(result.first))
    max_scalar = Scalar.from_libcudf(move(result.second))
    return (min_scalar, max_scalar)


cpdef bool is_valid_reduce_aggregation(DataType source, Aggregation agg):
    """
    Return if an aggregation is supported for a given datatype.

    Parameters
    ----------
    source
        The type of the column the aggregation is being performed on.
    agg
        The aggregation.

    Returns
    -------
    True if the aggregation is supported.
    """
    return cpp_is_valid_aggregation(source.c_obj, agg.kind())


cpdef size_type unique_count(
    Column source,
    null_policy null_handling,
    nan_policy nan_handling,
    object stream=None
):
    """Returns the number of unique consecutive elements in the input column.

    For details, see :cpp:func:`cudf::unique_count`.

    Parameters
    ----------
    source : Column
        The input column to count the unique elements of.
    null_handling : null_policy
        Flag to include or exclude nulls from the count. If included, all
        nulls compare equal.
    nan_handling : nan_policy
        Whether to treat NaNs as null, or valid elements. If valid all NaNs
        compare equal.

    Returns
    -------
    size_type
        The number of unique consecutive elements in the input column.

    Notes
    -----
    If the input column is sorted, then unique_count can produce the
    same result as distinct_count, but faster.
    """
    cdef Stream _stream = _get_stream(stream)
    cdef column_view c_source = source.view()

    with nogil:
        return cpp_unique_count.unique_count(
            c_source, null_handling, nan_handling, _stream.view().value()
        )


cpdef size_type distinct_count(
    Column source,
    null_policy null_handling,
    nan_policy nan_handling,
    object stream=None
):
    """Returns the number of distinct elements in the input column.

    For details, see :cpp:func:`cudf::distinct_count`.

    Parameters
    ----------
    source : Column
        The input column to count the unique elements of.
    null_handling : null_policy
        Flag to include or exclude nulls from the count. If included, all
        nulls compare equal.
    nan_handling : nan_policy
        Whether to treat NaNs as null, or valid elements. If valid all NaNs
        compare equal.

    Returns
    -------
    size_type
        The number of distinct elements in the input column.
    """
    cdef Stream _stream = _get_stream(stream)
    cdef column_view c_source = source.view()

    with nogil:
        return cpp_distinct_count.distinct_count(
            c_source, null_handling, nan_handling, _stream.view().value()
        )


cpdef size_type unique_count_table(
    Table source,
    null_equality nulls_equal,
    object stream=None
):
    """Returns the number of unique consecutive rows in the input table.

    For details, see :cpp:func:`cudf::unique_count`.

    Parameters
    ----------
    source : Table
        The input table to count the unique elements of.
    nulls_equal : null_equality
        Whether nulls should compare equal.

    Returns
    -------
    size_type
        The number of unique consecutive rows.

    Notes
    -----
    NaNs compare equal in this comparison.
    """
    cdef Stream _stream = _get_stream(stream)
    cdef table_view c_source = source.view()

    with nogil:
        return cpp_unique_count.unique_count(
            c_source, nulls_equal, _stream.view().value()
        )


cpdef size_type distinct_count_table(
    Table source,
    null_equality nulls_equal,
    object stream=None
):
    """Returns the number of distinct rows in the input table.

    For details, see :cpp:func:`cudf::distinct_count`.

    Parameters
    ----------
    source : Table
        The input table to count the unique rows of.
    nulls_equal : null_equality
        Whether nulls should compare equal.

    Returns
    -------
    size_type
        The number of distinct rows.

    Notes
    -----
    NaNs compare equal in this comparison.
    """
    cdef Stream _stream = _get_stream(stream)
    cdef table_view c_source = source.view()

    with nogil:
        return cpp_distinct_count.distinct_count(
            c_source, nulls_equal, _stream.view().value()
        )


ScanType.__str__ = ScanType.__repr__
