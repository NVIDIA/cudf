# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from cython.operator cimport dereference
from libcpp cimport bool
from libcpp.memory cimport make_unique, unique_ptr
from libcpp.pair cimport pair
from libcpp.utility cimport move
from libcpp.vector cimport vector
from pylibcudf.libcudf cimport rolling as cpp_rolling
from pylibcudf.libcudf.aggregation cimport rolling_aggregation
from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.column.column_view cimport column_view
from pylibcudf.libcudf.table.table cimport table
from pylibcudf.libcudf.table.table_view cimport table_view
from pylibcudf.libcudf.types cimport size_type
from rmm.pylibrmm.stream cimport Stream
from rmm.pylibrmm.memory_resource cimport DeviceMemoryResource

from .aggregation cimport Aggregation
from .column cimport Column
from .scalar cimport Scalar
from .types cimport DataType
from .utils cimport _get_stream, _get_memory_resource
from cuda.bindings.cyruntime cimport cudaStream_t


__all__ = [
    "BoundedClosed",
    "BoundedOpen",
    "CurrentRow",
    "RollingRequest",
    "Unbounded",
    "grouped_range_rolling_window",
    "rolling_window",
]

cdef class Unbounded:
    """
    An unbounded rolling window.

    This window runs to the begin/end of the current row's group.
    """
    def __cinit__(self):
        self.c_obj = move(make_unique[cpp_rolling.unbounded]())

cdef class CurrentRow:
    """
    A current row rolling window.

    This window contains all rows that are equal to the current row in the group.
    """
    def __cinit__(self):
        self.c_obj = move(make_unique[cpp_rolling.current_row]())

cdef class BoundedClosed:
    """
    A bounded closed window.

    This window contains rows with delta of the current row, endpoints included.

    Parameters
    ----------
    delta
        Offset from current row, must be valid. If floating point must not be inf/nan.
    """
    def __cinit__(self, Scalar delta not None):
        self.delta = delta
        self.c_obj = move(
            make_unique[cpp_rolling.bounded_closed](dereference(delta.get()))
        )

cdef class BoundedOpen:
    """
    A bounded open window.

    This window contains rows with delta of the current row, endpoints excluded.

    Parameters
    ----------
    delta
        Offset from current row, must be valid. If floating point must not be inf/nan.
    """
    def __cinit__(self, Scalar delta not None):
        self.delta = delta
        self.c_obj = move(
            make_unique[cpp_rolling.bounded_open](dereference(delta.get()))
        )


cdef class RollingRequest:
    """
    A request for a rolling aggregation.

    Parameters
    ----------
    values
        The column of values to aggregate.
    min_periods
        The minimum number of observations required for a valid result
        in a given window.
    aggregation
        The aggregation to perform.
    """
    def __init__(
            self,
            Column values not None,
            size_type min_periods,
            Aggregation aggregation not None,
    ):
        self.values = values
        self.min_periods = min_periods
        self.aggregation = aggregation

    cdef rolling_request view(self) except *:
        cdef rolling_request c_obj
        c_obj.values = self.values.view()
        c_obj.min_periods = self.min_periods
        c_obj.aggregation = move(self.aggregation.clone_underlying_as_rolling())
        return move(c_obj)


cpdef Table grouped_range_rolling_window(
    Table group_keys,
    Column orderby,
    order order,
    null_order null_order,
    PrecedingRangeWindowType preceding,
    FollowingRangeWindowType following,
    list requests,
    object stream=None,
    DeviceMemoryResource mr=None,
):
    """
    Perform grouping-aware range-based rolling window aggregations on some columns.

    Parameters
    ----------
    group_keys
        Possibly empty table of sorted keys defining groups.
    orderby
        Column defining window ranges. Must be sorted, if
        ``group_keys`` is not empty, must be sorted groupwise.
    order
        Sort order of the ``orderby`` column.
    null_order
        Null sort order in the sorted ``orderby`` column
    preceding
        The type of the preceding window offset.
    following
        The type of the following window offset.
    requests
        List of :class:`RollingRequest` objects.
    stream : Stream | None
        CUDA stream on which to perform the operation.
    mr : DeviceMemoryResource | None
        Device memory resource used to allocate the returned table's device memory.

    Returns
    -------
    A table of results, one column per input request, in order of the
    input requests.
    """
    cdef vector[cpp_rolling.rolling_request] crequests
    cdef unique_ptr[table] result
    crequests.reserve(len(requests))
    for req in requests:
        crequests.push_back(move((<RollingRequest?>req).view()))

    cdef Stream _stream = _get_stream(stream)
    cdef cudaStream_t _cs = _stream.view().value()
    mr = _get_memory_resource(mr)

    cdef table_view c_group_keys = group_keys.view()
    cdef column_view c_orderby = orderby.view()
    with nogil:
        result = cpp_rolling.grouped_range_rolling_window(
            c_group_keys,
            c_orderby,
            order,
            null_order,
            dereference(preceding.c_obj.get()),
            dereference(following.c_obj.get()),
            crequests,
            _cs,
            mr.get_mr()
        )
    return Table.from_libcudf(move(result), _stream, mr)


cpdef Column rolling_window(
    Column source,
    WindowType preceding_window,
    WindowType following_window,
    size_type min_periods,
    Aggregation agg,
    object stream=None,
    DeviceMemoryResource mr=None,
):
    """Perform a rolling window operation on a column

    For details, see ``cudf::rolling_window`` documentation.

    Parameters
    ----------
    source : Column
        The column to perform the rolling window operation on.
    preceding_window : Union[Column, size_type]
        The column containing the preceding window sizes or a scalar value
        indicating the sizes of all windows.
    following_window : Union[Column, size_type]
        The column containing the following window sizes or a scalar value
        indicating the sizes of all windows.
    min_periods : int
        The minimum number of periods to include in the result.
    agg : Aggregation
        The aggregation to perform.
    stream : Stream | None
        CUDA stream on which to perform the operation.
    mr : DeviceMemoryResource | None
        Device memory resource used to allocate the returned column's device memory.

    Returns
    -------
    Column
        The result of the rolling window operation.
    """
    cdef unique_ptr[column] result
    # TODO: Consider making all the conversion functions nogil functions that
    # reclaim the GIL internally for just the necessary scope like column.view()
    cdef const rolling_aggregation *c_agg = agg.view_underlying_as_rolling()

    cdef Stream _stream = _get_stream(stream)
    cdef cudaStream_t _cs = _stream.view().value()
    mr = _get_memory_resource(mr)

    cdef column_view c_source = source.view()
    cdef column_view c_preceding_window
    cdef column_view c_following_window
    if WindowType is Column:
        c_preceding_window = preceding_window.view()
        c_following_window = following_window.view()
        with nogil:
            result = cpp_rolling.rolling_window(
                c_source,
                c_preceding_window,
                c_following_window,
                min_periods,
                dereference(c_agg),
                _cs,
                mr.get_mr()
            )
    else:
        with nogil:
            result = cpp_rolling.rolling_window(
                c_source,
                preceding_window,
                following_window,
                min_periods,
                dereference(c_agg),
                _cs,
                mr.get_mr()
            )

    return Column.from_libcudf(move(result), _stream, mr)


cpdef bool is_valid_rolling_aggregation(DataType source, Aggregation agg):
    """
    Return if a rolling aggregation is supported for a given datatype.

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
    return cpp_rolling.is_valid_rolling_aggregation(source.c_obj, agg.kind())


cpdef tuple make_range_windows(
    Table group_keys,
    Column orderby,
    order order,
    null_order null_order,
    PrecedingRangeWindowType preceding,
    FollowingRangeWindowType following,
    object stream=None,
    DeviceMemoryResource mr=None,
):
    """
    Constructs preceding and following columns given window range specifications.

    Parameters
    ----------
    group_keys
        Possibly empty table of sorted keys defining groups.
    orderby
        Column defining window ranges. Must be sorted, if
       ``group_keys`` is not empty, must be sorted groupwise.
    order
        Sort order of the ``orderby`` column.
    null_order
        Null sort order in the sorted ``orderby`` column
    preceding
        The type of the preceding window offset.
    following
        The type of the following window offset.

    Returns
    -------
    tuple[Column, Column]
        A tuple of preceding and following columns that define the window bounds
        for each row suitable for passing to `rolling_window`.
    """
    cdef pair[unique_ptr[column], unique_ptr[column]] result

    cdef Stream _stream = _get_stream(stream)
    cdef cudaStream_t _cs = _stream.view().value()
    mr = _get_memory_resource(mr)

    cdef table_view c_group_keys = group_keys.view()
    cdef column_view c_orderby = orderby.view()
    with nogil:
        result = cpp_rolling.make_range_windows(
            c_group_keys,
            c_orderby,
            order,
            null_order,
            dereference(preceding.c_obj.get()),
            dereference(following.c_obj.get()),
            _cs,
            mr.get_mr()
        )
    return (
        Column.from_libcudf(move(result.first), _stream, mr),
        Column.from_libcudf(move(result.second), _stream, mr)
    )
