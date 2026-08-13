# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move
from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.column.column_view cimport column_view
from pylibcudf.libcudf.datetime cimport (
    add_calendrical_months as cpp_add_calendrical_months,
    ceil_datetimes as cpp_ceil_datetimes,
    datetime_component,
    day_of_year as cpp_day_of_year,
    days_in_month as cpp_days_in_month,
    extract_datetime_component as cpp_extract_datetime_component,
    extract_quarter as cpp_extract_quarter,
    floor_datetimes as cpp_floor_datetimes,
    is_leap_year as cpp_is_leap_year,
    last_day_of_month as cpp_last_day_of_month,
    round_datetimes as cpp_round_datetimes,
    rounding_frequency,
)

from pylibcudf.libcudf.datetime import \
    datetime_component as DatetimeComponent  # no-cython-lint
from pylibcudf.libcudf.datetime import \
    rounding_frequency as RoundingFrequency  # no-cython-lint

from cython.operator cimport dereference
from rmm.pylibrmm.memory_resource cimport DeviceMemoryResource
from rmm.pylibrmm.stream cimport Stream

from .column cimport Column
from .scalar cimport Scalar
from .utils cimport _get_stream, _get_memory_resource
from cuda.bindings.cyruntime cimport cudaStream_t

__all__ = [
    "DatetimeComponent",
    "RoundingFrequency",
    "add_calendrical_months",
    "ceil_datetimes",
    "day_of_year",
    "days_in_month",
    "extract_datetime_component",
    "extract_quarter",
    "floor_datetimes",
    "is_leap_year",
    "last_day_of_month",
    "round_datetimes",
]

cpdef Column extract_datetime_component(
    Column input,
    datetime_component component,
    object stream=None,
    DeviceMemoryResource mr=None,
):
    """
    Extract a datetime component from a datetime column.

    For details, see :cpp:func:`extract_datetime_component`.

    Parameters
    ----------
    input : Column
        The column to extract the component from.
    component : DatetimeComponent
        The datetime component to extract.
    stream : Stream | None
        CUDA stream on which to perform the operation.

    Returns
    -------
    Column
        Column with the extracted component.
    """
    cdef unique_ptr[column] result

    cdef Stream _stream = _get_stream(stream)
    cdef cudaStream_t _cs = _stream.view().value()
    mr = _get_memory_resource(mr)

    cdef column_view c_input = input.view()
    with nogil:
        result = cpp_extract_datetime_component(
            c_input, component, _cs, mr.get_mr()
        )
    return Column.from_libcudf(move(result), _stream, mr)

cpdef Column ceil_datetimes(
    Column input,
    rounding_frequency freq,
    object stream=None,
    DeviceMemoryResource mr=None,
):
    """
    Round datetimes up to the nearest multiple of the given frequency.

    For details, see :cpp:func:`ceil_datetimes`.

    Parameters
    ----------
    input : Column
        The column of input datetime values.
    freq : rounding_frequency
        The frequency to round up to.
    stream : Stream | None
        CUDA stream on which to perform the operation.

    Returns
    -------
    Column
        Column of the same datetime resolution as the input column.
    """
    cdef unique_ptr[column] result

    cdef Stream _stream = _get_stream(stream)
    cdef cudaStream_t _cs = _stream.view().value()
    mr = _get_memory_resource(mr)

    cdef column_view c_input = input.view()
    with nogil:
        result = cpp_ceil_datetimes(c_input, freq, _cs, mr.get_mr())
    return Column.from_libcudf(move(result), _stream, mr)

cpdef Column floor_datetimes(
    Column input,
    rounding_frequency freq,
    object stream=None,
    DeviceMemoryResource mr=None,
):
    """
    Round datetimes down to the nearest multiple of the given frequency.

    For details, see :cpp:func:`floor_datetimes`.

    Parameters
    ----------
    input : Column
        The column of input datetime values.
    freq : rounding_frequency
        The frequency to round down to.
    stream : Stream | None
        CUDA stream on which to perform the operation.

    Returns
    -------
    Column
        Column of the same datetime resolution as the input column.
    """
    cdef unique_ptr[column] result

    cdef Stream _stream = _get_stream(stream)
    cdef cudaStream_t _cs = _stream.view().value()
    mr = _get_memory_resource(mr)

    cdef column_view c_input = input.view()
    with nogil:
        result = cpp_floor_datetimes(c_input, freq, _cs, mr.get_mr())
    return Column.from_libcudf(move(result), _stream, mr)

cpdef Column round_datetimes(
    Column input,
    rounding_frequency freq,
    object stream=None,
    DeviceMemoryResource mr=None,
):
    """
    Round datetimes to the nearest multiple of the given frequency.

    For details, see :cpp:func:`round_datetimes`.

    Parameters
    ----------
    input : Column
        The column of input datetime values.
    freq : rounding_frequency
        The frequency to round to.
    stream : Stream | None
        CUDA stream on which to perform the operation.

    Returns
    -------
    Column
        Column of the same datetime resolution as the input column.
    """
    cdef unique_ptr[column] result

    cdef Stream _stream = _get_stream(stream)
    cdef cudaStream_t _cs = _stream.view().value()
    mr = _get_memory_resource(mr)

    cdef column_view c_input = input.view()
    with nogil:
        result = cpp_round_datetimes(c_input, freq, _cs, mr.get_mr())
    return Column.from_libcudf(move(result), _stream, mr)

cpdef Column add_calendrical_months(
    Column input,
    ColumnOrScalar months,
    object stream=None,
    DeviceMemoryResource mr=None,
):
    """
    Adds or subtracts a number of months from the datetime
    type and returns a timestamp column that is of the same
    type as the input timestamps column.

    For details, see :cpp:func:`add_calendrical_months`.

    Parameters
    ----------
    input : Column
        The column of input timestamp values.
    months : ColumnOrScalar
        The number of months to add.
    stream : Stream | None
        CUDA stream on which to perform the operation.

    Returns
    -------
    Column
        Column of computed timestamps.
    """
    cdef column_view c_input
    cdef column_view c_months_column

    if not isinstance(months, (Column, Scalar)):
        raise TypeError("Must pass a Column or Scalar")

    cdef unique_ptr[column] result

    cdef Stream _stream = _get_stream(stream)
    cdef cudaStream_t _cs = _stream.view().value()
    mr = _get_memory_resource(mr)

    c_input = input.view()
    if ColumnOrScalar is Column:
        c_months_column = months.view()
    with nogil:
        result = cpp_add_calendrical_months(
            c_input,
            c_months_column if ColumnOrScalar is Column else
            dereference(months.get()),
            _cs,
            mr.get_mr()
        )
    return Column.from_libcudf(move(result), _stream, mr)

cpdef Column day_of_year(
    Column input, object stream=None, DeviceMemoryResource mr=None
):
    """
    Computes the day number since the start of
    the year from the datetime. The value is between
    [1, {365-366}].

    For details, see :cpp:func:`day_of_year`.

    Parameters
    ----------
    input : Column
        The column of input datetime values.
    stream : Stream | None
        CUDA stream on which to perform the operation.

    Returns
    -------
    Column
        Column of day numbers.
    """
    cdef unique_ptr[column] result

    cdef Stream _stream = _get_stream(stream)
    cdef cudaStream_t _cs = _stream.view().value()
    mr = _get_memory_resource(mr)

    cdef column_view c_input = input.view()
    with nogil:
        result = cpp_day_of_year(c_input, _cs, mr.get_mr())
    return Column.from_libcudf(move(result), _stream, mr)

cpdef Column is_leap_year(
    Column input, object stream=None, DeviceMemoryResource mr=None
):
    """
    Check if the year of the given date is a leap year.

    For details, see :cpp:func:`is_leap_year`.

    Parameters
    ----------
    input : Column
        The column of input datetime values.
    stream : Stream | None
        CUDA stream on which to perform the operation.

    Returns
    -------
    Column
        Column of bools indicating whether the given year
        is a leap year.
    """
    cdef unique_ptr[column] result

    cdef Stream _stream = _get_stream(stream)
    cdef cudaStream_t _cs = _stream.view().value()
    mr = _get_memory_resource(mr)

    cdef column_view c_input = input.view()
    with nogil:
        result = cpp_is_leap_year(c_input, _cs, mr.get_mr())
    return Column.from_libcudf(move(result), _stream, mr)

cpdef Column last_day_of_month(
    Column input, object stream=None, DeviceMemoryResource mr=None
):
    """
    Computes the last day of the month.

    For details, see :cpp:func:`last_day_of_month`.

    Parameters
    ----------
    input : Column
        The column of input datetime values.
    stream : Stream | None
        CUDA stream on which to perform the operation.

    Returns
    -------
    Column
        Column of ``TIMESTAMP_DAYS`` representing the last day
        of the month.
    """
    cdef unique_ptr[column] result

    cdef Stream _stream = _get_stream(stream)
    cdef cudaStream_t _cs = _stream.view().value()
    mr = _get_memory_resource(mr)

    cdef column_view c_input = input.view()
    with nogil:
        result = cpp_last_day_of_month(c_input, _cs, mr.get_mr())
    return Column.from_libcudf(move(result), _stream, mr)

cpdef Column extract_quarter(
    Column input, object stream=None, DeviceMemoryResource mr=None
):
    """
    Returns the quarter (ie. a value from {1, 2, 3, 4})
    that the date is in.

    For details, see :cpp:func:`extract_quarter`.

    Parameters
    ----------
    input : Column
        The column of input datetime values.
    stream : Stream | None
        CUDA stream on which to perform the operation.

    Returns
    -------
    Column
        Column indicating which quarter the date is in.
    """
    cdef unique_ptr[column] result

    cdef Stream _stream = _get_stream(stream)
    cdef cudaStream_t _cs = _stream.view().value()
    mr = _get_memory_resource(mr)

    cdef column_view c_input = input.view()
    with nogil:
        result = cpp_extract_quarter(c_input, _cs, mr.get_mr())
    return Column.from_libcudf(move(result), _stream, mr)

cpdef Column days_in_month(
    Column input, object stream=None, DeviceMemoryResource mr=None
):
    """
    Extract the number of days in the month.

    For details, see :cpp:func:`days_in_month`.

    Parameters
    ----------
    input : Column
        The column of input datetime values.
    stream : Stream | None
        CUDA stream on which to perform the operation.

    Returns
    -------
    Column
        Column of the number of days in the given month.
    """
    cdef unique_ptr[column] result

    cdef Stream _stream = _get_stream(stream)
    cdef cudaStream_t _cs = _stream.view().value()
    mr = _get_memory_resource(mr)

    cdef column_view c_input = input.view()
    with nogil:
        result = cpp_days_in_month(c_input, _cs, mr.get_mr())
    return Column.from_libcudf(move(result), _stream, mr)

DatetimeComponent.__str__ = DatetimeComponent.__repr__
RoundingFrequency.__str__ = RoundingFrequency.__repr__
