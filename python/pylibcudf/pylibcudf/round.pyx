# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from libc.stdint cimport int32_t
from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move
from pylibcudf.libcudf.round cimport (
    round as cpp_round,
    round_decimal as cpp_round_decimal,
    rounding_method,
)

from pylibcudf.libcudf.round import \
    rounding_method as RoundingMethod  # no-cython-lint

from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.column.column_view cimport column_view

from rmm.pylibrmm.stream cimport Stream
from rmm.pylibrmm.memory_resource cimport DeviceMemoryResource

from .column cimport Column
from .utils cimport _get_stream, _get_memory_resource
from cuda.bindings.cyruntime cimport cudaStream_t

__all__ = ["RoundingMethod", "round"]

cpdef Column round(
    Column source,
    int32_t decimal_places = 0,
    rounding_method round_method = rounding_method.HALF_UP,
    object stream=None,
    DeviceMemoryResource mr=None
):
    """Rounds all the values in a column to the specified number of decimal places.

    .. deprecated:: release 26.04
        round is deprecated for float type.
        Use ``round_decimal`` for decimal and integer types.

    For details, see :cpp:func:`round`.

    Parameters
    ----------
    source : Column
        The Column for which to round values.
    decimal_places: int32_t, optional
        The number of decimal places to round to (default 0)
    round_method: rounding_method, optional
        The method by which to round each value.
        Can be one of { RoundingMethod.HALF_UP, RoundingMethod.HALF_EVEN }
        (default rounding_method.HALF_UP)
    stream : Stream | None
        CUDA stream on which to perform the operation.
    mr : DeviceMemoryResource | None
        Device memory resource used to allocate the returned column's device memory.

    Returns
    -------
    pylibcudf.Column
        A Column with values rounded
    """
    cdef unique_ptr[column] c_result
    cdef Stream _stream = _get_stream(stream)
    cdef cudaStream_t _cs = _stream.view().value()
    mr = _get_memory_resource(mr)

    cdef column_view c_source = source.view()
    with nogil:
        c_result = cpp_round(
            c_source,
            decimal_places,
            round_method,
            _cs,
            mr.get_mr()
        )

    return Column.from_libcudf(move(c_result), _stream, mr)


cpdef Column round_decimal(
    Column source,
    int32_t decimal_places = 0,
    rounding_method round_method = rounding_method.HALF_UP,
    object stream=None,
    DeviceMemoryResource mr=None
):
    """Rounds all the values in a column to the specified number of decimal places.
    Only decimal and integer types are supported.

    For details, see :cpp:func:`round_decimal`.

    Parameters
    ----------
    source : Column
        The Column for which to round values.
    decimal_places: int32_t, optional
        The number of decimal places to round to (default 0)
    round_method: rounding_method, optional
        The method by which to round each value.
        Can be one of { RoundingMethod.HALF_UP, RoundingMethod.HALF_EVEN }
        (default rounding_method.HALF_UP)
    stream : Stream | None
        CUDA stream on which to perform the operation.
    mr : DeviceMemoryResource | None
        Device memory resource used to allocate the returned column's device memory.

    Returns
    -------
    pylibcudf.Column
        A Column with values rounded
    """
    cdef unique_ptr[column] c_result
    cdef Stream _stream = _get_stream(stream)
    cdef cudaStream_t _cs = _stream.view().value()
    mr = _get_memory_resource(mr)

    cdef column_view c_source = source.view()
    with nogil:
        c_result = cpp_round_decimal(
            c_source,
            decimal_places,
            round_method,
            _cs,
            mr.get_mr()
        )

    return Column.from_libcudf(move(c_result), _stream, mr)

RoundingMethod.__str__ = RoundingMethod.__repr__
