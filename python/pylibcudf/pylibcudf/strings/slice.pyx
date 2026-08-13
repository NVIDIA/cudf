# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move
from pylibcudf.column cimport Column
from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.column.column_view cimport column_view
from pylibcudf.libcudf.scalar.scalar cimport numeric_scalar
from pylibcudf.libcudf.scalar.scalar_factories cimport (
    make_fixed_width_scalar as cpp_make_fixed_width_scalar,
)
from pylibcudf.libcudf.strings cimport substring as cpp_slice
from pylibcudf.libcudf.types cimport size_type
from pylibcudf.scalar cimport Scalar

from cython.operator import dereference
from rmm.pylibrmm.memory_resource cimport DeviceMemoryResource
from rmm.pylibrmm.stream cimport Stream

from ..utils cimport _get_stream, _get_memory_resource
from cuda.bindings.cyruntime cimport cudaStream_t

__all__ = ["slice_strings"]

cpdef Column slice_strings(
    Column input,
    ColumnOrScalar start=None,
    ColumnOrScalar stop=None,
    Scalar step=None,
    object stream=None,
    DeviceMemoryResource mr=None,
):
    """Perform a slice operation on a strings column.

    ``start`` and ``stop`` may be a
    :py:class:`~pylibcudf.column.Column` or a
    :py:class:`~pylibcudf.scalar.Scalar`. But ``step`` must be a
    :py:class:`~pylibcudf.scalar.Scalar`.

    For details, see :cpp:func:`slice_strings`.

    Parameters
    ----------
    input : Column
        Strings column for this operation
    start : Union[Column, Scalar]
        The start character position or positions.
    stop : Union[Column, Scalar]
        The end character position or positions
    step : Scalar
        Distance between input characters retrieved
    stream : Stream | None
        CUDA stream on which to perform the operation.

    Returns
    -------
    pylibcudf.Column
        The result of the slice operation
    """
    cdef unique_ptr[column] c_result
    cdef numeric_scalar[size_type]* cpp_start
    cdef numeric_scalar[size_type]* cpp_stop
    cdef numeric_scalar[size_type]* cpp_step
    cdef Stream _stream = _get_stream(stream)
    cdef cudaStream_t _cs = _stream.view().value()
    mr = _get_memory_resource(mr)
    cdef column_view c_input
    cdef column_view c_start
    cdef column_view c_stop

    if input is None:
        raise ValueError("input cannot be None")

    if ColumnOrScalar is Column:
        if step is not None:
            raise ValueError("Column-wise slice does not support step")

        if start is None or stop is None:
            raise ValueError(
                "start and stop must be provided for Column-wise slice"
            )

        c_input = input.view()
        c_start = start.view()
        c_stop = stop.view()
        with nogil:
            c_result = cpp_slice.slice_strings(
                c_input,
                c_start,
                c_stop,
                _cs,
                mr.get_mr()
            )

    elif ColumnOrScalar is Scalar:
        if start is None:
            start = Scalar.from_libcudf(
                cpp_make_fixed_width_scalar(0, _stream.view().value(), mr.get_mr())
            )
        if stop is None:
            stop = Scalar.from_libcudf(
                cpp_make_fixed_width_scalar(0, _stream.view().value(), mr.get_mr())
            )
        if step is None:
            step = Scalar.from_libcudf(
                cpp_make_fixed_width_scalar(1, _stream.view().value(), mr.get_mr())
            )

        cpp_start = <numeric_scalar[size_type]*>start.c_obj.get()
        cpp_stop = <numeric_scalar[size_type]*>stop.c_obj.get()
        cpp_step = <numeric_scalar[size_type]*>step.c_obj.get()

        c_input = input.view()
        with nogil:
            c_result = cpp_slice.slice_strings(
                c_input,
                dereference(cpp_start),
                dereference(cpp_stop),
                dereference(cpp_step),
                _cs,
                mr.get_mr()
            )
    else:
        raise ValueError("start, stop, and step must be either Column or Scalar")

    return Column.from_libcudf(move(c_result), _stream, mr)
