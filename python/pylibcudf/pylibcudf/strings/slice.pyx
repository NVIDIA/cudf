# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from libcpp.memory cimport unique_ptr
from libcpp.optional cimport make_optional, nullopt, optional
from libcpp.utility cimport move
from pylibcudf.column cimport Column
from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.column.column_view cimport column_view
from pylibcudf.libcudf.strings cimport substring as cpp_slice
from pylibcudf.libcudf.types cimport size_type

from rmm.pylibrmm.memory_resource cimport DeviceMemoryResource
from rmm.pylibrmm.stream cimport Stream

from ..utils cimport _get_stream, _get_memory_resource
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pylibcudf.typing import CudaStreamLike
from cuda.bindings.cyruntime cimport cudaStream_t

__all__ = ["slice_strings"]

cpdef Column slice_strings(
    Column input,
    ColumnOrScalar start=None,
    ColumnOrScalar stop=None,
    object step=None,
    object stream: CudaStreamLike | None = None,
    DeviceMemoryResource mr=None,
):
    """Perform a slice operation on a strings column.

    ``start`` and ``stop`` may be a
    :py:class:`~pylibcudf.column.Column` or an ``int``. But ``step``
    must be an ``int``.

    For details, see :cpp:func:`slice_strings`.

    Parameters
    ----------
    input : Column
        Strings column for this operation
    start : Union[Column, int]
        The start character position or positions.
    stop : Union[Column, int]
        The end character position or positions
    step : int
        Distance between input characters retrieved
    stream : Stream | None
        CUDA stream on which to perform the operation.

    Returns
    -------
    pylibcudf.Column
        The result of the slice operation
    """
    cdef unique_ptr[column] c_result
    cdef Stream _stream = _get_stream(stream)
    cdef cudaStream_t _cs = _stream.view().get()
    mr = _get_memory_resource(mr)
    cdef column_view c_input
    cdef column_view c_start
    cdef column_view c_stop
    cdef optional[size_type] c_start_scalar = nullopt
    cdef optional[size_type] c_stop_scalar = nullopt
    cdef optional[size_type] c_step_scalar = nullopt

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

    else:
        if not (isinstance(start, int) or start is None):
            raise ValueError("start, stop, and step must be either Column or int")
        if not (isinstance(stop, int) or stop is None):
            raise ValueError("start, stop, and step must be either Column or int")
        if not (isinstance(step, int) or step is None):
            raise ValueError("start, stop, and step must be either Column or int")

        if start is not None:
            c_start_scalar = make_optional[size_type](<size_type>start)
        if stop is not None:
            c_stop_scalar = make_optional[size_type](<size_type>stop)
        if step is not None:
            c_step_scalar = make_optional[size_type](<size_type>step)

        c_input = input.view()
        with nogil:
            c_result = cpp_slice.slice_strings(
                c_input,
                c_start_scalar,
                c_stop_scalar,
                c_step_scalar,
                _cs,
                mr.get_mr()
            )

    return Column.from_libcudf(move(c_result), _stream, mr)
