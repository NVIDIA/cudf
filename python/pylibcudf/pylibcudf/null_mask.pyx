# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from libc.stdint cimport uintptr_t
from libcpp.memory cimport make_unique
from libcpp.pair cimport pair
from libcpp.utility cimport move
from pylibcudf.libcudf cimport null_mask as cpp_null_mask
from pylibcudf.libcudf.column.column_view cimport column_view
from pylibcudf.libcudf.table.table_view cimport table_view
from pylibcudf.libcudf.types cimport mask_state, size_type, bitmask_type

from rmm.librmm.device_buffer cimport device_buffer
from rmm.pylibrmm.device_buffer cimport DeviceBuffer
from rmm.pylibrmm.stream cimport Stream
from rmm.pylibrmm.memory_resource cimport DeviceMemoryResource

from pylibcudf.libcudf.types import mask_state as MaskState  # no-cython-lint

from .span import is_span as py_is_span

from .column cimport Column
from .table cimport Table
from .utils cimport _get_stream, _get_memory_resource
from cuda.bindings.cyruntime cimport cudaStream_t

__all__ = [
    "bitmask_allocation_size_bytes",
    "bitmask_and",
    "bitmask_or",
    "copy_bitmask",
    "create_null_mask",
    "null_count",
    "index_of_first_set_bit",
]

cdef DeviceBuffer buffer_to_python(
    device_buffer buf, object stream, DeviceMemoryResource mr
):
    return DeviceBuffer.c_from_unique_ptr(
        make_unique[device_buffer](move(buf)), stream, mr
    )


cpdef DeviceBuffer copy_bitmask(
    Column col,
    object stream=None,
    DeviceMemoryResource mr=None
):
    """Copies ``col``'s bitmask into a ``DeviceBuffer``.

    For details, see :cpp:func:`copy_bitmask`.

    Parameters
    ----------
    col : Column
        Column whose bitmask needs to be copied
    stream : Stream | None
        CUDA stream on which to perform the operation.
    mr : DeviceMemoryResource | None
        Device memory resource for allocations.

    Returns
    -------
    rmm.DeviceBuffer
        A ``DeviceBuffer`` containing ``col``'s bitmask, or an empty
        ``DeviceBuffer`` if ``col`` is not nullable
    """
    cdef device_buffer db
    cdef Stream _stream = _get_stream(stream)
    cdef cudaStream_t _cs = _stream.view().value()
    mr = _get_memory_resource(mr)

    cdef column_view c_col = col.view()
    with nogil:
        db = cpp_null_mask.copy_bitmask(c_col, _cs, mr.get_mr())

    return buffer_to_python(move(db), _stream, mr)


cpdef DeviceBuffer copy_bitmask_from_bitmask(
    object bitmask,
    size_type begin_bit,
    size_type end_bit,
    object stream=None,
    DeviceMemoryResource mr=None
):
    """Copies a portion of a bitmask into a ``DeviceBuffer``.

    For details, see :cpp:func:`copy_bitmask`.

    Parameters
    ----------
    bitmask : Span-like object
        Object with ptr and size attributes (e.g., gpumemoryview, Buffer, DeviceBuffer).
    begin_bit : size_type
        The starting bit index (inclusive).
    end_bit : size_type
        The ending bit index (exclusive).
    stream : Stream | None
        CUDA stream on which to perform the operation.
    mr : DeviceMemoryResource | None
        Device memory resource for allocations.

    Returns
    -------
    rmm.DeviceBuffer
        A ``DeviceBuffer`` containing ``col``'s bitmask, or an empty
        ``DeviceBuffer`` if ``col`` is not nullable
    """
    if not py_is_span(bitmask):
        raise TypeError(
            f"bitmask must satisfy Span protocol (have .ptr and .size), "
            f"got {type(bitmask).__name__}"
        )
    cdef device_buffer db
    cdef Stream _stream = _get_stream(stream)
    cdef cudaStream_t _cs = _stream.view().value()
    mr = _get_memory_resource(mr)
    cdef uintptr_t ptr = bitmask.ptr

    with nogil:
        db = cpp_null_mask.copy_bitmask(
            <bitmask_type*>ptr,
            begin_bit,
            end_bit,
            _cs,
            mr.get_mr()
        )

    return buffer_to_python(move(db), _stream, mr)


cpdef size_t bitmask_allocation_size_bytes(size_type number_of_bits):
    """
    Computes the required bytes necessary to represent the specified number of bits
    with a 64B padding boundary.

    For details, see :cpp:func:`bitmask_allocation_size_bytes`.

    Parameters
    ----------
    number_of_bits : size_type
        The number of bits that need to be represented

    Returns
    -------
    size_t
        The necessary number of bytes
    """
    with nogil:
        return cpp_null_mask.bitmask_allocation_size_bytes(number_of_bits, 64)


cpdef DeviceBuffer create_null_mask(
    size_type size,
    mask_state state = mask_state.UNINITIALIZED,
    object stream=None,
    DeviceMemoryResource mr=None
):
    """Creates a ``DeviceBuffer`` for use as a null value indicator bitmask of a
    ``Column``.

    For details, see :cpp:func:`create_null_mask`.

    Parameters
    ----------
    size : size_type
        The number of elements to be represented by the mask
    state : mask_state, optional
        The desired state of the mask. Can be one of { MaskState.UNALLOCATED,
        MaskState.UNINITIALIZED, MaskState.ALL_VALID, MaskState.ALL_NULL }
        (default MaskState.UNINITIALIZED)
    stream : Stream | None
        CUDA stream on which to perform the operation.
    mr : DeviceMemoryResource | None
        Device memory resource for allocations.

    Returns
    -------
    rmm.DeviceBuffer
        A ``DeviceBuffer`` for use as a null bitmask satisfying the desired size and
        state
    """
    cdef device_buffer db
    cdef Stream _stream = _get_stream(stream)
    cdef cudaStream_t _cs = _stream.view().value()
    mr = _get_memory_resource(mr)

    with nogil:
        db = cpp_null_mask.create_null_mask(size, state, _cs, mr.get_mr())

    return buffer_to_python(move(db), _stream, mr)


cpdef tuple bitmask_and(columns, object stream=None, DeviceMemoryResource mr=None):
    """Performs bitwise AND of the bitmasks of a list of columns.

    For details, see :cpp:func:`bitmask_and`.

    Parameters
    ----------
    columns : Sequence[Column]
        The columns
    stream : Stream | None
        CUDA stream on which to perform the operation.
    mr : DeviceMemoryResource | None
        Device memory resource for allocations.

    Returns
    -------
    tuple[DeviceBuffer, size_type]
        A tuple of the resulting mask and count of unset bits
    """
    cdef Table c_table = Table(columns)
    cdef pair[device_buffer, size_type] c_result
    cdef Stream _stream = _get_stream(stream)
    cdef cudaStream_t _cs = _stream.view().value()
    mr = _get_memory_resource(mr)

    cdef table_view c_input = c_table.view()
    with nogil:
        c_result = cpp_null_mask.bitmask_and(
            c_input, _cs, mr.get_mr()
        )

    return buffer_to_python(move(c_result.first), _stream, mr), c_result.second


cpdef tuple bitmask_or(columns, object stream=None, DeviceMemoryResource mr=None):
    """Performs bitwise OR of the bitmasks of a list of columns.

    For details, see :cpp:func:`bitmask_or`.

    Parameters
    ----------
    columns : Sequence[Column]
        The columns
    stream : Stream | None
        CUDA stream on which to perform the operation.
    mr : DeviceMemoryResource | None
        Device memory resource for allocations.

    Returns
    -------
    tuple[DeviceBuffer, size_type]
        A tuple of the resulting mask and count of unset bits
    """
    cdef Table c_table = Table(columns)
    cdef pair[device_buffer, size_type] c_result
    cdef Stream _stream = _get_stream(stream)
    cdef cudaStream_t _cs = _stream.view().value()
    mr = _get_memory_resource(mr)

    cdef table_view c_input = c_table.view()
    with nogil:
        c_result = cpp_null_mask.bitmask_or(c_input, _cs, mr.get_mr())

    return buffer_to_python(move(c_result.first), _stream, mr), c_result.second


cpdef size_type null_count(
    object bitmask,
    size_type start,
    size_type stop,
    object stream=None
):
    """Given a validity bitmask, counts the number of null elements.

    For details, see :cpp:func:`null_count`.

    Parameters
    ----------
    bitmask : Span-like object
        Object with ptr and size attributes (e.g., gpumemoryview, Buffer, DeviceBuffer).
    start : int
        Index of the first bit to count (inclusive).
    stop : int
        Index of the last bit to count (exclusive).
    stream : Stream | None
        CUDA stream on which to perform the operation.

    Returns
    -------
    int
        The number of null elements in the specified range.
    """
    if not py_is_span(bitmask):
        raise TypeError(
            f"bitmask must satisfy Span protocol (have .ptr and .size), "
            f"got {type(bitmask).__name__}"
        )
    cdef uintptr_t ptr = bitmask.ptr
    cdef Stream _stream = _get_stream(stream)
    cdef cudaStream_t _cs = _stream.view().value()
    with nogil:
        return cpp_null_mask.null_count(
            <bitmask_type*>ptr,
            start,
            stop,
            _cs
        )

cpdef size_type index_of_first_set_bit(
    object bitmask,
    size_type start,
    size_type stop,
    object stream=None
):
    """Given a validity bitmask, returns the index of the first valid element
    relative to ``start``.

    For details, see :cpp:func:`index_of_first_set_bit`.

    Parameters
    ----------
    bitmask : Span-like object
        Object with ptr and size attributes (e.g., gpumemoryview, Buffer, DeviceBuffer).
    start : int
        Index of the first bit to check (inclusive).
    stop : int
        Index of the last bit to check (exclusive).
    stream : Stream | None
        CUDA stream on which to perform the operation.

    Returns
    -------
    int
        The index of the first set bit relative to ``start``
    """
    if not py_is_span(bitmask):
        raise TypeError(
            f"bitmask must satisfy Span protocol (have .ptr and .size), "
            f"got {type(bitmask).__name__}"
        )
    cdef uintptr_t ptr = bitmask.ptr
    cdef Stream _stream = _get_stream(stream)
    cdef cudaStream_t _cs = _stream.view().value()
    with nogil:
        return cpp_null_mask.index_of_first_set_bit(
            <bitmask_type*>ptr,
            start,
            stop,
            _cs
        )
