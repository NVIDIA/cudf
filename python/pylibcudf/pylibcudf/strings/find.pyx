# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move
from pylibcudf.column cimport Column
from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.column.column_view cimport column_view
from pylibcudf.libcudf.strings cimport find as cpp_find
from pylibcudf.libcudf.types cimport size_type
from pylibcudf.scalar cimport Scalar
from pylibcudf.utils cimport _get_stream, _get_memory_resource
from rmm.pylibrmm.memory_resource cimport DeviceMemoryResource
from rmm.pylibrmm.stream cimport Stream

from cython.operator import dereference

from pylibcudf.libcudf.scalar.scalar cimport string_scalar
from cuda.bindings.cyruntime cimport cudaStream_t

__all__ = ["contains", "ends_with", "find", "rfind", "starts_with"]

cpdef Column find(
    Column input,
    ColumnOrScalar target,
    size_type start=0,
    size_type stop=-1,
    object stream=None,
    DeviceMemoryResource mr=None,
):
    """Returns a column of character position values where the target string is
    first found in each string of the provided column.

    ``target`` may be a
    :py:class:`~pylibcudf.column.Column` or a
    :py:class:`~pylibcudf.scalar.Scalar`.

    If ``target`` is a scalar, the scalar will be searched for in each string.
    If ``target`` is a column, the corresponding string in the column will be
    searched for in each string.

    For details, see :cpp:func:`find`.

    Parameters
    ----------
    input : Column
        The input strings
    target : Union[Column, Scalar]
        String to search for in each string
    start : size_type
        First character position to include in the search
    stop : size_type
        Last position (exclusive) to include in the search. Default of -1 will
        search to the end of the string.
    stream : Stream | None
        CUDA stream on which to perform the operation.

    Returns
    -------
    pylibcudf.Column
        New integer column with character position values
    """
    cdef unique_ptr[column] result
    cdef Stream _stream = _get_stream(stream)
    cdef cudaStream_t _cs = _stream.view().value()
    mr = _get_memory_resource(mr)
    cdef column_view c_input
    cdef column_view c_target
    if ColumnOrScalar is Column:
        c_input = input.view()
        c_target = target.view()
        with nogil:
            result = cpp_find.find(
                c_input,
                c_target,
                start,
                _cs,
                mr.get_mr()
            )
    elif ColumnOrScalar is Scalar:
        c_input = input.view()
        with nogil:
            result = cpp_find.find(
                c_input,
                dereference(<string_scalar*>(target.c_obj.get())),
                start,
                stop,
                _cs,
                mr.get_mr()
            )
    else:
        raise ValueError(f"Invalid target {target}")

    return Column.from_libcudf(move(result), _stream, mr)


cpdef Column rfind(
    Column input,
    Scalar target,
    size_type start=0,
    size_type stop=-1,
    object stream=None,
    DeviceMemoryResource mr=None,
):
    """
    Returns a column of character position values where the target string is
    first found searching from the end of each string.

    For details, see :cpp:func:`rfind`.

    Parameters
    ----------
    input : Column
        The input strings
    target : Scalar
        String to search for in each string
    start : size_type
        First character position to include in the search
    stop : size_type
        Last position (exclusive) to include in the search. Default of -1 will
        search to the end of the string.
    stream : Stream | None
        CUDA stream on which to perform the operation.

    Returns
    -------
    pylibcudf.Column
        New integer column with character position values
    """
    cdef unique_ptr[column] result
    cdef Stream _stream = _get_stream(stream)
    cdef cudaStream_t _cs = _stream.view().value()
    mr = _get_memory_resource(mr)
    cdef column_view c_input = input.view()
    with nogil:
        result = cpp_find.rfind(
            c_input,
            dereference(<string_scalar*>(target.c_obj.get())),
            start,
            stop,
            _cs,
            mr.get_mr()
        )
    return Column.from_libcudf(move(result), _stream, mr)


cpdef Column contains(
    Column input,
    ColumnOrScalar target,
    object stream=None,
    DeviceMemoryResource mr=None,
):
    """
    Returns a column of boolean values for each string where true indicates the
    corresponding target string was found within that string in the provided
    column.

    ``target`` may be a
    :py:class:`~pylibcudf.column.Column` or a
    :py:class:`~pylibcudf.scalar.Scalar`.

    If ``target`` is a scalar, the scalar will be searched for in each string.
    If ``target`` is a column, the corresponding string in the column will be
    searched for in each string.

    For details, see :cpp:func:`contains`.

    Parameters
    ----------
    input : Column
        The input strings
    target : Union[Column, Scalar]
        String to search for in each string
    stream : Stream | None
        CUDA stream on which to perform the operation.

    Returns
    -------
    pylibcudf.Column
        New boolean column with True for each string that contains the target
    """
    cdef unique_ptr[column] result
    cdef Stream _stream = _get_stream(stream)
    cdef cudaStream_t _cs = _stream.view().value()
    mr = _get_memory_resource(mr)
    cdef column_view c_input
    cdef column_view c_target
    if ColumnOrScalar is Column:
        c_input = input.view()
        c_target = target.view()
        with nogil:
            result = cpp_find.contains(
                c_input,
                c_target,
                _cs,
                mr.get_mr()
            )
    elif ColumnOrScalar is Scalar:
        c_input = input.view()
        with nogil:
            result = cpp_find.contains(
                c_input,
                dereference(<string_scalar*>(target.c_obj.get())),
                _cs,
                mr.get_mr()
            )
    else:
        raise ValueError(f"Invalid target {target}")

    return Column.from_libcudf(move(result), _stream, mr)


cpdef Column starts_with(
    Column input,
    ColumnOrScalar target,
    object stream=None,
    DeviceMemoryResource mr=None,
):
    """
    Returns a column of boolean values for each string where true indicates the
    target string was found at the beginning of the string in the provided
    column.

    ``target`` may be a
    :py:class:`~pylibcudf.column.Column` or a
    :py:class:`~pylibcudf.scalar.Scalar`.

    If ``target`` is a scalar, the scalar will be searched for in each string.
    If ``target`` is a column, the corresponding string in the column will be
    searched for in each string.

    For details, see :cpp:func:`starts_with`.

    Parameters
    ----------
    input : Column
        The input strings
    target : Union[Column, Scalar]
        String to search for at the beginning of each string
    stream : Stream | None
        CUDA stream on which to perform the operation.

    Returns
    -------
    pylibcudf.Column
        New boolean column with True for each string that starts with the target
    """
    cdef unique_ptr[column] result
    cdef Stream _stream = _get_stream(stream)
    cdef cudaStream_t _cs = _stream.view().value()
    mr = _get_memory_resource(mr)
    cdef column_view c_input
    cdef column_view c_target

    if ColumnOrScalar is Column:
        c_input = input.view()
        c_target = target.view()
        with nogil:
            result = cpp_find.starts_with(
                c_input,
                c_target,
                _cs,
                mr.get_mr()
            )
    elif ColumnOrScalar is Scalar:
        c_input = input.view()
        with nogil:
            result = cpp_find.starts_with(
                c_input,
                dereference(<string_scalar*>(target.c_obj.get())),
                _cs,
                mr.get_mr()
            )
    else:
        raise ValueError(f"Invalid target {target}")

    return Column.from_libcudf(move(result), _stream, mr)

cpdef Column ends_with(
    Column input,
    ColumnOrScalar target,
    object stream=None,
    DeviceMemoryResource mr=None,
):
    """
    Returns a column of boolean values for each string where true indicates the
    target string was found at the end of the string in the provided column.

    ``target`` may be a
    :py:class:`~pylibcudf.column.Column` or a
    :py:class:`~pylibcudf.scalar.Scalar`.

    If ``target`` is a scalar, the scalar will be searched for in each string.
    If ``target`` is a column, the corresponding string in the column will be
    searched for in each string.

    For details, see :cpp:func:`ends_with`.

    Parameters
    ----------
    input : Column
        The input strings
    target : Union[Column, Scalar]
        String to search for at the end of each string
    stream : Stream | None
        CUDA stream on which to perform the operation.

    Returns
    -------
    pylibcudf.Column
        New boolean column with True for each string that ends with the target
    """
    cdef unique_ptr[column] result
    cdef Stream _stream = _get_stream(stream)
    cdef cudaStream_t _cs = _stream.view().value()
    mr = _get_memory_resource(mr)
    cdef column_view c_input
    cdef column_view c_target
    if ColumnOrScalar is Column:
        c_input = input.view()
        c_target = target.view()
        with nogil:
            result = cpp_find.ends_with(
                c_input,
                c_target,
                _cs,
                mr.get_mr()
            )
    elif ColumnOrScalar is Scalar:
        c_input = input.view()
        with nogil:
            result = cpp_find.ends_with(
                c_input,
                dereference(<string_scalar*>(target.c_obj.get())),
                _cs,
                mr.get_mr()
            )
    else:
        raise ValueError(f"Invalid target {target}")

    return Column.from_libcudf(move(result), _stream, mr)
