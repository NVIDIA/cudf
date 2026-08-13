# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move
from pylibcudf.column cimport Column
from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.column.column_view cimport column_view
from pylibcudf.libcudf.scalar.scalar cimport string_scalar
from pylibcudf.libcudf.scalar.scalar_factories cimport (
    make_string_scalar as cpp_make_string_scalar,
)
from pylibcudf.libcudf.strings.replace cimport (
    replace as cpp_replace,
    replace_multiple as cpp_replace_multiple,
    replace_slice as cpp_replace_slice,
)
from pylibcudf.libcudf.types cimport size_type
from pylibcudf.scalar cimport Scalar
from pylibcudf.utils cimport _get_stream, _get_memory_resource
from rmm.pylibrmm.memory_resource cimport DeviceMemoryResource
from rmm.pylibrmm.stream cimport Stream
from cuda.bindings.cyruntime cimport cudaStream_t

__all__ = ["replace", "replace_multiple", "replace_slice"]

cpdef Column replace(
    Column input,
    Scalar target,
    Scalar repl,
    size_type maxrepl=-1,
    object stream=None,
    DeviceMemoryResource mr=None,
):
    """Replaces target string within each string with the specified replacement string.

    Null string entries will return null output string entries.

    For details, see :cpp:func:`replace`.

    Parameters
    ----------
    input : Column
        The input strings
    target : Scalar
        String to search for in each string.
    repl : Scalar
        String to replace target with.
    maxrepl : size_type, default -1
        Maximum times to replace if target appears multiple times in the input string.
        Default of -1 specifies to replace all occurrences of target in each string.

    Returns
    -------
    pylibcudf.Column
        New string column with target replaced.
    """
    cdef:
        unique_ptr[column] c_result
        const string_scalar* target_str
        const string_scalar* repl_str

    target_str = <string_scalar *>(target.c_obj.get())
    repl_str = <string_scalar *>(repl.c_obj.get())
    cdef Stream _stream = _get_stream(stream)
    cdef cudaStream_t _cs = _stream.view().value()
    mr = _get_memory_resource(mr)
    cdef column_view c_input = input.view()
    with nogil:
        c_result = cpp_replace(
            c_input,
            target_str[0],
            repl_str[0],
            maxrepl,
            _cs,
            mr.get_mr()
        )

    return Column.from_libcudf(move(c_result), _stream, mr)


cpdef Column replace_multiple(
    Column input,
    Column target,
    Column repl,
    size_type maxrepl=-1,
    object stream=None,
    DeviceMemoryResource mr=None,
):
    """Replaces target string within each string with the specified replacement string.

    Null string entries will return null output string entries.

    For details, see :cpp:func:`replace_multiple`.

    Parameters
    ----------
    input : Column
        The input strings
    target : Column
        Column containing strings to search for in the input column.
    repl : Column
        Column containing strings to replace target with.
        Each target, when found, will be replaced by the value at the
        corresponding index in the repl Column.

        Must be of the same length as target.

    Returns
    -------
    pylibcudf.Column
        New string column with target replaced.
    """
    cdef unique_ptr[column] c_result
    cdef Stream _stream = _get_stream(stream)
    cdef cudaStream_t _cs = _stream.view().value()
    mr = _get_memory_resource(mr)
    cdef column_view c_input = input.view()
    cdef column_view c_target = target.view()
    cdef column_view c_repl = repl.view()
    with nogil:
        c_result = cpp_replace_multiple(
            c_input,
            c_target,
            c_repl,
            _cs,
            mr.get_mr()
        )

    return Column.from_libcudf(move(c_result), _stream, mr)


cpdef Column replace_slice(
    Column input,
    # TODO: default scalar values
    # https://github.com/rapidsai/cudf/issues/15505
    Scalar repl=None,
    size_type start=0,
    size_type stop=-1,
    object stream=None,
    DeviceMemoryResource mr=None,
):
    """Replaces each string in the column with the provided repl string
    within the [start,stop) character position range.

    Null string entries will return null output string entries.
    This function can be used to insert a string into specific position
    by specifying the same position value for start and stop.
    The repl string can be appended to each string by specifying -1
    for both start and stop.

    For details, see :cpp:func:`replace_slice`.

    Parameters
    ----------
    input : Column
        The input strings
    repl : Scalar, default ""
        String scalar to replace target with.
    start : size_type, default 0
        Start position where repl will be added.
    stop : size_type, default -1
        End position (exclusive) to use for replacement.

    Returns
    -------
    pylibcudf.Column
        New string column
    """
    cdef unique_ptr[column] c_result
    cdef Stream _stream = _get_stream(stream)
    cdef cudaStream_t _cs = _stream.view().value()
    mr = _get_memory_resource(mr)
    cdef column_view c_input

    if repl is None:
        repl = Scalar.from_libcudf(
            cpp_make_string_scalar("".encode(), _stream.view().value(), mr.get_mr())
        )

    cdef const string_scalar* scalar_str = <string_scalar*>(repl.c_obj.get())

    c_input = input.view()
    with nogil:
        c_result = cpp_replace_slice(
            c_input,
            scalar_str[0],
            start,
            stop,
            _cs,
            mr.get_mr()
        )

    return Column.from_libcudf(move(c_result), _stream, mr)
