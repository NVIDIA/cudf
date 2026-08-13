# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from cython.operator cimport dereference
from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move
from pylibcudf.column cimport Column
from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.column.column_view cimport column_view
from pylibcudf.libcudf.nvtext.replace cimport (
    filter_tokens as cpp_filter_tokens,
    replace_tokens as cpp_replace_tokens,
)
from pylibcudf.libcudf.scalar.scalar cimport string_scalar
from pylibcudf.libcudf.scalar.scalar_factories cimport (
    make_string_scalar as cpp_make_string_scalar,
)
from pylibcudf.libcudf.types cimport size_type
from pylibcudf.scalar cimport Scalar
from pylibcudf.utils cimport _get_stream, _get_memory_resource
from rmm.pylibrmm.memory_resource cimport DeviceMemoryResource
from rmm.pylibrmm.stream cimport Stream
from cuda.bindings.cyruntime cimport cudaStream_t

__all__ = ["filter_tokens", "replace_tokens"]

cpdef Column replace_tokens(
    Column input,
    Column targets,
    Column replacements,
    Scalar delimiter=None,
    object stream=None,
    DeviceMemoryResource mr=None,
):
    """
    Replaces specified tokens with corresponding replacement strings.

    For details, see :cpp:func:`replace_tokens`

    Parameters
    ----------
    input : Column
        Strings column to replace
    targets : Column
        Strings to compare against tokens found in ``input``
    replacements : Column
        Replacement strings for each string in ``targets``
    delimiter : Scalar, optional
        Characters used to separate each string into tokens.
        The default of empty string will identify tokens using whitespace.
    stream : Stream | None
        CUDA stream on which to perform the operation.

    Returns
    -------
    Column
        New strings column with replaced strings
    """
    cdef unique_ptr[column] c_result
    cdef Stream _stream = _get_stream(stream)
    cdef cudaStream_t _cs = _stream.view().value()
    mr = _get_memory_resource(mr)
    if delimiter is None:
        delimiter = Scalar.from_libcudf(
            cpp_make_string_scalar("".encode(), _stream.view().value(), mr.get_mr())
        )
    cdef column_view c_input = input.view()
    cdef column_view c_targets = targets.view()
    cdef column_view c_replacements = replacements.view()
    with nogil:
        c_result = cpp_replace_tokens(
            c_input,
            c_targets,
            c_replacements,
            dereference(<const string_scalar*>delimiter.get()),
            _cs,
            mr.get_mr()
        )
    return Column.from_libcudf(move(c_result), _stream, mr)


cpdef Column filter_tokens(
    Column input,
    size_type min_token_length,
    Scalar replacement=None,
    Scalar delimiter=None,
    object stream=None,
    DeviceMemoryResource mr=None,
):
    """
    Removes tokens whose lengths are less than a specified number of characters.

    For details, see :cpp:func:`filter_tokens`

    Parameters
    ----------
    input : Column
        Strings column to replace
    min_token_length : size_type
        The minimum number of characters to retain a
        token in the output string
    replacement : Scalar, optional
        Optional replacement string to be used in place of removed tokens
    delimiter : Scalar, optional
        Characters used to separate each string into tokens.
        The default of empty string will identify tokens using whitespace.
    stream : Stream | None
        CUDA stream on which to perform the operation.

    Returns
    -------
    Column
        New strings column of filtered strings
    """
    cdef unique_ptr[column] c_result
    cdef Stream _stream = _get_stream(stream)
    cdef cudaStream_t _cs = _stream.view().value()
    mr = _get_memory_resource(mr)
    if delimiter is None:
        delimiter = Scalar.from_libcudf(
            cpp_make_string_scalar("".encode(), _stream.view().value(), mr.get_mr())
        )
    if replacement is None:
        replacement = Scalar.from_libcudf(
            cpp_make_string_scalar("".encode(), _stream.view().value(), mr.get_mr())
        )

    cdef column_view c_input = input.view()
    with nogil:
        c_result = cpp_filter_tokens(
            c_input,
            min_token_length,
            dereference(<const string_scalar*>replacement.get()),
            dereference(<const string_scalar*>delimiter.get()),
            _cs,
            mr.get_mr()
        )

    return Column.from_libcudf(move(c_result), _stream, mr)
