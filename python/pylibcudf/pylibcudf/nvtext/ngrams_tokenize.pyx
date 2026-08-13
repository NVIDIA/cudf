# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from cython.operator cimport dereference
from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move
from pylibcudf.column cimport Column
from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.column.column_view cimport column_view
from pylibcudf.libcudf.nvtext.ngrams_tokenize cimport (
    ngrams_tokenize as cpp_ngrams_tokenize,
)
from pylibcudf.libcudf.scalar.scalar cimport string_scalar
from pylibcudf.libcudf.types cimport size_type
from pylibcudf.scalar cimport Scalar
from pylibcudf.utils cimport _get_stream, _get_memory_resource
from rmm.pylibrmm.memory_resource cimport DeviceMemoryResource
from rmm.pylibrmm.stream cimport Stream
from cuda.bindings.cyruntime cimport cudaStream_t

__all__ = ["ngrams_tokenize"]

cpdef Column ngrams_tokenize(
    Column input,
    size_type ngrams,
    Scalar delimiter,
    Scalar separator,
    object stream=None,
    DeviceMemoryResource mr=None,
):
    """
    Returns a single column of strings by tokenizing the input strings column
    and then producing ngrams of each string.

    For details, see :cpp:func:`ngrams_tokenize`

    Parameters
    ----------
    input : Column
        Input strings
    ngrams : size_type
        The ngram number to generate
    delimiter : Scalar
        UTF-8 characters used to separate each string into tokens.
        An empty string will separate tokens using whitespace.
    separator : Scalar
        The string to use for separating ngram tokens
    stream : Stream | None
        CUDA stream on which to perform the operation.

    Returns
    -------
    Column
        New strings columns of tokens
    """
    cdef unique_ptr[column] c_result
    cdef Stream _stream = _get_stream(stream)
    cdef cudaStream_t _cs = _stream.view().value()
    mr = _get_memory_resource(mr)

    cdef column_view c_input = input.view()
    with nogil:
        c_result = cpp_ngrams_tokenize(
            c_input,
            ngrams,
            dereference(<const string_scalar*>delimiter.get()),
            dereference(<const string_scalar*>separator.get()),
            _cs,
            mr.get_mr()
        )
    return Column.from_libcudf(move(c_result), _stream, mr)
