# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from cython.operator cimport dereference
from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move
from pylibcudf.column cimport Column
from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.column.column_view cimport column_view
from pylibcudf.libcudf.nvtext.tokenize cimport (
    character_tokenize as cpp_character_tokenize,
    count_tokens as cpp_count_tokens,
    detokenize as cpp_detokenize,
    load_vocabulary as cpp_load_vocabulary,
    tokenize as cpp_tokenize,
    tokenize_with_vocabulary as cpp_tokenize_with_vocabulary,
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

__all__ = [
    "TokenizeVocabulary",
    "character_tokenize",
    "count_tokens_column",
    "count_tokens_scalar",
    "detokenize",
    "tokenize_column",
    "tokenize_scalar",
    "tokenize_with_vocabulary",
]

cdef class TokenizeVocabulary:
    """The Vocabulary object to be used with ``tokenize_with_vocabulary``.

    For details, see :cpp:class:`cudf::nvtext::tokenize_vocabulary`.
    """
    def __cinit__(self, Column vocab, object stream=None, DeviceMemoryResource mr=None):
        cdef column_view c_vocab = vocab.view()
        cdef Stream _stream = _get_stream(stream)
        cdef cudaStream_t _cs = _stream.view().value()
        mr = _get_memory_resource(mr)
        with nogil:
            self.c_obj = move(cpp_load_vocabulary(c_vocab, _cs, mr.get_mr()))

    __hash__ = None

cpdef Column tokenize_scalar(
    Column input,
    Scalar delimiter=None,
    object stream=None,
    DeviceMemoryResource mr=None,
):
    """
    Returns a single column of strings by tokenizing the input
    strings column using the provided characters as delimiters.

    For details, see cpp:func:`cudf::nvtext::tokenize`

    Parameters
    ----------
    input : Column
        Strings column to tokenize
    delimiter : Scalar
        String scalar used to separate individual strings into tokens
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

    if delimiter is None:
        delimiter = Scalar.from_libcudf(
            cpp_make_string_scalar("".encode(), _stream.view().value(), mr.get_mr())
        )

    cdef column_view c_input = input.view()
    with nogil:
        c_result = cpp_tokenize(
            c_input,
            dereference(<const string_scalar*>delimiter.c_obj.get()),
            _cs,
            mr.get_mr()
        )

    return Column.from_libcudf(move(c_result), _stream, mr)

cpdef Column tokenize_column(
    Column input, Column delimiters, object stream=None, DeviceMemoryResource mr=None
):
    """
    Returns a single column of strings by tokenizing the input
    strings column using multiple strings as delimiters.

    For details, see cpp:func:`cudf::nvtext::tokenize`

    Parameters
    ----------
    input : Column
        Strings column to tokenize
    delimiters : Column
        Strings column used to separate individual strings into tokens
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
    cdef column_view c_delimiters = delimiters.view()
    with nogil:
        c_result = cpp_tokenize(
            c_input,
            c_delimiters,
            _cs,
            mr.get_mr()
        )

    return Column.from_libcudf(move(c_result), _stream, mr)

cpdef Column count_tokens_scalar(
    Column input,
    Scalar delimiter=None,
    object stream=None,
    DeviceMemoryResource mr=None,
):
    """
    Returns the number of tokens in each string of a strings column
    using the provided characters as delimiters.

    For details, see cpp:func:`cudf::nvtext::count_tokens`

    Parameters
    ----------
    input : Column
        Strings column to count tokens
    delimiters : Scalar
        String scalar used to separate each string into tokens
    stream : Stream | None
        CUDA stream on which to perform the operation.

    Returns
    -------
    Column
        New column of token counts
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
    with nogil:
        c_result = cpp_count_tokens(
            c_input,
            dereference(<const string_scalar*>delimiter.c_obj.get()),
            _cs,
            mr.get_mr()
        )

    return Column.from_libcudf(move(c_result), _stream, mr)

cpdef Column count_tokens_column(
    Column input, Column delimiters, object stream=None, DeviceMemoryResource mr=None
):
    """
    Returns the number of tokens in each string of a strings column
    using multiple strings as delimiters.

    For details, see cpp:func:`cudf::nvtext::count_tokens`

    Parameters
    ----------
    input : Column
        Strings column to count tokens
    delimiters : Column
        Strings column used to separate each string into tokens
    stream : Stream | None
        CUDA stream on which to perform the operation.

    Returns
    -------
    Column
        New column of token counts
    """
    cdef unique_ptr[column] c_result
    cdef Stream _stream = _get_stream(stream)
    cdef cudaStream_t _cs = _stream.view().value()
    mr = _get_memory_resource(mr)

    cdef column_view c_input = input.view()
    cdef column_view c_delimiters = delimiters.view()
    with nogil:
        c_result = cpp_count_tokens(
            c_input,
            c_delimiters,
            _cs,
            mr.get_mr()
        )

    return Column.from_libcudf(move(c_result), _stream, mr)

cpdef Column character_tokenize(
    Column input, object stream=None, DeviceMemoryResource mr=None
):
    """
    Returns a single column of strings by converting
    each character to a string.

    For details, see cpp:func:`cudf::nvtext::character_tokens`

    Parameters
    ----------
    input : Column
        Strings column to tokenize
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
        c_result = cpp_character_tokenize(c_input, _cs, mr.get_mr())

    return Column.from_libcudf(move(c_result), _stream, mr)

cpdef Column detokenize(
    Column input,
    Column row_indices,
    Scalar separator=None,
    object stream=None,
    DeviceMemoryResource mr=None,
):
    """
    Creates a strings column from a strings column of tokens
    and an associated column of row ids.

    For details, see cpp:func:`cudf::nvtext::detokenize`

    Parameters
    ----------
    input : Column
        Strings column to detokenize
    row_indices : Column
        The relative output row index assigned for each token in the input column
    separator : Scalar
        String to append after concatenating each token to the proper output row
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

    if separator is None:
        separator = Scalar.from_libcudf(
            cpp_make_string_scalar(" ".encode(), _stream.view().value(), mr.get_mr())
        )

    cdef column_view c_input = input.view()
    cdef column_view c_row_indices = row_indices.view()
    with nogil:
        c_result = cpp_detokenize(
            c_input,
            c_row_indices,
            dereference(<const string_scalar*>separator.c_obj.get()),
            _cs,
            mr.get_mr()
        )

    return Column.from_libcudf(move(c_result), _stream, mr)

cpdef Column tokenize_with_vocabulary(
    Column input,
    TokenizeVocabulary vocabulary,
    Scalar delimiter,
    size_type default_id=-1,
    object stream=None,
    DeviceMemoryResource mr=None,
):
    """
    Returns the token ids for the input string by looking
    up each delimited token in the given vocabulary.

    For details, see cpp:func:`cudf::nvtext::tokenize_with_vocabulary`

    Parameters
    ----------
    input : Column
        Strings column to tokenize
    vocabulary : TokenizeVocabulary
        Used to lookup tokens within ``input``
    delimiter : Scalar
        Used to identify tokens within ``input``
    default_id : size_type
        The token id to be used for tokens not found in the vocabulary; Default is -1
    stream : Stream | None
        CUDA stream on which to perform the operation.

    Returns
    -------
    Column
        Lists column of token ids
    """
    cdef unique_ptr[column] c_result
    cdef Stream _stream = _get_stream(stream)
    cdef cudaStream_t _cs = _stream.view().value()
    mr = _get_memory_resource(mr)

    cdef column_view c_input = input.view()
    with nogil:
        c_result = cpp_tokenize_with_vocabulary(
            c_input,
            dereference(vocabulary.c_obj.get()),
            dereference(<const string_scalar*>delimiter.c_obj.get()),
            default_id,
            _cs,
            mr.get_mr()
        )

    return Column.from_libcudf(move(c_result), _stream, mr)
