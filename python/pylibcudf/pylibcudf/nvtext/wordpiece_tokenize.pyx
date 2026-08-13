# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from cython.operator cimport dereference
from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move
from pylibcudf.column cimport Column
from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.column.column_view cimport column_view
from pylibcudf.libcudf.nvtext.wordpiece_tokenize cimport (
    load_wordpiece_vocabulary as cpp_load_wordpiece_vocabulary,
    wordpiece_tokenize as cpp_wordpiece_tokenize,
)
from pylibcudf.libcudf.types cimport size_type
from pylibcudf.utils cimport _get_stream, _get_memory_resource
from rmm.pylibrmm.memory_resource cimport DeviceMemoryResource
from rmm.pylibrmm.stream cimport Stream
from cuda.bindings.cyruntime cimport cudaStream_t

__all__ = [
    "WordPieceVocabulary",
    "wordpiece_tokenize",
]

cdef class WordPieceVocabulary:
    """The Vocabulary object to be used with ``wordpiece_tokenize``.

    For details, see :cpp:class:`cudf::nvtext::wordpiece_tokenize`.
    """
    def __cinit__(
        self,
        Column vocab,
        object stream=None,
        DeviceMemoryResource mr=None
    ):
        cdef column_view c_vocab = vocab.view()
        cdef Stream _stream = _get_stream(stream)
        cdef cudaStream_t _cs = _stream.view().value()
        mr = _get_memory_resource(mr)
        with nogil:
            self.c_obj = move(cpp_load_wordpiece_vocabulary(
                c_vocab, _cs, mr.get_mr()
            ))

    __hash__ = None

cpdef Column wordpiece_tokenize(
    Column input,
    WordPieceVocabulary vocabulary,
    size_type max_words_per_row,
    object stream=None,
    DeviceMemoryResource mr=None,
):
    """
    Returns the token ids for the input string by looking
    up each delimited token in the given vocabulary.
    The input is expected to be normalized.

    For details, see cpp:func:`cudf::nvtext::wordpiece_tokenize`

    Parameters
    ----------
    input : Column
        Normalized strings column to tokenize
    vocabulary : WordPieceVocabulary
        Used to lookup tokens within ``input``
    max_words_per_row : size_type
        Maximum number of words to tokenize per input row
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
        c_result = cpp_wordpiece_tokenize(
            c_input,
            dereference(vocabulary.c_obj.get()),
            max_words_per_row,
            _cs,
            mr.get_mr()
        )

    return Column.from_libcudf(move(c_result), _stream, mr)
