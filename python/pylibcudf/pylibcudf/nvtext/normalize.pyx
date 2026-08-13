# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from cython.operator cimport dereference
from libcpp cimport bool
from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move
from pylibcudf.column cimport Column
from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.column.column_view cimport column_view
from pylibcudf.libcudf.nvtext cimport normalize as cpp_normalize
from pylibcudf.utils cimport _get_stream, _get_memory_resource
from rmm.pylibrmm.memory_resource cimport DeviceMemoryResource
from rmm.pylibrmm.stream cimport Stream
from cuda.bindings.cyruntime cimport cudaStream_t

__all__ = [
    "CharacterNormalizer"
    "normalize_characters",
    "normalize_spaces",
]

cdef class CharacterNormalizer:
    """The normalizer object to be used with ``normalize_characters``.

    For details, see :cpp:class:`cudf::nvtext::character_normalizer`.
    """
    def __cinit__(
        self,
        bool do_lower_case,
        Column tokens,
        object stream=None,
        DeviceMemoryResource mr=None
    ):
        cdef column_view c_tokens = tokens.view()
        cdef Stream _stream = _get_stream(stream)
        cdef cudaStream_t _cs = _stream.view().value()
        mr = _get_memory_resource(mr)
        with nogil:
            self.c_obj = move(
                cpp_normalize.create_character_normalizer(
                    do_lower_case,
                    c_tokens,
                    _cs,
                    mr.get_mr()
                )
            )

    __hash__ = None

cpdef Column normalize_spaces(
    Column input, object stream=None, DeviceMemoryResource mr=None
):
    """
    Returns a new strings column by normalizing the whitespace in
    each string in the input column.

    For details, see :cpp:func:`normalize_spaces`

    Parameters
    ----------
    input : Column
        Input strings
    stream : Stream | None
        CUDA stream on which to perform the operation.

    Returns
    -------
    Column
        New strings columns of normalized strings.
    """
    cdef unique_ptr[column] c_result
    cdef Stream _stream = _get_stream(stream)
    cdef cudaStream_t _cs = _stream.view().value()
    mr = _get_memory_resource(mr)

    cdef column_view c_input = input.view()
    with nogil:
        c_result = cpp_normalize.normalize_spaces(
            c_input, _cs, mr.get_mr()
        )

    return Column.from_libcudf(move(c_result), _stream, mr)


cpdef Column normalize_characters(
    Column input,
    CharacterNormalizer normalizer,
    object stream=None,
    DeviceMemoryResource mr=None,
):
    """
    Normalizes strings characters for tokenizing.

    For details, see :cpp:func:`normalize_characters`

    Parameters
    ----------
    input : Column
        Input strings
    normalizer : CharacterNormalizer
        Normalizer object used for modifying the input column text
    stream : Stream | None
        CUDA stream on which to perform the operation.

    Returns
    -------
    Column
        Normalized strings column
    """
    cdef unique_ptr[column] c_result
    cdef Stream _stream = _get_stream(stream)
    cdef cudaStream_t _cs = _stream.view().value()
    mr = _get_memory_resource(mr)

    cdef column_view c_input = input.view()
    with nogil:
        c_result = cpp_normalize.normalize_characters(
            c_input,
            dereference(normalizer.c_obj.get()),
            _cs,
            mr.get_mr()
        )

    return Column.from_libcudf(move(c_result), _stream, mr)
