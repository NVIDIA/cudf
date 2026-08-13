# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from libc.stdint cimport uint32_t, uint64_t
from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move
from pylibcudf.column cimport Column
from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.column.column_view cimport column_view
from pylibcudf.libcudf.nvtext.minhash cimport (
    minhash as cpp_minhash,
    minhash64 as cpp_minhash64,
    minhash_ngrams as cpp_minhash_ngrams,
    minhash64_ngrams as cpp_minhash64_ngrams,
)
from pylibcudf.libcudf.types cimport size_type
from pylibcudf.utils cimport _get_stream, _get_memory_resource
from rmm.pylibrmm.memory_resource cimport DeviceMemoryResource
from rmm.pylibrmm.stream cimport Stream
from cuda.bindings.cyruntime cimport cudaStream_t

__all__ = [
    "minhash",
    "minhash64",
    "minhash_ngrams",
    "minhash64_ngrams",
]

cpdef Column minhash(
    Column input,
    uint32_t seed,
    Column a,
    Column b,
    size_type width,
    object stream=None,
    DeviceMemoryResource mr=None
):
    """
    Returns the minhash values for each string.
    This function uses MurmurHash3_x86_32 for the hash algorithm.

    For details, see :cpp:func:`minhash`.

    Parameters
    ----------
    input : Column
        Strings column to compute minhash
    seed : uint32_t
        Seed used for the hash function
    a : Column
        1st parameter value used for the minhash algorithm.
    b : Column
        2nd parameter value used for the minhash algorithm.
    width : size_type
        Character width used for apply substrings;

    Returns
    -------
    Column
        List column of minhash values for each string per seed
    """
    cdef unique_ptr[column] c_result
    cdef Stream _stream = _get_stream(stream)
    cdef cudaStream_t _cs = _stream.view().value()
    mr = _get_memory_resource(mr)

    cdef column_view c_input = input.view()
    cdef column_view c_a = a.view()
    cdef column_view c_b = b.view()
    with nogil:
        c_result = cpp_minhash(
            c_input,
            seed,
            c_a,
            c_b,
            width,
            _cs,
            mr.get_mr()
        )

    return Column.from_libcudf(move(c_result), _stream, mr)

cpdef Column minhash64(
    Column input,
    uint64_t seed,
    Column a,
    Column b,
    size_type width,
    object stream=None,
    DeviceMemoryResource mr=None
):
    """
    Returns the minhash values for each string.
    This function uses MurmurHash3_x64_128 for the hash algorithm.

    For details, see :cpp:func:`minhash64`.

    Parameters
    ----------
    input : Column
        Strings column to compute minhash
    seed : uint64_t
        Seed used for the hash function
    a : Column
        1st parameter value used for the minhash algorithm.
    b : Column
        2nd parameter value used for the minhash algorithm.
    width : size_type
        Character width used for apply substrings;
    stream : Stream | None
        CUDA stream on which to perform the operation.

    Returns
    -------
    Column
        List column of minhash values for each string per seed
    """
    cdef unique_ptr[column] c_result
    cdef Stream _stream = _get_stream(stream)
    cdef cudaStream_t _cs = _stream.view().value()
    mr = _get_memory_resource(mr)

    cdef column_view c_input = input.view()
    cdef column_view c_a = a.view()
    cdef column_view c_b = b.view()
    with nogil:
        c_result = cpp_minhash64(
            c_input,
            seed,
            c_a,
            c_b,
            width,
            _cs,
            mr.get_mr()
        )

    return Column.from_libcudf(move(c_result), _stream, mr)

cpdef Column minhash_ngrams(
    Column input,
    size_type ngrams,
    uint32_t seed,
    Column a,
    Column b,
    object stream=None,
    DeviceMemoryResource mr=None
):
    """
    Returns the minhash values for each input row of strings.
    This function uses MurmurHash3_x86_32 for the hash algorithm.

    For details, see :cpp:func:`minhash_ngrams`.

    Parameters
    ----------
    input : Column
        List column of strings to compute minhash
    ngrams : size_type
        Number of consecutive strings to hash in each row
    seed : uint32_t
        Seed used for the hash function
    a : Column
        1st parameter value used for the minhash algorithm.
    b : Column
        2nd parameter value used for the minhash algorithm.
    stream : Stream | None
        CUDA stream on which to perform the operation.

    Returns
    -------
    Column
        List column of minhash values for each row per
        value in columns a and b.
    """
    cdef unique_ptr[column] c_result
    cdef Stream _stream = _get_stream(stream)
    cdef cudaStream_t _cs = _stream.view().value()
    mr = _get_memory_resource(mr)

    cdef column_view c_input = input.view()
    cdef column_view c_a = a.view()
    cdef column_view c_b = b.view()
    with nogil:
        c_result = cpp_minhash_ngrams(
            c_input,
            ngrams,
            seed,
            c_a,
            c_b,
            _cs,
            mr.get_mr()
        )

    return Column.from_libcudf(move(c_result), _stream, mr)

cpdef Column minhash64_ngrams(
    Column input,
    size_type ngrams,
    uint64_t seed,
    Column a,
    Column b,
    object stream=None,
    DeviceMemoryResource mr=None
):
    """
    Returns the minhash values for each input row of strings.
    This function uses MurmurHash3_x64_128 for the hash algorithm.

    For details, see :cpp:func:`minhash64_ngrams`.

    Parameters
    ----------
    input : Column
        Strings column to compute minhash
    ngrams : size_type
        Number of consecutive strings to hash in each row
    seed : uint64_t
        Seed used for the hash function
    a : Column
        1st parameter value used for the minhash algorithm.
    b : Column
        2nd parameter value used for the minhash algorithm.
    stream : Stream | None
        CUDA stream on which to perform the operation.

    Returns
    -------
    Column
        List column of minhash values for each row per
        value in columns a and b.
    """
    cdef unique_ptr[column] c_result
    cdef Stream _stream = _get_stream(stream)
    cdef cudaStream_t _cs = _stream.view().value()
    mr = _get_memory_resource(mr)

    cdef column_view c_input = input.view()
    cdef column_view c_a = a.view()
    cdef column_view c_b = b.view()
    with nogil:
        c_result = cpp_minhash64_ngrams(
            c_input,
            ngrams,
            seed,
            c_a,
            c_b,
            _cs,
            mr.get_mr()
        )

    return Column.from_libcudf(move(c_result), _stream, mr)
