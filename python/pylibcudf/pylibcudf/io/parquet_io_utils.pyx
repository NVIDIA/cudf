# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""IO utilities for Parquet."""

from libc.stddef cimport size_t
from libc.stdint cimport uint8_t, uintptr_t
from libcpp.memory cimport make_unique, shared_ptr, unique_ptr
from libcpp.pair cimport pair
from libcpp.utility cimport move
from libcpp.vector cimport vector
from cython.operator cimport dereference

from rmm.librmm.device_buffer cimport device_buffer
from rmm.pylibrmm.device_buffer cimport DeviceBuffer
from rmm.pylibrmm.memory_resource cimport DeviceMemoryResource
from rmm.pylibrmm.stream cimport Stream

from pylibcudf.gpumemoryview cimport gpumemoryview
from pylibcudf.io.text cimport ByteRangeInfo
from pylibcudf.io.types cimport SourceInfo
from pylibcudf.libcudf.io.datasource cimport datasource, make_datasources
from pylibcudf.libcudf.io.parquet_io_utils cimport (
    bloom_filter_fetch,
    const_uint8_t,
    cpp_fetch_bloom_filters_to_device,
    cpp_fetch_byte_ranges_to_device,
    fetch_page_index_to_host as cpp_fetch_page_index_to_host,
    io_submission_policy as cpp_io_submission_policy,
)

from pylibcudf.libcudf.io.text cimport byte_range_info
from pylibcudf.libcudf.utilities.span cimport device_span
from pylibcudf.utils cimport _get_memory_resource, _get_stream
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pylibcudf.typing import CudaStreamLike

import pylibcudf.libcudf.io.parquet_io_utils

IOSubmissionPolicy = pylibcudf.libcudf.io.parquet_io_utils.io_submission_policy

__all__ = [
    "IOSubmissionPolicy",
    "fetch_bloom_filters_to_device",
    "fetch_byte_ranges_to_device",
    "fetch_page_index_to_host",
]


cdef list _group_byte_ranges_by_source(
    list byte_ranges,
    list source_indices,
    Py_ssize_t num_sources,
    vector[vector[byte_range_info]]& ranges_per_source,
):
    """Group byte ranges by source, returning the source index of each range."""
    if source_indices is None:
        if num_sources != 1:
            raise ValueError(
                f"source_indices is required unless there is exactly one "
                f"source, got {num_sources}"
            )
        source_indices = [0] * len(byte_ranges)
    elif len(source_indices) != len(byte_ranges):
        raise ValueError(
            f"Expected one source index per byte range, got "
            f"{len(source_indices)} indices for {len(byte_ranges)} byte ranges"
        )

    ranges_per_source.resize(num_sources)
    cdef ByteRangeInfo bri
    cdef Py_ssize_t source
    for bri, source in zip(byte_ranges, source_indices):
        if not 0 <= source < num_sources:
            raise ValueError(
                f"Source index {source} does not refer to one of "
                f"{num_sources} sources"
            )
        ranges_per_source[source].push_back(bri.c_obj)
    return source_indices


cdef class _BloomFilterBuffers:
    cdef shared_ptr[void] c_obj


cdef class _BloomFilterBitset:
    """Device memory of one bloom filter bitset, owned by a fetch result."""

    cdef _BloomFilterBuffers owner
    cdef uintptr_t ptr
    cdef size_t size

    @property
    def __cuda_array_interface__(self):
        return {
            "data": (self.ptr, False),
            "shape": (self.size,),
            "typestr": "|u1",
            "version": 3,
        }


cpdef list fetch_bloom_filters_to_device(
    SourceInfo source_info,
    list byte_ranges,
    cpp_io_submission_policy policy,
    object stream: CudaStreamLike | None = None,
    DeviceMemoryResource mr=None,
    list source_indices=None,
):
    """Fetch the bloom filter bitsets of parquet column chunks into device memory.

    Unlike :func:`fetch_byte_ranges_to_device`, the bloom filter headers are
    removed and every bitset is aligned as required by
    :meth:`~pylibcudf.io.experimental.HybridScanReader.filter_row_groups_with_bloom_filters`
    and
    :meth:`~pylibcudf.io.experimental.HybridScanMultiFile.filter_row_groups_with_bloom_filters`.

    Parameters
    ----------
    source_info : SourceInfo
        Source describing one or more Parquet files.
    byte_ranges : list[ByteRangeInfo]
        Byte ranges of complete bloom filters, as returned by
        :meth:`~pylibcudf.io.experimental.HybridScanReader.bloom_filters_byte_ranges`
        or
        :meth:`~pylibcudf.io.experimental.HybridScanMultiFile.bloom_filters_byte_ranges`.
    policy : IOSubmissionPolicy
        Whether to serialize I/O submissions from callers.
    stream : Stream, optional
        CUDA stream.
    mr : DeviceMemoryResource, optional
        Device memory resource.
    source_indices : list[int], optional
        Index of the source each byte range is fetched from. Required unless
        ``source_info`` describes exactly one source.

    Returns
    -------
    list[gpumemoryview]
        One view of the bitset per byte range, in the order of
        ``byte_ranges``. Each view keeps the allocation that owns its memory
        alive.

    Raises
    ------
    ValueError
        If ``source_indices`` is omitted and ``source_info`` does not describe
        exactly one source, if ``source_indices`` does not hold one index per
        byte range, or if an index does not refer to a source.
    """
    cdef Stream _stream = _get_stream(stream)
    cdef DeviceMemoryResource _mr = _get_memory_resource(mr)
    cdef vector[unique_ptr[datasource]] sources = make_datasources(source_info.c_obj)
    cdef Py_ssize_t num_sources = sources.size()
    cdef vector[vector[byte_range_info]] ranges_per_source
    source_indices = _group_byte_ranges_by_source(
        byte_ranges, source_indices, num_sources, ranges_per_source
    )

    cdef bloom_filter_fetch fetched
    with nogil:
        fetched = cpp_fetch_bloom_filters_to_device(
            sources,
            ranges_per_source,
            policy,
            _stream.view().get(),
            _mr.get_mr(),
        )

    cdef _BloomFilterBuffers buffers = _BloomFilterBuffers.__new__(_BloomFilterBuffers)
    buffers.c_obj = fetched.buffers
    cdef _BloomFilterBitset bitset
    cdef Py_ssize_t source
    views_per_source = []
    for source in range(num_sources):
        views = []
        for i in range(fetched.spans[source].size()):
            bitset = _BloomFilterBitset.__new__(_BloomFilterBitset)
            bitset.owner = buffers
            bitset.ptr = <uintptr_t>fetched.spans[source][i].data()
            bitset.size = fetched.spans[source][i].size()
            views.append(gpumemoryview(bitset))
        views_per_source.append(views)

    next_view = [0] * num_sources
    result = []
    for source in source_indices:
        result.append(views_per_source[source][next_view[source]])
        next_view[source] += 1
    return result


cpdef list fetch_byte_ranges_to_device(
    SourceInfo source_info,
    list byte_ranges,
    cpp_io_submission_policy policy,
    object stream: CudaStreamLike | None = None,
    DeviceMemoryResource mr=None,
    list source_indices=None,
):
    """Fetch byte ranges from Parquet sources into device memory.

    Parameters
    ----------
    source_info : SourceInfo
        Source describing one or more Parquet files.
    byte_ranges : list[ByteRangeInfo]
        Byte ranges to fetch, as returned by
        :meth:`~pylibcudf.io.experimental.HybridScanReader.filter_column_chunks_byte_ranges`,
        :meth:`~pylibcudf.io.experimental.HybridScanReader.payload_column_chunks_byte_ranges`,
        or
        :meth:`~pylibcudf.io.experimental.HybridScanReader.all_column_chunks_byte_ranges`,
        or by the corresponding
        :class:`~pylibcudf.io.experimental.HybridScanMultiFile` methods.
    policy : IOSubmissionPolicy
        Whether to serialize I/O submissions from callers.
    stream : Stream, optional
        CUDA stream.
    mr : DeviceMemoryResource, optional
        Device memory resource.
    source_indices : list[int], optional
        Index of the source each byte range is fetched from, as returned
        alongside the byte ranges by the
        :class:`~pylibcudf.io.experimental.HybridScanMultiFile` methods.
        Required unless ``source_info`` describes exactly one source.

    Returns
    -------
    list[gpumemoryview]
        One view per byte range, in the order of ``byte_ranges``. Each view
        holds a reference to the :class:`~rmm.DeviceBuffer` that owns its
        memory, keeping the allocation alive for as long as the view is
        referenced.

    Raises
    ------
    ValueError
        If ``source_indices`` is omitted and ``source_info`` does not describe
        exactly one source, if ``source_indices`` does not hold one index per
        byte range, or if an index does not refer to a source.
    """
    cdef Stream _stream = _get_stream(stream)
    cdef DeviceMemoryResource _mr = _get_memory_resource(mr)
    cdef vector[unique_ptr[datasource]] sources = make_datasources(source_info.c_obj)
    cdef Py_ssize_t num_sources = sources.size()
    cdef vector[vector[byte_range_info]] ranges_per_source
    cdef Py_ssize_t source
    source_indices = _group_byte_ranges_by_source(
        byte_ranges, source_indices, num_sources, ranges_per_source
    )

    cdef pair[
        vector[device_buffer], vector[vector[device_span[const_uint8_t]]]
    ] fetched
    with nogil:
        fetched = cpp_fetch_byte_ranges_to_device(
            sources,
            ranges_per_source,
            policy,
            _stream.view().get(),
            _mr.get_mr(),
        )

    if <Py_ssize_t>fetched.first.size() != num_sources:
        raise RuntimeError(
            f"Expected one device buffer per source, got {fetched.first.size()}"
        )
    cdef DeviceBuffer owner
    cdef gpumemoryview owner_gv
    cdef uintptr_t base
    cdef uintptr_t ptr
    cdef size_t n
    views_per_source = []
    for source in range(num_sources):
        views = []
        if fetched.second[source].size() != 0:
            owner = DeviceBuffer.c_from_unique_ptr(
                make_unique[device_buffer](move(fetched.first[source])),
                _stream,
                _mr,
            )
            owner_gv = gpumemoryview(owner)
            base = owner_gv.ptr
            for i in range(fetched.second[source].size()):
                ptr = <uintptr_t>fetched.second[source][i].data()
                n = fetched.second[source][i].size()
                views.append(
                    owner_gv.byte_slice(slice(ptr - base, ptr - base + n))
                )
        views_per_source.append(views)

    next_view = [0] * num_sources
    result = []
    for source in source_indices:
        result.append(views_per_source[source][next_view[source]])
        next_view[source] += 1
    return result


cpdef bytes fetch_page_index_to_host(
    SourceInfo source_info,
    ByteRangeInfo page_index_range,
):
    """Fetch parquet page index bytes to host memory.

    Parameters
    ----------
    source_info : SourceInfo
        Source describing a single Parquet file.
    page_index_range : ByteRangeInfo
        Byte range of the page index, as returned by
        :meth:`~pylibcudf.io.experimental.HybridScanReader.page_index_byte_range`.

    Returns
    -------
    bytes
        Raw page index bytes copied to Python host memory.

    Raises
    ------
    ValueError
        If ``source_info`` does not describe exactly one source.
    """
    cdef vector[unique_ptr[datasource]] sources = make_datasources(source_info.c_obj)
    if sources.size() != 1:
        raise ValueError(
            f"fetch_page_index_to_host requires exactly one source, "
            f"got {sources.size()}"
        )

    cdef unique_ptr[datasource.buffer] buf
    with nogil:
        buf = move(cpp_fetch_page_index_to_host(
            dereference(sources[0]),
            (<ByteRangeInfo>page_index_range).c_obj,
        ))

    if buf.get() is NULL:
        raise RuntimeError("fetch_page_index_to_host returned no buffer")
    cdef const uint8_t* ptr = buf.get().data()
    cdef size_t n = buf.get().size()
    return bytes(ptr[:n])
