# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from libc.stdint cimport uint8_t
from libc.stddef cimport size_t
from libcpp cimport bool
from libcpp.memory cimport make_unique
from libcpp.pair cimport pair
from libcpp.utility cimport move
from libcpp.vector cimport vector

from rmm.pylibrmm.memory_resource cimport DeviceMemoryResource
from rmm.pylibrmm.stream cimport Stream

from pylibcudf.column cimport Column
from pylibcudf.io.experimental.hybrid_scan cimport _get_device_span
from pylibcudf.io.parquet cimport ParquetReaderOptions
from pylibcudf.io.parquet_metadata cimport FileMetaData as c_FileMetaData
from pylibcudf.io.text cimport ByteRangeInfo
from pylibcudf.io.types cimport TableWithMetadata
from pylibcudf.libcudf.column.column_view cimport column_view
from pylibcudf.libcudf.io.hybrid_scan cimport const_device_span_const_uint8_t
from pylibcudf.libcudf.io.hybrid_scan_multifile cimport (
    const_host_span_const_uint8_t,
    const_uint8_t,
    const_vector_size_type,
    host_span_const_uint8_t,
    hybrid_scan_multifile as cpp_hybrid_scan_multifile,
)
from pylibcudf.libcudf.io.parquet_metadata cimport const_FileMetaData
from pylibcudf.libcudf.io.parquet_schema cimport FileMetaData as cpp_FileMetaData
from pylibcudf.libcudf.io.text cimport byte_range_info
from pylibcudf.libcudf.io.types cimport table_with_metadata
from pylibcudf.libcudf.types cimport size_type
from pylibcudf.libcudf.utilities.span cimport device_span, host_span
from pylibcudf.utils cimport _get_memory_resource, _get_stream
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing_extensions import Buffer
    from pylibcudf.typing import CudaStreamLike

from pylibcudf.io.parquet_metadata import FileMetaData

__all__ = ["HybridScanMultiFile"]


cdef vector[vector[size_type]] _get_row_group_indices(
    object row_group_indices
) except *:
    """Convert per-source row group indices to a vector of vectors."""
    cdef vector[vector[size_type]] indices
    cdef vector[size_type] source_indices
    for source in row_group_indices:
        source_indices = source
        indices.push_back(source_indices)
    return indices


cdef class HybridScanMultiFile:
    """Experimental multi-source Parquet reader for highly selective filters.

    Vectorizes the :class:`HybridScanReader` APIs over multiple Parquet sources.
    Inputs and outputs are indexed by source order, except for the row mask which
    is a single boolean column spanning all rows of all sources concatenated in
    source order, then in row-group order within a source.

    For details, see
    :cpp:class:`cudf::io::parquet::experimental::hybrid_scan_multifile`

    Examples
    --------
    >>> import pylibcudf as plc
    >>> reader = plc.io.experimental.HybridScanMultiFile.from_parquet_metadatas(
    ...     metadatas, options)
    >>> byte_ranges, sources = reader.payload_pages_byte_ranges(
    ...     row_groups, row_mask, options)
    """

    def __init__(self):
        raise ValueError(
            "HybridScanMultiFile cannot be constructed directly. "
            "Use from_parquet_metadatas()."
        )

    @staticmethod
    def from_parquet_metadatas(
        object parquet_metadatas: Sequence[FileMetaData],
        ParquetReaderOptions options,
    ) -> HybridScanMultiFile:
        """Create a HybridScanMultiFile from pre-populated metadata.

        Parameters
        ----------
        parquet_metadatas : Sequence[FileMetaData]
            Pre-populated Parquet file metadata, one per source
        options : ParquetReaderOptions
            Parquet reader options

        Returns
        -------
        HybridScanMultiFile
        """
        cdef HybridScanMultiFile reader = HybridScanMultiFile.__new__(
            HybridScanMultiFile
        )
        cdef vector[cpp_FileMetaData] metadatas
        cdef c_FileMetaData metadata
        for metadata in parquet_metadatas:
            metadatas.push_back(metadata.c_obj.get()[0])
        with nogil:
            reader.c_obj = make_unique[cpp_hybrid_scan_multifile](
                host_span[const_FileMetaData](
                    <const_FileMetaData*>metadatas.data(), metadatas.size()
                ),
                options.c_obj
            )
        return reader

    def parquet_metadatas(self) -> list[FileMetaData]:
        """Get the Parquet file footer metadata of all sources.

        Returns
        -------
        list[FileMetaData]
            Parquet file footer metadata, one per source
        """
        cdef vector[cpp_FileMetaData] c_result
        with nogil:
            c_result = move(self.c_obj.get()[0].parquet_metadatas())
        return [
            c_FileMetaData.from_libcudf(
                make_unique[cpp_FileMetaData](move(c_result[source]))
            )
            for source in range(c_result.size())
        ]

    def page_index_byte_ranges(self) -> list[ByteRangeInfo]:
        """Get the byte range of the page index of all sources.

        Returns
        -------
        list[ByteRangeInfo]
            Byte range of the page index, one per source
        """
        cdef vector[byte_range_info] ranges
        with nogil:
            ranges = move(self.c_obj.get()[0].page_index_byte_ranges())
        return [ByteRangeInfo(r.offset(), r.size()) for r in ranges]

    def setup_page_indexes(
        self, object page_index_bytes: Sequence[Buffer]
    ) -> None:
        """Setup the page index within the Parquet file metadata of all sources.

        Parameters
        ----------
        page_index_bytes : Sequence[Buffer]
            Parquet page index buffer bytes, one per source
        """
        cdef vector[host_span_const_uint8_t] spans
        cdef const uint8_t[::1] page_index
        # The spans point into page_index_bytes, which outlives this call
        for page_index in page_index_bytes:
            if len(page_index) == 0:
                spans.push_back(host_span[const_uint8_t](<const_uint8_t*>0, 0))
            else:
                spans.push_back(
                    host_span[const_uint8_t](&page_index[0], len(page_index))
                )
        with nogil:
            self.c_obj.get()[0].setup_page_indexes(
                host_span[const_host_span_const_uint8_t](
                    <const_host_span_const_uint8_t*>spans.data(), spans.size()
                )
            )

    def total_rows_in_row_groups(
        self, list row_group_indices: list[list[int]]
    ) -> int:
        """Get the total number of top-level rows in the row groups.

        Parameters
        ----------
        row_group_indices : list[list[int]]
            Input row group indices, one list per source

        Returns
        -------
        int
            Total number of top-level rows across all sources
        """
        cdef vector[vector[size_type]] indices = _get_row_group_indices(
            row_group_indices
        )
        cdef size_type result
        with nogil:
            result = self.c_obj.get()[0].total_rows_in_row_groups(
                host_span[const_vector_size_type](
                    <const_vector_size_type*>indices.data(), indices.size()
                )
            )
        return result

    def payload_pages_byte_ranges(
        self,
        list row_group_indices: list[list[int]],
        Column row_mask,
        ParquetReaderOptions options,
        object stream: CudaStreamLike | None = None
    ) -> tuple[list[ByteRangeInfo], list[int]]:
        """Get byte ranges of the pages of payload columns.

        Byte ranges are flattened in source, row group, column chunk, and page
        order. Dictionary pages precede data pages within a column chunk, and
        pruned pages are returned as empty byte ranges.

        Parameters
        ----------
        row_group_indices : list[list[int]]
            Input row group indices, one list per source
        row_mask : Column
            Boolean column indicating which rows need to be read
        options : ParquetReaderOptions
            Parquet reader options
        stream : Stream, optional
            CUDA stream

        Returns
        -------
        tuple[list[ByteRangeInfo], list[int]]
            Flattened byte ranges to the pages of payload columns and the source
            index of each byte range
        """
        cdef vector[vector[size_type]] indices = _get_row_group_indices(
            row_group_indices
        )
        cdef Stream _stream = _get_stream(stream)
        cdef column_view mask_view = row_mask.view()
        cdef pair[vector[byte_range_info], vector[size_type]] c_result
        with nogil:
            c_result = move(self.c_obj.get()[0].payload_pages_byte_ranges(
                host_span[const_vector_size_type](
                    <const_vector_size_type*>indices.data(), indices.size()
                ),
                mask_view,
                options.c_obj,
                _stream.view().value()
            ))
        return (
            [ByteRangeInfo(r.offset(), r.size()) for r in c_result.first],
            list(c_result.second),
        )

    def setup_chunking_for_payload_columns(
        self,
        size_t chunk_read_limit,
        size_t pass_read_limit,
        list row_group_indices: list[list[int]],
        Column row_mask,
        object page_data: Sequence,
        ParquetReaderOptions options,
        object stream: CudaStreamLike | None = None,
        DeviceMemoryResource mr=None
    ) -> None:
        """Setup chunking information for payload columns read page by page.

        The data page mask is inferred from ``page_data``, which must have the
        same shape as the byte ranges returned by
        :meth:`payload_pages_byte_ranges`.

        Parameters
        ----------
        chunk_read_limit : int
            Limit on bytes returned per chunk (0 for no limit)
        pass_read_limit : int
            Limit on memory for reading/decompressing (0 for no limit)
        row_group_indices : list[list[int]]
            Input row group indices, one list per source
        row_mask : Column
            Boolean column indicating which rows need to be read
        page_data : Sequence
            Span-like objects containing the page data of payload columns, in
            the same order as the byte ranges returned by
            :meth:`payload_pages_byte_ranges`. ``None`` indicates a pruned page
        options : ParquetReaderOptions
            Parquet reader options
        stream : Stream, optional
            CUDA stream
        mr : DeviceMemoryResource, optional
            Device memory resource
        """
        cdef vector[vector[size_type]] indices = _get_row_group_indices(
            row_group_indices
        )

        cdef vector[device_span[const_uint8_t]] spans_vec
        for page in page_data:
            if page is None:
                spans_vec.push_back(device_span[const_uint8_t]())
            else:
                spans_vec.push_back(_get_device_span(page))

        self._stream = _get_stream(stream)
        self.mr = _get_memory_resource(mr)
        # keep reference to avoid use-after-free of device spans
        self._payload_page_data = page_data

        cdef column_view mask_view = row_mask.view()
        with nogil:
            self.c_obj.get()[0].setup_chunking_for_payload_columns(
                chunk_read_limit,
                pass_read_limit,
                host_span[const_vector_size_type](
                    <const_vector_size_type*>indices.data(), indices.size()
                ),
                mask_view,
                host_span[const_device_span_const_uint8_t](
                    <const_device_span_const_uint8_t*>spans_vec.data(),
                    spans_vec.size()
                ),
                options.c_obj,
                self._stream.view().value(),
                self.mr.get_mr()
            )

    def materialize_payload_columns_chunk(
        self,
        Column row_mask,
    ) -> TableWithMetadata:
        """Materialize a chunk of payload columns.

        Parameters
        ----------
        row_mask : Column
            Boolean column indicating which rows need to be read

        Returns
        -------
        TableWithMetadata
            Table chunk of materialized payload columns and metadata
        """
        cdef column_view mask_view = row_mask.view()
        cdef table_with_metadata c_result
        cdef bool more_chunks
        with nogil:
            c_result = move(self.c_obj.get()[0].materialize_payload_columns_chunk(
                mask_view
            ))
            more_chunks = self.c_obj.get()[0].has_next_table_chunk()
        if not more_chunks:
            self._payload_page_data = None
        return TableWithMetadata.from_libcudf(c_result, self._stream, self.mr)

    def construct_row_group_passes(
        self,
        list row_group_indices: list[list[int]],
        size_t pass_read_limit,
    ) -> list[list[list[int]]]:
        """Partition row groups into passes such that the GPU memory required to
        materialize a pass is bounded by the specified limit.

        Note that ``pass_read_limit`` is a hint, not an absolute limit. i.e. if
        a row group cannot fit within the limit, it will still constitute a valid
        pass.

        Parameters
        ----------
        row_group_indices : list[list[int]]
            Input row group indices, one list per source
        pass_read_limit : int
            Limit on the amount of memory used for reading and decompressing data
            or 0 if there is no limit

        Returns
        -------
        list[list[list[int]]]
            Per-source row group indices, one list per pass

        Raises
        ------
        ValueError
            If ``row_group_indices`` is empty.
        """
        cdef vector[vector[size_type]] indices = _get_row_group_indices(
            row_group_indices
        )
        cdef vector[vector[vector[size_type]]] passes
        with nogil:
            passes = move(self.c_obj.get()[0].construct_row_group_passes(
                host_span[const_vector_size_type](
                    <const_vector_size_type*>indices.data(), indices.size()
                ),
                pass_read_limit
            ))
        return passes

    def has_next_table_chunk(self) -> bool:
        """Check if there is any parquet data left to read.

        Returns
        -------
        bool
            True if there is data left to read
        """
        cdef bool result
        with nogil:
            result = self.c_obj.get()[0].has_next_table_chunk()
        return result
