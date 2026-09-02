# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from cython.operator cimport dereference
from libc.stdint cimport uint8_t
from libc.stddef cimport size_t
from libcpp.memory cimport make_unique, unique_ptr
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
from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.io.hybrid_scan cimport (
    const_device_span_const_uint8_t,
    const_uint8_t,
)
from pylibcudf.libcudf.io.hybrid_scan_multifile cimport (
    const_FileMetaData,
    const_host_span_const_uint8_t,
    const_vector_size_type,
    hybrid_scan_multifile as cpp_hybrid_scan_multifile,
)
from pylibcudf.libcudf.io.parquet_schema cimport FileMetaData as c_cpp_FileMetaData
from pylibcudf.libcudf.io.text cimport byte_range_info
from pylibcudf.libcudf.io.types cimport table_with_metadata
from pylibcudf.libcudf.types cimport size_type
from pylibcudf.libcudf.utilities.span cimport device_span, host_span
from pylibcudf.utils cimport _get_memory_resource, _get_stream


__all__ = ["HybridScanMultifile"]


cdef class HybridScanMultifile:
    """Multi-file variant of the experimental Hybrid Scan Parquet reader.

    Vectorizes the single-source HybridScanReader APIs to support multiple
    Parquet sources. Inputs and outputs are indexed by source order except
    for the row mask which is a single BOOL8 column spanning all rows from
    all sources concatenated in source order, then row-group order within
    a source.

    For details, see :cpp:class:`cudf::io::parquet::experimental::hybrid_scan_multifile`

    Parameters
    ----------
    footer_bytes : list[Buffer]
        Parquet file footer bytes, one per source
    options : ParquetReaderOptions
        Parquet reader options

    Examples
    --------
    >>> import pylibcudf as plc
    >>> reader = plc.io.experimental.HybridScanMultifile(footer_bytes, options)
    >>> metadatas = reader.parquet_metadatas()
    >>> row_groups = reader.all_row_groups(options)
    """

    def __init__(self, list footer_bytes, ParquetReaderOptions options):
        cdef vector[host_span[const_uint8_t]] spans_vec
        cdef const uint8_t[::1] footer_view
        for footer in footer_bytes:
            footer_view = footer
            spans_vec.push_back(
                host_span[const_uint8_t](&footer_view[0], len(footer_view))
            )
        self.c_obj = make_unique[cpp_hybrid_scan_multifile](
            host_span[const_host_span_const_uint8_t](
                <const_host_span_const_uint8_t*>spans_vec.data(), spans_vec.size()
            ),
            options.c_obj
        )

    @staticmethod
    def from_parquet_metadata(list metadata, ParquetReaderOptions options):
        """Create a HybridScanMultifile from pre-populated metadata.

        Parameters
        ----------
        metadata : list[FileMetaData]
            Pre-populated Parquet file metadata, one per source
        options : ParquetReaderOptions
            Parquet reader options

        Returns
        -------
        HybridScanMultifile
        """
        cdef HybridScanMultifile reader = HybridScanMultifile.__new__(
            HybridScanMultifile
        )
        cdef vector[c_cpp_FileMetaData] meta_vec
        cdef c_FileMetaData m
        for m in metadata:
            meta_vec.push_back(dereference(m.c_obj))
        reader.c_obj = make_unique[cpp_hybrid_scan_multifile](
            host_span[const_FileMetaData](
                <const_FileMetaData*>meta_vec.data(), meta_vec.size()
            ),
            options.c_obj
        )
        return reader

    def parquet_metadatas(self):
        """Get the Parquet file footer metadata for all sources.

        Returns
        -------
        list[FileMetaData]
            Parquet file footer metadata, one per source
        """
        cdef vector[c_cpp_FileMetaData] metas = \
            self.c_obj.get()[0].parquet_metadatas()
        cdef unique_ptr[c_cpp_FileMetaData] meta_ptr
        cdef size_t i
        result = []
        for i in range(metas.size()):
            meta_ptr = make_unique[c_cpp_FileMetaData](metas[i])
            result.append(c_FileMetaData.from_libcudf(move(meta_ptr)))
        return result

    def page_index_byte_ranges(self):
        """Get the byte ranges of the page indexes for all sources.

        Returns
        -------
        list[ByteRangeInfo]
            Byte ranges of the page indexes, one per source
        """
        cdef vector[byte_range_info] ranges = \
            self.c_obj.get()[0].page_index_byte_ranges()
        return [ByteRangeInfo(r.offset(), r.size()) for r in ranges]

    def setup_page_indexes(self, list page_index_bytes):
        """Setup the per-source page index within each Parquet file metadata.

        Parameters
        ----------
        page_index_bytes : list[Buffer]
            Parquet page index buffer bytes, one per source
        """
        cdef vector[host_span[const_uint8_t]] spans_vec
        cdef const uint8_t[::1] page_view
        for page in page_index_bytes:
            page_view = page
            spans_vec.push_back(
                host_span[const_uint8_t](&page_view[0], len(page_view))
            )
        self.c_obj.get()[0].setup_page_indexes(
            host_span[const_host_span_const_uint8_t](
                <const_host_span_const_uint8_t*>spans_vec.data(), spans_vec.size()
            )
        )

    def all_row_groups(self, ParquetReaderOptions options):
        """Get all available per-source row group indices.

        Parameters
        ----------
        options : ParquetReaderOptions
            Parquet reader options

        Returns
        -------
        list[list[int]]
            Row group indices, one inner list per source
        """
        cdef vector[vector[size_type]] groups = \
            self.c_obj.get()[0].all_row_groups(options.c_obj)
        return [list(g) for g in groups]

    def total_rows_in_row_groups(self, list row_group_indices):
        """Get the total number of top-level rows across all sources.

        Parameters
        ----------
        row_group_indices : list[list[int]]
            Input row group indices, one inner list per source

        Returns
        -------
        int
            Total number of top-level rows across all sources
        """
        cdef vector[vector[size_type]] groups_vec = _to_per_source(row_group_indices)
        return self.c_obj.get()[0].total_rows_in_row_groups(
            host_span[const_vector_size_type](
                <const_vector_size_type*>groups_vec.data(), groups_vec.size()
            )
        )

    def reset_column_selection(self):
        """Reset the column selection state.

        Resets the internal column selection state forcing re-selection of columns in
        subsequent filter and read operations
        """
        self.c_obj.get()[0].reset_column_selection()

    def filter_row_groups_with_byte_range(
        self,
        list row_group_indices,
        ParquetReaderOptions options
    ):
        """Filter row groups using the byte range from the options.

        Parameters
        ----------
        row_group_indices : list[list[int]]
            Input row group indices, one inner list per source
        options : ParquetReaderOptions
            Parquet reader options

        Returns
        -------
        list[list[int]]
            Filtered row group indices, one inner list per source
        """
        cdef vector[vector[size_type]] groups_vec = _to_per_source(row_group_indices)
        cdef vector[vector[size_type]] filtered = \
            self.c_obj.get()[0].filter_row_groups_with_byte_range(
                host_span[const_vector_size_type](
                    <const_vector_size_type*>groups_vec.data(), groups_vec.size()
                ),
                options.c_obj
            )
        return [list(g) for g in filtered]

    def filter_row_groups_with_stats(
        self,
        list row_group_indices,
        ParquetReaderOptions options,
        object stream=None
    ):
        """Filter row groups using column chunk statistics.

        Parameters
        ----------
        row_group_indices : list[list[int]]
            Input row group indices, one inner list per source
        options : ParquetReaderOptions
            Parquet reader options
        stream : Stream, optional
            CUDA stream

        Returns
        -------
        list[list[int]]
            Filtered row group indices, one inner list per source
        """
        cdef Stream _stream = _get_stream(stream)
        cdef vector[vector[size_type]] groups_vec = _to_per_source(row_group_indices)
        cdef vector[vector[size_type]] filtered = \
            self.c_obj.get()[0].filter_row_groups_with_stats(
                host_span[const_vector_size_type](
                    <const_vector_size_type*>groups_vec.data(), groups_vec.size()
                ),
                options.c_obj,
                _stream.view().value()
            )
        return [list(g) for g in filtered]

    def bloom_filters_byte_ranges(
        self,
        list row_group_indices,
        ParquetReaderOptions options
    ):
        """Get byte ranges of bloom filters for row group pruning.

        Parameters
        ----------
        row_group_indices : list[list[int]]
            Input row group indices, one inner list per source
        options : ParquetReaderOptions
            Parquet reader options

        Returns
        -------
        tuple[list[ByteRangeInfo], list[int]]
            Flattened bloom filter byte ranges and their corresponding source indices
        """
        cdef vector[vector[size_type]] groups_vec = _to_per_source(row_group_indices)
        cdef pair[vector[byte_range_info], vector[size_type]] result = \
            self.c_obj.get()[0].bloom_filters_byte_ranges(
                host_span[const_vector_size_type](
                    <const_vector_size_type*>groups_vec.data(), groups_vec.size()
                ),
                options.c_obj
            )
        ranges = [ByteRangeInfo(r.offset(), r.size()) for r in result.first]
        source_indices = list(result.second)
        return (ranges, source_indices)

    def dictionary_pages_byte_ranges(
        self,
        list row_group_indices,
        ParquetReaderOptions options
    ):
        """Get byte ranges of column chunk dictionary pages for row group pruning.

        Parameters
        ----------
        row_group_indices : list[list[int]]
            Input row group indices, one inner list per source
        options : ParquetReaderOptions
            Parquet reader options

        Returns
        -------
        tuple[list[ByteRangeInfo], list[int]]
            Flattened dictionary page byte ranges and their corresponding source indices
        """
        cdef vector[vector[size_type]] groups_vec = _to_per_source(row_group_indices)
        cdef pair[vector[byte_range_info], vector[size_type]] result = \
            self.c_obj.get()[0].dictionary_pages_byte_ranges(
                host_span[const_vector_size_type](
                    <const_vector_size_type*>groups_vec.data(), groups_vec.size()
                ),
                options.c_obj
            )
        ranges = [ByteRangeInfo(r.offset(), r.size()) for r in result.first]
        source_indices = list(result.second)
        return (ranges, source_indices)

    def build_all_true_row_mask(
        self,
        list row_group_indices,
        object stream=None,
        DeviceMemoryResource mr=None
    ):
        """Build an all-true boolean column spanning all selected rows.

        Parameters
        ----------
        row_group_indices : list[list[int]]
            Input row group indices, one inner list per source
        stream : Stream, optional
            CUDA stream
        mr : DeviceMemoryResource, optional
            Device memory resource

        Returns
        -------
        Column
            An all-true boolean column spanning all selected rows across all sources
        """
        cdef Stream _stream = _get_stream(stream)
        mr = _get_memory_resource(mr)
        cdef vector[vector[size_type]] groups_vec = _to_per_source(row_group_indices)
        cdef unique_ptr[column] c_result = \
            self.c_obj.get()[0].build_all_true_row_mask(
                host_span[const_vector_size_type](
                    <const_vector_size_type*>groups_vec.data(), groups_vec.size()
                ),
                _stream.view().value(),
                mr.get_mr()
            )
        return Column.from_libcudf(move(c_result), _stream, mr)

    def build_row_mask_with_page_index_stats(
        self,
        list row_group_indices,
        ParquetReaderOptions options,
        object stream=None,
        DeviceMemoryResource mr=None
    ):
        """Build a boolean column indicating surviving rows from page stats.

        Parameters
        ----------
        row_group_indices : list[list[int]]
            Input row group indices, one inner list per source
        options : ParquetReaderOptions
            Parquet reader options
        stream : Stream, optional
            CUDA stream
        mr : DeviceMemoryResource, optional
            Device memory resource

        Returns
        -------
        Column
            Boolean column indicating surviving rows across all sources
        """
        cdef Stream _stream = _get_stream(stream)
        mr = _get_memory_resource(mr)
        cdef vector[vector[size_type]] groups_vec = _to_per_source(row_group_indices)
        cdef unique_ptr[column] c_result = \
            self.c_obj.get()[0].build_row_mask_with_page_index_stats(
                host_span[const_vector_size_type](
                    <const_vector_size_type*>groups_vec.data(), groups_vec.size()
                ),
                options.c_obj,
                _stream.view().value(),
                mr.get_mr()
            )
        return Column.from_libcudf(move(c_result), _stream, mr)

    def all_column_chunks_byte_ranges(
        self,
        list row_group_indices,
        ParquetReaderOptions options
    ):
        """Get byte ranges of column chunks of all columns.

        Parameters
        ----------
        row_group_indices : list[list[int]]
            Input row group indices, one inner list per source
        options : ParquetReaderOptions
            Parquet reader options

        Returns
        -------
        tuple[list[ByteRangeInfo], list[int]]
            Flattened byte ranges to column chunks of all columns and their
            corresponding source indices
        """
        cdef vector[vector[size_type]] groups_vec = _to_per_source(row_group_indices)
        cdef pair[vector[byte_range_info], vector[size_type]] result = \
            self.c_obj.get()[0].all_column_chunks_byte_ranges(
                host_span[const_vector_size_type](
                    <const_vector_size_type*>groups_vec.data(), groups_vec.size()
                ),
                options.c_obj
            )
        ranges = [ByteRangeInfo(r.offset(), r.size()) for r in result.first]
        source_indices = list(result.second)
        return (ranges, source_indices)

    def materialize_all_columns(
        self,
        list row_group_indices,
        list column_chunk_data,
        ParquetReaderOptions options,
        object stream=None,
        DeviceMemoryResource mr=None
    ):
        """Materialize all columns.

        Parameters
        ----------
        row_group_indices : list[list[int]]
            Input row group indices, one inner list per source
        column_chunk_data : list[Span]
            Span-like objects containing flattened column chunk data of all
            columns, in the same order as ``all_column_chunks_byte_ranges``
        options : ParquetReaderOptions
            Parquet reader options
        stream : Stream, optional
            CUDA stream
        mr : DeviceMemoryResource, optional
            Device memory resource

        Returns
        -------
        TableWithMetadata
            Table of materialized all columns and metadata
        """
        cdef Stream _stream = _get_stream(stream)
        mr = _get_memory_resource(mr)
        cdef vector[vector[size_type]] groups_vec = _to_per_source(row_group_indices)
        cdef vector[device_span[const_uint8_t]] spans_vec
        for span in column_chunk_data:
            spans_vec.push_back(_get_device_span(span))
        cdef table_with_metadata c_result = \
            self.c_obj.get()[0].materialize_all_columns(
                host_span[const_vector_size_type](
                    <const_vector_size_type*>groups_vec.data(), groups_vec.size()
                ),
                host_span[const_device_span_const_uint8_t](
                    <const_device_span_const_uint8_t*>spans_vec.data(), spans_vec.size()
                ),
                options.c_obj,
                _stream.view().value(),
                mr.get_mr()
            )
        return TableWithMetadata.from_libcudf(c_result, _stream, mr)


cdef vector[vector[size_type]] _to_per_source(list row_group_indices) except *:
    cdef vector[vector[size_type]] out
    cdef vector[size_type] inner
    for src in row_group_indices:
        inner = src
        out.push_back(inner)
    return out
