# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from libc.stdint cimport uint8_t
from libcpp.memory cimport unique_ptr
from libcpp.pair cimport pair
from libcpp.vector cimport vector
from pylibcudf.exception_handler cimport libcudf_exception_handler
from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.io.hybrid_scan cimport (
    const_device_span_const_uint8_t,
    const_uint8_t,
)
from pylibcudf.libcudf.io.parquet cimport parquet_reader_options
from pylibcudf.libcudf.io.parquet_schema cimport FileMetaData
from pylibcudf.libcudf.io.text cimport byte_range_info
from pylibcudf.libcudf.io.types cimport table_with_metadata
from pylibcudf.libcudf.types cimport size_type
from pylibcudf.libcudf.utilities.span cimport host_span
from cuda.bindings.cyruntime cimport cudaStream_t
from rmm.librmm.memory_resource cimport device_async_resource_ref

ctypedef const host_span[const_uint8_t] const_host_span_const_uint8_t
ctypedef const vector[size_type] const_vector_size_type
ctypedef const FileMetaData const_FileMetaData


cdef extern from "cudf/io/experimental/hybrid_scan_multifile.hpp" \
        namespace "cudf::io::parquet::experimental" nogil:

    cdef cppclass hybrid_scan_multifile:
        hybrid_scan_multifile(
            host_span[const_host_span_const_uint8_t] footer_bytes,
            const parquet_reader_options& options
        ) except +libcudf_exception_handler

        hybrid_scan_multifile(
            host_span[const_FileMetaData] parquet_metadata,
            const parquet_reader_options& options
        ) except +libcudf_exception_handler

        vector[FileMetaData] parquet_metadatas(
        ) except +libcudf_exception_handler

        vector[byte_range_info] page_index_byte_ranges(
        ) except +libcudf_exception_handler

        void setup_page_indexes(
            host_span[const_host_span_const_uint8_t] page_index_bytes
        ) except +libcudf_exception_handler

        vector[vector[size_type]] all_row_groups(
            const parquet_reader_options& options
        ) except +libcudf_exception_handler

        size_type total_rows_in_row_groups(
            host_span[const_vector_size_type] row_group_indices
        ) except +libcudf_exception_handler

        void reset_column_selection() except +libcudf_exception_handler

        vector[vector[size_type]] filter_row_groups_with_byte_range(
            host_span[const_vector_size_type] row_group_indices,
            const parquet_reader_options& options
        ) except +libcudf_exception_handler

        vector[vector[size_type]] filter_row_groups_with_stats(
            host_span[const_vector_size_type] row_group_indices,
            const parquet_reader_options& options,
            cudaStream_t stream
        ) except +libcudf_exception_handler

        pair[
            vector[byte_range_info], vector[size_type]
        ] bloom_filters_byte_ranges(
            host_span[const_vector_size_type] row_group_indices,
            const parquet_reader_options& options
        ) except +libcudf_exception_handler

        pair[
            vector[byte_range_info], vector[size_type]
        ] dictionary_pages_byte_ranges(
            host_span[const_vector_size_type] row_group_indices,
            const parquet_reader_options& options
        ) except +libcudf_exception_handler

        unique_ptr[column] build_all_true_row_mask(
            host_span[const_vector_size_type] row_group_indices,
            cudaStream_t stream,
            device_async_resource_ref mr
        ) except +libcudf_exception_handler

        unique_ptr[column] build_row_mask_with_page_index_stats(
            host_span[const_vector_size_type] row_group_indices,
            const parquet_reader_options& options,
            cudaStream_t stream,
            device_async_resource_ref mr
        ) except +libcudf_exception_handler

        pair[
            vector[byte_range_info], vector[size_type]
        ] all_column_chunks_byte_ranges(
            host_span[const_vector_size_type] row_group_indices,
            const parquet_reader_options& options
        ) except +libcudf_exception_handler

        table_with_metadata materialize_all_columns(
            host_span[const_vector_size_type] row_group_indices,
            host_span[const_device_span_const_uint8_t] column_chunk_data,
            const parquet_reader_options& options,
            cudaStream_t stream,
            device_async_resource_ref mr
        ) except +libcudf_exception_handler
