# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from libc.stdint cimport uint8_t
from libcpp cimport bool
from libcpp.memory cimport unique_ptr
from libcpp.pair cimport pair
from libcpp.span cimport span as std_span
from libcpp.vector cimport vector
from pylibcudf.exception_handler cimport libcudf_exception_handler
from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.column.column_view cimport column_view, mutable_column_view
from pylibcudf.libcudf.io.parquet cimport parquet_reader_options
from pylibcudf.libcudf.io.parquet_schema cimport FileMetaData
from pylibcudf.libcudf.io.text cimport byte_range_info
from pylibcudf.libcudf.io.types cimport table_with_metadata
from pylibcudf.libcudf.types cimport size_type
from pylibcudf.libcudf.utilities.span cimport device_span, host_span
from cuda.bindings.cyruntime cimport cudaStream_t
from rmm.librmm.memory_resource cimport device_async_resource_ref

ctypedef const uint8_t const_uint8_t
ctypedef const size_type const_size_type
ctypedef const device_span[const_uint8_t] const_device_span_const_uint8_t

cdef extern from "cudf/io/experimental/hybrid_scan.hpp" \
        namespace "cudf::io::parquet::experimental" nogil:

    cpdef enum class use_data_page_mask(bool):
        YES
        NO

    cdef cppclass hybrid_scan_reader:
        hybrid_scan_reader(
            host_span[const_uint8_t] footer_bytes,
            const parquet_reader_options& options
        ) except +libcudf_exception_handler

        hybrid_scan_reader(
            const FileMetaData& parquet_metadata,
            const parquet_reader_options& options
        ) except +libcudf_exception_handler

        FileMetaData parquet_metadata() except +libcudf_exception_handler

        byte_range_info page_index_byte_range() except +libcudf_exception_handler

        void setup_page_index(
            host_span[const_uint8_t] page_index_bytes
        ) except +libcudf_exception_handler

        vector[size_type] all_row_groups(
            const parquet_reader_options& options
        ) except +libcudf_exception_handler

        size_type total_rows_in_row_groups(
            std_span[const_size_type] row_group_indices
        ) except +libcudf_exception_handler

        void reset_column_selection() except +libcudf_exception_handler

        vector[size_type] filter_row_groups_with_stats(
            std_span[const_size_type] row_group_indices,
            const parquet_reader_options& options,
            cudaStream_t stream
        ) except +libcudf_exception_handler

        pair[
            vector[byte_range_info], vector[byte_range_info]
        ] secondary_filters_byte_ranges(
            std_span[const_size_type] row_group_indices,
            const parquet_reader_options& options
        ) except +libcudf_exception_handler

        vector[size_type] filter_row_groups_with_dictionary_pages(
            std_span[const_device_span_const_uint8_t] dictionary_page_data,
            std_span[const_size_type] row_group_indices,
            const parquet_reader_options& options,
            cudaStream_t stream
        ) except +libcudf_exception_handler

        vector[size_type] filter_row_groups_with_bloom_filters(
            std_span[const_device_span_const_uint8_t] bloom_filter_data,
            std_span[const_size_type] row_group_indices,
            const parquet_reader_options& options,
            cudaStream_t stream
        ) except +libcudf_exception_handler

        unique_ptr[column] build_row_mask_with_page_index_stats(
            std_span[const_size_type] row_group_indices,
            const parquet_reader_options& options,
            cudaStream_t stream,
            device_async_resource_ref mr
        ) except +libcudf_exception_handler

        vector[byte_range_info] filter_column_chunks_byte_ranges(
            std_span[const_size_type] row_group_indices,
            const parquet_reader_options& options
        ) except +libcudf_exception_handler

        table_with_metadata materialize_filter_columns(
            std_span[const_size_type] row_group_indices,
            std_span[const_device_span_const_uint8_t] column_chunk_data,
            mutable_column_view& row_mask,
            use_data_page_mask mask_data_pages,
            const parquet_reader_options& options,
            cudaStream_t stream,
            device_async_resource_ref mr
        ) except +libcudf_exception_handler

        vector[byte_range_info] payload_column_chunks_byte_ranges(
            std_span[const_size_type] row_group_indices,
            const parquet_reader_options& options
        ) except +libcudf_exception_handler

        table_with_metadata materialize_payload_columns(
            std_span[const_size_type] row_group_indices,
            std_span[const_device_span_const_uint8_t] column_chunk_data,
            const column_view& row_mask,
            use_data_page_mask mask_data_pages,
            const parquet_reader_options& options,
            cudaStream_t stream,
            device_async_resource_ref mr
        ) except +libcudf_exception_handler

        vector[byte_range_info] all_column_chunks_byte_ranges(
            std_span[const_size_type] row_group_indices,
            const parquet_reader_options& options
        ) except +libcudf_exception_handler

        table_with_metadata materialize_all_columns(
            std_span[const_size_type] row_group_indices,
            std_span[const_device_span_const_uint8_t] column_chunk_data,
            const parquet_reader_options& options,
            cudaStream_t stream,
            device_async_resource_ref mr
        ) except +libcudf_exception_handler

        void setup_chunking_for_filter_columns(
            size_t chunk_read_limit,
            size_t pass_read_limit,
            std_span[const_size_type] row_group_indices,
            const column_view& row_mask,
            use_data_page_mask mask_data_pages,
            std_span[const_device_span_const_uint8_t] column_chunk_data,
            const parquet_reader_options& options,
            cudaStream_t stream,
            device_async_resource_ref mr
        ) except +libcudf_exception_handler

        table_with_metadata materialize_filter_columns_chunk(
            mutable_column_view& row_mask
        ) except +libcudf_exception_handler

        void setup_chunking_for_payload_columns(
            size_t chunk_read_limit,
            size_t pass_read_limit,
            std_span[const_size_type] row_group_indices,
            const column_view& row_mask,
            use_data_page_mask mask_data_pages,
            std_span[const_device_span_const_uint8_t] column_chunk_data,
            const parquet_reader_options& options,
            cudaStream_t stream,
            device_async_resource_ref mr
        ) except +libcudf_exception_handler

        table_with_metadata materialize_payload_columns_chunk(
            const column_view& row_mask
        ) except +libcudf_exception_handler

        vector[vector[size_type]] construct_row_group_passes(
            std_span[const_size_type] row_group_indices,
            size_t pass_read_limit,
        ) except +libcudf_exception_handler

        bool has_next_table_chunk() except +libcudf_exception_handler
