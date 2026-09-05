# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from libcpp cimport bool
from libcpp.pair cimport pair
from libcpp.vector cimport vector
from pylibcudf.exception_handler cimport libcudf_exception_handler
from pylibcudf.libcudf.column.column_view cimport column_view
from pylibcudf.libcudf.io.hybrid_scan cimport (
    const_device_span_const_uint8_t,
    const_uint8_t,
)
from pylibcudf.libcudf.io.parquet cimport parquet_reader_options
from pylibcudf.libcudf.io.parquet_metadata cimport const_FileMetaData
from pylibcudf.libcudf.io.parquet_schema cimport FileMetaData
from pylibcudf.libcudf.io.text cimport byte_range_info
from pylibcudf.libcudf.io.types cimport table_with_metadata
from pylibcudf.libcudf.types cimport size_type
from pylibcudf.libcudf.utilities.span cimport host_span
from cuda.bindings.cyruntime cimport cudaStream_t
from rmm.librmm.memory_resource cimport device_async_resource_ref

ctypedef const vector[size_type] const_vector_size_type
ctypedef host_span[const_uint8_t] host_span_const_uint8_t
ctypedef const host_span_const_uint8_t const_host_span_const_uint8_t

cdef extern from "cudf/io/experimental/hybrid_scan_multifile.hpp" \
        namespace "cudf::io::parquet::experimental" nogil:

    cdef cppclass hybrid_scan_multifile:
        hybrid_scan_multifile(
            host_span[const_FileMetaData] parquet_metadata,
            const parquet_reader_options& options
        ) except +libcudf_exception_handler

        vector[FileMetaData] parquet_metadatas() except +libcudf_exception_handler

        vector[byte_range_info] page_index_byte_ranges() \
            except +libcudf_exception_handler

        void setup_page_indexes(
            host_span[const_host_span_const_uint8_t] page_index_bytes
        ) except +libcudf_exception_handler

        size_type total_rows_in_row_groups(
            host_span[const_vector_size_type] row_group_indices
        ) except +libcudf_exception_handler

        pair[vector[byte_range_info], vector[size_type]] payload_pages_byte_ranges(
            host_span[const_vector_size_type] row_group_indices,
            const column_view& row_mask,
            const parquet_reader_options& options,
            cudaStream_t stream
        ) except +libcudf_exception_handler

        void setup_chunking_for_payload_columns(
            size_t chunk_read_limit,
            size_t pass_read_limit,
            host_span[const_vector_size_type] row_group_indices,
            const column_view& row_mask,
            host_span[const_device_span_const_uint8_t] page_data,
            const parquet_reader_options& options,
            cudaStream_t stream,
            device_async_resource_ref mr
        ) except +libcudf_exception_handler

        table_with_metadata materialize_payload_columns_chunk(
            const column_view& row_mask
        ) except +libcudf_exception_handler

        vector[vector[vector[size_type]]] construct_row_group_passes(
            host_span[const_vector_size_type] row_group_indices,
            size_t pass_read_limit,
        ) except +libcudf_exception_handler

        bool has_next_table_chunk() except +libcudf_exception_handler
