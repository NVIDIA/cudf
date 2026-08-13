# SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from libc.stdint cimport int32_t, int64_t, uint8_t
from libcpp cimport bool
from libcpp.functional cimport reference_wrapper
from libcpp.map cimport map
from libcpp.memory cimport shared_ptr, unique_ptr
from libcpp.optional cimport optional
from libcpp.string cimport string
from libcpp.vector cimport vector
from pylibcudf.exception_handler cimport libcudf_exception_handler
from pylibcudf.libcudf.expressions cimport expression
from pylibcudf.libcudf.io.datasource cimport datasource
from pylibcudf.libcudf.io.parquet_schema cimport FileMetaData
from pylibcudf.libcudf.io.types cimport (
    compression_type,
    dictionary_policy,
    partition_info,
    sink_info,
    source_info,
    statistics_freq,
    table_input_metadata,
    table_with_metadata,
)
from pylibcudf.libcudf.table.table_view cimport table_view
from pylibcudf.libcudf.types cimport data_type, size_type, type_id
from cuda.bindings.cyruntime cimport cudaStream_t
from rmm.librmm.memory_resource cimport device_async_resource_ref


cdef extern from "cudf/io/parquet.hpp" namespace "cudf::io" nogil:
    cdef cppclass parquet_reader_options:
        parquet_reader_options() except +libcudf_exception_handler
        source_info get_source() except +libcudf_exception_handler
        vector[vector[size_type]] get_row_groups() except +libcudf_exception_handler
        const optional[reference_wrapper[expression]]& get_filter()\
            except +libcudf_exception_handler
        data_type get_timestamp_type() except +libcudf_exception_handler
        type_id get_decimal_width() noexcept
        bool is_enabled_use_pandas_metadata() except +libcudf_exception_handler
        bool is_enabled_arrow_schema() except +libcudf_exception_handler
        bool is_enabled_allow_mismatched_pq_schemas() except +libcudf_exception_handler
        bool is_enabled_ignore_missing_columns() except +libcudf_exception_handler
        bool is_enabled_use_jit_filter() noexcept
        bool is_enabled_case_sensitive_names() noexcept
        # setter

        void set_source(source_info src) except +libcudf_exception_handler
        void set_filter(expression &filter) except +libcudf_exception_handler
        void set_columns(vector[string] col_names) except +libcudf_exception_handler
        void set_column_names(
                vector[string] col_names) except +libcudf_exception_handler
        void set_column_indices(
            vector[size_type] col_indices
        ) except +libcudf_exception_handler
        void set_column_field_ids(
            vector[int32_t] column_field_ids
        ) except +libcudf_exception_handler
        void set_num_rows(int64_t val) except +libcudf_exception_handler
        void set_row_groups(
            vector[vector[size_type]] row_grp
        ) except +libcudf_exception_handler
        void set_skip_rows(int64_t val) except +libcudf_exception_handler
        void enable_use_arrow_schema(bool val) except +libcudf_exception_handler
        void enable_allow_mismatched_pq_schemas(
            bool val
        ) except +libcudf_exception_handler
        void enable_ignore_missing_columns(
            bool val
        ) except +libcudf_exception_handler
        void enable_use_pandas_metadata(bool val) except +libcudf_exception_handler
        void set_timestamp_type(data_type type) except +libcudf_exception_handler
        void set_decimal_width(type_id width) noexcept
        void enable_case_sensitive_names(bool val) noexcept

        @staticmethod
        parquet_reader_options_builder builder(
            source_info src
        ) except +libcudf_exception_handler

    cdef cppclass parquet_reader_options_builder:
        parquet_reader_options_builder() except +libcudf_exception_handler
        parquet_reader_options_builder(
            source_info src
        ) except +libcudf_exception_handler
        parquet_reader_options_builder& columns(
            vector[string] col_names
        ) except +libcudf_exception_handler
        parquet_reader_options_builder& column_names(
            vector[string] col_names
        ) except +libcudf_exception_handler
        parquet_reader_options_builder& column_indices(
            vector[size_type] col_indices
        ) except +libcudf_exception_handler
        parquet_reader_options_builder& column_field_ids(
            vector[int32_t] column_field_ids
        ) except +libcudf_exception_handler
        parquet_reader_options_builder& row_groups(
            vector[vector[size_type]] row_grp
        ) except +libcudf_exception_handler
        parquet_reader_options_builder& convert_strings_to_categories(
            bool val
        ) except +libcudf_exception_handler
        parquet_reader_options_builder& use_pandas_metadata(
            bool val
        ) except +libcudf_exception_handler
        parquet_reader_options_builder& use_arrow_schema(
            bool val
        ) except +libcudf_exception_handler
        parquet_reader_options_builder& allow_mismatched_pq_schemas(
            bool val
        ) except +libcudf_exception_handler
        parquet_reader_options_builder& ignore_missing_columns(
            bool val
        ) except +libcudf_exception_handler
        parquet_reader_options_builder& timestamp_type(
            data_type type
        ) except +libcudf_exception_handler
        parquet_reader_options_builder& decimal_width(
            type_id width
        ) noexcept
        parquet_reader_options_builder& filter(
            const expression & f
        ) except +libcudf_exception_handler
        parquet_reader_options_builder& use_jit_filter(
            bool use_jit_filter
        ) noexcept
        parquet_reader_options_builder& case_sensitive_names(
            bool val
        ) noexcept
        parquet_reader_options build() except +libcudf_exception_handler

    cdef table_with_metadata read_parquet(
        parquet_reader_options args,
        cudaStream_t stream,
        device_async_resource_ref mr
    ) except +libcudf_exception_handler
    cdef table_with_metadata read_parquet(
        vector[unique_ptr[datasource]] sources,
        vector[FileMetaData] parquet_metadatas,
        const parquet_reader_options& args,
        cudaStream_t stream,
        device_async_resource_ref mr
    ) except +libcudf_exception_handler

    cdef cppclass parquet_writer_options_base:
        parquet_writer_options_base() except +libcudf_exception_handler
        sink_info get_sink_info() except +libcudf_exception_handler
        compression_type get_compression() except +libcudf_exception_handler
        statistics_freq get_stats_level() except +libcudf_exception_handler
        const optional[table_input_metadata]& get_metadata(
        ) except +libcudf_exception_handler
        size_t get_row_group_size_bytes() except +libcudf_exception_handler
        size_type get_row_group_size_rows() except +libcudf_exception_handler
        size_t get_max_page_size_bytes() except +libcudf_exception_handler
        size_type get_max_page_size_rows() except +libcudf_exception_handler
        size_t get_max_dictionary_size() except +libcudf_exception_handler
        bool is_enabled_write_arrow_schema() except +libcudf_exception_handler

        void set_metadata(
            table_input_metadata m
        ) except +libcudf_exception_handler
        void set_key_value_metadata(
            vector[map[string, string]] kvm
        ) except +libcudf_exception_handler
        void set_stats_level(
            statistics_freq sf
        ) except +libcudf_exception_handler
        void set_compression(
            compression_type compression
        ) except +libcudf_exception_handler
        void set_int96_timestamps(
            bool enabled
        ) except +libcudf_exception_handler
        void set_utc_timestamps(
            bool enabled
        ) except +libcudf_exception_handler
        void set_row_group_size_bytes(size_t val) except +libcudf_exception_handler
        void set_row_group_size_rows(size_type val) except +libcudf_exception_handler
        void set_max_page_size_bytes(size_t val) except +libcudf_exception_handler
        void set_max_page_size_rows(size_type val) except +libcudf_exception_handler
        void set_max_dictionary_size(size_t val) except +libcudf_exception_handler
        void enable_write_v2_headers(bool val) except +libcudf_exception_handler
        void enable_write_arrow_schema(bool val) except +libcudf_exception_handler
        void enable_page_level_compression(bool val) except +libcudf_exception_handler
        void set_dictionary_policy(
            dictionary_policy policy
        ) except +libcudf_exception_handler

    cdef cppclass parquet_writer_options(parquet_writer_options_base):
        parquet_writer_options() except +libcudf_exception_handler
        table_view get_table() except +libcudf_exception_handler
        string get_column_chunks_file_paths() except +libcudf_exception_handler
        void set_partitions(
            vector[partition_info] partitions
        ) except +libcudf_exception_handler
        void set_column_chunks_file_paths(
            vector[string] column_chunks_file_paths
        ) except +libcudf_exception_handler

        @staticmethod
        parquet_writer_options_builder builder(
            sink_info sink_,
            table_view table_
        ) except +libcudf_exception_handler

    cdef cppclass parquet_writer_options_builder_base[BuilderT, OptionsT]:
        parquet_writer_options_builder_base() except +libcudf_exception_handler

        BuilderT& metadata(
            table_input_metadata m
        ) except +libcudf_exception_handler
        BuilderT& key_value_metadata(
            vector[map[string, string]] metadata
        ) except +libcudf_exception_handler
        BuilderT& stats_level(
            statistics_freq sf
        ) except +libcudf_exception_handler
        BuilderT& compression(
            compression_type compression
        ) except +libcudf_exception_handler
        BuilderT& int96_timestamps(
            bool enabled
        ) except +libcudf_exception_handler
        BuilderT& utc_timestamps(
            bool enabled
        ) except +libcudf_exception_handler
        BuilderT& write_arrow_schema(
            bool enabled
        ) except +libcudf_exception_handler
        BuilderT& row_group_size_bytes(
            size_t val
        ) except +libcudf_exception_handler
        BuilderT& row_group_size_rows(
            size_type val
        ) except +libcudf_exception_handler
        BuilderT& max_page_size_bytes(
            size_t val
        ) except +libcudf_exception_handler
        BuilderT& max_page_size_rows(
            size_type val
        ) except +libcudf_exception_handler
        BuilderT& max_dictionary_size(
            size_t val
        ) except +libcudf_exception_handler
        BuilderT& write_v2_headers(
            bool enabled
        ) except +libcudf_exception_handler
        BuilderT& page_level_compression(
            bool enabled
        ) except +libcudf_exception_handler
        BuilderT& dictionary_policy(
            dictionary_policy val
        ) except +libcudf_exception_handler
        OptionsT build() except +libcudf_exception_handler

    cdef cppclass parquet_writer_options_builder(
            parquet_writer_options_builder_base[parquet_writer_options_builder,
                                                parquet_writer_options]):
        parquet_writer_options_builder() except +libcudf_exception_handler
        parquet_writer_options_builder(
            sink_info sink_,
            table_view table_
        ) except +libcudf_exception_handler
        parquet_writer_options_builder& partitions(
            vector[partition_info] partitions
        ) except +libcudf_exception_handler
        parquet_writer_options_builder& column_chunks_file_paths(
            vector[string] column_chunks_file_paths
        ) except +libcudf_exception_handler

    cdef unique_ptr[vector[uint8_t]] write_parquet(
        parquet_writer_options options,
        cudaStream_t stream,
    ) except +libcudf_exception_handler

    cdef bool is_supported_read_parquet(
        compression_type compression
    ) except +libcudf_exception_handler

    cdef bool is_supported_write_parquet(
        compression_type compression
    ) except +libcudf_exception_handler

    cdef cppclass chunked_parquet_writer_options(parquet_writer_options_base):
        chunked_parquet_writer_options() except +libcudf_exception_handler

        @staticmethod
        chunked_parquet_writer_options_builder builder(
            sink_info sink_,
        ) except +libcudf_exception_handler

    cdef cppclass chunked_parquet_writer_options_builder(
            parquet_writer_options_builder_base[chunked_parquet_writer_options_builder,
                                                chunked_parquet_writer_options]
            ):
        chunked_parquet_writer_options_builder() except +libcudf_exception_handler
        chunked_parquet_writer_options_builder(
            sink_info sink_,
        ) except +libcudf_exception_handler

    cdef cppclass chunked_parquet_writer:
        chunked_parquet_writer() except +libcudf_exception_handler
        chunked_parquet_writer(
            const chunked_parquet_writer_options& args,
            cudaStream_t stream
        ) except +libcudf_exception_handler
        chunked_parquet_writer& write(
            const table_view& table_,
            const vector[partition_info]& partitions
        ) except +libcudf_exception_handler
        unique_ptr[vector[uint8_t]] close(
            vector[string] column_chunks_file_paths,
        ) except +libcudf_exception_handler

    cdef cppclass chunked_parquet_reader:
        chunked_parquet_reader() except +libcudf_exception_handler
        chunked_parquet_reader(
            size_t chunk_read_limit,
            const parquet_reader_options& options,
            cudaStream_t stream,
            device_async_resource_ref mr
        ) except +libcudf_exception_handler
        chunked_parquet_reader(
            size_t chunk_read_limit,
            vector[unique_ptr[datasource]] sources,
            vector[FileMetaData] parquet_metadatas,
            const parquet_reader_options& options,
            cudaStream_t stream,
            device_async_resource_ref mr
        ) except +libcudf_exception_handler
        chunked_parquet_reader(
            size_t chunk_read_limit,
            size_t pass_read_limit,
            const parquet_reader_options& options,
            cudaStream_t stream,
            device_async_resource_ref mr
        ) except +libcudf_exception_handler
        chunked_parquet_reader(
            size_t chunk_read_limit,
            size_t pass_read_limit,
            vector[unique_ptr[datasource]] sources,
            vector[FileMetaData] parquet_metadatas,
            const parquet_reader_options& options,
            cudaStream_t stream,
            device_async_resource_ref mr
        ) except +libcudf_exception_handler
        bool has_next() except +libcudf_exception_handler
        table_with_metadata read_chunk() except +libcudf_exception_handler

    cdef unique_ptr[vector[uint8_t]] merge_row_group_metadata(
        const vector[unique_ptr[vector[uint8_t]]]& metadata_list
    ) except +libcudf_exception_handler
