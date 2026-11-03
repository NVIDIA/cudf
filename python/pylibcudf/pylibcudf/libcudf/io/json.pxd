# SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
cimport pylibcudf.libcudf.io.types as cudf_io_types
cimport pylibcudf.libcudf.table.table_view as cudf_table_view
from libc.stdint cimport int32_t, uint8_t
from libcpp cimport bool
from libcpp.map cimport map
from libcpp.memory cimport shared_ptr, unique_ptr
from libcpp.optional cimport optional
from libcpp.string cimport string
from libcpp.vector cimport vector
from pylibcudf.exception_handler cimport libcudf_exception_handler
from pylibcudf.libcudf.types cimport data_type, size_type
from cuda.bindings.cyruntime cimport cudaStream_t
from rmm.librmm.memory_resource cimport device_async_resource_ref


cdef extern from "cudf/io/json.hpp" namespace "cudf::io" nogil:

    cdef struct schema_element:
        data_type type
        map[string, schema_element] child_types
        optional[vector[string]] column_order

    cpdef enum class json_recovery_mode_t(int32_t):
        FAIL
        RECOVER_WITH_NULL

    cdef cppclass json_reader_options:
        json_reader_options() except +libcudf_exception_handler
        cudf_io_types.source_info get_source() except +libcudf_exception_handler
        vector[string] get_dtypes() except +libcudf_exception_handler
        cudf_io_types.compression_type get_compression()\
            except +libcudf_exception_handler
        size_t get_byte_range_offset() except +libcudf_exception_handler
        size_t get_byte_range_size() except +libcudf_exception_handler
        size_t get_byte_range_size_with_padding() except +libcudf_exception_handler
        size_t get_byte_range_padding() except +libcudf_exception_handler
        char get_delimiter() except +libcudf_exception_handler
        bool is_enabled_lines() except +libcudf_exception_handler
        bool is_enabled_mixed_types_as_string() except +libcudf_exception_handler
        bool is_enabled_prune_columns() except +libcudf_exception_handler
        bool is_enabled_experimental() except +libcudf_exception_handler
        bool is_enabled_dayfirst() except +libcudf_exception_handler
        bool is_enabled_keep_quotes() except +libcudf_exception_handler
        bool is_enabled_normalize_single_quotes() except +libcudf_exception_handler
        bool is_enabled_normalize_whitespace() except +libcudf_exception_handler
        json_recovery_mode_t recovery_mode() except +libcudf_exception_handler
        bool is_strict_validation() except +libcudf_exception_handler
        bool is_allowed_numeric_leading_zeros() except +libcudf_exception_handler
        bool is_allowed_nonnumeric_numbers() except +libcudf_exception_handler
        bool is_allowed_unquoted_control_chars() except +libcudf_exception_handler
        vector[string] get_na_values() except +libcudf_exception_handler

        # setter
        void set_source(cudf_io_types.source_info src) except +libcudf_exception_handler
        void set_dtypes(vector[data_type] types) except +libcudf_exception_handler
        void set_dtypes(map[string, data_type] types) except +libcudf_exception_handler
        void set_dtypes(map[string, schema_element] types)\
            except +libcudf_exception_handler
        void set_dtypes(schema_element types) except +libcudf_exception_handler
        void set_compression(cudf_io_types.compression_type comp_type)\
            except +libcudf_exception_handler
        void set_byte_range_offset(size_t offset) except +libcudf_exception_handler
        void set_byte_range_size(size_t size) except +libcudf_exception_handler
        void set_delimiter(char delimiter) except +libcudf_exception_handler
        void enable_lines(bool val) except +libcudf_exception_handler
        void enable_mixed_types_as_string(bool val) except +libcudf_exception_handler
        void enable_prune_columns(bool val) except +libcudf_exception_handler
        void enable_experimental(bool val) except +libcudf_exception_handler
        void enable_dayfirst(bool val) except +libcudf_exception_handler
        void enable_keep_quotes(bool val) except +libcudf_exception_handler
        void enable_normalize_single_quotes(bool val) except +libcudf_exception_handler

        void enable_normalize_whitespace(bool val) except +libcudf_exception_handler
        void set_recovery_mode(json_recovery_mode_t val)\
            except +libcudf_exception_handler
        void set_strict_validation(bool val) except +libcudf_exception_handler
        void allow_numeric_leading_zeros(bool val) except +libcudf_exception_handler
        void allow_nonnumeric_numbers(bool val) except +libcudf_exception_handler
        void allow_unquoted_control_chars(bool val) except +libcudf_exception_handler
        void set_na_values(vector[string] vals) except +libcudf_exception_handler

        @staticmethod
        json_reader_options_builder builder(
            cudf_io_types.source_info src
        ) except +libcudf_exception_handler

    cdef cppclass json_reader_options_builder:
        json_reader_options_builder() except +libcudf_exception_handler
        json_reader_options_builder(
            cudf_io_types.source_info src
        ) except +libcudf_exception_handler
        json_reader_options_builder& dtypes(
            vector[data_type] types
        ) except +libcudf_exception_handler
        json_reader_options_builder& dtypes(
            map[string, schema_element] types
        ) except +libcudf_exception_handler
        json_reader_options_builder& dtypes(
            map[string, data_type] types
        ) except +libcudf_exception_handler
        json_reader_options_builder& dtypes(
            schema_element types
        ) except +libcudf_exception_handler
        json_reader_options_builder& compression(
            cudf_io_types.compression_type compression
        ) except +libcudf_exception_handler
        json_reader_options_builder& byte_range_offset(
            size_t offset
        ) except +libcudf_exception_handler
        json_reader_options_builder& byte_range_size(
            size_t size
        ) except +libcudf_exception_handler
        json_reader_options_builder& delimiter(
            char delimiter
        ) except +libcudf_exception_handler
        json_reader_options_builder& lines(
            bool val
        ) except +libcudf_exception_handler
        json_reader_options_builder& mixed_types_as_string(
            bool val
        ) except +libcudf_exception_handler
        json_reader_options_builder& prune_columns(
            bool val
        ) except +libcudf_exception_handler
        json_reader_options_builder& experimental(
            bool val
        ) except +libcudf_exception_handler
        json_reader_options_builder& dayfirst(
            bool val
        ) except +libcudf_exception_handler
        json_reader_options_builder& keep_quotes(
            bool val
        ) except +libcudf_exception_handler
        json_reader_options_builder& normalize_single_quotes(
            bool val
        ) except +libcudf_exception_handler
        json_reader_options_builder& normalize_whitespace(
            bool val
        ) except +libcudf_exception_handler
        json_reader_options_builder& recovery_mode(
            json_recovery_mode_t val
        ) except +libcudf_exception_handler

        json_reader_options_builder& strict_validation(bool val)\
            except +libcudf_exception_handler
        json_reader_options_builder& numeric_leading_zeros(bool val)\
            except +libcudf_exception_handler
        json_reader_options_builder& nonnumeric_numbers(bool val)\
            except +libcudf_exception_handler
        json_reader_options_builder& unquoted_control_chars(bool val)\
            except +libcudf_exception_handler
        json_reader_options_builder& na_values(vector[string] vals)\
            except +libcudf_exception_handler

        json_reader_options build() except +libcudf_exception_handler

    cdef cudf_io_types.table_with_metadata read_json(
        json_reader_options &options,
        cudaStream_t stream,
        device_async_resource_ref mr
    ) except +libcudf_exception_handler

    cdef cppclass json_writer_options:
        json_writer_options() except +libcudf_exception_handler
        cudf_io_types.sink_info get_sink() except +libcudf_exception_handler
        cudf_table_view.table_view get_table() except +libcudf_exception_handler
        string get_na_rep() except +libcudf_exception_handler
        bool is_enabled_include_nulls() except +libcudf_exception_handler
        bool is_enabled_lines() except +libcudf_exception_handler
        bool is_enabled_experimental() except +libcudf_exception_handler
        size_type get_rows_per_chunk() except +libcudf_exception_handler
        string get_true_value() except +libcudf_exception_handler
        string get_false_value() except +libcudf_exception_handler
        cudf_io_types.compression_type get_compression()\
            except +libcudf_exception_handler

        # setter
        void set_table(
            cudf_table_view.table_view tbl
        ) except +libcudf_exception_handler
        void set_metadata(
            cudf_io_types.table_metadata meta
        ) except +libcudf_exception_handler
        void set_na_rep(string val) except +libcudf_exception_handler
        void enable_include_nulls(bool val) except +libcudf_exception_handler
        void enable_lines(bool val) except +libcudf_exception_handler
        void set_rows_per_chunk(size_type val) except +libcudf_exception_handler
        void set_true_value(string val) except +libcudf_exception_handler
        void set_false_value(string val) except +libcudf_exception_handler
        void set_compression(
            cudf_io_types.compression_type comptype
        ) except +libcudf_exception_handler

        @staticmethod
        json_writer_options_builder builder(
            cudf_io_types.sink_info sink,
            cudf_table_view.table_view tbl
        ) except +libcudf_exception_handler

    cdef cppclass json_writer_options_builder:
        json_writer_options_builder() except +libcudf_exception_handler
        json_writer_options_builder(
            cudf_io_types.source_info src,
            cudf_table_view.table_view tbl
        ) except +libcudf_exception_handler
        json_writer_options_builder& table(
            cudf_table_view.table_view tbl
        ) except +libcudf_exception_handler
        json_writer_options_builder& metadata(
            cudf_io_types.table_metadata meta
        ) except +libcudf_exception_handler
        json_writer_options_builder& na_rep(
            string val
        ) except +libcudf_exception_handler
        json_writer_options_builder& include_nulls(
            bool val
        ) except +libcudf_exception_handler
        json_writer_options_builder& lines(
            bool val
        ) except +libcudf_exception_handler
        json_writer_options_builder& rows_per_chunk(
            size_type val
        ) except +libcudf_exception_handler
        json_writer_options_builder& true_value(
            string val
        ) except +libcudf_exception_handler
        json_writer_options_builder& false_value(
            string val
        ) except +libcudf_exception_handler
        json_writer_options_builder& compression(
            cudf_io_types.compression_type comptype
        ) except +libcudf_exception_handler
        json_writer_options_builder& utf8_escaped(
            bool val
        ) except +libcudf_exception_handler

        json_writer_options build() except +libcudf_exception_handler

    cdef cudf_io_types.table_with_metadata write_json(
        json_writer_options &options,
        cudaStream_t stream
    ) except +libcudf_exception_handler

    cdef bool is_supported_write_json(
        data_type type
    ) except +libcudf_exception_handler
