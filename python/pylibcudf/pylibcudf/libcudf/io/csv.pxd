# SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
cimport pylibcudf.libcudf.io.types as cudf_io_types
cimport pylibcudf.libcudf.table.table_view as cudf_table_view
from libc.stdint cimport uint8_t
from libcpp cimport bool
from libcpp.map cimport map
from libcpp.memory cimport shared_ptr, unique_ptr
from libcpp.string cimport string
from libcpp.vector cimport vector
from pylibcudf.exception_handler cimport libcudf_exception_handler
from pylibcudf.libcudf.types cimport data_type, size_type
from cuda.bindings.cyruntime cimport cudaStream_t
from rmm.librmm.memory_resource cimport device_async_resource_ref

cdef extern from "cudf/io/csv.hpp" \
        namespace "cudf::io" nogil:

    cdef cppclass csv_reader_options:
        csv_reader_options() except +libcudf_exception_handler

        # Getter

        cudf_io_types.source_info get_source() except +libcudf_exception_handler
        # Reader settings
        cudf_io_types.compression_type get_compression()\
            except +libcudf_exception_handler
        size_t get_byte_range_offset() except +libcudf_exception_handler
        size_t get_byte_range_size() except +libcudf_exception_handler
        vector[string] get_names() except +libcudf_exception_handler
        string get_prefix() except +libcudf_exception_handler
        bool is_enabled_mangle_dupe_cols() except +libcudf_exception_handler

        # Filter settings
        vector[string] get_use_cols_names() except +libcudf_exception_handler
        vector[int] get_use_cols_indexes() except +libcudf_exception_handler
        size_type get_nrows() except +libcudf_exception_handler
        size_type get_skiprows() except +libcudf_exception_handler
        size_type get_skipfooter() except +libcudf_exception_handler
        size_type get_header() except +libcudf_exception_handler

        # Parsing settings
        char get_lineterminator() except +libcudf_exception_handler
        char get_delimiter() except +libcudf_exception_handler
        char get_thousands() except +libcudf_exception_handler
        char get_decimal() except +libcudf_exception_handler
        char get_comment() except +libcudf_exception_handler
        bool is_enabled_windowslinetermination() except +libcudf_exception_handler
        bool is_enabled_delim_whitespace() except +libcudf_exception_handler
        bool is_enabled_skipinitialspace() except +libcudf_exception_handler
        bool is_enabled_skip_blank_lines() except +libcudf_exception_handler
        cudf_io_types.quote_style get_quoting() except +libcudf_exception_handler
        char get_quotechar() except +libcudf_exception_handler
        bool is_enabled_doublequote() except +libcudf_exception_handler
        bool is_enabled_updated_quotes_detection() except +libcudf_exception_handler
        vector[string] get_parse_dates_names() except +libcudf_exception_handler
        vector[int] get_parse_dates_indexes() except +libcudf_exception_handler
        vector[string] get_parse_hex_names() except +libcudf_exception_handler
        vector[int] get_parse_hex_indexes() except +libcudf_exception_handler

        # Conversion settings
        vector[string] get_dtype() except +libcudf_exception_handler
        vector[string] get_true_values() except +libcudf_exception_handler
        vector[string] get_false_values() except +libcudf_exception_handler
        vector[string] get_na_values() except +libcudf_exception_handler
        bool is_enabled_keep_default_na() except +libcudf_exception_handler
        bool is_enabled_na_filter() except +libcudf_exception_handler
        bool is_enabled_dayfirst() except +libcudf_exception_handler

        # setter

        void set_source(cudf_io_types.source_info src) except +libcudf_exception_handler
        # Reader settings
        void set_compression(
            cudf_io_types.compression_type comp
        ) except +libcudf_exception_handler
        void set_byte_range_offset(size_t val) except +libcudf_exception_handler
        void set_byte_range_size(size_t val) except +libcudf_exception_handler
        void set_names(vector[string] val) except +libcudf_exception_handler
        void set_prefix(string pfx) except +libcudf_exception_handler
        void set_mangle_dupe_cols(bool val) except +libcudf_exception_handler

        # Filter settings
        void set_use_cols_names(
            vector[string] col_names
        ) except +libcudf_exception_handler
        void set_use_cols_indexes(
            vector[int] col_ind
        ) except +libcudf_exception_handler
        void set_nrows(size_type n_rows) except +libcudf_exception_handler
        void set_skiprows(size_type val) except +libcudf_exception_handler
        void set_skipfooter(size_type val) except +libcudf_exception_handler
        void set_header(size_type hdr) except +libcudf_exception_handler

        # Parsing settings
        void set_lineterminator(char val) except +libcudf_exception_handler
        void set_delimiter(char val) except +libcudf_exception_handler
        void set_thousands(char val) except +libcudf_exception_handler
        void set_decimal(char val) except +libcudf_exception_handler
        void set_comment(char val) except +libcudf_exception_handler
        void enable_windowslinetermination(bool val) except +libcudf_exception_handler
        void enable_delim_whitespace(bool val) except +libcudf_exception_handler
        void enable_skipinitialspace(bool val) except +libcudf_exception_handler
        void enable_skip_blank_lines(bool val) except +libcudf_exception_handler
        void set_quoting(
            cudf_io_types.quote_style style
        ) except +libcudf_exception_handler
        void set_quotechar(char val) except +libcudf_exception_handler
        void set_doublequote(bool val) except +libcudf_exception_handler
        void set_detect_whitespace_around_quotes(
            bool val
        ) except +libcudf_exception_handler
        void set_parse_dates(vector[string]) except +libcudf_exception_handler
        void set_parse_dates(vector[int]) except +libcudf_exception_handler
        void set_parse_hex(vector[string]) except +libcudf_exception_handler
        void set_parse_hex(vector[int]) except +libcudf_exception_handler

        # Conversion settings
        void set_dtypes(vector[data_type] types) except +libcudf_exception_handler
        void set_dtypes(map[string, data_type] types) except +libcudf_exception_handler
        void set_true_values(vector[string] vals) except +libcudf_exception_handler
        void set_false_values(vector[string] vals) except +libcudf_exception_handler
        void set_na_values(vector[string] vals) except +libcudf_exception_handler
        void enable_keep_default_na(bool val) except +libcudf_exception_handler
        void enable_na_filter(bool val) except +libcudf_exception_handler
        void enable_dayfirst(bool val) except +libcudf_exception_handler
        void set_timestamp_type(data_type type) except +libcudf_exception_handler

        @staticmethod
        csv_reader_options_builder builder(
            cudf_io_types.source_info src
        ) except +libcudf_exception_handler

    cdef cppclass csv_reader_options_builder:

        csv_reader_options_builder() except +libcudf_exception_handler
        csv_reader_options_builder(
            cudf_io_types.source_info src
        ) except +libcudf_exception_handler

        csv_reader_options_builder& source(
            cudf_io_types.source_info info
        ) except +libcudf_exception_handler
        # Reader settings
        csv_reader_options_builder& compression(
            cudf_io_types.compression_type comp
        ) except +libcudf_exception_handler
        csv_reader_options_builder& byte_range_offset(
            size_t val
        ) except +libcudf_exception_handler
        csv_reader_options_builder& byte_range_size(
            size_t val
        ) except +libcudf_exception_handler
        csv_reader_options_builder& names(
            vector[string] val
        ) except +libcudf_exception_handler
        csv_reader_options_builder& prefix(
            string pfx
        ) except +libcudf_exception_handler
        csv_reader_options_builder& mangle_dupe_cols(
            bool val
        ) except +libcudf_exception_handler

        # Filter settings
        csv_reader_options_builder& use_cols_names(
            vector[string] col_names
        ) except +libcudf_exception_handler
        csv_reader_options_builder& use_cols_indexes(
            vector[int] col_ind
        ) except +libcudf_exception_handler
        csv_reader_options_builder& nrows(
            size_type n_rows
        ) except +libcudf_exception_handler
        csv_reader_options_builder& skiprows(
            size_type val
        ) except +libcudf_exception_handler
        csv_reader_options_builder& skipfooter(
            size_type val
        ) except +libcudf_exception_handler
        csv_reader_options_builder& header(
            size_type hdr
        ) except +libcudf_exception_handler

        # Parsing settings
        csv_reader_options_builder& lineterminator(
            char val
        ) except +libcudf_exception_handler
        csv_reader_options_builder& delimiter(
            char val
        ) except +libcudf_exception_handler
        csv_reader_options_builder& thousands(
            char val
        ) except +libcudf_exception_handler
        csv_reader_options_builder& decimal(
            char val
        ) except +libcudf_exception_handler
        csv_reader_options_builder& comment(
            char val
        ) except +libcudf_exception_handler
        csv_reader_options_builder& windowslinetermination(
            bool val
        ) except +libcudf_exception_handler
        csv_reader_options_builder& delim_whitespace(
            bool val
        ) except +libcudf_exception_handler
        csv_reader_options_builder& skipinitialspace(
            bool val
        ) except +libcudf_exception_handler
        csv_reader_options_builder& skip_blank_lines(
            bool val
        ) except +libcudf_exception_handler
        csv_reader_options_builder& quoting(
            cudf_io_types.quote_style style
        ) except +libcudf_exception_handler
        csv_reader_options_builder& quotechar(
            char val
        ) except +libcudf_exception_handler
        csv_reader_options_builder& doublequote(
            bool val
        ) except +libcudf_exception_handler
        csv_reader_options_builder& detect_whitespace_around_quotes(
            bool val
        ) except +libcudf_exception_handler
        csv_reader_options_builder& parse_dates(
            vector[string]
        ) except +libcudf_exception_handler
        csv_reader_options_builder& parse_dates(
            vector[int]
        ) except +libcudf_exception_handler

        # Conversion settings
        csv_reader_options_builder& dtypes(
            vector[string] types) except +libcudf_exception_handler
        csv_reader_options_builder& dtypes(
            vector[data_type] types
        ) except +libcudf_exception_handler
        csv_reader_options_builder& dtypes(
            map[string, data_type] types
        ) except +libcudf_exception_handler
        csv_reader_options_builder& true_values(
            vector[string] vals
        ) except +libcudf_exception_handler
        csv_reader_options_builder& false_values(
            vector[string] vals
        ) except +libcudf_exception_handler
        csv_reader_options_builder& na_values(
            vector[string] vals
        ) except +libcudf_exception_handler
        csv_reader_options_builder& keep_default_na(
            bool val
        ) except +libcudf_exception_handler
        csv_reader_options_builder& na_filter(
            bool val
        ) except +libcudf_exception_handler
        csv_reader_options_builder& dayfirst(
            bool val
        ) except +libcudf_exception_handler
        csv_reader_options_builder& timestamp_type(
            data_type type
        ) except +libcudf_exception_handler

        csv_reader_options build() except +libcudf_exception_handler

    cdef cudf_io_types.table_with_metadata read_csv(
        csv_reader_options &options,
        cudaStream_t stream,
        device_async_resource_ref mr
    ) except +libcudf_exception_handler

    cdef cppclass csv_writer_options:
        csv_writer_options() except +libcudf_exception_handler

        cudf_io_types.sink_info get_sink() except +libcudf_exception_handler
        cudf_table_view.table_view get_table() except +libcudf_exception_handler
        cudf_io_types.table_metadata get_metadata() except +libcudf_exception_handler
        string get_na_rep() except +libcudf_exception_handler
        bool is_enabled_include_header() except +libcudf_exception_handler
        size_type get_rows_per_chunk() except +libcudf_exception_handler
        string get_line_terminator() except +libcudf_exception_handler
        char get_inter_column_delimiter() except +libcudf_exception_handler
        string get_true_value() except +libcudf_exception_handler
        string get_false_value() except +libcudf_exception_handler
        vector[string] get_names() except +libcudf_exception_handler

        # setter
        void set_metadata(
            cudf_io_types.table_metadata* val
        ) except +libcudf_exception_handler
        void set_na_rep(string val) except +libcudf_exception_handler
        void enable_include_header(bool val) except +libcudf_exception_handler
        void set_rows_per_chunk(size_type val) except +libcudf_exception_handler
        void set_line_terminator(string term) except +libcudf_exception_handler
        void set_inter_column_delimiter(char delim) except +libcudf_exception_handler
        void set_true_value(string val) except +libcudf_exception_handler
        void set_false_value(string val) except +libcudf_exception_handler
        void set_names(vector[string] val) except +libcudf_exception_handler

        @staticmethod
        csv_writer_options_builder builder(
            cudf_io_types.sink_info sink,
            cudf_table_view.table_view table
        ) except +libcudf_exception_handler

    cdef cppclass csv_writer_options_builder:
        csv_writer_options_builder() except +libcudf_exception_handler
        csv_writer_options_builder(
            cudf_io_types.sink_info sink,
            cudf_table_view.table_view table
        ) except +libcudf_exception_handler

        csv_writer_options_builder& names(
            vector[string] val
        ) except +libcudf_exception_handler
        csv_writer_options_builder& na_rep(
            string val
        ) except +libcudf_exception_handler
        csv_writer_options_builder& include_header(
            bool val
        ) except +libcudf_exception_handler
        csv_writer_options_builder& rows_per_chunk(
            size_type val
        ) except +libcudf_exception_handler
        csv_writer_options_builder& line_terminator(
            string term
        ) except +libcudf_exception_handler
        csv_writer_options_builder& inter_column_delimiter(
            char delim
        ) except +libcudf_exception_handler
        csv_writer_options_builder& true_value(
            string val
        ) except +libcudf_exception_handler
        csv_writer_options_builder& false_value(
            string val
        ) except +libcudf_exception_handler
        csv_writer_options_builder& quoting(
            cudf_io_types.quote_style style
        ) except +libcudf_exception_handler

        csv_writer_options build() except +libcudf_exception_handler

    cdef void write_csv(
        csv_writer_options args,
        cudaStream_t stream,
    ) except +libcudf_exception_handler

    cdef bool is_supported_write_csv(
        data_type type
    ) except +libcudf_exception_handler
