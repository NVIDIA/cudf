# SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
cimport pylibcudf.libcudf.io.data_sink as cudf_io_data_sink
cimport pylibcudf.libcudf.io.datasource as cudf_io_datasource
cimport pylibcudf.libcudf.table.table_view as cudf_table_view
from libc.stdint cimport int32_t, uint8_t
from libcpp cimport bool
from libcpp.map cimport map
from libcpp.memory cimport unique_ptr
from libcpp.string cimport string
from libcpp.unordered_map cimport unordered_map
from libcpp.vector cimport vector
from libcpp.optional cimport optional
from pylibcudf.exception_handler cimport libcudf_exception_handler
from pylibcudf.libcudf.table.table cimport table
from pylibcudf.libcudf.types cimport size_type
from pylibcudf.libcudf.utilities.span cimport host_span, device_span

cdef extern from "<cstddef>" namespace "std":
    cdef cppclass byte:
        pass

ctypedef const byte const_byte

cdef extern from "cudf/io/types.hpp" \
        namespace "cudf::io" nogil:

    cpdef enum class quote_style(int32_t):
        MINIMAL
        ALL
        NONNUMERIC
        NONE

    cpdef enum class compression_type(int32_t):
        NONE
        AUTO
        SNAPPY
        GZIP
        BZIP2
        BROTLI
        ZIP
        XZ
        ZLIB
        LZ4
        LZO
        ZSTD

    cpdef enum class io_type(int32_t):
        FILEPATH
        HOST_BUFFER
        DEVICE_BUFFER
        VOID
        USER_IMPLEMENTED

    cpdef enum class statistics_freq(int32_t):
        STATISTICS_NONE
        STATISTICS_ROWGROUP
        STATISTICS_PAGE
        STATISTICS_COLUMN

    cpdef enum class dictionary_policy(int32_t):
        NEVER
        ADAPTIVE
        ALWAYS

    cpdef enum class column_encoding(int32_t):
        USE_DEFAULT
        DICTIONARY
        PLAIN
        DELTA_BINARY_PACKED
        DELTA_LENGTH_BYTE_ARRAY
        DELTA_BYTE_ARRAY
        BYTE_STREAM_SPLIT
        DIRECT
        DIRECT_V2
        DICTIONARY_V2

    cdef cppclass column_name_info:
        string name
        vector[column_name_info] children

    cdef cppclass table_metadata:
        table_metadata() except +libcudf_exception_handler

        map[string, string] user_data
        vector[unordered_map[string, string]] per_file_user_data
        vector[column_name_info] schema_info
        vector[size_t] num_rows_per_source
        # The following variables are currently only computed for Parquet reader
        size_type num_input_row_groups
        optional[size_type] num_row_groups_after_stats_filter
        optional[size_type] num_row_groups_after_bloom_filter

    cdef cppclass table_with_metadata:
        unique_ptr[table] tbl
        table_metadata metadata

    cdef cppclass column_in_metadata:
        column_in_metadata& set_name(const string& name)
        column_in_metadata& set_nullability(bool nullable)
        column_in_metadata& set_list_column_as_map()
        column_in_metadata& set_int96_timestamps(bool req)
        column_in_metadata& set_decimal_precision(uint8_t precision)
        column_in_metadata& child(size_type i)
        column_in_metadata& set_output_as_binary(bool binary)
        column_in_metadata& set_type_length(int32_t type_length)
        column_in_metadata& set_skip_compression(bool skip)
        column_in_metadata& set_encoding(column_encoding enc)
        string get_name()

    cdef cppclass table_input_metadata:
        table_input_metadata() except +libcudf_exception_handler
        table_input_metadata(
            const cudf_table_view.table_view& table
        ) except +libcudf_exception_handler

        vector[column_in_metadata] column_metadata

    cdef cppclass partition_info:
        size_type start_row
        size_type num_rows

        partition_info()
        partition_info(
            size_type start_row, size_type num_rows
        ) except +libcudf_exception_handler

    cdef cppclass filepath_source:
        string path
        optional[size_t] size

        filepath_source() except +libcudf_exception_handler
        filepath_source(string path) except +libcudf_exception_handler

    cdef cppclass source_info:
        const vector[string]& filepaths() except +libcudf_exception_handler
        const vector[filepath_source]& filepath_sources() \
            except +libcudf_exception_handler

        source_info() except +libcudf_exception_handler
        source_info(
            const vector[string] &filepaths
        ) except +libcudf_exception_handler
        source_info(
            vector[filepath_source] sources
        ) except +libcudf_exception_handler
        source_info(
            cudf_io_datasource.datasource *source
        ) except +libcudf_exception_handler
        source_info(
            const vector[cudf_io_datasource.datasource*] &datasources
        ) except +libcudf_exception_handler
        source_info(
            const host_span[const byte]& hspan
        ) except +libcudf_exception_handler
        source_info(
            const host_span[host_span[const byte]]& hspans
        ) except +libcudf_exception_handler
        source_info(
            device_span[const byte] dspan
        ) except +libcudf_exception_handler
        source_info(
            host_span[device_span[const byte]] dspan
        ) except +libcudf_exception_handler

    cdef cppclass sink_info:
        const vector[string]& filepaths()
        const vector[cudf_io_data_sink.data_sink *]& user_sinks()

        sink_info() except +libcudf_exception_handler
        sink_info(string file_path) except +libcudf_exception_handler
        sink_info(vector[string] file_path) except +libcudf_exception_handler
        sink_info(vector[char] * buffer) except +libcudf_exception_handler
        sink_info(
            cudf_io_data_sink.data_sink * user_sink
        ) except +libcudf_exception_handler
        sink_info(
            vector[cudf_io_data_sink.data_sink *] user_sink
        ) except +libcudf_exception_handler
