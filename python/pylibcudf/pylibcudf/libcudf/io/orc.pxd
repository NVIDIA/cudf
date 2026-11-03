# SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
cimport pylibcudf.libcudf.io.types as cudf_io_types
cimport pylibcudf.libcudf.table.table_view as cudf_table_view
from libc.stdint cimport int64_t, uint8_t
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


cdef extern from "cudf/io/orc.hpp" namespace "cudf::io" nogil:

    cdef cppclass orc_reader_options:
        orc_reader_options() except +libcudf_exception_handler

        cudf_io_types.source_info get_source() except +libcudf_exception_handler
        vector[vector[size_type]] get_stripes() except +libcudf_exception_handler
        int64_t get_skip_rows() except +libcudf_exception_handler
        optional[int64_t] get_num_rows() except +libcudf_exception_handler
        bool is_enabled_use_index() except +libcudf_exception_handler
        bool is_enabled_use_np_dtypes() except +libcudf_exception_handler
        data_type get_timestamp_type() except +libcudf_exception_handler
        bool is_enabled_decimals_as_float64() except +libcudf_exception_handler
        int get_forced_decimals_scale() except +libcudf_exception_handler

        void set_source(cudf_io_types.source_info src) except +libcudf_exception_handler
        void set_columns(vector[string] col_names) except +libcudf_exception_handler
        void set_stripes(
            vector[vector[size_type]] strps
        ) except +libcudf_exception_handler
        void set_skip_rows(int64_t rows) except +libcudf_exception_handler
        void set_num_rows(int64_t nrows) except +libcudf_exception_handler
        void enable_use_index(bool val) except +libcudf_exception_handler
        void enable_use_np_dtypes(bool val) except +libcudf_exception_handler
        void set_timestamp_type(data_type type) except +libcudf_exception_handler
        void set_decimal128_columns(
            vector[string] val
        ) except +libcudf_exception_handler

        @staticmethod
        orc_reader_options_builder builder(
            cudf_io_types.source_info src
        ) except +libcudf_exception_handler

    cdef cppclass orc_reader_options_builder:
        orc_reader_options_builder() except +libcudf_exception_handler
        orc_reader_options_builder(
            cudf_io_types.source_info &src
        ) except +libcudf_exception_handler

        orc_reader_options_builder& columns(
            vector[string] col_names
        ) except +libcudf_exception_handler
        orc_reader_options_builder& \
            stripes(vector[vector[size_type]] strps) except +libcudf_exception_handler
        orc_reader_options_builder& skip_rows(
            int64_t rows
        ) except +libcudf_exception_handler
        orc_reader_options_builder& num_rows(
            int64_t nrows
        ) except +libcudf_exception_handler
        orc_reader_options_builder& use_index(
            bool val
        ) except +libcudf_exception_handler
        orc_reader_options_builder& use_np_dtypes(
            bool val
        ) except +libcudf_exception_handler
        orc_reader_options_builder& timestamp_type(
            data_type type
        ) except +libcudf_exception_handler

        orc_reader_options build() except +libcudf_exception_handler

    cdef cudf_io_types.table_with_metadata read_orc(
        orc_reader_options opts,
        cudaStream_t stream,
        device_async_resource_ref mr,
    ) except +libcudf_exception_handler

    cdef cppclass orc_writer_options:
        orc_writer_options()
        cudf_io_types.sink_info get_sink() except +libcudf_exception_handler
        cudf_io_types.compression_type get_compression()\
            except +libcudf_exception_handler
        bool is_enabled_statistics() except +libcudf_exception_handler
        size_t get_stripe_size_bytes() except +libcudf_exception_handler
        size_type get_stripe_size_rows() except +libcudf_exception_handler
        size_type get_row_index_stride() except +libcudf_exception_handler
        cudf_table_view.table_view get_table() except +libcudf_exception_handler
        const optional[cudf_io_types.table_input_metadata]& get_metadata(
        ) except +libcudf_exception_handler

        # setter
        void set_compression(
            cudf_io_types.compression_type comp
        ) except +libcudf_exception_handler
        void enable_statistics(bool val) except +libcudf_exception_handler
        void set_stripe_size_bytes(size_t val) except +libcudf_exception_handler
        void set_stripe_size_rows(size_type val) except +libcudf_exception_handler
        void set_row_index_stride(size_type val) except +libcudf_exception_handler
        void set_table(cudf_table_view.table_view tbl) except +libcudf_exception_handler
        void set_metadata(
            cudf_io_types.table_input_metadata meta
        ) except +libcudf_exception_handler
        void set_key_value_metadata(
            map[string, string] kvm
        ) except +libcudf_exception_handler

        @staticmethod
        orc_writer_options_builder builder(
            cudf_io_types.sink_info &sink,
            cudf_table_view.table_view &tbl
        ) except +libcudf_exception_handler

    cdef cppclass orc_writer_options_builder:
        # setter
        orc_writer_options_builder& compression(
            cudf_io_types.compression_type comp
        ) except +libcudf_exception_handler
        orc_writer_options_builder& enable_statistics(
            cudf_io_types.statistics_freq val
        ) except +libcudf_exception_handler
        orc_writer_options_builder& stripe_size_bytes(
            size_t val
        ) except +libcudf_exception_handler
        orc_writer_options_builder& stripe_size_rows(
            size_type val
        ) except +libcudf_exception_handler
        orc_writer_options_builder& row_index_stride(
            size_type val
        ) except +libcudf_exception_handler
        orc_writer_options_builder& table(
            cudf_table_view.table_view tbl
        ) except +libcudf_exception_handler
        orc_writer_options_builder& metadata(
            cudf_io_types.table_input_metadata meta
        ) except +libcudf_exception_handler
        orc_writer_options_builder& key_value_metadata(
            map[string, string] kvm
        ) except +libcudf_exception_handler

        orc_writer_options build() except +libcudf_exception_handler

    cdef void write_orc(
        orc_writer_options options,
        cudaStream_t stream,
    ) except +libcudf_exception_handler

    cdef bool is_supported_read_orc(
        cudf_io_types.compression_type compression
    ) except +libcudf_exception_handler

    cdef bool is_supported_write_orc(
        cudf_io_types.compression_type compression
    ) except +libcudf_exception_handler

    cdef cppclass chunked_orc_writer_options:
        chunked_orc_writer_options() except +libcudf_exception_handler
        cudf_io_types.sink_info get_sink() except +libcudf_exception_handler
        cudf_io_types.compression_type get_compression()\
            except +libcudf_exception_handler
        bool enable_statistics() except +libcudf_exception_handler
        size_t stripe_size_bytes() except +libcudf_exception_handler
        size_type stripe_size_rows() except +libcudf_exception_handler
        size_type row_index_stride() except +libcudf_exception_handler
        cudf_table_view.table_view get_table() except +libcudf_exception_handler
        const optional[cudf_io_types.table_input_metadata]& get_metadata(
        ) except +libcudf_exception_handler

        # setter
        void set_compression(
            cudf_io_types.compression_type comp
        ) except +libcudf_exception_handler
        void enable_statistics(bool val) except +libcudf_exception_handler
        void set_stripe_size_bytes(size_t val) except +libcudf_exception_handler
        void set_stripe_size_rows(size_type val) except +libcudf_exception_handler
        void set_row_index_stride(size_type val) except +libcudf_exception_handler
        void set_table(cudf_table_view.table_view tbl) except +libcudf_exception_handler
        void set_metadata(
            cudf_io_types.table_input_metadata meta
        ) except +libcudf_exception_handler
        void set_key_value_metadata(
            map[string, string] kvm
        ) except +libcudf_exception_handler

        @staticmethod
        chunked_orc_writer_options_builder builder(
            cudf_io_types.sink_info &sink
        ) except +libcudf_exception_handler

    cdef cppclass chunked_orc_writer_options_builder:
        # setter
        chunked_orc_writer_options_builder& compression(
            cudf_io_types.compression_type comp
        ) except +libcudf_exception_handler
        chunked_orc_writer_options_builder& enable_statistics(
            cudf_io_types.statistics_freq val
        ) except +libcudf_exception_handler
        orc_writer_options_builder& stripe_size_bytes(
            size_t val
        ) except +libcudf_exception_handler
        orc_writer_options_builder& stripe_size_rows(
            size_type val
        ) except +libcudf_exception_handler
        orc_writer_options_builder& row_index_stride(
            size_type val
        ) except +libcudf_exception_handler
        chunked_orc_writer_options_builder& table(
            cudf_table_view.table_view tbl
        ) except +libcudf_exception_handler
        chunked_orc_writer_options_builder& metadata(
            cudf_io_types.table_input_metadata meta
        ) except +libcudf_exception_handler
        chunked_orc_writer_options_builder& key_value_metadata(
            map[string, string] kvm
        ) except +libcudf_exception_handler

        chunked_orc_writer_options build() except +libcudf_exception_handler

    cdef cppclass orc_chunked_writer:
        orc_chunked_writer() except +libcudf_exception_handler
        orc_chunked_writer(
            chunked_orc_writer_options args,
            cudaStream_t stream
        ) except +libcudf_exception_handler
        orc_chunked_writer& write(
            cudf_table_view.table_view table_,
        ) except +libcudf_exception_handler
        void close() except +libcudf_exception_handler
