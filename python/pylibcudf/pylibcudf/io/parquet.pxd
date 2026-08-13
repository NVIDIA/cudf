# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from libc.stdint cimport int64_t, uint8_t

from libcpp cimport bool
from libcpp.memory cimport unique_ptr
from libcpp.vector cimport vector

from rmm.pylibrmm.memory_resource cimport DeviceMemoryResource
from rmm.pylibrmm.stream cimport Stream

from pylibcudf.expressions cimport Expression
from pylibcudf.io.parquet_metadata cimport FileMetaData

from pylibcudf.io.types cimport (
    compression_type,
    dictionary_policy,
    statistics_freq,
    SinkInfo,
    SourceInfo,
    TableInputMetadata,
    TableWithMetadata,
)

from pylibcudf.libcudf.io.parquet cimport (
    chunked_parquet_writer as cpp_chunked_parquet_writer,
    chunked_parquet_reader as cpp_chunked_parquet_reader,
    parquet_writer_options,
    parquet_writer_options_builder,
    parquet_reader_options,
    parquet_reader_options_builder,
    chunked_parquet_writer_options,
    chunked_parquet_writer_options_builder,
)

from pylibcudf.libcudf.types cimport size_type, type_id

from pylibcudf.table cimport Table

from pylibcudf.types cimport DataType


cdef class ParquetReaderOptions:
    cdef parquet_reader_options c_obj
    cdef SourceInfo source
    cpdef void set_row_groups(self, list row_groups)
    cpdef void set_num_rows(self, int64_t nrows)
    cpdef void set_skip_rows(self, int64_t skip_rows)
    cpdef void set_columns(self, list col_names)
    cpdef void set_column_names(self, list col_names)
    cpdef void set_column_indices(self, list col_indices)
    cpdef void set_column_field_ids(self, list column_field_ids)
    cpdef void set_filter(self, Expression filter)
    cpdef void set_source(self, SourceInfo src)
    cpdef bool is_enabled_use_jit_filter(self)
    cpdef bool is_enabled_case_sensitive_names(self)
    cpdef void enable_case_sensitive_names(self, bool val)


cdef class ParquetReaderOptionsBuilder:
    cdef parquet_reader_options_builder c_obj
    cdef SourceInfo source
    cpdef ParquetReaderOptionsBuilder convert_strings_to_categories(self, bool val)
    cpdef ParquetReaderOptionsBuilder use_pandas_metadata(self, bool val)
    cpdef ParquetReaderOptionsBuilder allow_mismatched_pq_schemas(self, bool val)
    cpdef ParquetReaderOptionsBuilder ignore_missing_columns(self, bool val)
    cpdef ParquetReaderOptionsBuilder use_arrow_schema(self, bool val)
    cpdef ParquetReaderOptionsBuilder filter(self, Expression filter)
    cpdef ParquetReaderOptionsBuilder columns(self, list col_names)
    cpdef ParquetReaderOptionsBuilder column_names(self, list col_names)
    cpdef ParquetReaderOptionsBuilder column_indices(self, list col_indices)
    cpdef ParquetReaderOptionsBuilder column_field_ids(self, list column_field_ids)
    cpdef ParquetReaderOptionsBuilder use_jit_filter(self, bool use_jit_filter)
    cpdef ParquetReaderOptionsBuilder case_sensitive_names(self, bool val)
    cpdef ParquetReaderOptionsBuilder decimal_width(self, type_id width)
    cpdef build(self)


cdef class ChunkedParquetReader:
    cdef Stream _stream
    cdef DeviceMemoryResource mr
    cdef unique_ptr[cpp_chunked_parquet_reader] reader

    cpdef bool has_next(self)
    cpdef TableWithMetadata read_chunk(self, DeviceMemoryResource mr=*)


cpdef read_parquet(
    ParquetReaderOptions options,
    object stream = *,
    DeviceMemoryResource mr=*,
    object parquet_metadatas=*,
)


cdef class ChunkedParquetWriter:
    cdef unique_ptr[cpp_chunked_parquet_writer] c_obj
    cpdef memoryview close(self, list column_chunks_file_paths)
    cpdef void write(self, Table table, object partitions_info=*)


cdef class ChunkedParquetWriterOptions:
    cdef chunked_parquet_writer_options c_obj
    cdef SinkInfo sink

    cpdef void set_dictionary_policy(self, dictionary_policy policy)


cdef class ChunkedParquetWriterOptionsBuilder:
    cdef chunked_parquet_writer_options_builder c_obj
    cdef SinkInfo sink

    cpdef ChunkedParquetWriterOptionsBuilder metadata(self, TableInputMetadata metadata)

    cpdef ChunkedParquetWriterOptionsBuilder key_value_metadata(self, metadata)

    cpdef ChunkedParquetWriterOptionsBuilder compression(
        self,
        compression_type compression
    )

    cpdef ChunkedParquetWriterOptionsBuilder stats_level(self, statistics_freq sf)

    cpdef ChunkedParquetWriterOptionsBuilder row_group_size_bytes(self, size_t val)

    cpdef ChunkedParquetWriterOptionsBuilder row_group_size_rows(self, size_type val)

    cpdef ChunkedParquetWriterOptionsBuilder max_page_size_bytes(self, size_t val)

    cpdef ChunkedParquetWriterOptionsBuilder max_page_size_rows(self, size_type val)

    cpdef ChunkedParquetWriterOptionsBuilder max_dictionary_size(self, size_t val)

    cpdef ChunkedParquetWriterOptionsBuilder write_arrow_schema(self, bool enabled)

    cpdef ChunkedParquetWriterOptions build(self)


cdef class ParquetWriterOptions:
    cdef parquet_writer_options c_obj
    cdef Table table_ref
    cdef SinkInfo sink_ref

    cpdef void set_partitions(self, list partitions)

    cpdef void set_column_chunks_file_paths(self, file_paths)

    cpdef void set_row_group_size_bytes(self, size_t size_bytes)

    cpdef void set_row_group_size_rows(self, size_type size_rows)

    cpdef void set_max_page_size_bytes(self, size_t size_bytes)

    cpdef void set_max_page_size_rows(self, size_type size_rows)

    cpdef void set_max_dictionary_size(self, size_t size_bytes)

cdef class ParquetWriterOptionsBuilder:
    cdef parquet_writer_options_builder c_obj
    cdef Table table_ref
    cdef SinkInfo sink_ref

    cpdef ParquetWriterOptionsBuilder metadata(self, TableInputMetadata metadata)

    cpdef ParquetWriterOptionsBuilder key_value_metadata(self, metadata)

    cpdef ParquetWriterOptionsBuilder compression(self, compression_type compression)

    cpdef ParquetWriterOptionsBuilder stats_level(self, statistics_freq sf)

    cpdef ParquetWriterOptionsBuilder int96_timestamps(self, bool enabled)

    cpdef ParquetWriterOptionsBuilder write_v2_headers(self, bool enabled)

    cpdef ParquetWriterOptionsBuilder page_level_compression(self, bool enabled)

    cpdef ParquetWriterOptionsBuilder dictionary_policy(self, dictionary_policy val)

    cpdef ParquetWriterOptionsBuilder utc_timestamps(self, bool enabled)

    cpdef ParquetWriterOptionsBuilder write_arrow_schema(self, bool enabled)

    cpdef ParquetWriterOptionsBuilder row_group_size_rows(self, size_type val)

    cpdef ParquetWriterOptionsBuilder max_page_size_bytes(self, size_t val)

    cpdef ParquetWriterOptions build(self)

cpdef memoryview write_parquet(ParquetWriterOptions options, object stream = *)

cpdef bool is_supported_read_parquet(compression_type compression)

cpdef bool is_supported_write_parquet(compression_type compression)

cpdef memoryview merge_row_group_metadata(list metdata_list)
