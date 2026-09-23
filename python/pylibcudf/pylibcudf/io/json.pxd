# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from libcpp cimport bool
from libcpp.map cimport map
from libcpp.vector cimport vector

from rmm.pylibrmm.memory_resource cimport DeviceMemoryResource

from pylibcudf.io.types cimport (
    SinkInfo,
    SourceInfo,
    TableWithMetadata,
    compression_type,
)
from pylibcudf.column cimport Column
from pylibcudf.scalar cimport Scalar

from pylibcudf.libcudf.io.json cimport (
    json_recovery_mode_t,
    json_reader_options,
    json_reader_options_builder,
    json_writer_options,
    json_writer_options_builder,
)

from pylibcudf.libcudf.types cimport size_type

from pylibcudf.table cimport Table

from pylibcudf.types cimport DataType


cdef class JsonReaderOptions:
    cdef json_reader_options c_obj
    cdef SourceInfo source
    cpdef void set_dtypes(self, list types)
    cpdef void enable_keep_quotes(self, bool keep_quotes)
    cpdef void enable_mixed_types_as_string(self, bool mixed_types_as_string)
    cpdef void enable_prune_columns(self, bool prune_columns)
    cpdef void set_byte_range_offset(self, size_t offset)
    cpdef void set_byte_range_size(self, size_t size)
    cpdef void enable_lines(self, bool val)
    cpdef void set_source(self, SourceInfo src)
    # These hidden options are subjected to change without deprecation cycle.
    # These are used to test libcudf JSON reader features, not used in cuDF.
    cpdef void set_delimiter(self, str val)
    cpdef void enable_dayfirst(self, bool val)
    cpdef void enable_experimental(self, bool val)
    cpdef void enable_normalize_single_quotes(self, bool val)
    cpdef void enable_normalize_whitespace(self, bool val)
    cpdef void set_strict_validation(self, bool val)
    cpdef void allow_unquoted_control_chars(self, bool val)
    cpdef void allow_numeric_leading_zeros(self, bool val)
    cpdef void allow_nonnumeric_numbers(self, bool val)
    cpdef void set_na_values(self, list vals)


cdef class JsonReaderOptionsBuilder:
    cdef json_reader_options_builder c_obj
    cdef SourceInfo source
    cpdef JsonReaderOptionsBuilder byte_range_offset(self, size_t byte_range_offset)
    cpdef JsonReaderOptionsBuilder byte_range_size(self, size_t byte_range_size)
    cpdef JsonReaderOptionsBuilder compression(self, compression_type compression)
    cpdef JsonReaderOptionsBuilder dayfirst(self, bool val)
    cpdef JsonReaderOptionsBuilder delimiter(self, str delimiter)
    cpdef JsonReaderOptionsBuilder dtypes(self, list types)
    cpdef JsonReaderOptionsBuilder experimental(self, bool val)
    cpdef JsonReaderOptionsBuilder keep_quotes(self, bool val)
    cpdef JsonReaderOptionsBuilder lines(self, bool val)
    cpdef JsonReaderOptionsBuilder mixed_types_as_string(self, bool val)
    cpdef JsonReaderOptionsBuilder na_values(self, list vals)
    cpdef JsonReaderOptionsBuilder nonnumeric_numbers(self, bool val)
    cpdef JsonReaderOptionsBuilder normalize_single_quotes(self, bool val)
    cpdef JsonReaderOptionsBuilder normalize_whitespace(self, bool val)
    cpdef JsonReaderOptionsBuilder numeric_leading_zeros(self, bool val)
    cpdef JsonReaderOptionsBuilder prune_columns(self, bool val)
    cpdef JsonReaderOptionsBuilder recovery_mode(
        self, json_recovery_mode_t recovery_mode
    )
    cpdef JsonReaderOptionsBuilder strict_validation(self, bool val)
    cpdef JsonReaderOptionsBuilder unquoted_control_chars(self, bool val)
    cpdef JsonReaderOptions build(self)

cpdef TableWithMetadata read_json(
    JsonReaderOptions options, object stream = *, DeviceMemoryResource mr = *
)

cpdef TableWithMetadata read_json_from_string_column(
    Column input,
    Scalar separator,
    Scalar narep,
    list dtypes = *,
    compression_type compression = *,
    json_recovery_mode_t recovery_mode = *,
    object stream = *,
    DeviceMemoryResource mr = *)

cdef class JsonWriterOptions:
    cdef json_writer_options c_obj
    cdef SinkInfo sink
    cdef Table table
    cpdef void set_rows_per_chunk(self, size_type val)
    cpdef void set_true_value(self, str val)
    cpdef void set_false_value(self, str val)
    cpdef void set_compression(self, compression_type comptype)

cdef class JsonWriterOptionsBuilder:
    cdef json_writer_options_builder c_obj
    cdef SinkInfo sink
    cdef Table table
    cpdef JsonWriterOptionsBuilder metadata(self, TableWithMetadata tbl_w_meta)
    cpdef JsonWriterOptionsBuilder na_rep(self, str val)
    cpdef JsonWriterOptionsBuilder include_nulls(self, bool val)
    cpdef JsonWriterOptionsBuilder lines(self, bool val)
    cpdef JsonWriterOptionsBuilder compression(self, compression_type comptype)
    cpdef JsonWriterOptionsBuilder utf8_escaped(self, bool val)
    cpdef JsonWriterOptions build(self)

cpdef void write_json(JsonWriterOptions options, object stream = *)

cpdef bool is_supported_write_json(DataType type)

cpdef tuple[list[Column], list[str], dict] chunked_read_json(
    JsonReaderOptions options,
    int chunk_size= *,
    object stream = *,
    DeviceMemoryResource mr = *,
)
