# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from libc.stdint cimport int8_t, int16_t, int32_t, int64_t, uint8_t
from libcpp cimport bool
from libcpp.optional cimport optional
from libcpp.string cimport string
from libcpp.vector cimport vector
from pylibcudf.exception_handler cimport libcudf_exception_handler
from pylibcudf.libcudf.types cimport size_type, type_id


cdef extern from "<cuda/std/optional>" nogil:
    cdef cppclass cuda_optional "cuda::std::optional" [T]:
        bool has_value()
        T& value()


cdef extern from "cudf/io/parquet_schema.hpp" namespace "cudf::io::parquet" nogil:
    cpdef enum class LogicalTypeId "cudf::io::parquet::LogicalType::Type" (uint8_t):
        UNDEFINED
        STRING
        MAP
        LIST
        ENUM
        DECIMAL
        DATE
        TIME
        TIMESTAMP
        INTEGER
        UNKNOWN
        JSON
        BSON
        VARIANT

    cpdef enum class TimeUnit "cudf::io::parquet::TimeUnit::Type" (uint8_t):
        UNDEFINED
        MILLIS
        MICROS
        NANOS

    cdef cppclass DecimalType:
        int32_t scale
        int32_t precision

    cdef cppclass TimeUnitInfo "cudf::io::parquet::TimeUnit":
        TimeUnit type

    cdef cppclass TimeType:
        bool isAdjustedToUTC
        TimeUnitInfo unit

    cdef cppclass TimestampType:
        bool isAdjustedToUTC
        TimeUnitInfo unit

    cdef cppclass IntType:
        int8_t bitWidth
        bool isSigned

    cdef cppclass LogicalType:
        LogicalTypeId type
        cuda_optional[DecimalType] decimal_type
        cuda_optional[TimeType] time_type
        cuda_optional[TimestampType] timestamp_type
        cuda_optional[IntType] int_type

    cpdef enum class Type(int8_t):
        UNDEFINED
        BOOLEAN
        INT32
        INT64
        INT96
        FLOAT
        DOUBLE
        BYTE_ARRAY
        FIXED_LEN_BYTE_ARRAY

    cpdef enum class FieldRepetitionType(int8_t):
        UNSPECIFIED
        REQUIRED
        OPTIONAL
        REPEATED

    cdef cppclass Statistics:
        optional[vector[uint8_t]] max
        optional[vector[uint8_t]] min
        optional[int64_t] null_count
        optional[int64_t] distinct_count
        optional[vector[uint8_t]] max_value
        optional[vector[uint8_t]] min_value
        optional[bool] is_max_value_exact
        optional[bool] is_min_value_exact

    cdef cppclass SortingColumn:
        int32_t column_idx
        bint descending
        bint nulls_first

    cdef cppclass ColumnChunkMetaData:
        vector[string] path_in_schema
        int64_t num_values
        int64_t total_uncompressed_size
        int64_t total_compressed_size
        int64_t data_page_offset
        int64_t index_page_offset
        int64_t dictionary_page_offset
        Statistics statistics

    cdef cppclass ColumnChunk:
        string file_path
        int64_t file_offset
        ColumnChunkMetaData meta_data
        int64_t offset_index_offset
        int32_t offset_index_length
        int64_t column_index_offset
        int32_t column_index_length
        int schema_idx

    cdef cppclass RowGroup:
        vector[ColumnChunk] columns
        int64_t total_byte_size
        int64_t num_rows
        optional[vector[SortingColumn]] sorting_columns
        optional[int64_t] file_offset
        optional[int64_t] total_compressed_size
        optional[int16_t] ordinal

    cdef cppclass SchemaElement:
        Type type
        int32_t type_length
        string name
        int32_t num_children
        optional[int32_t] field_id
        cuda_optional[LogicalType] logical_type
        FieldRepetitionType repetition_type
        bool output_as_byte_array
        optional[type_id] arrow_type
        int max_definition_level
        int max_repetition_level
        size_type parent_idx
        vector[size_type] children_idx
        bool is_stub()
        bool is_one_level_list(const SchemaElement& parent)
        bool is_struct()

    cdef cppclass FileMetaData:
        FileMetaData() except +libcudf_exception_handler
        int32_t version
        vector[SchemaElement] schema
        int64_t num_rows
        vector[RowGroup] row_groups
        string created_by
