# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from libc.stdint cimport int16_t, int32_t, int64_t
from libcpp.optional cimport optional
from libcpp.string cimport string
from libcpp.vector cimport vector
from pylibcudf.exception_handler cimport libcudf_exception_handler


cdef extern from "cudf/io/parquet_schema.hpp" namespace "cudf::io::parquet" nogil:
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

    cdef cppclass FileMetaData:
        FileMetaData() except +libcudf_exception_handler
        int32_t version
        int64_t num_rows
        vector[RowGroup] row_groups
        string created_by
