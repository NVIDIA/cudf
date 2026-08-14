# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from libc.stdint cimport int64_t
from libcpp.memory cimport unique_ptr
from libcpp.string cimport string
from libcpp.unordered_map cimport unordered_map
from libcpp.vector cimport vector
from pylibcudf.exception_handler cimport libcudf_exception_handler
from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.io.datasource cimport datasource
from pylibcudf.libcudf.io.parquet_schema cimport FileMetaData
from pylibcudf.libcudf.table.table cimport table
from pylibcudf.libcudf.types cimport data_type, size_type
from pylibcudf.libcudf.io.types cimport source_info
from pylibcudf.libcudf.utilities.span cimport host_span
from cuda.bindings.cyruntime cimport cudaStream_t
from rmm.librmm.memory_resource cimport device_async_resource_ref

ctypedef const unique_ptr[datasource] const_unique_ptr_datasource
ctypedef const string const_string


cdef extern from "cudf/io/parquet_metadata.hpp" namespace "cudf::io" nogil:
    cdef cppclass parquet_column_schema:
        parquet_column_schema() except +libcudf_exception_handler
        string name() except +libcudf_exception_handler
        size_type num_children() except +libcudf_exception_handler
        parquet_column_schema child(int idx) except +libcudf_exception_handler
        vector[parquet_column_schema] children() except +libcudf_exception_handler
        data_type cudf_type() noexcept

    cdef cppclass parquet_schema:
        parquet_schema() except +libcudf_exception_handler
        parquet_column_schema root() except +libcudf_exception_handler

    cdef cppclass parquet_metadata:
        parquet_metadata() except +libcudf_exception_handler
        parquet_schema schema() except +libcudf_exception_handler
        int64_t num_rows() except +libcudf_exception_handler
        size_type num_rowgroups() except +libcudf_exception_handler
        vector[size_type] num_rowgroups_per_file() except +libcudf_exception_handler
        unordered_map[string, string] metadata() except +libcudf_exception_handler
        vector[unordered_map[string, int64_t]] rowgroup_metadata()\
            except +libcudf_exception_handler
        unordered_map[string, vector[int64_t]] \
            columnchunk_metadata() except +libcudf_exception_handler

    cdef cppclass column_chunk_bounds_result:
        unique_ptr[column] file_indices
        unique_ptr[column] row_group_indices
        vector[unique_ptr[table]] bounds

    cdef parquet_metadata read_parquet_metadata(
        source_info src_info
    ) except +libcudf_exception_handler

    cdef vector[FileMetaData] read_parquet_footers(
        host_span[const_unique_ptr_datasource] sources
    ) except +libcudf_exception_handler

    cdef column_chunk_bounds_result column_chunk_bounds(
        vector[FileMetaData] parquet_metadatas,
        host_span[const_string] column_names,
        cudaStream_t stream,
        device_async_resource_ref mr,
    ) except +libcudf_exception_handler
