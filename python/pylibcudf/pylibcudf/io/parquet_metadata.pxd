# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from pylibcudf.io.types cimport SourceInfo
from pylibcudf.libcudf.io.parquet_schema cimport (
    ColumnChunk as cpp_ColumnChunk,
    ColumnChunkMetaData as cpp_ColumnChunkMetaData,
    FileMetaData as cpp_FileMetaData,
    RowGroup as cpp_RowGroup,
    SortingColumn as cpp_SortingColumn,
)
from pylibcudf.libcudf.io.parquet_metadata cimport(
    parquet_metadata,
    parquet_schema,
    parquet_column_schema,
)
from pylibcudf.types cimport DataType

cdef class ParquetColumnSchema:
    cdef parquet_column_schema column_schema

    @staticmethod
    cdef from_column_schema(parquet_column_schema column_schema)

    cpdef str name(self)

    cpdef int num_children(self)

    cpdef ParquetColumnSchema child(self, int idx)

    cpdef list children(self)

    cpdef DataType cudf_type(self)


cdef class ParquetSchema:
    cdef parquet_schema schema

    @staticmethod
    cdef from_schema(parquet_schema schema)

    cpdef ParquetColumnSchema root(self)

    cpdef dict column_types(self)


cdef class ParquetMetadata:
    cdef parquet_metadata meta

    @staticmethod
    cdef from_metadata(parquet_metadata meta)

    cpdef ParquetSchema schema(self)

    cpdef int num_rows(self)

    cpdef int num_rowgroups(self)

    cpdef list num_rowgroups_per_file(self)

    cpdef dict metadata(self)

    cpdef list rowgroup_metadata(self)

    cpdef dict columnchunk_metadata(self)

cdef class FileMetaData:
    cdef cpp_FileMetaData c_obj

    @staticmethod
    cdef FileMetaData from_cpp(cpp_FileMetaData metadata)

cdef class SortingColumn:
    cdef cpp_SortingColumn c_obj

    @staticmethod
    cdef SortingColumn from_cpp(cpp_SortingColumn sorting_column)

cdef class ColumnChunk:
    cdef cpp_ColumnChunk c_obj

    @staticmethod
    cdef ColumnChunk from_cpp(cpp_ColumnChunk column_chunk)

cdef class ColumnChunkMetaData:
    cdef cpp_ColumnChunkMetaData c_obj

    @staticmethod
    cdef ColumnChunkMetaData from_cpp(cpp_ColumnChunkMetaData meta_data)

cdef class RowGroup:
    cdef cpp_RowGroup c_obj

    @staticmethod
    cdef RowGroup from_cpp(cpp_RowGroup row_group)

cpdef ParquetMetadata read_parquet_metadata(SourceInfo src_info)
cpdef list read_parquet_footers(SourceInfo src_info)
