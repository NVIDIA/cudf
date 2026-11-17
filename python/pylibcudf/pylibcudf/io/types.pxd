# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from libc.stdint cimport uint8_t, int32_t

from libcpp cimport bool
from libcpp.memory cimport unique_ptr
from libcpp.vector cimport vector

from pylibcudf.libcudf.io.data_sink cimport data_sink
from pylibcudf.libcudf.io.types cimport (
    const_byte,
    column_encoding,
    column_in_metadata,
    column_name_info,
    compression_type,
    dictionary_policy,
    filepath_source,
    io_type,
    partition_info,
    quote_style,
    sink_info,
    source_info,
    statistics_freq,
    table_input_metadata,
    table_metadata,
    table_with_metadata,
)
from pylibcudf.libcudf.types cimport size_type
from pylibcudf.libcudf.utilities.span cimport host_span

from pylibcudf.table cimport Table

from rmm.pylibrmm.memory_resource cimport DeviceMemoryResource

cdef class PartitionInfo:
    cdef partition_info c_obj

cdef class ColumnInMetadata:
    cdef column_in_metadata* c_obj
    cdef TableInputMetadata owner

    cdef TableInputMetadata table

    cpdef ColumnInMetadata set_name(self, str name)

    cpdef ColumnInMetadata set_name(self, str name)

    cpdef ColumnInMetadata set_nullability(self, bool nullable)

    cpdef ColumnInMetadata set_list_column_as_map(self)

    cpdef ColumnInMetadata set_int96_timestamps(self, bool req)

    cpdef ColumnInMetadata set_decimal_precision(self, uint8_t precision)

    cpdef ColumnInMetadata child(self, size_type i)

    cpdef ColumnInMetadata set_output_as_binary(self, bool binary)

    cpdef ColumnInMetadata set_type_length(self, int32_t type_length)

    cpdef ColumnInMetadata set_skip_compression(self, bool skip)

    cpdef ColumnInMetadata set_encoding(self, column_encoding encoding)

    cpdef str get_name(self)

    @staticmethod
    cdef ColumnInMetadata from_libcudf(
        column_in_metadata* metadata, TableInputMetadata owner
    )

cdef class TableInputMetadata:
    cdef table_input_metadata c_obj

cdef class TableWithMetadata:
    cdef public Table tbl
    cdef table_metadata metadata

    cdef vector[column_name_info] _make_column_info(self, list column_names)

    cdef list _make_columns_list(self, dict child_dict)

    @staticmethod
    cdef dict _parse_col_names(vector[column_name_info] infos)

    @staticmethod
    cdef TableWithMetadata from_libcudf(
        table_with_metadata& tbl, object stream, DeviceMemoryResource mr
    )

cdef class FilepathSource:
    cdef public object path
    cdef public object size

cdef class SourceInfo:
    cdef source_info c_obj
    # Keep the bytes converted from stringio alive
    # (otherwise we end up with a use after free when they get gc'ed)
    cdef list byte_sources
    cdef vector[host_span[const_byte]] _hspans
    cdef list device_sources

cdef class SinkInfo:
    # This vector just exists to keep the unique_ptrs to the sinks alive
    cdef vector[unique_ptr[data_sink]] sink_storage
    cdef sink_info c_obj
