# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
from libc.stdint cimport uint64_t, int64_t

from libcpp cimport bool
from libcpp.map cimport map
from libcpp.memory cimport unique_ptr
from libcpp.optional cimport optional
from libcpp.string cimport string
from libcpp.vector cimport vector

from rmm.pylibrmm.memory_resource cimport DeviceMemoryResource

from pylibcudf.io.types cimport (
    SourceInfo,
    SinkInfo,
    TableWithMetadata,
    TableInputMetadata,
)

from pylibcudf.libcudf.io.orc_metadata cimport (
    column_statistics,
    parsed_orc_statistics,
    statistics_type,
)

from pylibcudf.libcudf.io.orc cimport (
    orc_chunked_writer,
    orc_reader_options,
    orc_reader_options_builder,
    orc_writer_options,
    orc_writer_options_builder,
    chunked_orc_writer_options,
    chunked_orc_writer_options_builder,
)

from pylibcudf.libcudf.io.types cimport (
    compression_type,
    statistics_freq,
)

from pylibcudf.libcudf.types cimport size_type

from pylibcudf.table cimport Table

from pylibcudf.types cimport DataType


cdef class OrcReaderOptions:
    cdef orc_reader_options c_obj
    cdef SourceInfo source
    cpdef void set_num_rows(self, int64_t nrows)
    cpdef void set_skip_rows(self, int64_t skip_rows)
    cpdef void set_stripes(self, list stripes)
    cpdef void set_decimal128_columns(self, list val)
    cpdef void set_timestamp_type(self, DataType type_)
    cpdef void set_columns(self, list col_names)
    cpdef void set_source(self, SourceInfo src)

cdef class OrcReaderOptionsBuilder:
    cdef orc_reader_options_builder c_obj
    cdef SourceInfo source
    cpdef OrcReaderOptionsBuilder use_index(self, bool use)
    cpdef OrcReaderOptions build(self)

cpdef TableWithMetadata read_orc(
    OrcReaderOptions options, object stream = *, DeviceMemoryResource mr=*
)

cdef class OrcColumnStatistics:
    cdef optional[uint64_t] number_of_values_c
    cdef optional[bool] has_null_c
    cdef statistics_type type_specific_stats_c
    cdef dict column_stats

    cdef void _init_stats_dict(self)

    @staticmethod
    cdef OrcColumnStatistics from_libcudf(column_statistics& col_stats)


cdef class ParsedOrcStatistics:
    cdef parsed_orc_statistics c_obj

    @staticmethod
    cdef ParsedOrcStatistics from_libcudf(parsed_orc_statistics& orc_stats)


cpdef ParsedOrcStatistics read_parsed_orc_statistics(
    SourceInfo source_info,
    object stream = *
)

cdef class OrcWriterOptions:
    cdef orc_writer_options c_obj
    cdef Table table
    cdef SinkInfo sink
    cpdef void set_stripe_size_bytes(self, size_t size_bytes)
    cpdef void set_stripe_size_rows(self, size_type size_rows)
    cpdef void set_row_index_stride(self, size_type stride)

cdef class OrcWriterOptionsBuilder:
    cdef orc_writer_options_builder c_obj
    cdef Table table
    cdef SinkInfo sink
    cpdef OrcWriterOptionsBuilder compression(self, compression_type comp)
    cpdef OrcWriterOptionsBuilder enable_statistics(self, statistics_freq val)
    cpdef OrcWriterOptionsBuilder key_value_metadata(self, dict kvm)
    cpdef OrcWriterOptionsBuilder metadata(self, TableInputMetadata meta)
    cpdef OrcWriterOptions build(self)

cpdef void write_orc(OrcWriterOptions options, object stream = *)

cdef class OrcChunkedWriter:
    cdef unique_ptr[orc_chunked_writer] c_obj
    cpdef void close(self)
    cpdef void write(self, Table table)

cdef class ChunkedOrcWriterOptions:
    cdef chunked_orc_writer_options c_obj
    cdef SinkInfo sink
    cpdef void set_stripe_size_bytes(self, size_t size_bytes)
    cpdef void set_stripe_size_rows(self, size_type size_rows)
    cpdef void set_row_index_stride(self, size_type stride)

cdef class ChunkedOrcWriterOptionsBuilder:
    cdef chunked_orc_writer_options_builder c_obj
    cdef SinkInfo sink
    cpdef ChunkedOrcWriterOptionsBuilder compression(self, compression_type comp)
    cpdef ChunkedOrcWriterOptionsBuilder enable_statistics(self, statistics_freq val)
    cpdef ChunkedOrcWriterOptionsBuilder key_value_metadata(
        self, dict kvm
    )
    cpdef ChunkedOrcWriterOptionsBuilder metadata(self, TableInputMetadata meta)
    cpdef ChunkedOrcWriterOptions build(self)
