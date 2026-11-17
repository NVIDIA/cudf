# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
from rmm.pylibrmm.memory_resource cimport DeviceMemoryResource

from pylibcudf.io.types cimport SourceInfo, TableWithMetadata

from pylibcudf.libcudf.io.avro cimport (
    avro_reader_options,
    avro_reader_options_builder,
)

from pylibcudf.libcudf.types cimport size_type


cdef class AvroReaderOptions:
    cdef avro_reader_options c_obj
    cdef SourceInfo source
    cpdef void set_columns(self, list col_names)
    cpdef void set_source(self, SourceInfo src)


cdef class AvroReaderOptionsBuilder:
    cdef avro_reader_options_builder c_obj
    cdef SourceInfo source
    cpdef AvroReaderOptionsBuilder columns(self, list col_names)
    cpdef AvroReaderOptionsBuilder skip_rows(self, size_type skip_rows)
    cpdef AvroReaderOptionsBuilder num_rows(self, size_type num_rows)
    cpdef AvroReaderOptions build(self)

cpdef TableWithMetadata read_avro(
    AvroReaderOptions options, object stream = *, DeviceMemoryResource mr=*
)
