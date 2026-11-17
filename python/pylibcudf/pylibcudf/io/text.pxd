# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from libcpp.memory cimport unique_ptr
from libcpp.string cimport string
from pylibcudf.column cimport Column
from rmm.pylibrmm.memory_resource cimport DeviceMemoryResource
from pylibcudf.libcudf.io.text cimport parse_options, data_chunk_source, byte_range_info


cdef class ByteRangeInfo:
    cdef byte_range_info c_obj

cdef class ParseOptions:
    cdef parse_options c_options

cdef class DataChunkSource:
    cdef unique_ptr[data_chunk_source] c_source
    cdef string data_ref

cpdef Column multibyte_split(
    DataChunkSource source,
    str delimiter,
    ParseOptions options=*,
    object stream = *,
    DeviceMemoryResource mr=*
)

cpdef DataChunkSource make_source(str data)

cpdef DataChunkSource make_source_from_file(str filename)

cpdef DataChunkSource make_source_from_bgzip_file(
    str filename,
    int virtual_begin=*,
    int virtual_end=*,
)
