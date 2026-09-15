# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from libc.stdint cimport uint8_t, uintptr_t
from libc.stddef cimport size_t
from libcpp cimport bool
from libcpp.memory cimport unique_ptr
from libcpp.vector cimport vector
from pylibcudf.libcudf.contiguous_split cimport packed_columns, chunked_pack
from pylibcudf.libcudf.types cimport bitmask_type
from pylibcudf.libcudf.utilities.span cimport device_span
from rmm.pylibrmm.device_buffer cimport DeviceBuffer
from rmm.pylibrmm.memory_resource cimport DeviceMemoryResource
from rmm.pylibrmm.stream cimport Stream

from .gpumemoryview cimport gpumemoryview
from .table cimport Table


cdef device_span[uint8_t] _get_device_span(object obj) except *

cdef class HostBuffer:
    cdef unique_ptr[vector[uint8_t]] c_obj
    cdef size_t nbytes
    cdef Py_ssize_t[1] shape
    cdef Py_ssize_t[1] strides

    @staticmethod
    cdef HostBuffer from_unique_ptr(
        unique_ptr[vector[uint8_t]] vec
    )

cdef class PackedColumns:
    cdef unique_ptr[packed_columns] c_obj
    cdef object stream
    cdef DeviceMemoryResource mr

    @staticmethod
    cdef PackedColumns from_libcudf(
        unique_ptr[packed_columns] data,
        object stream,
        DeviceMemoryResource mr
    )
    cpdef tuple[memoryview, gpumemoryview] release(self)

cdef class ChunkedPack:
    cdef unique_ptr[chunked_pack] c_obj
    cdef Table table
    cdef DeviceMemoryResource mr
    cdef Stream stream

    cpdef bool has_next(self)
    cpdef size_t next(self, object buf)
    cpdef size_t get_total_contiguous_size(self)
    cpdef memoryview build_metadata(self)
    cpdef tuple[memoryview, memoryview] pack_to_host(self, object buf)


cpdef PackedColumns pack(Table input)

cpdef Table unpack(PackedColumns input, object stream = *)

cpdef Table unpack_from_memoryviews(
    memoryview metadata,
    object gpu_data,
    object stream = *,
)
