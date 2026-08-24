# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from libc.stddef cimport size_t
from libc.stdint cimport uint64_t
from libcpp cimport bool as bool_t
from libcpp.memory cimport unique_ptr
from libcpp.utility cimport pair
from pylibcudf.libcudf.table.table_view cimport table_view as cpp_table_view
from pylibcudf.libcudf.types cimport size_type
from rmm.librmm.cuda_stream_view cimport cuda_stream_view
from rmm.pylibrmm.stream cimport Stream

from rapidsmpf._detail.exception_handling cimport ex_handler
from rapidsmpf.memory.buffer cimport MemoryType
from rapidsmpf.memory.buffer_resource cimport (BufferResource,
                                               cpp_BufferResource)
from rapidsmpf.memory.memory_reservation cimport cpp_MemoryReservation
from rapidsmpf.memory.packed_data cimport cpp_PackedData


cdef extern from "<cudf_streaming/table_chunk.hpp>" nogil:
    cdef cppclass cpp_TableChunk "cudf_streaming::table_chunk":
        cpp_TableChunk(unique_ptr[cpp_PackedData]) except +ex_handler
        cuda_stream_view stream() noexcept
        size_t data_alloc_size(MemoryType mem_type) except +ex_handler
        bool_t is_available() noexcept
        size_t make_available_cost() noexcept
        cpp_table_view table_view() except +ex_handler
        bool_t is_spillable() noexcept
        cpp_TableChunk copy(cpp_MemoryReservation& reservation) except +ex_handler
        pair[size_type, size_type] shape() noexcept
        unique_ptr[cpp_PackedData] into_packed_data(
            cpp_BufferResource* br
        ) except +ex_handler

cdef class TableChunk:
    cdef unique_ptr[cpp_TableChunk] _handle
    # Keep the BufferResource alive as long as this object is so that when this
    # object is deallocated the associated stream and memory resource are still alive.
    cdef BufferResource _br

    @staticmethod
    cdef TableChunk from_handle(unique_ptr[cpp_TableChunk] handle, BufferResource br)
    cdef const cpp_TableChunk* handle_ptr(self)
    cdef unique_ptr[cpp_TableChunk] release_handle(self)
