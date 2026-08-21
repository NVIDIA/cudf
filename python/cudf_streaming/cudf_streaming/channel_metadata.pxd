# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from libc.stdint cimport int32_t, uint64_t
from libcpp cimport bool as bool_t
from libcpp.memory cimport shared_ptr, unique_ptr
from libcpp.optional cimport optional
from libcpp.utility cimport pair
from libcpp.vector cimport vector
from pylibcudf.libcudf.types cimport null_order as cpp_null_order
from pylibcudf.libcudf.types cimport order as cpp_order
from rmm.librmm.cuda_stream_view cimport cuda_stream_view

from rapidsmpf._detail.exception_handling cimport ex_handler

from rapidsmpf.memory.buffer_resource cimport cpp_BufferResource
from rapidsmpf.streaming.core.message cimport cpp_Message
from cudf_streaming.table_chunk cimport TableChunk, cpp_TableChunk


cdef extern from "<cudf_streaming/channel_metadata.hpp>" \
        namespace "cudf_streaming" nogil:

    cdef cppclass cpp_HashScheme "cudf_streaming::hash_scheme":
        vector[int32_t] column_indices
        int modulus
        cpp_HashScheme() except +ex_handler
        cpp_HashScheme(vector[int32_t], int) except +ex_handler
        bool_t operator==(const cpp_HashScheme&)

    cdef cppclass cpp_OrderKey "cudf_streaming::order_key":
        cpp_OrderKey() noexcept
        cpp_OrderKey(int32_t, cpp_order, cpp_null_order) noexcept
        int32_t column_index
        cpp_order order
        cpp_null_order null_order
        bool_t operator==(const cpp_OrderKey&) noexcept

    cdef cppclass cpp_Ordering "cudf_streaming::ordering":
        cpp_Ordering() noexcept
        cpp_Ordering(
            vector[cpp_OrderKey], unique_ptr[cpp_TableChunk], bool_t
        ) except +ex_handler
        vector[cpp_OrderKey] keys
        shared_ptr[cpp_TableChunk] boundaries
        bool_t strict_boundaries
        cpp_Ordering with_keys(vector[cpp_OrderKey]) except +ex_handler
        bool_t boundaries_aligned_with(
            const cpp_Ordering&, const cpp_BufferResource&
        ) except +ex_handler

    cdef cppclass cpp_OrderScheme "cudf_streaming::order_scheme":
        cpp_OrderScheme() noexcept
        cpp_OrderScheme(
            vector[cpp_OrderKey], unique_ptr[cpp_TableChunk], bool_t
        ) except +ex_handler
        cpp_OrderScheme(vector[cpp_Ordering]) except +ex_handler
        vector[cpp_Ordering] orderings

    cdef cppclass cpp_PartitioningSpec "cudf_streaming::partitioning_spec":
        enum cpp_Type "cudf_streaming::partitioning_spec::type":
            NONE "cudf_streaming::partitioning_spec::type::NONE"
            INHERIT "cudf_streaming::partitioning_spec::type::INHERIT"
            HASH "cudf_streaming::partitioning_spec::type::HASH"
            ORDER "cudf_streaming::partitioning_spec::type::ORDER"

        cpp_Type type
        optional[cpp_HashScheme] hash
        optional[cpp_OrderScheme] order

        @staticmethod
        cpp_PartitioningSpec none()

        @staticmethod
        cpp_PartitioningSpec inherit()

        @staticmethod
        cpp_PartitioningSpec from_hash(cpp_HashScheme)

        @staticmethod
        cpp_PartitioningSpec from_order(cpp_OrderScheme)

    cdef cppclass cpp_Partitioning "cudf_streaming::partitioning":
        cpp_PartitioningSpec inter_rank
        cpp_PartitioningSpec local
        cpp_Partitioning() except +ex_handler
        cpp_Partitioning(const cpp_Partitioning&) except +ex_handler

    cdef cppclass cpp_ChannelMetadata "cudf_streaming::channel_metadata":
        uint64_t local_count
        cpp_Partitioning partitioning
        bool_t duplicated
        cpp_ChannelMetadata(
            uint64_t,
            cpp_Partitioning,
            bool_t
        ) except +ex_handler

    cpp_Message cpp_to_message_channel_metadata \
        "cudf_streaming::to_message"(
            uint64_t, unique_ptr[cpp_ChannelMetadata]
        ) except +ex_handler


cdef class HashScheme:
    cdef cpp_HashScheme _handle

    @staticmethod
    cdef HashScheme from_cpp(cpp_HashScheme scheme)


cdef class OrderKey:
    cdef cpp_OrderKey _handle

    @staticmethod
    cdef OrderKey from_cpp(cpp_OrderKey key)


cdef class Ordering:
    cdef cpp_Ordering _handle

    @staticmethod
    cdef Ordering from_cpp(cpp_Ordering ordering)


cdef class OrderScheme:
    cdef cpp_OrderScheme _handle

    @staticmethod
    cdef OrderScheme from_cpp(cpp_OrderScheme scheme)


cdef class Partitioning:
    cdef cpp_Partitioning _handle

    @staticmethod
    cdef Partitioning from_cpp(cpp_Partitioning data)


cdef class ChannelMetadata:
    cdef unique_ptr[cpp_ChannelMetadata] _handle

    @staticmethod
    cdef ChannelMetadata from_handle(unique_ptr[cpp_ChannelMetadata] handle)

    cdef const cpp_ChannelMetadata* handle_ptr(self) except NULL

    cdef unique_ptr[cpp_ChannelMetadata] release_handle(self)
