# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from libc.stddef cimport size_t
from libc.stdint cimport uint64_t
from libcpp.memory cimport shared_ptr, unique_ptr

from rapidsmpf._detail.exception_handling cimport ex_handler
from rapidsmpf.communicator.communicator cimport Communicator, cpp_Communicator
from rapidsmpf.streaming.core.context cimport cpp_Context


cdef extern from "<cudf_streaming/bloom_filter.hpp>" nogil:
    cdef cppclass cpp_BloomFilter "cudf_streaming::bloom_filter":
        cpp_BloomFilter(
            shared_ptr[cpp_Context] ctx,
            shared_ptr[cpp_Communicator] comm,
            uint64_t seed,
            size_t filter_size,
        ) except +ex_handler
        const shared_ptr[cpp_Communicator]& comm() noexcept


cdef extern from "<cudf_streaming/bloom_filter.hpp>" nogil:
    size_t cpp_aligned_size \
        "cudf_streaming::bloom_filter::aligned_size"(
            size_t size
        ) noexcept


cdef class BloomFilter:
    cdef unique_ptr[cpp_BloomFilter] _handle
    cdef Communicator _comm
