# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from libc.stddef cimport size_t
from libc.stdint cimport int32_t
from libcpp.memory cimport shared_ptr, unique_ptr

from rapidsmpf._detail.exception_handling cimport ex_handler
from rapidsmpf.communicator.communicator cimport Communicator, cpp_Communicator
from rapidsmpf.streaming.core.context cimport cpp_Context


cdef extern from "<cudf_streaming/approx_distinct_count.hpp>" \
        namespace "cudf_streaming" nogil:
    cdef cppclass cpp_CardinalityEstimate \
            "cudf_streaming::cardinality_estimate":
        size_t row_count
        size_t distinct_count

    cdef cppclass cpp_CardinalityEstimator \
            "cudf_streaming::cardinality_estimator":
        cpp_CardinalityEstimator(
            shared_ptr[cpp_Context] ctx,
            shared_ptr[cpp_Communicator] comm,
            int32_t tag,
            int32_t precision,
        ) except +ex_handler
        const shared_ptr[cpp_Communicator]& comm() noexcept
        int32_t tag() noexcept
        int32_t precision() noexcept


cdef class CardinalityEstimate:
    cdef unique_ptr[cpp_CardinalityEstimate] _handle

    @staticmethod
    cdef CardinalityEstimate from_handle(
        unique_ptr[cpp_CardinalityEstimate] handle
    )


cdef class CardinalityEstimator:
    cdef unique_ptr[cpp_CardinalityEstimator] _handle
    cdef Communicator _comm
