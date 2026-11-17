# SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from libc.stddef cimport size_t
from libc.stdint cimport int32_t, uint32_t, int8_t, uint8_t
from libcpp cimport bool
from pylibcudf.exception_handler cimport libcudf_exception_handler


cdef extern from "cudf/types.hpp" namespace "cudf" nogil:
    ctypedef int32_t size_type
    ctypedef uint32_t bitmask_type
    ctypedef uint32_t char_utf8

    # A Hack to let cython compile with __int128_t symbol
    # https://stackoverflow.com/questions/27582001/how-to-use-128-bit-integers-in-cython/27609033#27609033
    ctypedef int int128 "__int128_t"

    cpdef enum class mask_state(int32_t):
        UNALLOCATED
        UNINITIALIZED
        ALL_VALID
        ALL_NULL

    cpdef enum class order(bool):
        ASCENDING
        DESCENDING

    cpdef enum class null_order(bool):
        AFTER
        BEFORE

    cpdef enum class sorted(bool):
        NO
        YES

    cdef cppclass order_info:
        sorted is_sorted
        order ordering
        null_order null_ordering

    cpdef enum class null_policy(bool):
        EXCLUDE
        INCLUDE

    cpdef enum class nan_policy(bool):
        NAN_IS_NULL
        NAN_IS_VALID

    cpdef enum class null_equality(bool):
        EQUAL
        UNEQUAL

    cpdef enum class null_aware(bool):
        NO
        YES

    cpdef enum class udf_source_type(uint8_t):
        CUDA
        PTX

    cpdef enum class output_nullability(int8_t):
        PRESERVE
        ALL_VALID

    cpdef enum class nan_equality(bool):
        ALL_EQUAL
        UNEQUAL

    cpdef enum class type_id(int32_t):
        EMPTY
        INT8
        INT16
        INT32
        INT64
        UINT8
        UINT16
        UINT32
        UINT64
        FLOAT32
        FLOAT64
        BOOL8
        TIMESTAMP_DAYS
        TIMESTAMP_SECONDS
        TIMESTAMP_MILLISECONDS
        TIMESTAMP_MICROSECONDS
        TIMESTAMP_NANOSECONDS
        DURATION_DAYS
        DURATION_SECONDS
        DURATION_MILLISECONDS
        DURATION_MICROSECONDS
        DURATION_NANOSECONDS
        DICTIONARY32
        STRING
        LIST
        DECIMAL32
        DECIMAL64
        DECIMAL128
        STRUCT
        NUM_TYPE_IDS

    cdef cppclass data_type:
        data_type() except +libcudf_exception_handler
        data_type(const data_type&) except +libcudf_exception_handler
        data_type(type_id id) except +libcudf_exception_handler
        data_type(type_id id, int32_t scale) except +libcudf_exception_handler
        type_id id() noexcept
        int32_t scale() noexcept
        bool operator==(const data_type&, const data_type&) noexcept

    cpdef enum class interpolation(int32_t):
        LINEAR
        LOWER
        HIGHER
        MIDPOINT
        NEAREST
        NEAREST_HALF_UP

    cdef size_t size_of(data_type t) except +libcudf_exception_handler
