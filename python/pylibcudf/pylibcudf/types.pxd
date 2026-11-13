# SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from libc.stddef cimport size_t
from libc.stdint cimport int32_t
from libcpp cimport bool as cbool
from pylibcudf.libcudf.types cimport (
    data_type,
    interpolation,
    mask_state,
    nan_equality,
    nan_policy,
    nan_semantics,
    null_equality,
    null_order,
    null_policy,
    null_aware,
    output_nullability,
    order,
    size_type,
    sorted,
    type_id,
)


cdef class DataType:
    cdef data_type c_obj

    cpdef type_id id(self)
    cpdef int32_t scale(self)

    @staticmethod
    cdef DataType from_libcudf(data_type dt)

cpdef size_t size_of(DataType t)
