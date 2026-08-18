# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from libc.stdint cimport int32_t
from libcpp cimport bool
from libcpp.memory cimport unique_ptr
from pylibcudf.libcudf.reduce cimport scan_type
from pylibcudf.libcudf.reduce cimport (
    approx_distinct_count as cpp_approx_distinct_count,
)
from pylibcudf.libcudf.types cimport nan_policy, null_equality, null_policy, size_type
from rmm.pylibrmm.memory_resource cimport DeviceMemoryResource

from .aggregation cimport Aggregation
from .column cimport Column
from .scalar cimport Scalar
from .table cimport Table
from .types cimport DataType


cpdef Scalar reduce(
    Column col,
    Aggregation agg,
    DataType data_type,
    Scalar init = *,
    object stream = *,
    DeviceMemoryResource mr = *,
)

cpdef Column scan(
    Column col,
    Aggregation agg,
    scan_type inclusive,
    object stream = *,
    DeviceMemoryResource mr = *,
)

cpdef tuple minmax(Column col, object stream = *, DeviceMemoryResource mr = *)

cpdef bool is_valid_reduce_aggregation(DataType source, Aggregation agg)

cpdef size_type unique_count(
    Column source,
    null_policy null_handling,
    nan_policy nan_handling,
    object stream = *
)

cpdef size_type distinct_count(
    Column source,
    null_policy null_handling,
    nan_policy nan_handling,
    object stream = *
)

cpdef size_type unique_count_table(
    Table source,
    null_equality nulls_equal,
    object stream = *
)

cpdef size_type distinct_count_table(
    Table source,
    null_equality nulls_equal,
    object stream = *
)

cdef class ApproxDistinctCount:
    cdef unique_ptr[cpp_approx_distinct_count] c_obj

    cpdef void add(self, Table input, object stream = *)
    cpdef void merge(self, ApproxDistinctCount other, object stream = *)
    cpdef size_t estimate(self, object stream = *)
    cpdef null_policy null_handling(self)
    cpdef nan_policy nan_handling(self)
    cpdef int32_t precision(self)
    cpdef double standard_error(self)
