# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from libcpp.memory cimport unique_ptr
from libcpp.pair cimport pair
from libcpp.vector cimport vector
from pylibcudf.libcudf.aggregation cimport (
    aggregation,
    groupby_aggregation,
    groupby_scan_aggregation,
)
from pylibcudf.libcudf.groupby cimport (
    aggregation_request,
    aggregation_result,
    groupby,
    scan_request,
)
from pylibcudf.libcudf.table.table cimport table
from pylibcudf.libcudf.types cimport null_order, order

from rmm.pylibrmm.memory_resource cimport DeviceMemoryResource

from .column cimport Column
from .table cimport Table


cdef class GroupByRequest:
    # The groupby APIs accept vectors of unique_ptrs to aggregation requests.
    # This ownership model means that if GroupByRequest owned the
    # corresponding C++ object, that object would have to be copied by e.g.
    # each groupby.aggregate call to avoid invalidating this object. Therefore,
    # this class instead stores only Python/Cython objects and constructs the
    # C++ object on the fly as requested.
    cdef Column _values
    cdef list _aggregations

    cdef aggregation_request _to_libcudf_agg_request(self) except *
    cdef scan_request _to_libcudf_scan_request(self) except *


cdef class GroupBy:
    cdef unique_ptr[groupby] c_obj
    cdef Table _keys
    cdef unique_ptr[vector[order]] _column_order
    cdef unique_ptr[vector[null_order]] _null_precedence

    cpdef tuple[Table, list[Table]] aggregate(
        self, list requests, object stream = *, DeviceMemoryResource mr=*
    )
    cpdef tuple[Table, list[Table]] scan(
        self, list requests, object stream = *, DeviceMemoryResource mr=*
    )
    cpdef tuple[Table, Table] shift(
        self,
        Table values,
        list offset,
        list fill_values,
        object stream = *,
        DeviceMemoryResource mr=*,
    )
    cpdef tuple[Table, Table] replace_nulls(
        self,
        Table values,
        list replace_policies,
        object stream = *,
        DeviceMemoryResource mr=*,
    )
    cpdef tuple[list[int], Table, object] get_groups(
        self, Table values=*, object stream = *, DeviceMemoryResource mr=*
    )

    @staticmethod
    cdef tuple[Table, list[Table]] _parse_outputs(
        pair[unique_ptr[table], vector[aggregation_result]] c_res,
        object stream,
        DeviceMemoryResource mr,
    )
