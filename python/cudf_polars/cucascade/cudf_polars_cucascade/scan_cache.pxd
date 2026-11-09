# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from libcpp.memory cimport unique_ptr

from cudf_polars_cucascade.advising_datasource cimport AdvisingDatasource
from cudf_polars_cucascade.column_admission cimport ColumnStats
from cudf_polars_cucascade.libcucascade.scan_cache cimport (
    scan_cache as cpp_scan_cache,
    scan_cache_config as cpp_scan_cache_config,
)


cdef class ScanCacheConfig:
    cdef cpp_scan_cache_config c_obj


cdef class ScanCache:
    cdef unique_ptr[cpp_scan_cache] c_obj

    cpdef AdvisingDatasource open(self, str path)

    cpdef void register_layout(
        self,
        str path,
        str table,
        list columns,
        list column_of_chunk,
        list offsets,
        list sizes,
    )

    cpdef list[ColumnStats] column_stats(self)

    cpdef str summary(self)
