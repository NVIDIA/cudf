# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from libc.stdint cimport uint32_t, uint64_t
from libcpp.memory cimport unique_ptr
from libcpp.optional cimport optional
from libcpp.span cimport span
from libcpp.string cimport string
from libcpp.vector cimport vector

from pylibcudf.libcudf.io.datasource cimport datasource

from cudf_polars_cucascade.exception_handler cimport libcucascade_exception_handler
from cudf_polars_cucascade.libcucascade.column_admission cimport column_stats


ctypedef const uint32_t const_uint32_t
ctypedef const uint64_t const_uint64_t


cdef extern from "cucascade/io/scan_cache.hpp" namespace "cucascade::io" nogil:

    cdef cppclass scan_cache_config:
        scan_cache_config() except +libcucascade_exception_handler
        size_t cache_bytes
        size_t block_bytes
        double eviction_threshold
        double eviction_batch
        double admission_budget
        size_t min_advise_bytes
        size_t reactors
        size_t connections
        long stall_seconds
        size_t pinning_threads
        optional[int] numa_node
        string endpoint
        string region
        string access_key
        string secret_key
        string session_token


cdef extern from "scan_cache.hpp" namespace "cudf_polars_cucascade" nogil:

    cdef cppclass scan_cache:
        scan_cache(const scan_cache_config&) except +libcucascade_exception_handler
        unique_ptr[datasource] open(
            const string& path
        ) except +libcucascade_exception_handler
        void register_layout(
            const string& path,
            const string& table,
            const vector[string]& columns,
            span[const_uint32_t] column_of_chunk,
            span[const_uint64_t] offsets,
            span[const_uint64_t] sizes,
        ) except +libcucascade_exception_handler
        vector[column_stats] stats() except +libcucascade_exception_handler const
        string summary() except +libcucascade_exception_handler const
