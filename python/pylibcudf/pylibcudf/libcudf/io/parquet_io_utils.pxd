# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from libc.stdint cimport uint8_t
from cuda.bindings.cyruntime cimport cudaStream_t
from libcpp.memory cimport shared_ptr, unique_ptr
from libcpp.pair cimport pair
from libcpp.vector cimport vector

from rmm.librmm.device_buffer cimport device_buffer
from rmm.librmm.memory_resource cimport device_async_resource_ref

from pylibcudf.exception_handler cimport libcudf_exception_handler
from pylibcudf.libcudf.io.datasource cimport datasource
from pylibcudf.libcudf.io.text cimport byte_range_info
from pylibcudf.libcudf.utilities.span cimport device_span

ctypedef const uint8_t const_uint8_t

cdef extern from "cudf/io/parquet_io_utils.hpp" \
        namespace "cudf::io::parquet" nogil:

    cpdef enum class io_submission_policy:
        SERIALIZE
        INTERLEAVE

cdef extern from * nogil:
    """
    #include <functional>
    #include <future>
    #include <cuda/stream>
    #include <cudf/io/parquet_io_utils.hpp>

    static std::vector<std::reference_wrapper<cudf::io::datasource>>
    as_datasource_refs(
        std::vector<std::unique_ptr<cudf::io::datasource>>& datasources)
    {
        std::vector<std::reference_wrapper<cudf::io::datasource>> sources;
        sources.reserve(datasources.size());
        for (auto& source : datasources) { sources.emplace_back(*source); }
        return sources;
    }

    struct bloom_filter_fetch {
        std::shared_ptr<void> buffers;
        std::vector<std::vector<cudf::device_span<uint8_t const>>> spans;
    };

    static bloom_filter_fetch cpp_fetch_bloom_filters_to_device(
        std::vector<std::unique_ptr<cudf::io::datasource>>& datasources,
        std::vector<std::vector<cudf::io::text::byte_range_info>> const&
            byte_ranges_per_source,
        cudf::io::parquet::io_submission_policy policy,
        cudaStream_t stream,
        rmm::device_async_resource_ref mr)
    {
        auto sources = as_datasource_refs(datasources);
        auto [buffers, spans] =
            cudf::io::parquet::fetch_bloom_filters_to_device(
                std::span<std::reference_wrapper<cudf::io::datasource> const>{sources},
                std::span<std::vector<cudf::io::text::byte_range_info> const>{
                    byte_ranges_per_source},
                policy,
                cuda::stream_ref{stream},
                mr);
        return {std::make_shared<decltype(buffers)>(std::move(buffers)),
                std::move(spans)};
    }

    static std::pair<std::vector<rmm::device_buffer>,
                     std::vector<std::vector<cudf::device_span<uint8_t const>>>>
    cpp_fetch_byte_ranges_to_device(
        std::vector<std::unique_ptr<cudf::io::datasource>>& datasources,
        std::vector<std::vector<cudf::io::text::byte_range_info>> const&
            byte_ranges_per_source,
        cudf::io::parquet::io_submission_policy policy,
        cudaStream_t stream,
        rmm::device_async_resource_ref mr)
    {
        auto sources = as_datasource_refs(datasources);
        auto [buffers, spans, fut] =
            cudf::io::parquet::fetch_byte_ranges_to_device_async(
                std::span<std::reference_wrapper<cudf::io::datasource> const>{sources},
                std::span<std::vector<cudf::io::text::byte_range_info> const>{
                    byte_ranges_per_source},
                policy,
                cuda::stream_ref{stream},
                mr);
        // Block until the async fetch completes so the returned buffers/spans
        // are fully populated. This also avoids exposing std::future to Cython.
        fut.get();
        return {std::move(buffers), std::move(spans)};
    }
    """
    cdef cppclass bloom_filter_fetch:
        shared_ptr[void] buffers
        vector[vector[device_span[const_uint8_t]]] spans

    bloom_filter_fetch cpp_fetch_bloom_filters_to_device(
        vector[unique_ptr[datasource]]& sources,
        const vector[vector[byte_range_info]]& byte_ranges_per_source,
        io_submission_policy policy,
        cudaStream_t stream,
        device_async_resource_ref mr,
    ) except +libcudf_exception_handler

    pair[vector[device_buffer], vector[vector[device_span[const_uint8_t]]]] \
        cpp_fetch_byte_ranges_to_device(
            vector[unique_ptr[datasource]]& sources,
            const vector[vector[byte_range_info]]& byte_ranges_per_source,
            io_submission_policy policy,
            cudaStream_t stream,
            device_async_resource_ref mr,
        ) except +libcudf_exception_handler

cdef extern from "cudf/io/parquet_io_utils.hpp" \
        namespace "cudf::io::parquet" nogil:

    unique_ptr[datasource.buffer] fetch_page_index_to_host(
        datasource& ds,
        byte_range_info page_index_bytes,
    ) except +libcudf_exception_handler
