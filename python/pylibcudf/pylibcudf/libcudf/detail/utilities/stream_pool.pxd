# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from cuda.bindings.cyruntime cimport cudaStream_t
from pylibcudf.exception_handler cimport libcudf_exception_handler
from pylibcudf.libcudf.utilities.span cimport host_span

ctypedef const cudaStream_t const_cudaStream_t


cdef extern from * nogil:
    """
    #include <cudf/detail/utilities/stream_pool.hpp>
    #include <cudf/utilities/span.hpp>
    #include <cuda/stream_ref>
    #include <vector>

    namespace {
    void join_streams_wrapper(
        cudf::host_span<cudaStream_t const> streams,
        cudaStream_t stream
    ) {
        std::vector<cuda::stream_ref> stream_refs;
        stream_refs.reserve(streams.size());
        for (auto const s : streams) {
            stream_refs.emplace_back(s);
        }
        cudf::detail::join_streams(stream_refs, cuda::stream_ref{stream});
    }
    }
    """
    cdef void join_streams "join_streams_wrapper"(
        host_span[const_cudaStream_t] streams,
        cudaStream_t stream
    ) except +libcudf_exception_handler
