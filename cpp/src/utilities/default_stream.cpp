/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/utilities/default_stream.hpp>

#include <cstdlib>

namespace cudf {

namespace detail {

#if defined(CUDF_USE_PER_THREAD_DEFAULT_STREAM)
cuda::stream_ref const default_stream_value{cudaStreamPerThread};
#else
cuda::stream_ref const default_stream_value{cudaStream_t{cudaStreamDefault}};
#endif

}  // namespace detail

/**
 * @brief Check if per-thread default stream is enabled.
 *
 * @return true if PTDS is enabled, false otherwise.
 */
bool is_ptds_enabled()
{
#ifdef CUDA_API_PER_THREAD_DEFAULT_STREAM
  return true;
#else
  return false;
#endif
}

cuda::stream_ref const get_default_stream()
{
  static auto const default_stream = []() {
    if (std::getenv("CUDF_PER_THREAD_STREAM") != nullptr) {
      return cuda::stream_ref{cudaStreamPerThread};
    } else {
      return detail::default_stream_value;
    }
  }();
  return default_stream;
}
}  // namespace cudf
