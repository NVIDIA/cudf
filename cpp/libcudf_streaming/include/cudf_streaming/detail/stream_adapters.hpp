/**
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <rmm/cuda_stream_view.hpp>

#include <cuda/stream>

#include <ranges>
#include <utility>

namespace cudf_streaming::detail {

inline rmm::cuda_stream_view as_rmm_cuda_stream_view(rmm::cuda_stream_view stream) noexcept
{
  return stream;
}

inline rmm::cuda_stream_view as_rmm_cuda_stream_view(cuda::stream_ref stream) noexcept
{
  return rmm::cuda_stream_view{stream.get()};
}

template <typename Range>
auto as_rmm_cuda_stream_view_range(Range&& streams)
{
  return std::forward<Range>(streams) |
         std::views::transform([](auto stream) { return as_rmm_cuda_stream_view(stream); });
}

}  // namespace cudf_streaming::detail
