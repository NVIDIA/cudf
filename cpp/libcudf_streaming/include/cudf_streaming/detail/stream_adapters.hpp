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

inline cuda::stream_ref as_cuda_stream_ref(cuda::stream_ref stream) noexcept { return stream; }

inline cuda::stream_ref as_cuda_stream_ref(rmm::cuda_stream_view stream) noexcept
{
  return cuda::stream_ref{stream.value()};
}

template <typename Range>
auto as_cuda_stream_ref_range(Range&& streams)
{
  return std::forward<Range>(streams) |
         std::views::transform([](auto stream) { return as_cuda_stream_ref(stream); });
}

}  // namespace cudf_streaming::detail
