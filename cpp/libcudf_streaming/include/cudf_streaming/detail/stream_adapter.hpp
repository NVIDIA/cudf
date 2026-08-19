/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <rmm/cuda_stream_view.hpp>

#include <cuda/stream_ref>

#include <rapidsmpf/cuda_event.hpp>
#include <rapidsmpf/cuda_stream.hpp>

#include <ranges>
#include <utility>

namespace cudf_streaming::detail {

inline rmm::cuda_stream_view as_rmm_stream(rmm::cuda_stream_view stream) { return stream; }

inline rmm::cuda_stream_view as_rmm_stream(cuda::stream_ref stream)
{
  return rmm::cuda_stream_view{stream.get()};
}

template <typename Range>
auto as_rmm_streams(Range&& streams)
{
  return std::forward<Range>(streams) |
         std::views::transform([](auto&& stream) { return as_rmm_stream(stream); });
}

template <typename Downstream, typename Upstream>
  requires(!std::ranges::range<Downstream> && !std::ranges::range<Upstream>)
void cuda_stream_join(Downstream downstream,
                      Upstream upstream,
                      rapidsmpf::CudaEvent* event = nullptr)
{
  rapidsmpf::cuda_stream_join(as_rmm_stream(downstream), as_rmm_stream(upstream), event);
}

template <typename DownstreamRange, typename UpstreamRange>
  requires(std::ranges::range<DownstreamRange> && std::ranges::range<UpstreamRange>)
void cuda_stream_join(DownstreamRange&& downstreams,
                      UpstreamRange&& upstreams,
                      rapidsmpf::CudaEvent* event = nullptr)
{
  rapidsmpf::cuda_stream_join(as_rmm_streams(std::forward<DownstreamRange>(downstreams)),
                              as_rmm_streams(std::forward<UpstreamRange>(upstreams)),
                              event);
}

}  // namespace cudf_streaming::detail
