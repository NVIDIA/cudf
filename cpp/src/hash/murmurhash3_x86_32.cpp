/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "murmurhash3_x86_32.cuh"

#include "murmurhash3_x86_32_rtcx.hpp"

#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/hashing.hpp>

namespace cudf::hashing {
namespace detail {

std::unique_ptr<column> murmurhash3_x86_32(table_view const& input,
                                           uint32_t seed,
                                           cuda::stream_ref stream,
                                           rmm::device_async_resource_ref mr)
{
  return murmurhash3_x86_32_rtcx(input, seed, stream, mr);
}

std::unique_ptr<column> murmurhash3_x86_32(
  std::shared_ptr<cudf::detail::row::equality::preprocessed_table> const& input,
  size_type num_rows,
  uint32_t seed,
  cuda::stream_ref stream,
  rmm::device_async_resource_ref mr)
{
  // These callers share preprocessing with equality and require nested-null checks.
  return murmurhash3_x86_32_rtcx_generic(input, num_rows, seed, true, stream, mr);
}

}  // namespace detail

std::unique_ptr<column> murmurhash3_x86_32(table_view const& input,
                                           uint32_t seed,
                                           cuda::stream_ref stream,
                                           rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::murmurhash3_x86_32(input, seed, stream, mr);
}

}  // namespace cudf::hashing
