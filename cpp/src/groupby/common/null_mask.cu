/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "utils.hpp"

#include <cudf/detail/valid_if.cuh>

#include <cuda/std/functional>
namespace cudf::groupby::detail {
std::pair<cuda::device_buffer<std::byte>, size_type> make_mask_from_validity(
  bool* begin, bool* end, cuda::stream_ref stream, cudf::memory_resources mr)
{
  return cudf::detail::valid_if(begin, end, cuda::std::identity{}, stream, mr);
}

std::pair<cuda::device_buffer<std::byte>, size_type> make_mask_from_counts(
  size_type const* begin, size_type const* end, cuda::stream_ref stream, cudf::memory_resources mr)
{
  return cudf::detail::valid_if(
    begin, end, [] __device__(size_type count) { return count > 0; }, stream, mr);
}

}  // namespace cudf::groupby::detail
