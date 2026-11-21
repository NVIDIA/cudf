/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "hash_graph_kernels.cuh"

#include <cudf/detail/sizes_to_offsets_iterator.cuh>
#include <cudf/utilities/memory_resource.hpp>

#include <cuda/iterator>

namespace cudf::detail {
namespace {

/// Match counts followed by a zero, so that the last exclusive-scan output is the total
struct padded_counts {
  size_type const* counts;
  size_type size;
  __device__ size_type operator()(cuda::std::int64_t idx) const
  {
    return idx < size ? counts[idx] : 0;
  }
};

}  // namespace

std::pair<cuda::device_buffer<cuda::std::int64_t>, std::size_t> hash_graph_match_offsets(
  device_span<size_type const> match_counts, cuda::stream_ref stream)
{
  auto const temp_mr  = cudf::get_current_device_resource_ref();
  auto const num_rows = static_cast<size_type>(match_counts.size());
  // 64-bit indices keep the end offset's index in range when there are INT32_MAX probe rows
  auto const sizes = cuda::transform_iterator(cuda::counting_iterator<cuda::std::int64_t>{0},
                                              padded_counts{match_counts.data(), num_rows});
  cuda::device_buffer<cuda::std::int64_t> offsets(
    stream, temp_mr, match_counts.size() + 1, cuda::no_init);
  auto const output_size = cudf::detail::sizes_to_offsets(
    sizes, sizes + num_rows + 1, offsets.begin(), 0, stream, temp_mr);
  return {std::move(offsets), static_cast<std::size_t>(output_size)};
}

}  // namespace cudf::detail
