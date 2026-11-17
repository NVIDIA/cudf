/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "counts.hpp"

#include <cudf/detail/null_mask.cuh>
#include <cudf/utilities/memory_resource.hpp>

#include <cuda/buffer>
#include <thrust/adjacent_difference.h>

namespace cudf {
namespace reduction {
namespace detail {

cuda::device_buffer<size_type> segmented_counts(bitmask_type const* null_mask,
                                                bool has_nulls,
                                                device_span<size_type const> offsets,
                                                null_policy null_handling,
                                                cuda::stream_ref stream,
                                                cudf::memory_resources mr)
{
  auto const num_segments = offsets.size() - 1;

  if (has_nulls && (null_handling == null_policy::EXCLUDE)) {
    return cudf::detail::segmented_count_bits(null_mask,
                                              offsets.begin(),
                                              offsets.end() - 1,
                                              offsets.begin() + 1,
                                              cudf::detail::count_bits_policy::SET_BITS,
                                              stream,
                                              mr);
  }

  cuda::device_buffer<size_type> valid_counts(
    stream, mr.get_output_mr(), num_segments, cuda::no_init);
  thrust::adjacent_difference(
    rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
    offsets.begin() + 1,
    offsets.end(),
    valid_counts.begin());
  return valid_counts;
}

}  // namespace detail
}  // namespace reduction
}  // namespace cudf
