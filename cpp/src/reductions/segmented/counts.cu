/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "counts.hpp"

#include <cudf/detail/null_mask.cuh>
#include <cudf/utilities/memory_resource.hpp>

#include <thrust/adjacent_difference.h>

namespace cudf {
namespace reduction {
namespace detail {

rmm::device_uvector<size_type> segmented_counts(bitmask_type const* null_mask,
                                                bool has_nulls,
                                                device_span<size_type const> offsets,
                                                null_policy null_handling,
                                                cuda::stream_ref stream,
                                                rmm::device_async_resource_ref mr)
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

  rmm::device_uvector<size_type> valid_counts(num_segments, stream, mr);
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
