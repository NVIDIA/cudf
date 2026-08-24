/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "distinct_helpers.hpp"

#include <cudf/detail/algorithms/copy_if.cuh>
#include <cudf/stream_compaction.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/exec_policy.hpp>
#include <rmm/resource_ref.hpp>

#include <cuda/functional>
#include <cuda/iterator>
#include <cuda/std/iterator>
#include <cuda/stream>
#include <thrust/uninitialized_fill.h>

namespace cudf::detail {

void initialize_reduction_results(size_type* results,
                                  size_type num_rows,
                                  duplicate_keep_option keep,
                                  cuda::stream_ref stream)
{
  thrust::uninitialized_fill(
    rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
    results,
    results + num_rows,
    reduction_init_value(keep));
}

size_type copy_reduction_results(size_type const* results,
                                 size_type num_rows,
                                 size_type* output,
                                 duplicate_keep_option keep,
                                 cuda::stream_ref stream)
{
  auto const output_end = [&] {
    if (keep == duplicate_keep_option::KEEP_NONE) {
      // KEEP_NONE stores group sizes; retain only singleton groups.
      return cudf::detail::copy_if(
        cuda::counting_iterator<size_type>{0},
        cuda::counting_iterator<size_type>{num_rows},
        output,
        cuda::proclaim_return_type<bool>(
          [results] __device__(auto const idx) { return results[idx] == size_type{1}; }),
        stream);
    }

    // KEEP_FIRST and KEEP_LAST store desired row indices or the mode's initial marker.
    return cudf::detail::copy_if(
      results,
      results + num_rows,
      output,
      cuda::proclaim_return_type<bool>([init_value = reduction_init_value(keep)] __device__(
                                         auto const idx) { return idx != init_value; }),
      stream);
  }();

  return cuda::std::distance(output, output_end);
}

}  // namespace cudf::detail
