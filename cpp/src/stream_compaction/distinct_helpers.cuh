/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "distinct_helpers.hpp"

#include <cudf/stream_compaction.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>
#include <rmm/resource_ref.hpp>

#include <cuco/operator.hpp>
#include <cuda/iterator>
#include <cuda/std/atomic>
#include <cuda/std/iterator>
#include <thrust/for_each.h>

namespace cudf::detail {

template <typename Set>
rmm::device_uvector<size_type> reduce_by_row_keep_any(Set& set,
                                                      size_type num_rows,
                                                      rmm::cuda_stream_view stream,
                                                      rmm::device_async_resource_ref mr)
{
  auto output_indices = rmm::device_uvector<size_type>(num_rows, stream, mr);

  auto const iter = cuda::counting_iterator<cudf::size_type>{0};
  set.insert_async(iter, iter + num_rows, stream.value());
  auto const output_end = set.retrieve_all(output_indices.begin(), stream.value());
  output_indices.resize(cuda::std::distance(output_indices.begin(), output_end), stream);
  return output_indices;
}

template <typename Set>
rmm::device_uvector<size_type> reduce_by_row_keep_first_last_none(Set& set,
                                                                  size_type num_rows,
                                                                  duplicate_keep_option keep,
                                                                  rmm::cuda_stream_view stream,
                                                                  rmm::device_async_resource_ref mr)
{
  auto output_indices = rmm::device_uvector<size_type>(num_rows, stream, mr);
  auto reduction_results =
    rmm::device_uvector<size_type>(num_rows, stream, cudf::get_current_device_resource_ref());
  initialize_reduction_results(reduction_results.data(), num_rows, keep, stream);

  auto set_ref = set.ref(cuco::op::insert_and_find);

  thrust::for_each(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                   cuda::counting_iterator<cudf::size_type>{0},
                   cuda::counting_iterator{num_rows},
                   [set_ref, keep, reduction_results = reduction_results.begin()] __device__(
                     size_type const idx) mutable -> void {
                     auto const [inserted_idx_ptr, _] = set_ref.insert_and_find(idx);

                     auto ref = cuda::atomic_ref<size_type, cuda::thread_scope_device>{
                       reduction_results[*inserted_idx_ptr]};
                     if (keep == duplicate_keep_option::KEEP_FIRST) {
                       // Store the smallest index of all rows that are equal.
                       ref.fetch_min(idx, cuda::memory_order_relaxed);
                     } else if (keep == duplicate_keep_option::KEEP_LAST) {
                       // Store the greatest index of all rows that are equal.
                       ref.fetch_max(idx, cuda::memory_order_relaxed);
                     } else {
                       // Count the number of rows in each group of rows that are compared equal.
                       ref.fetch_add(size_type{1}, cuda::memory_order_relaxed);
                     }
                   });

  auto const output_size =
    copy_reduction_results(reduction_results.data(), num_rows, output_indices.data(), keep, stream);
  output_indices.resize(output_size, stream);
  return output_indices;
}

}  // namespace cudf::detail
