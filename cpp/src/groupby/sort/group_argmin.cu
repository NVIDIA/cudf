/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "groupby/sort/group_single_pass_reduction_util.cuh"

#include <cudf/detail/gather.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/device_uvector.hpp>

#include <cuda/stream>
#include <thrust/gather.h>

namespace cudf {
namespace groupby {
namespace detail {
std::unique_ptr<column> group_argmin(column_view const& values,
                                     size_type num_groups,
                                     cudf::device_span<size_type const> group_labels,
                                     column_view const& key_sort_order,
                                     cuda::stream_ref stream,
                                     rmm::device_async_resource_ref mr)
{
  auto dispatch_type = cudf::is_dictionary(values.type())
                         ? dictionary_column_view(values).keys().type()
                         : values.type();
  auto indices       = type_dispatcher(dispatch_type,
                                 group_reduction_dispatcher<aggregation::ARGMIN>{},
                                 values,
                                 num_groups,
                                 group_labels,
                                 stream,
                                 mr);

  // The functor returns the indices of minimums based on the sorted keys.
  // We need the indices of minimums from the original unsorted keys
  // so we use these and the key_sort_order to map to the correct indices.
  // We do not use cudf::gather since we can move the null-mask separately.
  auto indices_view = indices->view();
  auto output       = rmm::device_uvector<size_type>(indices_view.size(), stream, mr);
  thrust::gather(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                 indices_view.begin<size_type>(),    // map first
                 indices_view.end<size_type>(),      // map last
                 key_sort_order.begin<size_type>(),  // input
                 output.data()                       // result (must not overlap map)
  );
  auto null_count = indices_view.null_count();
  auto null_mask  = indices->release().null_mask.release();
  return std::make_unique<column>(std::move(output), std::move(*null_mask), null_count);
}

}  // namespace detail
}  // namespace groupby
}  // namespace cudf
