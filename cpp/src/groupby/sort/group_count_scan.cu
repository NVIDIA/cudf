/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "group_scan.hpp"

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/types.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/exec_policy.hpp>

#include <cuda/iterator>
#include <cuda/stream>
#include <thrust/scan.h>

namespace cudf {
namespace groupby {
namespace detail {
std::unique_ptr<column> count_scan(column_view const& values,
                                   null_policy nulls,
                                   cudf::device_span<size_type const> group_labels,
                                   cuda::stream_ref stream,
                                   rmm::device_async_resource_ref mr)
{
  std::unique_ptr<column> result = make_fixed_width_column(
    data_type{type_id::INT32}, group_labels.size(), mask_state::UNALLOCATED, stream, mr);

  if (group_labels.empty()) { return result; }

  auto resultview = result->mutable_view();
  // aggregation::COUNT_ALL
  if (nulls == null_policy::INCLUDE) {
    thrust::inclusive_scan_by_key(
      rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
      group_labels.begin(),
      group_labels.end(),
      cuda::make_constant_iterator<size_type>(1),
      resultview.begin<size_type>());
  } else {  // aggregation::COUNT_VALID
    auto d_values = cudf::column_device_view::create(values, stream);
    auto itr      = cudf::detail::make_counting_transform_iterator(
      0, [d_values = *d_values] __device__(auto idx) -> cudf::size_type {
        return d_values.is_valid(idx);
      });
    thrust::inclusive_scan_by_key(
      rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
      group_labels.begin(),
      group_labels.end(),
      itr,
      resultview.begin<size_type>());
  }
  return result;
}

}  // namespace detail
}  // namespace groupby
}  // namespace cudf
