/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/aggregation/aggregation.hpp>
#include <cudf/detail/algorithms/reduce.cuh>
#include <cudf/detail/gather.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/types.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/exec_policy.hpp>

#include <cuda/functional>
#include <cuda/iterator>
#include <cuda/stream>
#include <thrust/scan.h>
#include <thrust/scatter.h>
#include <thrust/transform.h>
#include <thrust/uninitialized_fill.h>

namespace cudf {
namespace groupby {
namespace detail {
std::unique_ptr<column> group_nth_element(column_view const& values,
                                          column_view const& group_sizes,
                                          cudf::device_span<size_type const> group_labels,
                                          cudf::device_span<size_type const> group_offsets,
                                          size_type num_groups,
                                          size_type n,
                                          null_policy null_handling,
                                          cuda::stream_ref stream,
                                          rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(static_cast<size_t>(values.size()) == group_labels.size(),
               "Size of values column should be same as that of group labels");

  if (num_groups == 0) { return empty_like(values); }

  auto nth_index = rmm::device_uvector<size_type>(num_groups, stream);
  // TODO: replace with async version
  thrust::uninitialized_fill_n(
    rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
    nth_index.begin(),
    num_groups,
    values.size());

  // nulls_policy::INCLUDE (equivalent to pandas nth(dropna=None) but return nulls for n
  if (null_handling == null_policy::INCLUDE || !values.has_nulls()) {
    // Returns index of nth value.
    thrust::transform_if(
      rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
      group_sizes.begin<size_type>(),
      group_sizes.end<size_type>(),
      group_offsets.begin(),
      group_sizes.begin<size_type>(),  // stencil
      nth_index.begin(),
      [n] __device__(auto group_size, auto group_offset) {
        return group_offset + ((n < 0) ? group_size + n : n);
      },
      [n] __device__(auto group_size) {  // nth within group
        return (n < 0) ? group_size >= (-n) : group_size > n;
      });
  } else {  // skip nulls (equivalent to pandas nth(dropna='any'))
    // The scan numbers the valid rows within each group, and its output visitor records the row
    // whose number is the requested one. `group_sizes` holds the valid count of each group, so a
    // negative `n` counts back from the last valid row.
    auto values_view = column_device_view::create(values, stream);
    auto bitmask_iterator =
      cuda::transform_iterator(cudf::detail::make_validity_iterator(*values_view),
                               cuda::proclaim_return_type<size_type>(
                                 [] __device__(auto b) { return static_cast<size_type>(b); }));
    auto const record_nth = cuda::tabulate_output_iterator{
      [n,
       bitmask_iterator,
       group_sizes  = group_sizes.begin<size_type>(),
       group_labels = group_labels.begin(),
       nth_index    = nth_index.begin()] __device__(cuda::std::ptrdiff_t i, size_type intra) {
        auto const group = group_labels[i];
        auto const nth   = n < 0 ? group_sizes[group] + n : n;
        if (bitmask_iterator[i] && intra == nth) { nth_index[group] = static_cast<size_type>(i); }
      }};
    thrust::exclusive_scan_by_key(
      rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
      group_labels.begin(),
      group_labels.end(),
      bitmask_iterator,
      record_nth);
  }

  auto output_table = cudf::detail::gather(table_view{{values}},
                                           nth_index,
                                           out_of_bounds_policy::NULLIFY,
                                           cudf::negative_index_policy::NOT_ALLOWED,
                                           stream,
                                           mr);
  if (!output_table->get_column(0).has_nulls())
    output_table->get_column(0).set_null_mask(
      cudf::create_null_mask(0, cudf::mask_state::UNALLOCATED), 0);
  return std::make_unique<column>(std::move(output_table->get_column(0)));
}
}  // namespace detail
}  // namespace groupby
}  // namespace cudf
