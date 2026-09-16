/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/detail/copy_if.cuh>
#include <cudf/detail/stream_compaction.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/span.hpp>

#include <cuda/iterator>
#include <thrust/scatter.h>
#include <thrust/uninitialized_fill.h>

namespace cudf {
namespace detail {

std::unique_ptr<table> stable_distinct(table_view const& input,
                                       std::vector<size_type> const& keys,
                                       duplicate_keep_option keep,
                                       null_equality nulls_equal,
                                       nan_equality nans_equal,
                                       cuda::stream_ref stream,
                                       rmm::device_async_resource_ref mr)
{
  if (input.num_rows() == 0 or input.num_columns() == 0 or keys.empty()) {
    return empty_like(input);
  }

  auto const distinct_indices = detail::distinct_indices(input.select(keys),
                                                         keep,
                                                         nulls_equal,
                                                         nans_equal,
                                                         stream,
                                                         cudf::get_current_device_resource_ref());

  // The only difference between this implementation and the unstable version
  // is that the stable implementation must retain the input order. The
  // distinct indices are not sorted, so we cannot simply copy the rows in the
  // order of the distinct indices and retain the input order. Instead, we use
  // a boolean mask to indicate which rows to copy to the output. This avoids
  // the need to sort the distinct indices, which is slower.

  auto const output_markers = [&] {
    auto markers = rmm::device_uvector<bool>(input.num_rows(), stream);
    thrust::uninitialized_fill(
      rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
      markers.begin(),
      markers.end(),
      false);
    thrust::scatter(
      rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
      cuda::constant_iterator<bool>(true, 0),
      cuda::constant_iterator<bool>(true, static_cast<size_type>(distinct_indices.size())),
      distinct_indices.begin(),
      markers.begin());
    return markers;
  }();

  return cudf::detail::apply_mask(
    input, cudf::device_span<bool const>(output_markers), mask_type::RETENTION, stream, mr);
}

}  // namespace detail

std::unique_ptr<table> stable_distinct(table_view const& input,
                                       std::vector<size_type> const& keys,
                                       duplicate_keep_option keep,
                                       null_equality nulls_equal,
                                       nan_equality nans_equal,
                                       cuda::stream_ref stream,
                                       rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::stable_distinct(input, keys, keep, nulls_equal, nans_equal, stream, mr);
}

}  // namespace cudf
