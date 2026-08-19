/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_view.hpp>
#include <cudf/detail/copy.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/reduction/detail/reduction_functions.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <cuda/functional>
#include <cuda/iterator>
#include <cuda/stream>
#include <thrust/binary_search.h>
#include <thrust/scan.h>

namespace cudf::reduction::detail {

std::unique_ptr<cudf::scalar> nth_element(column_view const& col,
                                          size_type n,
                                          null_policy null_handling,
                                          cuda::stream_ref stream,
                                          rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(n >= -col.size() and n < col.size(), "Index out of bounds");
  auto wrap_n = [n](size_type size) { return (n < 0 ? size + n : n); };
  if (null_handling == null_policy::EXCLUDE and col.has_nulls()) {
    auto valid_count = col.size() - col.null_count();
    n                = wrap_n(valid_count);
    CUDF_EXPECTS(n >= 0 and n < valid_count, "Index out of bounds");
    auto dcol = column_device_view::create(col, stream);
    auto bitmask_iterator =
      cuda::transform_iterator(cudf::detail::make_validity_iterator(*dcol),
                               cuda::proclaim_return_type<size_type>(
                                 [] __device__(auto b) { return static_cast<size_type>(b); }));
    rmm::device_uvector<size_type> null_skipped_index(col.size(), stream);
    // null skipped index for valids only.
    thrust::inclusive_scan(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                           bitmask_iterator,
                           bitmask_iterator + col.size(),
                           null_skipped_index.begin());

    auto n_pos =
      thrust::upper_bound(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                          null_skipped_index.begin(),
                          null_skipped_index.end(),
                          n);
    auto null_skipped_n = n_pos - null_skipped_index.begin();
    return cudf::detail::get_element(col, null_skipped_n, stream, mr);
  } else {
    n = wrap_n(col.size());
    return cudf::detail::get_element(col, n, stream, mr);
  }
}

}  // namespace cudf::reduction::detail
