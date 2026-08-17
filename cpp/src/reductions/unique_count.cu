/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/row_operator/equality.cuh>
#include <cudf/reduction/detail/unique_count.hpp>
#include <cudf/reduction/unique_count.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <rmm/exec_policy.hpp>

#include <cuda/iterator>
#include <cuda/stream>
#include <thrust/count.h>
#include <thrust/execution_policy.h>
#include <thrust/transform.h>

namespace cudf {
namespace detail {

cudf::size_type unique_count(table_view const& keys,
                             null_equality nulls_equal,
                             cuda::stream_ref stream)
{
  auto const row_comp = cudf::detail::row::equality::self_comparator(keys, stream);
  if (cudf::detail::has_nested_columns(keys)) {
    auto const comp =
      row_comp.equal_to<true>(nullate::DYNAMIC{has_nested_nulls(keys)}, nulls_equal);
    // Using a temporary buffer for intermediate transform results from the lambda containing
    // the comparator speeds up compile-time significantly without much degradation in
    // runtime performance over using the comparator directly in thrust::count_if.
    auto d_results = rmm::device_uvector<bool>(keys.num_rows(), stream);
    thrust::transform(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                      cuda::counting_iterator<size_type>{0},
                      cuda::counting_iterator<size_type>{keys.num_rows()},
                      d_results.begin(),
                      [comp] __device__(auto i) { return (i == 0 or not comp(i, i - 1)); });

    return static_cast<size_type>(
      thrust::count(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                    d_results.begin(),
                    d_results.end(),
                    true));
  } else {
    auto const comp =
      row_comp.equal_to<false>(nullate::DYNAMIC{has_nested_nulls(keys)}, nulls_equal);
    // Using thrust::copy_if with the comparator directly will compile more slowly but
    // improves runtime by up to 2x over the transform/count approach above.
    return thrust::count_if(
      rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
      cuda::counting_iterator<cudf::size_type>{0},
      cuda::counting_iterator<cudf::size_type>{keys.num_rows()},
      [comp] __device__(cudf::size_type i) { return (i == 0 or not comp(i, i - 1)); });
  }
}

}  // namespace detail

cudf::size_type unique_count(table_view const& input,
                             null_equality nulls_equal,
                             cuda::stream_ref stream)
{
  CUDF_FUNC_RANGE();
  return detail::unique_count(input, nulls_equal, stream);
}

}  // namespace cudf
