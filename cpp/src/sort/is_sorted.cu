/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/detail/algorithms/reduce.cuh>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/row_operator/lexicographic.cuh>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/table/table_device_view.cuh>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <cuda/iterator>
#include <thrust/sort.h>
#include <thrust/transform.h>

namespace cudf {
namespace detail {

bool is_sorted(cudf::table_view const& in,
               std::vector<order> const& column_order,
               std::vector<null_order> const& null_precedence,
               rmm::cuda_stream_view stream)
{
  auto const comparator =
    detail::row::lexicographic::self_comparator{in, column_order, null_precedence, stream};

  if (cudf::detail::has_nested_columns(in)) {
    auto const device_comparator = comparator.less<true>(has_nested_nulls(in));

    // Using a temporary buffer for intermediate transform results from the lambda containing
    // the comparator speeds up compile-time significantly over using the comparator directly
    // in thrust::is_sorted.
    auto d_results = rmm::device_uvector<bool>(in.num_rows(), stream);
    thrust::transform(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                      cuda::counting_iterator<size_type>{0},
                      cuda::counting_iterator<size_type>{in.num_rows()},
                      d_results.begin(),
                      [device_comparator] __device__(auto idx) -> bool {
                        return (idx == 0) || device_comparator(idx - 1, idx);
                      });

    return cudf::detail::all_of(
      d_results.begin(),
      d_results.end(),
      [] __device__(bool result) -> bool { return result; },
      stream);
  } else {
    auto const device_comparator = comparator.less<false>(has_nested_nulls(in));

    return thrust::is_sorted(
      rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
      cuda::counting_iterator<size_type>{0},
      cuda::counting_iterator<size_type>{in.num_rows()},
      device_comparator);
  }
}

}  // namespace detail

bool is_sorted(cudf::table_view const& in,
               std::vector<order> const& column_order,
               std::vector<null_order> const& null_precedence,
               rmm::cuda_stream_view stream)
{
  CUDF_FUNC_RANGE();
  if (in.num_columns() == 0 || in.num_rows() == 0) { return true; }

  if (not column_order.empty()) {
    CUDF_EXPECTS(static_cast<unsigned int>(in.num_columns()) == column_order.size(),
                 "Number of columns in the table doesn't match the vector column_order's size .\n");
  }

  if (not null_precedence.empty()) {
    CUDF_EXPECTS(
      static_cast<unsigned int>(in.num_columns()) == null_precedence.size(),
      "Number of columns in the table doesn't match the vector null_precedence's size .\n");
  }

  return detail::is_sorted(in, column_order, null_precedence, stream);
}

}  // namespace cudf
