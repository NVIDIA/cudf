/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/detail/algorithms/reduce.cuh>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/row_operator/common_utils.cuh>
#include <cudf/detail/row_operator/equality.cuh>
#include <cudf/table/equality.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/type_checks.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <cub/device/device_transform.cuh>
#include <cuda/iterator>

namespace cudf {
namespace detail {
namespace {

template <bool has_nested_columns>
[[nodiscard]] bool tables_equal(table_view const& left,
                                table_view const& right,
                                null_equality nulls_equal,
                                rmm::cuda_stream_view stream)
{
  auto const comparator = detail::row::equality::two_table_comparator{left, right, stream};
  auto const rows_equal = comparator.equal_to<has_nested_columns>(
    nullate::DYNAMIC{has_nested_nulls(left) or has_nested_nulls(right)}, nulls_equal);
  rmm::device_uvector<bool> eq_rows{
    static_cast<std::size_t>(left.num_rows()), stream, cudf::get_current_device_resource_ref()};
  CUDF_CUDA_TRY(cub::DeviceTransform::Transform(
    cuda::counting_iterator<size_type>{0},
    eq_rows.begin(),
    eq_rows.size(),
    [rows_equal] __device__(size_type i) -> bool {
      return rows_equal(detail::row::lhs_index_type{i}, detail::row::rhs_index_type{i});
    },
    stream.value()));
  return cudf::detail::all_of(
    eq_rows.begin(),
    eq_rows.end(),
    [] __device__(bool row_equal) -> bool { return row_equal; },
    stream);
}

}  // namespace

[[nodiscard]] bool tables_equal(table_view const& left,
                                table_view const& right,
                                null_equality nulls_equal,
                                rmm::cuda_stream_view stream)
{
  if (left.num_rows() != right.num_rows() || left.num_columns() != right.num_columns() ||
      !have_same_types(left, right)) {
    return false;
  } else if (left.num_rows() == 0) {
    return true;
  }

  return cudf::has_nested_columns(left) || cudf::has_nested_columns(right)
           ? tables_equal<true>(left, right, nulls_equal, stream)
           : tables_equal<false>(left, right, nulls_equal, stream);
}
}  // namespace detail

bool tables_equal(table_view const& left,
                  table_view const& right,
                  null_equality nulls_equal,
                  rmm::cuda_stream_view stream)
{
  CUDF_FUNC_RANGE();
  return detail::tables_equal(left, right, nulls_equal, stream);
}

}  // namespace cudf
