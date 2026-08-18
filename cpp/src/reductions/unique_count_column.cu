/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_view.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/row_operator/equality.cuh>
#include <cudf/reduction/detail/unique_count.hpp>
#include <cudf/reduction/unique_count.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/exec_policy.hpp>

#include <cuda/iterator>
#include <cuda/std/cmath>
#include <cuda/std/type_traits>
#include <cuda/stream>
#include <thrust/count.h>
#include <thrust/execution_policy.h>

namespace cudf {
namespace detail {
namespace {
/**
 * @brief A functor to be used along with device type_dispatcher to check if
 * the row `index` of `column_device_view` is `NaN`.
 */
struct check_nan {
  template <typename T>
  __device__ inline bool operator()(column_device_view const& input, size_type index)
    requires(cuda::std::is_floating_point_v<T>)
  {
    return cuda::std::isnan(input.data<T>()[index]);
  }
  template <typename T>
  __device__ inline bool operator()(column_device_view const&, size_type)
    requires(not cuda::std::is_floating_point_v<T>)
  {
    return false;
  }
};
}  // namespace

cudf::size_type unique_count(column_view const& input,
                             null_policy null_handling,
                             nan_policy nan_handling,
                             cuda::stream_ref stream)
{
  auto const num_rows = input.size();

  if (num_rows == 0 or num_rows == input.null_count()) { return 0; }

  auto const count_nulls      = null_handling == null_policy::INCLUDE;
  auto const nan_is_null      = nan_handling == nan_policy::NAN_IS_NULL;
  auto const should_check_nan = cudf::is_floating_point(input.type());
  auto input_device_view      = cudf::column_device_view::create(input, stream);
  auto device_view            = *input_device_view;
  auto input_table_view       = table_view{{input}};

  auto temp_mr = cudf::get_current_device_resource_ref();
  auto const comparator =
    cudf::detail::row::equality::self_comparator{input_table_view, stream, temp_mr};
  auto const comp = comparator.equal_to<false>(
    nullate::DYNAMIC{cudf::has_nulls(input_table_view)},
    null_equality::EQUAL,
    cudf::detail::row::equality::nan_equal_physical_equality_comparator{});

  return thrust::count_if(
    rmm::exec_policy_nosync(stream, temp_mr),
    cuda::counting_iterator<cudf::size_type>{0},
    cuda::counting_iterator<cudf::size_type>{num_rows},
    [count_nulls, nan_is_null, should_check_nan, device_view, comp] __device__(cudf::size_type i) {
      auto const is_null = device_view.is_null(i);
      auto const is_nan  = nan_is_null and should_check_nan and
                          cudf::type_dispatcher(device_view.type(), check_nan{}, device_view, i);
      if (not count_nulls and (is_null or (nan_is_null and is_nan))) { return false; }
      if (i == 0) { return true; }
      if (count_nulls and nan_is_null and (is_nan or is_null)) {
        auto const prev_is_nan =
          should_check_nan and
          cudf::type_dispatcher(device_view.type(), check_nan{}, device_view, i - 1);
        return not(prev_is_nan or device_view.is_null(i - 1));
      }
      return not comp(i, i - 1);
    });
}
}  // namespace detail

cudf::size_type unique_count(column_view const& input,
                             null_policy null_handling,
                             nan_policy nan_handling,
                             cuda::stream_ref stream)
{
  CUDF_FUNC_RANGE();
  return detail::unique_count(input, null_handling, nan_handling, stream);
}

}  // namespace cudf
