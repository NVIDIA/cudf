/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/detail/aggregation/aggregation.cuh>
#include <cudf/detail/aggregation/aggregation.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/exec_policy.hpp>

#include <thrust/fill.h>

namespace cudf {
namespace detail {
namespace {

/**
 * @brief Dispatched functor to initialize a column with the identity of an
 * aggregation operation.
 *
 * Given a type `T` and `aggregation kind k`, determines and sets the value of
 * each element of the passed column to the appropriate initial value for the
 * aggregation.
 *
 * The initial values set as per aggregation are:
 * SUM: 0
 * COUNT_VALID: 0 and VALID
 * COUNT_ALL:   0 and VALID
 * MIN: Max value of type `T`
 * MAX: Min value of type `T`
 * ARGMAX: `ARGMAX_SENTINEL`
 * ARGMIN: `ARGMIN_SENTINEL`
 *
 * Only works on columns of fixed-width types.
 */
struct identity_initializer {
 private:
  template <typename T, aggregation::Kind k>
  static constexpr bool is_supported()
  {
    return is_identity_supported<T, k>() or
           (k == aggregation::SUM_OVERFLOW and std::is_same_v<T, cudf::struct_view>);
  }

 public:
  template <typename T, aggregation::Kind k>
  void operator()(mutable_column_view const& col, rmm::cuda_stream_view stream)
    requires(is_supported<T, k>())
  {
    if constexpr (k == aggregation::SUM_OVERFLOW) {
      // SUM_OVERFLOW uses a struct with sum and overflow children
      auto sum_col      = col.child(0);
      auto overflow_col = col.child(1);

      // Initialize sum column using standard SUM aggregation dispatch
      dispatch_type_and_aggregation(
        sum_col.type(), aggregation::SUM, identity_initializer{}, sum_col, stream);
      thrust::uninitialized_fill_n(
        rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
        overflow_col.begin<bool>(),
        col.size(),
        false);
    } else if constexpr (std::is_same_v<T, cudf::struct_view>) {
      // This should only happen for SUM_OVERFLOW, but handle it just in case
      CUDF_FAIL("Struct columns are only supported for SUM_OVERFLOW aggregation");
    } else {
      using DeviceType = device_storage_type_t<T>;
      thrust::fill(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                   col.begin<DeviceType>(),
                   col.end<DeviceType>(),
                   get_identity<DeviceType, k>());
    }
  }

  template <typename T, aggregation::Kind k>
  void operator()(mutable_column_view const& col, rmm::cuda_stream_view stream)
    requires(not is_supported<T, k>())
  {
    CUDF_FAIL("Unsupported aggregation for initializing values");
  }
};
}  // namespace

void initialize_with_identity(mutable_table_view const& table,
                              host_span<cudf::aggregation::Kind const> aggs,
                              rmm::cuda_stream_view stream)
{
  // TODO: Initialize all the columns in a single kernel instead of invoking one
  // kernel per column
  for (size_type i = 0; i < table.num_columns(); ++i) {
    auto col = table.column(i);
    dispatch_type_and_aggregation(col.type(), aggs[i], identity_initializer{}, col, stream);
  }
}

}  // namespace detail
}  // namespace cudf
