/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/aggregation/aggregation.hpp>
#include <cudf/detail/row_operator/equality.cuh>
#include <cudf/detail/scan.hpp>
#include <cudf/detail/structs/utilities.hpp>
#include <cudf/detail/utilities/device_operators.cuh>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/exec_policy.hpp>

#include <cuda/iterator>
#include <cuda/stream>
#include <thrust/scan.h>
#include <thrust/transform.h>

namespace cudf {
namespace detail {
namespace {

template <typename device_comparator_type, typename value_resolver>
struct rank_equality_functor {
  rank_equality_functor(device_comparator_type comparator, value_resolver resolver)
    : _comparator(comparator), _resolver(resolver)
  {
  }

  auto __device__ operator()(size_type row_index) const noexcept
  {
    return _resolver(row_index == 0 || !_comparator(row_index, row_index - 1), row_index);
  }

 private:
  device_comparator_type _comparator;
  value_resolver _resolver;
};

/**
 * @brief generate row ranks or dense ranks using a row comparison then scan the results
 *
 * @tparam value_resolver flag value resolver with boolean first and row number arguments
 * @tparam scan_operator scan function ran on the flag values
 * @param order_by input column to generate ranks for
 * @param resolver flag value resolver
 * @param scan_op scan operation ran on the flag results
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @return std::unique_ptr<column> rank values
 */
template <typename value_resolver, typename scan_operator>
std::unique_ptr<column> rank_generator(column_view const& order_by,
                                       value_resolver resolver,
                                       scan_operator scan_op,
                                       cuda::stream_ref stream,
                                       rmm::device_async_resource_ref mr)
{
  auto const order_by_tview = table_view{{order_by}};
  auto comp                 = cudf::detail::row::equality::self_comparator(order_by_tview, stream);

  auto ranks = make_fixed_width_column(
    data_type{type_to_id<size_type>()}, order_by.size(), mask_state::UNALLOCATED, stream, mr);
  auto mutable_ranks = ranks->mutable_view();

  auto const comparator_helper = [&](auto const device_comparator) {
    thrust::transform(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                      cuda::counting_iterator<size_type>(0),
                      cuda::counting_iterator<size_type>(order_by.size()),
                      mutable_ranks.begin<size_type>(),
                      rank_equality_functor<decltype(device_comparator), value_resolver>(
                        device_comparator, resolver));
  };

  if (cudf::detail::has_nested_columns(order_by_tview)) {
    auto const device_comparator =
      comp.equal_to<true>(nullate::DYNAMIC{has_nested_nulls(table_view({order_by}))});
    comparator_helper(device_comparator);
  } else {
    auto const device_comparator =
      comp.equal_to<false>(nullate::DYNAMIC{has_nested_nulls(table_view({order_by}))});
    comparator_helper(device_comparator);
  }

  thrust::inclusive_scan(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                         mutable_ranks.begin<size_type>(),
                         mutable_ranks.end<size_type>(),
                         mutable_ranks.begin<size_type>(),
                         scan_op);
  return ranks;
}

}  // namespace

std::unique_ptr<column> inclusive_dense_rank_scan(column_view const& order_by,
                                                  cuda::stream_ref stream,
                                                  rmm::device_async_resource_ref mr)
{
  return rank_generator(
    order_by,
    [] __device__(bool const unequal, size_type const) { return unequal ? 1 : 0; },
    DeviceSum{},
    stream,
    mr);
}

std::unique_ptr<column> inclusive_rank_scan(column_view const& order_by,
                                            cuda::stream_ref stream,
                                            rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(!cudf::structs::detail::is_or_has_nested_lists(order_by),
               "Unsupported list type in rank scan.");
  return rank_generator(
    order_by,
    [] __device__(bool unequal, auto row_index) { return unequal ? row_index + 1 : 0; },
    DeviceMax{},
    stream,
    mr);
}

std::unique_ptr<column> inclusive_one_normalized_percent_rank_scan(
  column_view const& order_by, cuda::stream_ref stream, rmm::device_async_resource_ref mr)
{
  auto const rank_column =
    inclusive_rank_scan(order_by, stream, cudf::get_current_device_resource_ref());
  auto const rank_view = rank_column->view();

  // Result type for min 0-index percent rank is independent of input type.
  using result_type        = double;
  auto percent_rank_result = cudf::make_fixed_width_column(
    data_type{type_to_id<result_type>()}, rank_view.size(), mask_state::UNALLOCATED, stream, mr);

  thrust::transform(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                    rank_view.begin<size_type>(),
                    rank_view.end<size_type>(),
                    percent_rank_result->mutable_view().begin<result_type>(),
                    [n_rows = rank_view.size()] __device__(auto const rank) {
                      return n_rows == 1 ? 0.0 : ((rank - 1.0) / (n_rows - 1));
                    });
  return percent_rank_result;
}

}  // namespace detail
}  // namespace cudf
