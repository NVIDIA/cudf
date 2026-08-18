/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "common_utils.cuh"

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/aggregation/aggregation.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/row_operator/equality.cuh>
#include <cudf/detail/utilities/device_operators.cuh>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/exec_policy.hpp>

#include <cuda/iterator>
#include <cuda/std/functional>
#include <cuda/std/iterator>
#include <cuda/std/limits>
#include <cuda/std/utility>
#include <cuda/stream>
#include <thrust/scan.h>
#include <thrust/transform.h>

namespace cudf {
namespace groupby {
namespace detail {
namespace {

template <bool forward, typename permuted_equal_t, typename value_resolver>
struct unique_identifier {
  unique_identifier(size_type const* labels,
                    size_type const* offsets,
                    permuted_equal_t permuted_equal,
                    value_resolver resolver)
    : _labels(labels), _offsets(offsets), _permuted_equal(permuted_equal), _resolver(resolver)
  {
  }

  auto __device__ operator()(size_type row_index) const noexcept
  {
    auto const group_start = _offsets[_labels[row_index]];
    if constexpr (forward) {
      // First value of equal values is 1.
      return _resolver(row_index == group_start || !_permuted_equal(row_index, row_index - 1),
                       row_index - group_start);
    } else {
      auto const group_end = _offsets[_labels[row_index] + 1];
      // Last value of equal values is 1.
      return _resolver(row_index + 1 == group_end || !_permuted_equal(row_index, row_index + 1),
                       row_index - group_start);
    }
  }

 private:
  size_type const* _labels;
  size_type const* _offsets;
  permuted_equal_t _permuted_equal;
  value_resolver _resolver;
};

/**
 * @brief generate grouped row ranks or dense ranks using a row comparison then scan the results
 *
 * @tparam forward true if the rank scan computation should use forward iterator traversal (default)
 * else reverse iterator traversal
 * @tparam value_resolver flag value resolver function with boolean first and row number arguments
 * @tparam scan_operator scan function ran on the flag values
 * @param grouped_values input column to generate ranks for
 * @param value_order column of type INT32 that contains the order of the values in the
 * grouped_values column
 * @param group_labels ID of group that the corresponding value belongs to
 * @param group_offsets group index offsets with group ID indices
 * @param resolver flag value resolver
 * @param scan_op scan operation ran on the flag results
 * @param has_nulls true if nulls are included in the `grouped_values` column
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @return std::unique_ptr<column> rank values
 */
template <bool forward, typename value_resolver, typename scan_operator>
std::unique_ptr<column> rank_generator(column_view const& grouped_values,
                                       column_view const& value_order,
                                       device_span<size_type const> group_labels,
                                       device_span<size_type const> group_offsets,
                                       value_resolver resolver,
                                       scan_operator scan_op,
                                       bool has_nulls,
                                       cuda::stream_ref stream,
                                       rmm::device_async_resource_ref mr)
{
  auto const grouped_values_view = table_view{{grouped_values}};
  auto const comparator = cudf::detail::row::equality::self_comparator{grouped_values_view, stream};

  auto ranks = make_fixed_width_column(
    data_type{type_to_id<size_type>()}, grouped_values.size(), mask_state::UNALLOCATED, stream, mr);
  auto mutable_ranks = ranks->mutable_view();

  auto const comparator_helper = [&](auto const d_equal) {
    auto const permuted_equal =
      permuted_row_equality_comparator(d_equal, value_order.begin<size_type>());

    thrust::transform(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                      cuda::counting_iterator<size_type>(0),
                      cuda::counting_iterator<size_type>(grouped_values.size()),
                      mutable_ranks.begin<size_type>(),
                      unique_identifier<forward, decltype(permuted_equal), value_resolver>(
                        group_labels.data(), group_offsets.data(), permuted_equal, resolver));
  };

  if (cudf::detail::has_nested_columns(grouped_values_view)) {
    auto const d_equal =
      comparator.equal_to<true>(cudf::nullate::DYNAMIC{has_nulls}, null_equality::EQUAL);
    comparator_helper(d_equal);
  } else {
    auto const d_equal =
      comparator.equal_to<false>(cudf::nullate::DYNAMIC{has_nulls}, null_equality::EQUAL);
    comparator_helper(d_equal);
  }

  auto [group_labels_begin, mutable_rank_begin] = [&]() {
    if constexpr (forward) {
      return cuda::std::pair{group_labels.begin(), mutable_ranks.begin<size_type>()};
    } else {
      return cuda::std::pair{cuda::std::reverse_iterator(group_labels.end()),
                             cuda::std::reverse_iterator(mutable_ranks.end<size_type>())};
    }
  }();
  thrust::inclusive_scan_by_key(
    rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
    group_labels_begin,
    group_labels_begin + group_labels.size(),
    mutable_rank_begin,
    mutable_rank_begin,
    cuda::std::equal_to{},
    scan_op);
  return ranks;
}
}  // namespace

std::unique_ptr<column> min_rank_scan(column_view const& grouped_values,
                                      column_view const& value_order,
                                      device_span<size_type const> group_labels,
                                      device_span<size_type const> group_offsets,
                                      cuda::stream_ref stream,
                                      rmm::device_async_resource_ref mr)
{
  return rank_generator<true>(
    grouped_values,
    value_order,
    group_labels,
    group_offsets,
    [] __device__(bool unequal, auto row_index_in_group) {
      return unequal ? row_index_in_group + 1 : 0;
    },
    DeviceMax{},
    has_nested_nulls(table_view{{grouped_values}}),
    stream,
    mr);
}

std::unique_ptr<column> max_rank_scan(column_view const& grouped_values,
                                      column_view const& value_order,
                                      device_span<size_type const> group_labels,
                                      device_span<size_type const> group_offsets,
                                      cuda::stream_ref stream,
                                      rmm::device_async_resource_ref mr)
{
  return rank_generator<false>(
    grouped_values,
    value_order,
    group_labels,
    group_offsets,
    [] __device__(bool unequal, auto row_index_in_group) {
      return unequal ? row_index_in_group + 1 : cuda::std::numeric_limits<size_type>::max();
    },
    DeviceMin{},
    has_nested_nulls(table_view{{grouped_values}}),
    stream,
    mr);
}

std::unique_ptr<column> first_rank_scan(column_view const& grouped_values,
                                        column_view const&,
                                        device_span<size_type const> group_labels,
                                        device_span<size_type const> group_offsets,
                                        cuda::stream_ref stream,
                                        rmm::device_async_resource_ref mr)
{
  auto ranks = make_fixed_width_column(
    data_type{type_to_id<size_type>()}, group_labels.size(), mask_state::UNALLOCATED, stream, mr);
  auto mutable_ranks = ranks->mutable_view();
  thrust::transform(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                    cuda::counting_iterator<size_type>(0),
                    cuda::counting_iterator<size_type>(group_labels.size()),
                    mutable_ranks.begin<size_type>(),
                    [labels  = group_labels.begin(),
                     offsets = group_offsets.begin()] __device__(size_type row_index) {
                      auto group_start = offsets[labels[row_index]];
                      return row_index - group_start + 1;
                    });
  return ranks;
}

std::unique_ptr<column> average_rank_scan(column_view const& grouped_values,
                                          column_view const& value_order,
                                          device_span<size_type const> group_labels,
                                          device_span<size_type const> group_offsets,
                                          cuda::stream_ref stream,
                                          rmm::device_async_resource_ref mr)
{
  auto max_rank = max_rank_scan(grouped_values,
                                value_order,
                                group_labels,
                                group_offsets,
                                stream,
                                cudf::get_current_device_resource_ref());
  auto min_rank = min_rank_scan(grouped_values,
                                value_order,
                                group_labels,
                                group_offsets,
                                stream,
                                cudf::get_current_device_resource_ref());
  auto ranks    = make_fixed_width_column(
    data_type{type_to_id<double>()}, group_labels.size(), mask_state::UNALLOCATED, stream, mr);
  auto mutable_ranks = ranks->mutable_view();
  thrust::transform(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                    max_rank->view().begin<size_type>(),
                    max_rank->view().end<size_type>(),
                    min_rank->view().begin<size_type>(),
                    mutable_ranks.begin<double>(),
                    [] __device__(auto max_rank, auto min_rank) -> double {
                      return min_rank + (max_rank - min_rank) / 2.0;
                    });
  return ranks;
}

std::unique_ptr<column> dense_rank_scan(column_view const& grouped_values,
                                        column_view const& value_order,
                                        device_span<size_type const> group_labels,
                                        device_span<size_type const> group_offsets,
                                        cuda::stream_ref stream,
                                        rmm::device_async_resource_ref mr)
{
  return rank_generator<true>(
    grouped_values,
    value_order,
    group_labels,
    group_offsets,
    [] __device__(bool const unequal, size_type const) { return unequal ? 1 : 0; },
    DeviceSum{},
    has_nested_nulls(table_view{{grouped_values}}),
    stream,
    mr);
}

std::unique_ptr<column> group_rank_to_percentage(rank_method const method,
                                                 rank_percentage const percentage,
                                                 column_view const& rank,
                                                 column_view const& count,
                                                 device_span<size_type const> group_labels,
                                                 device_span<size_type const> group_offsets,
                                                 cuda::stream_ref stream,
                                                 rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(percentage != rank_percentage::NONE, "Percentage cannot be NONE");
  auto ranks = make_fixed_width_column(
    data_type{type_to_id<double>()}, group_labels.size(), mask_state::UNALLOCATED, stream, mr);
  auto mutable_ranks = ranks->mutable_view();

  auto one_normalized = [] __device__(auto const rank, auto const group_size) {
    return group_size == 1 ? 0.0 : ((rank - 1.0) / (group_size - 1));
  };
  if (method == rank_method::DENSE) {
    thrust::transform(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                      cuda::counting_iterator<size_type>(0),
                      cuda::counting_iterator<size_type>(group_labels.size()),
                      mutable_ranks.begin<double>(),
                      [percentage,
                       one_normalized,
                       is_double = rank.type().id() == type_id::FLOAT64,
                       dcount    = count.begin<size_type>(),
                       labels    = group_labels.begin(),
                       offsets   = group_offsets.begin(),
                       d_rank    = rank.begin<double>(),
                       s_rank = rank.begin<size_type>()] __device__(size_type row_index) -> double {
                        double const r   = is_double ? d_rank[row_index] : s_rank[row_index];
                        auto const count = dcount[labels[row_index]];
                        size_type const last_rank_index = offsets[labels[row_index]] + count - 1;
                        auto const last_rank = last_rank_index < 0 ? 1 : s_rank[last_rank_index];
                        return percentage == rank_percentage::ZERO_NORMALIZED
                                 ? r / last_rank
                                 : one_normalized(r, last_rank);
                      });
  } else {
    thrust::transform(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                      cuda::counting_iterator<size_type>(0),
                      cuda::counting_iterator<size_type>(group_labels.size()),
                      mutable_ranks.begin<double>(),
                      [percentage,
                       one_normalized,
                       is_double = rank.type().id() == type_id::FLOAT64,
                       dcount    = count.begin<size_type>(),
                       labels    = group_labels.begin(),
                       d_rank    = rank.begin<double>(),
                       s_rank = rank.begin<size_type>()] __device__(size_type row_index) -> double {
                        double const r   = is_double ? d_rank[row_index] : s_rank[row_index];
                        auto const count = dcount[labels[row_index]];
                        return percentage == rank_percentage::ZERO_NORMALIZED
                                 ? r / count
                                 : one_normalized(r, count);
                      });
  }

  ranks->set_null_count(0);
  return ranks;
}

}  // namespace detail
}  // namespace groupby
}  // namespace cudf
