/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "detail/range_rolling.hpp"
#include "detail/rolling.hpp"
#include "detail/rolling_utils.cuh"

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/detail/groupby/sort_helper.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/rolling.hpp>
#include <cudf/rolling.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/span.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/exec_policy.hpp>
#include <rmm/resource_ref.hpp>

#include <cuda/functional>
#include <cuda/iterator>
#include <cuda/std/tuple>
#include <cuda/stream>
#include <thrust/copy.h>

#include <memory>
#include <optional>
#include <utility>
#include <variant>
#include <vector>

namespace CUDF_EXPORT cudf {
namespace detail {
namespace {

/**
 * @brief Computes preceding and following offsets from one group lookup
 */
template <typename Grouping>
struct unbounded_distance_fn {
  Grouping const groups;

  [[nodiscard]] __device__ cuda::std::tuple<size_type, size_type> operator()(
    size_type i) const noexcept
  {
    auto const row_info = groups.row_info(i);
    return cuda::std::tuple<size_type, size_type>{i - row_info.group_start() + 1,
                                                  row_info.group_end() - i - 1};
  }
};

template <typename Grouping>
struct directional_unbounded_distance_fn {
  Grouping const groups;
  rolling::direction const direction;

  [[nodiscard]] __device__ size_type operator()(size_type i) const noexcept
  {
    auto const row_info = groups.row_info(i);
    return direction == rolling::direction::PRECEDING ? i - row_info.group_start() + 1
                                                      : row_info.group_end() - i - 1;
  }
};

/**
 * @brief Computes preceding and following offsets from different groupings.
 */
template <typename Preceding, typename Following>
struct mixed_unbounded_distance_fn {
  directional_unbounded_distance_fn<Preceding> const preceding_fn;
  directional_unbounded_distance_fn<Following> const following_fn;

  [[nodiscard]] __device__ cuda::std::tuple<size_type, size_type> operator()(
    size_type i) const noexcept
  {
    return cuda::std::tuple<size_type, size_type>{preceding_fn(i), following_fn(i)};
  }
};

/**
 * @brief Selects grouping and writes preceding and following offsets.
 */
template <typename GroupHelper>
void select_and_write_offsets(range_window_type const& preceding,
                              range_window_type const& following,
                              std::optional<rolling::grouped> const& peers,
                              std::optional<GroupHelper>& group_helper,
                              mutable_column_view preceding_view,
                              mutable_column_view following_view,
                              size_type num_rows,
                              cuda::stream_ref stream)
{
  // Write the offsets to the output columns
  auto const write_offsets = [&](auto offset_fn) {
    auto const src_iter = cudf::detail::make_counting_transform_iterator(
      size_type{0}, cuda::proclaim_return_type<cuda::std::tuple<size_type, size_type>>(offset_fn));
    thrust::copy_n(
      rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
      src_iter,
      num_rows,
      cuda::zip_iterator(preceding_view.begin<size_type>(), following_view.begin<size_type>()));
  };

  // Write offsets for the current row case
  if (std::holds_alternative<current_row>(preceding) &&
      std::holds_alternative<current_row>(following)) {
    write_offsets(unbounded_distance_fn<rolling::grouped>{peers.value()});
    // Write offsets for the unbounded case
  } else if (std::holds_alternative<unbounded>(preceding) &&
             std::holds_alternative<unbounded>(following)) {
    if (group_helper.has_value()) {
      write_offsets(unbounded_distance_fn<rolling::grouped>{
        {group_helper->group_labels(stream).data(), group_helper->group_offsets(stream).data()}});
    } else {
      write_offsets(unbounded_distance_fn<rolling::ungrouped>{{num_rows}});
    }
  } else {
    // Select groupings for preceding and following windows
    auto const select_grouping =
      [&](range_window_type const& window) -> std::variant<rolling::grouped, rolling::ungrouped> {
      if (std::holds_alternative<current_row>(window)) {
        return peers.value();
      } else if (group_helper.has_value()) {
        return rolling::grouped{group_helper->group_labels(stream).data(),
                                group_helper->group_offsets(stream).data()};
      } else {
        return rolling::ungrouped{num_rows};
      }
    };

    auto preceding_grouping = select_grouping(preceding);
    auto following_grouping = select_grouping(following);
    std::visit(
      [&](auto preceding_grouping, auto following_grouping) {
        write_offsets(
          mixed_unbounded_distance_fn<decltype(preceding_grouping), decltype(following_grouping)>{
            {preceding_grouping, rolling::direction::PRECEDING},
            {following_grouping, rolling::direction::FOLLOWING}});
      },
      preceding_grouping,
      following_grouping);
  }
}

}  // namespace

std::pair<std::unique_ptr<column>, std::unique_ptr<column>> make_range_windows(
  table_view const& group_keys,
  table_view const& orderby,
  host_span<order const> orders,
  host_span<null_order const> null_orders,
  range_window_type preceding,
  range_window_type following,
  cuda::stream_ref stream,
  rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  CUDF_EXPECTS(orderby.num_columns() > 0, "orderby must be non-empty");
  CUDF_EXPECTS(group_keys.num_columns() == 0 || group_keys.num_rows() == orderby.num_rows(),
               "Size mismatch between group_keys and orderby table.");
  CUDF_EXPECTS(orderby.num_columns() == static_cast<size_type>(orders.size()),
               "ORDER BY column count must match order vector");
  CUDF_EXPECTS(orderby.num_columns() == static_cast<size_type>(null_orders.size()),
               "ORDER BY column count must match null-order vector");

  if (orderby.num_columns() == 1) {
    return detail::make_range_windows(group_keys,
                                      orderby.column(0),
                                      orders.front(),
                                      null_orders.front(),
                                      preceding,
                                      following,
                                      stream,
                                      mr);
  }

  auto const is_peer_bound = [](range_window_type const& window) {
    return std::holds_alternative<unbounded>(window) || std::holds_alternative<current_row>(window);
  };
  CUDF_EXPECTS(is_peer_bound(preceding) && is_peer_bound(following),
               "Multi-column RANGE windows support only UNBOUNDED and CURRENT ROW bounds");

  using sort_helper = cudf::groupby::detail::sort::sort_groupby_helper;

  std::optional<sort_helper> peer_helper;
  std::optional<rolling::grouped> peers;
  if (std::holds_alternative<current_row>(preceding) ||
      std::holds_alternative<current_row>(following)) {
    std::vector<column_view> peer_keys(group_keys.begin(), group_keys.end());
    peer_keys.insert(peer_keys.end(), orderby.begin(), orderby.end());
    peer_helper.emplace(
      table_view{peer_keys}, null_policy::INCLUDE, sorted::YES, std::vector<null_order>{});
    peers = rolling::grouped{peer_helper->group_labels(stream).data(),
                             peer_helper->group_offsets(stream).data()};
  }

  std::optional<sort_helper> group_helper;
  if (group_keys.num_columns() > 0 && (std::holds_alternative<unbounded>(preceding) ||
                                       std::holds_alternative<unbounded>(following))) {
    group_helper.emplace(group_keys, null_policy::INCLUDE, sorted::YES, std::vector<null_order>{});
  }

  auto const num_rows   = orderby.num_rows();
  auto preceding_result = make_numeric_column(
    data_type{type_to_id<size_type>()}, num_rows, mask_state::UNALLOCATED, stream, mr);
  auto following_result = make_numeric_column(
    data_type{type_to_id<size_type>()}, num_rows, mask_state::UNALLOCATED, stream, mr);

  select_and_write_offsets(preceding,
                           following,
                           peers,
                           group_helper,
                           preceding_result->mutable_view(),
                           following_result->mutable_view(),
                           num_rows,
                           stream);

  return std::pair{std::move(preceding_result), std::move(following_result)};
}

}  // namespace detail
}  // namespace CUDF_EXPORT cudf
