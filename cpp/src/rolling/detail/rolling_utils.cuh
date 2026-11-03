/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/detail/iterator.cuh>
#include <cudf/detail/rolling.hpp>
#include <cudf/types.hpp>

#include <cuda/functional>
#include <cuda/std/type_traits>

namespace cudf {
namespace detail::rolling {

/**
 * @brief Information about group bounds of the current row's group.
 */
struct group_info {
  size_type const start;  ///< The start of the group
  size_type const end;    ///< The end of the group
  [[nodiscard]] __device__ constexpr inline bool is_null(size_type) const noexcept { return false; }
  [[nodiscard]] __device__ constexpr inline size_type group_start() const noexcept { return start; }
  [[nodiscard]] __device__ constexpr inline size_type group_end() const noexcept { return end; }
  [[nodiscard]] __device__ constexpr inline size_type non_null_start() const noexcept
  {
    return start;
  }
  [[nodiscard]] __device__ constexpr inline size_type non_null_end() const noexcept { return end; }
  [[nodiscard]] __device__ constexpr inline size_type null_start() const noexcept { return start; }
  [[nodiscard]] __device__ constexpr inline size_type null_end() const noexcept { return start; }
};

/**
 * @brief Information about group bounds of the current row's group.
 *
 * The groups can contain nulls.
 */
struct group_info_with_nulls {
  size_type const group_start_;  ///< The start of the group
  size_type const group_end_;    ///< The end of the group
  size_type const null_count_;   ///< The null count of the group
  bool const nulls_at_start_;    ///< Do nulls sort at the start or the end of the group?
  [[nodiscard]] __device__ constexpr inline bool is_null(size_type i) const noexcept
  {
    return i < non_null_start() || i >= non_null_end();
  }
  [[nodiscard]] __device__ constexpr inline size_type group_start() const noexcept
  {
    return group_start_;
  }
  [[nodiscard]] __device__ constexpr inline size_type group_end() const noexcept
  {
    return group_end_;
  }
  [[nodiscard]] __device__ constexpr inline size_type non_null_start() const noexcept
  {
    return nulls_at_start_ ? group_start_ + null_count_ : group_start_;
  }
  [[nodiscard]] constexpr __device__ size_type inline non_null_end() const noexcept
  {
    return nulls_at_start_ ? group_end_ : group_end_ - null_count_;
  }
  [[nodiscard]] constexpr __device__ size_type inline null_start() const noexcept
  {
    return nulls_at_start_ ? group_start_ : group_end_ - null_count_;
  }
  [[nodiscard]] constexpr __device__ size_type inline null_end() const noexcept
  {
    return nulls_at_start_ ? group_start_ + null_count_ : group_end_;
  }
};

/**
 * @brief A group descriptor for an ungrouped rolling window.
 *
 * @param num_rows The number of rows to be rolled over.
 *
 * @note This is used for uniformity of interface between grouped and ungrouped iterator
 * construction.
 */
struct ungrouped {
  cudf::size_type const num_rows;

  /**
   * @brief Return information about the current row.
   *
   * @param i The row
   * @returns `range_group_info` with the information about the row.
   */
  [[nodiscard]] __device__ constexpr inline group_info row_info(size_type i) const noexcept
  {
    return {0, num_rows};
  }
};

/**
 * @brief A group descriptor for a grouped rolling window.
 *
 * @param labels The group labels, mapping from input rows to group.
 * @param offsets The group offsets providing the endpoints of each group.
 *
 * @note This is used for uniformity of interface between grouped and ungrouped iterator
 * construction.
 */
struct grouped {
  // Taking raw pointers here to avoid stealing two registers for the sizes which are never needed.
  cudf::size_type const* labels;
  cudf::size_type const* offsets;

  /**
   * @copydoc ungrouped::row_info
   */
  [[nodiscard]] __device__ constexpr inline group_info row_info(size_type i) const noexcept
  {
    auto const label       = labels[i];
    auto const group_start = offsets[label];
    auto const group_end   = offsets[label + 1];
    return {group_start, group_end};
  }
};

/**
 * @brief A group descriptor for an ungrouped rolling window with nulls
 *
 * @param nulls_at_start Are the nulls at the start or end?
 * @param num_rows The number of rows to be rolled over.
 * @param null_count The number of nulls.
 *
 * @note This is used for uniformity of interface between grouped and ungrouped
 * iterator construction.
 */
struct ungrouped_with_nulls {
  bool const nulls_at_start;
  cudf::size_type const num_rows;
  cudf::size_type const null_count;

  /**
   * @copydoc ungrouped::row_info
   */
  [[nodiscard]] __device__ constexpr inline group_info_with_nulls row_info(
    size_type i) const noexcept
  {
    return {0, num_rows, null_count, nulls_at_start};
  }
};

/**
 * @brief A group descriptor for a grouped rolling window with nulls
 *
 * @param nulls_at_start Are the nulls at the start of each group?
 * @param labels The group labels, mapping from input rows to group.
 * @param offsets The group offsets providing the endpoints of each group.
 * @param null_counts The null counts per group.
 * @param orderby The orderby column, sorted groupwise.
 *
 * @note This is used for uniformity of interface between grouped and ungrouped
 * iterator construction.
 */
struct grouped_with_nulls {
  bool const nulls_at_start;
  // Taking raw pointers here to avoid stealing three registers for the sizes which are never
  // needed.
  cudf::size_type const* labels;
  cudf::size_type const* offsets;
  cudf::size_type const* null_counts;

  /**
   * @copydoc ungrouped::row_info
   */
  [[nodiscard]] __device__ constexpr inline group_info_with_nulls row_info(
    size_type i) const noexcept
  {
    auto const label       = labels[i];
    auto const null_count  = null_counts[label];
    auto const group_start = offsets[label];
    auto const group_end   = offsets[label + 1];
    return {group_start, group_end, null_count, nulls_at_start};
  }
};

template <direction Direction, typename Grouping>
struct fixed_window_clamper {
  Grouping groups;
  cudf::size_type delta;
  static_assert(cuda::std::is_same_v<Grouping, ungrouped> ||
                  cuda::std::is_same_v<Grouping, grouped>,
                "Invalid grouping descriptor");

  [[nodiscard]] __device__ constexpr cudf::size_type operator()(cudf::size_type i) const
  {
    group_info const row_info = groups.row_info(i);
    if constexpr (Direction == direction::PRECEDING) {
      return cuda::std::min(i + 1 - row_info.group_start(),
                            cuda::std::max(delta, i + 1 - row_info.group_end()));
    } else {
      return cuda::std::max(row_info.group_start() - i - 1,
                            cuda::std::min(delta, row_info.group_end() - i - 1));
    }
  }
};

/**
 * @brief Construct a clamped counting iterator for a row-based window offset
 *
 * @tparam Direction the direction of the window `PRECEDING` or `FOLLOWING`.
 * @tparam Grouping the group specification.
 * @param delta the window offset.
 * @param grouper the grouping object.
 *
 * @return An iterator suitable for passing to `cudf::detail::rolling_window`
 */
template <direction Direction, typename Grouping>
[[nodiscard]] auto inline make_clamped_window_iterator(cudf::size_type delta, Grouping grouper)
{
  return cudf::detail::make_counting_transform_iterator(
    cudf::size_type{0}, fixed_window_clamper<Direction, Grouping>{grouper, delta});
}
}  // namespace detail::rolling
}  // namespace cudf
