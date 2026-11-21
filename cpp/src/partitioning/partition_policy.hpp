// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cudf/table/table_view.hpp>
#include <cudf/utilities/traits.hpp>

#include <algorithm>
#include <optional>

/**
 * @brief Host-side policy for choosing partition execution plans
 */
namespace cudf::detail::partition_policy {
/**
 * @brief Strategies for assigning input rows to output positions
 */
enum class algorithm {
  saved,      ///< Save per-row tile ranks during counting and reuse them when writing output.
  recompute,  ///< Count one tile per block and recompute its row ranks when writing output.
  super2,     ///< Recompute row ranks with two tiles per block to reduce the block-count array.
  reserve,    ///< Reserve output ranges per tile and partition using atomic partition cursors.
  binary      ///< Use a two-way partition of row indices for exactly two partitions.
};

/**
 * @brief Partition algorithm and output-materialization choices
 */
struct partition_plan {
  algorithm method;  ///< Strategy used to compute each row's destination.

  /**
   * @brief Whether to write fixed-width column values directly to their output positions
   *
   * A direct plan avoids an intermediate row map. Otherwise, the map engine emits a gather map
   * that a separate gather operation uses to materialize the output table.
   */
  bool direct;

  /**
   * @brief Whether to stage tile destinations in shared memory before writing output
   *
   * Reordering groups writes by destination within each tile to improve output locality. It is
   * supported by the `recompute` and `super2` methods and does not change partition membership.
   */
  bool reordered;
};

/**
 * @brief Select a partition plan from table metadata and the partition count
 *
 * Uses the row count, column types and widths, and cached null counts to choose the mapping
 * algorithm, direct materialization, and write reordering. This helper does not inspect column
 * data or partition-map values and launches no device work. Execution backends are dispatched
 * separately; callers handle trivial inputs and retain their fallback when no plan is selected.
 *
 * @param input Table whose metadata determines the eligible execution plans
 * @param partitions Number of destination partitions
 * @return The selected plan, or `std::nullopt` when the row/partition counts fall outside the
 * optimized policy's limits or no plan applies to the schema
 */
inline std::optional<partition_plan> select_partition_plan(table_view const& input,
                                                           size_type partitions)
{
  namespace plans = cudf::detail::partition_policy;
  if (partitions <= 1 || partitions > 256 || input.num_rows() < 1'024) { return std::nullopt; }
  auto const fixed = std::all_of(
    input.begin(), input.end(), [](auto const& col) { return is_fixed_width(col.type()); });
  auto const nullable =
    std::any_of(input.begin(), input.end(), [](auto const& col) { return col.null_count() > 0; });
  std::size_t bytes_per_row = 0;
  if (fixed) {
    for (auto const& col : input) {
      bytes_per_row += cudf::size_of(col.type());
    }
  }
  constexpr size_type large_input = 1'000'000;
  bool const large                = input.num_rows() >= large_input;
  bool const dense_nulls =
    fixed && nullable && large && std::all_of(input.begin(), input.end(), [&](auto const& col) {
      return int64_t{col.null_count()} * 4 >= int64_t{input.num_rows()} * 3;
    });
  // With few valid rows, skip payload reads and reorder writes without a row map.
  if (dense_nulls && input.num_columns() > 2) {
    return partition_plan{plans::algorithm::recompute, true, true};
  }
  if (partitions == 2) {
    bool const direct = fixed && input.num_columns() <= 2 &&
                        (input.num_columns() == 1 || !nullable || !large || bytes_per_row >= 16);
    partition_plan plan{plans::algorithm::binary, direct, false};
    if (fixed && !nullable && large) {
      plan.direct = true;
      if (input.num_columns() > 2) {
        // Reordered output pays off for a few wide columns and removes saved ranks.
        bool const reorder = input.num_columns() <= 4 && bytes_per_row >= 32;
        plan.method        = reorder ? plans::algorithm::recompute : plans::algorithm::saved;
        plan.reordered     = reorder;
      } else if (bytes_per_row > 4) {
        plan.method    = plans::algorithm::recompute;
        plan.reordered = input.num_columns() == 1 && bytes_per_row == 16;
      }
    }
    return plan;
  }
  if (fixed && large) {
    if (input.num_columns() == 1 ||
        (input.num_columns() == 2 && (!nullable || bytes_per_row >= 16))) {
      // Saved ranks avoid recomputing sparse nullable two-column tables; the
      // direct sink still removes the larger N-entry gather map.
      auto const method = nullable && input.num_columns() == 2 && !dense_nulls
                            ? plans::algorithm::saved
                          : partitions >= 16 ? plans::algorithm::reserve
                                             : plans::algorithm::recompute;
      return partition_plan{method, true, false};
    }
    if (!nullable) {
      // A one-tile prefix is more robust than supertiles for narrow multi-column
      // output with uniform partition maps, while retaining direct-output savings.
      return partition_plan{plans::algorithm::recompute, true, true};
    }
  }
  return std::nullopt;
}

/**
 * @brief Determine whether partitioning can use an identity-copy or all-null-output path
 *
 * @param rows Number of input rows
 * @param partitions Number of destination partitions
 * @param all_null Whether every column is a supported all-null fixed-width or STRING column
 * @return Whether there is one row, one partition, or a table of supported all-null columns
 */
inline bool use_trivial(size_type rows, size_type partitions, bool all_null)
{
  return partitions == 1 || rows == 1 || all_null;
}
}  // namespace cudf::detail::partition_policy
