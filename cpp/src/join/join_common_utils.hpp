/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/join/join.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/resource_ref.hpp>

#include <memory>
#include <utility>

namespace cudf::detail {

constexpr int DEFAULT_JOIN_BLOCK_SIZE = 128;

/**
 * @brief Validates and returns a hash-table load factor.
 *
 * @param load_factor The load factor to validate
 * @return The validated load factor
 * @throws std::invalid_argument if `load_factor` is not in (0, 1]
 */
double checked_load_factor(double load_factor);

// Convenient alias for a pair of unique pointers to device uvectors.
using VectorPair = std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
                             std::unique_ptr<rmm::device_uvector<size_type>>>;

/**
 * @brief Computes the trivial left join operation for the case when the
 * right table is empty.
 *
 * In this case all the valid indices of the left table
 * are returned with their corresponding right indices being set to
 * `JoinNoMatch`, i.e. `cuda::std::numeric_limits<size_type>::min()`.
 *
 * @param left Table of left columns to join
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the result
 *
 * @return Join output indices vector pair
 */
VectorPair get_trivial_left_join_indices(table_view const& left,
                                         rmm::cuda_stream_view stream,
                                         rmm::device_async_resource_ref mr);

/**
 * @brief Finalize a full-join result from a single `(left, right)` index pair.
 *
 * Takes ownership of `indices`, resizes both vectors to `indices.first->size() +
 * right_table_num_rows`, and appends the complement (unmatched right rows paired with
 * `JoinNoMatch`) into the tail. The vectors are then resized down to the true output length.
 *
 * Used by the non-partitioned full-join paths (hash/mixed/conditional); consuming the caller's
 * buffers in-place avoids a redundant concat memcpy over the left-side data.
 *
 * @param indices `(left, right)` index vectors (consumed).
 * @param left_table_num_rows Number of rows in the left table (0 → every right row is
 *                            unmatched, fast path).
 * @param right_table_num_rows Number of rows in the right table.
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @param mr Device memory resource used to allocate working storage.
 *
 * @return `[left_indices, right_indices]` of the complete full-join output.
 */
VectorPair finalize_full_join(VectorPair&& indices,
                              size_type left_table_num_rows,
                              size_type right_table_num_rows,
                              rmm::cuda_stream_view stream,
                              rmm::device_async_resource_ref mr);

/**
 * @brief Finalize a full-join result from per-partition index spans.
 *
 * Concatenates every `(left_partials[i], right_partials[i])` pair into the head of the output
 * and appends the complement (unmatched right rows paired with `JoinNoMatch`) into the tail.
 * Internally delegates to the `VectorPair&&` overload, so the mark/compact path is shared.
 *
 * Used by `cudf::hash_join::finalize_partitioned_full_join` for partitioned full joins where the
 * partials live in separate buffers and must be gathered.
 *
 * @param left_partials Per-partition left index spans.
 * @param right_partials Per-partition right index spans.
 * @param left_table_num_rows Number of rows in the left table.
 * @param right_table_num_rows Number of rows in the right table.
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @param mr Device memory resource used to allocate the returned vectors.
 *
 * @return `[left_indices, right_indices]` sized `sum(left_partials[i].size()) + num_unmatched`.
 */
VectorPair finalize_full_join(
  cudf::host_span<cudf::device_span<size_type const> const> left_partials,
  cudf::host_span<cudf::device_span<size_type const> const> right_partials,
  size_type left_table_num_rows,
  size_type right_table_num_rows,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr);

}  // namespace cudf::detail
