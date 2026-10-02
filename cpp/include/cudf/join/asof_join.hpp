/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/column/column_view.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/export.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/device_uvector.hpp>

#include <cuda/stream>

#include <memory>

/**
 * @file
 * @brief Class and supporting types for as-of joins.
 */

namespace CUDF_EXPORT cudf {

/**
 * @addtogroup column_join
 * @{
 */

/**
 * @brief Selects how an as-of join searches for a matching right row.
 */
enum class asof_join_strategy {
  BACKWARD,  ///< Select the last right row whose ordered key is less than or equal to the left key.
  FORWARD,   ///< Select the first right row whose ordered key is greater than or equal to the left
             ///< key.
  NEAREST    ///< Select the right row whose ordered key is nearest to the left key.
};

namespace detail {
class asof_join;
}  // namespace detail

/**
 * @brief Finds one ordered right-side match for each left-side row.
 *
 * An as-of join optionally matches rows on equality keys (`by`) and then selects one right row by
 * comparing an ordered key (`on`). The right side is preprocessed on construction and may be probed
 * repeatedly with different left inputs.
 *
 * The right input must be sorted in ascending lexicographic order by `(right_by..., right_on)`,
 * with nulls ordered before non-nulls. Consequently, rows with equal `right_by` keys must be
 * physically contiguous; interleaved right-side groups are not supported. The left input may be in
 * any order, and the result preserves its row order. These ordering preconditions are not
 * validated.
 *
 * Grouping-key nulls compare unequal. A null left ordered key never matches, and a null right
 * ordered key is never a valid match candidate.
 *
 * Currently, only `asof_join_strategy::BACKWARD` with exact matches enabled is supported. It
 * selects the last right row in the same group whose ordered key is less than or equal to the left
 * ordered key. If the selected ordered key has duplicates, the last duplicate is selected.
 *
 * The ordered columns must have the same type and must be integer, timestamp, or duration columns.
 * Corresponding grouping columns must have the same type and must be strings or fixed-width scalar
 * types. Nested and dictionary columns are not supported.
 *
 * @note The `asof_join` object must not outlive the columns viewed by `right_by` or `right_on`.
 */
class asof_join {
 public:
  asof_join() = delete;
  ~asof_join();
  asof_join(asof_join const&)            = delete;
  asof_join(asof_join&&)                 = delete;
  asof_join& operator=(asof_join const&) = delete;
  asof_join& operator=(asof_join&&)      = delete;

  /**
   * @brief Constructs an as-of join object by preprocessing the right-side keys.
   *
   * Passing an empty `right_by` performs an ungrouped as-of join. `right_on` must be sorted in
   * ascending order in this case.
   *
   * @throws cudf::logic_error if `right_by` has columns and
   *         `right_by.num_rows() != right_on.size()`
   * @throws cudf::logic_error if the right-side keys do not satisfy the supported type contract
   *
   * @param right_by Right-side equality grouping keys
   * @param right_on Right-side ordered key
   * @param stream CUDA stream used for device memory operations and kernel launches
   */
  asof_join(table_view const& right_by,
            column_view const& right_on,
            cuda::stream_ref stream = cudf::get_default_stream());

  /**
   * @brief Returns the matching right-side row index for every left-side row.
   *
   * The result has one entry per left row in left input order. An entry is `cudf::JoinNoMatch` when
   * no eligible right row exists in the same group or the left ordered key is null.
   *
   * This method is thread-safe and can be called concurrently from multiple threads on the same
   * instance.
   *
   * Right-side preprocessing is asynchronous on the stream passed to the constructor. Before
   * probing on a different stream, the caller must synchronize the constructor stream or establish
   * an equivalent CUDA event dependency so that preprocessing completes before the probe executes.
   * Probes on the constructor stream are already ordered and require no additional synchronization.
   *
   * @throws cudf::logic_error if the left and right inputs have different numbers of grouping keys
   * @throws cudf::logic_error if `left_by` has columns and
   *         `left_by.num_rows() != left_on.size()`
   * @throws cudf::logic_error if corresponding left and right key types do not match
   * @throws cudf::logic_error if the left-side keys do not satisfy the supported type contract
   * @throws cudf::logic_error if `strategy` is not `asof_join_strategy::BACKWARD`
   * @throws cudf::logic_error if `allow_exact_matches` is `false`
   *
   * @param left_by Left-side equality grouping keys
   * @param left_on Left-side ordered key
   * @param strategy Direction in which to search for a right-side match
   * @param allow_exact_matches Whether equal ordered keys may match
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @param mr Device memory resource used to allocate the returned indices
   * @return Right-side gather map containing a row index or `cudf::JoinNoMatch` per left row
   */
  std::unique_ptr<rmm::device_uvector<size_type>> join(
    table_view const& left_by,
    column_view const& left_on,
    asof_join_strategy strategy,
    bool allow_exact_matches,
    cuda::stream_ref stream           = cudf::get_default_stream(),
    rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref()) const;

 private:
  using impl_type = cudf::detail::asof_join;
  std::unique_ptr<impl_type const> _impl;
};

/** @} */  // end of group
}  // namespace CUDF_EXPORT cudf
