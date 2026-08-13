/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/join/join.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/export.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>

#include <memory>

/**
 * @file
 * @brief Class for mark-based hash joins used to implement semi/anti joins
 */

namespace CUDF_EXPORT cudf {

/**
 * @addtogroup column_join
 * @{
 */

namespace detail {
class mark_join;
}  // namespace detail

/**
 * @brief Mark-based hash join for semi/anti join with left table reuse.
 *
 * Builds a hash table from the left (build) table using a multiset that allows
 * duplicate keys. The probe kernel atomically marks matching left entries via
 * CAS on the hash MSB, then a retrieve kernel collects marked (semi) or unmarked
 * (anti) entries.
 *
 * This class enables building the hash table once and probing multiple times
 * with different right (probe) tables, amortizing the build cost. Probe-side
 * prefiltering can be enabled at construction time via `join_prefilter`.
 *
 * @note This class is designed for the case where the **left table is reused**
 * across multiple semi/anti join operations. It should only be used when:
 * - The left (build) table is **smaller** than the right (probe) table.
 *   Building a hash table from the larger table is memory-inefficient and
 *   leads to poor probe performance due to longer collision chains.
 * - The left table is **reasonably small** (e.g. ≤1M rows). The mark-based
 *   probe walks the hash table linearly per right row, so performance degrades
 *   with large hash tables. For large left tables, consider alternative join
 *   strategies.
 * - The left table is probed **multiple times**. If only a single join is needed,
 *   there is no benefit to reuse and standard join APIs may be more efficient.
 *
 * For the common case where the **right (filter) table** is reused, use
 * `cudf::filtered_join` instead, which builds a distinct set from the right table.
 *
 * @note All NaNs are considered as equal
 */
class mark_join {
 public:
  mark_join() = delete;
  ~mark_join();
  mark_join(mark_join const&)            = delete;
  mark_join(mark_join&&)                 = delete;
  mark_join& operator=(mark_join const&) = delete;
  mark_join& operator=(mark_join&&)      = delete;

  /**
   * @brief Constructs a mark join object with explicit prefilter selection.
   *
   * @param left The left table; the hash table is built from this table
   * @param compare_nulls Controls whether null join-key values should match or not
   * @param prefilter Controls whether an optional probe-side prefilter is enabled
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @param mr Device memory resource used to allocate the internal hash table and prefilter
   */
  mark_join(cudf::table_view const& left,
            cudf::null_equality compare_nulls,
            cudf::join_prefilter prefilter,
            rmm::cuda_stream_view stream = cudf::get_default_stream(),
            cuda::mr::any_resource<cuda::mr::device_accessible> mr =
              cudf::get_current_device_resource_ref());

  /**
   * @brief Constructs a mark join object with explicit prefilter selection and the given load
   * factor.
   *
   * @throws std::invalid_argument if `load_factor` is not in (0, 1]
   *
   * @param left The left table; the hash table is built from this table
   * @param load_factor Hash table load factor in range (0,1]
   * @param compare_nulls Controls whether null join-key values should match or not
   * @param prefilter Controls whether an optional probe-side prefilter is enabled
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @param mr Device memory resource used to allocate the internal hash table and prefilter
   */
  mark_join(cudf::table_view const& left,
            double load_factor,
            cudf::null_equality compare_nulls = cudf::null_equality::EQUAL,
            cudf::join_prefilter prefilter    = cudf::join_prefilter::NO,
            rmm::cuda_stream_view stream      = cudf::get_default_stream(),
            cuda::mr::any_resource<cuda::mr::device_accessible> mr =
              cudf::get_current_device_resource_ref());

  /**
   * @brief Returns left row indices that have at least one match in the right table.
   *
   * @param right The right table; probed against the hash table built from the left table
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @param mr Device memory resource used to allocate the returned device memory
   * @return Device vector of left row indices
   */
  [[nodiscard]] std::unique_ptr<rmm::device_uvector<size_type>> semi_join(
    cudf::table_view const& right,
    rmm::cuda_stream_view stream      = cudf::get_default_stream(),
    rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref()) const;

  /**
   * @brief Returns left row indices that have no match in the right table.
   *
   * @param right The right table; probed against the hash table built from the left table
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @param mr Device memory resource used to allocate the returned device memory
   * @return Device vector of left row indices
   */
  [[nodiscard]] std::unique_ptr<rmm::device_uvector<size_type>> anti_join(
    cudf::table_view const& right,
    rmm::cuda_stream_view stream      = cudf::get_default_stream(),
    rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref()) const;

 private:
  std::unique_ptr<cudf::detail::mark_join> _impl;
};

/** @} */  // end of group

}  // namespace CUDF_EXPORT cudf
