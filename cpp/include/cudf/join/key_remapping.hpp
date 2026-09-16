/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/column/column.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/export.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <cuda/stream>

#include <memory>

/**
 * @file
 * @brief Class and APIs for remapping join keys to unique integer IDs
 */

namespace CUDF_EXPORT cudf {

/**
 * @addtogroup column_join
 * @{
 */

/**
 * @brief Enum to control whether key remapping metrics should be computed
 */
enum class compute_metrics : bool { NO = false, YES = true };

namespace detail {
/**
 * @brief Forward declaration for key remapping implementation
 */
class key_remapping_impl;
}  // namespace detail

/**
 * @brief Sentinel value for left-side keys not found in the right table
 *
 * This constant is exposed primarily for testing purposes.
 * Application code should check for negative values rather than relying on specific sentinel
 * values.
 */
constexpr size_type KEY_REMAP_NOT_FOUND = -1;

/**
 * @brief Sentinel value for right-side rows with null keys (when nulls are not equal)
 *
 * This constant is exposed primarily for testing purposes.
 * Application code should check for negative values rather than relying on specific sentinel
 * values.
 */
constexpr size_type KEY_REMAP_RIGHT_NULL = -2;

/**
 * @brief Remaps keys to unique integer IDs
 *
 * Each distinct key in the right table is assigned a unique non-negative integer ID.
 * Rows with equal keys will map to the same ID. Keys that cannot be mapped (e.g., not found
 * in the left table, or null keys when nulls are unequal) receive negative sentinel values.
 * The specific ID values are stable for the lifetime of this object but are otherwise unspecified.
 *
 * @note The right table is the build side: the internal hash table is built from its keys, and
 *       keys passed to remap_left_keys() form the probe side matched against it.
 * @note The right table must remain valid for the lifetime of this object,
 *       as the hash table references it directly without copying.
 * @note All NaNs are considered equal
 */
class key_remapping {
 public:
  key_remapping() = delete;
  ~key_remapping();
  key_remapping(key_remapping const&)            = delete;
  key_remapping(key_remapping&&)                 = delete;
  key_remapping& operator=(key_remapping const&) = delete;
  key_remapping& operator=(key_remapping&&)      = delete;

  /**
   * @brief Constructs a key remapping structure from the given right keys.
   *
   * @throw cudf::logic_error if the right table has no columns
   *
   * @param right The right table containing the keys to remap; the internal hash table is built
   *        from this table
   * @param compare_nulls Controls whether null key values should match or not.
   *        When EQUAL, null keys are treated as equal and assigned a valid non-negative ID.
   *        When UNEQUAL, rows with null keys receive a negative sentinel value.
   * @param metrics Controls whether to compute distinct_count and max_duplicate_count.
   *        If YES (default), compute metrics for later retrieval via get_distinct_count()
   *        and get_max_duplicate_count(). If NO, skip metrics computation for better performance;
   *        calling get_distinct_count() or get_max_duplicate_count() will throw.
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @param mr Device memory resource used to allocate the internal hash table
   */
  key_remapping(cudf::table_view const& right,
                null_equality compare_nulls   = null_equality::EQUAL,
                cudf::compute_metrics metrics = cudf::compute_metrics::YES,
                cuda::stream_ref stream       = cudf::get_default_stream(),
                cuda::mr::any_resource<cuda::mr::device_accessible> mr =
                  cudf::get_current_device_resource_ref());

  /**
   * @brief Remap right keys to integer IDs.
   *
   * Recomputes the remapped right table from the cached right keys. This does not cache
   * the remapped table; each call will recompute it from the key remapping.
   *
   * For each row in the cached right table, returns the integer ID assigned to that key.
   * Non-negative integers represent valid mapped keys, while negative values represent
   * keys that cannot be mapped (e.g., null keys when nulls are unequal).
   *
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @param mr Device memory resource used to allocate the returned column's device memory
   *
   * @return A column of INT32 values with the remapped key IDs
   */
  [[nodiscard]] std::unique_ptr<cudf::column> remap_right_keys(
    cuda::stream_ref stream           = cudf::get_default_stream(),
    rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref()) const;

  /**
   * @brief Remap left keys to integer IDs.
   *
   * For each row in the input, returns the integer ID assigned to that key.
   * Non-negative integers represent keys found in the right table, while negative values
   * represent keys that were not found or cannot be matched (e.g., null keys when nulls
   * are unequal, or keys not present in the right table).
   *
   * @throw std::invalid_argument if keys has different number of columns than the right table
   * @throw cudf::data_type_error if keys has different column types than the right table
   *
   * @param keys The left keys to remap (must have same schema as the right table)
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @param mr Device memory resource used to allocate the returned column's device memory
   *
   * @return A column of INT32 values with the remapped key IDs
   */
  [[nodiscard]] std::unique_ptr<cudf::column> remap_left_keys(
    cudf::table_view const& keys,
    cuda::stream_ref stream           = cudf::get_default_stream(),
    rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref()) const;

  /**
   * @brief Check if metrics (distinct_count, max_duplicate_count) were computed.
   *
   * @return true if metrics are available, false if metrics was NO during construction
   */
  [[nodiscard]] bool has_metrics() const;

  /**
   * @brief Get the number of distinct keys in the right table
   *
   * @throw cudf::logic_error if metrics was NO during construction
   *
   * @return The count of unique key combinations found during build
   */
  [[nodiscard]] size_type get_distinct_count() const;

  /**
   * @brief Get the maximum number of times any single key appears
   *
   * @throw cudf::logic_error if metrics was NO during construction
   *
   * @return The maximum duplicate count across all distinct keys
   */
  [[nodiscard]] size_type get_max_duplicate_count() const;

 private:
  using impl_type = cudf::detail::key_remapping_impl;

  std::unique_ptr<impl_type> _impl;
};

/** @} */  // end of group

}  // namespace CUDF_EXPORT cudf
