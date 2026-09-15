/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <cuda/stream>

namespace cudf {

class scalar;

namespace jni {

/**
 * @brief Given a column-view of LIST<STRUCT<K,V>>, an instance of this class
 * provides an abstraction of a column of maps.
 *
 * Each list row is treated as a map of key->value, with possibly repeated keys.
 * The list may be looked up by a scalar key, or by a column of keys, to
 * retrieve the corresponding value.
 */
class maps_column_view {
 public:
  maps_column_view(lists_column_view const& lists_of_structs,
                   cuda::stream_ref stream = cudf::get_default_stream());

  // Rule of 5.
  maps_column_view(maps_column_view const& maps_view)  = default;
  maps_column_view(maps_column_view&& maps_view)       = default;
  maps_column_view& operator=(maps_column_view const&) = default;
  maps_column_view& operator=(maps_column_view&&)      = default;
  ~maps_column_view()                                  = default;

  /**
   * @brief Returns number of map rows in the column.
   */
  size_type size() const { return keys_.size(); }

  /**
   * @brief Getter for keys as a list column.
   *
   * Note: Keys are not deduped. Repeated keys are returned in order.
   */
  lists_column_view const& keys() const { return keys_; }

  /**
   * @brief Getter for values as a list column.
   *
   * Note: Values for repeated keys are not dropped.
   */
  lists_column_view const& values() const { return values_; }

  /**
   * @brief Map lookup by a column of keys.
   *
   * The lookup column must have as many rows as the map column,
   * and must match the key-type of the map.
   * A column of values is returned, with the same number of rows as the map column.
   * If a key is repeated in a map row, the value corresponding to the last matching
   * key is returned.
   * If a lookup key is null or not found, the corresponding value is null.
   *
   * @param keys Column of keys to be looked up in each corresponding map row.
   * @param stream CUDA stream used for device memory operations and kernel launches.
   * @param mr Device memory resource used to allocate the returned column's device memory.
   * @return std::unique_ptr<column> Column of values corresponding the value of the lookup key.
   */
  std::unique_ptr<column> get_values_for(
    column_view const& keys,
    cuda::stream_ref stream           = cudf::get_default_stream(),
    rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref()) const;

  /**
   * @brief Map lookup by a scalar key.
   *
   * The type of the lookup scalar must match the key-type of the map.
   * A column of values is returned, with the same number of rows as the map column.
   * If a key is repeated in a map row, the value corresponding to the last matching
   * key is returned.
   * If the lookup key is null or not found, the corresponding value is null.
   *
   * @param keys Column of keys to be looked up in each corresponding map row.
   * @param stream CUDA stream used for device memory operations and kernel launches.
   * @param mr Device memory resource used to allocate the returned column's device memory.
   * @return std::unique_ptr<column>
   */
  std::unique_ptr<column> get_values_for(
    scalar const& key,
    cuda::stream_ref stream           = cudf::get_default_stream(),
    rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref()) const;

  /**
   * @brief Check if each map row contains a specified scalar key.
   *
   * The type of the lookup scalar must match the key-type of the map.
   * A column of values is returned, with the same number of rows as the map column.
   *
   * Each row in the returned column contains a bool indicating whether the row contains
   * the specified key (`true`) or not (`false`).
   * The returned column contains no nulls. i.e. If the search key is null, or if the
   * map row is null, the result row is `false`.
   *
   * @param key Scalar key to be looked up in each corresponding map row.
   * @param stream CUDA stream used for device memory operations and kernel launches.
   * @param mr Device memory resource used to allocate the returned column's device memory.
   * @return std::unique_ptr<column>
   */
  std::unique_ptr<column> contains(
    scalar const& key,
    cuda::stream_ref stream           = cudf::get_default_stream(),
    rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref()) const;

  /**
   * @brief Check if each map row contains keys specified by a column
   *
   * The type of the lookup column must match the key-type of the map.
   * A column of values is returned, with the same number of rows as the map column.
   *
   * Each row in the returned column contains a bool indicating whether the row contains
   * the specified key (`true`) or not (`false`).
   * The returned column contains no nulls. i.e. If the search key is null, or if the
   * map row is null, the result row is `false`.
   *
   * @param keys Column of keys to be looked up in each corresponding map row.
   * @param stream CUDA stream used for device memory operations and kernel launches.
   * @param mr Device memory resource used to allocate the returned column's device memory.
   * @return std::unique_ptr<column>
   */

  std::unique_ptr<column> contains(
    column_view const& key,
    cuda::stream_ref stream           = cudf::get_default_stream(),
    rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref()) const;

 private:
  lists_column_view keys_, values_;
};

}  // namespace jni
}  // namespace cudf
