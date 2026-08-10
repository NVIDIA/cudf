/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/column/column.hpp>
#include <cudf/dictionary/dictionary_column_view.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/span.hpp>

#include <span>

/**
 * @file
 * @brief APIs for adding, removing, and matching the keys of dictionary columns.
 */

namespace CUDF_EXPORT cudf {
namespace dictionary {
/**
 * @addtogroup dictionary_update
 * @{
 */

/**
 * @brief Create a new dictionary column by adding the new keys elements
 * to the existing dictionary_column
 *
 * The indices remain the same while unique new keys are added.
 *
 * @code{.pseudo}
 * d1 = { keys=["a", "c", "d"], indices=[2, 0, 1, 0, 1]}
 * d2 = add_keys( d1, ["b", "c"] )
 * d2 is now {keys=["a", "c", "d", "b"], indices=[2, 0, 1, 0, 1]}
 * @endcode
 *
 * The output column will have the same number of rows as the input column.
 * Null entries from the input column are copied to the output column.
 * No new null entries are created by this operation.
 *
 * @throw cudf::data_type_error if the new_keys type does not match the keys type in
 *        the dictionary_column.
 * @throw std::invalid_argument if the new_keys contain nulls.
 *
 * @param dictionary_column Existing dictionary column.
 * @param new_keys New keys to incorporate into the dictionary_column.
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @param mr Device memory resource used to allocate the returned column's device memory.
 * @return New dictionary column.
 */
std::unique_ptr<column> add_keys(
  dictionary_column_view const& dictionary_column,
  column_view const& new_keys,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Create a new dictionary column by removing the specified keys
 * from the existing dictionary_column.
 *
 * The output column will have the same number of rows as the input column.
 * Null entries from the input column and copied to the output column.
 * The indices are updated to the new positions of the remaining keys.
 * Any indices pointing to removed keys sets that row to null.
 *
 * @code{.pseudo}
 * d1 = {keys=["a", "c", "d"], indices=[2, 0, 1, 0, 2]}
 * d2 = remove_keys( d1, ["b", "c"] )
 * d2 is now {keys=["a", "d"], indices=[1, 0, x, 0, 1], valids=[1, 1, 0, 1, 1]}
 * @endcode
 * Note that "a" has been removed so output row[2] becomes null.
 *
 * @throw cudf::data_type_error if the keys_to_remove type does not match the keys type in
 *        the dictionary_column.
 * @throw std::invalid_argument if the keys_to_remove contain nulls.
 *
 * @param dictionary_column Existing dictionary column.
 * @param keys_to_remove The keys to remove from the dictionary_column.
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @param mr Device memory resource used to allocate the returned column's device memory.
 * @return New dictionary column.
 */
std::unique_ptr<column> remove_keys(
  dictionary_column_view const& dictionary_column,
  column_view const& keys_to_remove,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Create a new dictionary column by removing any keys
 * that are not referenced by any of the indices.
 *
 * The indices are updated to the new position values of the remaining keys.
 *
 * @code{.pseudo}
 * d1 = {["a","c","d"],[2,0,2,0]}
 * d2 = remove_unused_keys(d1)
 * d2 is now {["a","d"],[1,0,1,0]}
 * @endcode
 *
 * @param dictionary_column Existing dictionary column.
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @param mr Device memory resource used to allocate the returned column's device memory.
 * @return New dictionary column.
 */
std::unique_ptr<column> remove_unused_keys(
  dictionary_column_view const& dictionary_column,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Create a new dictionary column by applying only the specified keys
 * to the existing dictionary_column.
 *
 * Any new elements found in the keys parameter are added to the output dictionary.
 * Any existing keys not in the keys parameter are removed.
 *
 * The number of rows in the output column will be the same as the number of rows
 * in the input column. Existing null entries are copied to the output column.
 * The indices are updated to reflect the position values of the new keys.
 * Any indices pointing to removed keys sets those rows to null.
 *
 * Although duplicate keys are allowed, indices in the returned dictionary may
 * only reference one of the duplicates.
 *
 * @code{.pseudo}
 * d1 = {keys=["a", "b", "c"], indices=[2, 0, 1, 2, 1]}
 * d2 = set_keys(existing_dict, ["b","c","d"])
 * d2 is now {keys=["b", "c", "d"], indices=[1, x, 0, 1, 0], valids=[1, 0, 1, 1, 1]}
 * @endcode
 *
 * @throw cudf::data_type_error if the keys type does not match the keys type in
 *        the dictionary_column.
 * @throw std::invalid_argument if the keys contain nulls.
 *
 * @param dictionary_column Existing dictionary column.
 * @param keys New keys to use for the output column. Must not contain nulls.
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @param mr Device memory resource used to allocate the returned column's device memory.
 * @return New dictionary column.
 */
std::unique_ptr<column> set_keys(
  dictionary_column_view const& dictionary_column,
  column_view const& keys,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Create new dictionaries that have keys merged from the input dictionaries.
 *
 * This will concatenate the keys for each dictionary and then call `set_keys` on each.
 * The result is a vector of new dictionaries with a common set of keys.
 *
 * @param input Dictionary columns to match keys.
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @param mr Device memory resource used to allocate the returned column's device memory.
 * @return New dictionary columns.
 */
std::vector<std::unique_ptr<column>> match_dictionaries(
  std::span<dictionary_column_view const> input,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/** @} */  // end of group
}  // namespace dictionary
}  // namespace CUDF_EXPORT cudf
