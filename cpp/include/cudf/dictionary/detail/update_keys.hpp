/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/column/column.hpp>
#include <cudf/dictionary/dictionary_column_view.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/span.hpp>

#include <cuda/stream>

#include <span>

namespace cudf {
namespace dictionary::detail {
/**
 * @copydoc cudf::dictionary::add_keys(dictionary_column_view const&,column_view
 * const&,rmm::device_async_resource_ref)
 *
 * @param stream CUDA stream used for device memory operations and kernel launches.
 */
std::unique_ptr<column> add_keys(dictionary_column_view const& dictionary_column,
                                 column_view const& new_keys,
                                 cuda::stream_ref stream,
                                 rmm::device_async_resource_ref mr);

/**
 * @copydoc cudf::dictionary::remove_keys(dictionary_column_view const&,column_view
 * const&,rmm::device_async_resource_ref)
 *
 * @param stream CUDA stream used for device memory operations and kernel launches.
 */
std::unique_ptr<column> remove_keys(dictionary_column_view const& dictionary_column,
                                    column_view const& keys_to_remove,
                                    cuda::stream_ref stream,
                                    rmm::device_async_resource_ref mr);

/**
 * @copydoc cudf::dictionary::remove_unused_keys(dictionary_column_view
 * const&,rmm::device_async_resource_ref)
 *
 * @param stream CUDA stream used for device memory operations and kernel launches.
 */
std::unique_ptr<column> remove_unused_keys(dictionary_column_view const& dictionary_column,
                                           cuda::stream_ref stream,
                                           rmm::device_async_resource_ref mr);

/**
 * @copydoc cudf::dictionary::set_keys(dictionary_column_view
 * const&,rmm::device_async_resource_ref)
 *
 * @param stream CUDA stream used for device memory operations and kernel launches.
 */
std::unique_ptr<column> set_keys(dictionary_column_view const& dictionary_column,
                                 column_view const& keys,
                                 cuda::stream_ref stream,
                                 rmm::device_async_resource_ref mr);

/**
 * @brief Remap the indices of a dictionary column to a new key set, returning
 * only the remapped index column with its null mask.
 *
 * Like set_keys() but does not copy the key set or build a dictionary column.
 * Rows whose key value is not found in new_keys are mapped to null.
 *
 * @throws std::invalid_argument if new_keys is empty or contains nulls
 * @throws cudf::data_type_error if new_keys and input.keys() data_types do not match
 *
 * @param input Dictionary column whose indices are to be remapped.
 * @param new_keys Key column to remap indices into. Must be non-empty, null-free,
 *        and the same type as input's keys.
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @param mr Device memory resource used to allocate the returned column's device memory.
 * @return Integer column of remapped indices with the null mask from the remap applied.
 */
std::unique_ptr<column> remap_indices(dictionary_column_view const& input,
                                      column_view const& new_keys,
                                      cuda::stream_ref stream,
                                      rmm::device_async_resource_ref mr);

/**
 * @copydoc
 * cudf::dictionary::match_dictionaries(std::vector<cudf::dictionary_column_view>,rmm::device_async_resource_ref)
 */
std::vector<std::unique_ptr<column>> match_dictionaries(
  std::span<dictionary_column_view const> input,
  cuda::stream_ref stream,
  rmm::device_async_resource_ref mr);

/**
 * @brief Create new dictionaries that have keys merged from dictionary columns
 * found in the provided tables.
 *
 * The result includes a vector of new dictionary columns along with a
 * vector of table_views with corresponding updated column_views.
 * And any column_views in the input tables that are not dictionary type
 * are simply copied.
 *
 * Merging the dictionary keys also adjusts the indices appropriately in the
 * output dictionary columns.
 *
 * Any null rows are left unchanged.
 *
 * @param tables Vector of cudf::table_views that include dictionary columns to be matched.
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @param mr Device memory resource used to allocate the returned column's device memory.
 * @return New dictionary columns and updated cudf::table_views.
 */
std::pair<std::vector<std::unique_ptr<column>>, std::vector<table_view>> match_dictionaries(
  std::vector<table_view> tables, cuda::stream_ref stream, rmm::device_async_resource_ref mr);

/**
 * @brief Like match_dictionaries() but returns index columns in place of the
 * dictionary columns
 *
 * Computes the merged unique key set across all input dictionaries, remaps each
 * dictionary's indices to that key set, and returns the resulting index columns
 * (with null masks). The merged keys are not returned.
 *
 * This is more efficient than match_dictionaries() when the caller only needs
 * to compare values by index and does not need the actual keys.
 *
 * @throws std::invalid_argument if input is empty
 *
 * @param input Span of dictionary column views to match
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned columns' device memory
 * @return One index column per input dictionary, in the same order as input
 */
std::vector<std::unique_ptr<column>> match_dictionaries_to_indices(
  std::span<dictionary_column_view const> input,
  cuda::stream_ref stream,
  rmm::device_async_resource_ref mr);

/**
 * @brief Like match_dictionaries() but substitutes index columns in place
 * of dictionary columns in the returned table_views
 *
 * For each dictionary column found across the input `tables`, computes the merged
 * unique key set and remaps each column's indices to it. The returned table_views
 * reference these index columns (and the originals for non-dictionary columns).
 * The merged keys are not returned.
 *
 * @throws std::invalid_argument if tables is empty
 *
 * @param tables Vector of table_views containing dictionary columns to be matched
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned columns' device memory
 * @return Index column owners and updated table_views with index columns substituted
 */
std::pair<std::vector<std::unique_ptr<column>>, std::vector<table_view>>
match_dictionaries_to_indices(std::vector<table_view> tables,
                              cuda::stream_ref stream,
                              rmm::device_async_resource_ref mr);

}  // namespace dictionary::detail
}  // namespace cudf
