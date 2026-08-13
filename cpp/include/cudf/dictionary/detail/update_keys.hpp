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

#include <rmm/cuda_stream_view.hpp>

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
                                 rmm::cuda_stream_view stream,
                                 rmm::device_async_resource_ref mr);

/**
 * @copydoc cudf::dictionary::remove_keys(dictionary_column_view const&,column_view
 * const&,rmm::device_async_resource_ref)
 *
 * @param stream CUDA stream used for device memory operations and kernel launches.
 */
std::unique_ptr<column> remove_keys(dictionary_column_view const& dictionary_column,
                                    column_view const& keys_to_remove,
                                    rmm::cuda_stream_view stream,
                                    rmm::device_async_resource_ref mr);

/**
 * @copydoc cudf::dictionary::remove_unused_keys(dictionary_column_view
 * const&,rmm::device_async_resource_ref)
 *
 * @param stream CUDA stream used for device memory operations and kernel launches.
 */
std::unique_ptr<column> remove_unused_keys(dictionary_column_view const& dictionary_column,
                                           rmm::cuda_stream_view stream,
                                           rmm::device_async_resource_ref mr);

/**
 * @copydoc cudf::dictionary::set_keys(dictionary_column_view
 * const&,rmm::device_async_resource_ref)
 *
 * @param stream CUDA stream used for device memory operations and kernel launches.
 */
std::unique_ptr<column> set_keys(dictionary_column_view const& dictionary_column,
                                 column_view const& keys,
                                 rmm::cuda_stream_view stream,
                                 rmm::device_async_resource_ref mr);

/**
 * @copydoc
 * cudf::dictionary::match_dictionaries(std::vector<cudf::dictionary_column_view>,rmm::device_async_resource_ref)
 *
 * @param stream CUDA stream used for device memory operations and kernel launches.
 */
std::vector<std::unique_ptr<column>> match_dictionaries(
  std::span<dictionary_column_view const> input,
  rmm::cuda_stream_view stream,
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
  std::vector<table_view> tables, rmm::cuda_stream_view stream, rmm::device_async_resource_ref mr);

}  // namespace dictionary::detail
}  // namespace cudf
