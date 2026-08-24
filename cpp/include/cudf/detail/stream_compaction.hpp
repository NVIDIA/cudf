/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/column/column_view.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/device_uvector.hpp>

#include <cuda/stream>

namespace cudf {
namespace detail {
/**
 * @copydoc cudf::drop_nulls(table_view const&, std::vector<size_type> const&,
 *                           cudf::size_type, cuda::stream_ref, rmm::device_async_resource_ref)
 */
std::unique_ptr<table> drop_nulls(table_view const& input,
                                  std::vector<size_type> const& keys,
                                  cudf::size_type keep_threshold,
                                  cuda::stream_ref stream,
                                  rmm::device_async_resource_ref mr);

/**
 * @copydoc cudf::drop_nans(table_view const&, std::vector<size_type> const&,
 *                          cudf::size_type, cuda::stream_ref, rmm::device_async_resource_ref)
 */
std::unique_ptr<table> drop_nans(table_view const& input,
                                 std::vector<size_type> const& keys,
                                 cudf::size_type keep_threshold,
                                 cuda::stream_ref stream,
                                 rmm::device_async_resource_ref mr);

/**
 * @brief Specifies how the boolean mask should be interpreted by `apply_mask` API.
 */
enum class mask_type : bool {
  DELETION  = false,  ///< `true` elements in the mask indicate deletions.
  RETENTION = true,   ///< `true` elements in the mask indicate retentions.
};

/**
 * @copydoc cudf::apply_boolean_mask
 *
 * @param mask_kind Specifies how the boolean mask is treated (retentions or deletions)
 */
std::unique_ptr<table> apply_mask(table_view const& input,
                                  column_view const& boolean_mask,
                                  mask_type mask_kind,
                                  cuda::stream_ref stream,
                                  rmm::device_async_resource_ref mr);

/**
 * @copydoc cudf::unique
 */
std::unique_ptr<table> unique(table_view const& input,
                              std::vector<size_type> const& keys,
                              duplicate_keep_option keep,
                              null_equality nulls_equal,
                              cuda::stream_ref stream,
                              rmm::device_async_resource_ref mr);

/**
 * @copydoc cudf::distinct
 */
std::unique_ptr<table> distinct(table_view const& input,
                                std::vector<size_type> const& keys,
                                duplicate_keep_option keep,
                                null_equality nulls_equal,
                                nan_equality nans_equal,
                                cuda::stream_ref stream,
                                rmm::device_async_resource_ref mr);

/**
 * @copydoc cudf::stable_distinct
 */
std::unique_ptr<table> stable_distinct(table_view const& input,
                                       std::vector<size_type> const& keys,
                                       duplicate_keep_option keep,
                                       null_equality nulls_equal,
                                       nan_equality nans_equal,
                                       cuda::stream_ref stream,
                                       rmm::device_async_resource_ref mr);

/**
 * @copydoc cudf::distinct_indices
 *
 * @return A device_uvector containing the result indices
 */
rmm::device_uvector<size_type> distinct_indices(table_view const& input,
                                                duplicate_keep_option keep,
                                                null_equality nulls_equal,
                                                nan_equality nans_equal,
                                                cuda::stream_ref stream,
                                                rmm::device_async_resource_ref mr);

}  // namespace detail
}  // namespace cudf
