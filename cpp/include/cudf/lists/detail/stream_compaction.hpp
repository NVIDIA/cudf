/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/column/column.hpp>
#include <cudf/detail/stream_compaction.hpp>
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/utilities/memory_resource.hpp>

namespace cudf {
namespace lists::detail {

/**
 * @copydoc cudf::lists::apply_boolean_mask
 *
 * @param mask_kind Specifies how the boolean mask is treated (retentions or deletions)
 */
std::unique_ptr<column> apply_mask(lists_column_view const& input,
                                   lists_column_view const& boolean_mask,
                                   cudf::detail::mask_type mask_kind,
                                   cuda::stream_ref stream,
                                   rmm::device_async_resource_ref mr);

/**
 * @copydoc cudf::lists::distinct(lists_column_view const&, null_equality, nan_equality,
 * duplicate_keep_option, cuda::stream_ref stream, rmm::device_async_resource_ref)
 */
std::unique_ptr<column> distinct(lists_column_view const& input,
                                 null_equality nulls_equal,
                                 nan_equality nans_equal,
                                 duplicate_keep_option keep_option,
                                 cuda::stream_ref stream,
                                 rmm::device_async_resource_ref mr);

}  // namespace lists::detail
}  // namespace cudf
