/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/labeling/label_bins.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <cuda/stream>

namespace CUDF_EXPORT cudf {

namespace detail {

/**
 * @addtogroup label_bins
 * @{
 * @file
 * @brief Internal APIs for labeling values by bin.
 */

/**
 * @copydoc cudf::label_bins(column_view const& input, column_view const& left_edges, inclusive
 * left_inclusive, column_view const& right_edges, inclusive right_inclusive, cuda::stream_ref,
 * rmm::device_async_resource_ref mr)
 *
 * @param stream Stream view on which to allocate resources and queue execution.
 */
std::unique_ptr<column> label_bins(column_view const& input,
                                   column_view const& left_edges,
                                   inclusive left_inclusive,
                                   column_view const& right_edges,
                                   inclusive right_inclusive,
                                   cuda::stream_ref stream,
                                   rmm::device_async_resource_ref mr);

/** @} */  // end of group
}  // namespace detail
}  // namespace CUDF_EXPORT cudf
