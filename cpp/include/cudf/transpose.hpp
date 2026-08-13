/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/column/column.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/export.hpp>
#include <cudf/utilities/memory_resource.hpp>

/**
 * @file
 * @brief APIs for transposing a table.
 */

namespace CUDF_EXPORT cudf {
/**
 * @addtogroup reshape_transpose
 * @{
 */

/**
 * @brief Transposes a table.
 *
 * Stores output in a contiguous column, exposing the transposed table as
 * a `table_view`.
 *
 * @throw cudf::logic_error if column types are non-homogeneous
 * @throw cudf::logic_error if column types are non-fixed-width
 *
 * @param[in] input   A table (M cols x N rows) to be transposed
 * @param[in] stream  CUDA stream used for device memory operations and kernel launches
 * @param[in] mr      Device memory resource used to allocate the device memory of returned value
 * @return            The transposed input (N cols x M rows) as a `column` and
 *                    `table_view`, representing the owner and transposed table,
 *                    respectively.
 */
std::pair<std::unique_ptr<column>, table_view> transpose(
  table_view const& input,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/** @} */  // end of group
}  // namespace CUDF_EXPORT cudf
