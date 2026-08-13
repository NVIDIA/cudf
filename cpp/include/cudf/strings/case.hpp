/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/column/column.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/memory_resource.hpp>

/**
 * @file
 * @brief Strings column APIs for converting to upper, lower, or swapped case.
 */

namespace CUDF_EXPORT cudf {
namespace strings {
/**
 * @addtogroup strings_case
 * @{
 */

/**
 * @brief Converts a column of strings to lower case.
 *
 * Only upper case alphabetical characters are converted. All other characters are copied.
 * Case conversion may result in strings that are longer or shorter than the
 * original string in bytes.
 *
 * Any null entries create null entries in the output column.
 *
 * @param strings Strings instance for this operation.
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory.
 * @return New column of strings with characters converted.
 */
std::unique_ptr<column> to_lower(
  strings_column_view const& strings,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Converts a column of strings to upper case.
 *
 * Only lower case alphabetical characters are converted. All other characters are copied.
 * Case conversion may result in strings that are longer or shorter than the
 * original string in bytes.
 *
 * Any null entries create null entries in the output column.
 *
 * @param strings Strings instance for this operation.
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory.
 * @return New column of strings with characters converted.
 */
std::unique_ptr<column> to_upper(
  strings_column_view const& strings,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Returns a column of strings converting lower case characters to
 * upper case and vice versa.
 *
 * Only upper or lower case alphabetical characters are converted. All other characters are copied.
 * Case conversion may result in strings that are longer or shorter than the
 * original string in bytes.
 *
 * Any null entries create null entries in the output column.
 *
 * @param strings Strings instance for this operation.
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory.
 * @return New column of strings with characters converted.
 */
std::unique_ptr<column> swapcase(
  strings_column_view const& strings,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/** @} */  // end of doxygen group
}  // namespace strings
}  // namespace CUDF_EXPORT cudf
