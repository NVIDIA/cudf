/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/column/column.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/memory_resource.hpp>

/**
 * @file
 * @brief APIs for converting strings columns to and from boolean values
 */

namespace CUDF_EXPORT cudf {
namespace strings {
/**
 * @addtogroup strings_convert
 * @{
 */

/**
 * @brief Returns a new BOOL8 column by parsing boolean values from the strings
 * in the provided strings column.
 *
 * Any null entries will result in corresponding null entries in the output column.
 *
 * @param input Strings instance for this operation
 * @param true_string String to expect for true. Non-matching strings are false
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @return New BOOL8 column converted from strings
 */
std::unique_ptr<column> to_booleans(
  strings_column_view const& input,
  string_scalar const& true_string,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Returns a new strings column converting the boolean values from the
 * provided column into strings.
 *
 * Any null entries will result in corresponding null entries in the output column.
 *
 * @throw cudf::logic_error if the input column is not BOOL8 type.
 *
 * @param booleans Boolean column to convert
 * @param true_string String to use for true in the output column
 * @param false_string String to use for false in the output column
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @return New strings column
 */
std::unique_ptr<column> from_booleans(
  column_view const& booleans,
  string_scalar const& true_string,
  string_scalar const& false_string,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/** @} */  // end of doxygen group
}  // namespace strings
}  // namespace CUDF_EXPORT cudf
