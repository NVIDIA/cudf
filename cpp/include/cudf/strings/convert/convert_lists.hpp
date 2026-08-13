/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/column/column.hpp>
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/memory_resource.hpp>

/**
 * @file
 * @brief Strings column APIs for converting a lists column into a formatted strings column.
 */

namespace CUDF_EXPORT cudf {
namespace strings {
/**
 * @addtogroup strings_convert
 * @{
 */

/**
 * @brief Convert a list column of strings into a formatted strings column.
 *
 * The `separators` column should contain 3 strings elements in the following order:
 * - element separator (default is comma `,`)
 * - left-hand enclosure (default is `[`)
 * - right-hand enclosure (default is `]`)
 *
 * @code{.pseudo}
 * l1 = { [[a,b,c], [d,e]], [[f,g], [h]] }
 * s1 = format_list_column(l1)
 * s1 is now ["[[a,b,c],[d,e]]", "[[f,g],[h]]"]
 *
 * l2 = { [[a,b,c], [d,e]], [NULL], [[f,g], NULL, [h]] }
 * s2 = format_list_column(l1, '-', [':', '{', '}'])
 * s2 is now ["{{a:b:c}:{d:e}}", "{-}", "{{f:g}:-:{h}}"]
 * @endcode
 *
 * @throw cudf::logic_error if the input column is not a LIST type with a STRING child.
 *
 * @param input Lists column to format
 * @param na_rep Replacement string for null elements
 * @param separators Strings to use for enclosing list components and separating elements
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @return New strings column
 */
std::unique_ptr<column> format_list_column(
  lists_column_view const& input,
  string_scalar const& na_rep           = string_scalar(""),
  strings_column_view const& separators = strings_column_view(column_view{
    data_type{type_id::STRING}, 0, nullptr, nullptr, 0}),
  rmm::cuda_stream_view stream          = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr     = cudf::get_current_device_resource_ref());

/** @} */  // end of doxygen group
}  // namespace strings
}  // namespace CUDF_EXPORT cudf
