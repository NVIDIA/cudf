/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/column/column.hpp>
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/utilities/export.hpp>
#include <cudf/utilities/memory_resource.hpp>

/**
 * @file
 * @brief API for counting the number of elements in each row of a list column
 */

namespace CUDF_EXPORT cudf {
namespace lists {
/**
 * @addtogroup lists_elements
 * @{
 */

/**
 * @brief Returns a numeric column containing the number of rows in
 * each list element in the given lists column.
 *
 * The output column will have the same number of rows as the
 * input lists column. Each `output[i]` will be `input[i].size()`.
 *
 * @code{.pseudo}
 * l = { {1, 2, 3}, {4}, {5, 6} }
 * r = count_elements(l)
 * r is now {3, 1, 2}
 * @endcode
 *
 * Any null input element will result in a corresponding null entry
 * in the output column.
 *
 * @param input Input lists column
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @return New column with the number of elements for each row
 */
std::unique_ptr<column> count_elements(
  lists_column_view const& input,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/** @} */  // end of lists_elements group

}  // namespace lists
}  // namespace CUDF_EXPORT cudf
