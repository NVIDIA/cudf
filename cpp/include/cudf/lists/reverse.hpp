/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/column/column.hpp>
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/utilities/export.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <memory>

/**
 * @file
 * @brief API for reversing the element order within each row of a list column
 */

namespace CUDF_EXPORT cudf {
namespace lists {
/**
 * @addtogroup lists_modify
 * @{
 */

/**
 * @brief Reverse the element order within each list of the input column.
 *
 * Any null input row will result in a corresponding null row in the output column.
 *
 * @code{.pseudo}
 * Example:
 * s = [ [1, 2, 3], [], null, [4, 5, null] ]
 * r = reverse(s)
 * r is now [ [3, 2, 1], [], null, [null, 5, 4] ]
 * @endcode
 *
 * @param input Lists column for this operation
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @return New lists column with reversed lists
 */
std::unique_ptr<column> reverse(
  lists_column_view const& input,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/** @} */  // end of doxygen group

}  // namespace lists
}  // namespace CUDF_EXPORT cudf
