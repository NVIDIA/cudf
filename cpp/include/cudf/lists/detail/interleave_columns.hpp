/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/column/column.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <cuda/stream>

namespace cudf {
namespace lists::detail {

/**
 * @brief Returns a single column by interleaving rows of the given table of list elements.
 *
 * @code{.pseudo}
 * s1 = [{0, 1}, {2, 3, 4}, {5}, {}, {6, 7}]
 * s2 = [{8}, {9}, {}, {10, 11, 12}, {13, 14, 15, 16}]
 * r = lists::interleave_columns(s1, s2)
 * r is now [{0, 1}, {8}, {2, 3, 4}, {9}, {5}, {}, {}, {10, 11, 12}, {6, 7}, {13, 14, 15, 16}]
 * @endcode
 *
 * @throws cudf::logic_error if any column of the input table is not a lists columns.
 * @throws cudf::logic_error if any lists column contains nested typed entry.
 * @throws cudf::logic_error if all lists columns do not have the same entry type.
 *
 * @param input Table containing lists columns to interleave.
 * @param has_null_mask A boolean flag indicating that the input columns have a null mask.
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @param mr Device memory resource used to allocate the returned column's device memory.
 * @return The interleaved columns as a single column.
 */
std::unique_ptr<column> interleave_columns(table_view const& input,
                                           bool has_null_mask,
                                           cuda::stream_ref stream,
                                           rmm::device_async_resource_ref mr);

}  // namespace lists::detail
}  // namespace cudf
