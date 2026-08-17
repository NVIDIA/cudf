/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/column/column_view.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/export.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/device_uvector.hpp>

#include <cuda/stream>

#include <memory>
#include <optional>

namespace CUDF_EXPORT cudf {
namespace reduction::detail {

/**
 * @brief Compute the frequency for each distinct row in the input table.
 *
 * @param input The input table to compute histogram
 * @param partial_counts An optional column containing count for each row
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate memory of the returned objects
 * @return A pair of array contains the (stable-order) indices of the distinct rows in the input
 * table, and their corresponding distinct counts
 */
[[nodiscard]] std::pair<std::unique_ptr<rmm::device_uvector<size_type>>, std::unique_ptr<column>>
compute_row_frequencies(table_view const& input,
                        std::optional<column_view> const& partial_counts,
                        cuda::stream_ref stream,
                        rmm::device_async_resource_ref mr);

/**
 * @brief Create an empty histogram column.
 *
 * A histogram column is a structs column `STRUCT<T, int64_t>` where T is type of the input
 * values.
 *
 * @param values The empty histogram value type will match the data type of this column
 * @returns An empty histogram column
 */
[[nodiscard]] std::unique_ptr<column> make_empty_histogram_like(column_view const& values);

}  // namespace reduction::detail
}  // namespace CUDF_EXPORT cudf
