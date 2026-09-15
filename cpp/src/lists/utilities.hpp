/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/column/column_view.hpp>
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <cuda/stream>

namespace cudf::lists::detail {

/**
 * @brief Generate list labels for elements in the child column of the input lists column.
 *
 * @param input The input lists column
 * @param n_elements The number of elements in the child column of the input lists column
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned object
 * @return A column containing list labels corresponding to each element in the child column
 */
std::unique_ptr<column> generate_labels(lists_column_view const& input,
                                        size_type n_elements,
                                        cuda::stream_ref stream,
                                        rmm::device_async_resource_ref mr);

/**
 * @brief Reconstruct an offsets column from the input list labels column.
 *
 * @param labels The list labels corresponding to each list element
 * @param n_lists The number of lists to build the offsets column
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned object
 * @return The output offsets column
 */
std::unique_ptr<column> reconstruct_offsets(column_view const& labels,
                                            size_type n_lists,
                                            cuda::stream_ref stream,
                                            rmm::device_async_resource_ref mr);

/**
 * @brief Generate 0-based list offsets from the offsets of the input lists column.
 *
 * @param input The input lists column
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned object
 * @return The output offsets column with values start from 0
 */
std::unique_ptr<column> get_normalized_offsets(lists_column_view const& input,
                                               cuda::stream_ref stream,
                                               rmm::device_async_resource_ref mr);

}  // namespace cudf::lists::detail
