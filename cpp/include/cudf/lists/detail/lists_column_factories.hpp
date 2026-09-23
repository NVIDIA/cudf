/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/column/column.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

namespace cudf {
namespace lists::detail {

/**
 * @brief Internal API to construct a lists column from a `list_scalar`, for public
 * use, use `cudf::make_column_from_scalar`.
 *
 * @param[in] value The `list_scalar` to construct from
 * @param[in] size The number of rows for the output column.
 * @param[in] stream CUDA stream used for device memory operations and kernel launches.
 * @param[in] mr Device memory resource used to allocate the returned column's device memory.
 */
std::unique_ptr<cudf::column> make_lists_column_from_scalar(list_scalar const& value,
                                                            size_type size,
                                                            cuda::stream_ref stream,
                                                            rmm::device_async_resource_ref mr);

/**
 * @brief Create an empty lists column.
 *
 * A list column requires a child type and so cannot be created with `make_empty_column`.
 *
 * @param child_type The type used for the empty child column
 */
std::unique_ptr<column> make_empty_lists_column(data_type child_type);

/**
 * @brief Create a lists column with all null rows.
 *
 * @param size Size of the output lists column
 * @param child_type The type used for the empty child column
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 */
std::unique_ptr<column> make_all_nulls_lists_column(size_type size,
                                                    data_type child_type,
                                                    cuda::stream_ref stream,
                                                    rmm::device_async_resource_ref mr);

}  // namespace lists::detail
}  // namespace cudf
