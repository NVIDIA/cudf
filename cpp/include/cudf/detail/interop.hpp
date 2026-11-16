/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/interop.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <cuda/stream>

#include <span>
#include <vector>

namespace cudf {
namespace detail {

/**
 * @copydoc cudf::from_dlpack
 */
std::unique_ptr<table> from_dlpack(DLManagedTensor const* managed_tensor,
                                   cuda::stream_ref stream,
                                   cudf::memory_resources mr);

/**
 * @copydoc cudf::to_dlpack
 */
DLManagedTensor* to_dlpack(table_view const& input,
                           cuda::stream_ref stream,
                           cudf::memory_resources mr);

/**
 * @brief The properties of a column that determine its Arrow schema
 *
 * The children follow the children of the column, so a list has its offsets and its child, and
 * a dictionary has its indices and its keys.
 */
struct arrow_schema_column {
  data_type type;    ///< Type of the column
  bool nullable{};   ///< Whether the column has a null mask
  bool has_nulls{};  ///< Whether the column has nulls, which requires `nullable`
  std::vector<arrow_schema_column> children;  ///< Properties of the children
};

/**
 * @brief Return the properties of `column` that determine its Arrow schema
 *
 * @param column The column
 * @return The properties of `column` and its children
 */
arrow_schema_column to_arrow_schema_column(column_view const& column);

/**
 * @brief Create the Arrow schema of a table from the properties of its columns
 *
 * Returns the same schema as `cudf::to_arrow_schema` of a table whose columns have these
 * properties. No column data is read.
 *
 * @throws cudf::logic_error if `metadata` does not have one entry per column
 * @throws std::invalid_argument if a list column does not have two children
 *
 * @param columns The properties of the columns
 * @param metadata Names and other metadata of the columns
 * @return Arrow schema of a struct with one child per column
 */
unique_schema_t to_arrow_schema(std::span<arrow_schema_column const> columns,
                                std::span<column_metadata const> metadata);

}  // namespace detail
}  // namespace cudf
