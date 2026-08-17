/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * @file writer_impl_helpers.hpp
 * @brief Helper function implementation for Parquet writer
 */

#pragma once

#include <cudf/detail/utilities/linked_column.hpp>
#include <cudf/io/detail/parquet.hpp>

namespace cudf::io::parquet::detail {

/**
 * @brief Fill the table metadata with default column names.
 *
 * @param table_meta The table metadata to fill
 */
void fill_table_meta(table_input_metadata& table_meta);

/**
 * @brief Compute size (in bytes) of the data stored in the given column.
 *
 * @param column The input column
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @return The data size of the input
 */
[[nodiscard]] size_t column_size(column_view const& column, cuda::stream_ref stream);

/**
 * @brief Indicates if the column should be marked as nullable in the output schema
 *
 * Returns `true` if the input column is nullable or if the write mode is not set to
 * write the table all at once instead of chunked.
 *
 * @param column A view of the (linked) column
 * @param column_metadata Metadata of the column
 * @param write_mode Flag to indicate that we are guaranteeing a single table write
 *
 * @return Whether the column is nullable.
 */
[[nodiscard]] bool is_output_column_nullable(cudf::detail::LinkedColPtr const& column,
                                             column_in_metadata const& column_metadata,
                                             ::cudf::io::detail::single_write_mode write_mode);

}  // namespace cudf::io::parquet::detail
