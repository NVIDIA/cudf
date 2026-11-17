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
#include <cudf/utilities/span.hpp>

#include <optional>

namespace cudf::io::parquet::detail {

struct PageFragment;
struct parquet_column_device_view;

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

/**
 * @brief Compute a smaller row span that keeps each page fragment within the Parquet page size
 * limit
 *
 * A fragment is never split across pages, so a fragment whose data and level bytes do not fit in
 * a single page has to be re-measured over fewer rows.
 *
 * @param fragments 2D span of measured fragments [column_idx][fragment_idx]
 * @param col_desc Span of column descriptors [column_idx]
 * @param input_fragment_size Input fragment size
 * @return New fragment size, or `std::nullopt` if all fragments already fit
 *
 * @throws std::overflow_error if a single row does not fit in a page
 */
[[nodiscard]] std::optional<size_type> compute_smaller_fragment_size(
  cudf::detail::host_2dspan<PageFragment const> fragments,
  host_span<parquet_column_device_view const> col_desc,
  size_type input_fragment_size);

}  // namespace cudf::io::parquet::detail
