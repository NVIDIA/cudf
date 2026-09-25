/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/column/column.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

/**
 * @brief Create a column from the given integer vector
 *
 * @param column_data  Input vector that the column will be created from
 * @return             Column with content matching that of the given integer vector
 */
std::unique_ptr<cudf::column> make_column_from_vector(std::vector<int32_t> const& column_data);

/**
 * @brief Create a table with the given row and column counts
 *
 * @param row_count     Number of rows in created table
 * @param column_count  Number of columns in created table
 * @return              Table with the given row and column counts
 */
cudf::table make_table(size_t row_count, size_t column_count);

/**
 * @brief Write out the given table view content to a string
 *
 * @param tbl_view  Input table view whose content will be written to a string
 * @return          String representation of the table view
 */
std::string table_view_to_string(cudf::table_view const& tbl_view);

/**
 * @brief Print the content of the given table view to the console
 *
 * @param header    Header that will be printed before table view content
 * @param tbl_view  Table view whose content will be printed to the console
 */
void print_table(std::string const& header, cudf::table_view const& tbl_view);
