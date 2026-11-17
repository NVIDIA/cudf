/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "io_source.hpp"

#include <cudf/ast/expressions.hpp>
#include <cudf/io/datasource.hpp>
#include <cudf/io/text/byte_range_info.hpp>
#include <cudf/io/types.hpp>
#include <cudf/table/table_view.hpp>

#include <rmm/mr/cuda_async_memory_resource.hpp>
#include <rmm/mr/pool_memory_resource.hpp>
#include <rmm/resource_ref.hpp>

#include <cuda/memory_resource>
#include <cuda/stream>

#include <string>
#include <vector>

/**
 * @file common_utils.hpp
 * @brief Common utilities for hybrid_scan examples
 */

/**
 * @brief Get boolean from they keyword
 *
 * @param input keyword affirmation string such as: Y, T, YES, TRUE, ON
 * @return true or false
 */
[[nodiscard]] bool get_boolean(std::string input);

/**
 * @brief Create memory resource for libcudf functions
 *
 * @param pool Whether to use a pool memory resource.
 * @return Memory resource instance
 */
cuda::mr::any_resource<cuda::mr::device_accessible> create_memory_resource(bool is_pool_used);

/**
 * @brief Create a filter expression of the form `column_name == literal` for string type point
 * lookups
 *
 * @param column_name String column name
 * @param literal String literal value
 * @return Filter expression
 */
cudf::ast::operation create_filter_expression(std::string const& column_name,
                                              std::string const& literal_value);

/**
 * @brief Combine columns from filter and payload tables into a single table
 *
 * @param filter_table Filter table
 * @param payload_table Payload table
 * @return Combined table
 */
std::unique_ptr<cudf::table> combine_tables(std::unique_ptr<cudf::table> filter_table,
                                            std::unique_ptr<cudf::table> payload_table);

/**
 * @brief Check if two tables are identical, throw an error otherwise
 *
 * @param lhs_table View to lhs table
 * @param rhs_table View to rhs table
 * @param stream CUDA stream to use
 */
void check_tables_equal(cudf::table_view const& lhs_table,
                        cudf::table_view const& rhs_table,
                        cuda::stream_ref stream = cudf::get_default_stream());

/**
 * @brief Function to process comma delimited input paths string to parquet files and/or dirs
 *        and convert them to specified io sources.
 *
 * Process the input path string containing directories (of parquet files) and/or individual
 * parquet files into a list of input parquet files, multiple the list by `input_multiplier`,
 * make sure to have at least `thread_count` files to satisfy at least file per parallel thread,
 * and convert the final list of files to a list of `io_source` and return.
 *
 * @param paths Comma delimited input paths string
 * @param input_multiplier Multiplier for the input files list
 * @param thread_count Number of threads being used in the example
 * @param io_source_type Specified IO source type to convert input files to
 * @param stream CUDA stream to use
 *
 * @return Vector of input sources for the given paths
 */
[[nodiscard]] std::vector<io_source> extract_input_sources(std::string const& paths,
                                                           int32_t input_multiplier,
                                                           int32_t thread_count,
                                                           io_source_type io_source_type,
                                                           cuda::stream_ref stream);
