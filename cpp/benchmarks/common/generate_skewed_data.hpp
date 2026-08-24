/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/column/column.hpp>

#include <string_view>

/** @brief Target substring embedded in skewed string benchmark data. */
constexpr std::string_view skewed_string_target_substring{"0987 5W43"};

/**
 * @brief Deterministically generates a string column with a skewed length distribution.
 *
 * The column is built from 10 template strings. The first template contains
 * `skewed_string_target_substring` within its prefix so that short rows still match.
 *
 * Rows are then assigned one of two lengths:
 * - `short_string_pct` percent of rows repeat the template until `short_length`
 * - The remaining rows repeat the template until `long_tail_length`
 *
 * Within the column, `hit_rate` percent of rows contain the target substring used by the
 * find/contains string benchmarks. Because hit rows are scattered uniformly, both the short
 * and long rows contain the target at approximately `hit_rate` percent.
 *
 * @param num_rows Number of rows in the output column
 * @param short_length Length of the cropped short strings
 * @param long_tail_length Length of the repeated long tail rows
 * @param short_string_pct Percentage of rows that are short strings (0-100)
 * @param hit_rate Percentage of rows that contain the target substring (0-100)
 */
std::unique_ptr<cudf::column> create_skewed_string_column(cudf::size_type num_rows,
                                                          cudf::size_type short_length,
                                                          cudf::size_type long_tail_length,
                                                          int32_t short_string_pct,
                                                          int32_t hit_rate);
