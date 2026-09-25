/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * @file
 * @brief Parquet I/O and generated data sources for NDS-H benchmarks and query tests.
 */

#pragma once

#include "io/cuio_common.hpp"
#include "ndsh/utilities.hpp"

#include <cudf/io/types.hpp>

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

/**
 * @brief Read a parquet file into a table
 *
 * @param source_info The source of the parquet file
 * @param columns The columns to read
 * @param predicate The filter predicate to pushdown
 */
[[nodiscard]] std::unique_ptr<table_with_names> read_parquet(
  cudf::io::source_info const& source_info,
  std::vector<std::string> const& columns                = {},
  std::unique_ptr<cudf::ast::operation> const& predicate = nullptr);

/**
 * @brief Write a `cudf::table` to a parquet cuio sink
 *
 * @param table The `cudf::table` to write
 * @param col_names The column names of the table
 * @param source The source sink pair to write the table to
 */
void write_to_parquet_device_buffer(std::unique_ptr<cudf::table> const& table,
                                    std::vector<std::string> const& col_names,
                                    cuio_source_sink_pair& source);

/**
 * @brief Generate NDS-H tables and write to parquet device buffers
 *
 * @param scale_factor The scale factor of NDS-H tables to generate
 * @param table_names The names of the tables to generate
 * @param sources The parquet data sources to populate
 */
void generate_parquet_data_sources(double scale_factor,
                                   std::vector<std::string> const& table_names,
                                   std::unordered_map<std::string, cuio_source_sink_pair>& sources);
