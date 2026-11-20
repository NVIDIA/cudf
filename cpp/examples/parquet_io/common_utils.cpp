/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "common_utils.hpp"

#include <cudf/concatenate.hpp>
#include <cudf/io/types.hpp>
#include <cudf/table/equality.hpp>
#include <cudf/table/table_view.hpp>

#include <rmm/mr/cuda_async_memory_resource.hpp>
#include <rmm/mr/cuda_memory_resource.hpp>
#include <rmm/mr/pool_memory_resource.hpp>

#include <cuda/stream>

#include <chrono>
#include <iomanip>
#include <string>

/**
 * @file common_utils.cpp
 * @brief Definitions for common utilities for `parquet_io` examples
 *
 */

cuda::mr::any_resource<cuda::mr::device_accessible> create_memory_resource(bool is_pool_used)
{
  if (is_pool_used) {
    return rmm::mr::pool_memory_resource{rmm::mr::cuda_memory_resource{},
                                         rmm::percent_of_free_device_memory(80)};
  }
  return rmm::mr::cuda_async_memory_resource{};
}

cudf::io::column_encoding get_encoding_type(std::string name)
{
  using encoding_type = cudf::io::column_encoding;

  static std::unordered_map<std::string_view, encoding_type> const map = {
    {"DEFAULT", encoding_type::USE_DEFAULT},
    {"DICTIONARY", encoding_type::DICTIONARY},
    {"PLAIN", encoding_type::PLAIN},
    {"DELTA_BINARY_PACKED", encoding_type::DELTA_BINARY_PACKED},
    {"DELTA_LENGTH_BYTE_ARRAY", encoding_type::DELTA_LENGTH_BYTE_ARRAY},
    {"DELTA_BYTE_ARRAY", encoding_type::DELTA_BYTE_ARRAY},
  };

  std::transform(name.begin(), name.end(), name.begin(), ::toupper);
  if (map.find(name) != map.end()) { return map.at(name); }
  throw std::invalid_argument(name +
                              " is not a valid encoding type.\n\n"
                              "Available encoding types: DEFAULT, DICTIONARY, PLAIN,\n"
                              "DELTA_BINARY_PACKED, DELTA_LENGTH_BYTE_ARRAY,\n"
                              "DELTA_BYTE_ARRAY\n\n");
}

cudf::io::compression_type get_compression_type(std::string name)
{
  using compression_type = cudf::io::compression_type;

  static std::unordered_map<std::string_view, compression_type> const map = {
    {"NONE", compression_type::NONE},
    {"AUTO", compression_type::AUTO},
    {"SNAPPY", compression_type::SNAPPY},
    {"LZ4", compression_type::LZ4},
    {"ZSTD", compression_type::ZSTD}};

  std::transform(name.begin(), name.end(), name.begin(), ::toupper);
  if (map.find(name) != map.end()) { return map.at(name); }
  throw std::invalid_argument(name +
                              " is not a valid compression type.\n\n"
                              "Available compression types: NONE, AUTO, SNAPPY,\n"
                              "LZ4, ZSTD\n\n");
}

bool get_boolean(std::string input)
{
  std::transform(input.begin(), input.end(), input.begin(), ::toupper);

  // Check if the input string matches to any of the following
  return input == "ON" or input == "TRUE" or input == "YES" or input == "Y" or input == "T";
}

void check_tables_equal(cudf::table_view const& lhs_table,
                        cudf::table_view const& rhs_table,
                        cuda::stream_ref stream)
{
  auto const tables_equal =
    cudf::tables_equal(lhs_table, rhs_table, cudf::null_equality::EQUAL, stream);
  std::cout << "Tables identical: " << std::boolalpha << tables_equal << "\n\n";
  if (not tables_equal) { throw std::logic_error("Table equality check failed"); }
}

std::unique_ptr<cudf::table> concatenate_tables(std::vector<std::unique_ptr<cudf::table>> tables,
                                                cuda::stream_ref stream)
{
  if (tables.size() == 1) { return std::move(tables[0]); }

  std::vector<cudf::table_view> table_views;
  table_views.reserve(tables.size());
  std::transform(
    tables.begin(), tables.end(), std::back_inserter(table_views), [&](auto const& tbl) {
      return tbl->view();
    });
  // Construct the final table
  return cudf::concatenate(table_views, stream);
}

std::string current_date_and_time()
{
  auto const time       = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
  auto const local_time = *std::localtime(&time);
  // Stringstream to format the date and time
  std::stringstream ss;
  ss << std::put_time(&local_time, "%Y-%m-%d-%H-%M-%S");
  return ss.str();
}
