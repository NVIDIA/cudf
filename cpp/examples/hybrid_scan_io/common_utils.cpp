/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "common_utils.hpp"

#include <cudf/ast/expressions.hpp>
#include <cudf/table/equality.hpp>
#include <cudf/table/table_view.hpp>

#include <rmm/mr/cuda_async_memory_resource.hpp>
#include <rmm/mr/cuda_memory_resource.hpp>
#include <rmm/mr/pool_memory_resource.hpp>

#include <cuda/iterator>
#include <cuda/stream>

#include <filesystem>
#include <string>
#include <vector>

/**
 * @file common_utils.cpp
 * @brief Definitions for common utilities for hybrid_scan examples
 */

bool get_boolean(std::string input)
{
  std::transform(input.begin(), input.end(), input.begin(), ::toupper);

  // Check if the input string matches to any of the following
  return input == "ON" or input == "TRUE" or input == "YES" or input == "Y" or input == "T";
}

cuda::mr::any_resource<cuda::mr::device_accessible> create_memory_resource(bool is_pool_used)
{
  if (is_pool_used) {
    return rmm::mr::pool_memory_resource{rmm::mr::cuda_memory_resource{},
                                         rmm::percent_of_free_device_memory(80)};
  }
  return rmm::mr::cuda_async_memory_resource{};
}

cudf::ast::operation create_filter_expression(std::string const& column_name,
                                              std::string const& literal_value)
{
  auto const column_reference = cudf::ast::column_name_reference(column_name);
  auto scalar                 = cudf::string_scalar(literal_value);
  auto literal                = cudf::ast::literal(scalar);
  return cudf::ast::operation(cudf::ast::ast_operator::EQUAL, column_reference, literal);
}

std::unique_ptr<cudf::table> combine_tables(std::unique_ptr<cudf::table> filter_table,
                                            std::unique_ptr<cudf::table> payload_table)
{
  auto filter_columns  = filter_table->release();
  auto payload_columns = payload_table->release();

  auto all_columns = std::vector<std::unique_ptr<cudf::column>>{};
  all_columns.reserve(filter_columns.size() + payload_columns.size());
  std::move(filter_columns.begin(), filter_columns.end(), std::back_inserter(all_columns));
  std::move(payload_columns.begin(), payload_columns.end(), std::back_inserter(all_columns));
  return std::make_unique<cudf::table>(std::move(all_columns));
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

std::vector<io_source> extract_input_sources(std::string const& paths,
                                             int32_t input_multiplier,
                                             int32_t thread_count,
                                             io_source_type io_source_type,
                                             cuda::stream_ref stream)
{
  // Get the delimited paths to directory and/or files.
  std::vector<std::string> const delimited_paths = [&]() {
    std::vector<std::string> paths_list;
    std::stringstream strstream{paths};
    std::string path;
    // Extract the delimited paths.
    while (std::getline(strstream, path, char{','})) {
      paths_list.push_back(path);
    }
    return paths_list;
  }();

  // List of parquet files
  std::vector<std::string> parquet_files;
  std::for_each(delimited_paths.cbegin(), delimited_paths.cend(), [&](auto const& path_string) {
    auto const path = std::filesystem::path{path_string};
    // If this is a parquet file, add it.
    if (std::filesystem::is_regular_file(path)) {
      parquet_files.push_back(path_string);
    }
    // If this is a directory, add all files in the directory.
    else if (std::filesystem::is_directory(path)) {
      for (auto const& file : std::filesystem::directory_iterator(path)) {
        if (std::filesystem::is_regular_file(file.path())) {
          parquet_files.push_back(file.path().string());
        } else {
          std::cout << "Skipping sub-directory: " << file.path().string() << "\n";
        }
      }
    } else {
      throw std::runtime_error("Encountered an invalid input path\n");
    }
  });

  // Current size of list of parquet files
  auto const initial_size = parquet_files.size();
  if (initial_size == 0) { throw std::runtime_error("No input files to read. Exiting early.\n"); }

  // Reserve space
  parquet_files.reserve(std::max<size_t>(thread_count, input_multiplier * parquet_files.size()));

  // Append the input files by input_multiplier times
  std::for_each(
    cuda::counting_iterator<int32_t>{1}, cuda::counting_iterator{input_multiplier}, [&](auto i) {
      parquet_files.insert(
        parquet_files.end(), parquet_files.begin(), parquet_files.begin() + initial_size);
    });

  if (parquet_files.size() < thread_count) {
    // Cycle append parquet files from the existing ones if less than the thread_count
    std::cout << "Warning: Number of input sources < thread count. Cycling from\n"
                 "and appending to current input sources such that the number of\n"
                 "input source == thread count\n";
    for (size_t idx = 0; thread_count > static_cast<int>(parquet_files.size()); idx++) {
      parquet_files.emplace_back(parquet_files[idx % initial_size]);
    }
  }

  // Vector of io sources
  std::vector<io_source> input_sources;
  input_sources.reserve(parquet_files.size());
  // Transform input files to the specified io sources
  std::transform(
    parquet_files.begin(),
    parquet_files.end(),
    std::back_inserter(input_sources),
    [&](auto const& file_name) { return io_source{file_name, io_source_type, stream}; });
  stream.sync();
  return input_sources;
}
