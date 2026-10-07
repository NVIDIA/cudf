/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "benchmark.hpp"
#include "common_utils.hpp"
#include "hybrid_scan_composer.hpp"
#include "io_source.hpp"

#include <cudf/table/table_view.hpp>

#include <rmm/mr/statistics_resource_adaptor.hpp>

#include <filesystem>
#include <string>
#include <unordered_set>

/**
 * @file hybrid_scan_io.cpp
 *
 * @brief This example demonstrates the use of libcudf next-gen parquet reader to optimally read
 * a parquet file subject to a highly selective string-type point lookup (col_name ==
 * literal) filter. The same file is also read using the libcudf legacy parquet reader and the read
 * times are compared.
 */

namespace {

/**
 * @brief Read parquet input using the legacy parquet reader from io source
 *
 * @param io_source io source to read
 * @return cudf::io::table_with_metadata
 */
cudf::io::table_with_metadata read_parquet(io_source const& io_source,
                                           cudf::ast::operation const& filter_expression,
                                           cuda::stream_ref stream)
{
  auto source_info = io_source.get_source_info();
  auto options =
    cudf::io::parquet_reader_options::builder(source_info).filter(filter_expression).build();
  return cudf::io::read_parquet(options);
}

/**
 * @brief Function to print example usage and argument information.
 */
void inline print_usage()
{
  std::cout
    << std::endl
    << "Usage: hybrid_scan <input parquet file> <column name> <literal> <io source type>\n"
    << "                   <iterations> <verbose:Y/N>\n\n"
    << "Available IO source types: FILEPATH, HOST_BUFFER, PINNED_BUFFER (Default), "
       "DEVICE_BUFFER\n\n"
    << "Note: Both the column name and literal must be of `string` type. The constructed filter\n"
    << "      expression will be of the form <column name> == <literal>\n\n"
    << "Example usage: hybrid_scan example.parquet string_col 0000001 PINNED_BUFFER 4 N\n\n";
}

}  // namespace

/**
 * @brief Main for hybrid scan example
 *
 * Command line parameters:
 * 1. parquet input file name/path (default: "example.parquet")
 * 2. column name for filter expression (default: "string_col")
 * 3. literal for filter expression (default: "0000001")
 * 4. io source type (default: "PINNED_BUFFER")
 * 5. iterations (default: 2)
 * 6. verbose (default: false)
 *
 * The filter expression will be of the form col_name == literal (default: string_col == 0000001)
 *
 * Example invocation from directory `cudf/cpp/examples/hybrid_scan`:
 * ./build/hybrid_scan example.parquet string_col 0000001 PINNED_BUFFER 2 NO
 *
 */
int main(int argc, char const** argv)
{
  auto input_filepath = std::string{"example.parquet"};
  auto column_name    = std::string{"string_col"};
  auto literal_value  = std::string{"0000001"};
  auto io_source_type = io_source_type::PINNED_BUFFER;
  auto iterations     = 2;
  auto verbose        = false;

  switch (argc) {
    case 7: verbose = get_boolean(argv[6]); [[fallthrough]];
    case 6: iterations = std::stoi(argv[5]); [[fallthrough]];
    case 5: io_source_type = get_io_source_type(argv[4]); [[fallthrough]];
    case 4: literal_value = argv[3]; [[fallthrough]];
    case 3: column_name = argv[2]; [[fallthrough]];
    case 2:  // Check if instead of input_paths, the first argument is `-h` or `--help`
      if (auto arg = std::string{argv[1]}; arg != "-h" and arg != "--help") {
        input_filepath = std::move(arg);
        break;
      }
      [[fallthrough]];
    default: print_usage(); throw std::invalid_argument("Invalid arguments");
  }

  // Check if input file exists
  if (not std::filesystem::is_regular_file(input_filepath)) {
    throw std::runtime_error("Input file does not exist");
  }

  // Initialize mr, default stream and stream pool
  auto constexpr is_pool_used = false;
  auto stream                 = cudf::get_default_stream();
  auto resource               = create_memory_resource(is_pool_used);
  auto stats_mr               = rmm::mr::statistics_resource_adaptor{resource};
  rmm::mr::set_current_device_resource(stats_mr);

  // Create filter expression
  auto const column_reference = cudf::ast::column_name_reference(column_name);
  auto scalar                 = cudf::string_scalar(literal_value, true, stream);
  stream.sync();
  auto literal = cudf::ast::literal(scalar);
  auto filter_expression =
    cudf::ast::operation(cudf::ast::ast_operator::EQUAL, column_reference, literal);

  // Create io source
  auto const data_source = io_source{input_filepath, io_source_type, stream};

  // Insert which filters to apply
  std::unordered_set<hybrid_scan_filter_type> filters;
  {
    filters.insert(hybrid_scan_filter_type::ROW_GROUPS_WITH_STATS);
    filters.insert(hybrid_scan_filter_type::ROW_GROUPS_WITH_DICT_PAGES);
    filters.insert(hybrid_scan_filter_type::ROW_GROUPS_WITH_BLOOM_FILTERS);
    // Deliberately disabled as it has a high cost to benefit ratio
    // filters.insert(hybrid_scan_filter_type::FILTER_COLUMN_PAGES_WITH_PAGE_INDEX);
    filters.insert(hybrid_scan_filter_type::PAYLOAD_COLUMN_PAGES_WITH_ROW_MASK);
  }

  // Hybrid scan parameters (must use page index for two-step read)
  auto constexpr single_step_read = false;
  auto constexpr use_page_index   = true;

  auto const filter_expression_opt =
    std::make_optional<cudf::ast::operation const>(filter_expression);
  {
    std::cout << "Benchmarking " << input_filepath << "read with next-gen parquet reader...\n";
    benchmark(
      [&] {
        std::ignore = hybrid_scan<single_step_read, use_page_index>(
          data_source, filter_expression_opt, filters, false, stream, stats_mr);
      },
      iterations);

    std::cout << "Benchmarking " << input_filepath << "read with main parquet reader...\n";
    benchmark([&] { std::ignore = read_parquet(data_source, filter_expression, stream); },
              iterations);
  }

  // Check for validity
  auto table_next_gen_reader = hybrid_scan<single_step_read, use_page_index>(
    data_source, filter_expression_opt, filters, verbose, stream, stats_mr);
  auto table_main_reader = std::move(read_parquet(data_source, filter_expression, stream).tbl);
  check_tables_equal(table_next_gen_reader->view(), table_main_reader->view(), stream);

  return 0;
}
