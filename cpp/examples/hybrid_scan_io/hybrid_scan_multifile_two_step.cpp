/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "benchmark.hpp"
#include "common_utils.hpp"
#include "hybrid_scan_composer.hpp"
#include "io_source.hpp"

#include <cudf/ast/expressions.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/io/types.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_stream_pool.hpp>
#include <rmm/mr/statistics_resource_adaptor.hpp>

#include <ranges>
#include <stdexcept>
#include <string>
#include <thread>

/**
 * @file hybrid_scan_multifile.cpp
 *
 * @brief This example demonstrates reading multiple parquet files subject to a point lookup filter
 * in parallel using multiple threads where each thread reads a subset of files using the next-gen
 * parquet reader in two-step mode. Profiles collected with Nsight Systems should demonstrate near
 * perfect parallelism and pipelining across IO and compute tasks. Note that the input parquet files
 * must contain a page index for two-step read to be used.
 */

/**
 * @brief Functor to read a subset of parquet files for a given thread using the next-gen parquet
 * reader in two-step mode
 */
struct hybrid_scan_two_step_fn {
  std::vector<io_source> const& input_sources;
  std::vector<cudf::ast::operation> const& filter_expressions;
  std::unordered_set<hybrid_scan_filter_type> const& filters;
  int const num_threads;
  bool const verbose;
  rmm::cuda_stream_pool const& stream_pool;
  rmm::device_async_resource_ref mr;

  void operator()(int tid)
  {
    timer timer;

    auto constexpr single_step_read = false;
    auto constexpr use_page_index   = true;

    auto const filter_expression_opt = std::make_optional<cudf::ast::operation const>(
      filter_expressions[tid % filter_expressions.size()]);
    auto const stream = stream_pool.get_stream();

    auto strided_indices = std::views::iota(size_t{0}, input_sources.size()) |
                           std::views::filter([&](auto idx) { return idx % num_threads == tid; });
    for (auto source_idx : strided_indices) {
      std::ignore = hybrid_scan<single_step_read, use_page_index>(
        input_sources[source_idx], filter_expression_opt, filters, false, stream, mr);
    }

    stream.synchronize_no_throw();

    if (verbose) {
      std::cout << "Thread " << tid << " ";
      timer.print_elapsed_millis();
    }
  }
};

/**
 * @brief Function to print example usage and argument information
 */
void inline print_usage()
{
  std::cout
    << "\nUsage: hybrid_scan_multifile_two_step <comma delimited list of dirs and/or files> "
       "<input multiplier>\n"
       "                                       <thread count> <column name> <literal> "
       "<io source type>\n"
       "                                       <iterations> <verbose:Y/N>\n\n"

       "Available IO source types: FILEPATH (Default), HOST_BUFFER, PINNED_BUFFER\n\n"
       "Note: Provide as many arguments as you like in the above order. Default values\n"
       "      for the unprovided arguments will be used. All input parquet files will\n"
       "      be converted to the specified IO source type before reading\n\n";
}

/**
 * @brief Main for hybrid scan multifile example
 *
 * Command line parameters:
 * 1. parquet input file name/path (default: "example.parquet")
 * 2. input multiplier (default: 1)
 * 3. thread count (default: 2)
 * 4. column name for filter expression (default: "string_col")
 * 5. literal for filter expression (default: "0000001")
 * 6. io source type (default: "FILEPATH")
 * 7. iterations (default: 1)
 * 8. verbose (default: false)
 *
 * Example invocation from directory `cudf/cpp/examples/hybrid_scan`:
 * ./build/hybrid_scan_multifile_two_step example.parquet 8 2 string_col 0000001 FILEPATH 1 NO
 *
 */
int main(int argc, char const** argv)
{
  auto const max_threads = std::thread::hardware_concurrency();

  // Set arguments to defaults
  auto input_paths      = std::string{"example.parquet"};
  auto input_multiplier = 1;
  auto num_threads      = 2;
  auto column_name      = std::string{"string_col"};
  auto literal_value    = std::string{"0000001"};
  auto io_source_type   = io_source_type::FILEPATH;
  auto iterations       = 1;
  auto verbose          = false;

  // Set to the provided args
  switch (argc) {
    case 9: verbose = get_boolean(argv[8]); [[fallthrough]];
    case 8: iterations = std::stoi(argv[7]); [[fallthrough]];
    case 7: io_source_type = get_io_source_type(argv[6]); [[fallthrough]];
    case 6: literal_value = argv[5]; [[fallthrough]];
    case 5: column_name = argv[4]; [[fallthrough]];
    case 4:
      num_threads =
        std::min<int>(max_threads, std::max(num_threads, std::stoi(std::string{argv[3]})));
      [[fallthrough]];
    case 3:
      input_multiplier = std::min<int>(
        max_threads, std::max<int>(input_multiplier, std::stoi(std::string{argv[2]})));
      [[fallthrough]];
    case 2:
      // Check if instead of input_paths, the first argument is `-h` or `--help`
      if (auto arg = std::string{argv[1]};
          arg != "-h" and arg != "--help" and not arg.starts_with("-")) {
        input_paths = std::move(arg);
        break;
      }
      [[fallthrough]];
    default: print_usage(); throw std::runtime_error("Exiting...");
  }

  // Initialize mr, stream pool and default stream
  auto constexpr is_pool_used = false;
  auto resource               = create_memory_resource(is_pool_used);
  auto stream_pool = rmm::cuda_stream_pool(1 + num_threads, rmm::cuda_stream::flags::non_blocking);
  auto default_stream = stream_pool.get_stream();
  auto stats_mr       = rmm::mr::statistics_resource_adaptor{resource};
  rmm::mr::set_current_device_resource(stats_mr);

  // Create filter expressions (one per thread; reused circularly if needed)
  auto const column_reference = cudf::ast::column_name_reference(column_name);
  auto scalar                 = cudf::string_scalar(literal_value, true, default_stream);
  default_stream.synchronize();
  auto literal = cudf::ast::literal(scalar);

  std::vector<cudf::ast::operation> filter_expressions;
  filter_expressions.emplace_back(cudf::ast::ast_operator::EQUAL, column_reference, literal);

  // Insert which hybrid scan filters to apply
  std::unordered_set<hybrid_scan_filter_type> filters;
  {
    filters.insert(hybrid_scan_filter_type::ROW_GROUPS_WITH_STATS);
    filters.insert(hybrid_scan_filter_type::ROW_GROUPS_WITH_DICT_PAGES);
    filters.insert(hybrid_scan_filter_type::ROW_GROUPS_WITH_BLOOM_FILTERS);
    // Deliberately disabled as it has a high cost to benefit ratio
    // filters.insert(hybrid_scan_filter_type::FILTER_COLUMN_PAGES_WITH_PAGE_INDEX);
    filters.insert(hybrid_scan_filter_type::PAYLOAD_COLUMN_PAGES_WITH_ROW_MASK);
  }

  // List of input sources from the input_paths string.
  auto const input_sources = [&]() {
    try {
      return extract_input_sources(
        input_paths, input_multiplier, num_threads, io_source_type, default_stream);
    } catch (std::exception const& e) {
      print_usage();
      throw std::runtime_error(e.what());
    }
  }();
  if (input_sources.empty()) {
    print_usage();
    throw std::runtime_error("No input files to read. Exiting early.\n");
  }

  // Read the input parquet sources specified times with multiple threads and discard the output
  std::cout << "\nReading " << input_sources.size() << " input sources " << iterations
            << " time(s) using " << num_threads
            << " threads and discarding output "
               "tables..\n";

  std::cout << "Note that the first read may include times for nvcomp, cufile loading and RMM "
               "growth.\n\n";

  benchmark(
    [&] {
      auto hybrid_scan_fn = hybrid_scan_two_step_fn{.input_sources      = input_sources,
                                                    .filter_expressions = filter_expressions,
                                                    .filters            = filters,
                                                    .num_threads        = num_threads,
                                                    .verbose            = verbose,
                                                    .stream_pool        = stream_pool,
                                                    .mr                 = stats_mr};

      hybrid_scan_multifile(num_threads, hybrid_scan_fn);
    },
    iterations);

  // Print peak memory
  std::cout << "Peak memory: " << (stats_mr.get_bytes_counter().peak / 1'048'576.0) << " MB\n\n";

  return 0;
}
