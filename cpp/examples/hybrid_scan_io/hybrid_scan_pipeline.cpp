/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "benchmark.hpp"
#include "common_utils.hpp"
#include "io_source.hpp"
#include "io_utils.hpp"
#include "timer.hpp"

#include <cudf/column/column_factories.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/io/experimental/hybrid_scan.hpp>
#include <cudf/io/types.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_stream_pool.hpp>
#include <rmm/mr/statistics_resource_adaptor.hpp>

#include <cuda/iterator>
#include <thrust/host_vector.h>

#include <filesystem>
#include <string>

/**
 * @file hybrid_scan_pipeline.cpp
 *
 * @brief This example demonstrates splitting a parquet file into multiple partitions, reading them
 * in a pipelined fashion, and concatenating the results using the next-gen parquet reader. The same
 * file is also read using the libcudf main parquet reader and the read times are compared.
 */

namespace {

using table_ptr = std::unique_ptr<cudf::table>;

enum class split_strategy : uint8_t {
  ROW_GROUPS  = 0,  //< Split by row groups
  BYTE_RANGES = 1,  //< Split by byte ranges
};

/**
 * @brief Get split strategy from the string keyword argument
 *
 * @param name split strategy keyword name
 * @return split strategy enum value
 */
[[nodiscard]] split_strategy get_split_strategy(std::string name)
{
  std::transform(name.begin(), name.end(), name.begin(), ::toupper);

  if (name == "ROW_GROUPS") {
    return split_strategy::ROW_GROUPS;
  } else if (name == "BYTE_RANGES") {
    return split_strategy::BYTE_RANGES;
  }
  throw std::invalid_argument("Invalid split strategy");
}

/**
 * @brief Concatenate a vector of tables and return the resultant table
 *
 * @param tables Vector of tables to concatenate
 * @param stream CUDA stream to use
 *
 * @return Unique pointer to the resultant concatenated table.
 */
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
/**
 * @brief Read parquet input using the main parquet reader from io source
 *
 * @param io_source io source to read
 * @return cudf::io::table_with_metadata
 */
cudf::io::table_with_metadata read_parquet(io_source const& io_source, cuda::stream_ref stream)
{
  auto source_info = io_source.get_source_info();
  auto options     = cudf::io::parquet_reader_options::builder(source_info).build();
  return cudf::io::read_parquet(options, stream);
}

struct hybrid_scan_fn {
  std::reference_wrapper<table_ptr> table;
  std::unique_ptr<cudf::io::parquet::experimental::hybrid_scan_reader> reader;
  std::reference_wrapper<cudf::io::datasource> datasource;
  cudf::host_span<cudf::size_type const> row_groups_indices;
  bool use_page_index;
  cudf::io::parquet_reader_options const& options;
  cuda::stream_ref const stream;
  rmm::device_async_resource_ref const mr;
  void operator()() const
  {
    CUDF_FUNC_RANGE();

    if (use_page_index) {
      auto const page_index_byte_range = reader->page_index_byte_range();
      if (not page_index_byte_range.is_empty()) {
        auto const page_index_buffer = fetch_page_index_bytes(datasource, page_index_byte_range);
        reader->setup_page_index(*page_index_buffer);
      }
    }

    auto const all_column_chunk_byte_ranges =
      reader->all_column_chunks_byte_ranges(row_groups_indices, options);
    auto [all_column_chunk_buffers, all_column_chunk_data, all_col_read_tasks] =
      fetch_byte_ranges_async(datasource, all_column_chunk_byte_ranges, stream, mr);
    all_col_read_tasks.get();
    table.get() = std::move(
      reader
        ->materialize_all_columns(row_groups_indices, all_column_chunk_data, options, stream, mr)
        .tbl);
    stream.sync();
  }
};

/**
 * @brief Split parquet row groups into partitions and pipeline their reads using the next-gen
 * parquet reader
 *
 * @param io_source io source to read
 * @param num_partitions Number of read partitions
 * @param stream_pool CUDA stream pool to use
 * @param mr Device memory resource
 *
 * @return Tuple of filter table, payload table, filter metadata, payload metadata, and the final
 *         row validity column
 */
template <bool verbose>
auto hybrid_scan_pipelined(io_source const& io_source,
                           cudf::size_type num_partitions,
                           split_strategy split_strategy,
                           bool use_page_index,
                           rmm::cuda_stream_pool const& stream_pool,
                           rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();

  if constexpr (verbose) { std::cout << "\nREADER: Setup, metadata and page index...\n"; }
  timer timer;

  // Input file buffer span
  auto datasource     = std::move(cudf::io::make_datasources(io_source.get_source_info()).front());
  auto datasource_ref = std::ref(*datasource);

  // Fetch footer bytes and setup reader
  auto const footer_buffer = fetch_footer_bytes(datasource_ref);
  auto const options       = cudf::io::parquet_reader_options::builder().build();

  auto reader =
    std::make_unique<cudf::io::parquet::experimental::hybrid_scan_reader>(*footer_buffer, options);

  auto const metadata = std::move(reader->parquet_metadata());

  auto const row_groups_indices = reader->all_row_groups(options);

  if constexpr (verbose) {
    timer.print_elapsed_millis();
    std::cout << "Setup partitions... \n";
  }

  timer.reset();

  // Adjust the number of partitions if needed
  num_partitions = std::min<cudf::size_type>(num_partitions, row_groups_indices.size());

  std::vector<table_ptr> tables(num_partitions);
  std::vector<std::unique_ptr<cudf::io::parquet::experimental::hybrid_scan_reader>> readers{};
  readers.reserve(num_partitions);
  readers.emplace_back(std::move(reader));
  for (cudf::size_type i = 1; i < num_partitions; ++i) {
    readers.emplace_back(
      std::make_unique<cudf::io::parquet::experimental::hybrid_scan_reader>(metadata, options));
  }

  if constexpr (verbose) { timer.print_elapsed_millis(); }

  if (num_partitions > 1) {
    if constexpr (verbose) { std::cout << "Creating row group partitions... \n"; }
    timer.reset();
  }

  if (num_partitions == 1) {
    if constexpr (verbose) { std::cout << "Reading as single partition... \n"; }
    timer.reset();
    hybrid_scan_fn{.table              = std::ref(tables.front()),
                   .reader             = std::move(readers.front()),
                   .datasource         = datasource_ref,
                   .row_groups_indices = row_groups_indices,
                   .use_page_index     = use_page_index,
                   .options            = options,
                   .stream             = stream_pool.get_stream(),
                   .mr                 = mr}();
    if constexpr (verbose) { timer.print_elapsed_millis(); }
    return std::move(tables.front());
  }

  auto const total_row_groups            = row_groups_indices.size();
  auto const one_row_group_per_partition = std::cmp_equal(num_partitions, total_row_groups);

  std::vector<std::vector<cudf::size_type>> row_group_parts(num_partitions);
  if (one_row_group_per_partition or split_strategy == split_strategy::ROW_GROUPS) {
    auto const chunk_size = total_row_groups / num_partitions;
    auto const remainder  = total_row_groups % num_partitions;

    size_t offset = 0;
    for (cudf::size_type i = 0; i < num_partitions; ++i) {
      auto const part_size = chunk_size + (static_cast<size_t>(i) < remainder ? 1 : 0);
      row_group_parts[i].assign(row_groups_indices.begin() + offset,
                                row_groups_indices.begin() + offset + part_size);
      offset += part_size;
    }
  } else {
    auto const datasource_size       = datasource->size();
    auto const datasource_chunk_size = datasource_size / num_partitions;
    size_t datasource_offset         = 0;
    for (cudf::size_type i = 0; i < num_partitions; ++i) {
      auto split_options = cudf::io::parquet_reader_options::builder()
                             .skip_bytes(datasource_offset)
                             .num_bytes(datasource_chunk_size)
                             .build();
      auto filtered_row_groups =
        readers.front()->filter_row_groups_with_byte_range(row_groups_indices, split_options);
      if (filtered_row_groups.empty()) {
        num_partitions--;
        if constexpr (verbose) {
          std::cout << "Adjusting number of partitions to " << num_partitions << "\n";
        }
      } else {
        row_group_parts[i].assign(filtered_row_groups.begin(), filtered_row_groups.end());
      }
      datasource_offset += datasource_chunk_size;
    }
  }

  if constexpr (verbose) {
    timer.print_elapsed_millis();
    std::cout << "Pipelining " << num_partitions << " table reads... \n";
  }

  timer.reset();

  std::vector<hybrid_scan_fn> read_tasks;
  read_tasks.reserve(num_partitions);
  std::for_each(
    cuda::counting_iterator<cudf::size_type>{0},
    cuda::counting_iterator{num_partitions},
    [&](auto task_id) {
      read_tasks.emplace_back(hybrid_scan_fn{.table              = std::ref(tables[task_id]),
                                             .reader             = std::move(readers[task_id]),
                                             .datasource         = datasource_ref,
                                             .row_groups_indices = row_group_parts[task_id],
                                             .use_page_index     = use_page_index,
                                             .options            = options,
                                             .stream             = stream_pool.get_stream(),
                                             .mr                 = mr});
    });

  std::vector<std::thread> threads;
  threads.reserve(num_partitions);
  for (auto& task : read_tasks) {
    threads.emplace_back(std::move(task));
  }
  for (auto& t : threads) {
    t.join();
  }

  if constexpr (verbose) {
    timer.print_elapsed_millis();
    std::cout << "Concatenating tables... \n";
  }

  timer.reset();

  auto table = concatenate_tables(std::move(tables), stream_pool.get_stream());

  if constexpr (verbose) { timer.print_elapsed_millis(); }

  return std::move(table);
}

/**
 * @brief Function to print example usage and argument information.
 */
void inline print_usage()
{
  std::cout
    << std::endl
    << "Usage: hybrid_scan_pipeline <input parquet file> <number of partitions> <io source "
       "type> <split strategy> <iterations> <verbose:Y/N>\n\n"
    << "Available IO source types: FILEPATH, HOST_BUFFER (Default), PINNED_BUFFER, "
       "DEVICE_BUFFER \n\n"
    << "Available split strategies: ROW_GROUPS (Default), BYTE_RANGES \n\n"
    << "Example usage: hybrid_scan_pipeline example.parquet 2 HOST_BUFFER ROW_GROUPS 4 true \n\n";
}

}  // namespace

/**
 * @brief Main for hybrid scan pipelined example
 *
 * Command line parameters:
 * 1. parquet input file name/path (default: "example.parquet")
 * 2. number of read partitions (default: 2)
 * 3. io source type (default: "HOST_BUFFER")
 * 4. split strategy (default: "ROW_GROUPS")
 * 5. iterations (default: 2)
 * 6. verbose (default: false)
 *
 * Example invocation from directory `cudf/cpp/examples/hybrid_scan`:
 * ./build/hybrid_scan_pipeline example.parquet 2 HOST_BUFFER ROW_GROUPS 2 NO
 *
 */
int main(int argc, char const** argv)
{
  auto input_filepath = std::string{"example.parquet"};
  auto num_partitions = 2;
  auto io_source_type = io_source_type::FILEPATH;
  auto split_strategy = split_strategy::ROW_GROUPS;
  auto iterations     = 2;
  auto verbose        = false;

  switch (argc) {
    case 7: verbose = get_boolean(argv[6]); [[fallthrough]];
    case 6: iterations = std::stoi(argv[5]); [[fallthrough]];
    case 5: split_strategy = get_split_strategy(argv[4]); [[fallthrough]];
    case 4: io_source_type = get_io_source_type(argv[3]); [[fallthrough]];
    case 3: num_partitions = std::max<cudf::size_type>(1, std::stoi(argv[2])); [[fallthrough]];
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
  auto stream_pool =
    rmm::cuda_stream_pool(1 + num_partitions, rmm::cuda_stream::flags::non_blocking);
  auto default_stream = stream_pool.get_stream();
  auto mr             = create_memory_resource(is_pool_used);
  auto stats_mr       = rmm::mr::statistics_resource_adaptor{mr};
  rmm::mr::set_current_device_resource(stats_mr);

  // Create io source
  auto const data_source = io_source{input_filepath, io_source_type, default_stream};

  constexpr bool use_page_index = false;
  {
    std::cout << "Reading " << input_filepath
              << " with next-gen parquet reader and no page index...\n";
    benchmark(
      [&] {
        std::ignore = hybrid_scan_pipelined<false>(
          data_source, num_partitions, split_strategy, use_page_index, stream_pool, stats_mr);
      },
      iterations);

    std::cout << "Reading " << input_filepath << " with main parquet reader...\n";
    benchmark([&] { std::ignore = read_parquet(data_source, default_stream); }, iterations);
  }

  // Check for validity
  auto pipeline_table = [&] {
    if (verbose) {
      return hybrid_scan_pipelined<true>(
        data_source, num_partitions, split_strategy, use_page_index, stream_pool, stats_mr);
    } else {
      return hybrid_scan_pipelined<false>(
        data_source, num_partitions, split_strategy, use_page_index, stream_pool, stats_mr);
    }
  }();
  auto main_table = std::move(read_parquet(data_source, default_stream).tbl);
  check_tables_equal(pipeline_table->view(), main_table->view(), default_stream);

  return 0;
}
