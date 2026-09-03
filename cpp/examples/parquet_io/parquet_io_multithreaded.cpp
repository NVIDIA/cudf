/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "benchmark.hpp"
#include "common_utils.hpp"
#include "io_source.hpp"

#include <cudf/concatenate.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/io/types.hpp>
#include <cudf/table/table_view.hpp>

#include <rmm/cuda_stream_pool.hpp>
#include <rmm/mr/statistics_resource_adaptor.hpp>

#include <cuda/iterator>

#include <exception>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <thread>

/**
 * @file parquet_io_multithreaded.cpp
 * @brief Demonstrates reading parquet data from the specified io source using multiple threads.
 *
 * The input parquet data is provided via files which are converted to the specified io source type
 * to be read using multiple threads. Optionally, the parquet data read by each thread can be
 * written to corresponding files and checked for validatity of the output files against the input
 * data.
 *
 * Run: ``parquet_io_multithreaded -h`` to see help with input args and more information.
 *
 * The following io source types are supported:
 * IO source types: FILEPATH, HOST_BUFFER, PINNED_BUFFER, DEVICE_BUFFER
 *
 */

// Type alias for unique ptr to cudf table
using table_t = std::unique_ptr<cudf::table>;

/**
 * @brief Behavior when handling the read tables by multiple threads
 */
enum class read_mode {
  NO_CONCATENATE,      ///< Only read and discard tables
  CONCATENATE_THREAD,  ///< Read and concatenate tables from each thread
  CONCATENATE_ALL,     ///< Read and concatenate everything to a single table
};

/**
 * @brief Functor for multithreaded parquet reading based on the provided read_mode
 */
template <read_mode read_mode>
struct read_fn {
  std::vector<io_source> const& input_sources;
  std::vector<table_t>& tables;
  int const thread_id;
  int const thread_count;
  cuda::stream_ref stream;

  void operator()()
  {
    // Tables read by this thread
    std::vector<table_t> tables_this_thread;

    // Sweep the available input files
    for (auto curr_file_idx = thread_id; curr_file_idx < input_sources.size();
         curr_file_idx += thread_count) {
      auto builder =
        cudf::io::parquet_reader_options::builder(input_sources[curr_file_idx].get_source_info());
      auto const options = builder.build();
      if constexpr (read_mode != read_mode::NO_CONCATENATE) {
        tables_this_thread.push_back(cudf::io::read_parquet(options, stream).tbl);
      } else {
        cudf::io::read_parquet(options, stream);
      }
    }

    // Concatenate the tables read by this thread if not NO_CONCATENATE read_mode.
    if constexpr (read_mode != read_mode::NO_CONCATENATE) {
      auto table = concatenate_tables(std::move(tables_this_thread), stream);
      stream.sync();
      tables[thread_id] = std::move(table);
    } else {
      // Just synchronize this stream and exit
      stream.sync();
    }
  }
};

/**
 * @brief Function to setup and launch multithreaded parquet reading.
 *
 * @tparam read_mode Specifies if to concatenate and return the actual
 *                    tables or discard them and return an empty vector
 *
 * @param input_sources List of input sources to read
 * @param thread_count Number of threads
 * @param stream_pool CUDA stream pool to use for threads
 *
 * @return Vector of read tables.
 */
template <read_mode read_mode>
std::vector<table_t> read_parquet_multithreaded(std::vector<io_source> const& input_sources,
                                                int32_t thread_count,
                                                rmm::cuda_stream_pool& stream_pool)
{
  // Tables read by each thread
  std::vector<table_t> tables(thread_count);

  // Table reading tasks
  std::vector<read_fn<read_mode>> read_tasks;
  read_tasks.reserve(thread_count);

  // Create the read tasks
  std::for_each(
    cuda::counting_iterator<int32_t>{0}, cuda::counting_iterator{thread_count}, [&](auto tid) {
      read_tasks.emplace_back(
        read_fn<read_mode>{input_sources, tables, tid, thread_count, stream_pool.get_stream()});
    });

  // Create threads with tasks
  std::vector<std::exception_ptr> exceptions(read_tasks.size());
  std::vector<std::thread> threads;
  threads.reserve(thread_count);
  for (std::size_t i = 0; i < read_tasks.size(); ++i) {
    threads.emplace_back([task = read_tasks[i], &exception = exceptions[i]]() mutable {
      try {
        task();
      } catch (...) {
        exception = std::current_exception();
      }
    });
  }
  for (auto& t : threads) {
    t.join();
  }
  for (auto const& exception : exceptions) {
    if (exception) { std::rethrow_exception(exception); }
  }

  // If CONCATENATE_ALL mode, then concatenate to a vector of one final table.
  if (read_mode == read_mode::CONCATENATE_ALL) {
    auto stream    = stream_pool.get_stream();
    auto final_tbl = concatenate_tables(std::move(tables), stream);
    stream.synchronize();
    tables.clear();
    tables.emplace_back(std::move(final_tbl));
  }

  return tables;
}

/**
 * @brief Functor for multithreaded parquet writing
 */
struct write_fn {
  std::string const& output_path;
  std::vector<cudf::table_view> const& table_views;
  int const thread_id;
  cuda::stream_ref stream;

  void operator()()
  {
    // Create a sink
    cudf::io::sink_info const sink_info{output_path + "/table_" + std::to_string(thread_id) +
                                        ".parquet"};
    // Writer options builder
    auto builder = cudf::io::parquet_writer_options::builder(sink_info, table_views[thread_id]);
    // Create a new metadata for the table
    auto table_metadata = cudf::io::table_input_metadata{table_views[thread_id]};

    builder.metadata(table_metadata);
    auto options = builder.build();

    // Write parquet data
    cudf::io::write_parquet(options, stream);

    // Done with this stream
    stream.sync();
  }
};

/**
 * @brief Function to setup and launch multithreaded writing parquet files.
 *
 * @param output_path Path to output directory
 * @param tables List of at least table views to be written
 * @param thread_count Number of threads to use for writing tables.
 * @param stream_pool CUDA stream pool to use for threads
 *
 */
void write_parquet_multithreaded(std::string const& output_path,
                                 std::vector<cudf::table_view> const& tables,
                                 int32_t thread_count,
                                 rmm::cuda_stream_pool& stream_pool)
{
  // Table writing tasks
  std::vector<write_fn> write_tasks;
  write_tasks.reserve(thread_count);
  std::for_each(
    cuda::counting_iterator<int32_t>{0}, cuda::counting_iterator{thread_count}, [&](auto tid) {
      write_tasks.emplace_back(write_fn{output_path, tables, tid, stream_pool.get_stream()});
    });

  // Writer threads
  std::vector<std::exception_ptr> exceptions(write_tasks.size());
  std::vector<std::thread> threads;
  threads.reserve(thread_count);
  for (std::size_t i = 0; i < write_tasks.size(); ++i) {
    threads.emplace_back([task = write_tasks[i], &exception = exceptions[i]]() mutable {
      try {
        task();
      } catch (...) {
        exception = std::current_exception();
      }
    });
  }
  for (auto& t : threads) {
    t.join();
  }
  for (auto const& exception : exceptions) {
    if (exception) { std::rethrow_exception(exception); }
  }
}

/**
 * @brief Function to print example usage and argument information.
 */
void print_usage()
{
  std::cout
    << "\nUsage: parquet_io_multithreaded <comma delimited list of dirs and/or files> <input "
       "multiplier>\n"
       "                                <io source type> <number of times to read> <thread count>\n"
       "                                <write to temp output files and validate: "
       "yes/no>\n\n"
       "Available IO source types: FILEPATH, HOST_BUFFER, PINNED_BUFFER (Default), "
       "DEVICE_BUFFER\n\n"
       "Note: Provide as many arguments as you like in the above order. Default values\n"
       "      for the unprovided arguments will be used. All input parquet files will\n"
       "      be converted to the specified IO source type before reading\n\n";
}

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
    std::filesystem::path path{path_string};
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
      print_usage();
      throw std::runtime_error("Encountered an invalid input path\n");
    }
  });

  // Current size of list of parquet files
  auto const initial_size = parquet_files.size();
  if (initial_size == 0) { return {}; }

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

/**
 * @brief The main function
 */
int32_t main(int argc, char const** argv)
{
  // Set arguments to defaults
  std::string input_paths       = "example.parquet";
  int32_t input_multiplier      = 1;
  int32_t num_reads             = 1;
  int32_t thread_count          = 1;
  io_source_type io_source_type = io_source_type::PINNED_BUFFER;
  bool write_and_validate       = false;

  // Set to the provided args
  switch (argc) {
    case 7: write_and_validate = get_boolean(argv[6]); [[fallthrough]];
    case 6: thread_count = std::max(thread_count, std::stoi(std::string{argv[5]})); [[fallthrough]];
    case 5: num_reads = std::max(num_reads, std::stoi(argv[4])); [[fallthrough]];
    case 4: io_source_type = get_io_source_type(argv[3]); [[fallthrough]];
    case 3:
      input_multiplier = std::max(input_multiplier, std::stoi(std::string{argv[2]}));
      [[fallthrough]];
    case 2:
      // Check if instead of input_paths, the first argument is `-h` or `--help`
      if (auto arg = std::string{argv[1]}; arg != "-h" and arg != "--help") {
        input_paths = std::move(arg);
        break;
      }
      [[fallthrough]];
    default: print_usage(); throw std::runtime_error("");
  }

  // Initialize mr, default stream and stream pool
  bool constexpr is_pool_used = true;
  auto resource               = create_memory_resource(is_pool_used);
  auto default_stream         = cudf::get_default_stream();
  auto stream_pool = rmm::cuda_stream_pool(thread_count, rmm::cuda_stream::flags::non_blocking);
  auto stats_mr    = rmm::mr::statistics_resource_adaptor{resource};
  rmm::mr::set_current_device_resource(stats_mr);

  // List of input sources from the input_paths string.
  auto const input_sources = extract_input_sources(
    input_paths, input_multiplier, thread_count, io_source_type, default_stream);

  // Check if there is nothing to do
  if (input_sources.empty()) {
    print_usage();
    throw std::runtime_error("No input files to read. Exiting early.\n");
  }

  // Read the same parquet files specified times with multiple threads and discard the read tables
  {
    // Print status
    std::cout << "\nReading " << input_sources.size() << " input sources " << num_reads
              << " time(s) using " << thread_count
              << " threads and discarding output "
                 "tables..\n";

    if (io_source_type == io_source_type::FILEPATH) {
      std::cout << "Note that the first read may include times for nvcomp, cufile loading and RMM "
                   "growth.\n\n";
    }

    benchmark(
      [&] {
        std::ignore = read_parquet_multithreaded<read_mode::NO_CONCATENATE>(
          input_sources, thread_count, stream_pool);
        default_stream.sync();
      },
      num_reads);

    std::cout << "Peak memory: " << (stats_mr.get_bytes_counter().peak / 1'048'576.0) << " MB\n\n";
  }

  // Write parquet files and validate if needed
  if (write_and_validate) {
    // read_mode::CONCATENATE_THREADS returns a vector of `thread_count` tables
    auto const tables = read_parquet_multithreaded<read_mode::CONCATENATE_THREAD>(
      input_sources, thread_count, stream_pool);
    default_stream.sync();

    // Construct a vector of table views for write_parquet_multithreaded
    auto const table_views = [&tables]() {
      std::vector<cudf::table_view> table_views;
      table_views.reserve(tables.size());
      std::transform(
        tables.cbegin(), tables.cend(), std::back_inserter(table_views), [](auto const& tbl) {
          return tbl->view();
        });
      return table_views;
    }();

    // Write tables to parquet
    std::cout << "Writing parquet output files..\n";

    // Create a directory at the tmpdir path.
    std::string output_path =
      std::filesystem::temp_directory_path().string() + "/output_" + current_date_and_time();
    std::filesystem::create_directory({output_path});

    benchmark(
      [&] {
        write_parquet_multithreaded(output_path, table_views, thread_count, stream_pool);
        default_stream.sync();
      },
      1);

    // Verify the output
    std::cout << "Verifying output..\n";

    // Simply concatenate the previously read tables from input sources
    auto const input_table = cudf::concatenate(table_views, default_stream);

    // Sources from written parquet files
    auto const written_pq_sources = extract_input_sources(
      output_path, input_multiplier, thread_count, io_source_type, default_stream);

    // read_mode::CONCATENATE_ALL returns a concatenated vector of 1 table only
    auto const transcoded_table = std::move(read_parquet_multithreaded<read_mode::CONCATENATE_ALL>(
                                              written_pq_sources, thread_count, stream_pool)
                                              .back());
    default_stream.sync();

    // Check if the tables are identical
    check_tables_equal(input_table->view(), transcoded_table->view(), default_stream);

    // Remove the created temp directory and parquet data
    std::filesystem::remove_all(output_path);
  }

  return 0;
}
