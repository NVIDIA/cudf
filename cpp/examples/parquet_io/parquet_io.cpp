/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "common_utils.hpp"
#include "timer.hpp"

#include <cudf/io/parquet.hpp>
#include <cudf/io/types.hpp>
#include <cudf/table/table_view.hpp>

#include <string>

/**
 * @file parquet_io.cpp
 * @brief Demonstrates usage of the libcudf APIs to read and write
 * parquet file format with different encodings and compression types
 *
 * The following encoding and compression ztypes are demonstrated:
 * Encoding Types: DEFAULT, DICTIONARY, PLAIN, DELTA_BINARY_PACKED,
 *                 DELTA_LENGTH_BYTE_ARRAY, DELTA_BYTE_ARRAY
 *
 * Compression Types: NONE, AUTO, SNAPPY, LZ4, ZSTD
 *
 */

/**
 * @brief Read parquet input from file
 *
 * @param filepath path to input parquet file
 * @return cudf::io::table_with_metadata
 */
cudf::io::table_with_metadata read_parquet(std::string filepath)
{
  auto source_info = cudf::io::source_info(filepath);
  auto builder     = cudf::io::parquet_reader_options::builder(source_info);
  auto options     = builder.build();
  return cudf::io::read_parquet(options);
}

/**
 * @brief Write parquet output to file
 *
 * @param input table to write
 * @param metadata metadata of input table read by parquet reader
 * @param filepath path to output parquet file
 * @param stats_level optional page size stats level
 */
void write_parquet(cudf::table_view input,
                   cudf::io::table_metadata metadata,
                   std::string filepath,
                   cudf::io::column_encoding encoding,
                   cudf::io::compression_type compression,
                   std::optional<cudf::io::statistics_freq> stats_level)
{
  // write the data for inspection
  auto sink_info      = cudf::io::sink_info(filepath);
  auto builder        = cudf::io::parquet_writer_options::builder(sink_info, input);
  auto table_metadata = cudf::io::table_input_metadata{metadata};

  std::for_each(table_metadata.column_metadata.begin(),
                table_metadata.column_metadata.end(),
                [=](auto& col_meta) { col_meta.set_encoding(encoding); });

  builder.metadata(table_metadata);
  auto options = builder.build();
  options.set_compression(compression);
  // Either use the input stats level or don't write stats
  options.set_stats_level(stats_level.value_or(cudf::io::statistics_freq::STATISTICS_NONE));

  // write parquet data
  cudf::io::write_parquet(options);
}

/**
 * @brief Function to print example usage and argument information.
 */
void print_usage()
{
  std::cout << "\nUsage: parquet_io <input parquet file> <output parquet file> <encoding type>\n"
               "                  <compression type> <write page stats: yes/no>\n\n"
               "Available encoding types: DEFAULT, DICTIONARY, PLAIN, DELTA_BINARY_PACKED,\n"
               "                 DELTA_LENGTH_BYTE_ARRAY, DELTA_BYTE_ARRAY\n\n"
               "Available compression types: NONE, AUTO, SNAPPY, LZ4, ZSTD\n\n";
}

/**
 * @brief Main for nested_types examples
 *
 * Command line parameters:
 * 1. parquet input file name/path (default: "example.parquet")
 * 2. parquet output file name/path (default: "output.parquet")
 * 3. encoding type for columns (default: "DELTA_BINARY_PACKED")
 * 4. compression type (default: "ZSTD")
 * 5. optional: use page size stats metadata (default: "NO")
 *
 * Example invocation from directory `cudf/cpp/examples/parquet_io`:
 * ./build/parquet_io example.parquet output.parquet DELTA_BINARY_PACKED ZSTD
 *
 */
int main(int argc, char const** argv)
{
  std::string input_filepath                          = "example.parquet";
  std::string output_filepath                         = "output.parquet";
  cudf::io::column_encoding encoding                  = get_encoding_type("DELTA_BINARY_PACKED");
  cudf::io::compression_type compression              = get_compression_type("ZSTD");
  std::optional<cudf::io::statistics_freq> page_stats = std::nullopt;

  switch (argc) {
    case 6:
      page_stats = get_boolean(argv[5])
                     ? std::make_optional(cudf::io::statistics_freq::STATISTICS_COLUMN)
                     : std::nullopt;
      [[fallthrough]];
    case 5: compression = get_compression_type(argv[4]); [[fallthrough]];
    case 4: encoding = get_encoding_type(argv[3]); [[fallthrough]];
    case 3: output_filepath = argv[2]; [[fallthrough]];
    case 2:  // Check if instead of input_paths, the first argument is `-h` or `--help`
      if (auto arg = std::string{argv[1]}; arg != "-h" and arg != "--help") {
        input_filepath = std::move(arg);
        break;
      }
      [[fallthrough]];
    default: print_usage(); throw std::runtime_error("");
  }

  auto resource = create_memory_resource();
  cudf::set_current_device_resource(resource);

  // Read input parquet file
  // We do not want to time the initial read time as it may include
  // time for nvcomp, cufile loading and RMM growth
  std::cout << "\nReading " << input_filepath << "...\n";
  std::cout << "Note: Not timing the initial parquet read as it may include\n"
               "times for nvcomp, cufile loading and RMM growth.\n\n";
  auto [input, metadata] = read_parquet(input_filepath);

  // Status string to indicate if page stats are set to be written or not
  auto page_stat_string = (page_stats.has_value()) ? "page stats" : "no page stats";
  // Write parquet file with the specified encoding and compression
  std::cout << "Writing " << output_filepath << " with encoding, compression and "
            << page_stat_string << "..\n";

  // `timer` is automatically started here
  timer timer;
  write_parquet(input->view(), metadata, output_filepath, encoding, compression, page_stats);
  timer.print_elapsed_millis();

  // Read the parquet file written with encoding and compression
  std::cout << "Reading " << output_filepath << "...\n";

  // Reset the timer
  timer.reset();
  auto [transcoded_input, transcoded_metadata] = read_parquet(output_filepath);
  timer.print_elapsed_millis();

  // Check for validity
  check_tables_equal(input->view(), transcoded_input->view());

  return 0;
}
