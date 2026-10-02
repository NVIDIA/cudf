/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/copying.hpp>
#include <cudf/groupby.hpp>
#include <cudf/io/json.hpp>
#include <cudf/io/types.hpp>
#include <cudf/join/hash_join.hpp>
#include <cudf/join/join.hpp>
#include <cudf/sorting.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>

#include <rmm/cuda_device.hpp>
#include <rmm/mr/cuda_memory_resource.hpp>
#include <rmm/mr/pool_memory_resource.hpp>

#include <cuda/memory_resource>

#include <iostream>
#include <string>

/**
 * @file deduplication.cpp
 * @brief Demonstrates usage of the libcudf APIs to perform operations on nested-type tables.
 *
 * The algorithms chosen to be demonstrated are to showcase nested-type row operators of three
 * kinds:
 * 1. hashing: Used by functions `count_aggregate` and `join_count` to hash inputs of any type
 * 2. equality: Used by functions `count_aggregate` and `join_count` in conjunction with hashing
 * to determine equality for nested types
 * 3. lexicographic: Used by function `sort_keys` to create a lexicographical order for nested-types
 * so as to enable sorting
 *
 */

/**
 * @brief Create memory resource for libcudf functions
 *
 * @param pool Whether to use a pool memory resource.
 * @return Memory resource instance
 */
cuda::mr::any_device_resource create_memory_resource(bool pool)
{
  rmm::mr::cuda_memory_resource cuda_mr{};
  if (pool) {
    return rmm::mr::pool_memory_resource{cuda_mr, rmm::percent_of_free_device_memory(50)};
  }
  return cuda_mr;
}

/**
 * @brief Read JSON input from file
 *
 * @param filepath path to input JSON file
 * @return cudf::io::table_with_metadata
 */
cudf::io::table_with_metadata read_json(std::string filepath)
{
  auto source_info = cudf::io::source_info(filepath);
  auto builder     = cudf::io::json_reader_options::builder(source_info).lines(true);
  auto options     = builder.build();
  return cudf::io::read_json(options);
}

/**
 * @brief Write JSON output to file
 *
 * @param input table to write
 * @param metadata metadata of input table read by JSON reader
 * @param filepath path to output JSON file
 */
void write_json(cudf::table_view input, cudf::io::table_metadata metadata, std::string filepath)
{
  // write the data for inspection
  auto sink_info = cudf::io::sink_info(filepath);
  auto builder   = cudf::io::json_writer_options::builder(sink_info, input).lines(true);
  builder.metadata(metadata);
  auto options = builder.build();
  cudf::io::write_json(options);
}

/**
 * @brief Aggregate count of duplicate rows in nested-type column
 *
 * @param input table to aggregate
 * @return std::unique_ptr<cudf::table>
 */
std::unique_ptr<cudf::table> count_aggregate(cudf::table_view input)
{
  // Get count for each key
  auto keys = cudf::table_view{{input.column(0)}};
  auto val  = cudf::make_numeric_column(cudf::data_type{cudf::type_id::INT32}, keys.num_rows());

  cudf::groupby::groupby grpby_obj(keys);
  std::vector<cudf::groupby::aggregation_request> requests;
  requests.emplace_back(cudf::groupby::aggregation_request());
  auto agg = cudf::make_count_aggregation<cudf::groupby_aggregation>();
  requests[0].aggregations.push_back(std::move(agg));
  requests[0].values = *val;
  auto agg_results   = grpby_obj.aggregate(requests);
  auto result_key    = std::move(agg_results.first);
  auto result_val    = std::move(agg_results.second[0].results[0]);

  auto left_cols = result_key->release();
  left_cols.push_back(std::move(result_val));

  return std::make_unique<cudf::table>(std::move(left_cols));
}

/**
 * @brief Join each row with its duplicate counts
 *
 * @param left left table
 * @param right right table
 * @return std::unique_ptr<cudf::table>
 */
std::unique_ptr<cudf::table> join_count(cudf::table_view left, cudf::table_view right)
{
  auto [left_indices, right_indices] =
    cudf::inner_join(cudf::table_view{{left.column(0)}}, cudf::table_view{{right.column(0)}});
  auto new_left  = cudf::gather(left, cudf::device_span<cudf::size_type const>{*left_indices});
  auto new_right = cudf::gather(right, cudf::device_span<cudf::size_type const>{*right_indices});

  auto left_cols  = new_left->release();
  auto right_cols = new_right->release();
  left_cols.push_back(std::move(right_cols[1]));

  return std::make_unique<cudf::table>(std::move(left_cols));
}

/**
 * @brief Sort nested-type column
 *
 * @param input table to sort
 * @return std::unique_ptr<cudf::table>
 *
 * @note if stability is desired, use `cudf::stable_sorted_order`
 */
std::unique_ptr<cudf::table> sort_keys(cudf::table_view input)
{
  auto sort_order = cudf::sorted_order(cudf::table_view{{input.column(0)}});
  return cudf::gather(input, *sort_order);
}

/**
 * @brief Main for nested_types examples
 *
 * Command line parameters:
 * 1. JSON input file name/path (default: "example.json")
 * 2. JSON output file name/path (default: "output.json")
 * 3. Memory resource (optional): "pool" or "cuda" (default: "pool")
 *
 * Example invocation from directory `cudf/cpp/examples/nested_types`:
 * ./build/deduplication example.json output.json pool
 *
 */
int main(int argc, char const** argv)
{
  std::string input_filepath;
  std::string output_filepath;
  std::string mr_name;
  if (argc != 4 && argc != 1) {
    std::cout << "Either provide all command-line arguments, or none to use defaults" << std::endl;
    return 1;
  }
  if (argc == 1) {
    input_filepath  = "example.json";
    output_filepath = "output.json";
    mr_name         = "pool";
  } else {
    input_filepath  = argv[1];
    output_filepath = argv[2];
    mr_name         = argv[3];
  }

  auto pool     = mr_name == "pool";
  auto resource = create_memory_resource(pool);
  cudf::set_current_device_resource(resource);

  std::cout << "Reading " << input_filepath << "..." << std::endl;
  // read input file
  auto [input, metadata] = read_json(input_filepath);

  auto count = count_aggregate(input->view());

  auto combined = join_count(input->view(), count->view());

  auto sorted = sort_keys(combined->view());

  metadata.schema_info.emplace_back("count");

  std::cout << "Writing " << output_filepath << "..." << std::endl;
  write_json(sorted->view(), metadata, output_filepath);

  return 0;
}
