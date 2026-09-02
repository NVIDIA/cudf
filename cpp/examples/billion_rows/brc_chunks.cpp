/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "common.hpp"
#include "groupby_results.hpp"

#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/io/csv.hpp>
#include <cudf/sorting.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>

#include <rmm/mr/statistics_resource_adaptor.hpp>

#include <chrono>
#include <filesystem>
#include <iostream>
#include <memory>
#include <string>

using elapsed_t = std::chrono::duration<double>;

std::unique_ptr<cudf::table> load_chunk(std::string const& input_file,
                                        std::size_t start,
                                        std::size_t size,
                                        cuda::stream_ref stream)
{
  cudf::io::csv_reader_options in_opts =
    cudf::io::csv_reader_options::builder(cudf::io::source_info{input_file})
      .header(-1)
      .delimiter(';')
      .doublequote(false)
      .byte_range_offset(start)
      .byte_range_size(size)
      .dtypes(std::vector<cudf::data_type>{cudf::data_type{cudf::type_id::STRING},
                                           cudf::data_type{cudf::type_id::FLOAT32}})
      .na_filter(false);
  return cudf::io::read_csv(in_opts, stream).tbl;
}

int main(int argc, char const** argv)
{
  if (argc < 2) {
    std::cout << "required parameter: input-file-path\n";
    std::cout << "optional parameter: chunk-count\n";
    return 1;
  }

  auto const input_file = std::string{argv[1]};
  auto const divider    = (argc < 3) ? 25 : std::stoi(std::string(argv[2]));

  std::cout << "Input: " << input_file << std::endl;
  std::cout << "Chunks: " << divider << std::endl;

  auto const mr_name = std::string("pool");
  auto resource      = create_memory_resource(mr_name);
  auto stats_mr      = rmm::mr::statistics_resource_adaptor{resource};
  rmm::mr::set_current_device_resource(stats_mr);
  auto stream = cudf::get_default_stream();

  std::filesystem::path p = input_file;
  auto const file_size    = std::filesystem::file_size(p);

  auto start = std::chrono::steady_clock::now();

  std::vector<std::unique_ptr<cudf::table>> agg_data;
  std::size_t chunk_size     = file_size / divider + ((file_size % divider) != 0);
  std::size_t start_pos      = 0;
  cudf::size_type total_rows = 0;
  do {
    auto const input_table = load_chunk(input_file, start_pos, chunk_size, stream);
    auto const read_rows   = input_table->num_rows();
    if (read_rows == 0) break;

    auto const cities = input_table->view().column(0);
    auto const temps  = input_table->view().column(1);

    std::vector<std::unique_ptr<cudf::groupby_aggregation>> aggregations;
    aggregations.emplace_back(cudf::make_min_aggregation<cudf::groupby_aggregation>());
    aggregations.emplace_back(cudf::make_max_aggregation<cudf::groupby_aggregation>());
    aggregations.emplace_back(cudf::make_sum_aggregation<cudf::groupby_aggregation>());
    aggregations.emplace_back(cudf::make_count_aggregation<cudf::groupby_aggregation>());
    auto result = compute_results(cities, temps, std::move(aggregations), stream);

    agg_data.emplace_back(
      cudf::sort_by_key(result->view(), result->view().select({0}), {}, {}, stream));
    start_pos += chunk_size;
    chunk_size = std::min(chunk_size, file_size - start_pos);
    total_rows += read_rows;
  } while (start_pos < file_size && chunk_size > 0);

  // now aggregate the aggregate results
  auto results = compute_final_aggregates(agg_data, stream);
  stream.sync();

  elapsed_t elapsed = std::chrono::steady_clock::now() - start;
  std::cout << "Number of keys: " << results->num_rows() << std::endl;
  std::cout << "Process time: " << elapsed.count() << " seconds\n";
  std::cout << "Peak memory: " << (stats_mr.get_bytes_counter().peak / 1048576.0) << " MB\n";

  return 0;
}
