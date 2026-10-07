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

#include <rmm/cuda_stream_pool.hpp>
#include <rmm/mr/statistics_resource_adaptor.hpp>

#include <chrono>
#include <exception>
#include <filesystem>
#include <iostream>
#include <memory>
#include <string>
#include <thread>

using elapsed_t  = std::chrono::duration<double>;
using byte_range = std::pair<std::size_t, std::size_t>;
using result_t   = std::unique_ptr<cudf::table>;

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

struct chunk_fn {
  std::string input_file;
  std::vector<result_t>& agg_data;
  cuda::stream_ref stream;

  std::vector<byte_range> byte_ranges{};
  bool first_range{};

  void add_range(std::size_t start, std::size_t size)
  {
    byte_ranges.push_back(byte_range{start, size});
    if (!first_range) { first_range = (start == 0); }
  }

  void operator()()
  {
    using namespace std::chrono_literals;

    // process each byte range assigned to this thread
    for (auto& br : byte_ranges) {
      auto const input_table = load_chunk(input_file, br.first, br.second, stream);
      auto const read_rows   = input_table->num_rows();
      if (read_rows == 0) continue;

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
    }
    // done with this stream
    stream.sync();
  }
};

int main(int argc, char const** argv)
{
  if (argc < 2) {
    std::cout << "required parameter: input-file-path\n";
    std::cout << "optional parameters: chunk-count thread-count\n";
    return 1;
  }

  auto const input_file   = std::string{argv[1]};
  auto const divider      = (argc < 3) ? 25 : std::stoi(std::string(argv[2]));
  auto const thread_count = (argc < 4) ? 2 : std::stoi(std::string(argv[3]));

  std::cout << "Input: " << input_file << std::endl;
  std::cout << "Chunks: " << divider << std::endl;
  std::cout << "Threads: " << thread_count << std::endl;

  auto const mr_name = std::string("pool");
  auto resource      = create_memory_resource(mr_name);
  auto stats_mr      = rmm::mr::statistics_resource_adaptor{resource};
  rmm::mr::set_current_device_resource(stats_mr);
  auto stream = cudf::get_default_stream();

  std::filesystem::path p = input_file;
  auto const file_size    = std::filesystem::file_size(p);

  auto start = std::chrono::steady_clock::now();

  std::size_t chunk_size = file_size / divider + ((file_size % divider) != 0);
  std::size_t start_pos  = 0;

  auto stream_pool = rmm::cuda_stream_pool(thread_count);
  std::vector<std::vector<result_t>> chunk_results(thread_count);

  std::vector<chunk_fn> chunk_tasks;
  for (auto& cr : chunk_results) {
    chunk_tasks.emplace_back(chunk_fn{input_file, cr, stream_pool.get_stream()});
  }
  for (std::size_t i = 0; i < divider; ++i) {
    auto start = i * chunk_size;
    auto size  = std::min(chunk_size, file_size - start);
    chunk_tasks[i % thread_count].add_range(start, size);
  }
  std::vector<std::exception_ptr> exceptions(chunk_tasks.size());
  std::vector<std::thread> threads;
  for (std::size_t i = 0; i < chunk_tasks.size(); ++i) {
    threads.emplace_back([task = chunk_tasks[i], &exception = exceptions[i]]() mutable {
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

  // in case some kernels are still running on the default stream
  stream.sync();

  // combine each thread's agg data into a single vector
  std::vector<result_t> agg_data(divider);
  auto begin = agg_data.begin();
  for (auto& c : chunk_results) {
    std::move(c.begin(), c.end(), begin);
    begin += c.size();
  }

  // now aggregate the aggregate results
  auto results = compute_final_aggregates(agg_data, stream);
  stream.sync();

  elapsed_t elapsed = std::chrono::steady_clock::now() - start;
  std::cout << "Number of keys: " << results->num_rows() << std::endl;
  std::cout << "Process time: " << elapsed.count() << " seconds\n";
  std::cout << "Peak memory: " << (stats_mr.get_bytes_counter().peak / 1048576.0) << " MB\n";

  return 0;
}
