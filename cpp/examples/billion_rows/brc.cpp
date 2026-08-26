/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "common.hpp"
#include "groupby_results.hpp"

#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/io/csv.hpp>
#include <cudf/io/types.hpp>
#include <cudf/sorting.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>

#include <rmm/mr/statistics_resource_adaptor.hpp>

#include <chrono>
#include <iostream>
#include <memory>
#include <string>

using elapsed_t = std::chrono::duration<double>;

int main(int argc, char const** argv)
{
  if (argc < 2) {
    std::cout << "required parameter: input-file-path\n";
    return 1;
  }

  auto const input_file = std::string{argv[1]};
  std::cout << "Input: " << input_file << std::endl;

  auto const mr_name = std::string("pool");
  auto resource      = create_memory_resource(mr_name);
  auto stats_mr      = rmm::mr::statistics_resource_adaptor{resource};
  rmm::mr::set_current_device_resource(stats_mr);
  auto stream = cudf::get_default_stream();

  auto start = std::chrono::steady_clock::now();

  auto const csv_result = [input_file, stream] {
    cudf::io::csv_reader_options in_opts =
      cudf::io::csv_reader_options::builder(cudf::io::source_info{input_file})
        .header(-1)
        .delimiter(';')
        .doublequote(false)
        .dtypes(std::vector<cudf::data_type>{cudf::data_type{cudf::type_id::STRING},
                                             cudf::data_type{cudf::type_id::FLOAT32}})
        .na_filter(false);
    return cudf::io::read_csv(in_opts, stream).tbl;
  }();
  elapsed_t elapsed = std::chrono::steady_clock::now() - start;
  std::cout << "File load time: " << elapsed.count() << " seconds\n";
  auto const csv_table = csv_result->view();
  std::cout << "Input rows: " << csv_table.num_rows() << std::endl;

  auto const cities = csv_table.column(0);
  auto const temps  = csv_table.column(1);

  std::vector<std::unique_ptr<cudf::groupby_aggregation>> aggregations;
  aggregations.emplace_back(cudf::make_min_aggregation<cudf::groupby_aggregation>());
  aggregations.emplace_back(cudf::make_max_aggregation<cudf::groupby_aggregation>());
  aggregations.emplace_back(cudf::make_mean_aggregation<cudf::groupby_aggregation>());

  auto result = compute_results(cities, temps, std::move(aggregations), stream);

  // The other 2 examples employ sorting for the sub-aggregates so enabling
  // the following line may be more comparable in performance with them.
  //
  // result      = cudf::sort_by_key(result->view(), result->view().select({0}), {}, {}, stream);

  stream.synchronize();

  elapsed = std::chrono::steady_clock::now() - start;
  std::cout << "Number of keys: " << result->num_rows() << std::endl;
  std::cout << "Process time: " << elapsed.count() << " seconds\n";
  std::cout << "Peak memory: " << (stats_mr.get_bytes_counter().peak / 1048576.0) << " MB\n";

  return 0;
}
