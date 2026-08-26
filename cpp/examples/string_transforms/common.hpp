/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/io/csv.hpp>
#include <cudf/io/datasource.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>

#include <rmm/cuda_device.hpp>
#include <rmm/mr/cuda_async_memory_resource.hpp>
#include <rmm/mr/cuda_memory_resource.hpp>
#include <rmm/mr/pool_memory_resource.hpp>
#include <rmm/mr/statistics_resource_adaptor.hpp>

#include <chrono>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

/**
 * @brief Main example function returns transformed string table.
 *
 * @param table Table to be transformed
 * @return Transformed result
 */
std::tuple<std::unique_ptr<cudf::column>, std::vector<int32_t>> transform(
  cudf::table_view const& table);

void write_csv(cudf::table_view const& tbl_view,
               std::string const& file_path,
               std::vector<std::string> const& names)
{
  auto sink_info = cudf::io::sink_info(file_path);
  auto builder   = cudf::io::csv_writer_options::builder(sink_info, tbl_view);
  auto options   = builder.include_header(true).names(names).rows_per_chunk(10'000'000).build();
  cudf::io::write_csv(options);
}

/**
 * @brief Main for strings examples
 *
 * Command line parameters:
 * 1. CSV file name/path
 * 2. Out file name/path
 * 3. Number of rows from the CSV to transform
 * 4. Memory resource (optional): 'pool' or 'cuda'
 *
 * The stdout includes the number of rows in the input and the output size in bytes.
 */
int main(int argc, char const** argv)
{
  if (argc < 3) {
    std::cerr
      << "insufficient arguments.\n\t\tin-csv-path out-csv-path [num_rows [memory_resource]]\n";
    return EXIT_FAILURE;
  }

  auto const in_csv   = std::string{argv[1]};
  auto const out_csv  = std::string{argv[2]};
  auto const num_rows = argc > 3 ? std::optional{std::stoi(std::string(argv[3]))} : std::nullopt;
  auto const memory_resource_name =
    std::string{argc > 4 ? std::string(argv[4]) : std::string("cuda")};
  auto const enable_stats = memory_resource_name.ends_with("-stats");

  auto stream = cudf::get_default_stream();

  rmm::mr::cuda_memory_resource cuda_mr{};
  rmm::mr::pool_memory_resource pool_mr{cuda_mr, rmm::percent_of_free_device_memory(50)};
  rmm::mr::cuda_async_memory_resource async_mr{};

  auto const base_name = enable_stats
                           ? memory_resource_name.substr(0, memory_resource_name.size() - 6)
                           : memory_resource_name;
  if (base_name == "pool") {
    cudf::set_current_device_resource(pool_mr);
  } else if (base_name == "async") {
    cudf::set_current_device_resource(async_mr);
  } else if (base_name == "cuda") {
    cudf::set_current_device_resource(cuda_mr);
  } else {
    CUDF_FAIL("Unrecognized memory resource name: " + memory_resource_name, std::invalid_argument);
  }

  rmm::mr::statistics_resource_adaptor stats_adaptor{cudf::get_current_device_resource_ref()};

  if (enable_stats) { cudf::set_current_device_resource(stats_adaptor); }

  cudf::io::csv_reader_options in_opts =
    cudf::io::csv_reader_options::builder(cudf::io::source_info{in_csv}).header(0);

  auto in_csv_table = cudf::io::read_csv(in_opts);
  auto& input       = in_csv_table.tbl;

  if (num_rows.has_value() && input->get_column(0).size() != num_rows.value()) {
    input = cudf::sample(*input, num_rows.value(), cudf::sample_with_replacement::TRUE);
  }

  auto table_view = input->view();

  std::chrono::duration<double> elapsed_cold{};
  {
    // warmup pass
    stream.synchronize();
    auto start_cold = std::chrono::steady_clock::now();
    nvtxRangePush("transform cold");
    auto [result_cold, input_indices_cold] = transform(table_view);
    stream.synchronize();
    nvtxRangePop();
    elapsed_cold = std::chrono::steady_clock::now() - start_cold;
  }

  stream.synchronize();

  auto start = std::chrono::steady_clock::now();
  nvtxRangePush("transform warm");
  auto [result, input_indices] = transform(table_view);

  // ensure transform operation completes and the wall-time is only for the transform computation
  stream.synchronize();
  nvtxRangePop();

  std::chrono::duration<double> elapsed = std::chrono::steady_clock::now() - start;

  std::vector<cudf::column_view> out_columns(table_view.begin(), table_view.end());

  out_columns.emplace_back(result->view());

  cudf::table_view table{out_columns};

  std::vector<std::string> output_column_names;
  std::transform(in_csv_table.metadata.schema_info.begin(),
                 in_csv_table.metadata.schema_info.end(),
                 std::back_inserter(output_column_names),
                 [](auto const& col) { return col.name; });
  output_column_names.emplace_back("Transformed");

  write_csv(table, out_csv, output_column_names);

  auto const transformed_size =
    std::transform_reduce(input_indices.begin(),
                          input_indices.end(),
                          static_cast<size_t>(0),
                          std::plus<>{},
                          [&](auto index) { return input->get_column(index).alloc_size(); });

  std::cout << "Memory Resource: " << memory_resource_name << "\n"
            << "Warmup Time: " << elapsed_cold.count() << " seconds\n"
            << "Wall Time: " << elapsed.count() << " seconds\n"
            << "Input Table: " << table_view.num_rows() << " rows x " << table_view.num_columns()
            << " columns, " << input->alloc_size() << " bytes\n"
            << "Transformed: " << table_view.num_rows() << " rows x " << input_indices.size()
            << " columns, " << transformed_size << " bytes\n\n";

  if (enable_stats) {
    auto bytes  = stats_adaptor.get_bytes_counter();
    auto allocs = stats_adaptor.get_allocations_counter();

    std::cout << "Peak Memory Allocated: " << bytes.peak << " bytes\n"
              << "Total Memory Allocated: " << bytes.total << " bytes\n";

    std::cout << "Peak Allocations: " << allocs.peak << " allocations\n"
              << "Total Allocations: " << allocs.total << " allocations\n\n";
  }

  std::cout.flush();

  return EXIT_SUCCESS;
}
