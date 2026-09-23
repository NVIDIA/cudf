/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/io/csv.hpp>
#include <cudf/io/datasource.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>

#include <rmm/cuda_device.hpp>
#include <rmm/mr/cuda_memory_resource.hpp>
#include <rmm/mr/pool_memory_resource.hpp>

#include <chrono>
#include <iostream>
#include <string>

/**
 * @brief Main example function returns redacted strings column.
 *
 * This function returns a redacted version of the input `names` column
 * using the `visibilities` column as in the following example
 * ```
 * names        visibility  --> redacted
 * John Doe     public          D John
 * Bobby Joe    private         X X
 * ```
 *
 * @param names First and last names separated with a single space
 * @param visibilities String values `public` or `private` only
 * @return Redacted strings column
 */
std::unique_ptr<cudf::column> redact_strings(cudf::column_view const& names,
                                             cudf::column_view const& visibilities);

/**
 * @brief Main for strings examples
 *
 * Command line parameters:
 * 1. CSV file name/path
 * 2. Memory resource (optional): 'pool' or 'cuda'
 *
 * The stdout includes the number of rows in the input and the output size in bytes.
 */
int main(int argc, char const** argv)
{
  if (argc < 2) {
    std::cout << "required parameter: csv-file-path\n";
    return 1;
  }

  auto const mr_name = std::string{argc > 2 ? std::string(argv[2]) : std::string("cuda")};
  rmm::mr::cuda_memory_resource cuda_mr{};
  rmm::mr::pool_memory_resource pool_mr{cuda_mr, rmm::percent_of_free_device_memory(50)};
  if (mr_name == "pool") {
    cudf::set_current_device_resource(pool_mr);
  } else {
    cudf::set_current_device_resource(cuda_mr);
  }

  auto const csv_file   = std::string{argv[1]};
  auto const csv_result = [csv_file] {
    cudf::io::csv_reader_options in_opts =
      cudf::io::csv_reader_options::builder(cudf::io::source_info{csv_file}).header(-1);
    return cudf::io::read_csv(in_opts).tbl;
  }();
  auto const csv_table = csv_result->view();

  std::cout << "table: " << csv_table.num_rows() << " rows " << csv_table.num_columns()
            << " columns\n";

  auto st     = std::chrono::steady_clock::now();
  auto result = redact_strings(csv_table.column(0), csv_table.column(1));

  std::chrono::duration<double> elapsed = std::chrono::steady_clock::now() - st;
  std::cout << "Wall time: " << elapsed.count() << " seconds\n";
  auto const scv = cudf::strings_column_view(result->view());
  std::cout << "Output size " << scv.chars_size(cudf::get_default_stream()) << " bytes\n";

  return 0;
}
