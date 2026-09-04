/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "common_utils.hpp"

#include <cudf/column/column.hpp>
#include <cudf/io/csv.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <rmm/device_buffer.hpp>

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

std::unique_ptr<cudf::column> make_column_from_vector(std::vector<int32_t> const& column_data)
{
  cudf::data_type dtype{cudf::type_id::INT32};
  rmm::device_buffer data{
    column_data.data(), column_data.size() * sizeof(int32_t), cudf::get_default_stream()};
  // Ensure the async H2D copy from `column_data` completes before the caller can reuse or
  // destroy the source vector.
  cudf::get_default_stream().synchronize();
  cudf::size_type size       = column_data.size();
  cudf::size_type null_count = 0;
  rmm::device_buffer null_mask{};

  return std::make_unique<cudf::column>(
    dtype, size, std::move(data), std::move(null_mask), null_count);
}

cudf::table make_table(size_t row_count, size_t column_count)
{
  std::vector<int32_t> column_data(row_count);
  std::vector<std::unique_ptr<cudf::column>> columns;

  int32_t current_value{0};
  for (size_t i = 1; i <= column_count; i++) {
    std::iota(column_data.begin(), column_data.end(), current_value);
    columns.emplace_back(make_column_from_vector(column_data));
    current_value += row_count;
  }
  return {std::move(columns)};
}

std::string table_view_to_string(cudf::table_view const& tbl_view)
{
  std::vector<char> output;
  auto sink_info = cudf::io::sink_info(&output);
  auto builder   = cudf::io::csv_writer_options::builder(sink_info, tbl_view);
  auto options   = builder.build();
  cudf::io::write_csv(options);
  return {output.begin(), output.end()};
}

void print_table(std::string const& header, cudf::table_view const& tbl_view)
{
  std::cout << header << ":\n" << table_view_to_string(tbl_view) << "\n";
}
