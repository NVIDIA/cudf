/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "parquet_io.hpp"

#include <benchmarks/common/ndsh_data_generator/ndsh_data_generator.hpp>
#include <benchmarks/common/nvtx_ranges.hpp>
#include <benchmarks/common/table_utilities.hpp>

#include <cudf/copying.hpp>
#include <cudf/detail/utilities/integer_utils.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <rmm/mr/managed_memory_resource.hpp>
#include <rmm/mr/pool_memory_resource.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <unordered_set>
#include <utility>

void write_parquet(table_with_names const& table, std::string const& filepath)
{
  CUDF_BENCHMARK_RANGE();
  auto const sink_info = cudf::io::sink_info(filepath);
  cudf::io::table_metadata metadata;
  metadata.schema_info = std::vector<cudf::io::column_name_info>(table.column_names().begin(),
                                                                 table.column_names().end());
  auto const table_input_metadata = cudf::io::table_input_metadata{metadata};
  auto builder = cudf::io::parquet_writer_options::builder(sink_info, table.table());
  builder.metadata(table_input_metadata);
  auto const options = builder.build();
  cudf::io::write_parquet(options);
}

std::unique_ptr<table_with_names> read_parquet(
  cudf::io::source_info const& source_info,
  std::vector<std::string> const& columns,
  std::unique_ptr<cudf::ast::operation> const& predicate)
{
  CUDF_BENCHMARK_RANGE();
  auto builder = cudf::io::parquet_reader_options_builder(source_info);
  if (!columns.empty()) { builder.column_names(columns); }
  if (predicate) { builder.filter(*predicate); }
  auto const options       = builder.build();
  auto table_with_metadata = cudf::io::read_parquet(options);
  std::vector<std::string> column_names;
  for (auto const& col_info : table_with_metadata.metadata.schema_info) {
    column_names.push_back(col_info.name);
  }
  return std::make_unique<table_with_names>(std::move(table_with_metadata.tbl), column_names);
}

void write_to_parquet_device_buffer(std::unique_ptr<cudf::table> const& table,
                                    std::vector<std::string> const& col_names,
                                    cuio_source_sink_pair& source)
{
  CUDF_BENCHMARK_RANGE();
  auto const stream = cudf::get_default_stream();

  // Prepare the table metadata
  cudf::io::table_metadata metadata;
  std::vector<cudf::io::column_name_info> col_name_infos;
  for (auto& col_name : col_names) {
    col_name_infos.push_back(cudf::io::column_name_info(col_name));
  }
  metadata.schema_info            = col_name_infos;
  auto const table_input_metadata = cudf::io::table_input_metadata{metadata};

  auto est_size                     = static_cast<std::size_t>(estimate_size(table->view()));
  constexpr auto PQ_MAX_TABLE_BYTES = 8ul << 30;  // 8GB
  // TODO: best to get this limit from percent_of_free_device_memory(50) of device memory resource.
  if (est_size > PQ_MAX_TABLE_BYTES) {
    auto builder = cudf::io::chunked_parquet_writer_options::builder(source.make_sink_info());
    builder.metadata(table_input_metadata);
    auto const options = builder.build();
    auto num_splits    = static_cast<cudf::size_type>(
      std::ceil(static_cast<long double>(est_size) / (PQ_MAX_TABLE_BYTES)));
    std::vector<cudf::size_type> splits(num_splits - 1);
    auto num_rows          = table->num_rows();
    auto num_row_per_chunk = cudf::util::div_rounding_up_safe(num_rows, num_splits);
    std::generate_n(splits.begin(), splits.size(), [num_row_per_chunk, i = 0]() mutable {
      return (i += num_row_per_chunk);
    });
    std::vector<cudf::table_view> split_tables = cudf::split(table->view(), splits, stream);
    auto writer                                = cudf::io::chunked_parquet_writer(options, stream);
    for (auto const& chunk_table : split_tables) {
      writer.write(chunk_table);
    }
    writer.close();
    return;
  }
  // Write parquet data to host buffer
  auto builder = cudf::io::parquet_writer_options::builder(source.make_sink_info(), table->view());
  builder.metadata(table_input_metadata);
  auto const options = builder.build();
  cudf::io::write_parquet(options, stream);
}

void generate_parquet_data_sources(double scale_factor,
                                   std::vector<std::string> const& table_names,
                                   std::unordered_map<std::string, cuio_source_sink_pair>& sources)
{
  CUDF_BENCHMARK_RANGE();

  // Use a managed pool for parquet generation.
  rmm::mr::pool_memory_resource managed_pool_mr{rmm::mr::managed_memory_resource{},
                                                rmm::percent_of_free_device_memory(50)};

  std::unordered_set<std::string> const requested_table_names = [&table_names]() {
    if (table_names.empty()) {
      return std::unordered_set<std::string>{
        "orders", "lineitem", "part", "partsupp", "supplier", "customer", "nation", "region"};
    }
    return std::unordered_set(table_names.begin(), table_names.end());
  }();
  std::for_each(
    requested_table_names.begin(), requested_table_names.end(), [&](auto const& table_name) {
      sources.emplace(table_name, cuio_source_sink_pair(io_type::HOST_BUFFER));
    });
  std::unordered_map<std::string, std::unique_ptr<cudf::table>> tables;

  auto const stream = cudf::get_default_stream();

  if (sources.count("orders") or sources.count("lineitem") or sources.count("part")) {
    auto [orders, lineitem, part] =
      cudf::datagen::generate_orders_lineitem_part(scale_factor, stream, managed_pool_mr);
    if (sources.count("orders")) {
      write_to_parquet_device_buffer(orders, ndsh_schema("orders"), sources.at("orders"));
      orders = {};
    }
    if (sources.count("part")) {
      write_to_parquet_device_buffer(part, ndsh_schema("part"), sources.at("part"));
      part = {};
    }
    if (sources.count("lineitem")) {
      write_to_parquet_device_buffer(lineitem, ndsh_schema("lineitem"), sources.at("lineitem"));
      lineitem = {};
    }
  }

  if (sources.count("partsupp")) {
    auto partsupp = cudf::datagen::generate_partsupp(scale_factor, stream, managed_pool_mr);
    write_to_parquet_device_buffer(partsupp, ndsh_schema("partsupp"), sources.at("partsupp"));
  }

  if (sources.count("supplier")) {
    auto supplier = cudf::datagen::generate_supplier(scale_factor, stream, managed_pool_mr);
    write_to_parquet_device_buffer(supplier, ndsh_schema("supplier"), sources.at("supplier"));
  }

  if (sources.count("customer")) {
    auto customer = cudf::datagen::generate_customer(scale_factor, stream, managed_pool_mr);
    write_to_parquet_device_buffer(customer, ndsh_schema("customer"), sources.at("customer"));
  }

  if (sources.count("nation")) {
    auto nation = cudf::datagen::generate_nation(stream, managed_pool_mr);
    write_to_parquet_device_buffer(nation, ndsh_schema("nation"), sources.at("nation"));
  }

  if (sources.count("region")) {
    auto region = cudf::datagen::generate_region(stream, managed_pool_mr);
    write_to_parquet_device_buffer(region, ndsh_schema("region"), sources.at("region"));
  }
}
