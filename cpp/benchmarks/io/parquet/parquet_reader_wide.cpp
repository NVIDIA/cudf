/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "reader_common.hpp"

#include <benchmarks/common/generate_input.hpp>
#include <benchmarks/common/memory_stats.hpp>
#include <benchmarks/io/cuio_common.hpp>
#include <benchmarks/io/nvbench_helpers.hpp>

#include <cudf/io/parquet.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <nvbench/nvbench.cuh>

template <data_type DataType>
void BM_parquet_read_wide_tables(nvbench::state& state,
                                 nvbench::type_list<nvbench::enum_type<DataType>> type_list)
{
  auto const d_type = get_type_or_group(static_cast<int32_t>(DataType));

  auto const n_col           = static_cast<cudf::size_type>(state.get_int64("num_cols"));
  auto const data_size_bytes = static_cast<size_t>(state.get_int64("data_size"));
  auto const cardinality     = static_cast<cudf::size_type>(state.get_int64("cardinality"));
  auto const run_length      = static_cast<cudf::size_type>(state.get_int64("run_length"));
  auto const source_type     = io_type::DEVICE_BUFFER;
  auto const rg_size_bytes   = state.get_int64("row_group_size_bytes");
  auto const rg_size_rows    = state.get_int64("row_group_size_rows");
  cuio_source_sink_pair source_sink(source_type);

  auto const num_rows_written = [&]() {
    auto const tbl = create_random_table(
      cycle_dtypes(d_type, n_col),
      table_size_bytes{data_size_bytes},
      data_profile_builder().cardinality(cardinality).avg_run_length(run_length));
    auto const view = tbl->view();

    cudf::io::parquet_writer_options write_opts =
      cudf::io::parquet_writer_options::builder(source_sink.make_sink_info(), view)
        .compression(cudf::io::compression_type::NONE);
    // Sentinel 0 == use cuDF default (parquet bytes default is size_t::max).
    if (rg_size_bytes > 0) write_opts.set_row_group_size_bytes(rg_size_bytes);
    if (rg_size_rows > 0) write_opts.set_row_group_size_rows(rg_size_rows);
    cudf::io::write_parquet(write_opts);
    return view.num_rows();
  }();

  parquet_read_common(num_rows_written, n_col, source_sink, state);
}

void BM_parquet_read_wide_tables_mixed(nvbench::state& state)
{
  auto const d_type = []() {
    auto d_type1 = get_type_or_group(static_cast<int32_t>(data_type::INTEGRAL));
    auto d_type2 = get_type_or_group(static_cast<int32_t>(data_type::FLOAT));
    d_type1.reserve(d_type1.size() + d_type2.size());
    std::move(d_type2.begin(), d_type2.end(), std::back_inserter(d_type1));
    return d_type1;
  }();

  auto const n_col           = static_cast<cudf::size_type>(state.get_int64("num_cols"));
  auto const data_size_bytes = static_cast<size_t>(state.get_int64("data_size"));
  auto const cardinality     = static_cast<cudf::size_type>(state.get_int64("cardinality"));
  auto const run_length      = static_cast<cudf::size_type>(state.get_int64("run_length"));
  auto const source_type     = io_type::DEVICE_BUFFER;
  auto const rg_size_bytes   = state.get_int64("row_group_size_bytes");
  auto const rg_size_rows    = state.get_int64("row_group_size_rows");
  cuio_source_sink_pair source_sink(source_type);

  auto const num_rows_written = [&]() {
    auto const tbl = create_random_table(
      cycle_dtypes(d_type, n_col),
      table_size_bytes{data_size_bytes},
      data_profile_builder().cardinality(cardinality).avg_run_length(run_length));
    auto const view = tbl->view();

    cudf::io::parquet_writer_options write_opts =
      cudf::io::parquet_writer_options::builder(source_sink.make_sink_info(), view)
        .compression(cudf::io::compression_type::NONE);
    if (rg_size_bytes > 0) write_opts.set_row_group_size_bytes(rg_size_bytes);
    if (rg_size_rows > 0) write_opts.set_row_group_size_rows(rg_size_rows);
    cudf::io::write_parquet(write_opts);
    return view.num_rows();
  }();

  parquet_read_common(num_rows_written, n_col, source_sink, state);
}

using d_type_list_wide_table = nvbench::enum_type_list<data_type::DECIMAL, data_type::STRING>;
NVBENCH_BENCH_TYPES(BM_parquet_read_wide_tables, NVBENCH_TYPE_AXES(d_type_list_wide_table))
  .set_name("parquet_read_wide_tables")
  .set_min_samples(4)
  .set_type_axes_names({"data_type"})
  .add_int64_axis("data_size", {1024L << 20, 2048L << 20})
  .add_int64_axis("num_cols", {256, 512, 1024})
  .add_int64_axis("cardinality", {0, 1000})
  .add_int64_axis("run_length", {1, 32})
  .add_int64_axis("row_group_size_bytes", {0})
  .add_int64_axis("row_group_size_rows", {0});

NVBENCH_BENCH(BM_parquet_read_wide_tables_mixed)
  .set_name("parquet_read_wide_tables_mixed")
  .set_min_samples(4)
  .add_int64_axis("data_size", {1024L << 20, 2048L << 20})
  .add_int64_axis("num_cols", {256, 512, 1024})
  .add_int64_axis("cardinality", {0, 1000})
  .add_int64_axis("run_length", {1, 32})
  .add_int64_axis("row_group_size_bytes", {0})
  .add_int64_axis("row_group_size_rows", {0});
