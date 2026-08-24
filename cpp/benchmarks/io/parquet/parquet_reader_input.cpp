/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION.
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
void BM_parquet_read_data_common(nvbench::state& state,
                                 data_profile const& profile,
                                 nvbench::type_list<nvbench::enum_type<DataType>>)
{
  auto const d_type        = get_type_or_group(static_cast<int32_t>(DataType));
  auto const source_type   = retrieve_io_type_enum(state.get_string("io_type"));
  auto const data_size     = static_cast<size_t>(state.get_int64("data_size"));
  auto const compression   = retrieve_compression_type_enum(state.get_string("compression_type"));
  auto const rg_size_bytes = state.get_int64("row_group_size_bytes");
  auto const rg_size_rows  = state.get_int64("row_group_size_rows");
  cuio_source_sink_pair source_sink(source_type);

  auto const num_rows_written = [&]() {
    auto const tbl =
      create_random_table(cycle_dtypes(d_type, num_cols), table_size_bytes{data_size}, profile);
    auto const view = tbl->view();

    cudf::io::parquet_writer_options write_opts =
      cudf::io::parquet_writer_options::builder(source_sink.make_sink_info(), view)
        .compression(compression);
    // Sentinel 0 == use cuDF default (parquet bytes default is size_t::max).
    if (rg_size_bytes > 0) write_opts.set_row_group_size_bytes(rg_size_bytes);
    if (rg_size_rows > 0) write_opts.set_row_group_size_rows(rg_size_rows);
    cudf::io::write_parquet(write_opts);
    return view.num_rows();
  }();

  parquet_read_common(num_rows_written, num_cols, source_sink, state);
}

template <data_type DataType>
void BM_parquet_read_data(nvbench::state& state,
                          nvbench::type_list<nvbench::enum_type<DataType>> type_list)
{
  auto const cardinality = static_cast<cudf::size_type>(state.get_int64("cardinality"));
  auto const run_length  = static_cast<cudf::size_type>(state.get_int64("run_length"));
  BM_parquet_read_data_common<DataType>(
    state, data_profile_builder().cardinality(cardinality).avg_run_length(run_length), type_list);
}

template <data_type DataType>
void BM_parquet_read_fixed_width_struct(nvbench::state& state,
                                        nvbench::type_list<nvbench::enum_type<DataType>> type_list)
{
  auto const cardinality = static_cast<cudf::size_type>(state.get_int64("cardinality"));
  auto const run_length  = static_cast<cudf::size_type>(state.get_int64("run_length"));
  std::vector<cudf::type_id> s_types{
    cudf::type_id::INT32, cudf::type_id::FLOAT32, cudf::type_id::INT64};
  BM_parquet_read_data_common<DataType>(state,
                                        data_profile_builder()
                                          .cardinality(cardinality)
                                          .avg_run_length(run_length)
                                          .struct_types(s_types),
                                        type_list);
}

void BM_parquet_read_io_small_mixed(nvbench::state& state)
{
  auto const d_type =
    std::pair<cudf::type_id, cudf::type_id>{cudf::type_id::STRING, cudf::type_id::INT32};

  auto const cardinality   = static_cast<cudf::size_type>(state.get_int64("cardinality"));
  auto const run_length    = static_cast<cudf::size_type>(state.get_int64("run_length"));
  auto const num_strings   = static_cast<cudf::size_type>(state.get_int64("num_string_cols"));
  auto const source_type   = retrieve_io_type_enum(state.get_string("io_type"));
  auto const rg_size_bytes = state.get_int64("row_group_size_bytes");
  auto const rg_size_rows  = state.get_int64("row_group_size_rows");
  cuio_source_sink_pair source_sink(source_type);

  // want 80 pages total, across 4 columns, so 20 pages per column
  cudf::size_type constexpr n_col          = 4;
  cudf::size_type constexpr page_size_rows = 10'000;
  cudf::size_type constexpr num_rows       = page_size_rows * (80 / n_col);

  {
    auto const tbl = create_random_table(
      mix_dtypes(d_type, n_col, num_strings),
      row_count{num_rows},
      data_profile_builder().cardinality(cardinality).avg_run_length(run_length));
    auto const view = tbl->view();

    cudf::io::parquet_writer_options write_opts =
      cudf::io::parquet_writer_options::builder(source_sink.make_sink_info(), view)
        .max_page_size_rows(10'000)
        .compression(cudf::io::compression_type::NONE);
    if (rg_size_bytes > 0) write_opts.set_row_group_size_bytes(rg_size_bytes);
    if (rg_size_rows > 0) write_opts.set_row_group_size_rows(rg_size_rows);
    cudf::io::write_parquet(write_opts);
  }

  parquet_read_common(num_rows, n_col, source_sink, state);
}

using d_type_list = nvbench::enum_type_list<data_type::INTEGRAL,
                                            data_type::FLOAT,
                                            data_type::BOOL8,
                                            data_type::DECIMAL,
                                            data_type::TIMESTAMP,
                                            data_type::DURATION,
                                            data_type::STRING,
                                            data_type::LIST,
                                            data_type::STRUCT>;

NVBENCH_BENCH_TYPES(BM_parquet_read_data, NVBENCH_TYPE_AXES(d_type_list))
  .set_name("parquet_read_decode")
  .set_type_axes_names({"data_type"})
  .add_string_axis("io_type", {"DEVICE_BUFFER"})
  .add_string_axis("compression_type", {"NONE"})
  .set_min_samples(4)
  .add_int64_axis("cardinality", {0, 1000})
  .add_int64_axis("run_length", {1, 32})
  .add_int64_axis("data_size", {512 << 20})
  .add_int64_axis("row_group_size_bytes", {0})
  .add_int64_axis("row_group_size_rows", {0});

NVBENCH_BENCH(BM_parquet_read_io_small_mixed)
  .set_name("parquet_read_io_small_mixed")
  .add_string_axis("io_type", {"FILEPATH"})
  .set_min_samples(4)
  .add_int64_axis("cardinality", {0, 1000})
  .add_int64_axis("run_length", {1, 32})
  .add_int64_axis("num_string_cols", {1, 2, 3})
  .add_int64_axis("data_size", {512 << 20})
  .add_int64_axis("row_group_size_bytes", {0})
  .add_int64_axis("row_group_size_rows", {0});

// a benchmark for structs that only contain fixed-width types
using d_type_list_struct_only = nvbench::enum_type_list<data_type::STRUCT>;
NVBENCH_BENCH_TYPES(BM_parquet_read_fixed_width_struct, NVBENCH_TYPE_AXES(d_type_list_struct_only))
  .set_name("parquet_read_fixed_width_struct")
  .set_type_axes_names({"data_type"})
  .add_string_axis("io_type", {"DEVICE_BUFFER"})
  .add_string_axis("compression_type", {"NONE"})
  .set_min_samples(4)
  .add_int64_axis("cardinality", {0, 1000})
  .add_int64_axis("run_length", {1, 32})
  .add_int64_axis("data_size", {512 << 20})
  .add_int64_axis("row_group_size_bytes", {0})
  .add_int64_axis("row_group_size_rows", {0});
