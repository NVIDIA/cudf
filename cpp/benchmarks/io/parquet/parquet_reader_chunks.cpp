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
void BM_parquet_read_chunks(nvbench::state& state, nvbench::type_list<nvbench::enum_type<DataType>>)
{
  auto const d_type           = get_type_or_group(static_cast<int32_t>(DataType));
  auto const cardinality      = static_cast<cudf::size_type>(state.get_int64("cardinality"));
  auto const run_length       = static_cast<cudf::size_type>(state.get_int64("run_length"));
  auto const chunk_read_limit = static_cast<cudf::size_type>(state.get_int64("chunk_read_limit"));
  auto const source_type      = retrieve_io_type_enum(state.get_string("io_type"));
  auto const data_size        = static_cast<size_t>(state.get_int64("data_size"));
  auto const compression      = cudf::io::compression_type::SNAPPY;
  auto const rg_size_bytes    = state.get_int64("row_group_size_bytes");
  auto const rg_size_rows     = state.get_int64("row_group_size_rows");
  cuio_source_sink_pair source_sink(source_type);

  auto const num_rows_written = [&]() {
    auto const tbl = create_random_table(
      cycle_dtypes(d_type, num_cols),
      table_size_bytes{data_size},
      data_profile_builder().cardinality(cardinality).avg_run_length(run_length));
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

  cudf::io::parquet_reader_options read_opts =
    cudf::io::parquet_reader_options::builder(source_sink.make_source_info());

  auto mem_stats_logger = cudf::memory_stats_logger();
  state.set_cuda_stream(nvbench::make_cuda_stream_view(cudf::get_default_stream().value()));
  state.exec(
    nvbench::exec_tag::sync | nvbench::exec_tag::timer, [&](nvbench::launch& launch, auto& timer) {
      drop_page_cache_if_enabled(read_opts.get_source().filepaths());

      timer.start();
      auto reader                   = cudf::io::chunked_parquet_reader(chunk_read_limit, read_opts);
      cudf::size_type num_rows_read = 0;
      do {
        auto const result = reader.read_chunk();
        num_rows_read += result.tbl->num_rows();
        CUDF_EXPECTS(result.tbl->num_columns() == num_cols, "Unexpected number of columns");
      } while (reader.has_next());
      timer.stop();

      CUDF_EXPECTS(num_rows_read == num_rows_written, "Benchmark did not read the entire table");
    });

  auto const time = state.get_summary("nv/cold/time/gpu/mean").get_float64("value");
  state.add_element_count(static_cast<double>(data_size) / time, "bytes_per_second");
  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");
  state.add_buffer_size(source_sink.size(), "encoded_file_size", "encoded_file_size");
}

template <data_type DataType>
void BM_parquet_read_subrowgroup_chunks(nvbench::state& state,
                                        nvbench::type_list<nvbench::enum_type<DataType>>)
{
  auto const d_type           = get_type_or_group(static_cast<int32_t>(DataType));
  auto const cardinality      = static_cast<cudf::size_type>(state.get_int64("cardinality"));
  auto const run_length       = static_cast<cudf::size_type>(state.get_int64("run_length"));
  auto const chunk_read_limit = static_cast<cudf::size_type>(state.get_int64("chunk_read_limit"));
  auto const pass_read_limit  = static_cast<cudf::size_type>(state.get_int64("pass_read_limit"));
  auto const source_type      = retrieve_io_type_enum(state.get_string("io_type"));
  auto const data_size        = static_cast<size_t>(state.get_int64("data_size"));
  auto const compression      = cudf::io::compression_type::SNAPPY;
  auto const rg_size_bytes    = state.get_int64("row_group_size_bytes");
  auto const rg_size_rows     = state.get_int64("row_group_size_rows");
  cuio_source_sink_pair source_sink(source_type);

  auto const num_rows_written = [&]() {
    auto const tbl = create_random_table(
      cycle_dtypes(d_type, num_cols),
      table_size_bytes{data_size},
      data_profile_builder().cardinality(cardinality).avg_run_length(run_length));
    auto const view = tbl->view();

    cudf::io::parquet_writer_options write_opts =
      cudf::io::parquet_writer_options::builder(source_sink.make_sink_info(), view)
        .compression(compression);
    if (rg_size_bytes > 0) write_opts.set_row_group_size_bytes(rg_size_bytes);
    if (rg_size_rows > 0) write_opts.set_row_group_size_rows(rg_size_rows);
    cudf::io::write_parquet(write_opts);
    return view.num_rows();
  }();

  cudf::io::parquet_reader_options read_opts =
    cudf::io::parquet_reader_options::builder(source_sink.make_source_info());

  auto mem_stats_logger = cudf::memory_stats_logger();
  state.set_cuda_stream(nvbench::make_cuda_stream_view(cudf::get_default_stream().value()));
  state.exec(
    nvbench::exec_tag::sync | nvbench::exec_tag::timer, [&](nvbench::launch& launch, auto& timer) {
      drop_page_cache_if_enabled(read_opts.get_source().filepaths());

      timer.start();
      auto reader = cudf::io::chunked_parquet_reader(chunk_read_limit, pass_read_limit, read_opts);
      cudf::size_type num_rows_read = 0;
      do {
        auto const result = reader.read_chunk();
        num_rows_read += result.tbl->num_rows();
        CUDF_EXPECTS(result.tbl->num_columns() == num_cols, "Unexpected number of columns");
      } while (reader.has_next());
      timer.stop();

      CUDF_EXPECTS(num_rows_read == num_rows_written, "Benchmark did not read the entire table");
    });

  auto const time = state.get_summary("nv/cold/time/gpu/mean").get_float64("value");
  state.add_element_count(static_cast<double>(data_size) / time, "bytes_per_second");
  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");
  state.add_buffer_size(source_sink.size(), "encoded_file_size", "encoded_file_size");
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

NVBENCH_BENCH_TYPES(BM_parquet_read_chunks, NVBENCH_TYPE_AXES(d_type_list))
  .set_name("parquet_read_chunks")
  .add_string_axis("io_type", {"DEVICE_BUFFER"})
  .set_min_samples(4)
  .add_int64_axis("cardinality", {0, 1000})
  .add_int64_axis("run_length", {1, 32})
  .add_int64_axis("chunk_read_limit", {0, 500'000})
  .add_int64_axis("data_size", {512 << 20})
  .add_int64_axis("row_group_size_bytes", {0})
  .add_int64_axis("row_group_size_rows", {0});

NVBENCH_BENCH_TYPES(BM_parquet_read_subrowgroup_chunks, NVBENCH_TYPE_AXES(d_type_list))
  .set_name("parquet_read_subrowgroup_chunks")
  .add_string_axis("io_type", {"DEVICE_BUFFER"})
  .set_min_samples(4)
  .add_int64_axis("cardinality", {0, 1000})
  .add_int64_axis("run_length", {1, 32})
  .add_int64_axis("chunk_read_limit", {0, 500'000})
  .add_int64_axis("pass_read_limit", {0, 500'000})
  .add_int64_axis("data_size", {512 << 20})
  .add_int64_axis("row_group_size_bytes", {0})
  .add_int64_axis("row_group_size_rows", {0});
