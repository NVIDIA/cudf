/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <benchmarks/common/generate_input.hpp>
#include <benchmarks/common/memory_stats.hpp>
#include <benchmarks/io/cuio_common.hpp>
#include <benchmarks/io/nvbench_helpers.hpp>

#include <cudf/column/column.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/table/table.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <nvbench/nvbench.cuh>

#include <array>
#include <iomanip>
#include <sstream>

std::string format_throughput(double bytes_per_second, int precision = 2)
{
  std::array const units = {"B/s", "KiB/s", "MiB/s", "GiB/s", "TiB/s"};
  int i                  = 0;

  while (bytes_per_second >= 1024.0 && i < static_cast<int>(units.size()) - 1) {
    bytes_per_second /= 1024.0;
    ++i;
  }

  std::ostringstream oss;
  oss << std::fixed << std::setprecision(precision) << bytes_per_second << " " << units[i];
  return oss.str();
}

// Size of the data in the benchmark dataframe; chosen to be low enough to allow benchmarks to
// run on most GPUs, but large enough to allow highest throughput
constexpr int64_t data_size = 512 << 20;

void PQ_write(nvbench::state& state)
{
  cudf::size_type const num_cols = state.get_int64("num_cols");
  auto const rg_size_bytes       = state.get_int64("row_group_size_bytes");
  auto const rg_size_rows        = state.get_int64("row_group_size_rows");

  auto const tbl  = create_random_table(cycle_dtypes({cudf::type_id::INT32}, num_cols),
                                       table_size_bytes{data_size});
  auto const view = tbl->view();

  std::size_t encoded_file_size = 0;
  auto const mem_stats_logger   = cudf::memory_stats_logger();

  state.set_cuda_stream(nvbench::make_cuda_stream_view(cudf::get_default_stream().value()));
  state.exec(nvbench::exec_tag::timer | nvbench::exec_tag::sync,
             [&](nvbench::launch& launch, auto& timer) {
               cuio_source_sink_pair source_sink(io_type::VOID);

               timer.start();
               cudf::io::parquet_writer_options opts =
                 cudf::io::parquet_writer_options::builder(source_sink.make_sink_info(), view);
               // Sentinel 0 == use cuDF default (parquet bytes default is size_t::max).
               if (rg_size_bytes > 0) opts.set_row_group_size_bytes(rg_size_bytes);
               if (rg_size_rows > 0) opts.set_row_group_size_rows(rg_size_rows);
               cudf::io::write_parquet(opts);
               timer.stop();

               encoded_file_size = source_sink.size();
             });

  auto const time       = state.get_summary("nv/cold/time/gpu/mean").get_float64("value");
  auto const throughput = static_cast<double>(data_size) / time;
  state.add_element_count(throughput, "bytes_per_second");
  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");
  state.add_buffer_size(encoded_file_size, "encoded_file_size", "encoded_file_size");
  auto& summary = state.add_summary("format_throughput");
  summary.set_string("name", "throughput");
  summary.set_string("value", format_throughput(throughput));
}

void PQ_write_chunked(nvbench::state& state)
{
  cudf::size_type const num_cols   = state.get_int64("num_cols");
  cudf::size_type const num_chunks = state.get_int64("num_chunks");
  auto const rg_size_bytes         = state.get_int64("row_group_size_bytes");
  auto const rg_size_rows          = state.get_int64("row_group_size_rows");

  std::vector<std::unique_ptr<cudf::table>> tables;
  for (cudf::size_type idx = 0; idx < num_chunks; idx++) {
    tables.push_back(create_random_table(cycle_dtypes({cudf::type_id::INT32}, num_cols),
                                         table_size_bytes{size_t(data_size / num_chunks)}));
  }

  auto const mem_stats_logger   = cudf::memory_stats_logger();
  std::size_t encoded_file_size = 0;

  state.set_cuda_stream(nvbench::make_cuda_stream_view(cudf::get_default_stream().value()));
  state.exec(
    nvbench::exec_tag::timer | nvbench::exec_tag::sync, [&](nvbench::launch& launch, auto& timer) {
      cuio_source_sink_pair source_sink(io_type::VOID);

      timer.start();
      cudf::io::chunked_parquet_writer_options opts =
        cudf::io::chunked_parquet_writer_options::builder(source_sink.make_sink_info());
      if (rg_size_bytes > 0) opts.set_row_group_size_bytes(rg_size_bytes);
      if (rg_size_rows > 0) opts.set_row_group_size_rows(rg_size_rows);
      cudf::io::chunked_parquet_writer writer(opts);
      std::for_each(tables.begin(),
                    tables.end(),
                    [&writer](std::unique_ptr<cudf::table> const& tbl) { writer.write(*tbl); });
      writer.close();
      timer.stop();

      encoded_file_size = source_sink.size();
    });

  auto const time       = state.get_summary("nv/cold/time/gpu/mean").get_float64("value");
  auto const throughput = static_cast<double>(data_size) / time;

  state.add_element_count(throughput, "bytes_per_second");
  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");
  state.add_buffer_size(encoded_file_size, "encoded_file_size", "encoded_file_size");

  auto& summary = state.add_summary("format_throughput");
  summary.set_string("name", "throughput");
  summary.set_string("value", format_throughput(throughput));
}

NVBENCH_BENCH(PQ_write)
  .set_name("parquet_write_num_cols")
  .set_min_samples(4)
  .add_int64_axis("num_cols", {8, 1024})
  .add_int64_axis("row_group_size_bytes", {0})
  .add_int64_axis("row_group_size_rows", {0});

NVBENCH_BENCH(PQ_write_chunked)
  .set_name("parquet_chunked_write")
  .set_min_samples(4)
  .add_int64_axis("num_cols", {8, 1024})
  .add_int64_axis("num_chunks", {1, 2, 4, 8, 32, 64})
  .add_int64_axis("row_group_size_bytes", {0})
  .add_int64_axis("row_group_size_rows", {0});
