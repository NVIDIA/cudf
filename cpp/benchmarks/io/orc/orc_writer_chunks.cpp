/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <benchmarks/common/generate_input.hpp>
#include <benchmarks/common/memory_stats.hpp>
#include <benchmarks/io/cuio_common.hpp>
#include <benchmarks/io/nvbench_helpers.hpp>

#include <cudf/column/column.hpp>
#include <cudf/io/orc.hpp>
#include <cudf/table/table.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <cuda/iterator>

#include <nvbench/nvbench.cuh>

// Size of the data in the benchmark dataframe; chosen to be low enough to allow benchmarks to
// run on most GPUs, but large enough to allow highest throughput
constexpr int64_t data_size = 512 << 20;

void nvbench_orc_write(nvbench::state& state)
{
  cudf::size_type num_cols     = state.get_int64("num_columns");
  auto const stripe_size_bytes = state.get_int64("stripe_size_bytes");
  auto const stripe_size_rows  = state.get_int64("stripe_size_rows");

  auto tbl = create_random_table(
    cycle_dtypes(get_type_or_group({static_cast<int32_t>(data_type::INTEGRAL_SIGNED),
                                    static_cast<int32_t>(data_type::FLOAT),
                                    static_cast<int32_t>(data_type::DECIMAL),
                                    static_cast<int32_t>(data_type::TIMESTAMP),
                                    static_cast<int32_t>(data_type::STRING),
                                    static_cast<int32_t>(data_type::STRUCT),
                                    static_cast<int32_t>(data_type::LIST)}),
                 num_cols),
    table_size_bytes{data_size});
  cudf::table_view view = tbl->view();

  auto mem_stats_logger = cudf::memory_stats_logger();

  state.add_global_memory_reads<int64_t>(data_size);
  state.add_element_count(view.num_columns() * view.num_rows());

  size_t encoded_file_size = 0;

  state.set_cuda_stream(nvbench::make_cuda_stream_view(cudf::get_default_stream().value()));
  state.exec(nvbench::exec_tag::timer | nvbench::exec_tag::sync,
             [&](nvbench::launch& launch, auto& timer) {
               cuio_source_sink_pair source_sink(io_type::VOID);
               timer.start();

               cudf::io::orc_writer_options opts =
                 cudf::io::orc_writer_options::builder(source_sink.make_sink_info(), view);
               // Sentinel 0 == use cuDF default.
               if (stripe_size_bytes > 0) opts.set_stripe_size_bytes(stripe_size_bytes);
               if (stripe_size_rows > 0) opts.set_stripe_size_rows(stripe_size_rows);
               cudf::io::write_orc(opts);

               timer.stop();
               encoded_file_size = source_sink.size();
             });

  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");
  state.add_buffer_size(encoded_file_size, "encoded_file_size", "encoded_file_size");
  state.add_element_count(view.num_rows(), "total_rows");
}

void nvbench_orc_chunked_write(nvbench::state& state)
{
  cudf::size_type num_cols     = state.get_int64("num_columns");
  cudf::size_type num_tables   = state.get_int64("num_chunks");
  auto const stripe_size_bytes = state.get_int64("stripe_size_bytes");
  auto const stripe_size_rows  = state.get_int64("stripe_size_rows");

  std::vector<std::unique_ptr<cudf::table>> tables;
  for (cudf::size_type idx = 0; idx < num_tables; idx++) {
    tables.push_back(create_random_table(
      cycle_dtypes(get_type_or_group({static_cast<int32_t>(data_type::INTEGRAL_SIGNED),
                                      static_cast<int32_t>(data_type::FLOAT),
                                      static_cast<int32_t>(data_type::DECIMAL),
                                      static_cast<int32_t>(data_type::TIMESTAMP),
                                      static_cast<int32_t>(data_type::STRING),
                                      static_cast<int32_t>(data_type::STRUCT),
                                      static_cast<int32_t>(data_type::LIST)}),
                   num_cols),
      table_size_bytes{size_t(data_size / num_tables)}));
  }

  auto mem_stats_logger = cudf::memory_stats_logger();

  auto size_iter = cuda::transform_iterator(
    tables.begin(), [](auto const& i) { return i->num_columns() * i->num_rows(); });
  auto row_count_iter =
    cuda::transform_iterator(tables.begin(), [](auto const& i) { return i->num_rows(); });
  auto total_elements = std::accumulate(size_iter, size_iter + num_tables, 0);
  auto total_rows     = std::accumulate(row_count_iter, row_count_iter + num_tables, 0);

  state.add_global_memory_reads<int64_t>(data_size);
  state.add_element_count(total_elements);

  size_t encoded_file_size = 0;

  state.set_cuda_stream(nvbench::make_cuda_stream_view(cudf::get_default_stream().value()));
  state.exec(
    nvbench::exec_tag::timer | nvbench::exec_tag::sync, [&](nvbench::launch& launch, auto& timer) {
      cuio_source_sink_pair source_sink(io_type::VOID);
      timer.start();

      cudf::io::chunked_orc_writer_options opts =
        cudf::io::chunked_orc_writer_options::builder(source_sink.make_sink_info());
      if (stripe_size_bytes > 0) opts.set_stripe_size_bytes(stripe_size_bytes);
      if (stripe_size_rows > 0) opts.set_stripe_size_rows(stripe_size_rows);
      cudf::io::orc_chunked_writer writer(opts);
      std::for_each(tables.begin(),
                    tables.end(),
                    [&writer](std::unique_ptr<cudf::table> const& tbl) { writer.write(*tbl); });
      writer.close();

      timer.stop();
      encoded_file_size = source_sink.size();
    });

  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");
  state.add_buffer_size(encoded_file_size, "encoded_file_size", "encoded_file_size");
  state.add_element_count(total_rows, "total_rows");
}

NVBENCH_BENCH(nvbench_orc_write)
  .set_name("orc_write")
  .set_min_samples(4)
  .add_int64_axis("num_columns", {8, 64})
  .add_int64_axis("stripe_size_bytes", {0})
  .add_int64_axis("stripe_size_rows", {0});

NVBENCH_BENCH(nvbench_orc_chunked_write)
  .set_name("orc_chunked_write")
  .set_min_samples(4)
  .add_int64_axis("num_columns", {8, 64})
  .add_int64_axis("num_chunks", {8, 64})
  .add_int64_axis("stripe_size_bytes", {0})
  .add_int64_axis("stripe_size_rows", {0});
