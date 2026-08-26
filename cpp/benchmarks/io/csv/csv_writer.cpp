/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <benchmarks/common/generate_input.hpp>
#include <benchmarks/common/memory_stats.hpp>
#include <benchmarks/io/cuio_common.hpp>
#include <benchmarks/io/nvbench_helpers.hpp>

#include <cudf/io/csv.hpp>

#include <nvbench/nvbench.cuh>

// Size of the data in the benchmark dataframe; chosen to be low enough to allow benchmarks to
// run on most GPUs, but large enough to allow highest throughput
constexpr size_t data_size         = 256 << 20;
constexpr cudf::size_type num_cols = 64;

template <data_type DataType, io_type IO>
void BM_csv_write_dtype_io(nvbench::state& state,
                           nvbench::type_list<nvbench::enum_type<DataType>, nvbench::enum_type<IO>>)
{
  auto const data_types = get_type_or_group(static_cast<int32_t>(DataType));
  auto const sink_type  = IO;

  auto const tbl =
    create_random_table(cycle_dtypes(data_types, num_cols), table_size_bytes{data_size});
  auto const view = tbl->view();

  std::size_t encoded_file_size = 0;

  auto const mem_stats_logger = cudf::memory_stats_logger();
  state.set_cuda_stream(nvbench::make_cuda_stream_view(cudf::get_default_stream().value()));
  state.exec(nvbench::exec_tag::timer | nvbench::exec_tag::sync,
             [&](nvbench::launch& launch, auto& timer) {
               cuio_source_sink_pair source_sink(sink_type);

               timer.start();
               cudf::io::csv_writer_options options =
                 cudf::io::csv_writer_options::builder(source_sink.make_sink_info(), view);
               cudf::io::write_csv(options);
               timer.stop();

               encoded_file_size = source_sink.size();
             });

  auto const time = state.get_summary("nv/cold/time/gpu/mean").get_float64("value");
  state.add_element_count(static_cast<double>(data_size) / time, "bytes_per_second");
  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");
  state.add_buffer_size(encoded_file_size, "encoded_file_size", "encoded_file_size");
}

void BM_csv_write_varying_options(nvbench::state& state)
{
  auto const na_per_len     = state.get_int64("na_per_len");
  auto const rows_per_chunk = state.get_int64("rows_per_chunk");

  auto const data_types = get_type_or_group({static_cast<int32_t>(data_type::INTEGRAL),
                                             static_cast<int32_t>(data_type::FLOAT),
                                             static_cast<int32_t>(data_type::DECIMAL),
                                             static_cast<int32_t>(data_type::TIMESTAMP),
                                             static_cast<int32_t>(data_type::DURATION),
                                             static_cast<int32_t>(data_type::STRING)});

  auto const tbl  = create_random_table(data_types, table_size_bytes{data_size});
  auto const view = tbl->view();

  std::string const na_per(na_per_len, '#');
  std::size_t encoded_file_size = 0;

  auto const mem_stats_logger = cudf::memory_stats_logger();
  state.set_cuda_stream(nvbench::make_cuda_stream_view(cudf::get_default_stream().value()));
  state.exec(nvbench::exec_tag::timer | nvbench::exec_tag::sync,
             [&](nvbench::launch& launch, auto& timer) {
               cuio_source_sink_pair source_sink(io_type::HOST_BUFFER);

               timer.start();
               cudf::io::csv_writer_options options =
                 cudf::io::csv_writer_options::builder(source_sink.make_sink_info(), view)
                   .na_rep(na_per)
                   .rows_per_chunk(rows_per_chunk);
               cudf::io::write_csv(options);
               timer.stop();

               encoded_file_size = source_sink.size();
             });

  auto const time = state.get_summary("nv/cold/time/gpu/mean").get_float64("value");
  state.add_element_count(static_cast<double>(data_size) / time, "bytes_per_second");
  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");
  state.add_buffer_size(encoded_file_size, "encoded_file_size", "encoded_file_size");
}

using d_type_list = nvbench::enum_type_list<data_type::INTEGRAL,
                                            data_type::FLOAT,
                                            data_type::DECIMAL,
                                            data_type::TIMESTAMP,
                                            data_type::DURATION,
                                            data_type::STRING>;

using io_list = nvbench::enum_type_list<io_type::FILEPATH, io_type::HOST_BUFFER, io_type::VOID>;

NVBENCH_BENCH_TYPES(BM_csv_write_dtype_io, NVBENCH_TYPE_AXES(d_type_list, io_list))
  .set_name("csv_write_dtype_io")
  .set_type_axes_names({"data_type", "io"})
  .set_min_samples(4);

NVBENCH_BENCH(BM_csv_write_varying_options)
  .set_name("csv_write_options")
  .set_min_samples(4)
  .add_int64_axis("na_per_len", {0, 16})
  .add_int64_power_of_two_axis("rows_per_chunk", nvbench::range(8, 20, 2));
