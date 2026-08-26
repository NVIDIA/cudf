/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <benchmarks/common/generate_input.hpp>
#include <benchmarks/common/memory_stats.hpp>
#include <benchmarks/io/cuio_common.hpp>
#include <benchmarks/io/nvbench_helpers.hpp>

#include <cudf/io/json.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <nvbench/nvbench.cuh>

// Size of the data in the benchmark dataframe; chosen to be low enough to allow benchmarks to
// run on most GPUs, but large enough to allow highest throughput
constexpr size_t data_size         = 512 << 20;
constexpr cudf::size_type num_cols = 64;

void json_write_common(cudf::io::json_writer_options const& write_opts,
                       cuio_source_sink_pair& source_sink,
                       size_t const data_size,
                       nvbench::state& state)
{
  auto mem_stats_logger = cudf::memory_stats_logger();
  state.set_cuda_stream(nvbench::make_cuda_stream_view(cudf::get_default_stream().value()));
  state.exec(nvbench::exec_tag::sync | nvbench::exec_tag::timer,
             [&](nvbench::launch& launch, auto& timer) {
               drop_page_cache_if_enabled(write_opts.get_sink().filepaths());

               timer.start();
               cudf::io::write_json(write_opts);
               timer.stop();
             });

  auto const time = state.get_summary("nv/cold/time/gpu/mean").get_float64("value");
  state.add_element_count(static_cast<double>(data_size) / time, "bytes_per_second");
  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");
  state.add_buffer_size(source_sink.size(), "encoded_file_size", "encoded_file_size");
}

template <io_type IO>
void BM_json_write_io(nvbench::state& state, nvbench::type_list<nvbench::enum_type<IO>>)
{
  auto const d_type = get_type_or_group({static_cast<int32_t>(data_type::INTEGRAL),
                                         static_cast<int32_t>(data_type::FLOAT),
                                         static_cast<int32_t>(data_type::DECIMAL),
                                         static_cast<int32_t>(data_type::TIMESTAMP),
                                         static_cast<int32_t>(data_type::DURATION),
                                         static_cast<int32_t>(data_type::STRING),
                                         static_cast<int32_t>(data_type::LIST),
                                         static_cast<int32_t>(data_type::STRUCT)});

  auto const source_type = IO;

  auto const tbl = create_random_table(
    cycle_dtypes(d_type, num_cols), table_size_bytes{data_size}, data_profile_builder());
  auto const view = tbl->view();

  cuio_source_sink_pair source_sink(source_type);
  cudf::io::json_writer_options write_opts =
    cudf::io::json_writer_options::builder(source_sink.make_sink_info(), view)
      .na_rep("null")
      .rows_per_chunk(view.num_rows() / 10);

  json_write_common(write_opts, source_sink, data_size, state);
}

void BM_json_writer_options(nvbench::state& state)
{
  auto const source_type    = io_type::HOST_BUFFER;
  bool const json_lines     = state.get_int64("json_lines");
  bool const include_nulls  = state.get_int64("include_nulls");
  auto const rows_per_chunk = state.get_int64("rows_per_chunk");

  if ((json_lines or include_nulls) and rows_per_chunk != 1 << 20) {
    state.skip("Skipping for unrequired rows_per_chunk combinations");
    return;
  }
  auto const d_type = get_type_or_group({static_cast<int32_t>(data_type::INTEGRAL),
                                         static_cast<int32_t>(data_type::FLOAT),
                                         static_cast<int32_t>(data_type::DECIMAL),
                                         static_cast<int32_t>(data_type::TIMESTAMP),
                                         static_cast<int32_t>(data_type::DURATION),
                                         static_cast<int32_t>(data_type::STRING),
                                         static_cast<int32_t>(data_type::LIST),
                                         static_cast<int32_t>(data_type::STRUCT)});

  auto const tbl = create_random_table(
    cycle_dtypes(d_type, num_cols), table_size_bytes{data_size}, data_profile_builder());
  auto const view = tbl->view();

  cuio_source_sink_pair source_sink(source_type);
  cudf::io::json_writer_options write_opts =
    cudf::io::json_writer_options::builder(source_sink.make_sink_info(), view)
      .na_rep("null")
      .lines(json_lines)
      .include_nulls(include_nulls)
      .rows_per_chunk(rows_per_chunk);

  json_write_common(write_opts, source_sink, data_size, state);
}

using io_list =
  nvbench::enum_type_list<io_type::FILEPATH, io_type::HOST_BUFFER, io_type::DEVICE_BUFFER>;

NVBENCH_BENCH_TYPES(BM_json_write_io, NVBENCH_TYPE_AXES(io_list))
  .set_name("json_write_io")
  .set_type_axes_names({"io"})
  .set_min_samples(4);

NVBENCH_BENCH(BM_json_writer_options)
  .set_name("json_write_options")
  .set_min_samples(4)
  .add_int64_axis("json_lines", {false, true})
  .add_int64_axis("include_nulls", {false, true})
  .add_int64_power_of_two_axis("rows_per_chunk", {10, 15, 16, 18, 20});
