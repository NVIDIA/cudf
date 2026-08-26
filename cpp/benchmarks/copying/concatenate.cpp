/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <benchmarks/common/generate_input.hpp>
#include <benchmarks/common/memory_stats.hpp>

#include <cudf/column/column_view.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <nvbench/nvbench.cuh>

#include <vector>

static void bench_concatenate(nvbench::state& state)
{
  auto const num_rows = static_cast<cudf::size_type>(state.get_int64("num_rows"));
  auto const num_cols = static_cast<cudf::size_type>(state.get_int64("num_cols"));
  auto const nulls    = static_cast<cudf::size_type>(state.get_float64("nulls"));

  auto input = create_sequence_table(
    cycle_dtypes({cudf::type_to_id<int64_t>()}, num_cols), row_count{num_rows}, nulls);
  auto input_columns = input->view();
  auto column_views  = std::vector<cudf::column_view>(input_columns.begin(), input_columns.end());

  auto stream = cudf::get_default_stream();
  state.set_cuda_stream(nvbench::make_cuda_stream_view(stream.value()));
  state.add_global_memory_reads<int64_t>(num_rows * num_cols);
  state.add_global_memory_writes<int64_t>(num_rows * num_cols);

  auto const mem_stats_logger = cudf::memory_stats_logger();

  state.exec(nvbench::exec_tag::sync,
             [&](nvbench::launch&) { auto result = cudf::concatenate(column_views); });

  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");
}

NVBENCH_BENCH(bench_concatenate)
  .set_name("concatenate")
  .add_int64_axis("num_rows", {64, 512, 4096, 32768, 262144})
  .add_int64_axis("num_cols", {2, 8, 64, 512, 1024})
  .add_float64_axis("nulls", {0.0, 0.3});

static void bench_concatenate_strings(nvbench::state& state)
{
  auto const num_rows  = static_cast<cudf::size_type>(state.get_int64("num_rows"));
  auto const num_cols  = static_cast<cudf::size_type>(state.get_int64("num_cols"));
  auto const row_width = static_cast<cudf::size_type>(state.get_int64("row_width"));
  auto const nulls     = static_cast<cudf::size_type>(state.get_float64("nulls"));

  data_profile const profile =
    data_profile_builder()
      .distribution(cudf::type_id::STRING, distribution_id::NORMAL, 0, row_width)
      .null_probability(nulls);
  auto const column = create_random_column(cudf::type_id::STRING, row_count{num_rows}, profile);
  auto const input  = column->view();

  auto column_views = std::vector<cudf::column_view>(num_cols, input);

  auto stream = cudf::get_default_stream();
  state.set_cuda_stream(nvbench::make_cuda_stream_view(stream.value()));
  auto const sv = cudf::strings_column_view(input);
  state.add_global_memory_reads<int8_t>(sv.chars_size(stream) * num_cols);
  state.add_global_memory_writes<int64_t>(sv.chars_size(stream) * num_cols);

  auto const mem_stats_logger = cudf::memory_stats_logger();

  state.exec(nvbench::exec_tag::sync,
             [&](nvbench::launch&) { auto result = cudf::concatenate(column_views); });

  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");
}

NVBENCH_BENCH(bench_concatenate_strings)
  .set_name("concatenate_strings")
  .add_int64_axis("num_rows", {256, 512, 4096, 16384})
  .add_int64_axis("num_cols", {2, 8, 64, 256})
  .add_int64_axis("row_width", {32, 128})
  .add_float64_axis("nulls", {0.0, 0.3});
