/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <benchmarks/common/generate_input.hpp>
#include <benchmarks/common/memory_stats.hpp>

#include <cudf/strings/case.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <nvbench/nvbench.cuh>

void bench_case(nvbench::state& state)
{
  auto const num_rows  = static_cast<cudf::size_type>(state.get_int64("num_rows"));
  auto const min_width = static_cast<cudf::size_type>(state.get_int64("min_width"));
  auto const max_width = static_cast<cudf::size_type>(state.get_int64("max_width"));
  auto const encoding  = state.get_string("encoding");

  data_profile const profile = data_profile_builder().distribution(
    cudf::type_id::STRING, distribution_id::NORMAL, min_width, max_width);
  auto const column = create_random_column(cudf::type_id::STRING, row_count{num_rows}, profile);

  auto col_view = column->view();
  auto col_size = column->alloc_size();

  cudf::column::contents ascii_contents;
  if (encoding == "ascii") {
    data_profile ascii_profile = data_profile_builder().no_validity().distribution(
      cudf::type_id::INT8, distribution_id::UNIFORM, 32, 126);  // nice ASCII range
    auto input        = cudf::strings_column_view(col_view);
    auto ascii_column = create_random_column(
      cudf::type_id::INT8,
      row_count{static_cast<cudf::size_type>(input.chars_size(cudf::get_default_stream()))},
      ascii_profile);
    auto ascii_data = ascii_column->view();

    col_view       = cudf::column_view(col_view.type(),
                                 col_view.size(),
                                 ascii_data.data<char>(),
                                 col_view.null_mask(),
                                 col_view.null_count(),
                                 0,
                                       {input.offsets()});
    col_size       = ascii_column->alloc_size();
    ascii_contents = ascii_column->release();
  }
  auto input = cudf::strings_column_view(col_view);

  state.set_cuda_stream(nvbench::make_cuda_stream_view(cudf::get_default_stream().value()));

  state.add_global_memory_reads<nvbench::int8_t>(col_size);
  state.add_global_memory_writes<nvbench::int8_t>(col_size);

  auto const mem_stats_logger = cudf::memory_stats_logger();
  state.exec(nvbench::exec_tag::sync,
             [&](nvbench::launch& launch) { auto result = cudf::strings::to_lower(input); });
  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");
}

NVBENCH_BENCH(bench_case)
  .set_name("case")
  .add_int64_axis("min_width", {0})
  .add_int64_axis("max_width", {32, 64, 128, 256})
  .add_int64_axis("num_rows", {32768, 262144, 2097152})
  .add_string_axis("encoding", {"ascii", "utf8"});
