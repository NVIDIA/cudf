/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <benchmarks/common/generate_input.hpp>
#include <benchmarks/common/memory_stats.hpp>

#include <cudf/strings/repeat_strings.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <nvbench/nvbench.cuh>

static void bench_repeat(nvbench::state& state)
{
  auto const num_rows   = static_cast<cudf::size_type>(state.get_int64("num_rows"));
  auto const min_width  = static_cast<cudf::size_type>(state.get_int64("min_width"));
  auto const max_width  = static_cast<cudf::size_type>(state.get_int64("max_width"));
  auto const min_repeat = static_cast<cudf::size_type>(state.get_int64("min_repeat"));
  auto const max_repeat = static_cast<cudf::size_type>(state.get_int64("max_repeat"));
  auto const api        = state.get_string("api");

  auto builder = data_profile_builder().distribution(
    cudf::type_id::STRING, distribution_id::NORMAL, min_width, max_width);
  builder.distribution(cudf::type_id::INT32, distribution_id::NORMAL, min_repeat, max_repeat);

  auto const table = create_random_table(
    {cudf::type_id::STRING, cudf::type_id::INT32}, row_count{num_rows}, data_profile{builder});
  auto const input = cudf::strings_column_view(table->view().column(0));

  auto stream = cudf::get_default_stream();
  state.set_cuda_stream(nvbench::make_cuda_stream_view(stream.value()));
  auto const data_size = table->alloc_size();
  state.add_global_memory_reads<nvbench::int8_t>(data_size);

  auto const mem_stats_logger = cudf::memory_stats_logger();
  if (api == "scalar") {
    state.add_global_memory_writes<nvbench::int8_t>(data_size * max_repeat);
    state.exec(nvbench::exec_tag::sync,
               [&](nvbench::launch& launch) { cudf::strings::repeat_strings(input, max_repeat); });
  } else if (api == "column") {
    auto repeats = table->view().column(1);
    {
      auto result = cudf::strings::repeat_strings(input, repeats);
      state.add_global_memory_writes<nvbench::int8_t>(result->alloc_size());
    }
    state.exec(nvbench::exec_tag::sync,
               [&](nvbench::launch& launch) { cudf::strings::repeat_strings(input, repeats); });
  }
  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");
}

NVBENCH_BENCH(bench_repeat)
  .set_name("repeat")
  .add_int64_axis("min_width", {0})
  .add_int64_axis("max_width", {32, 64, 128, 256})
  .add_int64_axis("min_repeat", {0})
  .add_int64_axis("max_repeat", {16})
  .add_int64_axis("num_rows", {32768, 262144, 2097152})
  .add_string_axis("api", {"scalar", "column"});
