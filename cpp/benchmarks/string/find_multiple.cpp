/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <benchmarks/common/generate_input.hpp>
#include <benchmarks/common/memory_stats.hpp>

#include <cudf_test/column_wrapper.hpp>

#include <cudf/strings/find.hpp>
#include <cudf/strings/find_multiple.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <nvbench/nvbench.cuh>

static void bench_find_string(nvbench::state& state)
{
  auto const n_rows       = static_cast<cudf::size_type>(state.get_int64("num_rows"));
  auto const row_width    = static_cast<cudf::size_type>(state.get_int64("row_width"));
  auto const hit_rate     = static_cast<cudf::size_type>(state.get_int64("hit_rate"));
  auto const target_count = static_cast<cudf::size_type>(state.get_int64("targets"));
  auto const api          = state.get_string("api");

  auto const stream = cudf::get_default_stream();
  auto const col    = create_string_column(n_rows, row_width, hit_rate);
  auto const input  = cudf::strings_column_view(col->view());

  // Note that these all match the first row of the raw_data in create_string_column.
  // This is so the hit_rate can properly accounted for.
  std::vector<std::string> const target_data(
    {" abc", "W43", "0987 5W43", "123 abc", "23 abc", "3 abc", "7 5W43", "87 5W43", "987 5W43"});
  auto h_targets = std::vector<std::string>{};
  for (cudf::size_type i = 0; i < target_count; i++) {
    h_targets.emplace_back(target_data[i % target_data.size()]);
  }
  cudf::test::strings_column_wrapper targets(h_targets.begin(), h_targets.end());

  state.set_cuda_stream(nvbench::make_cuda_stream_view(stream.value()));
  auto const data_size = col->alloc_size();
  state.add_global_memory_reads<nvbench::int8_t>(data_size);
  if (api == "find") {
    state.add_global_memory_writes<nvbench::int32_t>(input.size());
  } else {
    state.add_global_memory_writes<nvbench::int8_t>(input.size());
  }

  auto const mem_stats_logger = cudf::memory_stats_logger();
  if (api == "find") {
    state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
      cudf::strings::find_multiple(input, cudf::strings_column_view(targets));
    });
  } else if (api == "contains") {
    state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
      cudf::strings::contains_multiple(input, cudf::strings_column_view(targets));
    });
  }
  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");
}

NVBENCH_BENCH(bench_find_string)
  .set_name("find_multiple")
  .add_string_axis("api", {"find", "contains"})
  .add_int64_axis("targets", {10, 20, 40})
  .add_int64_axis("row_width", {32, 64, 128, 256})
  .add_int64_axis("num_rows", {32768, 262144, 2097152})
  .add_int64_axis("hit_rate", {20, 80});  // percentage
