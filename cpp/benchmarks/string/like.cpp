/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <benchmarks/common/generate_input.hpp>
#include <benchmarks/common/memory_stats.hpp>

#include <cudf_test/column_wrapper.hpp>

#include <cudf/strings/contains.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <nvbench/nvbench.cuh>

#include <string>
#include <utility>
#include <vector>

static void bench_like(nvbench::state& state)
{
  auto const n_rows    = static_cast<cudf::size_type>(state.get_int64("num_rows"));
  auto const row_width = static_cast<cudf::size_type>(state.get_int64("row_width"));
  auto const hit_rate  = static_cast<int32_t>(state.get_int64("hit_rate"));
  auto const ptn_idx   = state.get_int64("pattern");

  auto patterns = std::vector<std::string_view>{
    "_% % 5W4_",          // forces reading the entire target string (when matched expected)
    "%abc%6789%0987%5W%"  // middle-only pattern is common in tpch queries
  };

  if (std::cmp_greater_equal(ptn_idx, patterns.size())) {
    state.skip("invalid pattern index");
    return;
  }

  auto col     = create_string_column(n_rows, row_width, hit_rate);
  auto input   = cudf::strings_column_view(col->view());
  auto pattern = std::string_view(patterns[ptn_idx]);

  state.set_cuda_stream(nvbench::make_cuda_stream_view(cudf::get_default_stream().value()));
  // gather some throughput statistics as well
  auto const data_size = col->alloc_size();
  state.add_global_memory_reads<nvbench::int8_t>(data_size);  // all bytes are read;
  state.add_global_memory_writes<nvbench::int8_t>(n_rows);    // writes are BOOL8

  auto const mem_stats_logger = cudf::memory_stats_logger();
  state.exec(nvbench::exec_tag::sync,
             [&](nvbench::launch&) { cudf::strings::like(input, pattern); });
  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");
}

NVBENCH_BENCH(bench_like)
  .set_name("strings_like")
  .add_int64_axis("row_width", {32, 64, 128, 256})
  .add_int64_axis("num_rows", {262144, 2097152, 20971520})
  .add_int64_axis("hit_rate", {10, 25, 70, 100})
  .add_int64_axis("pattern", {0, 1});
