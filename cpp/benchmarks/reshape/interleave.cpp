/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <benchmarks/common/generate_input.hpp>
#include <benchmarks/common/memory_stats.hpp>

#include <cudf/reshape.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <nvbench/nvbench.cuh>

static void bench_interleave(nvbench::state& state)
{
  auto const num_rows  = static_cast<cudf::size_type>(state.get_int64("num_rows"));
  auto const row_width = static_cast<cudf::size_type>(state.get_int64("row_width"));
  auto const num_cols  = static_cast<cudf::size_type>(state.get_int64("columns"));

  if (static_cast<std::size_t>(num_rows) * static_cast<std::size_t>(row_width) * num_cols >=
      static_cast<std::size_t>(std::numeric_limits<cudf::size_type>::max())) {
    state.skip("Skip benchmarks greater than size_type limit");
    return;
  }

  data_profile const str_profile = data_profile_builder().distribution(
    cudf::type_id::STRING, distribution_id::NORMAL, 0, row_width);
  std::vector<cudf::type_id> types(num_cols, cudf::type_id::STRING);
  auto const source_table = create_random_table(types, row_count{num_rows}, str_profile);

  auto const source_view = source_table->view();
  auto const stream      = cudf::get_default_stream();

  state.set_cuda_stream(nvbench::make_cuda_stream_view(stream.value()));
  auto input_bytes = source_table->alloc_size();
  state.add_global_memory_reads<nvbench::int8_t>(input_bytes);   // all bytes are read
  state.add_global_memory_writes<nvbench::int8_t>(input_bytes);  // all bytes are written

  auto const mem_stats_logger = cudf::memory_stats_logger();
  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
    [[maybe_unused]] auto result = cudf::interleave_columns(source_view);
  });
  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");
}

NVBENCH_BENCH(bench_interleave)
  .set_name("interleave_strings")
  .add_int64_axis("row_width", {32, 64, 128, 256, 512, 1024})
  .add_int64_axis("num_rows", {32768, 262144, 2097152, 16777216})
  .add_int64_axis("columns", {2, 10, 100});
