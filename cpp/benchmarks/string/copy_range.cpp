/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <benchmarks/common/generate_input.hpp>
#include <benchmarks/common/memory_stats.hpp>

#include <cudf/copying.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <nvbench/nvbench.cuh>

static void bench_copy_range(nvbench::state& state)
{
  auto const num_rows  = static_cast<cudf::size_type>(state.get_int64("num_rows"));
  auto const min_width = static_cast<cudf::size_type>(state.get_int64("min_width"));
  auto const max_width = static_cast<cudf::size_type>(state.get_int64("max_width"));

  data_profile const table_profile =
    data_profile_builder()
      .distribution(cudf::type_id::STRING, distribution_id::NORMAL, min_width, max_width)
      .no_validity();
  auto const source_tables = create_random_table(
    {cudf::type_id::STRING, cudf::type_id::STRING}, row_count{num_rows}, table_profile);

  auto const start  = num_rows / 4;
  auto const end    = (num_rows * 3) / 4;
  auto const source = source_tables->view().column(0);
  auto const target = source_tables->view().column(1);

  state.set_cuda_stream(nvbench::make_cuda_stream_view(cudf::get_default_stream().value()));
  auto data_size = source_tables->alloc_size();
  state.add_global_memory_reads<nvbench::int8_t>(data_size);   // all bytes are read;
  state.add_global_memory_writes<nvbench::int8_t>(data_size);  // both columns are similar size

  auto const mem_stats_logger = cudf::memory_stats_logger();
  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
    [[maybe_unused]] auto result = cudf::copy_range(source, target, start, end, start / 2);
  });
  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");
}

NVBENCH_BENCH(bench_copy_range)
  .set_name("copy_range")
  .add_int64_axis("min_width", {0})
  .add_int64_axis("max_width", {32, 64, 128, 256})
  .add_int64_axis("num_rows", {32768, 262144, 2097152});
