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

static void bench_copy(nvbench::state& state)
{
  auto const num_rows  = static_cast<cudf::size_type>(state.get_int64("num_rows"));
  auto const min_width = static_cast<cudf::size_type>(state.get_int64("min_width"));
  auto const max_width = static_cast<cudf::size_type>(state.get_int64("max_width"));

  data_profile const str_profile = data_profile_builder().distribution(
    cudf::type_id::STRING, distribution_id::NORMAL, min_width, max_width);
  auto const source_table =
    create_random_table({cudf::type_id::STRING}, row_count{num_rows}, str_profile);
  auto const target_table =
    create_random_table({cudf::type_id::STRING}, row_count{num_rows}, str_profile);
  data_profile const bool_profile = data_profile_builder().no_validity();
  auto const booleans =
    create_random_table({cudf::type_id::BOOL8}, row_count{num_rows}, bool_profile);

  auto const source     = source_table->view().column(0);
  auto const target     = target_table->view().column(0);
  auto const left_right = booleans->view().column(0);

  state.set_cuda_stream(nvbench::make_cuda_stream_view(cudf::get_default_stream().value()));
  auto data_size = target_table->alloc_size();
  state.add_global_memory_reads<nvbench::int8_t>(data_size);   // all bytes are read;
  state.add_global_memory_writes<nvbench::int8_t>(data_size);  // both columns are similar size

  auto const mem_stats_logger = cudf::memory_stats_logger();
  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
    [[maybe_unused]] auto result = cudf::copy_if_else(source, target, left_right);
  });
  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");
}

NVBENCH_BENCH(bench_copy)
  .set_name("copy_if_else")
  .add_int64_axis("min_width", {0})
  .add_int64_axis("max_width", {32, 64, 128, 256})
  .add_int64_axis("num_rows", {32768, 262144, 2097152});
