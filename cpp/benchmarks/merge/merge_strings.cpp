/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <benchmarks/common/generate_input.hpp>
#include <benchmarks/common/memory_stats.hpp>

#include <cudf/merge.hpp>
#include <cudf/sorting.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <nvbench/nvbench.cuh>

void nvbench_merge_strings(nvbench::state& state)
{
  auto stream = cudf::get_default_stream();

  auto const num_rows  = static_cast<cudf::size_type>(state.get_int64("num_rows"));
  auto const row_width = static_cast<cudf::size_type>(state.get_int64("row_width"));
  if (static_cast<std::size_t>(2 * num_rows) * static_cast<std::size_t>(row_width) >=
      static_cast<std::size_t>(std::numeric_limits<cudf::size_type>::max())) {
    state.skip("Skip benchmarks greater than size_type limit");
    return;
  }

  data_profile const table_profile =
    data_profile_builder()
      .distribution(cudf::type_id::STRING, distribution_id::NORMAL, 0, row_width)
      .no_validity();
  auto const source_tables = create_random_table(
    {cudf::type_id::STRING, cudf::type_id::STRING}, row_count{num_rows}, table_profile);

  auto const sorted_lhs = cudf::sort(cudf::table_view({source_tables->view().column(0)}));
  auto const sorted_rhs = cudf::sort(cudf::table_view({source_tables->view().column(1)}));
  auto const lhs        = sorted_lhs->view().column(0);
  auto const rhs        = sorted_rhs->view().column(0);

  state.set_cuda_stream(nvbench::make_cuda_stream_view(stream.value()));
  auto chars_size = cudf::strings_column_view(lhs).chars_size(stream) +
                    cudf::strings_column_view(rhs).chars_size(stream);
  state.add_global_memory_reads<nvbench::int8_t>(chars_size);   // all bytes are read
  state.add_global_memory_writes<nvbench::int8_t>(chars_size);  // all bytes are written

  auto const mem_stats_logger = cudf::memory_stats_logger();

  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
    [[maybe_unused]] auto result = cudf::merge(
      {cudf::table_view({lhs}), cudf::table_view({rhs})}, {0}, {cudf::order::ASCENDING});
  });

  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");
}

NVBENCH_BENCH(nvbench_merge_strings)
  .set_name("merge_strings")
  .add_int64_axis("row_width", {32, 64, 128, 256, 512, 1024, 2048, 4096})
  .add_int64_axis("num_rows", {4096, 32768, 262144, 2097152, 16777216});
