/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <benchmarks/common/generate_input.hpp>
#include <benchmarks/common/memory_stats.hpp>

#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/replace.hpp>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/table/table.hpp>
#include <cudf/types.hpp>

#include <nvbench/nvbench.cuh>

static void replace_nulls(nvbench::state& state)
{
  auto const n_rows    = static_cast<cudf::size_type>(state.get_int64("num_rows"));
  auto const max_width = static_cast<int32_t>(state.get_int64("row_width"));

  if (static_cast<std::size_t>(n_rows) * static_cast<std::size_t>(max_width) >=
      static_cast<std::size_t>(std::numeric_limits<cudf::size_type>::max())) {
    state.skip("Skip benchmarks greater than size_type limit");
    return;
  }

  data_profile const table_profile = data_profile_builder().distribution(
    cudf::type_id::STRING, distribution_id::NORMAL, 0, max_width);

  auto const input_table = create_random_table(
    {cudf::type_id::STRING, cudf::type_id::STRING}, row_count{n_rows}, table_profile);
  auto const input = input_table->view().column(0);
  auto const repl  = input_table->view().column(1);

  state.set_cuda_stream(nvbench::make_cuda_stream_view(cudf::get_default_stream().value()));
  auto chars_size = cudf::strings_column_view(input).chars_size(cudf::get_default_stream());
  state.add_global_memory_reads<nvbench::int8_t>(chars_size);  // all bytes are read;
  state.add_global_memory_writes<nvbench::int8_t>(chars_size);

  auto const mem_stats_logger = cudf::memory_stats_logger();

  state.exec(nvbench::exec_tag::sync,
             [&](nvbench::launch& launch) { auto result = cudf::replace_nulls(input, repl); });

  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");
}

NVBENCH_BENCH(replace_nulls)
  .set_name("replace_nulls")
  .add_int64_axis("row_width", {32, 64, 128, 256, 512, 1024, 2048})
  .add_int64_axis("num_rows", {32768, 262144, 2097152, 16777216});
