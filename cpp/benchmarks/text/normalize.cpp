/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <benchmarks/common/generate_input.hpp>
#include <benchmarks/common/memory_stats.hpp>

#include <cudf/scalar/scalar.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <nvtext/normalize.hpp>

#include <nvbench/nvbench.cuh>

static void bench_normalize(nvbench::state& state)
{
  auto const num_rows       = static_cast<cudf::size_type>(state.get_int64("num_rows"));
  auto const min_width      = static_cast<cudf::size_type>(state.get_int64("min_width"));
  auto const max_width      = static_cast<cudf::size_type>(state.get_int64("max_width"));
  auto const normalize_type = state.get_string("type");

  data_profile const profile = data_profile_builder().distribution(
    cudf::type_id::STRING, distribution_id::NORMAL, min_width, max_width);
  auto const column = create_random_column(cudf::type_id::STRING, row_count{num_rows}, profile);
  cudf::strings_column_view input(column->view());

  state.set_cuda_stream(nvbench::make_cuda_stream_view(cudf::get_default_stream().value()));

  auto chars_size = input.chars_size(cudf::get_default_stream());
  state.add_global_memory_reads<nvbench::int8_t>(chars_size);
  state.add_global_memory_writes<nvbench::int8_t>(chars_size);

  auto const mem_stats_logger = cudf::memory_stats_logger();

  if (normalize_type == "spaces") {
    state.exec(nvbench::exec_tag::sync,
               [&](nvbench::launch& launch) { auto result = nvtext::normalize_spaces(input); });
  } else {
    bool const to_lower = (normalize_type == "to_lower");
    // we expect the normalizer to be created once and re-used
    // so creating it is not measured
    auto normalizer = nvtext::create_character_normalizer(to_lower);
    state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
      auto result = nvtext::normalize_characters(input, *normalizer);
    });
  }

  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");
}

NVBENCH_BENCH(bench_normalize)
  .set_name("normalize")
  .add_int64_axis("min_width", {0})
  .add_int64_axis("max_width", {128, 256})
  .add_int64_axis("num_rows", {32768, 262144, 2097152})
  .add_string_axis("type", {"spaces", "characters", "to_lower"});
