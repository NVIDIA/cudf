/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <benchmarks/common/generate_input.hpp>
#include <benchmarks/common/memory_stats.hpp>

#include <cudf/scalar/scalar.hpp>
#include <cudf/strings/strings_column_view.hpp>

#include <nvtext/generate_ngrams.hpp>

#include <nvbench/nvbench.cuh>

static void bench_ngrams(nvbench::state& state)
{
  auto const num_rows   = static_cast<cudf::size_type>(state.get_int64("num_rows"));
  auto const row_width  = static_cast<cudf::size_type>(state.get_int64("row_width"));
  auto const ngram_type = state.get_string("type");

  data_profile const profile = data_profile_builder().distribution(
    cudf::type_id::STRING, distribution_id::NORMAL, 0, row_width);
  auto const column = create_random_column(cudf::type_id::STRING, row_count{num_rows}, profile);
  cudf::strings_column_view input(column->view());
  auto const separator = cudf::string_scalar("_");

  state.set_cuda_stream(nvbench::make_cuda_stream_view(cudf::get_default_stream().value()));

  auto chars_size = input.chars_size(cudf::get_default_stream());
  state.add_global_memory_reads<nvbench::int8_t>(chars_size);
  state.add_global_memory_writes<nvbench::int8_t>(chars_size * 2);

  auto const mem_stats_logger = cudf::memory_stats_logger();

  if (ngram_type == "chars") {
    state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
      auto result = nvtext::generate_character_ngrams(input);
    });
  } else {
    state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
      auto result = nvtext::generate_ngrams(input, 2, separator);
    });
  }

  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");
}

NVBENCH_BENCH(bench_ngrams)
  .set_name("ngrams")
  .add_int64_axis("num_rows", {131072, 262144, 524288, 1048578})
  .add_int64_axis("row_width", {10, 20, 40, 100})
  .add_string_axis("type", {"chars", "tokens"});
