/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <benchmarks/common/generate_input.hpp>
#include <benchmarks/common/memory_stats.hpp>

#include <cudf/scalar/scalar.hpp>
#include <cudf/strings/char_types/char_types.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/strings/strip.hpp>
#include <cudf/strings/translate.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <nvbench/nvbench.cuh>

#include <vector>

static void bench_filter(nvbench::state& state)
{
  auto const num_rows  = static_cast<cudf::size_type>(state.get_int64("num_rows"));
  auto const min_width = static_cast<cudf::size_type>(state.get_int64("min_width"));
  auto const max_width = static_cast<cudf::size_type>(state.get_int64("max_width"));
  auto const api       = state.get_string("api");

  data_profile const profile = data_profile_builder().distribution(
    cudf::type_id::STRING, distribution_id::NORMAL, min_width, max_width);
  auto const column = create_random_column(cudf::type_id::STRING, row_count{num_rows}, profile);
  auto const input  = cudf::strings_column_view(column->view());

  auto stream = cudf::get_default_stream();
  state.set_cuda_stream(nvbench::make_cuda_stream_view(stream.value()));
  auto data_size = column->alloc_size();
  state.add_global_memory_reads<nvbench::int8_t>(data_size);

  if (api == "filter") {
    auto const types = cudf::strings::string_character_types::SPACE;
    {
      auto result = cudf::strings::filter_characters_of_type(input, types);
      state.add_global_memory_writes<nvbench::int8_t>(result->alloc_size());
    }
    auto const mem_stats_logger = cudf::memory_stats_logger();
    state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
      cudf::strings::filter_characters_of_type(input, types);
    });
    state.add_buffer_size(
      mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");
  } else if (api == "chars") {
    state.add_global_memory_writes<nvbench::int8_t>(data_size);
    std::vector<std::pair<cudf::char_utf8, cudf::char_utf8>> filter_table{
      {cudf::char_utf8{'a'}, cudf::char_utf8{'c'}}};
    auto const mem_stats_logger = cudf::memory_stats_logger();
    state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
      cudf::strings::filter_characters(input, filter_table);
    });
    state.add_buffer_size(
      mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");
  } else if (api == "strip") {
    {
      auto result = cudf::strings::strip(input);
      state.add_global_memory_writes<nvbench::int8_t>(result->alloc_size());
    }
    auto const mem_stats_logger = cudf::memory_stats_logger();
    state.exec(nvbench::exec_tag::sync,
               [&](nvbench::launch& launch) { cudf::strings::strip(input); });
    state.add_buffer_size(
      mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");
  }
}

NVBENCH_BENCH(bench_filter)
  .set_name("filter")
  .add_int64_axis("min_width", {0})
  .add_int64_axis("max_width", {32, 64, 128, 256})
  .add_int64_axis("num_rows", {32768, 262144, 2097152})
  .add_string_axis("api", {"filter", "chars", "strip"});
