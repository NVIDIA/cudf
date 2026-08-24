/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <benchmarks/common/memory_stats.hpp>

#include <cudf_test/column_wrapper.hpp>

#include <cudf/strings/strings_column_view.hpp>

#include <nvtext/replace.hpp>

#include <nvbench/nvbench.cuh>

#include <random>

static void bench_replace(nvbench::state& state)
{
  auto const num_rows  = static_cast<cudf::size_type>(state.get_int64("num_rows"));
  auto const row_width = static_cast<cudf::size_type>(state.get_int64("row_width"));

  std::vector<std::string> words{" ",        "one  ",    "two ",       "three ",     "four ",
                                 "five ",    "six  ",    "sevén  ",    "eight ",     "nine ",
                                 "ten   ",   "eleven ",  "twelve ",    "thirteen  ", "fourteen ",
                                 "fifteen ", "sixteen ", "seventeen ", "eighteen ",  "nineteen "};

  std::default_random_engine generator;
  std::uniform_int_distribution<int> tokens_dist(0, words.size() - 1);
  std::string row;  // build a row of random tokens
  while (static_cast<cudf::size_type>(row.size()) < row_width)
    row += words[tokens_dist(generator)];

  std::uniform_int_distribution<int> position_dist(0, 16);

  auto elements = cudf::detail::make_counting_transform_iterator(
    0, [&](auto idx) { return row.c_str() + position_dist(generator); });
  cudf::test::strings_column_wrapper input(elements, elements + num_rows);
  cudf::strings_column_view view(input);

  cudf::test::strings_column_wrapper targets({"one", "two", "sevén", "zero"});
  cudf::test::strings_column_wrapper replacements({"1", "2", "7", "0"});

  state.set_cuda_stream(nvbench::make_cuda_stream_view(cudf::get_default_stream().value()));

  auto chars_size = view.chars_size(cudf::get_default_stream());
  state.add_global_memory_reads<nvbench::int8_t>(chars_size);
  state.add_global_memory_writes<nvbench::int8_t>(chars_size);

  auto const mem_stats_logger = cudf::memory_stats_logger();

  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
    auto result = nvtext::replace_tokens(
      view, cudf::strings_column_view(targets), cudf::strings_column_view(replacements));
  });

  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");
}

NVBENCH_BENCH(bench_replace)
  .set_name("replace")
  .add_int64_axis("row_width", {32, 64, 128, 256})
  .add_int64_axis("num_rows", {32768, 262144, 2097152});
