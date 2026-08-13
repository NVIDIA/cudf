/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <benchmarks/common/generate_input.hpp>
#include <benchmarks/common/memory_stats.hpp>

#include <cudf/strings/regex/regex_program.hpp>
#include <cudf/strings/replace_re.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <nvbench/nvbench.cuh>

static std::vector<std::string> const patterns = {
  "\\d+",                  // 0: builtins class and quantifier
  " ",                     // 1: literal
  "[a-z]+[A-Z]+",          // 2: multiple classes
  "[a-f]+|[0-5]+",         // 3: alternation (comparable density to \d+)
  "[a-z][0-9]{0,3}[A-Z]",  // 4: bounded repetition / gap transitions
  ".+[0-9]",               // 5: late-failure stress (~97% hit rate)
  "[a-z]+Z",               // 6: late-failure + low hit rate (~23% on 32-char, ~79% on 256-char)
};

static void bench_replace(nvbench::state& state)
{
  auto const num_rows      = static_cast<cudf::size_type>(state.get_int64("num_rows"));
  auto const min_width     = static_cast<cudf::size_type>(state.get_int64("min_width"));
  auto const max_width     = static_cast<cudf::size_type>(state.get_int64("max_width"));
  auto const pattern_index = state.get_int64("pattern");
  auto const rtype         = state.get_string("type");

  if (pattern_index < 0 || std::cmp_greater_equal(pattern_index, patterns.size())) {
    state.skip("invalid pattern index");
    return;
  }

  data_profile const profile = data_profile_builder().distribution(
    cudf::type_id::STRING, distribution_id::NORMAL, min_width, max_width);
  auto const column = create_random_column(cudf::type_id::STRING, row_count{num_rows}, profile);
  cudf::strings_column_view input(column->view());

  // Wrap in a capture group for backref replace so \1 references the whole match
  auto const pat =
    (rtype == "backref") ? "(" + patterns[pattern_index] + ")" : patterns[pattern_index];
  auto program = cudf::strings::regex_program::create(pat);

  auto const data_size = column->alloc_size();
  state.add_global_memory_reads<nvbench::int8_t>(data_size);
  state.add_global_memory_writes<nvbench::int8_t>(data_size);

  auto const mem_stats_logger = cudf::memory_stats_logger();
  if (rtype == "backref") {
    auto replacement = std::string("#\\1X");
    state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
      cudf::strings::replace_with_backrefs(input, *program, replacement);
    });
  } else {
    auto replacement = std::string_view("77");
    state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
      cudf::strings::replace_re(input, *program, replacement);
    });
  }
  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");
}

NVBENCH_BENCH(bench_replace)
  .set_name("replace_re")
  .add_int64_axis("min_width", {0})
  .add_int64_axis("max_width", {64, 128, 256})
  .add_int64_axis("num_rows", {262144, 2097152})
  .add_int64_axis("pattern", {0, 1, 2, 3, 4, 5, 6})
  .add_string_axis("type", {"replace", "backref"});
