/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <benchmarks/common/generate_input.hpp>
#include <benchmarks/common/memory_stats.hpp>

#include <cudf_test/column_wrapper.hpp>

#include <cudf/sorting.hpp>
#include <cudf/strings/combine.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/table/table.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>

#include <nvbench/nvbench.cuh>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <random>
#include <string>
#include <vector>

namespace {

constexpr unsigned seed = 1;

void run_sorted_order_benchmark(nvbench::state& state, std::unique_ptr<cudf::column> const& input)
{
  state.set_cuda_stream(nvbench::make_cuda_stream_view(cudf::get_default_stream().get()));
  state.add_global_memory_reads<nvbench::int8_t>(input->alloc_size());
  state.add_global_memory_writes<cudf::size_type>(input->size());

  auto const mem_stats_logger = cudf::memory_stats_logger();

  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
    cudf::sorted_order(cudf::table_view{{input->view()}});
  });

  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");
}

std::unique_ptr<cudf::column> make_prefixed_input(cudf::size_type num_rows,
                                                  cudf::size_type prefix_width,
                                                  cudf::size_type suffix_width,
                                                  cudf::size_type prefix_cardinality)
{
  data_profile const prefix_profile =
    data_profile_builder()
      .no_validity()
      .cardinality(prefix_cardinality)
      .avg_run_length(1)
      .distribution(cudf::type_id::STRING, distribution_id::UNIFORM, prefix_width, prefix_width);
  data_profile const suffix_profile =
    data_profile_builder().no_validity().cardinality(0).avg_run_length(1).distribution(
      cudf::type_id::STRING, distribution_id::UNIFORM, 0, suffix_width);
  auto const prefix =
    create_random_column(cudf::type_id::STRING, row_count{num_rows}, prefix_profile, seed);
  // The general STRING generator includes non-ASCII characters; this suffix needs printable ASCII.
  auto const suffix = create_ascii_string_column(suffix_profile, num_rows, seed + 1);
  return cudf::strings::concatenate(cudf::table_view{{prefix->view(), suffix->view()}});
}

std::unique_ptr<cudf::column> make_cardinality_input(cudf::size_type num_rows,
                                                     cudf::size_type max_width,
                                                     cudf::size_type cardinality)
{
  data_profile const profile =
    data_profile_builder()
      .no_validity()
      .cardinality(cardinality)
      .avg_run_length(1)
      .distribution(cudf::type_id::STRING, distribution_id::UNIFORM, 0, max_width);
  return create_random_column(cudf::type_id::STRING, row_count{num_rows}, profile, seed);
}

std::unique_ptr<cudf::column> make_distribution_input(cudf::size_type num_rows,
                                                      std::string const& profile_name)
{
  if (profile_name == "cardinality_1_width_32") { return make_cardinality_input(num_rows, 32, 1); }
  if (profile_name == "cardinality_64_width_128") {
    return make_cardinality_input(num_rows, 128, 64);
  }
  if (profile_name == "shared_prefix_64") { return make_prefixed_input(num_rows, 64, 32, 1); }
  if (profile_name == "variable_128") { return make_cardinality_input(num_rows, 128, 0); }
  CUDF_FAIL("Unknown string distribution profile: " + profile_name);
}

std::unique_ptr<cudf::column> make_nullable_input(cudf::size_type num_rows,
                                                  std::string const& profile_name,
                                                  double null_probability)
{
  auto min_width = cudf::size_type{0};
  auto max_width = cudf::size_type{0};
  if (profile_name == "fixed_8") {
    min_width = max_width = 8;
  } else if (profile_name == "variable_128") {
    max_width = 128;
  } else {
    CUDF_FAIL("Unknown nullable string profile: " + profile_name);
  }

  data_profile const profile =
    data_profile_builder()
      .null_probability(null_probability)
      .cardinality(0)
      .avg_run_length(1)
      .distribution(cudf::type_id::STRING, distribution_id::UNIFORM, min_width, max_width);
  auto result = create_random_column(cudf::type_id::STRING, row_count{num_rows}, profile, seed);
  if (null_probability == 1.0) { result->set_null_count(num_rows); }
  return result;
}

}  // namespace

static void bench_sort_strings(nvbench::state& state)
{
  auto const num_rows  = static_cast<cudf::size_type>(state.get_int64("num_rows"));
  auto const min_width = static_cast<cudf::size_type>(state.get_int64("min_width"));
  auto const max_width = static_cast<cudf::size_type>(state.get_int64("max_width"));

  data_profile const profile = data_profile_builder().distribution(
    cudf::type_id::STRING, distribution_id::NORMAL, min_width, max_width);

  auto const table = create_random_table({cudf::type_id::STRING}, row_count{num_rows}, profile);
  auto const bytes = table->alloc_size();

  state.set_cuda_stream(nvbench::make_cuda_stream_view(cudf::get_default_stream().get()));
  state.add_global_memory_reads<nvbench::int8_t>(bytes);
  state.add_global_memory_writes<nvbench::int8_t>(bytes);

  auto const mem_stats_logger = cudf::memory_stats_logger();

  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) { cudf::sort(table->view()); });

  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");
}

NVBENCH_BENCH(bench_sort_strings)
  .set_name("sort_strings")
  .add_int64_axis("min_width", {0})
  .add_int64_axis("max_width", {32, 64, 128, 256})
  .add_int64_axis("num_rows", {32768, 262144, 2097152});

// Measures the `sorted_order` fast-path case: a single strings column with no nulls
static void bench_sorted_order_strings(nvbench::state& state)
{
  auto const num_rows  = static_cast<cudf::size_type>(state.get_int64("num_rows"));
  auto const min_width = static_cast<cudf::size_type>(state.get_int64("min_width"));
  auto const max_width = static_cast<cudf::size_type>(state.get_int64("max_width"));

  data_profile const profile =
    data_profile_builder()
      .distribution(cudf::type_id::STRING, distribution_id::NORMAL, min_width, max_width)
      .no_validity();

  auto const table = create_random_table({cudf::type_id::STRING}, row_count{num_rows}, profile);
  auto const bytes = table->alloc_size();

  state.set_cuda_stream(nvbench::make_cuda_stream_view(cudf::get_default_stream().get()));
  state.add_global_memory_reads<nvbench::int8_t>(bytes);
  state.add_global_memory_writes<cudf::size_type>(num_rows);

  auto const mem_stats_logger = cudf::memory_stats_logger();

  state.exec(nvbench::exec_tag::sync,
             [&](nvbench::launch& launch) { cudf::sorted_order(table->view()); });

  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");
}

NVBENCH_BENCH(bench_sorted_order_strings)
  .set_name("sorted_order_strings")
  .add_int64_axis("min_width", {1})
  .add_int64_axis("max_width", {8, 32, 64, 128, 256})
  .add_int64_axis("num_rows", {32768, 262144, 2097152, 16777216});

// Measures the multi-column (lexicographic row comparator) strings path
static void bench_sorted_order_strings_multi(nvbench::state& state)
{
  auto const num_rows  = static_cast<cudf::size_type>(state.get_int64("num_rows"));
  auto const num_cols  = static_cast<cudf::size_type>(state.get_int64("num_cols"));
  auto const max_width = static_cast<cudf::size_type>(state.get_int64("max_width"));

  data_profile const profile =
    data_profile_builder()
      .distribution(cudf::type_id::STRING, distribution_id::NORMAL, 1, max_width)
      .no_validity();

  auto const table = create_random_table(
    cycle_dtypes({cudf::type_id::STRING}, num_cols), row_count{num_rows}, profile);

  state.set_cuda_stream(nvbench::make_cuda_stream_view(cudf::get_default_stream().get()));
  state.add_global_memory_reads<nvbench::int8_t>(table->alloc_size());
  state.add_global_memory_writes<cudf::size_type>(num_rows);

  auto const mem_stats_logger = cudf::memory_stats_logger();

  state.exec(nvbench::exec_tag::sync,
             [&](nvbench::launch& launch) { cudf::sorted_order(table->view()); });

  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");
}

NVBENCH_BENCH(bench_sorted_order_strings_multi)
  .set_name("sorted_order_strings_multi")
  .add_int64_axis("max_width", {8, 32, 64})
  .add_int64_axis("num_cols", {2, 4})
  .add_int64_axis("num_rows", {262144, 2097152});

static void bench_sorted_order_strings_distribution(nvbench::state& state)
{
  auto const num_rows = static_cast<cudf::size_type>(state.get_int64("num_rows"));
  auto const profile  = state.get_string("profile");
  run_sorted_order_benchmark(state, make_distribution_input(num_rows, profile));
}

NVBENCH_BENCH(bench_sorted_order_strings_distribution)
  .set_name("sorted_order_strings_distribution")
  .add_int64_axis("num_rows", {262144, 2097152})
  .add_string_axis(
    "profile",
    {"cardinality_1_width_32", "cardinality_64_width_128", "shared_prefix_64", "variable_128"});

static void bench_sorted_order_strings_cardinality(nvbench::state& state)
{
  auto const num_rows    = static_cast<cudf::size_type>(state.get_int64("num_rows"));
  auto const max_width   = static_cast<cudf::size_type>(state.get_int64("max_width"));
  auto const cardinality = static_cast<cudf::size_type>(state.get_int64("cardinality"));
  run_sorted_order_benchmark(state, make_cardinality_input(num_rows, max_width, cardinality));
}

NVBENCH_BENCH(bench_sorted_order_strings_cardinality)
  .set_name("sorted_order_strings_cardinality")
  .add_int64_axis("num_rows", {262144, 2097152})
  .add_int64_axis("max_width", {32, 128})
  .add_int64_axis("cardinality", {1, 64, 0});

static void bench_sorted_order_strings_prefixes(nvbench::state& state)
{
  auto const num_rows     = static_cast<cudf::size_type>(state.get_int64("num_rows"));
  auto const prefix_width = static_cast<cudf::size_type>(state.get_int64("prefix_width"));
  auto const suffix_width = static_cast<cudf::size_type>(state.get_int64("suffix_width"));
  auto const prefix_cardinality =
    static_cast<cudf::size_type>(state.get_int64("prefix_cardinality"));
  run_sorted_order_benchmark(
    state, make_prefixed_input(num_rows, prefix_width, suffix_width, prefix_cardinality));
}

NVBENCH_BENCH(bench_sorted_order_strings_prefixes)
  .set_name("sorted_order_strings_prefixes")
  .add_int64_axis("num_rows", {262144, 2097152})
  .add_int64_axis("prefix_width", {64})
  .add_int64_axis("suffix_width", {32})
  .add_int64_axis("prefix_cardinality", {1, 64});

static void bench_sorted_order_strings_nulls(nvbench::state& state)
{
  auto const num_rows     = static_cast<cudf::size_type>(state.get_int64("num_rows"));
  auto const profile      = state.get_string("profile");
  auto const null_percent = static_cast<double>(state.get_int64("null_percent"));
  run_sorted_order_benchmark(state, make_nullable_input(num_rows, profile, null_percent / 100.0));
}

NVBENCH_BENCH(bench_sorted_order_strings_nulls)
  .set_name("sorted_order_strings_nulls")
  .add_int64_axis("num_rows", {262144, 2097152})
  .add_string_axis("profile", {"fixed_8", "variable_128"})
  .add_int64_axis("null_percent", {0, 50, 100});

// Equal eight-byte prefixes form runs of controlled sizes; each row's suffix is distinct.
static void bench_sorted_order_strings_runs(nvbench::state& state)
{
  auto const num_rows = static_cast<cudf::size_type>(state.get_int64("num_rows"));
  auto const shared   = static_cast<std::size_t>(state.get_int64("shared_suffix"));
  auto const profile  = state.get_string("run_profile");
  std::vector<std::string> strings;
  strings.reserve(num_rows);
  cudf::size_type group = 0;
  while (static_cast<cudf::size_type>(strings.size()) < num_rows) {
    cudf::size_type length = 1;
    if (profile == "pairs")
      length = 2;
    else if (profile == "tiny32")
      length = 32;
    else if (profile == "block256")
      length = 256;
    else if (profile == "logarithmic")
      length = 1 << (1 + group % 13);
    else if (profile == "hot90")
      length = group == 0 ? num_rows / 10 * 9 : 32;
    else if (profile == "few_giants")
      length = group < 8 ? num_rows / 64 : 32;
    else if (profile == "one_run")
      length = num_rows;
    else
      CUDF_EXPECTS(profile == "singletons", "Unknown run profile");
    length      = std::min(length, num_rows - static_cast<cudf::size_type>(strings.size()));
    auto prefix = std::to_string(group++);
    prefix.insert(0, 8 - prefix.size(), '0');
    prefix += std::string(4 + shared, 'x');
    for (cudf::size_type i = 0; i < length; ++i) {
      auto value = static_cast<std::uint64_t>(strings.size()) * 0x9e3779b97f4a7c15ULL;
      value ^= value >> 31;
      auto suffix = std::to_string(value % 100000000);
      suffix.insert(0, 8 - suffix.size(), '0');
      strings.push_back(prefix + suffix);
    }
  }
  std::shuffle(strings.begin(), strings.end(), std::mt19937{seed});
  auto const input = cudf::test::strings_column_wrapper(strings.begin(), strings.end()).release();
  run_sorted_order_benchmark(state, input);
}

NVBENCH_BENCH(bench_sorted_order_strings_runs)
  .set_name("sorted_order_strings_runs")
  .add_int64_axis("num_rows", {32768, 2097152})
  .add_int64_axis("shared_suffix", {0, 64})
  .add_string_axis(
    "run_profile",
    {"singletons", "pairs", "tiny32", "block256", "logarithmic", "hot90", "few_giants", "one_run"});
