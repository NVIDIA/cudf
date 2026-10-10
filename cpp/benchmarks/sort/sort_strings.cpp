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
#include <string_view>
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

void run_sort_benchmark(nvbench::state& state, std::unique_ptr<cudf::column> const& input)
{
  auto const bytes = input->alloc_size();
  state.set_cuda_stream(nvbench::make_cuda_stream_view(cudf::get_default_stream().get()));
  state.add_global_memory_reads<nvbench::int8_t>(bytes);
  state.add_global_memory_writes<nvbench::int8_t>(bytes);

  auto const mem_stats_logger = cudf::memory_stats_logger();
  state.exec(nvbench::exec_tag::sync,
             [&](nvbench::launch& launch) { cudf::sort(cudf::table_view{{input->view()}}); });
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

std::unique_ptr<cudf::column> make_workload_input(cudf::size_type num_rows,
                                                  cudf::size_type min_width,
                                                  cudf::size_type max_width,
                                                  std::string const& workload)
{
  if (workload == "duplicates") {
    data_profile const profile =
      data_profile_builder().no_validity().cardinality(64).avg_run_length(1).distribution(
        cudf::type_id::STRING, distribution_id::NORMAL, min_width, max_width);
    return create_random_column(cudf::type_id::STRING, row_count{num_rows}, profile, seed);
  }
  if (workload == "shared_prefix") {
    constexpr cudf::size_type suffix_width = 8;
    return make_prefixed_input(num_rows, max_width - suffix_width, suffix_width, 1);
  }
  if (workload == "normal") {
    data_profile const profile = data_profile_builder().no_validity().distribution(
      cudf::type_id::STRING, distribution_id::NORMAL, min_width, max_width);
    return create_random_column(cudf::type_id::STRING, row_count{num_rows}, profile, seed);
  }
  if (workload == "variable") {
    data_profile const profile =
      data_profile_builder().no_validity().cardinality(0).avg_run_length(1).distribution(
        cudf::type_id::STRING, distribution_id::UNIFORM, min_width, max_width);
    return create_random_column(cudf::type_id::STRING, row_count{num_rows}, profile, seed);
  }
  CUDF_FAIL("Unknown string workload: " + workload);
}

std::unique_ptr<cudf::column> make_sensitivity_input(cudf::size_type num_rows,
                                                     std::string const& profile_name)
{
  constexpr cudf::size_type width = 128;
  if (profile_name.starts_with("duplicates_")) {
    auto cardinality = cudf::size_type{0};
    if (profile_name == "duplicates_1") {
      cardinality = 1;
    } else if (profile_name == "duplicates_64") {
      cardinality = 64;
    } else if (profile_name == "duplicates_4096") {
      cardinality = 4096;
    } else if (profile_name != "duplicates_unique") {
      CUDF_FAIL("Unknown string sensitivity profile: " + profile_name);
    }
    data_profile const profile =
      data_profile_builder()
        .no_validity()
        .cardinality(cardinality)
        .avg_run_length(1)
        .distribution(cudf::type_id::STRING, distribution_id::UNIFORM, 0, width);
    return create_random_column(cudf::type_id::STRING, row_count{num_rows}, profile, seed);
  }

  auto prefix_percent = cudf::size_type{-1};
  if (profile_name == "prefix_0") {
    prefix_percent = 0;
  } else if (profile_name == "prefix_50") {
    prefix_percent = 50;
  } else if (profile_name == "prefix_75") {
    prefix_percent = 75;
  } else if (profile_name == "prefix_90") {
    prefix_percent = 90;
  } else {
    CUDF_FAIL("Unknown string sensitivity profile: " + profile_name);
  }
  auto const prefix_width = width * prefix_percent / 100;
  return make_prefixed_input(num_rows, prefix_width, width - prefix_width, 1);
}

std::string fixed_width_token(std::uint64_t value, cudf::size_type width)
{
  constexpr auto alphabet =
    std::string_view{"0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"};
  auto result = std::string(static_cast<std::size_t>(width), alphabet.front());
  for (auto position = width; position > 0; --position) {
    result[static_cast<std::size_t>(position - 1)] = alphabet[value % alphabet.size()];
    value /= alphabet.size();
  }
  return result;
}

std::unique_ptr<cudf::column> make_diagnostic_input(cudf::size_type num_rows,
                                                    std::string const& profile_name)
{
  auto strings = std::vector<std::string>(static_cast<std::size_t>(num_rows));
  if (profile_name.starts_with("finish_")) {
    auto const unresolved_rows = static_cast<cudf::size_type>(std::stoll(profile_name.substr(7)));
    CUDF_EXPECTS(unresolved_rows <= num_rows, "Unresolved run exceeds diagnostic row count");
    for (cudf::size_type row = 0; row < num_rows; ++row) {
      strings[static_cast<std::size_t>(row)] = row < unresolved_rows
                                                 ? std::string(24, 'z') + fixed_width_token(row, 8)
                                                 : fixed_width_token(row, 6) + std::string(26, 'x');
    }
  } else if (profile_name.starts_with("active_")) {
    auto const tenths_percent = profile_name == "active_50"    ? 500
                                : profile_name == "active_10"  ? 100
                                : profile_name == "active_1"   ? 10
                                : profile_name == "active_0_1" ? 1
                                                               : -1;
    CUDF_EXPECTS(tenths_percent >= 0, "Unknown active-coverage diagnostic profile");
    auto const active_rows =
      static_cast<cudf::size_type>(static_cast<std::int64_t>(num_rows) * tenths_percent / 1000);
    for (cudf::size_type row = 0; row < num_rows; ++row) {
      strings[static_cast<std::size_t>(row)] = row < active_rows
                                                 ? std::string{"zzzzzz"} + fixed_width_token(row, 8)
                                                 : fixed_width_token(row, 6) + std::string(8, 'x');
    }
  } else if (profile_name.starts_with("duplicates_")) {
    auto const width = static_cast<cudf::size_type>(std::stoll(profile_name.substr(11)));
    for (cudf::size_type row = 0; row < num_rows; ++row) {
      auto value = fixed_width_token(row % 64, std::min<cudf::size_type>(width, 6));
      value.append(static_cast<std::size_t>(width - value.size()), 'd');
      strings[static_cast<std::size_t>(row)] = std::move(value);
    }
  } else if (profile_name == "zero_collision_pass1" || profile_name == "zero_collision_pass2") {
    auto const preceding_prefix =
      profile_name == "zero_collision_pass1" ? std::string{} : std::string(8, 'q');
    auto const short_value = preceding_prefix + 'a';
    auto const long_value  = short_value + std::string(7, '\0');
    for (cudf::size_type row = 0; row < num_rows; ++row) {
      strings[static_cast<std::size_t>(row)] = row % 2 == 0 ? long_value : short_value;
    }
  } else if (profile_name == "rle_misaligned") {
    auto const stride = std::max<cudf::size_type>(1, (num_rows + 4095) / 4096);
    for (cudf::size_type row = 0; row < num_rows; ++row) {
      auto const block  = row / stride;
      auto const offset = row % stride;
      auto value        = std::string(40, 'q') + fixed_width_token(block, 6);
      if (offset == 0) {
        value += "a" + fixed_width_token(row, 6);
      } else if (offset <= std::min<cudf::size_type>(33, stride - 1)) {
        // Placing every eligible run immediately after a global stride boundary makes the old
        // position-based sampler systematically miss it.
        value += "b-repeat";
      } else {
        value += "c" + fixed_width_token(offset, 6);
      }
      strings[static_cast<std::size_t>(row)] = std::move(value);
    }
  } else {
    CUDF_FAIL("Unknown segmented string diagnostic profile: " + profile_name);
  }
  return cudf::test::strings_column_wrapper(strings.begin(), strings.end()).release();
}

std::unique_ptr<cudf::column> make_source_parity_input(cudf::size_type num_rows,
                                                       std::string const& profile_name)
{
  auto strings                 = std::vector<std::string>(static_cast<std::size_t>(num_rows));
  auto const raw_duplicates    = profile_name == "raw_duplicates_40";
  auto const raw_shared_prefix = profile_name == "raw_shared_prefix_24_width_40";
  if (raw_duplicates || raw_shared_prefix || profile_name == "raw_unique_40") {
    constexpr cudf::size_type width = 40;
    std::mt19937 random(seed);
    auto dictionary = std::vector<std::string>(64, std::string(width, '\0'));
    // Matching the harness's RNG consumption makes these inputs byte-identical to its corpora,
    // rather than replacing prefix collisions with a different nominally equivalent distribution.
    for (auto& value : dictionary) {
      for (auto& byte : value) {
        byte = static_cast<char>(1 + random() % 255);
      }
    }
    for (cudf::size_type row = 0; row < num_rows; ++row) {
      auto& value = strings[static_cast<std::size_t>(row)];
      if (raw_duplicates) {
        value = dictionary[static_cast<std::size_t>(row) % dictionary.size()];
      } else {
        value = std::string(width, 'p');
        for (cudf::size_type byte = raw_shared_prefix ? 24 : 0; byte < width; ++byte) {
          value[static_cast<std::size_t>(byte)] = static_cast<char>(1 + random() % 255);
        }
      }
    }
    return cudf::test::strings_column_wrapper(strings.begin(), strings.end()).release();
  }
  constexpr cudf::size_type width = 32;
  for (cudf::size_type row = 0; row < num_rows; ++row) {
    auto const token = fixed_width_token(static_cast<std::uint64_t>(row), width);
    if (profile_name == "unique_32") {
      strings[static_cast<std::size_t>(row)] = token;
    } else if (profile_name == "duplicates_32") {
      strings[static_cast<std::size_t>(row)] =
        fixed_width_token(static_cast<std::uint64_t>(row % 64), width);
    } else if (profile_name == "shared_prefix_24") {
      strings[static_cast<std::size_t>(row)] = std::string(24, 'p') + token.substr(24);
    } else {
      CUDF_FAIL("Unknown source-parity profile: " + profile_name);
    }
  }
  return cudf::test::strings_column_wrapper(strings.begin(), strings.end()).release();
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

static void bench_sorted_order_strings_workload(nvbench::state& state)
{
  auto const num_rows  = static_cast<cudf::size_type>(state.get_int64("num_rows"));
  auto const min_width = static_cast<cudf::size_type>(state.get_int64("min_width"));
  auto const max_width = static_cast<cudf::size_type>(state.get_int64("max_width"));
  auto const workload  = state.get_string("workload");
  run_sorted_order_benchmark(state, make_workload_input(num_rows, min_width, max_width, workload));
}

NVBENCH_BENCH(bench_sorted_order_strings_workload)
  .set_name("sorted_order_strings_workload")
  .add_int64_axis("min_width", {0})
  .add_int64_axis("max_width", {32, 64, 128, 256})
  .add_int64_axis("num_rows", {32768, 262144, 2097152})
  .add_string_axis("workload", {"normal", "duplicates", "shared_prefix", "variable"});

static void bench_sort_strings_workload(nvbench::state& state)
{
  auto const num_rows  = static_cast<cudf::size_type>(state.get_int64("num_rows"));
  auto const min_width = static_cast<cudf::size_type>(state.get_int64("min_width"));
  auto const max_width = static_cast<cudf::size_type>(state.get_int64("max_width"));
  auto const workload  = state.get_string("workload");
  run_sort_benchmark(state, make_workload_input(num_rows, min_width, max_width, workload));
}

NVBENCH_BENCH(bench_sort_strings_workload)
  .set_name("sort_strings_workload")
  .add_int64_axis("min_width", {0})
  .add_int64_axis("max_width", {32, 64, 128, 256})
  .add_int64_axis("num_rows", {32768, 262144, 2097152})
  .add_string_axis("workload", {"normal", "duplicates", "shared_prefix", "variable"});

static void bench_sorted_order_strings_sensitivity(nvbench::state& state)
{
  auto const num_rows = static_cast<cudf::size_type>(state.get_int64("num_rows"));
  auto const profile  = state.get_string("profile");
  run_sorted_order_benchmark(state, make_sensitivity_input(num_rows, profile));
}

NVBENCH_BENCH(bench_sorted_order_strings_sensitivity)
  .set_name("sorted_order_strings_sensitivity")
  .add_int64_axis("num_rows", {262144, 2097152})
  .add_string_axis("profile",
                   {"duplicates_1",
                    "duplicates_64",
                    "duplicates_4096",
                    "duplicates_unique",
                    "prefix_0",
                    "prefix_50",
                    "prefix_75",
                    "prefix_90"});

static void bench_sorted_order_strings_segmented_diagnostics(nvbench::state& state)
{
  auto const num_rows = static_cast<cudf::size_type>(state.get_int64("num_rows"));
  auto const profile  = state.get_string("profile");
  run_sorted_order_benchmark(state, make_diagnostic_input(num_rows, profile));
}

NVBENCH_BENCH(bench_sorted_order_strings_segmented_diagnostics)
  .set_name("sorted_order_strings_segmented_diagnostics")
  .add_int64_axis("num_rows", {262144, 2097152})
  .add_string_axis("profile",
                   {"finish_257",
                    "finish_1024",
                    "finish_4096",
                    "active_50",
                    "active_10",
                    "active_1",
                    "active_0_1",
                    "duplicates_6",
                    "duplicates_8",
                    "duplicates_12",
                    "duplicates_16",
                    "duplicates_24",
                    "duplicates_32",
                    "zero_collision_pass1",
                    "zero_collision_pass2",
                    "rle_misaligned"});

static void bench_sorted_order_strings_source_parity(nvbench::state& state)
{
  auto const num_rows = static_cast<cudf::size_type>(state.get_int64("num_rows"));
  auto const profile  = state.get_string("profile");
  run_sorted_order_benchmark(state, make_source_parity_input(num_rows, profile));
}

NVBENCH_BENCH(bench_sorted_order_strings_source_parity)
  .set_name("sorted_order_strings_source_parity")
  .add_int64_axis("num_rows", {262144, 2097152})
  .add_string_axis("profile",
                   {"unique_32",
                    "duplicates_32",
                    "shared_prefix_24",
                    "raw_unique_40",
                    "raw_duplicates_40",
                    "raw_shared_prefix_24_width_40"});
