/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <benchmarks/common/generate_input.hpp>
#include <benchmarks/common/memory_stats.hpp>

#include <cudf_test/column_wrapper.hpp>

#include <cudf/copying.hpp>
#include <cudf/experimental/strings/regex.hpp>
#include <cudf/strings/contains.hpp>
#include <cudf/strings/regex/regex_program.hpp>
#include <cudf/strings/replace_re.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <nvbench/nvbench.cuh>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace {

struct count_case {
  std::string_view name;
  std::string_view pattern;
};

struct replace_case {
  std::string_view name;
  std::string_view pattern;
  std::string_view replacement;
};

// The canonical regex-redux expressions from the Benchmarks Game specification.
constexpr std::array count_cases{
  count_case{"variant-0", R"(agggtaaa|tttaccct)"},
  count_case{"variant-1", R"([cgt]gggtaaa|tttaccc[acg])"},
  count_case{"variant-2", R"(a[act]ggtaaa|tttacc[agt]t)"},
  count_case{"variant-3", R"(ag[act]gtaaa|tttac[agt]ct)"},
  count_case{"variant-4", R"(agg[act]taaa|ttta[agt]cct)"},
  count_case{"variant-5", R"(aggg[acg]aaa|ttt[cgt]ccct)"},
  count_case{"variant-6", R"(agggt[cgt]aa|tt[acg]accct)"},
  count_case{"variant-7", R"(agggta[cgt]a|t[acg]taccct)"},
  count_case{"variant-8", R"(agggtaa[cgt]|[acg]ttaccct)"},
};

constexpr std::array replace_cases{
  replace_case{"iub-4", R"(tHa[Nt])", "<4>"},
  replace_case{"iub-3", R"(aND|caN|Ha[DS]|WaS)", "<3>"},
  replace_case{"iub-2", R"(a[NSt]|BY)", "<2>"},
  replace_case{"tags", R"(<[^>]*>)", "|"},
  replace_case{"bars", R"(\|[^|][^|]*\|)", "-"},
};

template <typename Case, std::size_t N>
std::vector<std::string> case_names(std::array<Case, N> const& cases)
{
  std::vector<std::string> result;
  result.reserve(cases.size());
  std::transform(cases.begin(), cases.end(), std::back_inserter(result), [](auto& item) {
    return std::string{item.name};
  });
  return result;
}

template <typename Case, std::size_t N>
Case const* find_case(std::array<Case, N> const& cases, std::string_view name)
{
  auto found =
    std::find_if(cases.begin(), cases.end(), [name](auto& item) { return item.name == name; });
  return found == cases.end() ? nullptr : &*found;
}

std::uint64_t random_value(std::uint64_t& state)
{
  state ^= state >> 12U;
  state ^= state << 25U;
  state ^= state >> 27U;
  return state * 0x2545f4914f6cdd1dULL;
}

void inject(std::string& row, std::string_view value, std::size_t position)
{
  if (value.size() <= row.size()) {
    row.replace(std::min(position, row.size() - value.size()), value.size(), value);
  }
}

std::string make_redux_row(cudf::size_type width, std::size_t sample)
{
  constexpr auto alphabet = std::string_view{"acgtBDHKMNRSVWY"};
  auto random_state       = std::uint64_t{0xa0761d6478bd642fULL + sample};
  std::string row(static_cast<std::size_t>(width), 'a');
  for (auto& character : row) {
    character = alphabet[random_value(random_state) % alphabet.size()];
  }

  if (sample % 2U == 0U) inject(row, "agggtaaa", row.size() / 8U);
  if (sample % 3U == 0U) inject(row, "tttaccct", row.size() / 4U);
  if (sample % 4U == 0U) inject(row, "cgggtaaa", row.size() / 3U);
  if (sample % 5U == 0U) inject(row, "agcggtaaa", row.size() / 2U);
  inject(row, "tHaNtaNDcaNHaDWaSaNStBY<tag>ACGT|abc|", row.size() * 3U / 5U);
  return row;
}

std::unique_ptr<cudf::table> make_input(cudf::size_type num_rows, cudf::size_type row_width)
{
  constexpr std::size_t sample_count = 64;
  std::vector<std::string> samples;
  samples.reserve(sample_count);
  for (std::size_t sample = 0; sample < sample_count; ++sample) {
    samples.push_back(make_redux_row(row_width, sample));
  }

  cudf::test::strings_column_wrapper samples_column(samples.begin(), samples.end());
  auto profile = data_profile_builder().no_validity().distribution(
    cudf::type_to_id<cudf::size_type>(), distribution_id::UNIFORM, 0ul, sample_count - 1);
  auto map =
    create_random_column(cudf::type_to_id<cudf::size_type>(), row_count{num_rows}, profile);
  return cudf::gather(
    cudf::table_view{{samples_column}}, map->view(), cudf::out_of_bounds_policy::DONT_CHECK);
}

void bench_regex_redux_count(nvbench::state& state)
{
  auto definition = find_case(count_cases, state.get_string("case"));
  if (definition == nullptr) {
    state.skip("unknown regex-redux count case");
    return;
  }

  auto num_rows  = static_cast<cudf::size_type>(state.get_int64("num_rows"));
  auto row_width = static_cast<cudf::size_type>(state.get_int64("row_width"));
  auto backend   = state.get_string("backend");
  auto input     = make_input(num_rows, row_width);
  auto strings   = cudf::strings_column_view(input->get_column(0).view());
  auto interpreter =
    backend == "interpreter" ? cudf::strings::regex_program::create(definition->pattern) : nullptr;
  auto jit = backend == "jit" ? cudf::experimental::regex_jit_program::create(
                                  definition->pattern, cudf::experimental::regex_operation::COUNT)
                              : nullptr;

  state.set_cuda_stream(nvbench::make_cuda_stream_view(cudf::get_default_stream().get()));
  state.add_global_memory_reads<nvbench::int8_t>(input->alloc_size());
  state.add_global_memory_writes<nvbench::int32_t>(strings.size());

  auto mem_stats_logger = cudf::memory_stats_logger();
  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch&) {
    if (backend == "jit") {
      static_cast<void>(cudf::experimental::count_re(strings, *jit));
    } else {
      static_cast<void>(cudf::strings::count_re(strings, *interpreter));
    }
  });
  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");
}

void bench_regex_redux_replace(nvbench::state& state)
{
  auto definition = find_case(replace_cases, state.get_string("case"));
  if (definition == nullptr) {
    state.skip("unknown regex-redux replacement case");
    return;
  }

  auto num_rows    = static_cast<cudf::size_type>(state.get_int64("num_rows"));
  auto row_width   = static_cast<cudf::size_type>(state.get_int64("row_width"));
  auto backend     = state.get_string("backend");
  auto input       = make_input(num_rows, row_width);
  auto strings     = cudf::strings_column_view(input->get_column(0).view());
  auto replacement = cudf::string_scalar(std::string{definition->replacement});
  auto interpreter =
    backend == "interpreter" ? cudf::strings::regex_program::create(definition->pattern) : nullptr;
  auto options = cudf::experimental::regex_jit_program_options{
    .replacement = std::string{definition->replacement}};
  auto jit = backend == "jit"
               ? cudf::experimental::regex_jit_program::create(
                   definition->pattern, cudf::experimental::regex_operation::REPLACE, options)
               : nullptr;

  state.set_cuda_stream(nvbench::make_cuda_stream_view(cudf::get_default_stream().get()));
  state.add_global_memory_reads<nvbench::int8_t>(input->alloc_size());
  state.add_global_memory_writes<nvbench::int8_t>(input->alloc_size());

  auto mem_stats_logger = cudf::memory_stats_logger();
  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch&) {
    if (backend == "jit") {
      static_cast<void>(cudf::experimental::replace_re(strings, *jit, replacement));
    } else {
      static_cast<void>(cudf::strings::replace_re(strings, *interpreter, replacement));
    }
  });
  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");
}

void bench_regex_redux_pipeline(nvbench::state& state)
{
  auto num_rows  = static_cast<cudf::size_type>(state.get_int64("num_rows"));
  auto row_width = static_cast<cudf::size_type>(state.get_int64("row_width"));
  auto backend   = state.get_string("backend");
  auto input     = make_input(num_rows, row_width);

  std::vector<std::unique_ptr<cudf::strings::regex_program>> interpreter;
  std::vector<std::unique_ptr<cudf::experimental::regex_jit_program>> jit;
  std::vector<std::unique_ptr<cudf::string_scalar>> replacements;
  interpreter.reserve(replace_cases.size());
  jit.reserve(replace_cases.size());
  replacements.reserve(replace_cases.size());
  for (auto& definition : replace_cases) {
    replacements.push_back(
      std::make_unique<cudf::string_scalar>(std::string{definition.replacement}));
    if (backend == "jit") {
      auto options = cudf::experimental::regex_jit_program_options{
        .replacement = std::string{definition.replacement}};
      jit.push_back(cudf::experimental::regex_jit_program::create(
        definition.pattern, cudf::experimental::regex_operation::REPLACE, options));
    } else {
      interpreter.push_back(cudf::strings::regex_program::create(definition.pattern));
    }
  }

  state.set_cuda_stream(nvbench::make_cuda_stream_view(cudf::get_default_stream().get()));
  state.add_global_memory_reads<nvbench::int8_t>(input->alloc_size());
  state.add_global_memory_writes<nvbench::int8_t>(input->alloc_size());

  auto mem_stats_logger = cudf::memory_stats_logger();
  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch&) {
    std::unique_ptr<cudf::column> output;
    for (std::size_t index = 0; index < replace_cases.size(); ++index) {
      auto strings =
        cudf::strings_column_view{output == nullptr ? input->get_column(0).view() : output->view()};
      auto next = backend == "jit"
                    ? cudf::experimental::replace_re(strings, *jit[index], *replacements[index])
                    : cudf::strings::replace_re(strings, *interpreter[index], *replacements[index]);
      output    = std::move(next);
    }
  });
  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");
}

}  // namespace

NVBENCH_BENCH(bench_regex_redux_count)
  .set_name("regex_redux_count")
  .add_int64_axis("row_width", {128, 512})
  .add_int64_axis("num_rows", {262144})
  .add_string_axis("case", case_names(count_cases))
  .add_string_axis("backend", {"interpreter", "jit"});

NVBENCH_BENCH(bench_regex_redux_replace)
  .set_name("regex_redux_replace")
  .add_int64_axis("row_width", {128, 512})
  .add_int64_axis("num_rows", {262144})
  .add_string_axis("case", case_names(replace_cases))
  .add_string_axis("backend", {"interpreter", "jit"});

NVBENCH_BENCH(bench_regex_redux_pipeline)
  .set_name("regex_redux_pipeline")
  .add_int64_axis("row_width", {128, 512})
  .add_int64_axis("num_rows", {262144})
  .add_string_axis("backend", {"interpreter", "jit"});
