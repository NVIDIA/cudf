/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <benchmarks/common/generate_input.hpp>
#include <benchmarks/common/memory_stats.hpp>

#include <cudf_test/column_wrapper.hpp>

#include <cudf/strings/extract.hpp>
#include <cudf/strings/regex/regex_program.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <nvbench/nvbench.cuh>

#include <random>

static void bench_extract(nvbench::state& state)
{
  auto const num_rows  = static_cast<cudf::size_type>(state.get_int64("num_rows"));
  auto const row_width = static_cast<cudf::size_type>(state.get_int64("row_width"));

  auto groups = static_cast<cudf::size_type>(state.get_int64("groups"));

  std::default_random_engine generator;
  std::uniform_int_distribution<int> words_dist(0, 999);
  std::vector<std::string> samples(100);  // 100 unique rows of data to reuse
  std::generate(samples.begin(), samples.end(), [&]() {
    std::string row;  // build a row of random tokens
    while (static_cast<cudf::size_type>(row.size()) < row_width) {
      row += std::to_string(words_dist(generator)) + " ";
    }
    return row;
  });

  std::string pattern{""};
  while (groups--) {
    pattern += "(\\d+) ";
  }

  cudf::test::strings_column_wrapper samples_column(samples.begin(), samples.end());
  data_profile const profile = data_profile_builder().no_validity().distribution(
    cudf::type_to_id<cudf::size_type>(), distribution_id::UNIFORM, 0ul, samples.size() - 1);
  auto map =
    create_random_column(cudf::type_to_id<cudf::size_type>(), row_count{num_rows}, profile);
  auto input = cudf::gather(
    cudf::table_view{{samples_column}}, map->view(), cudf::out_of_bounds_policy::DONT_CHECK);
  cudf::strings_column_view strings_view(input->get_column(0).view());
  auto prog = cudf::strings::regex_program::create(pattern);

  state.set_cuda_stream(nvbench::make_cuda_stream_view(cudf::get_default_stream().value()));
  // gather some throughput statistics as well
  auto data_size = input->alloc_size();
  state.add_global_memory_reads<nvbench::int8_t>(data_size);   // all bytes are read;
  state.add_global_memory_writes<nvbench::int8_t>(data_size);  // all bytes are written

  auto const mem_stats_logger = cudf::memory_stats_logger();
  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
    auto result = cudf::strings::extract(strings_view, *prog);
  });
  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");
}

NVBENCH_BENCH(bench_extract)
  .set_name("extract")
  .add_int64_axis("row_width", {32, 64, 128, 256})
  .add_int64_axis("num_rows", {32768, 262144, 2097152})
  .add_int64_axis("groups", {1, 2, 4});
