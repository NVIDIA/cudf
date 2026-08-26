/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <benchmarks/common/generate_input.hpp>
#include <benchmarks/common/memory_stats.hpp>

#include <cudf/strings/strings_column_view.hpp>

#include <nvtext/deduplicate.hpp>

#include <nvbench/nvbench.cuh>

static void bench_suffix_array(nvbench::state& state)
{
  auto const num_rows  = static_cast<cudf::size_type>(state.get_int64("num_rows"));
  auto const min_width = static_cast<cudf::size_type>(state.get_int64("min_width"));
  auto const max_width = static_cast<cudf::size_type>(state.get_int64("max_width"));
  auto const dup_width = static_cast<cudf::size_type>(state.get_int64("dup_width"));

  data_profile const strings_profile = data_profile_builder().distribution(
    cudf::type_id::STRING, distribution_id::NORMAL, min_width, max_width);
  auto const strings_table =
    create_random_table({cudf::type_id::STRING}, row_count{num_rows}, strings_profile);
  cudf::strings_column_view input(strings_table->view().column(0));

  state.set_cuda_stream(nvbench::make_cuda_stream_view(cudf::get_default_stream().value()));

  auto chars_size = input.chars_size(cudf::get_default_stream());
  state.add_global_memory_reads<nvbench::int8_t>(chars_size);
  state.add_global_memory_writes<nvbench::int32_t>(num_rows);
  auto const mem_stats_logger = cudf::memory_stats_logger();

  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
    auto result = nvtext::build_suffix_array(input, dup_width);
  });

  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");
}

static void bench_resolve_duplicates(nvbench::state& state)
{
  auto const num_rows  = static_cast<cudf::size_type>(state.get_int64("num_rows"));
  auto const min_width = static_cast<cudf::size_type>(state.get_int64("min_width"));
  auto const max_width = static_cast<cudf::size_type>(state.get_int64("max_width"));
  auto const dup_width = static_cast<cudf::size_type>(state.get_int64("dup_width"));

  data_profile const strings_profile = data_profile_builder().distribution(
    cudf::type_id::STRING, distribution_id::NORMAL, min_width, max_width);
  auto const strings_table = create_random_table(
    {cudf::type_id::STRING, cudf::type_id::STRING}, row_count{num_rows}, strings_profile);
  cudf::strings_column_view input1(strings_table->view().column(0));
  cudf::strings_column_view input2(strings_table->view().column(1));

  auto stream = cudf::get_default_stream();
  state.set_cuda_stream(nvbench::make_cuda_stream_view(stream.value()));

  auto sa1 = nvtext::build_suffix_array(input1, 0);
  auto sa2 = nvtext::build_suffix_array(input2, 0);

  auto chars_size = input1.chars_size(stream) + input2.chars_size(stream);
  state.add_global_memory_reads<nvbench::int8_t>(chars_size);

  auto const mem_stats_logger = cudf::memory_stats_logger();

  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
    auto result = nvtext::resolve_duplicates_pair(input1, *sa1, input2, *sa2, dup_width);
  });

  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");
}

NVBENCH_BENCH(bench_suffix_array)
  .set_name("suffix_array")
  .add_int64_axis("dup_width", {50})
  .add_int64_axis("min_width", {0})
  .add_int64_axis("max_width", {32, 64, 128, 256})
  .add_int64_axis("num_rows", {32768, 262144});

NVBENCH_BENCH(bench_resolve_duplicates)
  .set_name("resolve_duplicates")
  .add_int64_axis("dup_width", {50})
  .add_int64_axis("min_width", {0})
  .add_int64_axis("max_width", {32, 64, 128, 256})
  .add_int64_axis("num_rows", {32768, 262144});
