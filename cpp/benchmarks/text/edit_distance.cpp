/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <benchmarks/common/generate_input.hpp>
#include <benchmarks/common/memory_stats.hpp>

#include <cudf/strings/strings_column_view.hpp>

#include <nvtext/edit_distance.hpp>

#include <nvbench/nvbench.cuh>

static void bench_edit_distance_utf8(nvbench::state& state)
{
  auto const num_rows  = static_cast<cudf::size_type>(state.get_int64("num_rows"));
  auto const min_width = static_cast<cudf::size_type>(state.get_int64("min_width"));
  auto const max_width = static_cast<cudf::size_type>(state.get_int64("max_width"));

  data_profile const strings_profile = data_profile_builder().distribution(
    cudf::type_id::STRING, distribution_id::NORMAL, min_width, max_width);
  auto const strings_table = create_random_table(
    {cudf::type_id::STRING, cudf::type_id::STRING}, row_count{num_rows}, strings_profile);
  auto input1 = strings_table->get_column(0);
  auto input2 = strings_table->get_column(1);
  auto sv1    = cudf::strings_column_view(input1.view());
  auto sv2    = cudf::strings_column_view(input2.view());

  state.set_cuda_stream(nvbench::make_cuda_stream_view(cudf::get_default_stream().get()));

  state.add_global_memory_reads<nvbench::int8_t>(input1.alloc_size() + input2.alloc_size());
  // output are integers (one per row)
  state.add_global_memory_writes<nvbench::int32_t>(num_rows);

  auto const mem_stats_logger = cudf::memory_stats_logger();
  state.exec(nvbench::exec_tag::sync,
             [&](nvbench::launch&) { auto result = nvtext::edit_distance(sv1, sv2); });
  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");
}

static void bench_edit_distance_ascii(nvbench::state& state)
{
  auto const num_rows  = static_cast<cudf::size_type>(state.get_int64("num_rows"));
  auto const min_width = static_cast<cudf::size_type>(state.get_int64("min_width"));
  auto const max_width = static_cast<cudf::size_type>(state.get_int64("max_width"));

  data_profile const strings_profile = data_profile_builder().no_validity().distribution(
    cudf::type_id::STRING, distribution_id::NORMAL, min_width, max_width);
  auto const input1 = create_ascii_string_column(strings_profile, num_rows, 1);
  auto const input2 = create_ascii_string_column(strings_profile, num_rows, 2);
  auto sv1          = cudf::strings_column_view(input1->view());
  auto sv2          = cudf::strings_column_view(input2->view());

  state.set_cuda_stream(nvbench::make_cuda_stream_view(cudf::get_default_stream().get()));

  state.add_global_memory_reads<nvbench::int8_t>(input1->alloc_size() + input2->alloc_size());
  // output are integers (one per row)
  state.add_global_memory_writes<nvbench::int32_t>(num_rows);

  auto const mem_stats_logger = cudf::memory_stats_logger();
  state.exec(nvbench::exec_tag::sync,
             [&](nvbench::launch&) { auto result = nvtext::edit_distance(sv1, sv2); });
  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");
}

static void bench_edit_distance(nvbench::state& state)
{
  auto const encode = state.get_string("encode");
  if (encode == "ascii") {
    bench_edit_distance_ascii(state);
  } else {
    bench_edit_distance_utf8(state);
  }
}

NVBENCH_BENCH(bench_edit_distance)
  .set_name("edit_distance")
  .add_int64_axis("min_width", {0})
  .add_int64_axis("max_width", {32, 64, 128, 256, 512})
  .add_int64_axis("num_rows", {262144, 524288, 1048576})
  .add_string_axis("encode", {"utf8", "ascii"});
