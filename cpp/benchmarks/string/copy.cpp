/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <benchmarks/common/generate_input.hpp>
#include <benchmarks/common/memory_stats.hpp>

#include <cudf/copying.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <nvbench/nvbench.cuh>

static void bench_copy(nvbench::state& state)
{
  auto const num_rows  = static_cast<cudf::size_type>(state.get_int64("num_rows"));
  auto const min_width = static_cast<cudf::size_type>(state.get_int64("min_width"));
  auto const max_width = static_cast<cudf::size_type>(state.get_int64("max_width"));
  auto const api       = state.get_string("api");

  data_profile const table_profile = data_profile_builder().distribution(
    cudf::type_id::STRING, distribution_id::NORMAL, min_width, max_width);
  auto const source =
    create_random_table({cudf::type_id::STRING}, row_count{num_rows}, table_profile);

  data_profile const map_profile = data_profile_builder().no_validity().distribution(
    cudf::type_to_id<cudf::size_type>(), distribution_id::UNIFORM, 0, num_rows);
  auto const map_table =
    create_random_table({cudf::type_to_id<cudf::size_type>()}, row_count{num_rows}, map_profile);
  auto const map_view = map_table->view().column(0);

  auto stream = cudf::get_default_stream();
  state.set_cuda_stream(nvbench::make_cuda_stream_view(stream.value()));

  if (api == "gather") {
    auto result =
      cudf::gather(source->view(), map_view, cudf::out_of_bounds_policy::NULLIFY, stream);
    auto data_size = result->alloc_size();
    state.add_global_memory_reads<nvbench::int8_t>(data_size +
                                                   (map_view.size() * sizeof(cudf::size_type)));
    state.add_global_memory_writes<nvbench::int8_t>(data_size);
    auto const mem_stats_logger = cudf::memory_stats_logger();

    state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
      cudf::gather(source->view(), map_view, cudf::out_of_bounds_policy::NULLIFY, stream);
    });
    state.add_buffer_size(
      mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");
  } else if (api == "scatter") {
    auto const target =
      create_random_table({cudf::type_id::STRING}, row_count{num_rows}, table_profile);
    auto result    = cudf::scatter(source->view(), map_view, target->view(), stream);
    auto data_size = result->alloc_size();
    state.add_global_memory_reads<nvbench::int8_t>(data_size +
                                                   (map_view.size() * sizeof(cudf::size_type)));
    state.add_global_memory_writes<nvbench::int8_t>(data_size);
    auto const mem_stats_logger = cudf::memory_stats_logger();
    state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
      cudf::scatter(source->view(), map_view, target->view(), stream);
    });
    state.add_buffer_size(
      mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");
  }
}

NVBENCH_BENCH(bench_copy)
  .set_name("copy")
  .add_int64_axis("min_width", {0})
  .add_int64_axis("max_width", {32, 64, 128, 256})
  .add_int64_axis("num_rows",
                  {131072, 262144, 524288, 1048576, 2097152, 4194304, 8388608, 16777216, 33554432})
  .add_string_axis("api", {"gather", "scatter"});
