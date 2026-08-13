/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <benchmarks/common/generate_input.hpp>
#include <benchmarks/common/memory_stats.hpp>

#include <cudf/filling.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/sorting.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <nvbench/nvbench.cuh>

void nvbench_segmented_sort(nvbench::state& state)
{
  auto const stable     = static_cast<bool>(state.get_int64("stable"));
  auto const dtype      = cudf::type_to_id<int32_t>();
  auto const size_bytes = static_cast<size_t>(state.get_int64("size_bytes"));
  auto const null_freq  = state.get_float64("null_frequency");
  auto const row_width  = static_cast<cudf::size_type>(state.get_int64("row_width"));

  data_profile const table_profile =
    data_profile_builder().null_probability(null_freq).distribution(
      dtype, distribution_id::UNIFORM, 0, 10);
  auto const input =
    create_random_table({cudf::type_id::INT32}, table_size_bytes{size_bytes}, table_profile);
  auto const rows = input->num_rows();

  auto const segments = cudf::sequence((rows / row_width) + 1,
                                       cudf::numeric_scalar<int32_t>(0),
                                       cudf::numeric_scalar<int32_t>(row_width));

  state.set_cuda_stream(nvbench::make_cuda_stream_view(cudf::get_default_stream().value()));
  state.add_element_count(size_bytes, "bytes");
  state.add_global_memory_reads<nvbench::int32_t>(rows * row_width);
  state.add_global_memory_writes<nvbench::int32_t>(rows);
  auto const mem_stats_logger = cudf::memory_stats_logger();

  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
    if (stable)
      cudf::stable_segmented_sorted_order(*input, *segments);
    else
      cudf::segmented_sorted_order(*input, *segments);
  });

  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");
}

NVBENCH_BENCH(nvbench_segmented_sort)
  .set_name("segmented_sort")
  .add_int64_axis("stable", {0, 1})
  .add_int64_power_of_two_axis("size_bytes", {16, 18, 20, 22, 24, 28})
  .add_float64_axis("null_frequency", {0, 0.1})
  .add_int64_axis("row_width", {16, 128, 1024});
