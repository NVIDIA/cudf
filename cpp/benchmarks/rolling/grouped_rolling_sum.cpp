/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <benchmarks/common/generate_input.hpp>
#include <benchmarks/common/memory_stats.hpp>

#include <cudf/aggregation.hpp>
#include <cudf/rolling.hpp>
#include <cudf/sorting.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <nvbench/nvbench.cuh>

template <typename Type>
void bench_row_grouped_rolling_sum(nvbench::state& state, nvbench::type_list<Type>)
{
  auto const num_rows       = static_cast<cudf::size_type>(state.get_int64("num_rows"));
  auto const cardinality    = static_cast<cudf::size_type>(state.get_int64("cardinality"));
  auto const preceding_size = static_cast<cudf::size_type>(state.get_int64("preceding_size"));
  auto const following_size = static_cast<cudf::size_type>(state.get_int64("following_size"));
  auto const min_periods    = static_cast<cudf::size_type>(state.get_int64("min_periods"));

  auto const keys = [&] {
    data_profile const profile =
      data_profile_builder()
        .cardinality(cardinality)
        .no_validity()
        .distribution(cudf::type_to_id<int32_t>(), distribution_id::UNIFORM, 0, num_rows);
    auto keys = create_random_column(cudf::type_to_id<int32_t>(), row_count{num_rows}, profile);
    return cudf::sort(cudf::table_view{{keys->view()}});
  }();
  data_profile const profile = data_profile_builder().cardinality(0).no_validity().distribution(
    cudf::type_to_id<Type>(), distribution_id::UNIFORM, 0, 100);
  auto vals = create_random_column(cudf::type_to_id<Type>(), row_count{num_rows}, profile);

  auto req = cudf::make_sum_aggregation<cudf::rolling_aggregation>();

  auto const mem_stats_logger = cudf::memory_stats_logger();
  state.set_cuda_stream(nvbench::make_cuda_stream_view(cudf::get_default_stream().value()));
  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
    auto const result = cudf::grouped_rolling_window(
      keys->view(), vals->view(), preceding_size, following_size, min_periods, *req);
  });
  auto const elapsed_time = state.get_summary("nv/cold/time/gpu/mean").get_float64("value");
  state.add_element_count(static_cast<double>(num_rows) / elapsed_time / 1'000'000., "Mrows/s");
  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");
}

NVBENCH_BENCH_TYPES(bench_row_grouped_rolling_sum,
                    NVBENCH_TYPE_AXES(nvbench::type_list<std::int32_t, double>))
  .set_name("row_grouped_rolling_sum")
  .add_int64_power_of_two_axis("num_rows", {14, 28})
  .add_int64_axis("preceding_size", {1, 10})
  .add_int64_axis("following_size", {2})
  .add_int64_axis("min_periods", {1})
  .add_int64_axis("cardinality", {10, 100, 1'000'000, 100'000'000});

NVBENCH_BENCH_TYPES(bench_row_grouped_rolling_sum,
                    NVBENCH_TYPE_AXES(nvbench::type_list<std::int32_t>))
  .set_name("row_grouped_rolling_sum_large_windows")
  .add_int64_power_of_two_axis("num_rows", {28})
  .add_int64_axis("preceding_size", {10'000, 40'000})
  .add_int64_axis("following_size", {0})
  .add_int64_axis("min_periods", {1})
  .add_int64_axis("cardinality", {10, 100});
