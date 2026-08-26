/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <benchmarks/common/generate_input.hpp>
#include <benchmarks/common/memory_stats.hpp>

#include <cudf/aggregation.hpp>
#include <cudf/binaryop.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/rolling.hpp>
#include <cudf/rolling/range_window_bounds.hpp>
#include <cudf/sorting.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/device_buffer.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/tabulate.h>

#include <nvbench/nvbench.cuh>

#include <algorithm>
#include <cstdint>
#include <optional>

void bench_range_rolling_sum(nvbench::state& state)
{
  auto const num_rows = static_cast<cudf::size_type>(state.get_int64("num_rows"));
  // Configurable parameter is window range in seconds.
  // Since orderby column is approximately equally spaced at 1s
  // intervals, this approximately controls the number of entries in
  // the window.
  auto const preceding_range = cudf::duration_scalar<cudf::duration_ms>{
    cudf::duration_ms{state.get_int64("preceding_range") * 1000}, true};
  auto const following_range = cudf::duration_scalar<cudf::duration_ms>{
    cudf::duration_ms{state.get_int64("following_range") * 1000}, true};
  auto const has_nulls = static_cast<bool>(state.get_int64("has_nulls"));

  auto vals = [&] {
    data_profile const profile = data_profile_builder().cardinality(0).no_validity().distribution(
      cudf::type_to_id<std::int32_t>(), distribution_id::UNIFORM, 0, 100);
    return create_random_column(cudf::type_to_id<std::int32_t>(), row_count{num_rows}, profile);
  }();

  auto orderby = [&] {
    auto seq = cudf::make_timestamp_column(cudf::data_type{cudf::type_to_id<cudf::timestamp_ms>()},
                                           num_rows);
    // Equally spaced rows separated by 1s
    thrust::tabulate(
      rmm::exec_policy_nosync(cudf::get_default_stream()),
      seq->mutable_view().begin<cudf::timestamp_ms>(),
      seq->mutable_view().end<cudf::timestamp_ms>(),
      [] __device__(cudf::size_type i) {
        return cudf::timestamp_ms{cudf::duration_ms{static_cast<std::int64_t>(i) * 1000}};
      });
    // Add some milliseconds of noise
    data_profile profile = data_profile_builder().cardinality(0).distribution(
      cudf::type_to_id<cudf::duration_ms>(), distribution_id::NORMAL, -2000, 2000);
    profile.set_null_probability(has_nulls ? std::optional<double>{400.0 / num_rows}
                                           : std::nullopt);
    auto noise =
      create_random_column(cudf::type_to_id<cudf::duration_ms>(), row_count{num_rows}, profile);
    auto result =
      cudf::binary_operation(seq->view(), noise->view(), cudf::binary_operator::ADD, seq->type());
    auto columns =
      cudf::sort(
        cudf::table_view{{result->view()}}, {cudf::order::ASCENDING}, {cudf::null_order::AFTER})
        ->release();
    return std::move(columns[0]);
  }();

  auto req = cudf::make_sum_aggregation<cudf::rolling_aggregation>();

  auto const mem_stats_logger = cudf::memory_stats_logger();
  state.set_cuda_stream(nvbench::make_cuda_stream_view(cudf::get_default_stream().value()));
  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
    auto const result =
      cudf::grouped_range_rolling_window(cudf::table_view{},
                                         orderby->view(),
                                         cudf::order::ASCENDING,
                                         vals->view(),
                                         cudf::range_window_bounds::get(preceding_range),
                                         cudf::range_window_bounds::get(following_range),
                                         1,
                                         *req);
  });
  auto const elapsed_time = state.get_summary("nv/cold/time/gpu/mean").get_float64("value");
  state.add_element_count(static_cast<double>(num_rows) / elapsed_time / 1'000'000., "Mrows/s");
  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");
}

NVBENCH_BENCH(bench_range_rolling_sum)
  .set_name("range_rolling_sum")
  .add_int64_power_of_two_axis("num_rows", {14, 22, 28})
  .add_int64_axis("preceding_range", {100})
  .add_int64_axis("following_range", {100})
  .add_int64_axis("has_nulls", {0, 1});

NVBENCH_BENCH(bench_range_rolling_sum)
  .set_name("range_rolling_sum_large_windows")
  .add_int64_power_of_two_axis("num_rows", {28})
  .add_int64_axis("preceding_range", {10'000, 40'000})
  .add_int64_axis("following_range", {0})
  .add_int64_axis("has_nulls", {0, 1});
