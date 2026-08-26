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
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/sorting.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/device_buffer.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/tabulate.h>

#include <nvbench/nvbench.cuh>

#include <algorithm>
#include <cstdint>
#include <optional>

void bench_grouped_range_rolling_sum(nvbench::state& state)
{
  auto const num_rows    = static_cast<cudf::size_type>(state.get_int64("num_rows"));
  auto const cardinality = static_cast<cudf::size_type>(state.get_int64("cardinality"));
  // Configurable parameter is window range.
  // Since orderby column is approximately equally spaced at unit
  // intervals, this approximately controls the number of entries in
  // the window.
  auto const preceding_range = cudf::numeric_scalar<cudf::size_type>{
    static_cast<cudf::size_type>(state.get_int64("preceding_range") * 1000), true};
  auto const following_range = cudf::numeric_scalar<cudf::size_type>{
    static_cast<cudf::size_type>(state.get_int64("preceding_range") * 1000), true};
  auto const has_nulls = static_cast<bool>(state.get_int64("has_nulls"));

  auto vals = [&] {
    data_profile const profile = data_profile_builder().cardinality(0).no_validity().distribution(
      cudf::type_to_id<std::int32_t>(), distribution_id::UNIFORM, 0, 100);
    return create_random_column(cudf::type_to_id<std::int32_t>(), row_count{num_rows}, profile);
  }();
  auto const keys = [&] {
    data_profile const profile =
      data_profile_builder()
        .cardinality(cardinality)
        .no_validity()
        .distribution(cudf::type_to_id<cudf::size_type>(), distribution_id::UNIFORM, 0, num_rows);
    auto keys =
      create_random_column(cudf::type_to_id<cudf::size_type>(), row_count{num_rows}, profile);
    return cudf::sort(cudf::table_view{{keys->view()}});
  }();
  auto orderby = [&] {
    auto seq =
      cudf::make_numeric_column(cudf::data_type{cudf::type_to_id<cudf::size_type>()}, num_rows);
    // Equally spaced rows separated by 1000 unit intervals
    thrust::tabulate(
      rmm::exec_policy_nosync(cudf::get_default_stream()),
      seq->mutable_view().begin<cudf::size_type>(),
      seq->mutable_view().end<cudf::size_type>(),
      [] __device__(cudf::size_type i) { return static_cast<cudf::size_type>(i) * 1000; });
    // Add some units of noise
    data_profile profile = data_profile_builder().cardinality(0).distribution(
      cudf::type_to_id<cudf::duration_ms>(), distribution_id::NORMAL, -2000, 2000);
    profile.set_null_probability(has_nulls ? std::optional<double>{400.0 / num_rows}
                                           : std::nullopt);
    auto noise =
      create_random_column(cudf::type_to_id<cudf::size_type>(), row_count{num_rows}, profile);
    auto result =
      cudf::binary_operation(seq->view(), noise->view(), cudf::binary_operator::ADD, seq->type());
    auto columns = cudf::sort_by_key(cudf::table_view{{result->view()}},
                                     cudf::table_view{{keys->get_column(0).view(), result->view()}},
                                     {cudf::order::ASCENDING, cudf::order::ASCENDING},
                                     {cudf::null_order::AFTER, cudf::null_order::AFTER})
                     ->release();
    return std::move(columns[0]);
  }();

  auto req = cudf::make_sum_aggregation<cudf::rolling_aggregation>();

  auto const mem_stats_logger = cudf::memory_stats_logger();
  state.set_cuda_stream(nvbench::make_cuda_stream_view(cudf::get_default_stream().value()));
  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
    auto const result =
      cudf::grouped_range_rolling_window(keys->view(),
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

NVBENCH_BENCH(bench_grouped_range_rolling_sum)
  .set_name("range_grouped_rolling_sum")
  .add_int64_power_of_two_axis("num_rows", {14, 22, 28})
  .add_int64_axis("preceding_range", {100})
  .add_int64_axis("following_range", {100})
  .add_int64_axis("has_nulls", {0, 1})
  .add_int64_axis("cardinality", {10, 100, 1'000'000, 100'000'000});

NVBENCH_BENCH(bench_grouped_range_rolling_sum)
  .set_name("range_grouped_rolling_sum_large_windows")
  .add_int64_power_of_two_axis("num_rows", {28})
  .add_int64_axis("preceding_range", {10'000, 40'000})
  .add_int64_axis("following_range", {0})
  .add_int64_axis("has_nulls", {0, 1})
  .add_int64_axis("cardinality", {100});
