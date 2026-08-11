/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <benchmarks/common/generate_input.hpp>
#include <benchmarks/common/memory_stats.hpp>

#include <cudf/aggregation.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/filling.hpp>
#include <cudf/rolling.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <nvbench/nvbench.cuh>

#include <cstdint>
#include <vector>

// Per-row (column-valued) RANGE bounds: `bounded_closed_column`. This mirrors the scalar
// `range_rolling_sum` benchmark so the two can be compared directly to quantify the overhead of
// reading a per-row delta instead of broadcasting a single scalar. Constant delta columns keep the
// window sizes (and therefore the aggregation work) identical to the scalar case, isolating the
// cost of the per-row read.
void bench_range_rolling_column_bounds_sum(nvbench::state& state)
{
  auto const num_rows        = static_cast<cudf::size_type>(state.get_int64("num_rows"));
  auto const preceding_range = state.get_int64("preceding_range");
  auto const following_range = state.get_int64("following_range");

  auto vals = [&] {
    data_profile const profile = data_profile_builder().cardinality(0).no_validity().distribution(
      cudf::type_to_id<std::int32_t>(), distribution_id::UNIFORM, 0, 100);
    return create_random_column(cudf::type_to_id<std::int32_t>(), row_count{num_rows}, profile);
  }();

  // Equally-spaced ascending integer orderby (1 unit apart), so `preceding_range`/`following_range`
  // approximately control the number of rows in each window.
  auto const orderby = cudf::sequence(
    num_rows, cudf::numeric_scalar<std::int64_t>(0), cudf::numeric_scalar<std::int64_t>(1));

  // Per-row delta columns (matching the orderby type), each filled with a single constant.
  auto const preceding_col =
    cudf::make_column_from_scalar(cudf::numeric_scalar<std::int64_t>(preceding_range), num_rows);
  auto const following_col =
    cudf::make_column_from_scalar(cudf::numeric_scalar<std::int64_t>(following_range), num_rows);

  std::vector<cudf::rolling_request> requests;
  requests.push_back({vals->view(), 1, cudf::make_sum_aggregation<cudf::rolling_aggregation>()});

  auto const mem_stats_logger = cudf::memory_stats_logger();
  state.set_cuda_stream(nvbench::make_cuda_stream_view(cudf::get_default_stream().value()));
  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
    auto const result =
      cudf::grouped_range_rolling_window(cudf::table_view{},
                                         orderby->view(),
                                         cudf::order::ASCENDING,
                                         cudf::null_order::BEFORE,
                                         cudf::bounded_closed_column{preceding_col->view()},
                                         cudf::bounded_closed_column{following_col->view()},
                                         requests);
  });
  auto const elapsed_time = state.get_summary("nv/cold/time/gpu/mean").get_float64("value");
  state.add_element_count(static_cast<double>(num_rows) / elapsed_time / 1'000'000., "Mrows/s");
  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");
}

NVBENCH_BENCH(bench_range_rolling_column_bounds_sum)
  .set_name("range_rolling_column_bounds_sum")
  .add_int64_power_of_two_axis("num_rows", {14, 22, 28})
  .add_int64_axis("preceding_range", {100})
  .add_int64_axis("following_range", {100});
