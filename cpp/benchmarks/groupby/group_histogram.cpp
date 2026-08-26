/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <benchmarks/common/generate_input.hpp>
#include <benchmarks/common/memory_stats.hpp>

#include <cudf/groupby.hpp>

#include <nvbench/nvbench.cuh>

template <typename Type>
void groupby_histogram_helper(nvbench::state& state,
                              cudf::size_type num_rows,
                              cudf::size_type cardinality,
                              double null_probability)
{
  auto const keys = [&] {
    data_profile const profile =
      data_profile_builder()
        .cardinality(cardinality)
        .no_validity()
        .distribution(cudf::type_to_id<int32_t>(), distribution_id::UNIFORM, 0, num_rows);
    return create_random_column(cudf::type_to_id<int32_t>(), row_count{num_rows}, profile);
  }();

  auto const values = [&] {
    auto builder = data_profile_builder().cardinality(0).distribution(
      cudf::type_to_id<Type>(), distribution_id::UNIFORM, 0, num_rows);
    if (null_probability > 0) {
      builder.null_probability(null_probability);
    } else {
      builder.no_validity();
    }
    return create_random_column(
      cudf::type_to_id<Type>(), row_count{num_rows}, data_profile{builder});
  }();

  // Vector of 1 request
  std::vector<cudf::groupby::aggregation_request> requests(1);
  requests.back().values = values->view();
  requests.back().aggregations.push_back(
    cudf::make_histogram_aggregation<cudf::groupby_aggregation>());

  auto const mem_stats_logger = cudf::memory_stats_logger();
  state.set_cuda_stream(nvbench::make_cuda_stream_view(cudf::get_default_stream().value()));
  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
    auto gb_obj       = cudf::groupby::groupby(cudf::table_view({keys->view()}));
    auto const result = gb_obj.aggregate(requests);
  });

  auto const elapsed_time = state.get_summary("nv/cold/time/gpu/mean").get_float64("value");
  state.add_element_count(static_cast<double>(num_rows) / elapsed_time, "rows/s");
  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");
}

template <typename Type>
void bench_groupby_histogram(nvbench::state& state, nvbench::type_list<Type>)
{
  auto const cardinality      = static_cast<cudf::size_type>(state.get_int64("cardinality"));
  auto const num_rows         = static_cast<cudf::size_type>(state.get_int64("num_rows"));
  auto const null_probability = state.get_float64("null_probability");

  if (cardinality > num_rows) {
    state.skip("cardinality > num_rows");
    return;
  }

  groupby_histogram_helper<Type>(state, num_rows, cardinality, null_probability);
}

NVBENCH_BENCH_TYPES(bench_groupby_histogram,
                    NVBENCH_TYPE_AXES(nvbench::type_list<int32_t, int64_t, float, double>))
  .set_name("groupby_histogram")
  .add_float64_axis("null_probability", {0, 0.1, 0.9})
  .add_int64_axis("cardinality", {100, 1'000, 10'000, 100'000, 1'000'000, 10'000'000})
  .add_int64_axis("num_rows", {100, 1'000, 10'000, 100'000, 1'000'000, 10'000'000});
