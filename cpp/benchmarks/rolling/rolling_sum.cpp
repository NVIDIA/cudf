/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <benchmarks/common/generate_input.hpp>
#include <benchmarks/common/memory_stats.hpp>

#include <cudf/aggregation.hpp>
#include <cudf/rolling.hpp>
#include <cudf/sorting.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <rmm/device_buffer.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <cuda/iterator>

#include <nvbench/nvbench.cuh>

#include <algorithm>

template <typename Type>
void bench_row_fixed_rolling_sum(nvbench::state& state, nvbench::type_list<Type>)
{
  auto const num_rows       = static_cast<cudf::size_type>(state.get_int64("num_rows"));
  auto const preceding_size = static_cast<cudf::size_type>(state.get_int64("preceding_size"));
  auto const following_size = static_cast<cudf::size_type>(state.get_int64("following_size"));
  auto const min_periods    = static_cast<cudf::size_type>(state.get_int64("min_periods"));

  data_profile const profile = data_profile_builder().cardinality(0).no_validity().distribution(
    cudf::type_to_id<Type>(), distribution_id::UNIFORM, 0, 100);
  auto vals = create_random_column(cudf::type_to_id<Type>(), row_count{num_rows}, profile);

  auto req = cudf::make_sum_aggregation<cudf::rolling_aggregation>();

  auto const mem_stats_logger = cudf::memory_stats_logger();
  state.set_cuda_stream(nvbench::make_cuda_stream_view(cudf::get_default_stream().value()));
  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
    auto const result =
      cudf::rolling_window(vals->view(), preceding_size, following_size, min_periods, *req);
  });
  auto const elapsed_time = state.get_summary("nv/cold/time/gpu/mean").get_float64("value");
  state.add_element_count(static_cast<double>(num_rows) / elapsed_time / 1'000'000., "Mrows/s");
  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");
}

template <typename Type>
void bench_row_variable_rolling_sum(nvbench::state& state, nvbench::type_list<Type>)
{
  auto const num_rows       = static_cast<cudf::size_type>(state.get_int64("num_rows"));
  auto const preceding_size = static_cast<cudf::size_type>(state.get_int64("preceding_size"));
  auto const following_size = static_cast<cudf::size_type>(state.get_int64("following_size"));

  auto vals = [&] {
    data_profile const profile = data_profile_builder().cardinality(0).no_validity().distribution(
      cudf::type_to_id<Type>(), distribution_id::UNIFORM, 0, 100);
    return create_random_column(cudf::type_to_id<Type>(), row_count{num_rows}, profile);
  }();

  auto preceding = [&] {
    auto data = std::vector<cudf::size_type>(num_rows);
    auto it   = cuda::counting_iterator<cudf::size_type>{0};
    std::transform(it, it + num_rows, data.begin(), [num_rows, preceding_size](auto i) {
      return std::min(i + 1, std::max(preceding_size, i + 1 - num_rows));
    });
    auto buf = rmm::device_buffer(
      data.data(), num_rows * sizeof(cudf::size_type), cudf::get_default_stream());
    cudf::get_default_stream().synchronize();
    return std::make_unique<cudf::column>(cudf::data_type(cudf::type_to_id<cudf::size_type>()),
                                          num_rows,
                                          std::move(buf),
                                          rmm::device_buffer{},
                                          0);
  }();

  auto following = [&] {
    auto data = std::vector<cudf::size_type>(num_rows);
    auto it   = cuda::counting_iterator<cudf::size_type>{0};
    std::transform(it, it + num_rows, data.begin(), [num_rows, following_size](auto i) {
      return std::max(-i - 1, std::min(following_size, num_rows - i - 1));
    });
    auto buf = rmm::device_buffer(
      data.data(), num_rows * sizeof(cudf::size_type), cudf::get_default_stream());
    cudf::get_default_stream().synchronize();
    return std::make_unique<cudf::column>(cudf::data_type(cudf::type_to_id<cudf::size_type>()),
                                          num_rows,
                                          std::move(buf),
                                          rmm::device_buffer{},
                                          0);
  }();

  auto req = cudf::make_sum_aggregation<cudf::rolling_aggregation>();

  auto const mem_stats_logger = cudf::memory_stats_logger();
  state.set_cuda_stream(nvbench::make_cuda_stream_view(cudf::get_default_stream().value()));
  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
    auto const result =
      cudf::rolling_window(vals->view(), preceding->view(), following->view(), 1, *req);
  });
  auto const elapsed_time = state.get_summary("nv/cold/time/gpu/mean").get_float64("value");
  state.add_element_count(static_cast<double>(num_rows) / elapsed_time / 1'000'000., "Mrows/s");
  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");
}

NVBENCH_BENCH_TYPES(bench_row_fixed_rolling_sum,
                    NVBENCH_TYPE_AXES(nvbench::type_list<std::int32_t, double>))
  .set_name("row_fixed_rolling_sum")
  .add_int64_power_of_two_axis("num_rows", {14, 22, 28})
  .add_int64_axis("preceding_size", {1, 10, 100})
  .add_int64_axis("following_size", {2})
  .add_int64_axis("min_periods", {1, 20});

NVBENCH_BENCH_TYPES(bench_row_variable_rolling_sum,
                    NVBENCH_TYPE_AXES(nvbench::type_list<std::int32_t, double>))
  .set_name("row_variable_rolling_sum")
  .add_int64_power_of_two_axis("num_rows", {14, 22, 28})
  .add_int64_axis("preceding_size", {10, 100})
  .add_int64_axis("following_size", {2});

NVBENCH_BENCH_TYPES(bench_row_fixed_rolling_sum,
                    NVBENCH_TYPE_AXES(nvbench::type_list<std::int32_t>))
  .set_name("row_fixed_rolling_sum_large_windows")
  .add_int64_power_of_two_axis("num_rows", {28})
  .add_int64_axis("preceding_size", {10'000, 40'000})
  .add_int64_axis("following_size", {0})
  .add_int64_axis("min_periods", {1});
