/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <benchmarks/common/generate_input.hpp>
#include <benchmarks/common/memory_stats.hpp>

#include <cudf/detail/utilities/stream_pool.hpp>
#include <cudf/groupby.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <BS_thread_pool.hpp>
#include <nvbench/nvbench.cuh>

template <typename Type>
void bench_groupby_max_multithreaded(nvbench::state& state, nvbench::type_list<Type>)
{
  auto const cardinality      = static_cast<cudf::size_type>(state.get_int64("cardinality"));
  auto const num_rows         = static_cast<cudf::size_type>(state.get_int64("num_rows"));
  auto const null_probability = state.get_float64("null_probability");
  auto const num_threads      = state.get_int64("num_threads");
  auto const num_aggregations = state.get_int64("num_aggregations");

  auto const keys = [&] {
    data_profile const profile =
      data_profile_builder()
        .cardinality(cardinality)
        .no_validity()
        .distribution(cudf::type_to_id<int32_t>(), distribution_id::UNIFORM, 0, num_rows);
    return create_random_column(cudf::type_to_id<int32_t>(), row_count{num_rows}, profile);
  }();

  auto const make_values = [&]() {
    auto builder = data_profile_builder().cardinality(0).distribution(
      cudf::type_to_id<Type>(), distribution_id::UNIFORM, 0, num_rows);
    if (null_probability > 0) {
      builder.null_probability(null_probability);
    } else {
      builder.no_validity();
    }
    return create_random_column(
      cudf::type_to_id<Type>(), row_count{num_rows}, data_profile{builder});
  };

  auto keys_view = keys->view();

  auto streams = cudf::detail::fork_streams(cudf::get_default_stream(), num_threads);
  BS::thread_pool threads(num_threads);

  std::vector<std::unique_ptr<cudf::column>> val_cols;
  std::vector<std::vector<cudf::groupby::aggregation_request>> requests(num_threads);
  for (auto& thread_requests : requests) {
    for (int64_t j = 0; j < num_aggregations; j++) {
      thread_requests.emplace_back();
      val_cols.emplace_back(make_values());
      thread_requests.back().values = val_cols.back()->view();
      thread_requests.back().aggregations.push_back(
        cudf::make_max_aggregation<cudf::groupby_aggregation>());
    }
  }

  auto const mem_stats_logger = cudf::memory_stats_logger();
  state.exec(
    nvbench::exec_tag::sync | nvbench::exec_tag::timer, [&](nvbench::launch& launch, auto& timer) {
      auto perform_agg = [&](int64_t index) {
        auto gb_obj = cudf::groupby::groupby(cudf::table_view({keys_view, keys_view, keys_view}));
        gb_obj.aggregate(requests[index], streams[index]);
      };
      timer.start();
      threads.detach_sequence(decltype(num_threads){0}, num_threads, perform_agg);
      threads.wait();
      cudf::detail::join_streams(streams, cudf::get_default_stream());
      cudf::get_default_stream().synchronize();
      timer.stop();
    });

  auto const elapsed_time = state.get_summary("nv/cold/time/gpu/mean").get_float64("value");
  state.add_element_count(
    static_cast<double>(num_rows * num_threads * num_aggregations) / elapsed_time / 1'000'000.,
    "Mrows/s");
  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");
}

NVBENCH_BENCH_TYPES(bench_groupby_max_multithreaded,
                    NVBENCH_TYPE_AXES(nvbench::type_list<int32_t, int64_t, float, double>))
  .set_name("groupby_max_multithreaded")
  .add_int64_axis("cardinality", {0})
  .add_int64_power_of_two_axis("num_rows", {12, 18})
  .add_float64_axis("null_probability", {0, 0.1, 0.9})
  .add_int64_axis("num_aggregations", {1})
  .add_int64_axis("num_threads", {1, 2, 4, 8});
