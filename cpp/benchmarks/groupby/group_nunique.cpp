/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <benchmarks/common/generate_input.hpp>
#include <benchmarks/common/memory_stats.hpp>

#include <cudf/groupby.hpp>

#include <nvbench/nvbench.cuh>

namespace {

template <typename... Args>
auto make_aggregation_request_vector(cudf::column_view const& values, Args&&... args)
{
  std::vector<std::unique_ptr<cudf::groupby_aggregation>> aggregations;
  (aggregations.emplace_back(std::forward<Args>(args)), ...);

  std::vector<cudf::groupby::aggregation_request> requests;
  requests.emplace_back(cudf::groupby::aggregation_request{values, std::move(aggregations)});

  return requests;
}

}  // namespace

template <typename Type>
void bench_groupby_nunique(nvbench::state& state, nvbench::type_list<Type>)
{
  auto const size        = static_cast<cudf::size_type>(state.get_int64("num_rows"));
  auto const cardinality = static_cast<cudf::size_type>(state.get_int64("cardinality"));

  auto const keys = [&] {
    data_profile profile =
      data_profile_builder()
        .cardinality(cardinality)
        .no_validity()
        .distribution(cudf::type_to_id<int32_t>(), distribution_id::UNIFORM, 0, size);
    return create_random_column(cudf::type_to_id<int32_t>(), row_count{size}, profile);
  }();

  auto const vals = [&] {
    data_profile profile =
      data_profile_builder()
        .cardinality(cardinality)
        .distribution(cudf::type_to_id<Type>(), distribution_id::UNIFORM, 0, size);
    if (auto const null_freq = state.get_float64("null_probability"); null_freq > 0) {
      profile.set_null_probability(null_freq);
    } else {
      profile.set_null_probability(std::nullopt);
    }
    return create_random_column(cudf::type_to_id<Type>(), row_count{size}, profile);
  }();

  auto const requests = make_aggregation_request_vector(
    *vals, cudf::make_nunique_aggregation<cudf::groupby_aggregation>());

  state.set_cuda_stream(nvbench::make_cuda_stream_view(cudf::get_default_stream().value()));
  auto const mem_stats_logger = cudf::memory_stats_logger();
  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
    auto gb_obj =
      cudf::groupby::groupby(cudf::table_view({keys->view(), keys->view(), keys->view()}));
    auto const result = gb_obj.aggregate(requests);
  });
  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");
}

NVBENCH_BENCH_TYPES(bench_groupby_nunique, NVBENCH_TYPE_AXES(nvbench::type_list<int32_t, int64_t>))
  .set_name("nunique")
  .add_int64_power_of_two_axis("num_rows", {12, 16, 20, 24})
  .add_int64_axis("cardinality", {0})
  .add_float64_axis("null_probability", {0, 0.5});
