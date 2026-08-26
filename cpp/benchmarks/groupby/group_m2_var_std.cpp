/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <benchmarks/common/generate_input.hpp>
#include <benchmarks/common/memory_stats.hpp>

#include <cudf/groupby.hpp>

#include <nvbench/nvbench.cuh>

namespace {

template <typename Type, cudf::aggregation::Kind Agg>
void run_benchmark(nvbench::state& state,
                   cudf::size_type num_rows,
                   cudf::size_type num_aggs,
                   cudf::size_type value_key_ratio,
                   double null_probability)
{
  auto const keys = [&] {
    data_profile const profile =
      data_profile_builder()
        .cardinality(num_rows / value_key_ratio)
        .no_validity()
        .distribution(cudf::type_to_id<int32_t>(), distribution_id::UNIFORM, 0, num_rows);
    return create_random_column(cudf::type_to_id<int32_t>(), row_count{num_rows}, profile);
  }();

  auto values_builder = data_profile_builder().cardinality(0).distribution(
    cudf::type_to_id<Type>(), distribution_id::UNIFORM, 0, num_rows);
  if (null_probability > 0) {
    values_builder.null_probability(null_probability);
  } else {
    values_builder.no_validity();
  }

  std::vector<std::unique_ptr<cudf::column>> values_cols;
  std::vector<cudf::groupby::aggregation_request> requests;
  values_cols.reserve(num_aggs);
  requests.reserve(num_aggs);
  for (cudf::size_type i = 0; i < num_aggs; i++) {
    auto values = create_random_column(
      cudf::type_to_id<Type>(), row_count{num_rows}, data_profile{values_builder});
    auto request   = cudf::groupby::aggregation_request{};
    request.values = values->view();
    if constexpr (Agg == cudf::aggregation::Kind::M2) {
      request.aggregations.push_back(cudf::make_m2_aggregation<cudf::groupby_aggregation>());
    } else if constexpr (Agg == cudf::aggregation::Kind::VARIANCE) {
      request.aggregations.push_back(cudf::make_variance_aggregation<cudf::groupby_aggregation>());
    } else if constexpr (Agg == cudf::aggregation::Kind::STD) {
      request.aggregations.push_back(cudf::make_std_aggregation<cudf::groupby_aggregation>());
    } else {
      CUDF_FAIL("Unsupported aggregation kind.");
    }
    values_cols.emplace_back(std::move(values));
    requests.emplace_back(std::move(request));
  }

  auto const mem_stats_logger = cudf::memory_stats_logger();
  state.set_cuda_stream(nvbench::make_cuda_stream_view(cudf::get_default_stream().value()));
  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch&) {
    auto gb_obj                        = cudf::groupby::groupby(cudf::table_view({keys->view()}));
    [[maybe_unused]] auto const result = gb_obj.aggregate(requests);
  });

  auto const elapsed_time = state.get_summary("nv/cold/time/gpu/mean").get_float64("value");
  state.add_element_count(static_cast<double>(num_rows) / elapsed_time, "rows/s");
  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");
}

}  // namespace

template <typename Type, cudf::aggregation::Kind Agg>
void bench_groupby_m2_var_std(nvbench::state& state,
                              nvbench::type_list<Type, nvbench::enum_type<Agg>>)
{
  auto const value_key_ratio  = static_cast<cudf::size_type>(state.get_int64("value_key_ratio"));
  auto const num_rows         = static_cast<cudf::size_type>(state.get_int64("num_rows"));
  auto const null_probability = state.get_float64("null_probability");
  auto const num_aggs         = static_cast<cudf::size_type>(state.get_int64("num_aggs"));
  run_benchmark<Type, Agg>(state, num_rows, num_aggs, value_key_ratio, null_probability);
}

using Types    = nvbench::type_list<int32_t, double>;
using AggKinds = nvbench::enum_type_list<cudf::aggregation::Kind::M2,
                                         cudf::aggregation::Kind::VARIANCE,
                                         cudf::aggregation::Kind::STD>;

NVBENCH_BENCH_TYPES(bench_groupby_m2_var_std, NVBENCH_TYPE_AXES(Types, AggKinds))
  .set_name("groupby_m2_var_std")
  .add_int64_axis("value_key_ratio", {20, 100})
  .add_int64_axis("num_rows", {100'000, 10'000'000})
  .add_float64_axis("null_probability", {0, 0.5})
  .add_int64_axis("num_aggs", {1, 10, 50, 100});
