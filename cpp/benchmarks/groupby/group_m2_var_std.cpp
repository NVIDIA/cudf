/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <benchmarks/common/generate_input.hpp>
#include <benchmarks/common/memory_stats.hpp>

#include <cudf/column/column_factories.hpp>
#include <cudf/detail/utilities/cuda_memcpy.hpp>
#include <cudf/groupby.hpp>

#include <nvbench/nvbench.cuh>

#include <algorithm>
#include <limits>
#include <random>
#include <vector>

namespace {

template <typename Type, cudf::aggregation::Kind Agg>
void run_benchmark(nvbench::state& state,
                   cudf::size_type num_rows,
                   cudf::size_type num_aggs,
                   cudf::column_view keys,
                   double null_probability)
{
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
  state.set_cuda_stream(nvbench::make_cuda_stream_view(cudf::get_default_stream().get()));
  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch&) {
    auto gb_obj                        = cudf::groupby::groupby(cudf::table_view({keys}));
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
  data_profile const profile  = data_profile_builder()
                                 .cardinality(num_rows / value_key_ratio)
                                 .no_validity()
                                 .distribution(cudf::type_to_id<int32_t>(),
                                               distribution_id::UNIFORM,
                                               static_cast<Type>(0),
                                               static_cast<Type>(num_rows));
  auto const keys = create_random_column(cudf::type_id::INT32, row_count{num_rows}, profile);
  run_benchmark<Type, Agg>(state, num_rows, num_aggs, keys->view(), null_probability);
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

// Controlled cases complement the random-cardinality benchmark above. Key generation and
// transfer are outside timing. Seed 1 fixes the shuffled layout; value generation also uses
// create_random_column's default seed 1. The groupby object is still fresh for every iteration.
namespace {

std::unique_ptr<cudf::column> make_controlled_keys(cudf::size_type num_rows,
                                                   cudf::size_type num_groups,
                                                   bool skewed)
{
  std::vector<int32_t> host_keys(num_rows);
  for (cudf::size_type row = 0; row < num_rows; ++row) {
    // Skew: every tenth row is a distinct cold key; all remaining rows share key zero.
    host_keys[row] = skewed ? (row % 10 == 0 ? row / 10 + 1 : 0) : row % num_groups;
  }
  std::mt19937 generator{1};
  std::shuffle(host_keys.begin(), host_keys.end(), generator);
  auto keys = cudf::make_numeric_column(
    cudf::data_type{cudf::type_id::INT32}, num_rows, cudf::mask_state::UNALLOCATED);
  cudf::detail::cuda_memcpy(
    cudf::device_span<int32_t>{keys->mutable_view().data<int32_t>(), host_keys.size()},
    cudf::host_span<int32_t const>{host_keys.data(), host_keys.size()},
    cudf::get_default_stream());
  return keys;
}

}  // namespace

template <typename Type, cudf::aggregation::Kind Agg>
void bench_groupby_m2_group_size(nvbench::state& state,
                                 nvbench::type_list<Type, nvbench::enum_type<Agg>>)
{
  auto const group_size = state.get_int64("group_size");
  auto const num_groups = state.get_int64("num_groups");
  CUDF_EXPECTS(group_size > 0 && num_groups > 0 &&
                 num_groups <= std::numeric_limits<cudf::size_type>::max() / group_size,
               "Group size and count must produce a positive, representable row count");
  auto const num_rows = static_cast<cudf::size_type>(group_size * num_groups);
  auto const keys = make_controlled_keys(num_rows, static_cast<cudf::size_type>(num_groups), false);
  run_benchmark<Type, Agg>(state,
                           num_rows,
                           static_cast<cudf::size_type>(state.get_int64("num_aggs")),
                           keys->view(),
                           state.get_float64("null_probability"));
}

template <typename Type, cudf::aggregation::Kind Agg>
void bench_groupby_m2_distribution(nvbench::state& state,
                                   nvbench::type_list<Type, nvbench::enum_type<Agg>>)
{
  auto const distribution = state.get_string("distribution");
  CUDF_EXPECTS(distribution == "balanced" || distribution == "hot_90_percent",
               "Unknown key distribution");
  auto const skewed = distribution == "hot_90_percent";
  auto const rows   = state.get_int64("num_rows");
  auto const groups = skewed ? int64_t{1} : state.get_int64("num_groups");
  CUDF_EXPECTS(
    rows > 0 && rows <= std::numeric_limits<cudf::size_type>::max() && groups > 0 && groups <= rows,
    "Require 0 < num_groups <= num_rows <= size_type max");
  auto const num_rows = static_cast<cudf::size_type>(rows);
  auto const keys = make_controlled_keys(num_rows, static_cast<cudf::size_type>(groups), skewed);
  run_benchmark<Type, Agg>(state,
                           num_rows,
                           static_cast<cudf::size_type>(state.get_int64("num_aggs")),
                           keys->view(),
                           state.get_float64("null_probability"));
}

// Exactly num_groups groups, each with group_size rows, including singleton groups.
NVBENCH_BENCH_TYPES(bench_groupby_m2_group_size, NVBENCH_TYPE_AXES(Types, AggKinds))
  .set_name("groupby_m2_group_size")
  .add_int64_axis("group_size", {1, 31, 32, 33, 1023, 1024, 1025})
  .add_int64_axis("num_groups", {4096})
  .add_float64_axis("null_probability", {0, 0.5})
  .add_int64_axis("num_aggs", {1});

// Balanced groups differ in size by at most one row.
NVBENCH_BENCH_TYPES(bench_groupby_m2_distribution, NVBENCH_TYPE_AXES(Types, AggKinds))
  .set_name("groupby_m2_cardinality")
  .add_string_axis("distribution", {"balanced"})
  .add_int64_axis("num_groups", {1, 16, 256})
  .add_int64_axis("num_rows", {100'000, 10'000'000})
  .add_float64_axis("null_probability", {0, 0.5})
  .add_int64_axis("num_aggs", {1, 10});

// One hot group plus ceil(num_rows / 10) singleton cold groups.
NVBENCH_BENCH_TYPES(bench_groupby_m2_distribution, NVBENCH_TYPE_AXES(Types, AggKinds))
  .set_name("groupby_m2_skew")
  .add_string_axis("distribution", {"hot_90_percent"})
  .add_int64_axis("num_rows", {100'000, 10'000'000})
  .add_float64_axis("null_probability", {0, 0.5})
  .add_int64_axis("num_aggs", {1, 10});
