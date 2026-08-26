/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <benchmarks/common/generate_input.hpp>
#include <benchmarks/common/memory_stats.hpp>

#include <cudf/aggregation.hpp>
#include <cudf/copying.hpp>
#include <cudf/groupby.hpp>
#include <cudf/sorting.hpp>

#include <nvbench/nvbench.cuh>

using Types = nvbench::type_list<int64_t, numeric::decimal64>;
NVBENCH_DECLARE_TYPE_STRINGS(numeric::decimal64, "decimal64", "decimal64");

template <typename DataType>
static void bench_groupby_basic_sum(nvbench::state& state, nvbench::type_list<DataType>)
{
  auto const num_rows     = static_cast<cudf::size_type>(state.get_int64("num_rows"));
  auto const data_type_id = cudf::type_to_id<DataType>();

  data_profile const profile = data_profile_builder().cardinality(0).no_validity().distribution(
    data_type_id, distribution_id::UNIFORM, 0, 100);
  auto keys = create_random_column(data_type_id, row_count{num_rows}, profile);
  auto vals = create_random_column(data_type_id, row_count{num_rows}, profile);

  std::vector<cudf::groupby::aggregation_request> requests;
  requests.emplace_back(cudf::groupby::aggregation_request());
  requests[0].values = vals->view();
  requests[0].aggregations.push_back(cudf::make_sum_aggregation<cudf::groupby_aggregation>());

  state.add_global_memory_reads<nvbench::int8_t>(vals->alloc_size());
  std::size_t write_size = 0;

  state.set_cuda_stream(nvbench::make_cuda_stream_view(cudf::get_default_stream().value()));
  auto const mem_stats_logger = cudf::memory_stats_logger();
  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
    cudf::groupby::groupby gb_obj(cudf::table_view({keys->view(), keys->view(), keys->view()}));
    auto const result = gb_obj.aggregate(requests);
    write_size = result.first->alloc_size() + result.second.front().results.front()->alloc_size();
  });
  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");

  state.add_global_memory_writes<nvbench::int8_t>(write_size);
}

NVBENCH_BENCH_TYPES(bench_groupby_basic_sum, NVBENCH_TYPE_AXES(Types))
  .set_name("sum")
  .add_int64_axis("num_rows", {100'000, 1'000'000, 10'000'000, 100'000'000});

template <typename DataType>
static void bench_groupby_pre_sorted_sum(nvbench::state& state, nvbench::type_list<DataType>)
{
  auto const num_rows     = static_cast<cudf::size_type>(state.get_int64("num_rows"));
  auto const data_type_id = cudf::type_to_id<DataType>();

  data_profile profile = data_profile_builder().cardinality(0).no_validity().distribution(
    data_type_id, distribution_id::UNIFORM, 0, 100);
  auto keys_table = create_random_table({data_type_id}, row_count{num_rows}, profile);
  profile.set_null_probability(0.1);
  auto vals = create_random_column(data_type_id, row_count{num_rows}, profile);

  auto sort_order  = cudf::sorted_order(*keys_table);
  auto sorted_keys = cudf::gather(*keys_table, *sort_order);
  // No need to sort values using sort_order because they were generated randomly

  std::vector<cudf::groupby::aggregation_request> requests;
  requests.emplace_back(cudf::groupby::aggregation_request());
  requests[0].values = vals->view();
  requests[0].aggregations.push_back(cudf::make_sum_aggregation<cudf::groupby_aggregation>());

  state.add_global_memory_reads<nvbench::int8_t>(vals->alloc_size());
  std::size_t write_size = 0;

  state.set_cuda_stream(nvbench::make_cuda_stream_view(cudf::get_default_stream().value()));
  auto const mem_stats_logger = cudf::memory_stats_logger();
  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
    cudf::groupby::groupby gb_obj(*sorted_keys, cudf::null_policy::EXCLUDE, cudf::sorted::YES);
    auto const result = gb_obj.aggregate(requests);
    write_size = result.first->alloc_size() + result.second.front().results.front()->alloc_size();
  });
  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");

  state.add_global_memory_writes<nvbench::int8_t>(write_size);
}

NVBENCH_BENCH_TYPES(bench_groupby_pre_sorted_sum, NVBENCH_TYPE_AXES(Types))
  .set_name("pre_sorted_sum")
  .add_int64_axis("num_rows", {100'000, 1'000'000, 10'000'000, 100'000'000});
