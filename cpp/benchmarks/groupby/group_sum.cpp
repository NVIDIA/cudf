/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <benchmarks/common/generate_input.hpp>
#include <benchmarks/common/memory_stats.hpp>

#include <cudf/aggregation.hpp>
#include <cudf/binaryop.hpp>
#include <cudf/copying.hpp>
#include <cudf/filling.hpp>
#include <cudf/fixed_point/fixed_point.hpp>
#include <cudf/groupby.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/sorting.hpp>

#include <nvbench/nvbench.cuh>

#include <algorithm>
#include <memory>
#include <vector>

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

  state.set_cuda_stream(nvbench::make_cuda_stream_view(cudf::get_default_stream().get()));
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

  state.set_cuda_stream(nvbench::make_cuda_stream_view(cudf::get_default_stream().get()));
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

static void bench_streaming_groupby_decimal128_sum(nvbench::state& state)
{
  auto const num_rows    = static_cast<cudf::size_type>(state.get_int64("num_rows"));
  auto const batch_size  = static_cast<cudf::size_type>(state.get_int64("batch_size"));
  auto const cardinality = static_cast<cudf::size_type>(state.get_int64("cardinality"));

  data_profile const key_profile =
    data_profile_builder()
      .cardinality(cardinality)
      .no_validity()
      .distribution(cudf::type_id::INT32, distribution_id::UNIFORM, 0, num_rows);
  data_profile const value_profile =
    data_profile_builder().cardinality(0).no_validity().distribution(
      cudf::type_id::DECIMAL128, distribution_id::UNIFORM, -100, 100, numeric::scale_type{-2});
  auto const keys = create_random_column(cudf::type_id::INT32, row_count{num_rows}, key_profile);
  auto const vals =
    create_random_column(cudf::type_id::DECIMAL128, row_count{num_rows}, value_profile);

  std::vector<cudf::size_type> slice_indices;
  for (cudf::size_type start = 0; start < num_rows; start += batch_size) {
    slice_indices.push_back(start);
    slice_indices.push_back(std::min(start + batch_size, num_rows));
  }
  auto const batches = cudf::slice(cudf::table_view({keys->view(), vals->view()}), slice_indices);

  std::vector<cudf::size_type> const key_indices{0};
  std::vector<cudf::groupby::streaming_aggregation_request> requests;
  requests.emplace_back();
  requests.back().column_index = 1;
  requests.back().aggregation  = cudf::make_sum_aggregation<cudf::groupby_aggregation>();

  state.add_element_count(num_rows);
  state.add_global_memory_reads<nvbench::int8_t>(keys->alloc_size() + vals->alloc_size());
  state.set_cuda_stream(nvbench::make_cuda_stream_view(cudf::get_default_stream().get()));
  auto const mem_stats_logger = cudf::memory_stats_logger();
  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
    // Streaming groupby also requires capacity for every row in an individual batch.
    auto sgb =
      cudf::groupby::streaming_groupby(key_indices, requests, std::max(batch_size, cardinality));
    for (auto const& batch : batches) {
      sgb.aggregate(batch);
    }
    auto const result = sgb.finalize();
  });
  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");
}

NVBENCH_BENCH(bench_streaming_groupby_decimal128_sum)
  .set_name("streaming_decimal128_sum")
  .add_int64_power_of_two_axis("num_rows", {20, 24})
  .add_int64_power_of_two_axis("batch_size", {12, 16, 20})
  .add_int64_axis("cardinality", {128, 4'096});

// Streaming groupby over batches whose keys are uniformly random, cyclic (`key[i] = i %
// cardinality`) or sorted runs within each batch.  The key orders differ in how many updates to
// the same group meet in a warp, which drives the atomic contention of the aggregation.
static void bench_streaming_groupby(nvbench::state& state)
{
  auto const num_rows    = static_cast<cudf::size_type>(state.get_int64("num_rows"));
  auto const batch_size  = static_cast<cudf::size_type>(state.get_int64("batch_size"));
  auto const cardinality = static_cast<cudf::size_type>(state.get_int64("cardinality"));
  auto const key_order   = state.get_string("key_order");
  auto const aggs        = state.get_string("aggs");
  // Sorted runs of a single row are the same keys as the cyclic order.
  if (key_order == "sorted" && batch_size / cardinality <= 1) {
    state.skip("sorted keys equal cyclic keys without runs of equal keys");
    return;
  }

  auto const keys = [&] {
    if (key_order == "random") {
      data_profile const profile = data_profile_builder().cardinality(0).no_validity().distribution(
        cudf::type_id::INT32, distribution_id::UNIFORM, 0, cardinality - 1);
      return create_random_column(cudf::type_id::INT32, row_count{num_rows}, profile);
    }
    auto const int32_type = cudf::data_type{cudf::type_id::INT32};
    auto const sequence   = cudf::sequence(num_rows, cudf::numeric_scalar<int32_t>(0));
    if (key_order == "cyclic") {
      return cudf::binary_operation(sequence->view(),
                                    cudf::numeric_scalar<int32_t>(cardinality),
                                    cudf::binary_operator::PYMOD,
                                    int32_type);
    }
    auto const run  = std::max(1, batch_size / cardinality);
    auto const runs = cudf::binary_operation(
      sequence->view(), cudf::numeric_scalar<int32_t>(run), cudf::binary_operator::DIV, int32_type);
    return cudf::binary_operation(runs->view(),
                                  cudf::numeric_scalar<int32_t>(cardinality),
                                  cudf::binary_operator::PYMOD,
                                  int32_type);
  }();
  data_profile const int_profile = data_profile_builder().cardinality(0).no_validity().distribution(
    cudf::type_id::INT32, distribution_id::UNIFORM, 0, 100);
  auto const int_vals =
    create_random_column(cudf::type_id::INT32, row_count{num_rows}, int_profile);
  data_profile const double_profile =
    data_profile_builder().cardinality(0).no_validity().distribution(
      cudf::type_id::FLOAT64, distribution_id::UNIFORM, 0, 100);
  auto const double_vals =
    create_random_column(cudf::type_id::FLOAT64, row_count{num_rows}, double_profile);

  std::vector<cudf::size_type> slice_indices;
  for (cudf::size_type start = 0; start < num_rows; start += batch_size) {
    slice_indices.push_back(start);
    slice_indices.push_back(std::min(start + batch_size, num_rows));
  }
  auto const batches = cudf::slice(
    cudf::table_view({keys->view(), int_vals->view(), double_vals->view()}), slice_indices);

  std::vector<cudf::groupby::streaming_aggregation_request> requests;
  auto const add_request = [&](cudf::size_type column_index,
                               std::unique_ptr<cudf::groupby_aggregation> aggregation) {
    requests.emplace_back();
    requests.back().column_index = column_index;
    requests.back().aggregation  = std::move(aggregation);
  };
  add_request(1, cudf::make_sum_aggregation<cudf::groupby_aggregation>());
  if (aggs == "mixed") {
    add_request(1, cudf::make_min_aggregation<cudf::groupby_aggregation>());
    add_request(2, cudf::make_max_aggregation<cudf::groupby_aggregation>());
    add_request(2, cudf::make_mean_aggregation<cudf::groupby_aggregation>());
  }

  std::vector<cudf::size_type> const key_indices{0};
  state.add_element_count(num_rows);
  state.set_cuda_stream(nvbench::make_cuda_stream_view(cudf::get_default_stream().get()));
  auto const mem_stats_logger = cudf::memory_stats_logger();
  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch&) {
    // Streaming groupby also requires capacity for every row in an individual batch.
    auto sgb =
      cudf::groupby::streaming_groupby(key_indices, requests, std::max(batch_size, cardinality));
    for (auto const& batch : batches) {
      sgb.aggregate(batch);
    }
    auto const result = sgb.finalize();
  });
  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");
}

NVBENCH_BENCH(bench_streaming_groupby)
  .set_name("streaming_groupby")
  .add_int64_power_of_two_axis("num_rows", {24})
  .add_int64_power_of_two_axis("batch_size", {10, 14, 20})
  .add_int64_axis("cardinality", {32, 256, 512, 65'536, 1'048'576})
  .add_string_axis("key_order", {"random", "cyclic", "sorted"})
  .add_string_axis("aggs", {"sum", "mixed"});
