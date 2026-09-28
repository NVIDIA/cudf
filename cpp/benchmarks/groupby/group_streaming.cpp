/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <benchmarks/common/generate_input.hpp>
#include <benchmarks/common/memory_stats.hpp>

#include <cudf/copying.hpp>
#include <cudf/groupby.hpp>
#include <cudf/table/table_view.hpp>

#include <nvbench/nvbench.cuh>

#include <memory>
#include <numeric>
#include <string>
#include <vector>

/*
 * The streaming groupby against the stateless one, over the two kinds of keys the streaming
 * groupby handles differently: one int32 key, which fits in the packed-key table, and three of
 * them, twelve bytes that the generic row-operator path takes.  The input is the same random
 * key repeated, so both key layouts have the same groups.  `num_batches` cuts the input into
 * that many batches for the streaming path; the stateless path always takes it whole.
 */
void bench_streaming_groupby(nvbench::state& state)
{
  auto const num_rows         = static_cast<cudf::size_type>(state.get_int64("num_rows"));
  auto const cardinality      = static_cast<cudf::size_type>(state.get_int64("cardinality"));
  auto const num_batches      = static_cast<cudf::size_type>(state.get_int64("num_batches"));
  auto const num_aggregations = state.get_int64("num_aggregations");
  auto const is_streaming     = state.get_string("api") == "streaming";
  auto const num_keys         = state.get_string("keys") == "int32" ? 1 : 3;

  if (!is_streaming && num_batches != 1) {
    state.skip("the stateless groupby takes the input whole");
    return;
  }

  auto const keys = [&] {
    data_profile const profile =
      data_profile_builder()
        .cardinality(cardinality)
        .no_validity()
        .distribution(cudf::type_to_id<int32_t>(), distribution_id::UNIFORM, 0, num_rows);
    return create_random_column(cudf::type_to_id<int32_t>(), row_count{num_rows}, profile);
  }();
  auto const values = [&] {
    data_profile const profile = data_profile_builder().cardinality(0).no_validity().distribution(
      cudf::type_to_id<int64_t>(), distribution_id::UNIFORM, 0, num_rows);
    return create_random_column(cudf::type_to_id<int64_t>(), row_count{num_rows}, profile);
  }();

  std::vector<cudf::column_view> columns(num_keys, keys->view());
  columns.push_back(values->view());
  auto const table = cudf::table_view(columns);

  std::vector<cudf::size_type> key_indices(num_keys);
  std::iota(key_indices.begin(), key_indices.end(), 0);
  auto const value_index = static_cast<cudf::size_type>(num_keys);

  auto const make_aggregations = [&] {
    std::vector<std::unique_ptr<cudf::groupby_aggregation>> aggregations;
    aggregations.push_back(cudf::make_sum_aggregation<cudf::groupby_aggregation>());
    if (num_aggregations > 1) {
      aggregations.push_back(cudf::make_min_aggregation<cudf::groupby_aggregation>());
    }
    if (num_aggregations > 2) {
      aggregations.push_back(cudf::make_max_aggregation<cudf::groupby_aggregation>());
    }
    return aggregations;
  };

  auto const mem_stats_logger = cudf::memory_stats_logger();
  state.set_cuda_stream(nvbench::make_cuda_stream_view(cudf::get_default_stream().get()));

  if (is_streaming) {
    std::vector<cudf::size_type> splits;
    for (cudf::size_type b = 1; b < num_batches; ++b) {
      splits.push_back(
        static_cast<cudf::size_type>(static_cast<int64_t>(num_rows) * b / num_batches));
    }
    auto const batches = cudf::split(table, splits);

    std::vector<cudf::groupby::streaming_aggregation_request> requests;
    for (auto& aggregation : make_aggregations()) {
      cudf::groupby::streaming_aggregation_request req;
      req.column_index = value_index;
      req.aggregation  = std::move(aggregation);
      requests.push_back(std::move(req));
    }

    // The distinct keys are hinted at exactly, as a caller with an estimate of them would.
    state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
      auto sgb = cudf::groupby::streaming_groupby(key_indices, requests, cardinality);
      for (auto const& batch : batches) {
        sgb.aggregate(batch);
      }
      auto const result = sgb.finalize();
    });
  } else {
    std::vector<cudf::groupby::aggregation_request> requests;
    requests.emplace_back();
    requests[0].values       = values->view();
    requests[0].aggregations = make_aggregations();
    state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
      auto gb_obj       = cudf::groupby::groupby(table.select(key_indices));
      auto const result = gb_obj.aggregate(requests);
    });
  }

  auto const elapsed_time = state.get_summary("nv/cold/time/gpu/mean").get_float64("value");
  state.add_element_count(static_cast<double>(num_rows) / elapsed_time / 1'000'000., "Mrows/s");
  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");
}

NVBENCH_BENCH(bench_streaming_groupby)
  .set_name("groupby_streaming")
  .add_string_axis("api", {"normal", "streaming"})
  .add_string_axis("keys", {"int32", "int32x3"})
  .add_int64_axis("num_rows", {16'777'216})
  .add_int64_axis("cardinality", {100, 10'000, 1'000'000})
  .add_int64_axis("num_batches", {1, 16})
  .add_int64_axis("num_aggregations", {1, 3});
