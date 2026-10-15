/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <benchmarks/common/generate_input.hpp>
#include <benchmarks/common/memory_stats.hpp>

#include <cudf/aggregation.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/groupby/direct_groupby.hpp>
#include <cudf/groupby.hpp>
#include <cudf/groupby/direct_groupby.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <thrust/execution_policy.h>
#include <thrust/tabulate.h>

#include <nvbench/nvbench.cuh>

#include <algorithm>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct dense_key_fn {
  std::uint64_t num_distinct;
  std::uint32_t stride;

  __device__ std::uint32_t operator()(std::uint64_t idx) const
  {
    auto z = idx + 0x9e3779b97f4a7c15ULL;
    z      = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z      = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    z      = z ^ (z >> 31);
    return static_cast<std::uint32_t>(z % num_distinct) * stride;
  }
};

std::vector<cudf::groupby::aggregation_request> make_requests(std::string const& aggs,
                                                              cudf::column_view v0,
                                                              cudf::column_view v1)
{
  std::vector<cudf::groupby::aggregation_request> requests;
  auto& r0  = requests.emplace_back();
  r0.values = v0;
  r0.aggregations.push_back(cudf::make_sum_aggregation<cudf::groupby_aggregation>());
  if (aggs == "multi") {
    r0.aggregations.push_back(cudf::make_min_aggregation<cudf::groupby_aggregation>());
    r0.aggregations.push_back(cudf::make_max_aggregation<cudf::groupby_aggregation>());
    r0.aggregations.push_back(cudf::make_mean_aggregation<cudf::groupby_aggregation>());
    r0.aggregations.push_back(
      cudf::make_count_aggregation<cudf::groupby_aggregation>(cudf::null_policy::INCLUDE));
    auto& r1  = requests.emplace_back();
    r1.values = v1;
    r1.aggregations.push_back(cudf::make_sum_aggregation<cudf::groupby_aggregation>());
  }
  return requests;
}

}  // namespace

// Apples-to-apples comparison of hash groupby and direct groupby on input that satisfies
// `direct_aggregate`'s preconditions: a single UINT32 key column with all values in
// [0, capacity). `occupancy_pct` percent of the slots are hit, spread evenly over the key range.
void nvbench_direct_groupby(nvbench::state& state)
{
  auto const num_rows      = static_cast<cudf::size_type>(state.get_int64("num_rows"));
  auto const capacity      = static_cast<std::uint32_t>(state.get_int64("capacity"));
  auto const occupancy_pct = state.get_int64("occupancy_pct");
  auto const null_pct      = state.get_int64("null_pct");
  auto const algorithm     = state.get_string("algorithm");
  auto const aggs          = state.get_string("aggs");
  auto const value_type    = state.get_string("value_type");

  if (static_cast<std::int64_t>(capacity) > static_cast<std::int64_t>(num_rows)) {
    state.skip("capacity larger than num_rows");
    return;
  }
  // Hash groupby has no decimal128 MIN/MAX and falls back to sort groupby
  if (value_type == "decimal128" and aggs == "multi") {
    state.skip("decimal128 MIN/MAX is not hash-based");
    return;
  }

  auto const num_distinct =
    std::max<std::uint64_t>(1, static_cast<std::uint64_t>(capacity) * occupancy_pct / 100);
  auto const stride = static_cast<std::uint32_t>(capacity / num_distinct);

  auto keys = cudf::make_numeric_column(
    cudf::data_type{cudf::type_id::UINT32}, num_rows, cudf::mask_state::UNALLOCATED);
  thrust::tabulate(thrust::device,
                   keys->mutable_view().begin<std::uint32_t>(),
                   keys->mutable_view().end<std::uint32_t>(),
                   dense_key_fn{num_distinct, stride});

  auto const value_type_id =
    value_type == "decimal128" ? cudf::type_id::DECIMAL128 : cudf::type_id::FLOAT64;
  data_profile profile = data_profile_builder().cardinality(0).distribution(
    cudf::type_id::FLOAT64, distribution_id::UNIFORM, 0, 1000);
  profile.set_distribution_params(
    cudf::type_id::DECIMAL128, distribution_id::UNIFORM, 0, 100'000, numeric::scale_type{-2});
  if (null_pct > 0) {
    profile.set_null_probability(null_pct / 100.0);
  } else {
    profile.set_null_probability(std::nullopt);
  }
  auto const v0 = create_random_column(value_type_id, row_count{num_rows}, profile);
  auto const v1 = create_random_column(value_type_id, row_count{num_rows}, profile);

  auto const requests  = make_requests(aggs, v0->view(), v1->view());
  auto const keys_view = keys->view();
  auto const path      = [&] {
    if (algorithm == "direct_shmem") { return cudf::groupby::detail::direct_aggregate_path::SHARED_MEMORY; }
    if (algorithm == "direct_global") { return cudf::groupby::detail::direct_aggregate_path::GLOBAL_MEMORY; }
    return cudf::groupby::detail::direct_aggregate_path::AUTO;
  }();
  auto const run_direct = [&] {
    return cudf::groupby::detail::direct_aggregate(keys_view,
                                                   requests,
                                                   capacity,
                                                   path,
                                                   cudf::get_default_stream(),
                                                   cudf::get_current_device_resource_ref());
  };

  if (algorithm != "hash") {
    try {
      auto const warmup = run_direct();
    } catch (std::invalid_argument const&) {
      state.skip("path not applicable");
      return;
    }
  }

  state.set_cuda_stream(nvbench::make_cuda_stream_view(cudf::get_default_stream().get()));
  state.add_element_count(num_rows, "rows");
  state.add_global_memory_reads<nvbench::int8_t>(
    keys->alloc_size() + v0->alloc_size() + (aggs == "multi" ? v1->alloc_size() : 0));

  auto const mem_stats_logger = cudf::memory_stats_logger();
  if (algorithm == "hash") {
    state.exec(nvbench::exec_tag::sync, [&](nvbench::launch&) {
      auto gb     = cudf::groupby::groupby{cudf::table_view{{keys_view}}};
      auto result = gb.aggregate(requests);
    });
  } else {
    state.exec(nvbench::exec_tag::sync, [&](nvbench::launch&) { auto result = run_direct(); });
  }
  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");
}

NVBENCH_BENCH(nvbench_direct_groupby)
  .set_name("direct_groupby")
  .add_string_axis("algorithm", {"hash", "direct", "direct_shmem", "direct_global"})
  .add_string_axis("aggs", {"sum", "multi"})
  .add_string_axis("value_type", {"float64", "decimal128"})
  .add_int64_axis("num_rows", {10'000'000, 100'000'000})
  .add_int64_axis("capacity", {4, 64, 1'024, 16'384, 262'144, 4'194'304, 67'108'864})
  .add_int64_axis("occupancy_pct", {100, 10})
  .add_int64_axis("null_pct", {0});
