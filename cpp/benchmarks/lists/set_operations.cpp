/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <benchmarks/common/generate_input.hpp>
#include <benchmarks/common/memory_stats.hpp>

#include <cudf/lists/set_operations.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <nvbench/nvbench.cuh>

namespace {

constexpr auto max_list_size = 20;

auto generate_random_lists(cudf::size_type num_rows, cudf::size_type depth, double null_freq)
{
  auto builder =
    data_profile_builder()
      .cardinality(0)
      .distribution(cudf::type_id::LIST, distribution_id::UNIFORM, 0, max_list_size)
      .list_depth(depth)
      .null_probability(null_freq > 0 ? std::optional<double>{null_freq} : std::nullopt);

  auto data_table =
    create_random_table({cudf::type_id::LIST}, row_count{num_rows}, data_profile{builder});
  return std::move(data_table->release().front());
}

template <typename BenchFuncPtr>
void nvbench_set_op(nvbench::state& state, BenchFuncPtr bfunc)
{
  auto const num_rows  = static_cast<cudf::size_type>(state.get_int64("num_rows"));
  auto const depth     = static_cast<cudf::size_type>(state.get_int64("depth"));
  auto const null_freq = state.get_float64("null_frequency");

  auto const lhs = generate_random_lists(num_rows, depth, null_freq);
  auto const rhs = generate_random_lists(num_rows, depth, null_freq);

  state.set_cuda_stream(nvbench::make_cuda_stream_view(cudf::get_default_stream().value()));
  auto const mem_stats_logger = cudf::memory_stats_logger();
  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
    bfunc(cudf::lists_column_view{*lhs},
          cudf::lists_column_view{*rhs},
          cudf::null_equality::EQUAL,
          cudf::nan_equality::ALL_EQUAL,
          cudf::get_default_stream(),
          cudf::get_current_device_resource_ref());
  });
  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");
}

}  // namespace

void nvbench_have_overlap(nvbench::state& state)
{
  nvbench_set_op(state, &cudf::lists::have_overlap);
}

void nvbench_intersect_distinct(nvbench::state& state)
{
  nvbench_set_op(state, &cudf::lists::intersect_distinct);
}

NVBENCH_BENCH(nvbench_have_overlap)
  .set_name("have_overlap")
  .add_int64_power_of_two_axis("num_rows", {10, 13, 16})
  .add_int64_axis("depth", {1, 4})
  .add_float64_axis("null_frequency", {0, 0.2, 0.8});

NVBENCH_BENCH(nvbench_intersect_distinct)
  .set_name("intersect_distinct")
  .add_int64_power_of_two_axis("num_rows", {10, 13, 16})
  .add_int64_axis("depth", {1, 4})
  .add_float64_axis("null_frequency", {0, 0.2, 0.8});
