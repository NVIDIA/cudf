/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <benchmarks/common/memory_stats.hpp>

#include <cudf/scalar/scalar.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <nvbench/nvbench.cuh>

#include <cstdint>
#include <string>

namespace {

void numeric_scalar_construction(nvbench::state& state)
{
  auto const is_valid = static_cast<bool>(state.get_int64("valid"));

  auto const mem_stats_logger = cudf::memory_stats_logger();
  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
    auto const stream           = rmm::cuda_stream_view{launch.get_stream()};
    [[maybe_unused]] auto scalar = cudf::numeric_scalar<int64_t>{
      42, is_valid, stream, cudf::get_current_device_resource_ref()};
  });
  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");
}

void string_scalar_construction(nvbench::state& state)
{
  auto const num_bytes = state.get_int64("num_bytes");
  auto const value     = std::string(static_cast<std::size_t>(num_bytes), 'a');

  auto const mem_stats_logger = cudf::memory_stats_logger();
  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
    auto const stream           = rmm::cuda_stream_view{launch.get_stream()};
    [[maybe_unused]] auto scalar = cudf::string_scalar{
      value, true, stream, cudf::get_current_device_resource_ref()};
  });
  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");
}

}  // namespace

NVBENCH_BENCH(numeric_scalar_construction)
  .set_name("numeric_scalar_construction")
  .add_int64_axis("valid", {0, 1});

NVBENCH_BENCH(string_scalar_construction)
  .set_name("string_scalar_construction")
  .add_int64_power_of_two_axis("num_bytes", {0, 4, 8, 12, 16, 20});
