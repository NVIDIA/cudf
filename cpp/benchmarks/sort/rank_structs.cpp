/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "rank_types_common.hpp"

#include <benchmarks/common/generate_nested_types.hpp>
#include <benchmarks/common/memory_stats.hpp>

#include <cudf/sorting.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <nvbench/nvbench.cuh>

template <cudf::rank_method method>
void nvbench_rank_structs(nvbench::state& state, nvbench::type_list<nvbench::enum_type<method>>)
{
  auto const table = create_structs_data(state);

  bool const nulls{static_cast<bool>(state.get_int64("Nulls"))};
  auto const mem_stats_logger = cudf::memory_stats_logger();

  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
    cudf::rank(table->view().column(0),
               method,
               cudf::order::ASCENDING,
               nulls ? cudf::null_policy::INCLUDE : cudf::null_policy::EXCLUDE,
               cudf::null_order::AFTER,
               false,
               cudf::get_default_stream(),
               cudf::get_current_device_resource_ref());
  });
  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");
}

NVBENCH_BENCH_TYPES(nvbench_rank_structs, NVBENCH_TYPE_AXES(methods))
  .set_name("rank_structs")
  .add_int64_power_of_two_axis("NumRows", {10, 18, 26})
  .add_int64_axis("Depth", {1, 8})
  .add_int64_axis("Nulls", {0, 1});
