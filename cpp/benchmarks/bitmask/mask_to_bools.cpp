/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/null_mask.hpp>
#include <cudf/transform.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <nvbench/nvbench.cuh>

static void bench_mask_to_bools(nvbench::state& state)
{
  auto const num_rows = static_cast<cudf::size_type>(state.get_int64("num_rows"));
  auto const mask     = cudf::create_null_mask(num_rows, cudf::mask_state::ALL_VALID);
  auto const bitmask  = reinterpret_cast<cudf::bitmask_type const*>(mask.data());

  state.set_cuda_stream(nvbench::make_cuda_stream_view(cudf::get_default_stream().get()));
  state.add_global_memory_reads<nvbench::int8_t>(cudf::bitmask_allocation_size_bytes(num_rows));
  state.add_global_memory_writes<bool>(num_rows);

  state.exec(nvbench::exec_tag::sync,
             [&](nvbench::launch& launch) { cudf::mask_to_bools(bitmask, 0, num_rows); });
}

NVBENCH_BENCH(bench_mask_to_bools)
  .set_name("mask_to_bools")
  .add_int64_power_of_two_axis("num_rows", {10, 16, 20, 24, 28});
