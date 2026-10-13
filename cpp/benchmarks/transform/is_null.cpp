/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <benchmarks/common/generate_input.hpp>

#include <cudf/null_mask.hpp>
#include <cudf/unary.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <nvbench/nvbench.cuh>

static void bench_is_null(nvbench::state& state)
{
  auto const num_rows = static_cast<cudf::size_type>(state.get_int64("num_rows"));
  auto const api      = state.get_string("api");

  auto const type            = api == "is_nan" ? cudf::type_id::FLOAT32 : cudf::type_id::INT32;
  data_profile const profile = data_profile_builder().null_probability(0.1).distribution(
    type, distribution_id::UNIFORM, 0, 100);
  auto const input = create_random_column(type, row_count{num_rows}, profile);

  auto const stream = cudf::get_default_stream();
  state.set_cuda_stream(nvbench::make_cuda_stream_view(stream.get()));
  // is_null only reads the null mask; is_nan also reads the data
  state.add_global_memory_reads<nvbench::int8_t>(cudf::bitmask_allocation_size_bytes(num_rows));
  if (api == "is_nan") { state.add_global_memory_reads<float>(num_rows); }
  state.add_global_memory_writes<bool>(num_rows);

  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch&) {
    if (api == "is_nan") {
      cudf::is_nan(input->view(), stream);
    } else {
      cudf::is_null(input->view(), stream);
    }
  });
}

NVBENCH_BENCH(bench_is_null)
  .set_name("is_null")
  .add_int64_power_of_two_axis("num_rows", {20, 24, 27})
  .add_string_axis("api", {"is_null", "is_nan"});
