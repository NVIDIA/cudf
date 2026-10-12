/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <benchmarks/common/generate_input.hpp>

#include <cudf/filling.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/transform.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <nvbench/nvbench.cuh>

static void bench_one_hot_encode(nvbench::state& state)
{
  auto const num_rows       = static_cast<cudf::size_type>(state.get_int64("num_rows"));
  auto const num_categories = static_cast<cudf::size_type>(state.get_int64("num_categories"));

  data_profile const profile = data_profile_builder().no_validity().distribution(
    cudf::type_id::INT32, distribution_id::UNIFORM, 0, num_categories - 1);
  auto const input      = create_random_column(cudf::type_id::INT32, row_count{num_rows}, profile);
  auto const categories = cudf::sequence(
    num_categories, cudf::numeric_scalar<int32_t>(0), cudf::numeric_scalar<int32_t>(1));

  auto stream = cudf::get_default_stream();
  state.set_cuda_stream(nvbench::make_cuda_stream_view(stream.get()));
  state.add_global_memory_reads<int32_t>(num_rows);
  state.add_global_memory_writes<bool>(static_cast<int64_t>(num_rows) * num_categories);

  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
    cudf::one_hot_encode(input->view(), categories->view(), stream);
  });
}

NVBENCH_BENCH(bench_one_hot_encode)
  .set_name("one_hot_encode")
  .add_int64_power_of_two_axis("num_rows", {16, 20, 24})
  .add_int64_axis("num_categories", {4, 16, 64});
