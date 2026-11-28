/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <benchmarks/common/generate_input.hpp>
#include <benchmarks/common/memory_stats.hpp>

#include <cudf/strings/strings_column_view.hpp>

#include <nvtext/minhash.hpp>

#include <nvbench/nvbench.cuh>

static void bench_minhash(nvbench::state& state)
{
  auto const num_rows   = static_cast<cudf::size_type>(state.get_int64("num_rows"));
  auto const row_width  = static_cast<cudf::size_type>(state.get_int64("row_width"));
  auto const hash_width = static_cast<cudf::size_type>(state.get_int64("hash_width"));
  auto const parameters = static_cast<cudf::size_type>(state.get_int64("parameters"));
  auto const base64     = state.get_int64("hash_type") == 64;

  data_profile const strings_profile = data_profile_builder().distribution(
    cudf::type_id::STRING, distribution_id::NORMAL, 0, row_width);
  auto const strings_table =
    create_random_table({cudf::type_id::STRING}, row_count{num_rows}, strings_profile);
  cudf::strings_column_view input(strings_table->view().column(0));

  data_profile const param_profile = data_profile_builder().no_validity().distribution(
    cudf::type_to_id<cudf::hash_value_type>(),
    distribution_id::NORMAL,
    0u,
    std::numeric_limits<cudf::hash_value_type>::max());
  auto const param_type = base64 ? cudf::type_id::UINT64 : cudf::type_id::UINT32;
  auto const param_table =
    create_random_table({param_type, param_type}, row_count{parameters}, param_profile);
  auto const parameters_a = param_table->view().column(0);
  auto const parameters_b = param_table->view().column(1);

  state.set_cuda_stream(nvbench::make_cuda_stream_view(cudf::get_default_stream().get()));

  state.add_global_memory_reads<nvbench::int8_t>(strings_table->alloc_size());
  // output is a hash value per parameter for each row
  auto const hash_size = base64 ? sizeof(uint64_t) : sizeof(uint32_t);
  state.add_global_memory_writes<nvbench::int8_t>(static_cast<std::size_t>(num_rows) * parameters *
                                                  hash_size);

  auto const mem_stats_logger = cudf::memory_stats_logger();
  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
    auto result = base64 ? nvtext::minhash64(input, 0, parameters_a, parameters_b, hash_width)
                         : nvtext::minhash(input, 0, parameters_a, parameters_b, hash_width);
  });

  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");
}

NVBENCH_BENCH(bench_minhash)
  .set_name("minhash")
  .add_int64_axis("num_rows", {15000, 30000, 60000})
  .add_int64_axis("row_width", {6000, 28000, 50000})
  .add_int64_axis("hash_width", {12, 24})
  .add_int64_axis("parameters", {26, 260})
  .add_int64_axis("hash_type", {32, 64});

// fewer rows with much longer strings
NVBENCH_BENCH(bench_minhash)
  .set_name("minhash_wide")
  .add_int64_axis("num_rows", {16, 256})
  .add_int64_axis("row_width", {1000000, 4000000})
  .add_int64_axis("hash_width", {24})
  .add_int64_axis("parameters", {26, 260})
  .add_int64_axis("hash_type", {32, 64});
