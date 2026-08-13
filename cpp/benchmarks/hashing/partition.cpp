/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <benchmarks/common/generate_input.hpp>
#include <benchmarks/common/memory_stats.hpp>

#include <cudf/partitioning.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <nvbench/nvbench.cuh>

#include <algorithm>
#include <numeric>

static void bench_hash_partition(nvbench::state& state)
{
  using T = double;

  auto const num_rows       = static_cast<cudf::size_type>(state.get_int64("num_rows"));
  auto const num_cols       = static_cast<cudf::size_type>(state.get_int64("num_cols"));
  auto const num_partitions = static_cast<cudf::size_type>(state.get_int64("num_partitions"));

  // Create owning columns
  auto input_table = create_sequence_table(cycle_dtypes({cudf::type_to_id<T>()}, num_cols),
                                           row_count{static_cast<cudf::size_type>(num_rows)});
  auto input       = cudf::table_view(*input_table);

  auto columns_to_hash = std::vector<cudf::size_type>(num_cols);
  std::iota(columns_to_hash.begin(), columns_to_hash.end(), 0);

  // Set up CUDA stream for nvbench
  auto stream = cudf::get_default_stream();
  state.set_cuda_stream(nvbench::make_cuda_stream_view(stream.value()));

  auto const mem_stats_logger = cudf::memory_stats_logger();

  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch&) {
    auto output = cudf::hash_partition(input, columns_to_hash, num_partitions);
  });

  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");

  // Set memory usage statistics for nvbench
  state.add_global_memory_reads<T>(static_cast<int64_t>(num_rows) * num_cols);
  state.add_global_memory_writes<T>(static_cast<int64_t>(num_rows) * num_cols);
  state.add_global_memory_writes<cudf::size_type>(num_partitions);
}

NVBENCH_BENCH(bench_hash_partition)
  .set_name("hash_partition")
  .add_int64_axis("num_rows", {1 << 17, 1 << 18, 1 << 19, 1 << 20, 1 << 21})
  .add_int64_axis("num_cols", {1, 16, 256})
  .add_int64_axis("num_partitions", {64, 128, 256, 512, 1024});
