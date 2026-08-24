/**
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/table/table.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/error.hpp>

#include <cudf_streaming/partition_utils.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/mr/cuda_memory_resource.hpp>
#include <rmm/mr/pool_memory_resource.hpp>

#include <benchmark/benchmark.h>
#include <rapidsmpf/utils/misc.hpp>

#include <memory>
#include <vector>

// Helper function to create a table with a single int column
std::unique_ptr<cudf::table> create_int_table(cudf::size_type num_rows,
                                              rmm::cuda_stream_view stream)
{
  auto data =
    rmm::device_buffer(rapidsmpf::safe_cast<std::size_t>(num_rows) * sizeof(std::int32_t), stream);
  auto validity = rmm::device_buffer(0, stream);  // No nulls

  auto column = std::make_unique<cudf::column>(
    cudf::data_type{cudf::type_id::INT32}, num_rows, std::move(data), std::move(validity), 0);

  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(std::move(column));
  return std::make_unique<cudf::table>(std::move(columns));
}

static void BM_PartitionAndPack(benchmark::State& state)
{
  std::int64_t const local_size = std::int64_t{state.range(1)} * 1000000;
  int num_rows                  = int(local_size / std::int64_t{sizeof(std::int32_t)});

  int const num_partitions = state.range(1);

  rmm::cuda_stream_view stream = rmm::cuda_stream_default;

  // Get total GPU memory
  cudaDeviceProp prop;
  CUDF_CUDA_TRY(cudaGetDeviceProperties(&prop, 0));
  std::size_t total_memory = prop.totalGlobalMem;

  // Calculate 50% of GPU memory
  auto pool_size = static_cast<std::size_t>(total_memory * 0.5);

  // Create a pool memory resource with 50% of GPU memory
  rmm::mr::pool_memory_resource pool_mr{rmm::mr::cuda_memory_resource{}, pool_size};
  auto br = rapidsmpf::BufferResource::create(pool_mr);

  // Create input table
  auto table = create_int_table(num_rows, stream);

  // Columns to hash (just the first column)
  std::vector<cudf::size_type> columns_to_hash{0};

  for (auto _ : state) {
    auto pack_partitions = cudf_streaming::partition_and_pack(*table,
                                                              columns_to_hash,
                                                              num_partitions,
                                                              cudf::hash_id::HASH_MURMUR3,
                                                              cudf::DEFAULT_HASH_SEED,
                                                              stream,
                                                              br.get());
    benchmark::DoNotOptimize(pack_partitions);
    CUDF_CUDA_TRY(cudaStreamSynchronize(stream));
  }

  // Set metrics
  state.SetBytesProcessed(state.iterations() * local_size);
  state.counters["num_rows"]     = num_rows;
  state.counters["total_nparts"] = num_partitions;
  state.counters["splits"]       = num_partitions;
}

static void BM_PartitionAndPackCurrentImpl(benchmark::State& state)
{
  int const nranks              = state.range(0);
  std::int64_t const local_size = std::int64_t{state.range(1)} * 1000000;
  int const num_partitions      = state.range(2);

  int total_npartitions = nranks * num_partitions;
  int num_rows =
    int(local_size / std::int64_t{sizeof(std::int32_t)} / std::int64_t{num_partitions});

  rmm::cuda_stream_view stream = rmm::cuda_stream_default;

  // Get total GPU memory
  cudaDeviceProp prop;
  CUDF_CUDA_TRY(cudaGetDeviceProperties(&prop, 0));
  std::size_t total_memory = prop.totalGlobalMem;

  // Calculate 50% of GPU memory
  auto pool_size = static_cast<std::size_t>(total_memory * 0.5);

  // Create a pool memory resource with 50% of GPU memory
  rmm::mr::pool_memory_resource pool_mr{rmm::mr::cuda_memory_resource{}, pool_size};
  auto br = rapidsmpf::BufferResource::create(pool_mr);

  // Create input table
  auto table = create_int_table(num_rows, stream);

  // Columns to hash (just the first column)
  std::vector<cudf::size_type> columns_to_hash{0};

  for (auto _ : state) {
    for (int i = 0; i < num_partitions; i++) {
      auto pack_partitions = cudf_streaming::partition_and_pack(*table,
                                                                columns_to_hash,
                                                                total_npartitions,
                                                                cudf::hash_id::HASH_MURMUR3,
                                                                cudf::DEFAULT_HASH_SEED,
                                                                stream,
                                                                br.get());
      benchmark::DoNotOptimize(pack_partitions);
    }
    CUDF_CUDA_TRY(cudaStreamSynchronize(stream));
  }

  // Set metrics
  state.SetBytesProcessed(state.iterations() * local_size);
  state.counters["num_rows"]     = num_rows;
  state.counters["total_nparts"] = total_npartitions;
  state.counters["splits"]       = total_npartitions * num_partitions;
}

// Custom argument generator for the benchmark
void CustomArguments(benchmark::Benchmark* b)
{
  // Test different combinations of table sizes and partitions
  for (auto nranks : {4}) {
    for (int size_mb : {4000}) {
      for (auto partitions : {2, 8, 32, 128, 512, 1024}) {
        b->Args({nranks, size_mb, partitions});
      }
    }
  }
}

// Register the benchmark with custom arguments
BENCHMARK(BM_PartitionAndPackCurrentImpl)
  ->Apply(CustomArguments)
  ->UseRealTime()
  ->Unit(benchmark::kMillisecond);

// Register the benchmark with custom arguments
BENCHMARK(BM_PartitionAndPack)
  ->Args({4000, 2})
  ->Args({4000, 8})
  ->Args({4000, 32})
  ->Args({4000, 128})
  ->Args({4000, 512})
  ->Args({4000, 1024})
  ->UseRealTime()
  ->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN();
