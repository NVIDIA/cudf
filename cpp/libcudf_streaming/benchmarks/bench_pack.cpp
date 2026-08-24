/**
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "utils/random_data.hpp"

#include <cudf/contiguous_split.hpp>
#include <cudf/table/table.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/cuda_device.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/mr/cuda_async_memory_resource.hpp>
#include <rmm/mr/pool_memory_resource.hpp>

#include <benchmark/benchmark.h>
#include <rapidsmpf/memory/cuda_memcpy_async.hpp>
#include <rapidsmpf/memory/pinned_memory_resource.hpp>
#include <rapidsmpf/utils/misc.hpp>

#include <algorithm>
#include <cstdint>

constexpr std::size_t MB = 1024 * 1024;

/**
 * @brief Runs the cudf::pack benchmark
 * @param state The benchmark state
 * @param table_size_mb The size of the table in MB
 * @param table_mr The memory resource for the table
 * @param pack_mr The memory resource for the packed data
 * @param stream The CUDA stream to use
 */
void run_pack(benchmark::State& state,
              std::size_t table_size_mb,
              rmm::device_async_resource_ref table_mr,
              rmm::device_async_resource_ref pack_mr,
              rmm::cuda_stream_view stream)
{
  auto const table_size_bytes = table_size_mb * MB;

  // Calculate number of rows for a single-column table of the desired size
  auto const nrows =
    rapidsmpf::safe_cast<cudf::size_type>(table_size_bytes / sizeof(random_data_t));
  auto table = random_table(1, nrows, 0, 1000, stream, table_mr);

  // Warm up
  auto warm_up = cudf::pack(table.view(), stream, pack_mr);
  stream.synchronize();

  for (auto _ : state) {
    auto packed = cudf::pack(table.view(), stream, pack_mr);
    benchmark::DoNotOptimize(packed);
    stream.synchronize();
  }

  state.SetBytesProcessed(static_cast<std::int64_t>(state.iterations()) *
                          static_cast<std::int64_t>(table_size_bytes));
  state.counters["table_size_mb"] = static_cast<double>(table_size_mb);
  state.counters["num_rows"]      = nrows;
}

/**
 * @brief Benchmark for cudf::pack with device memory
 */
static void BM_Pack_device(benchmark::State& state)
{
  auto const table_size_mb = static_cast<std::size_t>(state.range(0));

  rmm::cuda_stream_view stream = rmm::cuda_stream_default;

  // Create memory resources
  rmm::mr::pool_memory_resource pool_mr{rmm::mr::cuda_async_memory_resource{},
                                        rmm::percent_of_free_device_memory(40)};
  run_pack(state, table_size_mb, pool_mr, pool_mr, stream);
}

/**
 * @brief Benchmark for cudf::pack with pinned memory
 */
static void BM_Pack_pinned(benchmark::State& state)
{
  state.SkipWithMessage("Skipping until cudf#20886 is fixed");
  /* if (!rapidsmpf::is_pinned_memory_resources_supported()) {
    state.SkipWithMessage("Pinned memory resources are not supported");
    return;
  }

  auto const table_size_mb = static_cast<std::size_t>(state.range(0));

  rmm::cuda_stream_view stream = rmm::cuda_stream_default;

  // Create memory resources
  rmm::mr::pool_memory_resource pool_mr{
    rmm::mr::cuda_async_memory_resource{}, rmm::percent_of_free_device_memory(40)
  };

  run_pack(state, table_size_mb, pool_mr, pinned_mr, stream); */
}

/**
 * @brief Runs the cudf::chunked_pack benchmark
 * @param state The benchmark state
 * @param bounce_buffer_size The size of the bounce buffer in bytes
 * @param table_size The size of the table in bytes
 * @param table_mr The memory resource for the table
 * @param pack_mr The memory resource for the packed data
 * @param stream The CUDA stream to use
 */
void run_chunked_pack(benchmark::State& state,
                      std::size_t bounce_buffer_size,
                      std::size_t table_size,
                      rmm::device_async_resource_ref table_mr,
                      rmm::device_async_resource_ref pack_mr,
                      rmm::cuda_stream_view stream)
{
  // Calculate number of rows for a single-column table of the desired size
  auto const nrows = rapidsmpf::safe_cast<cudf::size_type>(table_size / sizeof(random_data_t));
  auto table       = random_table(1, nrows, 0, 1000, stream, table_mr);

  // Create the chunked_pack instance to get total output size
  std::size_t total_size;
  {
    cudf::chunked_pack packer(table.view(), bounce_buffer_size, stream, table_mr);
    total_size = packer.get_total_contiguous_size();
  }

  // Allocate bounce buffer and destination buffer using the pack_mr
  rmm::device_buffer bounce_buffer(bounce_buffer_size, stream, pack_mr);
  rmm::device_buffer destination(total_size, stream, pack_mr);

  auto run_packer = [&] {
    cudf::chunked_pack packer(table.view(), bounce_buffer_size, stream, pack_mr);

    std::size_t offset = 0;
    while (packer.has_next()) {
      auto const bytes_copied = packer.next(cudf::device_span<std::uint8_t>(
        static_cast<std::uint8_t*>(bounce_buffer.data()), bounce_buffer_size));
      RAPIDSMPF_CUDA_TRY(
        rapidsmpf::cuda_memcpy_async(static_cast<std::uint8_t*>(destination.data()) + offset,
                                     bounce_buffer.data(),
                                     bytes_copied,
                                     stream));
      offset += bytes_copied;
    }
  };

  {
    run_packer();
    stream.synchronize();
  }

  for (auto _ : state) {
    run_packer();
    benchmark::DoNotOptimize(destination);
    stream.synchronize();
  }

  state.SetBytesProcessed(static_cast<std::int64_t>(state.iterations()) *
                          static_cast<std::int64_t>(table_size));
  state.counters["table_size_mb"] = static_cast<double>(table_size) / static_cast<double>(MB);
  state.counters["num_rows"]      = nrows;
  state.counters["bounce_buffer_mb"] =
    static_cast<double>(bounce_buffer_size) / static_cast<double>(MB);
}

/**
 * @brief Benchmark for cudf::chunked_pack with device memory
 */
static void BM_ChunkedPack_device(benchmark::State& state)
{
  auto const table_size_mb    = static_cast<std::size_t>(state.range(0));
  auto const table_size_bytes = table_size_mb * MB;

  // Bounce buffer size: max(1MB, table_size / 10)
  auto const bounce_buffer_size = std::max(MB, table_size_bytes / 10);

  rmm::cuda_stream_view stream = rmm::cuda_stream_default;

  rmm::mr::pool_memory_resource pool_mr{rmm::mr::cuda_async_memory_resource{},
                                        rmm::percent_of_free_device_memory(40)};

  run_chunked_pack(state, bounce_buffer_size, table_size_bytes, pool_mr, pool_mr, stream);
}

/**
 * @brief Benchmark for cudf::chunked_pack pinned memory
 */
static void BM_ChunkedPack_pinned(benchmark::State& state)
{
  state.SkipWithMessage("Skipping until cudf#20886 is fixed");
  /*     if (!rapidsmpf::is_pinned_memory_resources_supported()) {
    state.SkipWithMessage("Pinned memory resources are not supported");
    return;
  }

  auto const table_size_mb = static_cast<std::size_t>(state.range(0));
  auto const table_size_bytes = table_size_mb * MB;

  // Bounce buffer size: max(1MB, table_size / 10)
  auto const bounce_buffer_size = std::max(MB, table_size_bytes / 10);

  rmm::cuda_stream_view stream = rmm::cuda_stream_default;

  rmm::mr::pool_memory_resource pool_mr{
    rmm::mr::cuda_async_memory_resource{}, rmm::percent_of_free_device_memory(40)
  };
  rapidsmpf::PinnedMemoryResource pinned_mr;

  run_chunked_pack(
    state, bounce_buffer_size, table_size_bytes, pool_mr, pinned_mr, stream
  ); */
}

// Custom argument generator for the benchmark
void PackArguments(benchmark::Benchmark* b)
{
  // Test different table sizes in MB (minimum 1MB as requested)
  for (auto size_mb : {1, 10, 100, 500, 1000, 2000, 4000}) {
    b->Args({size_mb});
  }
}

// Register the benchmarks
BENCHMARK(BM_Pack_device)->Apply(PackArguments)->UseRealTime()->Unit(benchmark::kMillisecond);
BENCHMARK(BM_Pack_pinned)->Apply(PackArguments)->UseRealTime()->Unit(benchmark::kMillisecond);

BENCHMARK(BM_ChunkedPack_device)
  ->Apply(PackArguments)
  ->UseRealTime()
  ->Unit(benchmark::kMillisecond);

BENCHMARK(BM_ChunkedPack_pinned)
  ->Apply(PackArguments)
  ->UseRealTime()
  ->Unit(benchmark::kMillisecond);

/**
 * @brief Benchmark for cudf::chunked_pack in device memory varying the bounce buffer size
 * and keeping table size fixed at 1GB
 */
static void BM_ChunkedPack_fixed_table_device(benchmark::State& state)
{
  auto const bounce_buffer_size          = static_cast<std::size_t>(state.range(0)) * MB;
  constexpr std::size_t table_size_bytes = 1024 * MB;

  rmm::cuda_stream_view stream = rmm::cuda_stream_default;

  // Create memory resources
  rmm::mr::pool_memory_resource pool_mr{rmm::mr::cuda_async_memory_resource{},
                                        rmm::percent_of_free_device_memory(40)};

  run_chunked_pack(state, bounce_buffer_size, table_size_bytes, pool_mr, pool_mr, stream);
}

/**
 * @brief Benchmark for cudf::chunked_pack in pinned memory varying the bounce buffer size
 * and keeping table size fixed at 1GB
 */
static void BM_ChunkedPack_fixed_table_pinned(benchmark::State& state)
{
  state.SkipWithMessage("Skipping until cudf#20886 is fixed");
  /* if (!rapidsmpf::is_pinned_memory_resources_supported()) {
    state.SkipWithMessage("Pinned memory resources are not supported");
    return;
  }

  auto const bounce_buffer_size = static_cast<std::size_t>(state.range(0)) * MB;
  constexpr std::size_t table_size_bytes = 1024 * MB;

  rmm::cuda_stream_view stream = rmm::cuda_stream_default;

  rmm::mr::pool_memory_resource pool_mr{
    rmm::mr::cuda_async_memory_resource{}, rmm::percent_of_free_device_memory(40)
  };
  rapidsmpf::PinnedMemoryResource pinned_mr;

  run_chunked_pack(
    state, bounce_buffer_size, table_size_bytes, pool_mr, pinned_mr, stream
  ); */
}

// Custom argument generator for the benchmark
void ChunkedPackArguments(benchmark::Benchmark* b)
{
  // Test different table sizes in MB (minimum 1MB as requested)
  for (auto bounce_buf_sz_mb : {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024}) {
    b->Args({bounce_buf_sz_mb});
  }
}

BENCHMARK(BM_ChunkedPack_fixed_table_device)
  ->Apply(ChunkedPackArguments)
  ->UseRealTime()
  ->Unit(benchmark::kMillisecond);

BENCHMARK(BM_ChunkedPack_fixed_table_pinned)
  ->Apply(ChunkedPackArguments)
  ->UseRealTime()
  ->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN();
