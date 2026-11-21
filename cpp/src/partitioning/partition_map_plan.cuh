// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "partition_policy.hpp"

#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/detail/utilities/cuda_memcpy.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/device_buffer.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <cub/block/block_scan.cuh>
#include <cub/device/device_histogram.cuh>
#include <cub/device/device_partition.cuh>
#include <cub/device/device_scan.cuh>
#include <cuda/atomic>
#include <cuda/iterator>
#include <cuda/std/bit>
#include <thrust/scan.h>

namespace cudf::detail::partition_map_plan {
using algorithm       = partition_policy::algorithm;
constexpr int threads = 256, tile_rows = 1'024, warps = 8, bins = 256;

// The same contiguous warp ordering as PR1: four 32-row rounds per warp.
// Histogram-only instantiation discards ranks; rank instantiation retains them in registers.
template <bool Rank, typename MapType>
__device__ void local_tile(MapType const* map,
                           int n,
                           int start,
                           int p,
                           int (&hist)[warps][bins],
                           int (&rank)[4],
                           int (&keys)[4])
{
  int lane = threadIdx.x % 32, w = threadIdx.x / 32;
  for (int q = lane; q < p; q += 32)
    hist[w][q] = 0;
  __syncwarp();
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    int row         = start + w * 128 + j * 32 + lane;
    bool valid      = row < n;
    unsigned active = __ballot_sync(0xffff'ffffu, valid);
    if (valid) {
      int q          = map[row];
      unsigned peers = __match_any_sync(active, q);
      int base       = hist[w][q];
      if constexpr (Rank) {
        keys[j] = q;
        rank[j] = base + cuda::std::popcount(peers & ((1u << lane) - 1u));
      }
      __syncwarp(active);
      if (lane == __ffs(peers) - 1) hist[w][q] = base + cuda::std::popcount(peers);
    }
    __syncwarp();
  }
  __syncthreads();
}

template <int Tiles, bool SaveRanks, bool PackBin = false, typename MapType>
CUDF_KERNEL void count_tiles(MapType const* map, int n, int p, uint16_t* saved, int32_t* counts)
{
  static_assert(!PackBin || tile_rows <= 32'768, "Binary packed ranks must fit in uint16_t");
  __shared__ int hist[warps][bins];
  int total = 0;
  for (int t = 0; t < Tiles; ++t) {
    int rank[4], keys[4];
    int start = (blockIdx.x * Tiles + t) * tile_rows;
    local_tile<SaveRanks>(map, n, start, p, hist, rank, keys);
    if (threadIdx.x < p) {
      int q = threadIdx.x, sum = 0;
#pragma unroll
      for (int w = 0; w < warps; ++w) {
        int c = hist[w][q];
        if constexpr (SaveRanks) hist[w][q] = sum;
        sum += c;
      }
      total += sum;
    }
    if constexpr (SaveRanks) {
      __syncthreads();
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        int i = start + (threadIdx.x / 32) * 128 + j * 32 + threadIdx.x % 32;
        if (i < n) {
          auto const local = hist[threadIdx.x / 32][keys[j]] + rank[j];
          // PackBin is used by the binary STRING plan: its 10-bit tile rank and
          // one-bit bin fit in uint16_t, avoiding a map read during byte copying.
          saved[i] = local + (PackBin ? keys[j] * tile_rows : 0);
        }
      }
    }
    __syncthreads();
  }
  if (threadIdx.x < p) counts[size_t(threadIdx.x) * gridDim.x + blockIdx.x] = total;
}

CUDF_KERNEL void get_offsets(int32_t const* counts, int blocks, int p, int n, int32_t* offsets)
{
  if (threadIdx.x < p) offsets[threadIdx.x] = counts[size_t(threadIdx.x) * blocks];
  if (threadIdx.x == 0) offsets[p] = n;
}

template <typename Sink, typename MapType>
CUDF_KERNEL void scatter_saved(MapType const* map,
                               uint16_t const* ranks,
                               int32_t const* offsets,
                               int n,
                               int blocks,
                               int p,
                               Sink sink)
{
  __shared__ int base[bins];
  if (threadIdx.x < p) base[threadIdx.x] = offsets[size_t(threadIdx.x) * blocks + blockIdx.x];
  __syncthreads();
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    int i = blockIdx.x * tile_rows + j * threads + threadIdx.x;
    if (i < n) sink(i, base[map[i]] + ranks[i]);
  }
}

template <typename Sink>
__device__ void emit_reordered(int const (&keys)[4],
                               int const (&rank)[4],
                               int (&hist)[warps][bins],
                               int const* base,
                               int next,
                               int p,
                               int n,
                               int start,
                               Sink sink)
{
  __shared__ typename cub::BlockScan<int, threads>::TempStorage scan_temp;
  __shared__ int bucket_begin[bins], source_rows[tile_rows], destinations[tile_rows];
  int count = threadIdx.x < p ? next - base[threadIdx.x] : 0, prefix = 0;
  cub::BlockScan<int, threads>(scan_temp).ExclusiveSum(count, prefix);
  bucket_begin[threadIdx.x] = prefix;
  __syncthreads();
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    int row = start + (threadIdx.x / 32) * 128 + j * 32 + threadIdx.x % 32;
    if (row < n) {
      int q = keys[j], local = hist[threadIdx.x / 32][q] + rank[j],
          position           = bucket_begin[q] + local;
      source_rows[position]  = row;
      destinations[position] = base[q] + local;
    }
  }
  __syncthreads();
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    int position = threadIdx.x + j * threads;
    if (position < min(tile_rows, n - start)) sink(source_rows[position], destinations[position]);
  }
  // The caller synchronizes before reusing the shared storage.
}

template <int Tiles, bool Reorder = false, typename Sink, typename MapType, typename Offsets>
CUDF_KERNEL void scatter_recompute(MapType const* map, Offsets offsets, int n, int p, Sink sink)
{
  __shared__ int hist[warps][bins], base[bins];
  if (threadIdx.x < p) base[threadIdx.x] = offsets[size_t(threadIdx.x) * gridDim.x + blockIdx.x];
  __syncthreads();
  for (int t = 0; t < Tiles; ++t) {
    int rank[4], keys[4];
    int start = (blockIdx.x * Tiles + t) * tile_rows;
    local_tile<true>(map, n, start, p, hist, rank, keys);
    int next = 0;
    if (threadIdx.x < p) {
      int q = threadIdx.x, sum = 0;
#pragma unroll
      for (int w = 0; w < warps; ++w) {
        int c      = hist[w][q];
        hist[w][q] = sum;
        sum += c;
      }
      next = base[q] + sum;
    }
    __syncthreads();
    if constexpr (Reorder)
      emit_reordered(keys, rank, hist, base, next, p, n, start, sink);
    else {
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        int i = start + (threadIdx.x / 32) * 128 + j * 32 + threadIdx.x % 32;
        if (i < n) {
          int q = keys[j];
          sink(i, base[q] + hist[threadIdx.x / 32][q] + rank[j]);
        }
      }
    }
    __syncthreads();
    if (threadIdx.x < p) base[threadIdx.x] = next;
    __syncthreads();
  }
}

template <typename Sink, typename MapType>
CUDF_KERNEL void scatter_reserve(MapType const* map, int n, int p, int32_t* cursors, Sink sink)
{
  __shared__ int hist[warps][bins], base[bins];
  int rank[4], keys[4];
  int start = blockIdx.x * tile_rows;
  local_tile<true>(map, n, start, p, hist, rank, keys);
  if (threadIdx.x < p) {
    int q = threadIdx.x, sum = 0;
#pragma unroll
    for (int w = 0; w < warps; ++w) {
      int c      = hist[w][q];
      hist[w][q] = sum;
      sum += c;
    }
    if (sum) {
      cuda::atomic_ref<int32_t, cuda::thread_scope_device> cursor{cursors[q]};
      base[q] = cursor.fetch_add(sum, cuda::memory_order_relaxed);
    }
  }
  __syncthreads();
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    int i = start + (threadIdx.x / 32) * 128 + j * 32 + threadIdx.x % 32;
    if (i < n) sink(i, base[keys[j]] + hist[threadIdx.x / 32][keys[j]] + rank[j]);
  }
}

template <typename Emit>
struct write_proxy {
  Emit sink;
  int destination;
  __device__ void operator=(int source) const { sink(source, destination); }
};
template <typename Emit>
struct make_write_proxy {
  Emit sink;
  __host__ __device__ write_proxy<Emit> operator()(int destination) const
  {
    return {sink, destination};
  }
};
template <typename Map>
struct is_zero {
  Map const* map;
  __device__ bool operator()(int row) const { return map[row] == 0; }
};

template <typename Map, typename Emit>
std::vector<int> binary(Map const* map, int n, Emit sink, cuda::stream_ref stream)
{
  rmm::device_uvector<int> selected(1, stream);
  auto source  = cuda::counting_iterator<int>{0};
  auto output  = cuda::transform_iterator(source, make_write_proxy<Emit>{sink});
  size_t bytes = 0;
  CUDF_CUDA_TRY(cub::DevicePartition::If(
    nullptr, bytes, source, output, selected.data(), n, is_zero<Map>{map}, stream.get()));
  rmm::device_buffer temporary(bytes, stream);
  CUDF_CUDA_TRY(cub::DevicePartition::If(
    temporary.data(), bytes, source, output, selected.data(), n, is_zero<Map>{map}, stream.get()));
  auto const count = cudf::detail::make_std_vector(selected, stream)[0];
  return {0, count, n};
}

inline void scan(int32_t* ptr, size_t count, cuda::stream_ref stream)
{
  // Same Thrust in-place scan policy as PR1.
  thrust::exclusive_scan(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                         ptr,
                         ptr + count,
                         ptr);
}

template <typename MapType>
inline void global_hist(MapType const* map, int n, int p, int32_t* offsets, cuda::stream_ref stream)
{
  size_t bytes = 0;
  CUDF_CUDA_TRY(cudaMemsetAsync(offsets + p, 0, sizeof(int), stream.get()));
  CUDF_CUDA_TRY(cub::DeviceHistogram::HistogramEven(
    nullptr, bytes, map, offsets, p + 1, 0, p, n, stream.get()));
  rmm::device_buffer tmp(bytes, stream);
  CUDF_CUDA_TRY(cub::DeviceHistogram::HistogramEven(
    tmp.data(), bytes, map, offsets, p + 1, 0, p, n, stream.get()));
  scan(offsets, p + 1, stream);
}

template <int Tiles, bool Reorder = false, typename Sink, typename MapType>
void recompute(
  MapType const* map, int n, int p, Sink sink, int32_t* offsets, cuda::stream_ref stream)
{
  int blocks = (n - 1) / (tile_rows * Tiles) + 1;
  rmm::device_uvector<int32_t> counts(size_t(p) * blocks, stream);
  count_tiles<Tiles, false>
    <<<blocks, threads, 0, stream.get()>>>(map, n, p, nullptr, counts.data());
  CUDF_CUDA_TRY(cudaPeekAtLastError());
  scan(counts.data(), counts.size(), stream);
  get_offsets<<<1, threads, 0, stream.get()>>>(counts.data(), blocks, p, n, offsets);
  scatter_recompute<Tiles, Reorder>
    <<<blocks, threads, 0, stream.get()>>>(map, counts.data(), n, p, sink);
  CUDF_CUDA_TRY(cudaPeekAtLastError());
}

template <typename MapType, typename Sink>
std::vector<size_type> run(MapType const* map,
                           size_type n,
                           size_type p,
                           algorithm method,
                           Sink sink,
                           bool reorder,
                           cuda::stream_ref stream)
{
  CUDF_EXPECTS(n > 0 && p > 0 && p <= bins, "Invalid block partition dimensions");
  if (method == algorithm::binary) {
    CUDF_EXPECTS(p == 2, "Binary plan requires two partitions");
    return binary(map, n, sink, stream);
  }
  int const blocks = (n - 1) / tile_rows + 1;
  rmm::device_uvector<int32_t> offsets(p + 1, stream);
  if (reorder) {
    if (method == algorithm::recompute)
      recompute<1, true>(map, n, p, sink, offsets.data(), stream);
    else if (method == algorithm::super2)
      recompute<2, true>(map, n, p, sink, offsets.data(), stream);
    else
      CUDF_FAIL("Unsupported reordered partition plan");
  } else if (method == algorithm::saved) {
    rmm::device_uvector<uint16_t> ranks(n, stream);
    rmm::device_uvector<int32_t> counts(size_t(p) * blocks, stream);
    count_tiles<1, true>
      <<<blocks, threads, 0, stream.get()>>>(map, n, p, ranks.data(), counts.data());
    CUDF_CUDA_TRY(cudaPeekAtLastError());
    scan(counts.data(), counts.size(), stream);
    get_offsets<<<1, threads, 0, stream.get()>>>(counts.data(), blocks, p, n, offsets.data());
    scatter_saved<<<blocks, threads, 0, stream.get()>>>(
      map, ranks.data(), counts.data(), n, blocks, p, sink);
    CUDF_CUDA_TRY(cudaPeekAtLastError());
  } else if (method == algorithm::recompute)
    recompute<1>(map, n, p, sink, offsets.data(), stream);
  else if (method == algorithm::super2)
    recompute<2>(map, n, p, sink, offsets.data(), stream);
  else if (method == algorithm::reserve) {
    global_hist(map, n, p, offsets.data(), stream);
    rmm::device_uvector<int32_t> cursors(p + 1, stream);
    CUDF_CUDA_TRY(cudf::detail::memcpy_async(
      cursors.data(), offsets.data(), (p + 1) * sizeof(int32_t), stream));
    scatter_reserve<<<blocks, threads, 0, stream.get()>>>(map, n, p, cursors.data(), sink);
    CUDF_CUDA_TRY(cudaPeekAtLastError());
  } else
    CUDF_FAIL("Unknown block partition plan");
  return cudf::detail::make_std_vector(offsets, stream);
}
}  // namespace cudf::detail::partition_map_plan
