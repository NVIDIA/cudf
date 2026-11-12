// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "partition_policy.hpp"

#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/detail/utilities/vector_factories.hpp>

#include <rmm/device_buffer.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <cub/device/device_partition.cuh>
#include <cuda/iterator>
#include <cuda/std/bit>
#include <thrust/scan.h>

namespace cudf::detail::partition_map_plan {
using algorithm       = partition_policy::algorithm;
constexpr int threads = 256, tile_rows = 1024, warps = 8, bins = 256;

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
    unsigned active = __ballot_sync(0xffffffffu, valid);
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
    static_assert(!Reorder, "Reordered sinks are not enabled yet");
    {
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

inline void scan(int32_t* ptr, size_t count, cuda::stream_ref stream)
{
  // Same Thrust in-place scan policy as PR1.
  thrust::exclusive_scan(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                         ptr,
                         ptr + count,
                         ptr);
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

template <typename Emit>
struct WriteProxy {
  Emit sink;
  int destination;
  __device__ void operator=(int source) const { sink(source, destination); }
};
template <typename Emit>
struct MakeWriteProxy {
  Emit sink;
  __host__ __device__ WriteProxy<Emit> operator()(int destination) const
  {
    return {sink, destination};
  }
};
template <typename Map>
struct IsZero {
  Map const* map;
  __device__ bool operator()(int row) const { return map[row] == 0; }
};

template <typename Map, typename Emit>
std::vector<int> binary(Map const* map, int n, Emit sink, cuda::stream_ref stream)
{
  rmm::device_uvector<int> selected(1, stream);
  auto source  = cuda::counting_iterator<int>{0};
  auto output  = cuda::transform_iterator(source, MakeWriteProxy<Emit>{sink});
  size_t bytes = 0;
  CUDF_CUDA_TRY(cub::DevicePartition::If(
    nullptr, bytes, source, output, selected.data(), n, IsZero<Map>{map}, stream.get()));
  rmm::device_buffer temporary(bytes, stream);
  CUDF_CUDA_TRY(cub::DevicePartition::If(
    temporary.data(), bytes, source, output, selected.data(), n, IsZero<Map>{map}, stream.get()));
  auto const count = cudf::detail::make_std_vector(selected, stream)[0];
  return {0, count, n};
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
  CUDF_EXPECTS(n > 0 && p > 0 && p <= bins && !reorder, "Invalid block partition dimensions");
  if (method == algorithm::binary) {
    CUDF_EXPECTS(p == 2, "Binary plan requires two partitions");
    return binary(map, n, sink, stream);
  }
  rmm::device_uvector<int32_t> offsets(p + 1, stream);
  if (method == algorithm::recompute) {
    recompute<1>(map, n, p, sink, offsets.data(), stream);
  } else if (method == algorithm::super2) {
    recompute<2>(map, n, p, sink, offsets.data(), stream);
  } else {
    CUDF_FAIL("Unsupported partition plan");
  }
  CUDF_CUDA_TRY(cudaPeekAtLastError());
  return cudf::detail::make_std_vector(offsets, stream);
}
}  // namespace cudf::detail::partition_map_plan
