/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/detail/utilities/cuda_memcpy.hpp>
#include <cudf/strings/string_view.cuh>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <cooperative_groups.h>
#include <cooperative_groups/scan.h>
#include <cub/block/block_exchange.cuh>
#include <cub/block/block_merge_sort.cuh>
#include <cub/block/block_reduce.cuh>
#include <cub/block/block_scan.cuh>
#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_run_length_encode.cuh>
#include <cub/device/device_scan.cuh>
#include <cub/device/device_select.cuh>
#include <cub/warp/warp_merge_sort.cuh>
#include <cuda/functional>
#include <cuda/iterator>
#include <cuda/std/limits>
#include <cuda/std/tuple>
#include <thrust/sequence.h>
#include <thrust/transform.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <type_traits>

namespace cudf::detail {

// Experimental radix key: null rank is most significant, followed by the prefix.
// Row IDs are values, never part of the radix key.
struct string_radix_prefix_key {
  uint64_t hi;
  uint32_t lo;
  uint32_t null_rank;
  __host__ __device__ bool operator==(string_radix_prefix_key const& rhs) const
  {
    return hi == rhs.hi && lo == rhs.lo && null_rank == rhs.null_rank;
  }
};

// Nonnullable prefixes carry no null rank or padding through radix sorting.
struct string_radix_prefix12 {
  uint32_t hi;
  uint32_t mid;
  uint32_t lo;
  __host__ __device__ bool operator==(string_radix_prefix12 const& rhs) const
  {
    return hi == rhs.hi && mid == rhs.mid && lo == rhs.lo;
  }
};
static_assert(sizeof(string_radix_prefix12) == 12);

template <typename Key>
__device__ bool radix_key_is_null(Key const& key, uint32_t null_rank)
{
  if constexpr (std::is_same_v<Key, string_radix_prefix_key>)
    return key.null_rank == null_rank;
  else
    return false;
}

template <int Bytes>
struct string_radix_decomposer {
  template <typename Key>
  __host__ __device__ auto operator()(Key& key) const
  {
    if constexpr (std::is_same_v<Key, string_radix_prefix12>) {
      return cuda::std::tuple<uint32_t&, uint32_t&, uint32_t&>{key.hi, key.mid, key.lo};
    } else if constexpr (Bytes <= 8) {
      return cuda::std::tuple<uint32_t&, uint64_t&>{key.null_rank, key.hi};
    } else {
      return cuda::std::tuple<uint32_t&, uint64_t&, uint32_t&>{key.null_rank, key.hi, key.lo};
    }
  }
};

struct radix_segment {
  size_type begin;
  size_type end;
};

// Locate the segment owning a tile by upper-bounding the scanned tile offsets.
__device__ inline size_type radix_tile_segment(size_type tile,
                                               size_type const* offsets,
                                               size_type count)
{
  size_type lo = 0, hi = count;
  while (lo < hi) {
    auto const mid = lo + (hi - lo) / 2;
    if (offsets[mid + 1] <= tile)
      lo = mid + 1;
    else
      hi = mid;
  }
  return lo;
}

template <int TileSize>
__global__ void radix_prepare_tiles(radix_segment const* segments,
                                    size_type count,
                                    size_type* tile_counts,
                                    size_type* max_length,
                                    size_type const* device_count)
{
  auto const i            = static_cast<size_type>(blockIdx.x * blockDim.x + threadIdx.x);
  auto const actual_count = device_count == nullptr ? count : *device_count;
  if (i < actual_count) {
    auto const length = segments[i].end - segments[i].begin;
    tile_counts[i]    = 1 + (length - 1) / TileSize;
    atomicMax(max_length, length);
  }
  if (i >= actual_count && i <= count) tile_counts[i] = 0;
}

template <int Threads, int Items, bool Stable, typename Comparator>
__device__ void radix_block_sort_tile(size_type const* input,
                                      size_type* output,
                                      radix_segment const* segments,
                                      size_type const* tile_offsets,
                                      size_type segment_count,
                                      size_type tile,
                                      Comparator comp)
{
  constexpr int tile_size = Threads * Items;
  auto const segment      = radix_tile_segment(tile, tile_offsets, segment_count);
  if constexpr (requires { comp.bind_segment(segment); }) {
    comp = comp.bind_segment(segment);
    if (comp.complete) return;
    if (comp.skip == 8) {
      radix_block_sort_tile<Threads, Items, Stable>(
        input, output, segments, tile_offsets, segment_count, tile, comp.base_comparator());
      return;
    }
  }
  auto const begin = segments[segment].begin + (tile - tile_offsets[segment]) * tile_size;
  auto const valid = min(tile_size, segments[segment].end - begin);
  using sorter     = cub::BlockMergeSort<size_type, Threads, Items>;
  __shared__ typename sorter::TempStorage storage;
  size_type rows[Items];
  for (int j = 0; j < Items; ++j) {
    auto const offset = static_cast<int>(threadIdx.x) * Items + j;
    rows[j]           = offset < valid ? input[begin + offset] : -1;
  }
  if constexpr (Stable)
    sorter(storage).StableSort(rows, comp, valid, -1);
  else
    sorter(storage).Sort(rows, comp, valid, -1);
  for (int j = 0; j < Items; ++j) {
    auto const offset = static_cast<int>(threadIdx.x) * Items + j;
    if (offset < valid) output[begin + offset] = rows[j];
  }
}

template <int Threads, int Items, bool Stable, typename Comparator>
__global__ void radix_segment_block_sort(size_type const* input,
                                         size_type* output,
                                         radix_segment const* segments,
                                         size_type const* tile_offsets,
                                         size_type segment_count,
                                         Comparator comp)
{
  radix_block_sort_tile<Threads, Items, Stable>(
    input, output, segments, tile_offsets, segment_count, static_cast<size_type>(blockIdx.x), comp);
}

// Each thread finds one position on the stable merge path and emits one row.
// Left-run equality precedes right-run equality, preserving stable radix order.
template <int Threads, int Items, typename Comparator>
__device__ void radix_merge_tile(size_type const* input,
                                 size_type* output,
                                 radix_segment const* segments,
                                 size_type const* tile_offsets,
                                 size_type segment_count,
                                 size_type tile,
                                 int64_t run_length,
                                 Comparator comp)
{
  constexpr int tile_size = Threads * Items;
  auto const segment      = radix_tile_segment(tile, tile_offsets, segment_count);
  if constexpr (requires { comp.bind_segment(segment); }) {
    comp = comp.bind_segment(segment);
    if (comp.complete) return;
    if (comp.skip == 8) {
      radix_merge_tile<Threads, Items>(input,
                                       output,
                                       segments,
                                       tile_offsets,
                                       segment_count,
                                       tile,
                                       run_length,
                                       comp.base_comparator());
      return;
    }
  }
  auto const seg         = segments[segment];
  auto const local_tile  = tile - tile_offsets[segment];
  auto const tile_begin  = static_cast<int64_t>(local_tile) * tile_size;
  auto const pair_begin  = tile_begin / (2 * run_length) * (2 * run_length);
  auto const length      = static_cast<int64_t>(seg.end - seg.begin);
  auto const left_count  = min(run_length, length - pair_begin);
  auto const right_count = max(int64_t{0}, min(run_length, length - pair_begin - run_length));
  auto const* left       = input + seg.begin + pair_begin;
  auto const* right      = left + left_count;
  if (right_count == 0) {
    for (int j = 0; j < Items; ++j) {
      auto const position = tile_begin + threadIdx.x + j * Threads;
      if (position < length) output[seg.begin + position] = input[seg.begin + position];
    }
    return;
  }
  // Partition each output tile once; thread searches then stay within this tile's inputs.
  __shared__ int64_t partition_a[2];
  if (threadIdx.x < 2) {
    auto const diagonal = min(tile_begin + threadIdx.x * tile_size, length) - pair_begin;
    int64_t lo          = max(int64_t{0}, diagonal - right_count);
    int64_t hi          = min(diagonal, left_count);
    while (lo < hi) {
      auto const a = lo + (hi - lo) / 2;
      auto const b = diagonal - a;
      if (b > 0 && a < left_count && !comp(right[b - 1], left[a]))
        lo = a + 1;
      else
        hi = a;
    }
    partition_a[threadIdx.x] = lo;
  }
  __syncthreads();
  auto const first_b = tile_begin - pair_begin - partition_a[0];
  auto const last_b  = min(tile_begin + tile_size, length) - pair_begin - partition_a[1];
  if constexpr (Items > 1) {
    if (comp.contiguous_merge) {
      // One merge-path search per thread, then emit adjacent items sequentially.
      size_type result[Items]{};
      auto const first_position = tile_begin + static_cast<int64_t>(threadIdx.x) * Items;
      if (first_position < length) {
        auto const diagonal = first_position - pair_begin;
        int64_t lo          = max(partition_a[0], diagonal - last_b);
        int64_t hi          = min(partition_a[1], diagonal - first_b);
        while (lo < hi) {
          auto const a = lo + (hi - lo) / 2;
          auto const b = diagonal - a;
          if (b > 0 && a < left_count && !comp(right[b - 1], left[a]))
            lo = a + 1;
          else
            hi = a;
        }
        auto a = lo;
        auto b = diagonal - a;
        for (int j = 0; j < Items && first_position + j < length; ++j) {
          if (a < left_count && (b >= right_count || !comp(right[b], left[a])))
            result[j] = left[a++];
          else
            result[j] = right[b++];
        }
      }
      // Preserve coalesced output despite computing contiguous runs per thread.
      using Exchange = cub::BlockExchange<size_type, Threads, Items>;
      __shared__ typename Exchange::TempStorage exchange;
      Exchange(exchange).BlockedToStriped(result, result);
      for (int j = 0; j < Items; ++j) {
        auto const position = tile_begin + threadIdx.x + j * Threads;
        if (position < length) output[seg.begin + position] = result[j];
      }
      return;
    }
  }
  for (int j = 0; j < Items; ++j) {
    auto const position = tile_begin + threadIdx.x + j * Threads;
    if (position >= length) continue;
    auto const diagonal = position - pair_begin;
    int64_t lo          = max(partition_a[0], diagonal - last_b);
    int64_t hi          = min(partition_a[1], diagonal - first_b);
    while (lo < hi) {
      auto const a = lo + (hi - lo) / 2;
      auto const b = diagonal - a;
      if (b > 0 && a < left_count && !comp(right[b - 1], left[a]))
        lo = a + 1;
      else
        hi = a;
    }
    auto const a = lo, b = diagonal - a;
    output[seg.begin + position] =
      a < left_count && (b >= right_count || !comp(right[b], left[a])) ? left[a] : right[b];
  }
}

template <int Threads, int Items, typename Comparator>
__global__ void radix_segment_merge(size_type const* input,
                                    size_type* output,
                                    radix_segment const* segments,
                                    size_type const* tile_offsets,
                                    size_type segment_count,
                                    int64_t run_length,
                                    Comparator comp)
{
  radix_merge_tile<Threads, Items>(input,
                                   output,
                                   segments,
                                   tile_offsets,
                                   segment_count,
                                   static_cast<size_type>(blockIdx.x),
                                   run_length,
                                   comp);
}

// A block reserves segment IDs and tile offsets together in a packed prefix sum.
// Reservations may reorder whole segments; row order inside each segment is unchanged.
template <int TileSize, typename Key>
__global__ void radix_compact_runs(size_type const* starts,
                                   size_type const* lengths,
                                   size_type const* run_count,
                                   Key const* sorted_keys,
                                   uint32_t null_rank,
                                   radix_segment* segments,
                                   size_type* tile_offsets,
                                   unsigned long long* metadata)
{
  using Scan   = cub::BlockScan<unsigned long long, 256>;
  using Reduce = cub::BlockReduce<size_type, 256>;
  __shared__ union {
    typename Scan::TempStorage scan;
    typename Reduce::TempStorage reduce;
  } temp;
  __shared__ unsigned long long reservation;
  auto const count = *run_count;
  for (size_type base = blockIdx.x * 256; base < count; base += gridDim.x * 256) {
    auto const i      = base + threadIdx.x;
    auto const begin  = i < count ? starts[i] : 0;
    auto const length = i < count ? lengths[i] : 0;
    bool const valid  = length > 1 && !radix_key_is_null(sorted_keys[begin], null_rank);
    auto const tiles  = valid ? 1 + (length - 1) / TileSize : 0;
    auto const packed = (static_cast<unsigned long long>(tiles) << 32) | (valid ? 1ULL : 0ULL);
    unsigned long long prefix, total;
    Scan(temp.scan).ExclusiveSum(packed, prefix, total);
    __syncthreads();
    auto const max_length = Reduce(temp.reduce).Reduce(valid ? length : 0, cuda::maximum<>{});
    if (threadIdx.x == 0) {
      reservation = total != 0 ? atomicAdd(metadata, total) : 0;
      atomicMax(reinterpret_cast<size_type*>(metadata + 1), max_length);
    }
    __syncthreads();
    if (valid) {
      auto const location   = reservation + prefix;
      auto const segment    = static_cast<uint32_t>(location);
      segments[segment]     = {begin, begin + length};
      tile_offsets[segment] = static_cast<size_type>(location >> 32);
    }
    __syncthreads();
  }
}

__global__ void radix_finish_offsets(size_type* offsets, unsigned long long const* metadata)
{
  auto const packed                      = *metadata;
  offsets[static_cast<uint32_t>(packed)] = static_cast<size_type>(packed >> 32);
}

// Schedule block sorts freely before the lighter cooperative merge kernel.
__global__ void radix_device_copy(size_type size,
                                  size_type const* input,
                                  size_type* output,
                                  unsigned long long const* metadata)
{
  if (static_cast<uint32_t>(metadata[0]) == 0) return;
  auto const stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (auto i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < size; i += stride)
    output[i] = input[i];
}

template <int Threads, int Items, bool Stable, typename Comparator>
__global__ void radix_device_block_sort(size_type const* input,
                                        size_type* output,
                                        radix_segment const* segments,
                                        size_type const* offsets,
                                        unsigned long long const* metadata,
                                        Comparator comp)
{
  auto const packed = metadata[0];
  auto const count  = static_cast<size_type>(static_cast<uint32_t>(packed));
  auto const tiles  = static_cast<size_type>(packed >> 32);
  for (size_type tile = blockIdx.x; tile < tiles; tile += gridDim.x) {
    radix_block_sort_tile<Threads, Items, Stable>(
      input, output, segments, offsets, count, tile, comp);
    __syncthreads();
  }
}

// Guarded launches retain the ordinary CUDA scheduler without host metadata.
template <int Threads, int Items, typename Comparator>
__global__ void radix_device_merge(size_type const* input,
                                   size_type* output,
                                   radix_segment const* segments,
                                   size_type const* offsets,
                                   unsigned long long const* metadata,
                                   int64_t run,
                                   Comparator comp)
{
  auto const packed = metadata[0];
  auto const count  = static_cast<size_type>(static_cast<uint32_t>(packed));
  if (count == 0 || run >= static_cast<int64_t>(metadata[1])) return;
  auto const tiles = static_cast<size_type>(packed >> 32);
  for (size_type tile = blockIdx.x; tile < tiles; tile += gridDim.x) {
    radix_merge_tile<Threads, Items>(input, output, segments, offsets, count, tile, run, comp);
    __syncthreads();
  }
}

template <int TileSize>
__global__ void radix_device_finish(size_type size,
                                    size_type const* scratch,
                                    size_type* output,
                                    unsigned long long const* metadata)
{
  if (static_cast<uint32_t>(metadata[0]) == 0) return;
  int levels = 0;
  for (int64_t run = TileSize; run < static_cast<int64_t>(metadata[1]); run *= 2)
    ++levels;
  if (levels % 2 != 0) return;  // Odd merge counts already leave the final result in output.
  auto const stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (auto i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < size; i += stride)
    output[i] = scratch[i];
}

template <int Threads, int Items, bool Stable, typename Comparator>
void radix_launch_device_schedule(size_type size,
                                  size_type* output,
                                  size_type* scratch,
                                  radix_segment const* segments,
                                  size_type const* offsets,
                                  unsigned long long const* metadata,
                                  Comparator comp,
                                  cuda::stream_ref stream)
{
  int device, multiprocessors;
  CUDF_CUDA_TRY(cudaGetDevice(&device));
  CUDF_CUDA_TRY(cudaDeviceGetAttribute(&multiprocessors, cudaDevAttrMultiProcessorCount, device));
  auto const blocks = std::min(multiprocessors * 32, std::max(1, size / 2));
  radix_device_copy<<<blocks, 256, 0, stream.get()>>>(size, output, scratch, metadata);
  CUDF_CUDA_TRY(cudaGetLastError());
  radix_device_block_sort<Threads, Items, Stable>
    <<<blocks, Threads, 0, stream.get()>>>(output, scratch, segments, offsets, metadata, comp);
  CUDF_CUDA_TRY(cudaGetLastError());
  auto* src = scratch;
  auto* dst = output;
  for (int64_t run = Threads * Items; run < size; run *= 2) {
    radix_device_merge<Threads, Items>
      <<<blocks, Threads, 0, stream.get()>>>(src, dst, segments, offsets, metadata, run, comp);
    CUDF_CUDA_TRY(cudaGetLastError());
    std::swap(src, dst);
  }
  radix_device_finish<Threads * Items>
    <<<blocks, 256, 0, stream.get()>>>(size, scratch, output, metadata);
  CUDF_CUDA_TRY(cudaGetLastError());
}

// All blocks remain resident; grid barriers replace host scheduling/readbacks.
template <int Threads, int Items, bool Stable, typename Comparator>
__global__ void radix_cooperative_refine(size_type size,
                                         size_type* output,
                                         size_type* scratch,
                                         radix_segment const* segments,
                                         size_type const* tile_offsets,
                                         unsigned long long const* metadata,
                                         Comparator comp)
{
  auto const packed        = metadata[0];
  auto const segment_count = static_cast<size_type>(static_cast<uint32_t>(packed));
  if (segment_count == 0) return;
  auto const tile_count = static_cast<size_type>(packed >> 32);
  auto const max_length = static_cast<size_type>(metadata[1]);
  auto const grid       = cooperative_groups::this_grid();
  auto const rank       = static_cast<int64_t>(blockIdx.x) * Threads + threadIdx.x;
  auto const stride     = static_cast<int64_t>(gridDim.x) * Threads;
  auto* src             = scratch;
  auto* dst             = output;
  for (int64_t run = Threads * Items; run < max_length; run *= 2) {
    for (size_type tile = blockIdx.x; tile < tile_count; tile += gridDim.x) {
      radix_merge_tile<Threads, Items>(
        src, dst, segments, tile_offsets, segment_count, tile, run, comp);
      __syncthreads();
    }
    grid.sync();
    auto* tmp = src;
    src       = dst;
    dst       = tmp;
  }
  if (src != output) {
    for (auto i = rank; i < size; i += stride)
      output[i] = src[i];
  }
}

template <int Threads, int Items, bool Stable, typename Comparator>
void radix_launch_cooperative(size_type size,
                              size_type* output,
                              size_type* scratch,
                              radix_segment const* segments,
                              size_type const* tile_offsets,
                              unsigned long long const* metadata,
                              Comparator comparator,
                              cuda::stream_ref stream)
{
  int device, supported, multiprocessors, blocks_per_sm;
  CUDF_CUDA_TRY(cudaGetDevice(&device));
  CUDF_CUDA_TRY(cudaDeviceGetAttribute(&supported, cudaDevAttrCooperativeLaunch, device));
  CUDF_EXPECTS(supported, "Cooperative radix refinement requires cooperative launch support");
  CUDF_CUDA_TRY(cudaDeviceGetAttribute(&multiprocessors, cudaDevAttrMultiProcessorCount, device));
  auto const work_blocks = std::min(multiprocessors * 16, std::max(1, size / 2));
  radix_device_copy<<<work_blocks, 256, 0, stream.get()>>>(size, output, scratch, metadata);
  CUDF_CUDA_TRY(cudaGetLastError());
  radix_device_block_sort<Threads, Items, Stable><<<work_blocks, Threads, 0, stream.get()>>>(
    output, scratch, segments, tile_offsets, metadata, comparator);
  CUDF_CUDA_TRY(cudaGetLastError());
  auto const kernel = radix_cooperative_refine<Threads, Items, Stable, Comparator>;
  CUDF_CUDA_TRY(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_sm, kernel, Threads, 0));
  auto const blocks = std::min(multiprocessors * blocks_per_sm, std::max(1, size / 2));
  CUDF_EXPECTS(blocks > 0, "No resident blocks available for cooperative refinement");
  void* args[] = {&size, &output, &scratch, &segments, &tile_offsets, &metadata, &comparator};
  CUDF_CUDA_TRY(
    cudaLaunchCooperativeKernel(kernel, dim3(blocks), dim3(Threads), args, 0, stream.get()));
}

template <int Threads, int Items, bool Stable, typename Comparator>
void radix_refine_segments(size_type size,
                           size_type* output,
                           size_type* scratch,
                           radix_segment const* segments,
                           size_type const* tile_offsets,
                           size_type segment_count,
                           size_type tile_count,
                           size_type max_length,
                           Comparator comparator,
                           cuda::stream_ref stream)
{
  if (segment_count == 0) return;
  constexpr int tile_size = Threads * Items;
  CUDF_CUDA_TRY(cudf::detail::memcpy_async(scratch, output, sizeof(size_type) * size, stream));
  radix_segment_block_sort<Threads, Items, Stable><<<tile_count, Threads, 0, stream.get()>>>(
    output, scratch, segments, tile_offsets, segment_count, comparator);
  CUDF_CUDA_TRY(cudaGetLastError());
  auto* src = scratch;
  auto* dst = output;
  for (int64_t run = tile_size; run < max_length; run *= 2) {
    radix_segment_merge<Threads, Items><<<tile_count, Threads, 0, stream.get()>>>(
      src, dst, segments, tile_offsets, segment_count, run, comparator);
    CUDF_CUDA_TRY(cudaGetLastError());
    std::swap(src, dst);
  }
  if (src != output) {
    CUDF_CUDA_TRY(cudf::detail::memcpy_async(output, src, sizeof(size_type) * size, stream));
  }
}

// LRB reorders whole prefix-tie descriptors, never the rows inside those descriptors.
// Bin b contains 2^(b-1) < length <= 2^b; all such segments share a merge depth.
struct radix_lrb_metadata {
  unsigned long long
    counts[32];  // bins 1..31: low32 segments, high32 initial tiles; bin 0: proof candidates
  unsigned long long cursors[32];
  size_type segment_base[33];
  uint64_t tile_base[33];
};

__device__ inline int radix_lrb_bin(size_type length)
{
  return 32 - __clz(static_cast<unsigned int>(length - 1));
}

__host__ __device__ inline int radix_lrb_tile(int bin, int mode, int fixed_tile)
{
  if (mode == 1) return fixed_tile;
  if (bin <= 5) return 1 << bin;
  if (bin <= 7) return 128;
  if (bin == 8) return 256;
  return fixed_tile;
}

__device__ inline unsigned int radix_lrb_peer_sum(unsigned int mask, unsigned int value)
{
#if __CUDA_ARCH__ >= 800
  return __reduce_add_sync(mask, value);
#else
  unsigned int sum = 0;
  for (auto peers = mask; peers != 0; peers &= peers - 1) {
    sum += __shfl_sync(mask, value, __ffs(peers) - 1);
  }
  return sum;
#endif
}

template <int FixedTile, typename Key, typename Comparator>
__global__ void radix_lrb_histogram(size_type const* starts,
                                    size_type const* lengths,
                                    size_type const* run_count,
                                    Key const* keys,
                                    uint32_t null_rank,
                                    int mode,
                                    radix_lrb_metadata* metadata,
                                    size_type const* row_ids,
                                    size_type size,
                                    bool probe,
                                    Comparator comp)
{
  __shared__ unsigned long long counts[32];
  if (threadIdx.x < 32) counts[threadIdx.x] = 0;
  __syncthreads();
  auto const lane = threadIdx.x & 31;
  for (int64_t base = static_cast<int64_t>(blockIdx.x) * blockDim.x; base < *run_count;
       base += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    auto const i      = base + threadIdx.x;
    auto const length = i < *run_count ? lengths[i] : 0;
    bool const valid  = length > 1 && !radix_key_is_null(keys[starts[i]], null_rank);
    auto const bin    = valid ? radix_lrb_bin(length) : 0;
    auto const tiles  = valid ? 1 + (length - 1) / radix_lrb_tile(bin, mode, FixedTile) : 0;
    auto const peers  = __match_any_sync(0xffffffffu, bin);
    if constexpr (requires { comp.enable_proof; }) {
      if (probe && valid && bin >= comp.proof_first_bin() &&
          comp.probe_run(row_ids,
                         radix_segment{starts[i], starts[i] + length},
                         size,
                         comp.enable_suffix_radix)) {
        atomicAdd(counts, 1ULL);
      }
    }
    if (valid) {
      auto const tile_sum = radix_lrb_peer_sum(peers, tiles);
      if (lane == static_cast<unsigned int>(__ffs(peers) - 1)) {
        atomicAdd(counts + bin, (static_cast<unsigned long long>(tile_sum) << 32) | __popc(peers));
      }
    }
  }
  __syncthreads();
  if (threadIdx.x < 32 && counts[threadIdx.x] != 0) {
    atomicAdd(metadata->counts + threadIdx.x, counts[threadIdx.x]);
  }
}

__global__ void radix_lrb_prefix(radix_lrb_metadata* metadata, size_type* offsets)
{
  auto const warp =
    cooperative_groups::tiled_partition<32>(cooperative_groups::this_thread_block());
  auto const bin    = threadIdx.x;
  auto const packed = bin == 0 ? 0ULL : metadata->counts[bin];
  // Counts fit uint32: at most n/2 segments and at most n initial tiles, with n < INT32_MAX.
  auto const inclusive        = cooperative_groups::inclusive_scan(warp, packed);
  auto const exclusive        = inclusive - packed;
  auto const count            = static_cast<uint32_t>(packed);
  auto const tiles            = static_cast<uint32_t>(packed >> 32);
  auto const segment_base     = static_cast<uint32_t>(exclusive);
  metadata->segment_base[bin] = segment_base;
  metadata->tile_base[bin]    = exclusive >> 32;
  // Each bin needs its own end sentinel, even when the next bin is empty.
  offsets[segment_base + bin + count] = tiles;
  if (bin == 31) {
    metadata->segment_base[32] = static_cast<uint32_t>(inclusive);
    metadata->tile_base[32]    = inclusive >> 32;
  }
}

template <int FixedTile, typename Key>
__global__ void radix_lrb_scatter(size_type const* starts,
                                  size_type const* lengths,
                                  size_type const* run_count,
                                  Key const* keys,
                                  uint32_t null_rank,
                                  int mode,
                                  radix_segment* segments,
                                  size_type* offsets,
                                  radix_lrb_metadata* metadata)
{
  __shared__ unsigned long long counts[32];
  __shared__ unsigned long long reservations[32];
  for (int64_t base = static_cast<int64_t>(blockIdx.x) * blockDim.x; base < *run_count;
       base += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    if (threadIdx.x < 32) counts[threadIdx.x] = 0;
    __syncthreads();
    auto const i             = base + threadIdx.x;
    auto const begin         = i < *run_count ? starts[i] : 0;
    auto const length        = i < *run_count ? lengths[i] : 0;
    bool const valid         = length > 1 && !radix_key_is_null(keys[begin], null_rank);
    auto const bin           = valid ? radix_lrb_bin(length) : 0;
    unsigned long long local = 0;
    if (valid) {
      auto const tiles = 1 + (length - 1) / radix_lrb_tile(bin, mode, FixedTile);
      local = atomicAdd(counts + bin, (static_cast<unsigned long long>(tiles) << 32) | 1ULL);
    }
    __syncthreads();
    if (threadIdx.x < 32 && counts[threadIdx.x] != 0) {
      reservations[threadIdx.x] = atomicAdd(metadata->cursors + threadIdx.x, counts[threadIdx.x]);
    }
    __syncthreads();
    if (valid) {
      auto const location    = reservations[bin] + local;
      auto const segment     = metadata->segment_base[bin] + static_cast<uint32_t>(location);
      segments[segment]      = {begin, begin + length};
      offsets[segment + bin] = static_cast<size_type>(location >> 32);
    }
    __syncthreads();
  }
}

__device__ inline int radix_lrb_find_bin(uint64_t tile,
                                         radix_lrb_metadata const* metadata,
                                         int first,
                                         int last)
{
  while (first < last) {
    auto const mid = first + (last - first) / 2;
    if (metadata->tile_base[mid + 1] <= tile)
      first = mid + 1;
    else
      last = mid;
  }
  return first;
}

template <int Threads, int Items, bool Stable, typename Comparator>
__global__ void radix_lrb_block_sort(size_type const* input,
                                     size_type* scratch,
                                     radix_segment const* segments,
                                     size_type const* offsets,
                                     radix_lrb_metadata const* metadata,
                                     int first_bin,
                                     int last_bin,
                                     Comparator comp)
{
  auto const begin = metadata->tile_base[first_bin];
  auto const end   = metadata->tile_base[last_bin];
  for (auto task = begin + blockIdx.x; task < end; task += gridDim.x) {
    auto const bin          = radix_lrb_find_bin(task, metadata, first_bin, last_bin);
    auto const segment_base = metadata->segment_base[bin];
    auto const count        = static_cast<uint32_t>(metadata->counts[bin]);
    auto bound_comp         = comp;
    if constexpr (requires { comp.bind_bin(segment_base); })
      bound_comp = comp.bind_bin(segment_base);
    radix_block_sort_tile<Threads, Items, Stable>(
      input,
      scratch,
      segments + segment_base,
      offsets + segment_base + bin,
      count,
      static_cast<size_type>(task - metadata->tile_base[bin]),
      bound_comp);
    __syncthreads();
  }
}

template <int Threads, int Items, typename Comparator>
__global__ void radix_lrb_merge(size_type const* input,
                                size_type* output,
                                radix_segment const* segments,
                                size_type const* offsets,
                                radix_lrb_metadata const* metadata,
                                int first_bin,
                                int64_t run,
                                Comparator comp)
{
  auto const begin = metadata->tile_base[first_bin];
  auto const end   = metadata->tile_base[32];
  for (auto task = begin + blockIdx.x; task < end; task += gridDim.x) {
    auto const bin          = radix_lrb_find_bin(task, metadata, first_bin, 32);
    auto const segment_base = metadata->segment_base[bin];
    auto const count        = static_cast<uint32_t>(metadata->counts[bin]);
    auto bound_comp         = comp;
    if constexpr (requires { comp.bind_bin(segment_base); })
      bound_comp = comp.bind_bin(segment_base);
    radix_merge_tile<Threads, Items>(input,
                                     output,
                                     segments + segment_base,
                                     offsets + segment_base + bin,
                                     count,
                                     static_cast<size_type>(task - metadata->tile_base[bin]),
                                     run,
                                     bound_comp);
    __syncthreads();
  }
}

template <bool Stable, typename Comparator>
__device__ bool radix_lrb_precedes(size_type lhs, size_type rhs, Comparator comp)
{
  if (lhs == -1 || rhs == -1) return rhs == -1 && lhs != -1;
  if constexpr (Stable && requires { comp.compare_three_way(lhs, rhs); }) {
    auto const comparison = comp.compare_three_way(lhs, rhs);
    return comparison < 0 || (comparison == 0 && lhs < rhs);
  } else if constexpr (Stable) {
    if (comp(lhs, rhs)) return true;
    if (comp(rhs, lhs)) return false;
    return lhs < rhs;  // Original row order, including descending string order.
  } else {
    return comp(lhs, rhs);
  }
}

template <int Group, bool Stable, typename Comparator>
__device__ void radix_lrb_warp_sort_group(
  size_type* output, radix_segment const* segments, size_type segment, bool active, Comparator comp)
{
  auto const lane  = threadIdx.x & (Group - 1);
  auto const seg   = active ? segments[segment] : radix_segment{0, 0};
  bool const valid = active && lane < seg.end - seg.begin;
  size_type row    = valid ? output[seg.begin + lane] : -1;
  for (int width = 2; width <= Group; width *= 2) {
    for (int distance = width / 2; distance > 0; distance /= 2) {
      auto const other      = __shfl_xor_sync(0xffffffffu, row, distance);
      bool const low_side   = (lane & distance) == 0;
      bool const forward    = (lane & width) == 0;
      bool const take_other = low_side == forward ? radix_lrb_precedes<Stable>(other, row, comp)
                                                  : radix_lrb_precedes<Stable>(row, other, comp);
      if (take_other) row = other;
    }
  }
  if (valid) output[seg.begin + lane] = row;
}

// Each physical warp owns its entire union. Sharing one union across a block
// would alias storage when neighboring warps process different size bins.
union radix_lrb_warp_storage {
  cub::WarpMergeSort<size_type, 1, 2>::TempStorage group2[16];
  cub::WarpMergeSort<size_type, 1, 4>::TempStorage group4[8];
  cub::WarpMergeSort<size_type, 1, 8>::TempStorage group8[4];
  cub::WarpMergeSort<size_type, 1, 16>::TempStorage group16[2];
  cub::WarpMergeSort<size_type, 1, 32>::TempStorage group32[1];
};

template <bool Stable, typename Comparator>
struct radix_lrb_warp_compare {
  Comparator comp;
  __device__ bool operator()(size_type lhs, size_type rhs) const
  {
    return radix_lrb_precedes<Stable>(lhs, rhs, comp);
  }
};

template <int Algorithm, int Group, bool Stable, typename Comparator>
__device__ void radix_lrb_warp_sort_group_tuned(size_type* output,
                                                radix_segment const* segments,
                                                size_type segment,
                                                bool active,
                                                Comparator comp,
                                                radix_lrb_warp_storage& storage)
{
  if constexpr (Algorithm == 0) {
    radix_lrb_warp_sort_group<Group, Stable>(output, segments, segment, active, comp);
    return;
  }
  auto const lane  = threadIdx.x & (Group - 1);
  auto const seg   = active ? segments[segment] : radix_segment{0, 0};
  bool const valid = active && lane < seg.end - seg.begin;
  size_type row    = valid ? output[seg.begin + lane] : -1;
  // Algorithm 1 shares each bitonic comparison; 2 uses CUB merge sorting;
  // 3 uses shared comparisons through eight lanes and CUB above that.
  if constexpr (Algorithm == 1 || (Algorithm == 3 && Group <= 8)) {
    for (int width = 2; width <= Group; width *= 2) {
      for (int distance = width / 2; distance > 0; distance /= 2) {
        auto const other        = __shfl_xor_sync(0xffffffffu, row, distance);
        bool const low          = (lane & distance) == 0;
        bool const forward      = (lane & width) == 0;
        int const swap          = low ? (forward ? radix_lrb_precedes<Stable>(other, row, comp)
                                                 : radix_lrb_precedes<Stable>(row, other, comp))
                                      : 0;
        auto const partner_swap = __shfl_xor_sync(0xffffffffu, swap, distance);
        if (low ? swap : partner_swap) row = other;
      }
    }
  } else if constexpr (Algorithm != 0) {
    auto const group = (threadIdx.x & 31) / Group;
    size_type keys[1]{row};
    radix_lrb_warp_compare<Stable, Comparator> less{comp};
    if constexpr (Group == 2)
      cub::WarpMergeSort<size_type, 1, Group>(storage.group2[group]).StableSort(keys, less);
    if constexpr (Group == 4)
      cub::WarpMergeSort<size_type, 1, Group>(storage.group4[group]).StableSort(keys, less);
    if constexpr (Group == 8)
      cub::WarpMergeSort<size_type, 1, Group>(storage.group8[group]).StableSort(keys, less);
    if constexpr (Group == 16)
      cub::WarpMergeSort<size_type, 1, Group>(storage.group16[group]).StableSort(keys, less);
    if constexpr (Group == 32)
      cub::WarpMergeSort<size_type, 1, Group>(storage.group32[group]).StableSort(keys, less);
    row = keys[0];
  }
  if (valid) output[seg.begin + lane] = row;
}

template <int Algorithm, bool Stable, typename Comparator>
__global__ void radix_lrb_warp_sort(size_type* output,
                                    radix_segment const* segments,
                                    radix_lrb_metadata const* metadata,
                                    Comparator comp)
{
  __shared__ radix_lrb_warp_storage storage[8];
  auto const warp =
    cooperative_groups::tiled_partition<32>(cooperative_groups::this_thread_block());
  auto const lane   = warp.thread_rank();
  auto const count  = lane >= 1 && lane <= 5 ? static_cast<uint32_t>(metadata->counts[lane]) : 0u;
  auto const groups = lane >= 1 && lane <= 5 ? 32 >> lane : 1;
  auto const tasks  = static_cast<uint64_t>(count + groups - 1) / groups;
  auto const inclusive = cooperative_groups::inclusive_scan(warp, tasks);
  auto const exclusive = inclusive - tasks;
  auto const total     = warp.shfl(inclusive, 31);
  auto const first     = metadata->segment_base[lane];
  auto const warp_id   = static_cast<uint64_t>(blockIdx.x) * (blockDim.x / 32) + threadIdx.x / 32;
  auto const stride    = static_cast<uint64_t>(gridDim.x) * (blockDim.x / 32);
  for (auto task = warp_id; task < total; task += stride) {
    auto const bin     = __ffs(warp.ballot(task < inclusive)) - 1;
    auto const local   = task - warp.shfl(exclusive, bin);
    auto const group   = 1 << bin;
    auto const segment = warp.shfl(first, bin) + local * (32 / group) + lane / group;
    auto const end     = warp.shfl(first + count, bin);
    bool const active  = segment < end;
    switch (bin) {
      case 1:
        radix_lrb_warp_sort_group_tuned<Algorithm, 2, Stable>(
          output, segments, segment, active, comp, storage[threadIdx.x / 32]);
        break;
      case 2:
        radix_lrb_warp_sort_group_tuned<Algorithm, 4, Stable>(
          output, segments, segment, active, comp, storage[threadIdx.x / 32]);
        break;
      case 3:
        radix_lrb_warp_sort_group_tuned<Algorithm, 8, Stable>(
          output, segments, segment, active, comp, storage[threadIdx.x / 32]);
        break;
      case 4:
        radix_lrb_warp_sort_group_tuned<Algorithm, 16, Stable>(
          output, segments, segment, active, comp, storage[threadIdx.x / 32]);
        break;
      case 5:
        radix_lrb_warp_sort_group_tuned<Algorithm, 32, Stable>(
          output, segments, segment, active, comp, storage[threadIdx.x / 32]);
        break;
    }
    // The next task can use a different union member for this physical warp.
    __syncwarp();
  }
}

// A proof is exact: samples decide whether to scan, never whether to skip bytes.
// Only runs larger than 32768 rows are eligible; tiny/medium workers are unchanged.
struct radix_lrb_proof {
  unsigned int prefix;
  size_type min_length;
  size_type max_length;
  unsigned int flags;  // bit 0: unequal; bit 1: scanned; bit 2: spans the whole input.
};

template <typename Comparator>
__global__ void radix_lrb_probe(size_type const* output,
                                radix_segment const* segments,
                                radix_lrb_metadata const* metadata,
                                radix_lrb_proof* proofs,
                                size_type count,
                                size_type size,
                                bool extra_radix,
                                Comparator comp)
{
  auto const ordinal = static_cast<size_type>(blockIdx.x * blockDim.x + threadIdx.x);
  if (ordinal >= count) return;
  auto const seg       = segments[metadata->segment_base[comp.proof_first_bin()] + ordinal];
  auto const reference = output[seg.begin];
  auto const length    = comp.keys.template element<string_view>(reference).size_bytes();
  bool const full      = seg.begin == 0 && seg.end == size;
  bool const enabled   = comp.probe_run(output, seg, size, extra_radix);
  proofs[ordinal] = {enabled ? ~0u : 8u, length, length, (enabled ? 2u : 1u) | (full ? 4u : 0u)};
}

static_assert(sizeof(radix_lrb_proof) == 16);

struct radix_lrb_proof_reduce {
  __device__ radix_lrb_proof operator()(radix_lrb_proof a, radix_lrb_proof b) const
  {
    return {min(a.prefix, b.prefix),
            min(a.min_length, b.min_length),
            max(a.max_length, b.max_length),
            a.flags | b.flags};
  }
};

template <int Threads, int Items, typename Comparator>
__global__ void radix_lrb_prove(size_type const* output,
                                radix_segment const* segments,
                                size_type const* offsets,
                                radix_lrb_metadata const* metadata,
                                radix_lrb_proof* proofs,
                                Comparator comp)
{
  using reduction = cub::BlockReduce<radix_lrb_proof, Threads>;
  __shared__ typename reduction::TempStorage storage;
  auto const begin = metadata->tile_base[comp.proof_first_bin()];
  auto const end   = metadata->tile_base[32];
  for (auto task = begin + blockIdx.x; task < end; task += gridDim.x) {
    auto const bin           = radix_lrb_find_bin(task, metadata, comp.proof_first_bin(), 32);
    auto const base          = metadata->segment_base[bin];
    auto const count         = static_cast<uint32_t>(metadata->counts[bin]);
    auto const* tile_offsets = offsets + base + bin;
    auto const tile          = static_cast<size_type>(task - metadata->tile_base[bin]);
    auto const segment       = radix_tile_segment(tile, tile_offsets, count);
    auto const ordinal       = base + segment - metadata->segment_base[comp.proof_first_bin()];
    // Atomic read avoids mixing ordinary reads with other blocks' atomic updates.
    auto flags = (threadIdx.x & 31) == 0 ? atomicOr(&proofs[ordinal].flags, 0u) : 0u;
    flags      = __shfl_sync(0xffffffffu, flags, 0);
    if ((flags & 2u) == 0) continue;
    auto const seg              = segments[base + segment];
    auto const reference        = output[seg.begin];
    auto const reference_length = comp.keys.template element<string_view>(reference).size_bytes();
    auto const start            = seg.begin + (tile - tile_offsets[segment]) * (Threads * Items);
    radix_lrb_proof value{~0u, cuda::std::numeric_limits<size_type>::max(), 0, 0};
    for (int j = 0; j < Items; ++j) {
      auto const i = start + threadIdx.x + j * Threads;
      if (i >= seg.end) continue;
      auto const row    = output[i];
      auto const length = comp.keys.template element<string_view>(row).size_bytes();
      auto const common = comp.common_prefix(reference, row);
      value.prefix      = min(value.prefix, common);
      value.min_length  = min(value.min_length, length);
      value.max_length  = max(value.max_length, length);
      value.flags |=
        (length != reference_length || common != static_cast<unsigned int>(length)) ? 1u : 0u;
    }
    auto const result = reduction(storage).Reduce(value, radix_lrb_proof_reduce{});
    if (threadIdx.x == 0) {
      atomicMin(&proofs[ordinal].prefix, result.prefix);
      atomicMin(&proofs[ordinal].min_length, result.min_length);
      atomicMax(&proofs[ordinal].max_length, result.max_length);
      atomicOr(&proofs[ordinal].flags, result.flags);
    }
    __syncthreads();
  }
}

template <int FixedTile, typename Comparator>
__global__ void radix_lrb_finish(size_type const* scratch,
                                 size_type* output,
                                 radix_segment const* segments,
                                 size_type const* offsets,
                                 radix_lrb_metadata const* metadata,
                                 int mode,
                                 int fixed_log,
                                 Comparator comp)
{
  auto const first_bin = mode == 2 ? 6 : 1;
  auto const begin     = metadata->tile_base[first_bin];
  auto const end       = metadata->tile_base[32];
  for (auto task = begin + blockIdx.x; task < end; task += gridDim.x) {
    auto const bin    = radix_lrb_find_bin(task, metadata, first_bin, 32);
    auto const levels = mode == 2 && bin < 9 ? 0 : max(0, bin - fixed_log);
    if ((levels & 1) != 0) continue;  // Its completed result is already in output.
    auto const base          = metadata->segment_base[bin];
    auto const count         = static_cast<uint32_t>(metadata->counts[bin]);
    auto const* tile_offsets = offsets + base + bin;
    auto const tile          = static_cast<size_type>(task - metadata->tile_base[bin]);
    auto const segment       = radix_tile_segment(tile, tile_offsets, count);
    if constexpr (requires { comp.bind_bin(base).bind_segment(segment); }) {
      if (comp.bind_bin(base).bind_segment(segment).complete) continue;
    }
    auto const seg       = segments[base + segment];
    auto const tile_size = radix_lrb_tile(bin, mode, FixedTile);
    auto const start =
      static_cast<int64_t>(seg.begin) + (tile - tile_offsets[segment]) * int64_t{tile_size};
    auto const stop = min(start + tile_size, static_cast<int64_t>(seg.end));
    for (auto i = start + threadIdx.x; i < stop; i += blockDim.x)
      output[i] = scratch[i];
  }
}

template <int Threads, int Items, bool Stable, typename Key, typename Comparator>
void radix_lrb_refine(size_type size,
                      size_type* output,
                      size_type* scratch,
                      size_type const* starts,
                      size_type const* lengths,
                      size_type const* run_count,
                      Key* sorted_keys,
                      Key* radix_scratch_keys,
                      uint32_t null_rank,
                      int mode,
                      int schedule,
                      int warp_sort,
                      int grid_warps,
                      Comparator comp,
                      cuda::stream_ref stream)
{
  CUDF_EXPECTS(schedule == 0 || schedule == 2, "LRB supports native or guarded device scheduling");
  constexpr int fixed_tile = Threads * Items;
  int fixed_log            = 0;
  for (int tile = fixed_tile; tile > 1; tile /= 2)
    ++fixed_log;
  auto const mr       = cudf::get_current_device_resource_ref();
  auto const capacity = size / 2;
  rmm::device_uvector<radix_segment> segments(capacity, stream, mr);
  rmm::device_uvector<size_type> offsets(capacity + 32, stream, mr);
  rmm::device_uvector<radix_lrb_metadata> metadata(1, stream, mr);
  CUDF_CUDA_TRY(cudaMemsetAsync(metadata.data(), 0, sizeof(radix_lrb_metadata), stream.get()));
  auto const build_blocks = std::max(1, std::min(256, (capacity + 255) / 256));
  bool probe              = false;
  if constexpr (requires { comp.enable_proof; })
    probe = schedule == 0 && mode == 2 && comp.enable_proof;
  radix_lrb_histogram<fixed_tile><<<build_blocks, 256, 0, stream.get()>>>(starts,
                                                                          lengths,
                                                                          run_count,
                                                                          sorted_keys,
                                                                          null_rank,
                                                                          mode,
                                                                          metadata.data(),
                                                                          output,
                                                                          size,
                                                                          probe,
                                                                          comp);
  CUDF_CUDA_TRY(cudaGetLastError());
  radix_lrb_prefix<<<1, 32, 0, stream.get()>>>(metadata.data(), offsets.data());
  CUDF_CUDA_TRY(cudaGetLastError());
  radix_lrb_scatter<fixed_tile><<<build_blocks, 256, 0, stream.get()>>>(starts,
                                                                        lengths,
                                                                        run_count,
                                                                        sorted_keys,
                                                                        null_rank,
                                                                        mode,
                                                                        segments.data(),
                                                                        offsets.data(),
                                                                        metadata.data());
  CUDF_CUDA_TRY(cudaGetLastError());
  unsigned long long counts[32]{};
  bool const native = schedule == 0;
  if (native) {
    CUDF_CUDA_TRY(
      cudf::detail::memcpy_async(counts, metadata.data()->counts, sizeof(counts), stream));
    CUDF_CUDA_TRY(cudaStreamSynchronize(stream.get()));
  }
  int device, multiprocessors;
  CUDF_CUDA_TRY(cudaGetDevice(&device));
  CUDF_CUDA_TRY(cudaDeviceGetAttribute(&multiprocessors, cudaDevAttrMultiProcessorCount, device));
  auto tiles = [&](int first, int last) {
    uint64_t total = 0;
    for (int bin = first; bin < last; ++bin)
      total += counts[bin] >> 32;
    return total;
  };
  auto blocks = [&](uint64_t tasks, int threads = 256) {
    // Equal warp budgets avoid underfilling the SM with 128-thread workers.
    auto const max_blocks =
      std::max(1, std::min(multiprocessors * grid_warps / (threads / 32), capacity));
    return native ? static_cast<int>(std::min<uint64_t>(tasks, max_blocks)) : max_blocks;
  };
  if (native && tiles(1, 32) == 0) return;
  auto execute = [&]<typename GiantComparator>(GiantComparator giant_comp) {
    if (mode == 1) {
      radix_lrb_block_sort<Threads, Items, Stable>
        <<<blocks(tiles(1, 32), Threads), Threads, 0, stream.get()>>>(
          output, scratch, segments.data(), offsets.data(), metadata.data(), 1, 32, giant_comp);
      CUDF_CUDA_TRY(cudaGetLastError());
    } else {
      uint64_t warp_tasks = 0;
      for (int bin = 1; bin <= 5; ++bin) {
        auto const groups = 32 >> bin;
        warp_tasks += (static_cast<uint32_t>(counts[bin]) + groups - 1) / groups;
      }
      if (!native || warp_tasks != 0) {
        auto launch = [&]<int Algorithm>() {
          radix_lrb_warp_sort<Algorithm, Stable>
            <<<blocks((warp_tasks + 7) / 8), 256, 0, stream.get()>>>(
              output, segments.data(), metadata.data(), comp);
        };
        switch (warp_sort) {
          case 0: launch.template operator()<0>(); break;
          case 1: launch.template operator()<1>(); break;
          case 2: launch.template operator()<2>(); break;
          case 3: launch.template operator()<3>(); break;
        }
        CUDF_CUDA_TRY(cudaGetLastError());
      }
      if (!native || tiles(6, 8) != 0) {
        radix_lrb_block_sort<128, 1, Stable><<<blocks(tiles(6, 8), 128), 128, 0, stream.get()>>>(
          output, scratch, segments.data(), offsets.data(), metadata.data(), 6, 8, comp);
        CUDF_CUDA_TRY(cudaGetLastError());
      }
      if (!native || tiles(8, 9) != 0) {
        radix_lrb_block_sort<256, 1, Stable><<<blocks(tiles(8, 9)), 256, 0, stream.get()>>>(
          output, scratch, segments.data(), offsets.data(), metadata.data(), 8, 9, comp);
        CUDF_CUDA_TRY(cudaGetLastError());
      }
      if (!native || tiles(9, 32) != 0) {
        radix_lrb_block_sort<Threads, Items, Stable>
          <<<blocks(tiles(9, 32), Threads), Threads, 0, stream.get()>>>(
            output, scratch, segments.data(), offsets.data(), metadata.data(), 9, 32, giant_comp);
        CUDF_CUDA_TRY(cudaGetLastError());
      }
    }
    auto* src = scratch;
    auto* dst = output;
    for (int run_log = fixed_log; run_log < 31 && (int64_t{1} << run_log) < size; ++run_log) {
      // Medium adaptive runs are already complete in their 128/256-row tile.
      // Giant runs can require several stages before the active bin advances.
      auto const first_bin    = std::max(mode == 2 ? 9 : 1, run_log + 1);
      auto const active_tiles = tiles(first_bin, 32);
      if (native && active_tiles == 0) break;
      auto const run = int64_t{1} << run_log;
      radix_lrb_merge<Threads, Items><<<blocks(active_tiles, Threads), Threads, 0, stream.get()>>>(
        src, dst, segments.data(), offsets.data(), metadata.data(), first_bin, run, giant_comp);
      CUDF_CUDA_TRY(cudaGetLastError());
      std::swap(src, dst);
    }
    auto const finish_tiles = tiles(mode == 2 ? 6 : 1, 32);
    if (!native || finish_tiles != 0) {
      radix_lrb_finish<fixed_tile><<<blocks(finish_tiles), 256, 0, stream.get()>>>(scratch,
                                                                                   output,
                                                                                   segments.data(),
                                                                                   offsets.data(),
                                                                                   metadata.data(),
                                                                                   mode,
                                                                                   fixed_log,
                                                                                   giant_comp);
      CUDF_CUDA_TRY(cudaGetLastError());
    }
  };
  if constexpr (requires {
                  comp.enable_proof;
                  comp.make_proof_comparator(nullptr, nullptr);
                }) {
    size_type eligible   = 0;
    auto const proof_bin = comp.proof_first_bin();
    for (int bin = proof_bin; bin < 32; ++bin)
      eligible += static_cast<uint32_t>(counts[bin]);
    // Guarded schedules retain the original device-only algorithm and graph topology.
    if (native && mode == 2 && eligible != 0 && counts[0] != 0 && comp.enable_proof) {
      rmm::device_uvector<radix_lrb_proof> proofs(eligible, stream, mr);
      radix_lrb_probe<<<(eligible + 127) / 128, 128, 0, stream.get()>>>(output,
                                                                        segments.data(),
                                                                        metadata.data(),
                                                                        proofs.data(),
                                                                        eligible,
                                                                        size,
                                                                        comp.enable_suffix_radix,
                                                                        comp);
      CUDF_CUDA_TRY(cudaGetLastError());
      radix_lrb_prove<Threads, Items>
        <<<blocks(tiles(proof_bin, 32), Threads), Threads, 0, stream.get()>>>(
          output, segments.data(), offsets.data(), metadata.data(), proofs.data(), comp);
      CUDF_CUDA_TRY(cudaGetLastError());
      if (eligible == 1 && comp.enable_suffix_radix &&
          tiles(proof_bin, 32) == static_cast<uint64_t>((size - 1) / fixed_tile + 1)) {
        radix_lrb_proof proof;
        CUDF_CUDA_TRY(cudf::detail::memcpy_async(&proof, proofs.data(), sizeof(proof), stream));
        CUDF_CUDA_TRY(cudaStreamSynchronize(stream.get()));
        if ((proof.flags & 4u) != 0) {
          if ((proof.flags & 1u) == 0) return;
          if constexpr (std::is_same_v<Key, uint64_t>) {
            if (comp.try_suffix_radix(
                  proof, size, output, scratch, sorted_keys, radix_scratch_keys, stream))
              return;
          }
        }
      }
      execute(comp.make_proof_comparator(proofs.data(), metadata.data()));
      return;
    }
  }
  execute(comp);
}

template <int Bytes, int Threads, int Items, bool Stable, typename Extractor, typename Comparator>
void radix_prefix_refine(size_type size,
                         size_type* output,
                         Extractor extractor,
                         Comparator comparator,
                         uint32_t null_rank,
                         bool use_rle,
                         bool device_metadata,
                         bool compact_runs,
                         int schedule,
                         int lrb_mode,
                         int warp_sort,
                         int grid_warps,
                         cuda::stream_ref stream)
{
  CUDF_EXPECTS(lrb_mode == 0 || use_rle, "LRB requires radix RLE");
  // Device metadata mode prepares all scheduling information before one host readback.
  device_metadata = device_metadata && use_rle;
  CUDF_EXPECTS(size < std::numeric_limits<size_type>::max(), "Radix boundary count overflow");
  auto const mr     = cudf::get_current_device_resource_ref();
  auto const policy = rmm::exec_policy_nosync(stream, mr);
  auto rows         = cuda::counting_iterator<size_type>{0};
  using key_type    = decltype(extractor(size_type{0}));
  rmm::device_uvector<key_type> keys_in(size, stream, mr);
  rmm::device_uvector<key_type> keys_out(size, stream, mr);
  rmm::device_uvector<size_type> scratch(size, stream, mr);
  thrust::transform(policy, rows, rows + size, keys_in.begin(), extractor);
  thrust::sequence(policy, scratch.begin(), scratch.end(), size_type{0});
  std::size_t bytes = 0;
  auto radix_sort   = [&](void* storage) {
    if constexpr (std::is_same_v<key_type, uint64_t>) {
      return cub::DeviceRadixSort::SortPairs(storage,
                                             bytes,
                                             keys_in.data(),
                                             keys_out.data(),
                                             scratch.data(),
                                             output,
                                             size,
                                             0,
                                             64,
                                             stream.get());
    } else {
      return cub::DeviceRadixSort::SortPairs(
        storage,
        bytes,
        keys_in.data(),
        keys_out.data(),
        scratch.data(),
        output,
        size,
        string_radix_decomposer<Bytes>{},
        0,
        std::is_same_v<key_type, string_radix_prefix12> ? 96 : (Bytes <= 8 ? 65 : 97),
        stream.get());
    }
  };
  CUDF_CUDA_TRY(radix_sort(nullptr));
  {
    rmm::device_buffer temp(bytes, stream, mr);
    CUDF_CUDA_TRY(radix_sort(temp.data()));
  }
  keys_in.resize(0, stream);

  rmm::device_uvector<size_type> count(3, stream, mr);
  rmm::device_uvector<radix_segment> segments(0, stream, mr);
  auto* sorted_keys = keys_out.data();
  if (use_rle) {
    auto const max_runs = size / 2;
    rmm::device_uvector<size_type> offsets(max_runs, stream, mr);
    rmm::device_uvector<size_type> lengths(max_runs, stream, mr);
    rmm::device_uvector<size_type> run_count(1, stream, mr);
    auto encode = [&](void* storage) {
      return cub::DeviceRunLengthEncode::NonTrivialRuns(storage,
                                                        bytes,
                                                        sorted_keys,
                                                        offsets.data(),
                                                        lengths.data(),
                                                        run_count.data(),
                                                        size,
                                                        stream.get());
    };
    bytes = 0;
    CUDF_CUDA_TRY(encode(nullptr));
    {
      rmm::device_buffer temp(bytes, stream, mr);
      CUDF_CUDA_TRY(encode(temp.data()));
    }
    if (lrb_mode != 0) {
      radix_lrb_refine<Threads, Items, Stable>(
        size,
        output,
        scratch.data(),
        offsets.data(),
        lengths.data(),
        run_count.data(),
        sorted_keys,
        keys_in.data(),  // resize(0) retains this allocation for optional suffix radix.
        null_rank,
        lrb_mode,
        schedule,
        warp_sort,
        grid_warps,
        comparator,
        stream);
      return;
    }
    segments.resize(max_runs, stream);
    if (compact_runs) {
      rmm::device_uvector<size_type> tile_offsets(max_runs + 1, stream, mr);
      rmm::device_uvector<unsigned long long> metadata(2, stream, mr);
      CUDF_CUDA_TRY(cudaMemsetAsync(
        metadata.data(), 0, metadata.size() * sizeof(unsigned long long), stream.get()));
      radix_compact_runs<Threads * Items>
        <<<std::min(256, (max_runs + 255) / 256), 256, 0, stream.get()>>>(offsets.data(),
                                                                          lengths.data(),
                                                                          run_count.data(),
                                                                          sorted_keys,
                                                                          null_rank,
                                                                          segments.data(),
                                                                          tile_offsets.data(),
                                                                          metadata.data());
      CUDF_CUDA_TRY(cudaGetLastError());
      radix_finish_offsets<<<1, 1, 0, stream.get()>>>(tile_offsets.data(), metadata.data());
      CUDF_CUDA_TRY(cudaGetLastError());
      if (schedule == 2) {
        radix_launch_device_schedule<Threads, Items, Stable>(size,
                                                             output,
                                                             scratch.data(),
                                                             segments.data(),
                                                             tile_offsets.data(),
                                                             metadata.data(),
                                                             comparator,
                                                             stream);
        return;
      }
      if (schedule == 1) {
        radix_launch_cooperative<Threads, Items, Stable>(size,
                                                         output,
                                                         scratch.data(),
                                                         segments.data(),
                                                         tile_offsets.data(),
                                                         metadata.data(),
                                                         comparator,
                                                         stream);
        return;
      }
      unsigned long long host_metadata[2];
      CUDF_CUDA_TRY(
        cudf::detail::memcpy_async(host_metadata, metadata.data(), sizeof(host_metadata), stream));
      CUDF_CUDA_TRY(cudaStreamSynchronize(stream.get()));
      auto const segment_count = static_cast<size_type>(static_cast<uint32_t>(host_metadata[0]));
      auto const tile_count    = static_cast<size_type>(host_metadata[0] >> 32);
      auto const max_length    = static_cast<size_type>(host_metadata[1]);
      radix_refine_segments<Threads, Items, Stable>(size,
                                                    output,
                                                    scratch.data(),
                                                    segments.data(),
                                                    tile_offsets.data(),
                                                    segment_count,
                                                    tile_count,
                                                    max_length,
                                                    comparator,
                                                    stream);
      return;
    }
    auto const* starts = offsets.data();
    auto const* sizes  = lengths.data();
    auto const* runs   = run_count.data();
    auto make_segment  = [starts, sizes, runs] __device__(size_type i) -> radix_segment {
      if (i >= *runs) return {0, 0};
      return {starts[i], starts[i] + sizes[i]};
    };
    auto segment_input = cuda::make_transform_iterator(rows, make_segment);
    auto non_null      = [sorted_keys, null_rank] __device__(radix_segment seg) -> bool {
      return seg.end > seg.begin && !radix_key_is_null(sorted_keys[seg.begin], null_rank);
    };
    auto filter = [&](void* storage) {
      return cub::DeviceSelect::If(storage,
                                   bytes,
                                   segment_input,
                                   segments.data(),
                                   count.data(),
                                   max_runs,
                                   non_null,
                                   stream.get());
    };
    bytes = 0;
    CUDF_CUDA_TRY(filter(nullptr));
    {
      rmm::device_buffer temp(bytes, stream, mr);
      CUDF_CUDA_TRY(filter(temp.data()));
    }
  } else {
    rmm::device_uvector<size_type> boundaries(static_cast<std::size_t>(size) + 1, stream, mr);
    auto is_boundary = [sorted_keys, size] __device__(size_type i) -> bool {
      return i == 0 || i == size || !(sorted_keys[i] == sorted_keys[i - 1]);
    };
    auto select_boundaries = [&](void* storage) {
      return cub::DeviceSelect::If(
        storage, bytes, rows, boundaries.data(), count.data(), size + 1, is_boundary, stream.get());
    };
    bytes = 0;
    CUDF_CUDA_TRY(select_boundaries(nullptr));
    {
      rmm::device_buffer temp(bytes, stream, mr);
      CUDF_CUDA_TRY(select_boundaries(temp.data()));
    }
    size_type boundary_count;
    CUDF_CUDA_TRY(
      cudf::detail::memcpy_async(&boundary_count, count.data(), sizeof(size_type), stream));
    CUDF_CUDA_TRY(cudaStreamSynchronize(stream.get()));
    segments.resize(boundary_count - 1, stream);
    auto const* starts = boundaries.data();
    auto make_segment  = [starts] __device__(size_type i) -> radix_segment {
      return radix_segment{starts[i], starts[i + 1]};
    };
    auto segment_input    = cuda::make_transform_iterator(rows, make_segment);
    auto needs_refinement = [sorted_keys, null_rank] __device__(radix_segment segment) -> bool {
      return segment.end - segment.begin > 1 &&
             !radix_key_is_null(sorted_keys[segment.begin], null_rank);
    };
    auto select_segments = [&](void* storage) {
      return cub::DeviceSelect::If(storage,
                                   bytes,
                                   segment_input,
                                   segments.data(),
                                   count.data(),
                                   boundary_count - 1,
                                   needs_refinement,
                                   stream.get());
    };
    bytes = 0;
    CUDF_CUDA_TRY(select_segments(nullptr));
    {
      rmm::device_buffer temp(bytes, stream, mr);
      CUDF_CUDA_TRY(select_segments(temp.data()));
    }
  }
  size_type segment_count = 0;
  if (!device_metadata) {
    CUDF_CUDA_TRY(
      cudf::detail::memcpy_async(&segment_count, count.data(), sizeof(size_type), stream));
    CUDF_CUDA_TRY(cudaStreamSynchronize(stream.get()));
    if (segment_count == 0) return;
  }

  constexpr int tile_size = Threads * Items;
  auto const capacity     = device_metadata ? size / 2 : segment_count;
  rmm::device_uvector<size_type> tile_counts(capacity + 1, stream, mr);
  rmm::device_uvector<size_type> tile_offsets(capacity + 1, stream, mr);
  CUDF_CUDA_TRY(cudaMemsetAsync(count.data() + 2, 0, sizeof(size_type), stream.get()));
  radix_prepare_tiles<tile_size>
    <<<(capacity + 256) / 256, 256, 0, stream.get()>>>(segments.data(),
                                                       capacity,
                                                       tile_counts.data(),
                                                       count.data() + 2,
                                                       device_metadata ? count.data() : nullptr);
  CUDF_CUDA_TRY(cudaGetLastError());
  bytes = 0;
  CUDF_CUDA_TRY(cub::DeviceScan::ExclusiveSum(
    nullptr, bytes, tile_counts.data(), tile_offsets.data(), capacity + 1, stream.get()));
  {
    rmm::device_buffer temp(bytes, stream, mr);
    CUDF_CUDA_TRY(cub::DeviceScan::ExclusiveSum(
      temp.data(), bytes, tile_counts.data(), tile_offsets.data(), capacity + 1, stream.get()));
  }
  CUDF_CUDA_TRY(cudf::detail::memcpy_async(
    count.data() + 1, tile_offsets.data() + capacity, sizeof(size_type), stream));
  size_type metadata[3];
  CUDF_CUDA_TRY(cudf::detail::memcpy_async(metadata, count.data(), sizeof(metadata), stream));
  CUDF_CUDA_TRY(cudaStreamSynchronize(stream.get()));
  segment_count         = metadata[0];
  auto const tile_count = metadata[1];
  auto const max_length = metadata[2];
  if (segment_count == 0) return;
  radix_refine_segments<Threads, Items, Stable>(size,
                                                output,
                                                scratch.data(),
                                                segments.data(),
                                                tile_offsets.data(),
                                                segment_count,
                                                tile_count,
                                                max_length,
                                                comparator,
                                                stream);
}
}  // namespace cudf::detail
