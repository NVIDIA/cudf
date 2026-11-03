/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/detail/utilities/cuda_memcpy.hpp>
#include <cudf/strings/string_view.cuh>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/exec_policy.hpp>

#include <cooperative_groups.h>
#include <cooperative_groups/scan.h>
#include <cub/block/block_exchange.cuh>
#include <cub/block/block_merge_sort.cuh>
#include <cub/block/block_reduce.cuh>
#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_run_length_encode.cuh>
#include <cub/warp/warp_merge_sort.cuh>
#include <cuda/buffer>
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
  uint32_t null_rank;
  __host__ __device__ bool operator==(string_radix_prefix_key const& rhs) const
  {
    return hi == rhs.hi && null_rank == rhs.null_rank;
  }
};

template <typename Key>
__device__ bool radix_key_is_null(Key const& key, uint32_t null_rank)
{
  if constexpr (std::is_same_v<Key, string_radix_prefix_key>)
    return key.null_rank == null_rank;
  else
    return false;
}

struct string_radix_decomposer {
  __host__ __device__ auto operator()(string_radix_prefix_key& key) const
  {
    return cuda::std::tuple<uint32_t&, uint64_t&>{key.null_rank, key.hi};
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

__host__ __device__ inline int radix_lrb_tile(int bin, int fixed_tile)
{
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
                                    radix_lrb_metadata* metadata,
                                    size_type const* row_ids,
                                    size_type size,
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
    auto const tiles  = valid ? 1 + (length - 1) / radix_lrb_tile(bin, FixedTile) : 0;
    auto const peers  = __match_any_sync(0xffffffffu, bin);
    if (valid && bin >= comp.proof_first_bin() &&
        comp.probe_run(row_ids, radix_segment{starts[i], starts[i] + length}, size)) {
      atomicAdd(counts, 1ULL);
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
      auto const tiles = 1 + (length - 1) / radix_lrb_tile(bin, FixedTile);
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

template <int Group, bool Stable, typename Comparator>
__device__ void radix_lrb_warp_sort_group(size_type* output,
                                          radix_segment const* segments,
                                          size_type segment,
                                          bool active,
                                          Comparator comp,
                                          radix_lrb_warp_storage& storage)
{
  auto const lane  = threadIdx.x & (Group - 1);
  auto const seg   = active ? segments[segment] : radix_segment{0, 0};
  bool const valid = active && lane < seg.end - seg.begin;
  size_type row    = valid ? output[seg.begin + lane] : -1;
  // Bitonic exchanges share one comparison per pair through eight lanes; CUB sorts larger groups.
  if constexpr (Group <= 8) {
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
  } else {
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

template <bool Stable, typename Comparator>
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
        radix_lrb_warp_sort_group<2, Stable>(
          output, segments, segment, active, comp, storage[threadIdx.x / 32]);
        break;
      case 2:
        radix_lrb_warp_sort_group<4, Stable>(
          output, segments, segment, active, comp, storage[threadIdx.x / 32]);
        break;
      case 3:
        radix_lrb_warp_sort_group<8, Stable>(
          output, segments, segment, active, comp, storage[threadIdx.x / 32]);
        break;
      case 4:
        radix_lrb_warp_sort_group<16, Stable>(
          output, segments, segment, active, comp, storage[threadIdx.x / 32]);
        break;
      case 5:
        radix_lrb_warp_sort_group<32, Stable>(
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
                                Comparator comp)
{
  auto const ordinal = static_cast<size_type>(blockIdx.x * blockDim.x + threadIdx.x);
  if (ordinal >= count) return;
  auto const seg       = segments[metadata->segment_base[comp.proof_first_bin()] + ordinal];
  auto const reference = output[seg.begin];
  auto const length    = comp.keys.template element<string_view>(reference).size_bytes();
  bool const full      = seg.begin == 0 && seg.end == size;
  bool const enabled   = comp.probe_run(output, seg, size);
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
                                 int fixed_log,
                                 Comparator comp)
{
  auto const first_bin = 6;
  auto const begin     = metadata->tile_base[first_bin];
  auto const end       = metadata->tile_base[32];
  for (auto task = begin + blockIdx.x; task < end; task += gridDim.x) {
    auto const bin    = radix_lrb_find_bin(task, metadata, first_bin, 32);
    auto const levels = bin < 9 ? 0 : max(0, bin - fixed_log);
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
    auto const tile_size = radix_lrb_tile(bin, FixedTile);
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
                      Comparator comp,
                      cuda::stream_ref stream,
                      rmm::device_async_resource_ref temp_mr)
{
  constexpr int fixed_tile = Threads * Items;
  constexpr int grid_warps = 32;
  int fixed_log            = 0;
  for (int tile = fixed_tile; tile > 1; tile /= 2)
    ++fixed_log;
  auto const capacity = size / 2;
  auto segments = cuda::device_buffer<radix_segment>(stream, temp_mr, capacity, cuda::no_init);
  auto offsets  = cuda::device_buffer<size_type>(stream, temp_mr, capacity + 32, cuda::no_init);
  auto metadata = cuda::device_buffer<radix_lrb_metadata>(stream, temp_mr, 1, cuda::no_init);
  CUDF_CUDA_TRY(cudaMemsetAsync(metadata.data(), 0, sizeof(radix_lrb_metadata), stream.get()));
  auto const build_blocks = std::max(1, std::min(256, (capacity + 255) / 256));
  radix_lrb_histogram<fixed_tile><<<build_blocks, 256, 0, stream.get()>>>(
    starts, lengths, run_count, sorted_keys, null_rank, metadata.data(), output, size, comp);
  CUDF_CUDA_TRY(cudaGetLastError());
  radix_lrb_prefix<<<1, 32, 0, stream.get()>>>(metadata.data(), offsets.data());
  CUDF_CUDA_TRY(cudaGetLastError());
  radix_lrb_scatter<fixed_tile><<<build_blocks, 256, 0, stream.get()>>>(starts,
                                                                        lengths,
                                                                        run_count,
                                                                        sorted_keys,
                                                                        null_rank,
                                                                        segments.data(),
                                                                        offsets.data(),
                                                                        metadata.data());
  CUDF_CUDA_TRY(cudaGetLastError());
  // One readback of the bin counts selects which workers and merge stages to launch.
  unsigned long long counts[32]{};
  CUDF_CUDA_TRY(
    cudf::detail::memcpy_async(counts, metadata.data()->counts, sizeof(counts), stream));
  stream.sync();
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
    return static_cast<int>(std::min<uint64_t>(tasks, max_blocks));
  };
  if (tiles(1, 32) == 0) return;
  auto execute = [&]<typename GiantComparator>(GiantComparator giant_comp) {
    uint64_t warp_tasks = 0;
    for (int bin = 1; bin <= 5; ++bin) {
      auto const groups = 32 >> bin;
      warp_tasks += (static_cast<uint32_t>(counts[bin]) + groups - 1) / groups;
    }
    if (warp_tasks != 0) {
      radix_lrb_warp_sort<Stable><<<blocks((warp_tasks + 7) / 8), 256, 0, stream.get()>>>(
        output, segments.data(), metadata.data(), comp);
      CUDF_CUDA_TRY(cudaGetLastError());
    }
    if (tiles(6, 8) != 0) {
      radix_lrb_block_sort<128, 1, Stable><<<blocks(tiles(6, 8), 128), 128, 0, stream.get()>>>(
        output, scratch, segments.data(), offsets.data(), metadata.data(), 6, 8, comp);
      CUDF_CUDA_TRY(cudaGetLastError());
    }
    if (tiles(8, 9) != 0) {
      radix_lrb_block_sort<256, 1, Stable><<<blocks(tiles(8, 9)), 256, 0, stream.get()>>>(
        output, scratch, segments.data(), offsets.data(), metadata.data(), 8, 9, comp);
      CUDF_CUDA_TRY(cudaGetLastError());
    }
    if (tiles(9, 32) != 0) {
      radix_lrb_block_sort<Threads, Items, Stable>
        <<<blocks(tiles(9, 32), Threads), Threads, 0, stream.get()>>>(
          output, scratch, segments.data(), offsets.data(), metadata.data(), 9, 32, giant_comp);
      CUDF_CUDA_TRY(cudaGetLastError());
    }
    auto* src = scratch;
    auto* dst = output;
    for (int run_log = fixed_log; run_log < 31 && (int64_t{1} << run_log) < size; ++run_log) {
      // Medium adaptive runs are already complete in their 128/256-row tile.
      // Giant runs can require several stages before the active bin advances.
      auto const first_bin    = std::max(9, run_log + 1);
      auto const active_tiles = tiles(first_bin, 32);
      if (active_tiles == 0) break;
      auto const run = int64_t{1} << run_log;
      radix_lrb_merge<Threads, Items><<<blocks(active_tiles, Threads), Threads, 0, stream.get()>>>(
        src, dst, segments.data(), offsets.data(), metadata.data(), first_bin, run, giant_comp);
      CUDF_CUDA_TRY(cudaGetLastError());
      std::swap(src, dst);
    }
    auto const finish_tiles = tiles(6, 32);
    if (finish_tiles != 0) {
      radix_lrb_finish<fixed_tile><<<blocks(finish_tiles), 256, 0, stream.get()>>>(
        scratch, output, segments.data(), offsets.data(), metadata.data(), fixed_log, giant_comp);
      CUDF_CUDA_TRY(cudaGetLastError());
    }
  };
  size_type eligible   = 0;
  auto const proof_bin = comp.proof_first_bin();
  for (int bin = proof_bin; bin < 32; ++bin)
    eligible += static_cast<uint32_t>(counts[bin]);
  if (eligible == 0 || counts[0] == 0) {
    execute(comp);
    return;
  }
  auto proofs = cuda::device_buffer<radix_lrb_proof>(stream, temp_mr, eligible, cuda::no_init);
  radix_lrb_probe<<<(eligible + 127) / 128, 128, 0, stream.get()>>>(
    output, segments.data(), metadata.data(), proofs.data(), eligible, size, comp);
  CUDF_CUDA_TRY(cudaGetLastError());
  radix_lrb_prove<Threads, Items>
    <<<blocks(tiles(proof_bin, 32), Threads), Threads, 0, stream.get()>>>(
      output, segments.data(), offsets.data(), metadata.data(), proofs.data(), comp);
  CUDF_CUDA_TRY(cudaGetLastError());
  if (eligible == 1 && tiles(proof_bin, 32) == static_cast<uint64_t>((size - 1) / fixed_tile + 1)) {
    radix_lrb_proof proof;
    CUDF_CUDA_TRY(cudf::detail::memcpy_async(&proof, proofs.data(), sizeof(proof), stream));
    stream.sync();
    if ((proof.flags & 4u) != 0) {
      if ((proof.flags & 1u) == 0) return;
      if constexpr (std::is_same_v<Key, uint64_t>) {
        if (comp.try_suffix_radix(
              proof, size, output, scratch, sorted_keys, radix_scratch_keys, stream, temp_mr))
          return;
      }
    }
  }
  execute(comp.make_proof_comparator(proofs.data(), metadata.data()));
}

/**
 * @brief Radix-sorts eight-byte prefixes, then refines equal-prefix runs with LRB workers.
 */
template <int Threads, int Items, bool Stable, typename Extractor, typename Comparator>
void radix_prefix_refine(size_type size,
                         size_type* output,
                         Extractor extractor,
                         Comparator comparator,
                         uint32_t null_rank,
                         cuda::stream_ref stream,
                         rmm::device_async_resource_ref temp_mr)
{
  CUDF_EXPECTS(size < std::numeric_limits<size_type>::max(), "Radix boundary count overflow");
  auto const policy = rmm::exec_policy_nosync(stream, temp_mr);
  auto rows         = cuda::counting_iterator<size_type>{0};
  using key_type    = decltype(extractor(size_type{0}));
  auto keys_in      = cuda::device_buffer<key_type>(stream, temp_mr, size, cuda::no_init);
  auto keys_out     = cuda::device_buffer<key_type>(stream, temp_mr, size, cuda::no_init);
  auto scratch      = cuda::device_buffer<size_type>(stream, temp_mr, size, cuda::no_init);
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
      return cub::DeviceRadixSort::SortPairs(storage,
                                             bytes,
                                             keys_in.data(),
                                             keys_out.data(),
                                             scratch.data(),
                                             output,
                                             size,
                                             string_radix_decomposer{},
                                             0,
                                             65,
                                             stream.get());
    }
  };
  CUDF_CUDA_TRY(radix_sort(nullptr));
  {
    auto temp = cuda::device_buffer<std::byte>(stream, temp_mr, bytes, cuda::no_init);
    CUDF_CUDA_TRY(radix_sort(temp.data()));
  }

  auto const max_runs = size / 2;
  auto offsets        = cuda::device_buffer<size_type>(stream, temp_mr, max_runs, cuda::no_init);
  auto lengths        = cuda::device_buffer<size_type>(stream, temp_mr, max_runs, cuda::no_init);
  auto run_count      = cuda::device_buffer<size_type>(stream, temp_mr, 1, cuda::no_init);
  auto encode         = [&](void* storage) {
    return cub::DeviceRunLengthEncode::NonTrivialRuns(storage,
                                                      bytes,
                                                      keys_out.data(),
                                                      offsets.data(),
                                                      lengths.data(),
                                                      run_count.data(),
                                                      size,
                                                      stream.get());
  };
  bytes = 0;
  CUDF_CUDA_TRY(encode(nullptr));
  {
    auto temp = cuda::device_buffer<std::byte>(stream, temp_mr, bytes, cuda::no_init);
    CUDF_CUDA_TRY(encode(temp.data()));
  }
  // keys_in is no longer needed after the radix pass; the optional suffix radix reuses it.
  radix_lrb_refine<Threads, Items, Stable>(size,
                                           output,
                                           scratch.data(),
                                           offsets.data(),
                                           lengths.data(),
                                           run_count.data(),
                                           keys_out.data(),
                                           keys_in.data(),
                                           null_rank,
                                           comparator,
                                           stream,
                                           temp_mr);
}
}  // namespace cudf::detail
