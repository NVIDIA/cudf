/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "string_sort_config.hpp"

#include <cudf/column/column_device_view.cuh>
#include <cudf/detail/device_scalar.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/detail/utilities/cuda_memcpy.hpp>
#include <cudf/detail/utilities/grid_1d.cuh>
#include <cudf/strings/string_view.cuh>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/device_buffer.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <cooperative_groups.h>
#include <cub/block/block_reduce.cuh>
#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_scan.cuh>
#include <cub/device/device_segmented_sort.cuh>
#include <cuda/pipeline>
#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/fill.h>
#include <thrust/functional.h>
#include <thrust/scan.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/transform_reduce.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <utility>

// CUDA's block pipeline state is intentionally initialized cooperatively in shared memory.
#pragma nv_diag_suppress static_var_with_dynamic_init

namespace cudf::detail {
namespace segmented_string_sort {

constexpr size_type comparison_chunk_size   = 512;
constexpr size_type radix_prefix_bytes      = sizeof(std::uint64_t);
constexpr size_type maximum_string_size     = std::numeric_limits<size_type>::max();
constexpr size_type invalid_index           = -1;
constexpr size_type shuffle_block_size      = 256;
constexpr size_type strings_per_shuffle_cta = 2048;
constexpr size_type shuffle_copy_bytes      = 2048;
constexpr size_type shuffle_pipeline_stages = 2;

__device__ constexpr std::uint64_t byte_swap(std::uint64_t value)
{
  value = ((value & 0x00ff00ff00ff00ffULL) << 8) | ((value & 0xff00ff00ff00ff00ULL) >> 8);
  value = ((value & 0x0000ffff0000ffffULL) << 16) | ((value & 0xffff0000ffff0000ULL) >> 16);
  return (value << 32) | (value >> 32);
}

struct finish_counts {
  size_type segments{};
  size_type rows{};
};

struct refinement_counts {
  size_type candidate_starts{};
  size_type candidate_ends{};
  size_type continuing_runs{};
  size_type continuing_rows{};
  size_type completed_runs{};
  size_type duplicate_rows{};
  size_type completed_rows{};
  size_type finish_runs{};
  size_type finish_rows{};
  size_type minimum_prefix{maximum_string_size};
  size_type maximum_prefix{};
};

struct remaining_bounds {
  size_type minimum{maximum_string_size};
  size_type maximum{};
};

struct endpoint_offset {
  size_type stride;
  size_type offset;

  __host__ __device__ size_type operator()(size_type segment) const
  {
    return segment * stride + offset;
  }
};

struct combine_remaining_bounds {
  __host__ __device__ remaining_bounds operator()(remaining_bounds lhs, remaining_bounds rhs) const
  {
    return {lhs.minimum < rhs.minimum ? lhs.minimum : rhs.minimum,
            lhs.maximum > rhs.maximum ? lhs.maximum : rhs.maximum};
  }
};

struct remaining_length {
  std::uint8_t const* byte_counts;

  __device__ remaining_bounds operator()(size_type row) const
  {
    auto const remaining = static_cast<size_type>(byte_counts[row]);
    return {remaining, remaining};
  }
};

__device__ inline bool strings_equal_after(column_device_view strings,
                                           size_type lhs,
                                           size_type rhs,
                                           size_type known_prefix_bytes);

__device__ inline size_type find_run_slot(size_type position,
                                          size_type const* run_begins,
                                          size_type const* run_ends,
                                          size_type num_runs)
{
  auto low  = size_type{0};
  auto high = num_runs;
  while (low < high) {
    auto const middle = low + (high - low) / 2;
    if (run_ends[middle] <= position) {
      low = middle + 1;
    } else {
      high = middle;
    }
  }
  return low < num_runs && run_begins[low] <= position ? low : invalid_index;
}

__device__ inline bool find_containing_run(size_type position,
                                           size_type const* run_begins,
                                           size_type const* run_ends,
                                           size_type num_runs,
                                           size_type size,
                                           size_type& begin,
                                           size_type& end)
{
  if (num_runs == 0) {
    begin = 0;
    end   = size;
    return position < size;
  }
  auto const low = find_run_slot(position, run_begins, run_ends, num_runs);
  if (low == invalid_index) { return false; }
  begin = run_begins[low];
  end   = run_ends[low];
  return true;
}

CUDF_KERNEL void reduce_run_remaining_lengths(remaining_length get_remaining,
                                              size_type const* indices,
                                              size_type size,
                                              size_type const* begins,
                                              size_type const* ends,
                                              size_type run_count,
                                              remaining_bounds* bounds)
{
  constexpr size_type tile_items = 1024;
  auto const tile_begin          = static_cast<size_type>(blockIdx.x) * tile_items;
  auto const tile_end            = tile_begin + min(size - tile_begin, tile_items);
  __shared__ size_type tile_run;
  if (threadIdx.x == 0) {
    auto const slot = find_run_slot(tile_begin, begins, ends, run_count);
    tile_run        = slot != invalid_index && ends[slot] >= tile_end ? slot : invalid_index;
  }
  __syncthreads();
  if (tile_run != invalid_index) {
    remaining_bounds local;
    for (auto position = int64_t{tile_begin} + threadIdx.x; position < tile_end;
         position += blockDim.x) {
      local = combine_remaining_bounds{}(local, get_remaining(indices[position]));
    }
    using block_reduce = cub::BlockReduce<remaining_bounds, 256>;
    __shared__ typename block_reduce::TempStorage scratch;
    auto const reduced = block_reduce(scratch).Reduce(local, combine_remaining_bounds{});
    if (threadIdx.x == 0) {
      atomicMin(&bounds[tile_run].minimum, reduced.minimum);
      atomicMax(&bounds[tile_run].maximum, reduced.maximum);
    }
  } else {
    // Only tiles crossing run boundaries need per-row atomics. Large runs otherwise contribute
    // one pair of bounds per tile, avoiding both serial walks and a contended atomic per row.
    for (auto position = int64_t{tile_begin} + threadIdx.x; position < tile_end;
         position += blockDim.x) {
      auto const slot = find_run_slot(static_cast<size_type>(position), begins, ends, run_count);
      if (slot == invalid_index) { continue; }
      auto const remaining = get_remaining(indices[position]);
      atomicMin(&bounds[slot].minimum, remaining.minimum);
      atomicMax(&bounds[slot].maximum, remaining.maximum);
    }
  }
}

CUDF_KERNEL void mark_tied_run_endpoints(std::uint64_t const* sorted_keys,
                                         size_type size,
                                         size_type const* previous_begins,
                                         size_type const* previous_ends,
                                         size_type previous_run_count,
                                         size_type* candidate_begins,
                                         size_type* candidate_ends,
                                         refinement_counts* counts)
{
  auto const position = cudf::detail::grid_1d::global_thread_id();
  if (position >= size) { return; }

  auto segment_begin = size_type{};
  auto segment_end   = size_type{};
  if (!find_containing_run(position,
                           previous_begins,
                           previous_ends,
                           previous_run_count,
                           size,
                           segment_begin,
                           segment_end)) {
    return;
  }

  auto const begins_tie =
    position + 1 < segment_end && sorted_keys[position] == sorted_keys[position + 1] &&
    (position == segment_begin || sorted_keys[position - 1] != sorted_keys[position]);
  auto const ends_tie =
    position > segment_begin && sorted_keys[position - 1] == sorted_keys[position] &&
    (position + 1 == segment_end || sorted_keys[position] != sorted_keys[position + 1]);
  if (begins_tie) {
    auto const slot        = atomicAdd(&counts->candidate_starts, size_type{1});
    candidate_begins[slot] = position;
  }
  if (ends_tie) {
    auto const slot      = atomicAdd(&counts->candidate_ends, size_type{1});
    candidate_ends[slot] = position + 1;
  }
}

template <bool eliminate_exact_duplicates>
CUDF_KERNEL void classify_tied_runs(remaining_bounds const* bounds,
                                    column_device_view strings,
                                    size_type const* indices,
                                    size_type const* candidate_begins,
                                    size_type const* candidate_ends,
                                    size_type candidate_count,
                                    size_type byte_offset,
                                    size_type radix_run_min,
                                    bool last_pass,
                                    bool trace,
                                    size_type* continuing_begins,
                                    size_type* continuing_ends,
                                    size_type* final_begins,
                                    size_type* final_ends,
                                    size_type* final_prefix_bytes,
                                    refinement_counts* refinement,
                                    finish_counts* finish)
{
  auto const run = static_cast<size_type>(blockIdx.x);
  if (run >= candidate_count) { return; }
  auto const begin = candidate_begins[run];
  auto const end   = candidate_ends[run];

  auto const minimum_bytes = bounds[run].minimum;
  auto const maximum_bytes = bounds[run].maximum;
  using block_reduce       = cub::BlockReduce<size_type, 256>;
  __shared__ typename block_reduce::TempStorage reduction_storage;

  auto const full_chunk    = minimum_bytes >= radix_prefix_bytes;
  auto const proven_prefix = full_chunk ? byte_offset + radix_prefix_bytes : byte_offset;
  if (trace && threadIdx.x == 0) {
    atomicMin(&refinement->minimum_prefix, proven_prefix);
    atomicMax(&refinement->maximum_prefix, proven_prefix);
  }
  if (maximum_bytes <= radix_prefix_bytes && minimum_bytes == maximum_bytes) {
    if (threadIdx.x == 0) {
      atomicAdd(&refinement->completed_runs, size_type{1});
      if (trace) { atomicAdd(&refinement->completed_rows, end - begin); }
    }
    return;
  }

  if (full_chunk && !last_pass && end - begin >= radix_run_min) {
    if (threadIdx.x != 0) { return; }
    auto const slot         = atomicAdd(&refinement->continuing_runs, size_type{1});
    continuing_begins[slot] = begin;
    continuing_ends[slot]   = end;
    atomicAdd(&refinement->continuing_rows, end - begin);
    return;
  }

  if constexpr (eliminate_exact_duplicates) {
    // Only terminal unresolved runs pay for exact verification. Comparing after the proven prefix
    // avoids rereading radix bytes without confusing a short value with a zero-padded longer one.
    auto mismatch = size_type{0};
    for (auto position = begin + static_cast<size_type>(threadIdx.x); position < end;
         position += static_cast<size_type>(blockDim.x)) {
      mismatch |= !strings_equal_after(strings, indices[begin], indices[position], proven_prefix);
    }
    __syncthreads();
    mismatch = block_reduce(reduction_storage).Reduce(mismatch, cuda::maximum<>{});
    if (threadIdx.x != 0) { return; }
    if (mismatch == 0) {
      atomicAdd(&refinement->completed_runs, size_type{1});
      atomicAdd(&refinement->duplicate_rows, end - begin);
      if (trace) { atomicAdd(&refinement->completed_rows, end - begin); }
      return;
    }
  } else {
    if (threadIdx.x != 0) { return; }
  }

  auto const slot          = atomicAdd(&finish->segments, size_type{1});
  final_begins[slot]       = begin;
  final_ends[slot]         = end;
  final_prefix_bytes[slot] = proven_prefix;
  atomicAdd(&finish->rows, end - begin);
  if (trace) {
    atomicAdd(&refinement->finish_runs, size_type{1});
    atomicAdd(&refinement->finish_rows, end - begin);
  }
}

__device__ inline bool strings_equal_after(column_device_view strings,
                                           size_type lhs,
                                           size_type rhs,
                                           size_type known_prefix_bytes)
{
  auto const left  = strings.element<string_view>(lhs);
  auto const right = strings.element<string_view>(rhs);
  if (left.size_bytes() != right.size_bytes()) { return false; }
  for (auto byte = known_prefix_bytes; byte < left.size_bytes(); ++byte) {
    if (left.data()[byte] != right.data()[byte]) { return false; }
  }
  return true;
}

template <typename Offset>
CUDF_KERNEL __launch_bounds__(shuffle_block_size) void make_first_keys(Offset const* offsets,
                                                                       char const* chars,
                                                                       size_type size,
                                                                       std::uint64_t* keys,
                                                                       std::uint8_t* byte_counts)
{
  __shared__ __align__(16) Offset shared_offsets[strings_per_shuffle_cta + 1];
  __shared__ __align__(16) char shared_chars[shuffle_copy_bytes * shuffle_pipeline_stages];
  __shared__
    cuda::pipeline_shared_state<cuda::thread_scope::thread_scope_block, shuffle_pipeline_stages>
      pipeline_state;

  auto const block       = cooperative_groups::this_thread_block();
  auto pipeline          = cuda::make_pipeline(block, &pipeline_state);
  auto const block_begin = static_cast<size_type>(blockIdx.x) * strings_per_shuffle_cta;
  auto const block_size  = min(strings_per_shuffle_cta, size - block_begin);
  for (auto index = static_cast<size_type>(threadIdx.x); index <= block_size;
       index += static_cast<size_type>(blockDim.x)) {
    shared_offsets[index] = offsets[block_begin + index];
  }
  block.sync();

  auto const chars_begin = static_cast<Offset>(shared_offsets[0] & ~Offset{15});
  auto const chars_end   = shared_offsets[block_size];
  // A large-offset column can span more than INT32_MAX bytes within one CTA even when each
  // individual string fits size_type. Keep window arithmetic in the offsets' representation.
  auto const char_count = chars_end - chars_begin;
  auto const iterations = max(size_type{1},
                              static_cast<size_type>(char_count / shuffle_copy_bytes +
                                                     (char_count % shuffle_copy_bytes != 0)));
  auto next_copy        = size_type{0};

  auto issue_copy = [&](size_type stage) {
    auto const copied = Offset{next_copy} * shuffle_copy_bytes;
    auto const bytes =
      static_cast<size_type>(min(Offset{shuffle_copy_bytes}, max(Offset{0}, char_count - copied)));
    pipeline.producer_acquire();
    if (bytes == shuffle_copy_bytes) {
      // An explicit alignment proof enables the source's sixteen-byte vector copies for full
      // stages. Tail copies cannot make the same guarantee.
      cuda::memcpy_async(block,
                         reinterpret_cast<int4*>(shared_chars + stage * shuffle_copy_bytes),
                         reinterpret_cast<int4 const*>(chars + chars_begin + copied),
                         cuda::aligned_size_t<16>{static_cast<std::size_t>(bytes)},
                         pipeline);
    } else {
      cuda::memcpy_async(block,
                         shared_chars + stage * shuffle_copy_bytes,
                         chars + chars_begin + copied,
                         static_cast<std::size_t>(bytes),
                         pipeline);
    }
    pipeline.producer_commit();
    ++next_copy;
  };
  issue_copy(0);

  auto local_row  = static_cast<size_type>(threadIdx.x);
  auto byte       = size_type{0};
  auto key        = std::uint64_t{0};
  auto skip_empty = [&] {
    while (local_row < block_size && shared_offsets[local_row] == shared_offsets[local_row + 1]) {
      keys[block_begin + local_row]        = 0;
      byte_counts[block_begin + local_row] = 0;
      local_row += static_cast<size_type>(blockDim.x);
    }
  };
  skip_empty();

  for (auto iteration = size_type{0}; iteration < iterations; ++iteration) {
    auto const current_stage = iteration % shuffle_pipeline_stages;
    if (next_copy < iterations) { issue_copy(next_copy % shuffle_pipeline_stages); }
    pipeline.consumer_wait();
    auto const window_begin = Offset{iteration} * shuffle_copy_bytes;
    auto const window_end =
      window_begin + min(Offset{shuffle_copy_bytes}, char_count - window_begin);

    while (local_row < block_size) {
      auto const row_begin = shared_offsets[local_row] - chars_begin;
      auto const row_size =
        static_cast<size_type>(shared_offsets[local_row + 1] - shared_offsets[local_row]);
      auto const key_size = min(size_type{8}, row_size);
      if (row_begin + byte >= window_end) { break; }
      if (byte == 0 && key_size == radix_prefix_bytes && row_begin + key_size <= window_end) {
        auto const aligned_begin = row_begin & ~(radix_prefix_bytes - 1);
        auto const shift         = static_cast<unsigned int>((row_begin - aligned_begin) * 8);
        if (window_end - aligned_begin >= radix_prefix_bytes * (shift == 0 ? 1 : 2)) {
          auto const* words = reinterpret_cast<std::uint64_t const*>(
            shared_chars + current_stage * shuffle_copy_bytes + aligned_begin - window_begin);
          auto word = words[0] >> shift;
          if (shift != 0) { word |= words[1] << (64 - shift); }
          key  = byte_swap(word);
          byte = key_size;
        }
      }
      while (byte < key_size && row_begin + byte < window_end) {
        key = (key << 8) |
              static_cast<unsigned char>(
                shared_chars[current_stage * shuffle_copy_bytes + row_begin + byte - window_begin]);
        ++byte;
      }
      if (byte != key_size) { break; }
      keys[block_begin + local_row] = key << (8 * (8 - key_size));
      byte_counts[block_begin + local_row] =
        static_cast<std::uint8_t>(min(row_size, radix_prefix_bytes + 1));
      local_row += static_cast<size_type>(blockDim.x);
      byte = 0;
      key  = 0;
      skip_empty();
    }
    pipeline.consumer_release();
  }
}

template <typename Offset>
CUDF_KERNEL void make_subsequent_keys(Offset const* offsets,
                                      char const* chars,
                                      size_type strings_offset,
                                      size_type const* indices,
                                      size_type const* run_begins,
                                      size_type const* run_ends,
                                      size_type num_runs,
                                      std::uint64_t* keys,
                                      std::uint8_t* byte_counts,
                                      size_type size,
                                      size_type byte_offset)
{
  auto const block_begin = static_cast<size_type>(blockIdx.x) * strings_per_shuffle_cta;
  auto const block_rows  = min(strings_per_shuffle_cta, size - block_begin);
  auto const block_end   = block_begin + block_rows;
  auto const lane        = static_cast<size_type>(threadIdx.x);
  auto const threads     = static_cast<size_type>(blockDim.x);
  __shared__ size_type first_run;
  __shared__ size_type last_run;
  __shared__ __align__(16) size_type block_indices[strings_per_shuffle_cta];
  if (lane == 0) {
    auto low  = size_type{0};
    auto high = num_runs;
    while (low < high) {
      auto const middle = low + (high - low) / 2;
      if (run_ends[middle] <= block_begin) {
        low = middle + 1;
      } else {
        high = middle;
      }
    }
    first_run = low;
    high      = num_runs;
    while (low < high) {
      auto const middle = low + (high - low) / 2;
      if (run_begins[middle] < block_end) {
        low = middle + 1;
      } else {
        high = middle;
      }
    }
    last_run = low;
  }
  __syncthreads();
  if (num_runs > 0 && first_run == last_run) { return; }

  // The source amortizes each CTA over 2,048 rows. Shared index staging and a block-local run
  // interval avoid launching a CTA per 256 rows and searching every run for every input row.
  if (block_rows == strings_per_shuffle_cta) {
    auto const* input_vectors = reinterpret_cast<int4 const*>(indices + block_begin);
    auto* shared_vectors      = reinterpret_cast<int4*>(block_indices);
    for (auto index = lane; index < strings_per_shuffle_cta / 4; index += threads) {
      shared_vectors[index] = input_vectors[index];
    }
  } else {
    for (auto index = lane; index < block_rows; index += threads) {
      block_indices[index] = indices[block_begin + index];
    }
  }
  __syncthreads();

  for (auto local_row = lane; local_row < block_rows; local_row += threads) {
    auto const position = block_begin + local_row;
    if (num_runs > 0 && find_run_slot(position,
                                      run_begins + first_run,
                                      run_ends + first_run,
                                      last_run - first_run) == invalid_index) {
      continue;
    }
    auto const row       = block_indices[local_row];
    auto const row_begin = offsets[strings_offset + row];
    auto const row_end   = offsets[strings_offset + row + 1];
    auto const remaining =
      row_end - row_begin > byte_offset ? row_end - row_begin - byte_offset : 0;
    byte_counts[row]    = static_cast<std::uint8_t>(min(Offset{radix_prefix_bytes + 1}, remaining));
    auto const key_size = min(Offset{radix_prefix_bytes}, remaining);
    auto key            = std::uint64_t{0};
    if (key_size == radix_prefix_bytes) {
      auto const start         = row_begin + byte_offset;
      auto const aligned_start = start & ~Offset{radix_prefix_bytes - 1};
      auto const shift         = static_cast<unsigned int>((start - aligned_start) * 8);
      auto const* words        = reinterpret_cast<std::uint64_t const*>(chars + aligned_start);
      auto native_word         = words[0] >> shift;
      if (shift != 0) {
        // Reading a complete aligned word is safe only inside the chars allocation. A final
        // misaligned string may require a few tail-byte loads instead of the reference's overread.
        if (offsets[strings_offset + size] - aligned_start >= 2 * radix_prefix_bytes) {
          native_word |= words[1] << (64 - shift);
        } else {
          auto const loaded_bytes = radix_prefix_bytes - shift / 8;
          for (auto byte = loaded_bytes; byte < radix_prefix_bytes; ++byte) {
            native_word |= std::uint64_t{static_cast<unsigned char>(chars[start + byte])}
                           << (8 * byte);
          }
        }
      }
      key = byte_swap(native_word);
    } else {
      for (Offset byte = 0; byte < key_size; ++byte) {
        key = (key << 8) | static_cast<unsigned char>(chars[row_begin + byte_offset + byte]);
      }
      if (key_size > 0) { key <<= 8 * (radix_prefix_bytes - key_size); }
    }
    keys[position] = key;
  }
}

struct valid_row_predicate {
  __device__ bool operator()(size_type row) const { return not strings.is_null(row); }
  column_device_view strings;
};

struct null_row_predicate {
  __device__ bool operator()(size_type row) const { return strings.is_null(row); }
  column_device_view strings;
};

struct comparison_value {
  size_type row;
  size_type bytes;
  char const* data;
};

template <typename Offset>
__device__ comparison_value load_comparison_value(size_type row,
                                                  size_type known_prefix_bytes,
                                                  Offset const* offsets,
                                                  char const* chars)
{
  if (row == invalid_index) { return {row, 0, nullptr}; }
  auto const begin = offsets[row];
  auto const bytes = static_cast<size_type>(offsets[row + 1] - begin) - known_prefix_bytes;
  return {row, bytes, chars + begin + known_prefix_bytes};
}

template <bool ascending>
__device__ bool stable_valid_string_less(comparison_value lhs, comparison_value rhs)
{
  // Equal byte bounds give the existing comparator one loop limit, matching the source's finish.
  auto const common_bytes = min(lhs.bytes, rhs.bytes);
  auto const comparison =
    string_view{lhs.data, common_bytes}.compare(string_view{rhs.data, common_bytes});
  if (comparison != 0) { return ascending ? comparison < 0 : comparison > 0; }
  if (lhs.bytes != rhs.bytes) { return ascending ? lhs.bytes < rhs.bytes : lhs.bytes > rhs.bytes; }
  return lhs.row < rhs.row;
}

template <bool ascending>
__device__ bool stable_string_less(comparison_value lhs, comparison_value rhs)
{
  if (lhs.row == invalid_index) { return false; }
  if (rhs.row == invalid_index) { return true; }
  return stable_valid_string_less<ascending>(lhs, rhs);
}

CUDF_KERNEL void compute_finish_chunk_counts(size_type const* final_begins,
                                             size_type const* final_ends,
                                             size_type num_segments,
                                             size_type* chunk_counts)
{
  auto const segment = cudf::detail::grid_1d::global_thread_id();
  if (segment >= num_segments) { return; }
  auto const length     = final_ends[segment] - final_begins[segment];
  chunk_counts[segment] = length / comparison_chunk_size + (length % comparison_chunk_size != 0);
}

CUDF_KERNEL void compute_total_finish_chunks(size_type const* chunk_offsets,
                                             size_type const* chunk_counts,
                                             size_type num_segments,
                                             size_type* total_chunks)
{
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    *total_chunks = chunk_offsets[num_segments - 1] + chunk_counts[num_segments - 1];
  }
}

CUDF_KERNEL void expand_finish_chunks(size_type const* final_begins,
                                      size_type const* final_ends,
                                      size_type const* final_prefix_bytes,
                                      size_type const* chunk_offsets,
                                      size_type num_segments,
                                      size_type* chunk_begins,
                                      size_type* chunk_sizes,
                                      size_type* chunk_run_offsets,
                                      size_type* chunk_run_counts,
                                      size_type* chunk_prefix_bytes)
{
  auto const segment = cudf::detail::grid_1d::global_thread_id();
  if (segment >= num_segments) { return; }
  auto const run_begin  = final_begins[segment];
  auto const run_length = final_ends[segment] - run_begin;
  auto const run_offset = chunk_offsets[segment];
  auto const chunk_count =
    run_length / comparison_chunk_size + (run_length % comparison_chunk_size != 0);
  for (size_type chunk = 0; chunk < chunk_count; ++chunk) {
    auto const slot          = run_offset + chunk;
    auto const begin         = run_begin + chunk * comparison_chunk_size;
    chunk_begins[slot]       = begin;
    chunk_sizes[slot]        = min(comparison_chunk_size, final_ends[segment] - begin);
    chunk_run_offsets[slot]  = run_offset;
    chunk_run_counts[slot]   = chunk_count;
    chunk_prefix_bytes[slot] = final_prefix_bytes[segment];
  }
}

template <bool ascending, typename Offset>
CUDF_KERNEL __launch_bounds__(comparison_chunk_size, 1) void bitonic_sort_finish_chunks(
  size_type const* input,
  size_type* sorted_chunks,
  size_type const* chunk_begins,
  size_type const* chunk_sizes,
  size_type const* chunk_prefix_bytes,
  Offset const* offsets,
  char const* chars)
{
  // Cache string metadata as in the source network so comparisons do not reload column offsets.
  __shared__ comparison_value values[comparison_chunk_size];
  auto const chunk              = static_cast<size_type>(blockIdx.x);
  auto const lane               = static_cast<size_type>(threadIdx.x);
  auto const begin              = chunk_begins[chunk];
  auto const length             = chunk_sizes[chunk];
  auto const known_prefix_bytes = chunk_prefix_bytes[chunk];
  values[lane]                  = load_comparison_value(
    lane < length ? input[begin + lane] : invalid_index, known_prefix_bytes, offsets, chars);
  __syncthreads();

  auto sort_size = comparison_chunk_size;
  if (length < comparison_chunk_size) {
    sort_size = length - 1;
    sort_size |= sort_size >> 1;
    sort_size |= sort_size >> 2;
    sort_size |= sort_size >> 4;
    sort_size |= sort_size >> 8;
    sort_size |= sort_size >> 16;
    ++sort_size;
  }
  for (size_type sequence = 2; sequence <= sort_size; sequence <<= 1) {
    for (size_type stride = sequence >> 1; stride > 0; stride >>= 1) {
      auto const peer = lane ^ stride;
      if (lane < sort_size && peer > lane) {
        auto const ascending_network = (lane & sequence) == 0;
        auto const should_swap       = ascending_network
                                         ? stable_string_less<ascending>(values[peer], values[lane])
                                         : stable_string_less<ascending>(values[lane], values[peer]);
        if (should_swap) {
          auto const temporary = values[lane];
          values[lane]         = values[peer];
          values[peer]         = temporary;
        }
      }
      __syncthreads();
    }
  }
  if (lane < length) { sorted_chunks[begin + lane] = values[lane].row; }
}

template <bool ascending, typename Offset>
CUDF_KERNEL __launch_bounds__(comparison_chunk_size,
                              1) void merge_all_sibling_chunks(size_type* output,
                                                               size_type const* sorted_chunks,
                                                               size_type const* chunk_begins,
                                                               size_type const* chunk_sizes,
                                                               size_type const* chunk_run_offsets,
                                                               size_type const* chunk_run_counts,
                                                               size_type const* chunk_prefix_bytes,
                                                               Offset const* offsets,
                                                               char const* chars)
{
  // Reconstruct addresses from global offsets instead of caching pointers in shared memory; this
  // keeps the source's smaller merge workspace and exposes the global origin of comparison bytes.
  __shared__ size_type sibling_rows[comparison_chunk_size];
  __shared__ size_type sibling_bytes[comparison_chunk_size];
  auto const chunk              = static_cast<size_type>(blockIdx.x);
  auto const lane               = static_cast<size_type>(threadIdx.x);
  auto const begin              = chunk_begins[chunk];
  auto const length             = chunk_sizes[chunk];
  auto const run_offset         = chunk_run_offsets[chunk];
  auto const chunk_count        = chunk_run_counts[chunk];
  auto const known_prefix_bytes = chunk_prefix_bytes[chunk];
  auto const local_chunk        = chunk - run_offset;
  auto const active             = lane < length;
  auto const value              = load_comparison_value(
    active ? sorted_chunks[begin + lane] : invalid_index, known_prefix_bytes, offsets, chars);
  auto rank = lane;

  for (size_type sibling = 0; sibling < chunk_count; ++sibling) {
    if (sibling == local_chunk) { continue; }
    __syncthreads();
    auto const sibling_slot   = run_offset + sibling;
    auto const sibling_begin  = chunk_begins[sibling_slot];
    auto const sibling_length = chunk_sizes[sibling_slot];
    auto const sibling_value  = load_comparison_value(
      lane < sibling_length ? sorted_chunks[sibling_begin + lane] : invalid_index,
      known_prefix_bytes,
      offsets,
      chars);
    sibling_rows[lane]  = sibling_value.row;
    sibling_bytes[lane] = sibling_value.bytes;
    __syncthreads();
    if (!active) { continue; }

    // The original row index makes the comparison a strict total order. The resulting lower-bound
    // rank is equivalent to the source's earlier-chunk upper/later-chunk lower bounds while also
    // preserving stability if radix refinement has rearranged equal rows across chunk boundaries.
    auto lower = size_type{0};
    auto upper = sibling_length;
    while (lower < upper) {
      auto const middle = lower + (upper - lower) / 2;
      // Only active lanes search, and the bound excludes padded sibling entries.
      auto const sibling_row = sibling_rows[middle];
      auto const candidate   = comparison_value{
        sibling_row, sibling_bytes[middle], chars + offsets[sibling_row] + known_prefix_bytes};
      if (stable_valid_string_less<ascending>(candidate, value)) {
        lower = middle + 1;
      } else {
        upper = middle;
      }
    }
    rank += lower;
  }
  if (active) { output[chunk_begins[run_offset] + rank] = value.row; }
}

inline void segmented_key_sort(cub::DoubleBuffer<std::uint64_t>& keys,
                               cub::DoubleBuffer<size_type>& values,
                               size_type size,
                               size_type num_segments,
                               size_type const* segment_begins,
                               size_type const* segment_ends,
                               bool ascending,
                               rmm::device_buffer& temp_storage,
                               cuda::stream_ref stream)
{
  std::size_t temp_storage_bytes = 0;
  auto invoke                    = [&](void* temp_storage) {
    if (ascending) {
      return cub::DeviceSegmentedSort::StableSortPairs(temp_storage,
                                                       temp_storage_bytes,
                                                       keys,
                                                       values,
                                                       size,
                                                       num_segments,
                                                       segment_begins,
                                                       segment_ends,
                                                       stream.get());
    }
    return cub::DeviceSegmentedSort::StableSortPairsDescending(temp_storage,
                                                               temp_storage_bytes,
                                                               keys,
                                                               values,
                                                               size,
                                                               num_segments,
                                                               segment_begins,
                                                               segment_ends,
                                                               stream.get());
  };
  CUDF_CUDA_TRY(invoke(nullptr));
  if (temp_storage_bytes > temp_storage.size()) {
    temp_storage =
      rmm::device_buffer(temp_storage_bytes, stream, cudf::get_current_device_resource_ref());
  }
  CUDF_CUDA_TRY(invoke(temp_storage.data()));
}

inline void global_radix_sort(cub::DoubleBuffer<std::uint64_t>& keys,
                              cub::DoubleBuffer<size_type>& values,
                              size_type size,
                              bool ascending,
                              rmm::device_buffer& temp_storage,
                              cuda::stream_ref stream)
{
  std::size_t temp_storage_bytes = 0;
  auto invoke                    = [&](void* storage) {
    if (ascending) {
      return cub::DeviceRadixSort::SortPairs(
        storage, temp_storage_bytes, keys, values, size, 0, 64, stream.get());
    }
    return cub::DeviceRadixSort::SortPairsDescending(
      storage, temp_storage_bytes, keys, values, size, 0, 64, stream.get());
  };
  CUDF_CUDA_TRY(invoke(nullptr));
  if (temp_storage_bytes > temp_storage.size()) {
    temp_storage =
      rmm::device_buffer(temp_storage_bytes, stream, cudf::get_current_device_resource_ref());
  }
  CUDF_CUDA_TRY(invoke(temp_storage.data()));
}

/**
 * @brief Sort string row indices using iterative big-endian prefix refinement.
 *
 * The work decomposition follows string_sort aa13a98: global prefix sorting, tied-run refinement,
 * 512-item bitonic chunks, and stable ranking against every sibling chunk. Length classification
 * protects libcudf lexical semantics when zero padding cannot establish a complete equal prefix.
 */
template <typename Comparator>
void sorted_order(column_view const& input,
                  mutable_column_view& output,
                  bool ascending,
                  null_order null_precedence,
                  Comparator comparator,
                  segmented_string_sort_config const& tuning,
                  cuda::stream_ref stream)
{
  auto const size         = input.size();
  auto const temp_mr      = cudf::get_current_device_resource_ref();
  auto const strings_view = strings_column_view{input};
  auto const offsets      = strings_view.offsets();
  auto const chars        = strings_view.chars_begin(stream);
  auto const exec         = rmm::exec_policy_nosync(stream, temp_mr);
  auto const valid_size   = size - input.null_count();
  auto const null_size    = size - valid_size;
  auto indices_a          = rmm::device_uvector<size_type>(valid_size, stream, temp_mr);
  auto indices_b          = rmm::device_uvector<size_type>(valid_size, stream, temp_mr);
  auto null_indices       = rmm::device_uvector<size_type>(null_size, stream, temp_mr);
  auto const rows         = cuda::counting_iterator<size_type>{0};
  if (null_size == 0) {
    thrust::sequence(exec, indices_a.begin(), indices_a.end(), 0);
  } else {
    thrust::copy_if(
      exec, rows, rows + size, indices_a.begin(), valid_row_predicate{comparator.d_column});
    thrust::copy_if(
      exec, rows, rows + size, null_indices.begin(), null_row_predicate{comparator.d_column});
  }

  auto const maximum_radix_passes = static_cast<size_type>(tuning.lexic_precision);
  auto synchronization_points     = size_type{0};
  auto trace_readbacks            = size_type{0};
  auto const radix_run_min        = static_cast<size_type>(tuning.radix_run_min);
  if (tuning.trace) {
    std::fprintf(stderr,
                 "segmented-string-sort bytes=%d precision=%d radix-run-min=%d "
                 "exact-duplicates=%d valid=%d nulls=%d\n",
                 radix_prefix_bytes,
                 maximum_radix_passes,
                 tuning.radix_run_min,
                 tuning.eliminate_exact_duplicates,
                 valid_size,
                 null_size);
  }

  // Extraction already knows each length. A clipped byte-count sidecar avoids random offset
  // gathers after sorting; nine distinguishes a full nonterminal chunk from terminal lengths
  // zero through eight. The unused tail of the key allocation survives radix and bounds reuse.
  auto const sidecar_words =
    (static_cast<std::size_t>(size) + sizeof(std::uint64_t) - 1) / sizeof(std::uint64_t);
  auto keys_in = rmm::device_uvector<std::uint64_t>(
    static_cast<std::size_t>(valid_size) + sidecar_words, stream, temp_mr);
  auto keys_out     = rmm::device_uvector<std::uint64_t>(valid_size, stream, temp_mr);
  auto* byte_counts = reinterpret_cast<std::uint8_t*>(keys_in.data() + valid_size);

  // Continuing runs contain at least radix_run_min rows; cutoff runs never enter these arrays.
  // Reserve one slot for the virtual first-pass run even on smaller inputs.
  auto const maximum_continuing_runs = std::max(size_type{1}, valid_size / radix_run_min);
  auto begins_a = rmm::device_uvector<size_type>(maximum_continuing_runs, stream, temp_mr);
  auto ends_a   = rmm::device_uvector<size_type>(maximum_continuing_runs, stream, temp_mr);
  auto begins_b = rmm::device_uvector<size_type>(maximum_continuing_runs, stream, temp_mr);
  auto ends_b   = rmm::device_uvector<size_type>(maximum_continuing_runs, stream, temp_mr);

  auto const endpoint_capacity = valid_size / 2;
  auto run_endpoints = rmm::device_uvector<size_type>(2 * endpoint_capacity, stream, temp_mr);
  auto* run_begins   = run_endpoints.data();
  auto* run_ends     = endpoint_capacity > 0 ? run_begins + endpoint_capacity : run_begins;

  // Every recorded final segment contains at least two rows, so half the row count is a tight
  // upper bound.
  auto const maximum_finish_items = valid_size / 2;
  auto final_begins       = rmm::device_uvector<size_type>(maximum_finish_items, stream, temp_mr);
  auto final_ends         = rmm::device_uvector<size_type>(maximum_finish_items, stream, temp_mr);
  auto final_prefix_bytes = rmm::device_uvector<size_type>(maximum_finish_items, stream, temp_mr);
  auto counts = cudf::detail::device_scalar<finish_counts>(finish_counts{}, stream, temp_mr);
  auto refinement =
    cudf::detail::device_scalar<refinement_counts>(refinement_counts{}, stream, temp_mr);

  thrust::fill_n(exec, begins_a.begin(), 1, size_type{0});
  thrust::fill_n(exec, ends_a.begin(), 1, valid_size);

  auto* current_indices        = indices_a.data();
  auto* other_indices          = indices_b.data();
  auto* current_begins         = begins_a.data();
  auto* current_ends           = ends_a.data();
  auto* next_begins            = begins_b.data();
  auto* next_ends              = ends_b.data();
  auto const config            = cudf::detail::grid_1d{valid_size, 256};
  auto const first_pass_config = cudf::detail::grid_1d{
    valid_size, shuffle_block_size, strings_per_shuffle_cta / shuffle_block_size};
  auto cub_temp_storage = rmm::device_buffer{};
  size_type num_segments{1};
  // This counts algorithm-visible workspace allocations rather than upstream pool growth. It makes
  // stage-to-stage buffer lifetime changes observable without replacing the caller's memory
  // resource.
  auto allocation_count = size_type{14 + (null_indices.size() > 0 ? 1 : 0)};
  auto active_rows      = valid_size;

  for (size_type pass = 0; pass < maximum_radix_passes && num_segments > 0; ++pass) {
    auto const launch_keys = [&]<typename Offset>() {
      auto const* typed_offsets = offsets.head<Offset>() + input.offset();
      if (pass == 0 && null_size == 0) {
        make_first_keys<Offset>
          <<<first_pass_config.num_blocks,
             first_pass_config.num_threads_per_block,
             0,
             stream.get()>>>(typed_offsets, chars, valid_size, keys_in.data(), byte_counts);
      } else {
        make_subsequent_keys<Offset><<<first_pass_config.num_blocks,
                                       first_pass_config.num_threads_per_block,
                                       0,
                                       stream.get()>>>(offsets.head<Offset>(),
                                                       chars,
                                                       input.offset(),
                                                       current_indices,
                                                       current_begins,
                                                       current_ends,
                                                       pass == 0 ? size_type{0} : num_segments,
                                                       keys_in.data(),
                                                       byte_counts,
                                                       valid_size,
                                                       pass * radix_prefix_bytes);
      }
    };
    if (offsets.type().id() == type_id::INT64) {
      launch_keys.template operator()<int64_t>();
    } else {
      launch_keys.template operator()<size_type>();
    }
    CUDF_CUDA_TRY(cudaGetLastError());
    auto const cub_storage_bytes_before = cub_temp_storage.size();
    // Reusing both pairs lets CUB avoid row-sized fixed-output scratch. Only valid_size keys are
    // exposed, so the original-row byte-count sidecar at the tail of keys_in remains untouched.
    auto key_buffers   = cub::DoubleBuffer<std::uint64_t>{keys_in.data(), keys_out.data()};
    auto value_buffers = cub::DoubleBuffer<size_type>{current_indices, other_indices};
    if (pass == 0) {
      global_radix_sort(
        key_buffers, value_buffers, valid_size, ascending, cub_temp_storage, stream);
    } else {
      CUDF_CUDA_TRY(cudf::detail::memcpy_async(
        other_indices, current_indices, sizeof(size_type) * valid_size, stream));
      // The source lets CUB choose its result buffer, avoiding fixed-output copies and row-sized
      // scratch. Both index buffers contain inactive rows because CUB only touches active runs.
      segmented_key_sort(key_buffers,
                         value_buffers,
                         valid_size,
                         num_segments,
                         current_begins,
                         current_ends,
                         ascending,
                         cub_temp_storage,
                         stream);
    }
    auto* sorted_keys = key_buffers.Current();
    if (value_buffers.Current() != current_indices) { std::swap(current_indices, other_indices); }
    if (cub_temp_storage.size() > cub_storage_bytes_before) { ++allocation_count; }
    auto const radix_scratch_capacity = cub_temp_storage.size();

    refinement.set_value_async(refinement_counts{}, stream);
    auto const previous_run_count = pass == 0 ? size_type{0} : num_segments;
    mark_tied_run_endpoints<<<config.num_blocks, config.num_threads_per_block, 0, stream.get()>>>(
      sorted_keys,
      valid_size,
      current_begins,
      current_ends,
      previous_run_count,
      run_begins,
      run_ends,
      refinement.data());
    CUDF_CUDA_TRY(cudaGetLastError());

    auto observed = refinement.value(stream);
    ++synchronization_points;
    CUDF_EXPECTS(observed.candidate_starts == observed.candidate_ends,
                 "Segmented string sort produced mismatched run endpoints");
    auto const candidate_count = observed.candidate_starts;
    if (candidate_count > 0) {
      auto* ordered_begins = run_begins;
      auto* ordered_ends   = run_ends;
      if (candidate_count > 1) {
        // Disjoint runs have the same begin/end order. Batching their independent endpoint sorts
        // matches the source pipeline and avoids two separate sorting workspaces and launches.
        // Prefix keys are dead after endpoint detection, so a key buffer holds the output.
        ordered_begins    = reinterpret_cast<size_type*>(keys_out.data());
        ordered_ends      = ordered_begins + endpoint_capacity;
        auto const begins = cudf::detail::make_counting_transform_iterator(
          size_type{0}, endpoint_offset{endpoint_capacity, 0});
        auto const ends = cudf::detail::make_counting_transform_iterator(
          size_type{0}, endpoint_offset{endpoint_capacity, candidate_count});
        std::size_t endpoint_sort_bytes = 0;
        auto const sort_endpoints       = [&](void* storage) {
          return cub::DeviceSegmentedSort::SortKeys(storage,
                                                    endpoint_sort_bytes,
                                                    run_endpoints.data(),
                                                    ordered_begins,
                                                    2 * endpoint_capacity,
                                                    2,
                                                    begins,
                                                    ends,
                                                    stream.get());
        };
        CUDF_CUDA_TRY(sort_endpoints(nullptr));
        if (endpoint_sort_bytes > cub_temp_storage.size()) {
          cub_temp_storage = rmm::device_buffer(endpoint_sort_bytes, stream, temp_mr);
          ++allocation_count;
        }
        CUDF_CUDA_TRY(sort_endpoints(cub_temp_storage.data()));
      }
      // A run can span the entire column. Tiling distributes its length proof across blocks instead
      // of making one classification block walk millions of rows. Prefix keys are dead
      // until the next extraction, so their storage holds these two-word bounds without allocation.
      static_assert(sizeof(remaining_bounds) == sizeof(std::uint64_t));
      auto* bounds = reinterpret_cast<remaining_bounds*>(keys_in.data());
      thrust::fill_n(exec, bounds, candidate_count, remaining_bounds{});
      auto const bounds_config = cudf::detail::grid_1d{valid_size, 256, 4};
      reduce_run_remaining_lengths<<<bounds_config.num_blocks,
                                     bounds_config.num_threads_per_block,
                                     0,
                                     stream.get()>>>(remaining_length{byte_counts},
                                                     current_indices,
                                                     valid_size,
                                                     ordered_begins,
                                                     ordered_ends,
                                                     candidate_count,
                                                     bounds);
      auto const last_pass             = pass + 1 == maximum_radix_passes;
      auto const launch_classification = [&]<bool eliminate_duplicates>() {
        classify_tied_runs<eliminate_duplicates>
          <<<candidate_count, 256, 0, stream.get()>>>(bounds,
                                                      comparator.d_column,
                                                      current_indices,
                                                      ordered_begins,
                                                      ordered_ends,
                                                      candidate_count,
                                                      pass * radix_prefix_bytes,
                                                      radix_run_min,
                                                      last_pass,
                                                      tuning.trace,
                                                      next_begins,
                                                      next_ends,
                                                      final_begins.data(),
                                                      final_ends.data(),
                                                      final_prefix_bytes.data(),
                                                      refinement.data(),
                                                      counts.data());
      };
      if (tuning.eliminate_exact_duplicates) {
        launch_classification.template operator()<true>();
      } else {
        launch_classification.template operator()<false>();
      }
      CUDF_CUDA_TRY(cudaGetLastError());
    }

    // No next-pass count is needed after the final pass or when no tied run was found.
    // Trace alone may request classification counters that ordinary sorting never reads back.
    if (candidate_count > 0 && (pass + 1 < maximum_radix_passes || tuning.trace)) {
      observed = refinement.value(stream);
      ++synchronization_points;
      if (pass + 1 == maximum_radix_passes && tuning.trace) { ++trace_readbacks; }
    }
    num_segments = observed.continuing_runs;
    if (num_segments > 0) {
      thrust::sort_by_key(exec, next_begins, next_begins + num_segments, next_ends);
    }
    if (tuning.trace) {
      auto const singleton_rows =
        active_rows - observed.continuing_rows - observed.finish_rows - observed.completed_rows;
      std::fprintf(stderr,
                   "segmented-string-sort pass=%d active-rows=%d tied-runs=%d "
                   "continuing-runs=%d continuing-rows=%d completed-runs=%d completed-rows=%d "
                   "singleton-runs=%d "
                   "finish-runs=%d finish-rows=%d prefix-min=%d prefix-max=%d duplicate-rows=%d "
                   "allocations=%d sync-points=%d trace-readbacks=%d "
                   "radix-scratch-capacity=%zu scratch-capacity=%zu\n",
                   pass,
                   active_rows,
                   candidate_count,
                   observed.continuing_runs,
                   observed.continuing_rows,
                   observed.completed_runs + singleton_rows,
                   observed.completed_rows + singleton_rows,
                   singleton_rows,
                   observed.finish_runs,
                   observed.finish_rows,
                   candidate_count > 0 ? observed.minimum_prefix : 0,
                   observed.maximum_prefix,
                   observed.duplicate_rows,
                   allocation_count,
                   synchronization_points,
                   trace_readbacks,
                   radix_scratch_capacity,
                   cub_temp_storage.size());
    }

    std::swap(current_begins, next_begins);
    std::swap(current_ends, next_ends);
    active_rows = observed.continuing_rows;
  }

  // Transfer the finish launch metadata together after refinement.
  auto const finish = counts.value(stream);
  ++synchronization_points;
  auto total_chunks = size_type{0};
  if (finish.segments > 0) {
    if (finish.segments > 1) {
      // Atomic append order need not follow row order. Ordering disjoint runs restores the source's
      // descriptor layout and keeps neighboring chunk CTAs near neighboring string data.
      thrust::sort_by_key(exec,
                          final_begins.begin(),
                          final_begins.begin() + finish.segments,
                          final_prefix_bytes.begin());
      thrust::sort(exec, final_ends.begin(), final_ends.begin() + finish.segments);
    }
    // Refinement endpoints are dead here and have enough capacity for one count/offset per finish
    // run, so the scan needs no additional row-sized storage.
    auto* chunk_counts      = run_begins;
    auto* chunk_offsets     = run_ends;
    auto const chunk_config = cudf::detail::grid_1d{finish.segments, 256};
    compute_finish_chunk_counts<<<chunk_config.num_blocks,
                                  chunk_config.num_threads_per_block,
                                  0,
                                  stream.get()>>>(
      final_begins.data(), final_ends.data(), finish.segments, chunk_counts);
    std::size_t scan_bytes = 0;
    CUDF_CUDA_TRY(cub::DeviceScan::ExclusiveSum(
      nullptr, scan_bytes, chunk_counts, chunk_offsets, finish.segments, stream.get()));
    if (scan_bytes > cub_temp_storage.size()) {
      cub_temp_storage = rmm::device_buffer(scan_bytes, stream, temp_mr);
      ++allocation_count;
    }
    CUDF_CUDA_TRY(cub::DeviceScan::ExclusiveSum(cub_temp_storage.data(),
                                                scan_bytes,
                                                chunk_counts,
                                                chunk_offsets,
                                                finish.segments,
                                                stream.get()));
    auto device_total_chunks =
      cudf::detail::device_scalar<size_type>(size_type{0}, stream, temp_mr);
    compute_total_finish_chunks<<<1, 1, 0, stream.get()>>>(
      chunk_offsets, chunk_counts, finish.segments, device_total_chunks.data());
    total_chunks = device_total_chunks.value(stream);
    ++synchronization_points;

    auto chunk_begins       = rmm::device_uvector<size_type>(total_chunks, stream, temp_mr);
    auto chunk_sizes        = rmm::device_uvector<size_type>(total_chunks, stream, temp_mr);
    auto chunk_run_offsets  = rmm::device_uvector<size_type>(total_chunks, stream, temp_mr);
    auto chunk_run_counts   = rmm::device_uvector<size_type>(total_chunks, stream, temp_mr);
    auto chunk_prefix_bytes = rmm::device_uvector<size_type>(total_chunks, stream, temp_mr);
    allocation_count += 6;
    expand_finish_chunks<<<chunk_config.num_blocks,
                           chunk_config.num_threads_per_block,
                           0,
                           stream.get()>>>(final_begins.data(),
                                           final_ends.data(),
                                           final_prefix_bytes.data(),
                                           chunk_offsets,
                                           finish.segments,
                                           chunk_begins.data(),
                                           chunk_sizes.data(),
                                           chunk_run_offsets.data(),
                                           chunk_run_counts.data(),
                                           chunk_prefix_bytes.data());
    // Nulls are already partitioned out; host-known offset width avoids repeated device dispatch
    // when sibling ranking reloads string metadata.
    auto const launch_finish = [&]<bool sort_ascending, typename Offset>() {
      auto const* typed_offsets = offsets.head<Offset>() + input.offset();
      bitonic_sort_finish_chunks<sort_ascending>
        <<<total_chunks, comparison_chunk_size, 0, stream.get()>>>(current_indices,
                                                                   other_indices,
                                                                   chunk_begins.data(),
                                                                   chunk_sizes.data(),
                                                                   chunk_prefix_bytes.data(),
                                                                   typed_offsets,
                                                                   chars);
      merge_all_sibling_chunks<sort_ascending>
        <<<total_chunks, comparison_chunk_size, 0, stream.get()>>>(current_indices,
                                                                   other_indices,
                                                                   chunk_begins.data(),
                                                                   chunk_sizes.data(),
                                                                   chunk_run_offsets.data(),
                                                                   chunk_run_counts.data(),
                                                                   chunk_prefix_bytes.data(),
                                                                   typed_offsets,
                                                                   chars);
    };
    auto const launch_typed_finish = [&]<typename Offset>() {
      if (ascending) {
        launch_finish.template operator()<true, Offset>();
      } else {
        launch_finish.template operator()<false, Offset>();
      }
    };
    if (offsets.type().id() == type_id::INT64) {
      launch_typed_finish.template operator()<int64_t>();
    } else {
      launch_typed_finish.template operator()<size_type>();
    }
    CUDF_CUDA_TRY(cudaGetLastError());
  }

  if (tuning.trace) {
    std::fprintf(stderr,
                 "segmented-string-sort finish-segments=%d finish-rows=%d chunks=%d "
                 "allocations=%d sync-points=%d trace-readbacks=%d\n",
                 finish.segments,
                 finish.rows,
                 total_chunks,
                 allocation_count,
                 synchronization_points,
                 trace_readbacks);
  }

  {
    auto const nulls_first  = ascending == (null_precedence == null_order::BEFORE);
    auto const valid_offset = nulls_first ? null_size : size_type{0};
    auto const null_offset  = nulls_first ? size_type{0} : valid_size;
    CUDF_CUDA_TRY(cudf::detail::memcpy_async(output.begin<size_type>() + valid_offset,
                                             current_indices,
                                             sizeof(size_type) * valid_size,
                                             stream));
    if (null_size > 0) {
      CUDF_CUDA_TRY(cudf::detail::memcpy_async(output.begin<size_type>() + null_offset,
                                               null_indices.data(),
                                               sizeof(size_type) * null_size,
                                               stream));
    }
  }
}

}  // namespace segmented_string_sort
}  // namespace cudf::detail
