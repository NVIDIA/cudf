/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "text/bpe/byte_pair_encoding.cuh"

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/algorithms/copy_if.cuh>
#include <cudf/detail/algorithms/reduce.cuh>
#include <cudf/detail/copy.hpp>
#include <cudf/detail/get_value.cuh>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/offsets_iterator_factory.cuh>
#include <cudf/detail/sizes_to_offsets_iterator.cuh>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/detail/utilities/grid_1d.cuh>
#include <cudf/detail/utilities/integer_utils.hpp>
#include <cudf/strings/detail/strings_children.cuh>
#include <cudf/strings/detail/utilities.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <nvtext/byte_pair_encoding.hpp>

#include <rmm/exec_policy.hpp>

#include <cub/block/block_reduce.cuh>
#include <cub/device/device_for.cuh>
#include <cub/device/device_scan.cuh>
#include <cub/device/device_transform.cuh>
#include <cuda/atomic>
#include <cuda/functional>
#include <cuda/iterator>
#include <cuda/std/bit>
#include <cuda/std/limits>
#include <cuda/stream>
#include <thrust/execution_policy.h>
#include <thrust/find.h>
#include <thrust/for_each.h>

#include <algorithm>
#include <bit>
#include <limits>

namespace nvtext {

/**
 * @brief Access the bpe_merge_pairs impl member
 *
 * This is used by the encoder to access the impl member functions.
 *
 * @param bpe The merge pairs struct
 * @return The impl object with detailed, internal member data
 */
bpe_merge_pairs::bpe_merge_pairs_impl const* get_bpe_merge_pairs_impl(bpe_merge_pairs const& bpe)
{
  return bpe.impl;
}

namespace detail {
namespace {

constexpr int block_size         = 512;  // for row-per-block kernels
constexpr int long_block_size    = 64;   // for encoding long segments
constexpr int positions_per_lane = 8;    // window of 256 bytes processed by a warp
constexpr int max_short_size     = 32 * positions_per_lane;  // larger segments use a block
constexpr int short_block_size   = 256;                      // for encoding short segments
constexpr int max_dedup_size     = 4096;  // larger segments are always encoded (not de-duplicated)

/**
 * @brief Identifies unpairable boundaries in the given chars array
 *
 * Launched as a thread per byte of the chars array.
 * Returns 1 if a segment starts at this byte position.
 * Adjacent characters `a|b` can never be merged across if no merge pair
 * has a left half ending in `a` and a right half starting with `b`.
 * Fortunately, this can be used as an artificial boundary providing
 * increased parallelism in the BPE kernel.
 *
 * @tparam SetRefType The type of the cross-character set finder object
 */
template <typename SetRefType>
struct bpe_unpairable_fn {
  char const* d_chars;
  SetRefType const d_set;
  __device__ int8_t operator()(int64_t idx) const
  {
    if (!cudf::strings::detail::is_begin_utf8_char(d_chars[idx])) { return 0; }
    if (idx == 0) { return 1; }
    auto prev = idx - 1;  // locate the beginning of the previous character
    while (prev > 0 && !cudf::strings::detail::is_begin_utf8_char(d_chars[prev])) {
      --prev;
    }
    return !d_set.contains(make_cross_key(d_chars + prev, d_chars + idx));
  }
};

/**
 * @brief Writes the output chars and separators
 *
 * Called with the inclusive count of separators for each input byte position.
 */
struct bpe_write_fn {
  char const* d_input_chars;
  int8_t const* d_spaces;
  char* d_output_chars;
  char separator;
  __device__ void operator()(int64_t idx, int64_t count) const
  {
    auto const pos      = idx + count;
    d_output_chars[pos] = d_input_chars[idx];
    if (d_spaces[idx] > 0) { d_output_chars[pos - 1] = separator; }
  }
};

/**
 * @brief De-duplicates segments so each distinct segment is encoded once
 *
 * Each segment (up to `max_dedup_size` bytes) is hashed into a fixed-size table of slots. The first
 * segment to claim a slot becomes the representative for all segments with the same content. Only
 * representatives and segments whose slot was claimed by a segment with different content
 * (collisions) are encoded. The remaining segments copy their representative's result. Segments are
 * independent of their surroundings so identical content encodes identically.
 */
template <typename IndexType>
struct bpe_dedup_fn {
  char const* d_chars;
  IndexType const* d_segments;
  IndexType* d_slots;
  IndexType slot_mask;  // number of slots - 1 (number of slots is a power of 2)

  __device__ IndexType size(IndexType seg) const { return d_segments[seg + 1] - d_segments[seg]; }
  __device__ bool is_short(IndexType seg) const { return size(seg) <= max_short_size; }
  __device__ bool is_candidate(IndexType seg) const { return size(seg) <= max_dedup_size; }
  __device__ IndexType slot(IndexType seg) const
  {
    auto const hash = cudf::hashing::detail::XXHash_64<cudf::string_view>{}.compute_bytes(
      reinterpret_cast<cuda::std::byte const*>(d_chars + d_segments[seg]), size(seg));
    return static_cast<IndexType>(hash & static_cast<uint64_t>(slot_mask));
  }
  __device__ bool same(IndexType lhs, IndexType rhs) const
  {
    auto const size_bytes = static_cast<cudf::size_type>(size(lhs));
    return size_bytes == size(rhs) && cudf::string_view(d_chars + d_segments[lhs], size_bytes) ==
                                        cudf::string_view(d_chars + d_segments[rhs], size_bytes);
  }
  /// Claims the slot for a segment if it is not already taken
  __device__ void claim(IndexType seg) const
  {
    if (!is_candidate(seg)) { return; }
    cuda::atomic_ref<IndexType, cuda::thread_scope_device> ref{d_slots[slot(seg)]};
    // frequent segments map to the same slot; avoid contention once it is claimed
    if (ref.load(cuda::std::memory_order_relaxed) != -1) { return; }
    IndexType expected = -1;
    ref.compare_exchange_strong(expected, seg, cuda::std::memory_order_relaxed);
  }
  /// Returns the segment to copy the result from or -1 if this segment must be encoded
  __device__ IndexType source(IndexType seg) const
  {
    if (!is_candidate(seg)) { return -1; }
    auto const rep = d_slots[slot(seg)];
    return (rep != seg && same(rep, seg)) ? rep : -1;
  }
};

/**
 * @brief Warp-uniform bit-set of window positions stored in shared memory
 */
struct window_bits {
  static constexpr int num_words = positions_per_lane;
  uint32_t* words;

  __device__ bool test(int p) const { return (words[p >> 5] >> (p & 31)) & 1; }
  /// Highest set position less than p or -1 if none
  __device__ int prev(int p) const
  {
    auto j = p >> 5;
    auto m = words[j] & ((1u << (p & 31)) - 1);
    while (m == 0 && j > 0) {
      m = words[--j];
    }
    return m ? (j << 5) + 31 - cuda::std::countl_zero(m) : -1;
  }
  /// Lowest set position greater than p or `end` if none
  __device__ int next(int p, int end) const
  {
    auto j = p >> 5;
    auto m = ((p & 31) == 31) ? 0u : (words[j] & (~0u << ((p & 31) + 1)));
    while (m == 0 && j + 1 < num_words) {
      m = words[++j];
    }
    return m ? (j << 5) + cuda::std::countr_zero(m) : end;
  }
};

/**
 * @brief Performs byte-pair-encoding on short segments
 *
 * Each warp processes 32 segments from the `d_ids` list. Segments are packed into windows
 * of up to `max_short_size` bytes so each lane is responsible for `positions_per_lane`
 * byte positions (position p is handled by lane p % 32). The segments in a window need not
 * be adjacent in memory; positions in the window are mapped back to each segment's bytes
 * (tokens never span segments). Segments larger than `max_short_size` are skipped
 * (see bpe_parallel_fn).
 *
 * The token and segment boundaries in a window are kept in bit-sets in shared memory.
 * Each round, all segments in the window are processed together:
 * the minimum rank for each segment is found, all non-overlapping occurrences (left-to-right)
 * of each segment's minimum ranked pair are merged, and the pairs adjacent to a merge are
 * re-ranked in parallel. The rounds continue until no pairs can be merged in any of the
 * window's segments. Packing many segments in a window amortizes the per-round work.
 *
 * @tparam FinderType The type of the merge pair rank finder
 * @tparam IndexType Type of the segment offsets and indices (int32_t or int64_t)
 * @param d_input_chars Pointer to the input character data
 * @param d_segments Offsets to each segment; the last entry is the end of the chars data
 * @param d_ids Indices of the segments to encode
 * @param num_ids Number of entries in `d_ids`
 * @param d_finder For looking up the rank of a pair of tokens
 * @param d_spaces_data Output the location where separator will be inserted
 */
template <typename FinderType, typename IndexType>
CUDF_KERNEL void bpe_short_fn(char const* d_input_chars,
                              IndexType const* d_segments,
                              IndexType const* d_ids,
                              IndexType num_ids,
                              FinderType const d_finder,
                              int8_t* d_spaces_data)
{
  auto constexpr warp_size   = cudf::detail::warp_size;
  auto constexpr window_size = max_short_size;
  auto constexpr num_words   = window_bits::num_words;
  auto constexpr full_mask   = 0xFFFF'FFFFu;
  auto constexpr max_rank    = cuda::std::numeric_limits<cudf::size_type>::max();
  auto constexpr num_warps   = short_block_size / warp_size;

  __shared__ cudf::size_type s_ranks[num_warps][window_size];
  __shared__ uint8_t s_seg_idx[num_warps][window_size];  // segment of each position
  __shared__ uint32_t s_bits[num_warps][4][num_words];   // token/segment starts, merged, re-rank
  __shared__ cudf::size_type s_seg_min[num_warps][warp_size];
  __shared__ int64_t s_seg_offset[num_warps][warp_size];  // segment's offset in d_input_chars
  __shared__ int s_seg_begin[num_warps][warp_size];       // segment's position in the window

  auto const warp_idx    = cudf::detail::grid_1d::global_thread_id() / warp_size;
  auto const lane        = static_cast<int>(threadIdx.x % warp_size);
  auto const widx        = threadIdx.x / warp_size;
  auto const ranks       = s_ranks[widx];
  auto const seg_idx     = s_seg_idx[widx];
  auto const seg_min     = s_seg_min[widx];
  auto const seg_offsets = s_seg_offset[widx];
  auto const seg_begins  = s_seg_begin[widx];
  auto const S           = window_bits{s_bits[widx][0]};  // token starts
  auto const G           = window_bits{s_bits[widx][1]};  // segment starts
  auto const M           = window_bits{s_bits[widx][2]};  // merged this round
  auto const R           = window_bits{s_bits[widx][3]};  // re-rank this round

  auto const first = warp_idx * warp_size;
  if (first >= num_ids) { return; }
  auto const last = cuda::std::min(first + warp_size, static_cast<int64_t>(num_ids));

  for (auto cur = first; cur < last;) {
    // pack as many (short) segments as will fit into the window
    auto size       = window_size + 1;  // stops the window
    auto seg_offset = int64_t{0};
    if (cur + lane < last) {
      auto const seg = d_ids[cur + lane];
      seg_offset     = d_segments[seg];
      auto const sz  = d_segments[seg + 1] - seg_offset;
      if (sz <= max_short_size) { size = static_cast<int>(sz); }
    }
    auto seg_end = size;  // inclusive scan of sizes
    for (int offset = 1; offset < warp_size; offset *= 2) {
      auto const value = __shfl_up_sync(full_mask, seg_end, offset);
      if (lane >= offset) { seg_end += value; }
    }
    auto const n = __popc(__ballot_sync(full_mask, seg_end <= window_size));
    if (n == 0) {  // long segment is encoded by bpe_parallel_fn
      ++cur;
      continue;
    }
    auto const seg_begin = seg_end - size;
    auto const width     = __shfl_sync(full_mask, seg_end, n - 1);
    auto const words     = (width + warp_size - 1) / warp_size;  // words used by this window

    __syncwarp();  // the previous window is done with the shared memory
    if (lane < n) {
      seg_offsets[lane] = seg_offset;
      seg_begins[lane]  = seg_begin;
    }
    if (lane < num_words) {
      S.words[lane] = 0;
      G.words[lane] = 0;
      M.words[lane] = 0;
      R.words[lane] = 0;
    }
    __syncwarp();
    if (lane < n) { atomicOr(&G.words[seg_begin >> 5], 1u << (seg_begin & 31)); }
    __syncwarp();

    // offset into d_input_chars for window position p
    auto chars_offset = [&](int p) {
      auto const k = seg_idx[p];
      return seg_offsets[k] + (p - seg_begins[k]);
    };
    // rank of the pair formed by the token starting at p and the token before it
    auto pair_rank = [&](int p) {
      if (!S.test(p) || G.test(p)) { return max_rank; }
      auto const q = S.prev(p);
      auto const e = S.next(p, width);
      return d_finder.find(d_input_chars + chars_offset(q), p - q, e - q);
    };

    // segment of each position and the initial token starts
    for (int j = 0; j < words; ++j) {
      auto const p   = lane + j * warp_size;
      auto const pos = cuda::std::min(p, width - 1);
      auto const k   = __popc(G.words[j] & ((2u << lane) - 1u)) - 1;  // within this word
      auto prefix    = 0;
      for (int i = 0; i < j; ++i) {
        prefix += __popc(G.words[i]);
      }
      seg_idx[p] = static_cast<uint8_t>(prefix + k);
      __syncwarp();
      auto const is_start =
        p < width && cudf::strings::detail::is_begin_utf8_char(d_input_chars[chars_offset(pos)]);
      auto const ballot = __ballot_sync(full_mask, is_start) | G.words[j];
      if (lane == 0) { S.words[j] = ballot; }
    }
    __syncwarp();
    for (int j = 0; j < words; ++j) {
      auto const p = lane + j * warp_size;
      ranks[p]     = p < width ? pair_rank(p) : max_rank;
    }

    while (true) {
      // find the minimum rank for each segment
      if (lane < n) { seg_min[lane] = max_rank; }
      __syncwarp();
      for (int j = 0; j < words; ++j) {
        auto const p = lane + j * warp_size;
        if (ranks[p] < max_rank) { atomicMin(&seg_min[seg_idx[p]], ranks[p]); }
      }
      __syncwarp();
      auto found = false;
      for (int j = 0; j < words; ++j) {
        auto const p = lane + j * warp_size;
        auto const c = ranks[p] < max_rank && ranks[p] == seg_min[seg_idx[p]];
        found        = found || c;
        auto const C = __ballot_sync(full_mask, c);
        // merge left-to-right skipping a pair whose left token was just merged
        if (lane == 0) {
          for (auto m = C; m; m &= m - 1) {
            auto const q = (j << 5) + cuda::std::countr_zero(m);
            if (!M.test(S.prev(q))) { M.words[j] |= 1u << (q & 31); }
          }
        }
        __syncwarp();
      }
      if (!__any_sync(full_mask, found)) { break; }
      if (lane < words) { S.words[lane] &= ~M.words[lane]; }
      __syncwarp();

      // re-rank the pairs adjacent to each merged position
      for (int j = 0; j < words; ++j) {
        auto const p = lane + j * warp_size;
        if (p < width && M.test(p)) {
          ranks[p]     = max_rank;  // no longer a token start
          auto const q = S.prev(p);
          if (!G.test(q)) { atomicOr(&R.words[q >> 5], 1u << (q & 31)); }
          auto const e = S.next(p, width);
          if (e < width && !G.test(e)) { atomicOr(&R.words[e >> 5], 1u << (e & 31)); }
        }
      }
      __syncwarp();
      for (int j = 0; j < words; ++j) {
        auto const p = lane + j * warp_size;
        if (p < width && R.test(p)) { ranks[p] = pair_rank(p); }
      }
      __syncwarp();
      if (lane < num_words) {
        M.words[lane] = 0;
        R.words[lane] = 0;
      }
      __syncwarp();
    }

    for (int j = 0; j < words; ++j) {
      auto const p = lane + j * warp_size;
      if (p < width) { d_spaces_data[chars_offset(p)] = static_cast<int8_t>(S.test(p)); }
    }
    cur += n;
  }
}

/**
 * @brief Performs byte-pair-encoding on long segments
 *
 * Computes the locations where the separator will be inserted in `d_spaces_data`.
 * This is launched as a segment per block for segments larger than `max_short_size`.
 *
 * The process first initializes all characters to 1 per position in `d_spaces_data`.
 * All pairs are realized and their ranks stored in `d_ranks_data`.
 *
 * Iteratively, the minimum rank is located, the corresponding `d_spaces_data` location
 * is set to 0 resulting in new potential pairs. The process repeats accounting for
 * the rank of the newly formed pairs.
 *
 * Once there are no more rankable pairs, the process finishes and the `d_spaces_data`
 * values identify the location to insert the separator.
 *
 * @tparam FinderType The type of the merge pair rank finder
 * @tparam IndexType Type of the segment offsets and indices (int32_t or int64_t)
 * @param d_input_chars Pointer to the input character data
 * @param d_segments Offsets to each segment; the last entry is the end of the chars data
 * @param d_long_ids Indices of the segments to encode (those larger than max_short_size)
 * @param d_scratch_offsets Offsets into the working memory for each segment in d_long_ids
 * @param num_long Number of segments to encode
 * @param d_finder For looking up the rank of a pair of tokens
 * @param d_spaces_data Output the location where separator will be inserted
 * @param d_ranks_data Working memory to hold pair ranks
 * @param d_rerank_data Working memory to hold locations where reranking is required
 */
template <typename FinderType, typename IndexType>
CUDF_KERNEL void bpe_parallel_fn(char const* d_input_chars,
                                 IndexType const* d_segments,
                                 IndexType const* d_long_ids,
                                 IndexType const* d_scratch_offsets,
                                 IndexType num_long,
                                 FinderType const d_finder,
                                 int8_t* d_spaces_data,          // working memory
                                 cudf::size_type* d_ranks_data,  // more working memory
                                 int8_t* d_rerank_data           // and one more working memory
)
{
  auto const lane_idx = static_cast<cudf::size_type>(threadIdx.x);

  auto constexpr max_rank = cuda::std::numeric_limits<cudf::size_type>::max();

  __shared__ cudf::size_type block_min_rank;
  using block_reduce = cub::BlockReduce<cudf::size_type, long_block_size>;
  __shared__ typename block_reduce::TempStorage temp_storage;

  // segment per block
  for (auto idx = static_cast<int64_t>(blockIdx.x); idx < num_long; idx += gridDim.x) {
    auto const seg_idx = d_long_ids[idx];
    auto const offset  = static_cast<int64_t>(d_segments[seg_idx]);
    auto const d_str   = cudf::string_view(
      d_input_chars + offset, static_cast<cudf::size_type>(d_segments[seg_idx + 1] - offset));

    auto const scratch    = static_cast<int64_t>(d_scratch_offsets[idx]);
    auto const d_spaces   = d_spaces_data + offset;
    auto const end_spaces = d_spaces + d_str.size_bytes();
    auto const d_ranks    = d_ranks_data + scratch;
    auto const end_ranks  = d_ranks + d_str.size_bytes();
    auto const d_rerank   = d_rerank_data + scratch;
    auto const end_rerank = d_rerank + d_str.size_bytes();
    auto const num_valid =
      long_block_size < d_str.size_bytes() ? long_block_size : d_str.size_bytes();

    // init all the re-rank identifiers to zero
    for (auto itr = d_rerank + lane_idx; itr < end_rerank; itr += long_block_size) {
      *itr = 0;
    }
    // init all ranks to max
    for (auto itr = d_ranks + lane_idx; itr < end_ranks; itr += long_block_size) {
      *itr = max_rank;
    }
    // init all spaces to 1 as appropriate
    for (auto itr = d_spaces + lane_idx; itr < end_spaces; itr += long_block_size) {
      auto const index = cuda::std::distance(d_spaces, itr);
      *itr = static_cast<int8_t>(cudf::strings::detail::is_begin_utf8_char(d_str.data()[index]));
    }
    __syncthreads();

    // for finding the next half of a pair
    auto next_substr = [d_str, d_spaces, end = end_spaces](int8_t* begin) {
      auto const next = thrust::find(thrust::seq, begin + 1, end, 1);
      auto const size = static_cast<cudf::size_type>(cuda::std::distance(begin, next));
      return cudf::string_view(d_str.data() + cuda::std::distance(d_spaces, begin), size);
    };
    // for locating adjacent pairs after merging a pair
    auto find_prev = [begin = d_spaces](int8_t* ptr) {
      while (ptr > begin && *ptr == 0) {
        --ptr;
      }
      return ptr;
    };

    auto min_rank = max_rank;

    // store all the initial ranks for each pair
    // every character but the first one will have a initial rank
    //
    // Example:
    // string:   abcdefghij
    // spaces:   1111111111
    // ranks:    *948516327
    for (auto itr = d_spaces + lane_idx; itr < end_spaces; itr += long_block_size) {
      if (*itr == 0) { continue; }  // skips any UTF-8 continuation bytes
      // resolve pair and lookup its rank
      auto const lhs      = next_substr(itr);  // retrieve lhs of the pair
      auto const next_itr = itr + lhs.size_bytes();
      if (next_itr < end_spaces) {
        auto const rhs = next_substr(next_itr);  // retrieve rhs of the pair
        if (!rhs.empty()) {
          // lookup pair in merges table (lhs and rhs are contiguous)
          auto const rank =
            d_finder.find(lhs.data(), lhs.size_bytes(), lhs.size_bytes() + rhs.size_bytes());
          d_ranks[cuda::std::distance(d_spaces, next_itr)] = rank;  // store the rank
          if (rank < min_rank) { min_rank = rank; }
        }
      }
    }
    // compute the min rank across the block
    auto const reduce_rank =
      block_reduce(temp_storage).Reduce(min_rank, cuda::minimum{}, num_valid);
    if (lane_idx == 0) { block_min_rank = reduce_rank; }
    __syncthreads();

    // loop through the ranks processing the current minimum until there are no more
    while (block_min_rank < max_rank) {
      // search the d_ranks for matches to block_min_rank
      for (auto itr = d_ranks + lane_idx; itr < end_ranks; itr += long_block_size) {
        if (*itr == block_min_rank) {
          auto ptr = itr - 1;  // check for adjacent min-rank (edge-case)
          while (ptr > d_ranks && *ptr == max_rank) {
            --ptr;
          }
          // set the output value to 0 at this position (erases separator, merges pair)
          // using example string above, the min-rank is 1 at position 5
          // string: abcdefghij
          // spaces: 1111101111  (set position 5 to 0)
          if (*ptr != block_min_rank) { d_spaces[cuda::std::distance(d_ranks, itr)] = 0; }
        }
      }
      __syncthreads();

      // identify all the re-rank locations (logic above invalidated adjacent pairs)
      // using example string above, the adjacent pairs have to be re-ranked
      // string: abcdefghij
      // spaces: 1111101111 (pair 'e,f' is now merged)
      // rerank: 0000101000 ('ef' and 'fg' need re-ranking as 'd,ef' and 'ef,g'
      for (auto itr = d_ranks + lane_idx; itr < end_ranks; itr += long_block_size) {
        auto const index = cuda::std::distance(d_ranks, itr);
        if (*itr == block_min_rank && d_spaces[index] == 0) {
          // find previous pair mid-point
          auto ptr = find_prev(d_spaces + index - 1);
          if (ptr > d_spaces) { d_rerank[cuda::std::distance(d_spaces, ptr)] = 1; }
          // find next pair mid-point
          ptr = thrust::find(thrust::seq, d_spaces + index + 1, end_spaces, 1);
          if (ptr < end_spaces) { d_rerank[cuda::std::distance(d_spaces, ptr)] = 1; }
          *itr = max_rank;  // reset this rank
        }
      }
      __syncthreads();

      // compute the ranks for the newly created pairs
      min_rank = max_rank;  // and record the new minimum along the way
      for (auto itr = d_rerank + lane_idx; itr < end_rerank; itr += long_block_size) {
        auto const index = cuda::std::distance(d_rerank, itr);
        auto rank        = d_ranks[index];
        if (*itr) {
          *itr = 0;  // reset re-rank
          // build lhs of pair
          auto const ptr = find_prev(d_spaces + index - 1);
          auto const size =
            static_cast<cudf::size_type>(cuda::std::distance(ptr, d_spaces + index));
          auto const lhs =
            cudf::string_view(d_str.data() + cuda::std::distance(d_spaces, ptr), size);
          auto const rhs = next_substr(d_spaces + index);  // retrieve rhs of pair
          rank           = max_rank;
          if (!rhs.empty()) {
            // lookup rank for this pair (lhs and rhs are contiguous)
            rank = d_finder.find(lhs.data(), lhs.size_bytes(), lhs.size_bytes() + rhs.size_bytes());
          }
          d_ranks[index] = rank;  // store new rank
        }
        if (rank < min_rank) { min_rank = rank; }
      }

      // re-compute the minimum rank across the block (since new pairs are created above)
      auto const reduce_rank =
        block_reduce(temp_storage).Reduce(min_rank, cuda::minimum{}, num_valid);
      if (lane_idx == 0) { block_min_rank = reduce_rank; }
      __syncthreads();
    }  // if no min ranks are found we are done, otherwise start again
    __syncthreads();  // shared memory is reused by the next segment
  }
}

/**
 * @brief Computes the output size of each strings row
 *
 * This launches as a string per block.
 * The non-zero values in `d_spaces_data` for each string is added to
 * the current string size to produce the total output bytes.
 *
 * @param d_strings Input data
 * @param d_input_chars Pointer to the input character data
 * @param d_spaces_data Output the location where separator will be inserted
 * @param d_sizes Output sizes of each row
 */
CUDF_KERNEL void bpe_finalize(cudf::column_device_view const d_strings,
                              char const* d_input_chars,
                              int8_t* d_spaces_data,    // where separators are inserted
                              cudf::size_type* d_sizes  // output sizes of encoded strings
)
{
  // string per block
  auto const str_idx =
    static_cast<cudf::size_type>(cudf::detail::grid_1d::global_thread_id() / block_size);
  auto const lane_idx = static_cast<cudf::size_type>(threadIdx.x);

  if (d_strings.is_null(str_idx)) {
    d_sizes[str_idx] = 0;
    return;
  }
  auto const d_str = d_strings.element<cudf::string_view>(str_idx);
  if (d_str.empty()) {
    d_sizes[str_idx] = 0;
    return;
  }

  auto const offset = cuda::std::distance(d_input_chars, d_str.data());

  auto const d_spaces   = d_spaces_data + offset;
  auto const end_spaces = d_spaces + d_str.size_bytes();
  auto const num_valid  = block_size < d_str.size_bytes() ? block_size : d_str.size_bytes();

  using block_reduce = cub::BlockReduce<cudf::size_type, block_size>;
  __shared__ typename block_reduce::TempStorage temp_storage;

  // reset the first position -- no separator to be added here
  if (lane_idx == 0) { *d_spaces = 0; }

  // compute the output size for this string by counting the resulting separator positions
  auto bytes = 0;
  for (auto itr = d_spaces + lane_idx; itr < end_spaces; itr += block_size) {
    bytes += (*itr > 0);
  }
  auto const total_bytes = block_reduce(temp_storage).Sum(bytes, num_valid);
  if (lane_idx == 0) { d_sizes[str_idx] = total_bytes + d_str.size_bytes(); }
}

/**
 * @brief Encodes the segments identified by the flags in `d_spaces`
 *
 * On input, `d_spaces` contains a 1 for each byte that starts a segment.
 * On output, it contains a 1 for each byte that starts a token.
 *
 * @tparam IndexType Type for segment offsets and indices: int32_t when the chars
 *                   size allows, to reduce the memory footprint
 */
template <typename IndexType>
void encode_segments(char const* d_input_chars,
                     int64_t chars_size,
                     bpe_merge_pairs const& merge_pairs,
                     int8_t* d_spaces,
                     cuda::stream_ref stream)
{
  auto const d_flags = cuda::transform_iterator(
    d_spaces,
    cuda::proclaim_return_type<IndexType>([] __device__(int8_t v) -> IndexType { return v; }));
  auto const num_segments = cudf::detail::reduce(
    d_flags, d_flags + chars_size, IndexType{0}, cuda::std::plus<IndexType>{}, stream);
  auto d_segments = rmm::device_uvector<IndexType>(num_segments + 1, stream);
  cudf::detail::copy_if(cuda::counting_iterator<IndexType>{0},
                        cuda::counting_iterator<IndexType>{static_cast<IndexType>(chars_size)},
                        d_spaces,
                        d_segments.begin(),
                        cuda::std::identity{},
                        stream);
  auto const end_offset = static_cast<IndexType>(chars_size);
  d_segments.set_element_async(num_segments, end_offset, stream);

  // de-duplicate the segments so only distinct segments are encoded;
  // the slot table size is bounded so collisions are encoded as well
  auto const num_slots =
    std::clamp(static_cast<IndexType>(std::bit_ceil(static_cast<uint64_t>(num_segments / 4 + 1))),
               IndexType{1024},
               IndexType{1} << 22);
  auto d_slots = rmm::device_uvector<IndexType>(num_slots, stream);
  CUDF_CUDA_TRY(cudaMemsetAsync(d_slots.data(), 0xFF, num_slots * sizeof(IndexType), stream.get()));
  auto const dedup =
    bpe_dedup_fn<IndexType>{d_input_chars, d_segments.data(), d_slots.data(), num_slots - 1};
  CUDF_CUDA_TRY(cub::DeviceFor::Bulk(
    num_segments, [dedup] __device__(IndexType seg) { dedup.claim(seg); }, stream.get()));

  auto const d_encode = cuda::transform_iterator(
    cuda::counting_iterator<IndexType>{0},
    cuda::proclaim_return_type<IndexType>([dedup] __device__(IndexType seg) -> IndexType {
      return dedup.is_short(seg) && dedup.source(seg) < 0;
    }));
  auto const num_encode = cudf::detail::reduce(
    d_encode, d_encode + num_segments, IndexType{0}, cuda::std::plus<IndexType>{}, stream);
  auto d_encode_ids = rmm::device_uvector<IndexType>(num_encode, stream);
  cudf::detail::copy_if(cuda::counting_iterator<IndexType>{0},
                        cuda::counting_iterator<IndexType>{num_segments},
                        d_encode,
                        d_encode_ids.begin(),
                        cuda::std::identity{},
                        stream);

  // encode the distinct short segments with a warp per 32 segments
  auto const finder = get_bpe_merge_pairs_impl(merge_pairs)->get_rank_finder();
  if (num_encode > 0) {
    auto const num_warps =
      cudf::util::div_rounding_up_safe(num_encode, static_cast<IndexType>(cudf::detail::warp_size));
    auto const grid = cudf::detail::grid_1d(
      static_cast<int64_t>(num_warps) * cudf::detail::warp_size, short_block_size);
    bpe_short_fn<decltype(finder), IndexType>
      <<<grid.num_blocks, grid.num_threads_per_block, 0, stream.get()>>>(
        d_input_chars, d_segments.data(), d_encode_ids.data(), num_encode, finder, d_spaces);
    CUDF_CUDA_TRY(cudaGetLastError());
  }

  // encode the distinct long segments with a block per segment
  auto const d_seg_sizes = cuda::transform_iterator(
    cuda::counting_iterator<IndexType>{0},
    cuda::proclaim_return_type<IndexType>([dedup] __device__(IndexType idx) -> IndexType {
      auto const size = dedup.size(idx);
      return (size > max_short_size && dedup.source(idx) < 0) ? size : IndexType{0};
    }));
  auto const d_is_long = cuda::transform_iterator(
    d_seg_sizes, cuda::proclaim_return_type<IndexType>([] __device__(IndexType size) -> IndexType {
      return size > 0;
    }));
  auto const num_long = cudf::detail::reduce(
    d_is_long, d_is_long + num_segments, IndexType{0}, cuda::std::plus<IndexType>{}, stream);
  if (num_long > 0) {
    auto d_long_ids = rmm::device_uvector<IndexType>(num_long, stream);
    cudf::detail::copy_if(cuda::counting_iterator<IndexType>{0},
                          cuda::counting_iterator<IndexType>{num_segments},
                          d_seg_sizes,
                          d_long_ids.begin(),
                          cuda::std::identity{},
                          stream);
    // working memory is only needed for the long segments
    auto d_scratch_offsets  = rmm::device_uvector<IndexType>(num_long + 1, stream);
    auto const d_long_sizes = cuda::transform_iterator(
      cuda::counting_iterator<IndexType>{0},
      cuda::proclaim_return_type<IndexType>(
        [d_seg_sizes, d_long_ids = d_long_ids.data(), num_long] __device__(IndexType idx) {
          return idx < num_long ? d_seg_sizes[d_long_ids[idx]] : IndexType{0};
        }));
    auto const scratch_size =
      cudf::detail::sizes_to_offsets(d_long_sizes,
                                     d_long_sizes + num_long + 1,
                                     d_scratch_offsets.begin(),
                                     0,
                                     stream,
                                     cudf::get_current_device_resource_ref());
    rmm::device_uvector<int8_t> d_rerank(scratch_size, stream);
    rmm::device_uvector<cudf::size_type> d_ranks(scratch_size, stream);
    auto const long_grid = static_cast<int>(std::min<int64_t>(num_long, int64_t{1} << 20));
    bpe_parallel_fn<decltype(finder), IndexType>
      <<<long_grid, long_block_size, 0, stream.get()>>>(d_input_chars,
                                                        d_segments.data(),
                                                        d_long_ids.data(),
                                                        d_scratch_offsets.data(),
                                                        num_long,
                                                        finder,
                                                        d_spaces,
                                                        d_ranks.data(),
                                                        d_rerank.data());
    CUDF_CUDA_TRY(cudaGetLastError());
  }

  // copy the results to the duplicate segments
  CUDF_CUDA_TRY(cub::DeviceFor::Bulk(
    num_segments,
    [dedup, d_segments = d_segments.data(), d_spaces = d_spaces] __device__(IndexType seg) {
      auto const rep = dedup.source(seg);
      if (rep < 0) { return; }
      auto const src = d_spaces + d_segments[rep];
      auto const dst = d_spaces + d_segments[seg];
      for (IndexType i = 0; i < dedup.size(seg); ++i) {
        dst[i] = src[i];
      }
    },
    stream.get()));
}

}  // namespace

std::unique_ptr<cudf::column> byte_pair_encoding(cudf::strings_column_view const& input,
                                                 bpe_merge_pairs const& merge_pairs,
                                                 cudf::string_scalar const& separator,
                                                 cuda::stream_ref stream,
                                                 rmm::device_async_resource_ref mr)
{
  if (input.is_empty()) { return cudf::make_empty_column(cudf::type_id::STRING); }
  if (input.chars_size(stream) == 0) {  // all rows are empty or null
    auto result = cudf::make_column_from_scalar(
      cudf::string_scalar("", true, stream), input.size(), stream, mr);
    result->set_null_mask(cudf::detail::copy_bitmask(input.parent(), stream, mr),
                          input.null_count());
    return result;
  }

  CUDF_EXPECTS(separator.is_valid(stream), "separator parameter must be valid");
  auto const d_separator = separator.value(stream);
  CUDF_EXPECTS(d_separator.size_bytes() == 1, "for now, separator must be a single-byte character");

  // the chars data for null rows would otherwise be encoded and written to the output
  if (input.has_nulls() && cudf::detail::has_nonempty_nulls(input.parent(), stream)) {
    auto const sanitized = cudf::detail::purge_nonempty_nulls(
      input.parent(), stream, cudf::get_current_device_resource_ref());
    return detail::byte_pair_encoding(
      cudf::strings_column_view(sanitized->view()), merge_pairs, separator, stream, mr);
  }

  auto const d_strings = cudf::column_device_view::create(input.parent(), stream);

  auto const first_offset  = (input.offset() == 0) ? 0L
                                                   : cudf::strings::detail::get_offset_value(
                                                      input.offsets(), input.offset(), stream);
  auto const last_offset   = (input.offset() == 0 && input.size() == input.offsets().size() - 1)
                               ? static_cast<int64_t>(input.chars_size(stream))
                               : cudf::strings::detail::get_offset_value(
                                 input.offsets(), input.size() + input.offset(), stream);
  auto const chars_size    = last_offset - first_offset;
  auto const d_input_chars = input.chars_begin(stream) + first_offset;

  auto const chars_begin = cuda::counting_iterator<int64_t>{0};

  // identifies non-merged pairs; also used to identify segment boundaries before encoding
  rmm::device_uvector<int8_t> d_spaces(chars_size, stream);

  {
    // locate unpairable boundaries to create artificial segments that can be encoded
    // independently; each row start is also a segment start
    auto const cross_set = get_bpe_merge_pairs_impl(merge_pairs)->get_cross_set_ref();
    // (thrust is limited to 32-bit sizes in libcudf; cub is used for all per-byte operations)
    CUDF_CUDA_TRY(cub::DeviceTransform::Transform(
      chars_begin,
      d_spaces.begin(),
      chars_size,
      bpe_unpairable_fn<decltype(cross_set)>{d_input_chars, cross_set},
      stream.get()));
    auto const input_offsets =
      cudf::detail::offsetalator_factory::make_input_iterator(input.offsets(), input.offset());
    thrust::for_each_n(
      rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
      cuda::counting_iterator<cudf::size_type>{0},
      input.size(),
      [input_offsets, first_offset, d_spaces = d_spaces.data()] __device__(cudf::size_type idx) {
        auto const begin = input_offsets[idx];
        if (begin < input_offsets[idx + 1]) { d_spaces[begin - first_offset] = 1; }
      });

    if (chars_size <= std::numeric_limits<int32_t>::max()) {
      encode_segments<int32_t>(d_input_chars, chars_size, merge_pairs, d_spaces.data(), stream);
    } else {
      encode_segments<int64_t>(d_input_chars, chars_size, merge_pairs, d_spaces.data(), stream);
    }
  }

  // compute the output sizes
  auto output_sizes = rmm::device_uvector<cudf::size_type>(input.size(), stream);
  bpe_finalize<<<input.size(), block_size, 0, stream.get()>>>(
    *d_strings, d_input_chars, d_spaces.data(), output_sizes.data());
  CUDF_CUDA_TRY(cudaGetLastError());

  // convert sizes to offsets in-place
  auto [offsets, bytes] = cudf::strings::detail::make_offsets_child_column(
    output_sizes.begin(), output_sizes.end(), stream, mr);

  // build the output: the inclusive count of separators gives each byte's output position
  rmm::device_uvector<char> chars(bytes, stream, mr);
  auto const sep_char = separator.to_string(stream)[0];
  auto const d_counts = cuda::transform_iterator(
    chars_begin,
    cuda::proclaim_return_type<int64_t>(
      [d_spaces = d_spaces.data()] __device__(int64_t idx) -> int64_t {
        return d_spaces[idx] > 0;  // separator to be inserted before this position
      }));
  auto const d_output = cuda::make_tabulate_output_iterator(
    bpe_write_fn{d_input_chars, d_spaces.data(), chars.data(), sep_char});
  std::size_t temp_bytes = 0;
  CUDF_CUDA_TRY(cub::DeviceScan::InclusiveSum(
    nullptr, temp_bytes, d_counts, d_output, chars_size, stream.get()));
  auto temp_storage = rmm::device_buffer(temp_bytes, stream);
  CUDF_CUDA_TRY(cub::DeviceScan::InclusiveSum(
    temp_storage.data(), temp_bytes, d_counts, d_output, chars_size, stream.get()));

  return cudf::make_strings_column(input.size(),
                                   std::move(offsets),
                                   chars.release(),
                                   input.null_count(),
                                   cudf::detail::copy_bitmask(input.parent(), stream, mr));
}

}  // namespace detail

std::unique_ptr<cudf::column> byte_pair_encoding(cudf::strings_column_view const& input,
                                                 bpe_merge_pairs const& merges_table,
                                                 cudf::string_scalar const& separator,
                                                 cuda::stream_ref stream,
                                                 rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::byte_pair_encoding(input, merges_table, separator, stream, mr);
}

}  // namespace nvtext
