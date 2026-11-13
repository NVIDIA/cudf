/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "delta_binary.cuh"
#include "io/utilities/block_utils.cuh"
#include "page_state_composed.cuh"
#include "page_string_utils.cuh"
#include "parquet_gpu.hpp"

#include <cudf/detail/utilities/cuda.cuh>

#include <cooperative_groups.h>
#include <cub/warp/warp_exchange.cuh>
#include <cuda/annotated_ptr>
#include <cuda/pipeline>
#include <cuda/std/algorithm.lower_bound.h>
#include <cuda/std/algorithm.max.h>
#include <cuda/std/algorithm.min.h>
#include <cuda/std/bit>

#include <algorithm>

namespace cudf::io::parquet::detail {

namespace {

namespace cg = cooperative_groups;

constexpr int decode_block_size              = 128;
constexpr int decode_delta_binary_block_size = 96;

// Size of the ring buffer that maps leaf-value ordinals to output rows (nz_idx). The level
// decoder runs up to two batches ahead of the value consumer and, on nested pages, overshoots
// its target by up to a warp of values, so this needs to exceed 3 * delta_max_batch_size +
// warp_size; anything smaller lets the level decoder wrap onto entries the consumer is reading.
constexpr int delta_nz_buf_size = 4 * delta_max_batch_size;

// One warp decodes and writes each mapped page.
constexpr int32_t decode_delta_binary_with_nz_map_block_size = cudf::detail::warp_size;

// DELTA_BYTE_ARRAY encoding (incremental encoding or front compression), is used for BYTE_ARRAY
// columns. For each element in a sequence of strings, a prefix length from the preceding string
// and a suffix is stored. The prefix lengths are DELTA_BINARY_PACKED encoded. The suffixes are
// encoded with DELTA_LENGTH_BYTE_ARRAY encoding, which is a DELTA_BINARY_PACKED list of suffix
// lengths, followed by the concatenated suffix data.
struct delta_byte_array_decoder {
  uint8_t const* last_string;       // pointer to last decoded string...needed for its prefix
  uint8_t const* suffix_char_data;  // pointer to the start of character data

  uint8_t* temp_buf;         // scratch for strings skipped over by a leading row range; the next
                             // batch overwrites it from its start each round
  uint8_t* prefix_seed;      // one reserved slot ahead of temp_buf holding a durable copy of the
                             // last decoded string, used to seed the next batch's first prefix
  uint32_t start_val;        // decoded strings up to this index will be dumped to temp_buf
  uint32_t last_string_len;  // length of the last decoded string

  delta_binary_decoder prefixes;  // state of decoder for prefix lengths
  delta_binary_decoder suffixes;  // state of decoder for suffix lengths

  // initialize the prefixes and suffixes blocks
  __device__ void init(
    uint8_t const* start, uint8_t const* end, uint32_t start_idx, uint8_t* temp, size_t temp_size)
  {
    auto const* suffix_start = prefixes.find_end_of_block(start, end);
    suffix_char_data         = suffixes.find_end_of_block(suffix_start, end);
    last_string              = nullptr;
    // the temp allocation holds one leading string slot (see
    // `compute_delta_page_string_sizes_kernel`) followed by delta_max_batch_size scratch slots.
    // reserve the leading slot for the last decoded string so it stays clear of the scratch, which
    // each round overwrites from its start.
    prefix_seed = temp;
    temp_buf    = temp + temp_size / (delta_max_batch_size + 1);
    start_val   = start_idx;
  }

  // kind of like an inclusive scan for strings. takes prefix_len bytes from preceding
  // string and prepends to the suffix we've already copied into place. called from
  // within loop over values_in_mb, so this only needs to handle a single warp worth of data
  // at a time.
  __device__ void string_scan(uint8_t* strings_out,
                              uint8_t const* last_string,
                              uint32_t start_idx,
                              uint32_t end_idx,
                              uint32_t offset,
                              uint32_t lane_id)
  {
    using cudf::detail::warp_size;

    // let p(n) === length(prefix(string_n))
    //
    // if p(n-1) > p(n), then string_n can be completed when string_n-2 is completed. likewise if
    // p(m) > p(n), then string_n can be completed with string_m-1. however, if p(m) < p(n), then m
    // is a "blocker" for string_n; string_n can be completed only after string_m is.
    //
    // we will calculate the nearest blocking position for each lane, and then fill in string_0. we
    // then iterate, finding all lanes that have had their "blocker" filled in and completing them.
    // when all lanes are filled in, we return. this will still hit the worst case if p(n-1) < p(n)
    // for all n
    __shared__ __align__(8) int64_t prefix_lens[warp_size];
    __shared__ __align__(8) uint8_t const* offsets[warp_size];

    uint32_t const ln_idx   = start_idx + lane_id;
    uint64_t prefix_len     = ln_idx < end_idx ? prefixes.value_at(ln_idx) : 0;
    uint8_t* const lane_out = ln_idx < end_idx ? strings_out + offset : nullptr;

    // if all prefix_len's are zero, then there's nothing to do
    if (__all_sync(0xffff'ffff, prefix_len == 0)) { return; }

    prefix_lens[lane_id] = prefix_len;
    offsets[lane_id]     = lane_out;
    __syncwarp();

    // find a neighbor to the left that has a prefix length less than this lane. once that
    // neighbor is complete, this lane can be completed.
    int blocker = lane_id - 1;
    while (blocker > 0 && prefix_lens[blocker] != 0 && prefix_len <= prefix_lens[blocker]) {
      blocker--;
    }

    // fill in lane 0 (if necessary)
    if (lane_id == 0 && prefix_len > 0) {
      memcpy(lane_out, last_string, prefix_len);
      prefix_lens[0] = prefix_len = 0;
    }
    __syncwarp();

    // now fill in blockers until done
    for (uint32_t i = 1; i < warp_size && i + start_idx < end_idx; i++) {
      // record whether this lane's blocker is complete before any lane writes below
      // prevents race condition accessing prefix_lens
      bool const completed = prefix_len != 0 && prefix_lens[blocker] == 0 && lane_out != nullptr;
      __syncwarp();
      if (completed) {
        memcpy(lane_out, offsets[blocker], prefix_len);
        prefix_lens[lane_id] = prefix_len = 0;
      }
      __syncwarp();

      // check for finished
      if (__all_sync(0xffff'ffff, prefix_len == 0)) { return; }
    }
  }

  // calculate a mini-batch of string values, writing the results to
  // `strings_out`. starting at global index `start_idx` and decoding
  // up to `num_values` strings.
  // called by all threads in a warp. used for strings <= 32 chars.
  // returns number of bytes written
  __device__ size_t calculate_string_values(uint8_t* strings_out,
                                            uint32_t start_idx,
                                            uint32_t num_values,
                                            uint32_t lane_id)
  {
    using cudf::detail::warp_size;
    using WarpScan = cub::WarpScan<uint64_t>;
    __shared__ WarpScan::TempStorage scan_temp;

    if (start_idx >= suffixes.value_count) { return 0; }
    auto end_idx = start_idx + min(suffixes.values_per_mb, num_values);
    end_idx      = min(end_idx, static_cast<uint32_t>(suffixes.value_count));

    auto p_strings_out = strings_out;
    auto p_temp_out    = temp_buf;

    auto copy_batch = [&](uint8_t* out, uint32_t idx, uint32_t end) {
      uint32_t const ln_idx = idx + lane_id;

      // calculate offsets into suffix data
      uint64_t const suffix_len = ln_idx < end ? suffixes.value_at(ln_idx) : 0;
      uint64_t suffix_off       = 0;
      WarpScan(scan_temp).ExclusiveSum(suffix_len, suffix_off);

      // calculate offsets into string data
      uint64_t const prefix_len = ln_idx < end ? prefixes.value_at(ln_idx) : 0;
      uint64_t const string_len = prefix_len + suffix_len;

      // get offset into output for each lane
      uint64_t string_off, warp_total;
      WarpScan(scan_temp).ExclusiveSum(string_len, string_off, warp_total);
      auto const so_ptr = out + string_off;

      // copy suffixes into string data
      if (ln_idx < end) { memcpy(so_ptr + prefix_len, suffix_char_data + suffix_off, suffix_len); }
      __syncwarp();

      // copy prefixes into string data.
      string_scan(out, last_string, idx, end, string_off, lane_id);
      __syncwarp();

      // save the position of the last computed string. this will be used in
      // the next iteration to reconstruct the string in lane 0.
      if (ln_idx == end - 1 || (ln_idx < end && lane_id == 31)) {
        // set last_string to this lane's string
        last_string     = out + string_off;
        last_string_len = string_len;
        // and consume used suffix_char_data
        suffix_char_data += suffix_off + suffix_len;
      }

      return warp_total;
    };

    uint64_t string_total = 0;
    for (int idx = start_idx; idx < end_idx; idx += warp_size) {
      auto const n_in_batch = min(warp_size, end_idx - idx);
      // account for the case where start_val occurs in the middle of this batch
      if (idx < start_val && idx + n_in_batch > start_val) {
        // dump idx...start_val into temp_buf
        copy_batch(p_temp_out, idx, start_val);
        __syncwarp();

        // start_val...idx + n_in_batch into strings_out
        auto nbytes = copy_batch(p_strings_out, start_val, idx + n_in_batch);
        p_strings_out += nbytes;
        string_total = nbytes;
      } else {
        if (idx < start_val) {
          p_temp_out += copy_batch(p_temp_out, idx, end_idx);
        } else {
          auto nbytes = copy_batch(p_strings_out, idx, end_idx);
          p_strings_out += nbytes;
          string_total += nbytes;
        }
      }
      __syncwarp();
    }

    // the next batch overwrites the temp scratch from its start, so if the last decoded string
    // lives there, preserve it in the reserved seed slot ahead of the scratch
    if (end_idx <= start_val && last_string != prefix_seed) {
      // snapshot before lane 0 overwrites last_string below; without this sync, other lanes'
      // (unused but still issued) reads of last_string can race with lane 0's write of it.
      uint8_t const* const last_string_snapshot = last_string;
      uint32_t const last_string_len_snapshot   = last_string_len;
      __syncwarp();
      if (lane_id == 0) {
        memcpy(prefix_seed, last_string_snapshot, last_string_len_snapshot);
        last_string = prefix_seed;
      }
      __syncwarp();
    }

    return string_total;
  }

  // character parallel version of CalculateStringValues(). This is faster for strings longer than
  // 32 chars.
  __device__ size_t calculate_string_values_cp(uint8_t* strings_out,
                                               uint32_t start_idx,
                                               uint32_t num_values,
                                               uint32_t lane_id)
  {
    using cudf::detail::warp_size;
    __shared__ __align__(8) uint8_t* so_ptr;

    if (start_idx >= suffixes.value_count) { return 0; }
    auto end_idx = start_idx + min(suffixes.values_per_mb, num_values);
    end_idx      = min(end_idx, static_cast<uint32_t>(suffixes.value_count));

    if (lane_id == 0) { so_ptr = start_idx < start_val ? temp_buf : strings_out; }
    __syncwarp();

    uint64_t string_total = 0;
    for (int idx = start_idx; idx < end_idx; idx++) {
      uint64_t const suffix_len = suffixes.value_at(idx);
      uint64_t const prefix_len = prefixes.value_at(idx);
      uint64_t const string_len = prefix_len + suffix_len;

      // copy prefix and suffix data into current strings_out position
      // for longer strings use a 4-byte version stolen from gather_chars_fn_string_parallel.
      if (string_len > 64) {
        if (prefix_len > 0) { wideStrcpy(so_ptr, last_string, prefix_len, lane_id); }
        if (suffix_len > 0) {
          wideStrcpy(so_ptr + prefix_len, suffix_char_data, suffix_len, lane_id);
        }
      } else {
        for (int i = lane_id; i < string_len; i += warp_size) {
          so_ptr[i] = i < prefix_len ? last_string[i] : suffix_char_data[i - prefix_len];
        }
      }
      __syncwarp();

      if (idx >= start_val) { string_total += string_len; }

      if (lane_id == 0) {
        last_string     = so_ptr;
        last_string_len = string_len;
        suffix_char_data += suffix_len;
        if (idx == start_val - 1) {
          so_ptr = strings_out;
        } else {
          so_ptr += string_len;
        }
      }
      __syncwarp();
    }

    // the next batch overwrites the temp scratch from its start, so if the last decoded string
    // lives there, preserve it in the reserved seed slot ahead of the scratch
    if (end_idx <= start_val && last_string != prefix_seed) {
      // snapshot before lane 0 overwrites last_string below; without this sync, other lanes'
      // (unused but still issued) reads of last_string can race with lane 0's write of it.
      uint8_t const* const last_string_snapshot = last_string;
      uint32_t const last_string_len_snapshot   = last_string_len;
      __syncwarp();
      if (lane_id == 0) {
        memcpy(prefix_seed, last_string_snapshot, last_string_len_snapshot);
        last_string = prefix_seed;
      }
      __syncwarp();
    }

    return string_total;
  }

  // dump strings before start_val to temp buf. decodes one warp_size-wide pass per round, so
  // any mini-block size is supported. called by all threads in a thread block.
  __device__ void skip(bool use_char_ll,
                       cg::thread_block const& block,
                       cg::thread_block_tile<cudf::detail::warp_size, cg::thread_block> const& warp)
  {
    using cudf::detail::warp_size;

    // is this even necessary? return if asking to skip the whole block.
    if (start_val >= prefixes.num_encoded_values(true)) { return; }

    uint32_t skip_pos = 0;
    while (skip_pos < start_val) {
      // warp 0 decodes a pass of prefixes and warp 1 a pass of suffixes. this will potentially
      // decode past start_val, and those values stay resident in the rolling buffers for the
      // decode loop that follows. dispatch on a compile-time-constant decoder per warp (as the
      // main decode loop does) rather than a runtime-selected `db` pointer: selecting the object
      // at runtime makes the compiler speculatively load both prefixes and suffixes on every
      // `db->` access, so the suffix warp reads prefix state (and vice versa) while the other
      // warp writes it, which compute-sanitizer racecheck reports as a cross-warp hazard.
      if (warp.meta_group_rank() == 0) {
        prefixes.decode_next_pass(warp);
      } else if (warp.meta_group_rank() == 1) {
        suffixes.decode_next_pass(warp);
      }
      block.sync();

      // warp 0 reconstructs this round's skipped strings into the temp scratch (the helpers
      // preserve the round's last string past the scratch area for the next round's prefixes)
      if (warp.meta_group_rank() == 0) {
        auto const num_to_decode = min(static_cast<uint32_t>(warp_size), start_val - skip_pos);
        if (use_char_ll) {
          calculate_string_values_cp(temp_buf, skip_pos, num_to_decode, warp.thread_rank());
        } else {
          calculate_string_values(temp_buf, skip_pos, num_to_decode, warp.thread_rank());
        }
      }
      skip_pos += warp_size;
      block.sync();
    }
  }
};

// Decode page data that is DELTA_BINARY_PACKED encoded. This encoding is
// only used for int32 and int64 physical types (and appears to only be used
// with V2 page headers; see https://www.mail-archive.com/dev@parquet.apache.org/msg11826.html).
// this kernel only needs 96 threads (3 warps)(for now).
template <typename level_t>
CUDF_KERNEL void __launch_bounds__(decode_delta_binary_block_size)
  decode_delta_binary_kernel(PageInfo* pages,
                             device_span<ColumnChunkDesc const> chunks,
                             size_t min_row,
                             size_t num_rows,
                             cudf::device_span<bool const> page_mask,
                             bool has_flat_nz_map,
                             kernel_error::pointer error_code)
{
  __shared__ __align__(16) delta_binary_decoder db_state;
  __shared__ __align__(16) full_page_decode_state state_g;
  __shared__ __align__(16) page_state_buffers_s<delta_nz_buf_size, 1, 1> state_buffers;

  auto* const s      = &state_g;
  auto* const sb     = &state_buffers;
  int const page_idx = cg::this_grid().block_rank();
  auto const block   = cg::this_thread_block();
  auto const warp    = cg::tiled_partition<cudf::detail::warp_size>(block);
  auto* const db     = &db_state;

  // Exit early if the page is pruned
  if (page_mask.size() > 0 and not page_mask[page_idx]) { return; }
  // A page carrying an nz map is decoded by `decode_delta_binary_kernel_with_nz_map` instead.
  // Gated on the same flag that launches that kernel, so the two routing decisions are one bit.
  if (has_flat_nz_map and pages[page_idx].has_nz_map(nz_map_kind::DELTA_FLAT)) { return; }

  [[maybe_unused]] null_count_back_copier _{s, static_cast<int>(block.thread_rank())};

  // Setup local page info
  if (!setup_local_page_info(s,
                             &pages[page_idx],
                             chunks,
                             min_row,
                             num_rows,
                             mask_filter{decode_kernel_mask::DELTA_BINARY},
                             page_processing_stage::DECODE)) {
    return;
  }

  // Must be evaluated after setup_local_page_info
  bool const has_repetition = s->setup.col.max_level[level_type::REPETITION] > 0;
  bool const process_nulls  = should_process_nulls(s);

  // Capture initial valid_map_offset before any processing that might modify it
  int const init_valid_map_offset =
    s->nesting.nesting_info[s->setup.col.max_nesting_depth - 1].valid_map_offset;

  // copying logic from gpuDecodePageData.
  PageNestingDecodeInfo const* nesting_info_base = s->nesting.nesting_info;

  // Get the level decode buffers for this page
  PageInfo* pp       = &pages[page_idx];
  level_t* const def = !process_nulls
                         ? nullptr
                         : reinterpret_cast<level_t*>(pp->lvl_decode_buf[level_type::DEFINITION]);
  auto* const rep    = reinterpret_cast<level_t*>(pp->lvl_decode_buf[level_type::REPETITION]);

  // skipped_leaf_values will always be 0 for flat hierarchies.
  uint32_t const skipped_leaf_values = s->setup.page.skipped_leaf_values;

  // initialize delta state
  if (block.thread_rank() == 0) { db->init_binary_block(s->stream.data_start, s->stream.data_end); }
  block.sync();

  if (db->error) {
    if (block.thread_rank() == 0) {
      set_error(static_cast<kernel_error::value_type>(decode_error::DELTA_PARAMS_UNSUPPORTED),
                error_code);
    }
    return;
  }

  bool const is_skip_resume = skipped_leaf_values > 0;

  // Number of values produced per main-loop iteration: up to two warp_size passes, so pages whose
  // mini-blocks hold at least two passes keep the schedule of the whole-mini-block decoder. When
  // resuming after skip_values() the producer emits a single pass per iteration: the skip leaves
  // up to warp_size not-yet-consumed values in the rolling buffer, and a larger batch could wrap
  // around and overwrite them before the consumer reads them.
  uint32_t const batch_size =
    is_skip_resume ? cudf::detail::warp_size
                   : min(db->values_per_mb, static_cast<uint32_t>(delta_max_batch_size));
  uint32_t const passes_per_batch = batch_size / cudf::detail::warp_size;

  // if skipped_leaf_values is non-zero, then we need to decode up to the first mini-block
  // that has a value we need.
  if (is_skip_resume) { db->skip_values(skipped_leaf_values, block, warp); }

  while (s->setup.error == 0 && (s->progress.input_value_count < s->setup.num_input_values ||
                                 s->progress.src_pos < s->progress.nz_count)) {
    uint32_t target_pos;
    uint32_t const src_pos = s->progress.src_pos;

    if (warp.meta_group_rank() < 2) {  // warp0..1
      target_pos = min(src_pos + 2 * batch_size, s->progress.nz_count + batch_size);
    } else {  // warp2
      target_pos = min(s->progress.nz_count, src_pos + batch_size);
    }
    // This needs to be here to prevent warp 2 modifying src_pos before all threads have read it
    block.sync();

    // warp0 will decode the rep/def levels, warp1 will unpack a mini-batch of deltas.
    // warp2 waits one cycle for warps 0/1 to produce a batch, and then stuffs values
    // into the proper location in the output.
    if (warp.meta_group_rank() == 0) {
      // warp 0
      // decode repetition and definition levels.
      // - update validity vectors
      // - updates offsets (for nested columns)
      // - produces non-NULL value indices in s->nz_idx for subsequent decoding
      gpuDecodeLevels<delta_nz_buf_size, level_t>(s, sb, target_pos, rep, def, warp);
    } else if (warp.meta_group_rank() == 1) {
      // warp 1
      for (uint32_t i = 0; i < passes_per_batch; i++) {
        // make lane 0's state updates from the previous pass visible to the whole warp; the
        // block-wide sync below covers the last pass of the iteration
        if (i > 0) { warp.sync(); }
        db->decode_next_pass(warp);
      }
    } else if (src_pos < target_pos) {
      // warp 2
      // nesting level that is storing actual leaf values
      int const leaf_level_index = s->setup.col.max_nesting_depth - 1;

      // process the mini-block using warps
      for (uint32_t sp = src_pos + warp.thread_rank(); sp < src_pos + batch_size;
           sp += warp.size()) {
        // the position in the output column/buffer
        int32_t dst_pos = sb->nz_idx[rolling_index<delta_nz_buf_size>(sp)];

        // handle skip_rows here. flat hierarchies can just skip up to first_row.
        if (!has_repetition) { dst_pos -= s->setup.first_row; }

        // place value for this thread
        if (dst_pos >= 0 && sp < target_pos) {
          void* const dst =
            nesting_info_base[leaf_level_index].data_out + dst_pos * s->output_cvt.dtype_len;
          auto const val = db->value_at(sp + skipped_leaf_values);
          switch (s->output_cvt.dtype_len) {
            case 1: *static_cast<int8_t*>(dst) = val; break;
            case 2: *static_cast<int16_t*>(dst) = val; break;
            case 4: *static_cast<int32_t*>(dst) = val; break;
            case 8: *static_cast<int64_t*>(dst) = val; break;
          }
        }
      }
      if (warp.thread_rank() == 0) { s->progress.src_pos = src_pos + batch_size; }
    }

    block.sync();
  }

  if (has_repetition) {
    // Zero-fill null positions after decoding valid values
    auto const& ni = s->nesting.nesting_info[s->setup.col.max_nesting_depth - 1];
    if (ni.valid_map != nullptr) {
      int const num_values = ni.valid_map_offset - init_valid_map_offset;
      zero_fill_null_positions_shared<decode_delta_binary_block_size>(
        s,
        s->output_cvt.dtype_len,
        init_valid_map_offset,
        num_values,
        static_cast<int>(block.thread_rank()));
    }
  }

  // `check_mini_block` can reject a later mini-block mid-decode. Report it like a bad header.
  if (block.thread_rank() == 0 and db->error) {
    set_error(static_cast<kernel_error::value_type>(decode_error::DELTA_PARAMS_UNSUPPORTED),
              error_code);
  }
  if (block.thread_rank() == 0 and s->setup.error != 0) { set_error(s->setup.error, error_code); }
}

// Null count for the rows a page contributes, for the nz map consumers: `num_rows` minus the map's
// valid values at or after `first_row`. The map also holds the valid values before `first_row`; it
// is sorted, so there are `first_row`'s lower bound of them. Only the first page of a read has
// `first_row > 0`, so the search is skipped otherwise.
[[nodiscard]] __device__ int nz_map_null_count(uint32_t const* nz_idx,
                                               int nz_count,
                                               int first_row,
                                               int num_rows)
{
  auto const prefix_valid_count =
    first_row > 0
      ? static_cast<int>(
          cuda::std::lower_bound(nz_idx, nz_idx + nz_count, static_cast<uint32_t>(first_row)) -
          nz_idx)
      : 0;
  return num_rows - (nz_count - prefix_valid_count);
}

/**
 * @brief Warp-wide, register-resident DELTA_BINARY_PACKED decoder used by
 * `decode_delta_binary_kernel_with_nz_map`.
 *
 * One warp decodes one page in tiles of 128 values, four adjacent values per lane. Every lane runs
 * the same warp-uniform walk over blocks, mini-blocks and 32-value passes in registers, so the tile
 * descriptors need no shared memory, no single-lane parsing and no warp syncs, and the running sum
 * stays in a register. Each lane loads the packed bits of its four values with aligned 8-byte
 * loads, one tile ahead of decoding. An aligned word holding at least one in-bounds byte cannot
 * leave the allocation, and bytes past the end of the page read as zero, matching the bounded byte
 * loads of `delta_binary_decoder`. Page data is prefetched into L2 ahead of the read position.
 */
namespace nz_register_delta {

using warp_tile = cg::thread_block_tile<cudf::detail::warp_size>;

constexpr uint32_t warp_lanes     = cudf::detail::warp_size;
constexpr uint32_t items_per_lane = 4;
constexpr uint32_t tile_size      = items_per_lane * warp_lanes;
// Lanes sharing one 32-value pass of a tile
constexpr uint32_t lanes_per_pass = warp_lanes / items_per_lane;
constexpr uint32_t byte_bits      = 8;
// Bytes of packed data in one 32-value pass per bit of width
constexpr uint32_t pass_bytes_per_bit = warp_lanes / byte_bits;
constexpr uint32_t word_bytes         = sizeof(uint64_t);
constexpr uint32_t word_bits          = word_bytes * byte_bits;
// Aligned words covering one lane's values: a start offset below one word plus four values of at
// most one word each
constexpr uint32_t words_per_lane = items_per_lane + 1;
// A zigzag ULEB128 encoding of a 64-bit value takes at most 10 bytes
constexpr uint32_t max_varint_bytes = 10;
// Block header bytes decoded from the staged words: the minimum delta and the bit widths
constexpr uint32_t header_bytes = 2 * word_bytes;
// Aligned words covering `header_bytes` from any start address
constexpr uint32_t header_words = header_bytes / word_bytes + 1;
// Bit widths that fit in one word
constexpr uint32_t max_packed_widths = word_bytes;
constexpr uint32_t cache_line_bytes  = 128;
// Bytes prefetched into L2 per burst; the next burst is issued once the read position is within
// half a window of the prefetched end
constexpr uint32_t prefetch_window = 4'096;
static_assert(prefetch_window % (cache_line_bytes * warp_lanes) == 0,
              "Each lane prefetches the same number of cache lines per burst");

using exchange_type = cub::WarpExchange<uint64_t, items_per_lane, warp_lanes>;

/**
 * @brief Shared memory used by `decode_page`
 */
struct shared_storage {
  typename exchange_type::TempStorage exchange;
  alignas(word_bytes) uint64_t header[header_words];
};

/**
 * @brief Returns `w[i]` for an index known to be below `K`.
 *
 * Selecting through a chain of conditional moves keeps `w` in registers; a dynamic index would
 * place it in local memory.
 */
template <uint32_t K, uint32_t N>
__device__ __forceinline__ uint64_t select_word(uint64_t const (&w)[N], uint32_t i)
{
  static_assert(K >= 1 && K <= N, "The index bound must lie within the array");
  uint64_t result = w[0];
#pragma unroll
  for (uint32_t k = 1; k < K; ++k) {
    result = i == k ? w[k] : result;
  }
  return result;
}

/**
 * @brief Returns the 64 bits starting at bit `shift` (< 64) of the 128-bit value `hi:lo`.
 */
__device__ __forceinline__ uint64_t extract64(uint64_t lo, uint64_t hi, uint32_t shift)
{
  bool const upper = shift >= 32;
  uint32_t const a = upper ? static_cast<uint32_t>(lo >> 32) : static_cast<uint32_t>(lo);
  uint32_t const b = upper ? static_cast<uint32_t>(hi) : static_cast<uint32_t>(lo >> 32);
  uint32_t const c = upper ? static_cast<uint32_t>(hi >> 32) : static_cast<uint32_t>(hi);
  return (static_cast<uint64_t>(__funnelshift_r(b, c, shift)) << 32) | __funnelshift_r(a, b, shift);
}

/**
 * @brief Returns the number of bytes up to and including the lowest byte of `stop` whose top bit is
 * set.
 *
 * `stop` must be nonzero and have only the top bit of each byte set. The lowest set bit, 8k+7 for
 * byte k, has a bit width of 8(k+1).
 */
__device__ __forceinline__ uint32_t stop_byte_count(uint64_t stop)
{
  return static_cast<uint32_t>(cuda::std::bit_width(stop & (0 - stop))) / byte_bits;
}

/**
 * @brief Returns the aligned word address containing `p`.
 */
__device__ __forceinline__ uintptr_t aligned_word(uint8_t const* p)
{
  return reinterpret_cast<uintptr_t>(p) & ~uintptr_t{word_bytes - 1};
}

/**
 * @brief Returns `w` with the bytes at or past `end` cleared, for the word at address `word`.
 */
__device__ __forceinline__ uint64_t clear_past_end(uint64_t w, uintptr_t word, uintptr_t end)
{
  if (word >= end) { return 0; }
  if (word + word_bytes > end) { w &= (uint64_t{1} << (byte_bits * (end - word))) - 1; }
  return w;
}

/**
 * @brief Warp-uniform walker over the blocks, mini-blocks and 32-value passes of a page.
 *
 * Block headers are staged one block ahead: on entering block k the walker computes where block
 * k+1 starts and copies the words holding its header into shared memory asynchronously. Entering
 * block k+1 then decodes the minimum delta and the bit widths from those words in registers, which
 * keeps the chain of dependent byte loads through each header off the critical path. Headers that
 * do not fit the staged words (a minimum delta longer than 10 bytes, more than 8 mini-blocks) are
 * parsed from global memory instead.
 */
struct walker {
  uint8_t const* end;
  uint64_t* staged;  ///< `header_words` words of shared memory
  uint32_t mini_block_count;
  uint32_t passes_per_mb;
  uint8_t const* mb_start  = nullptr;  ///< Packed data of the current mini-block
  uint8_t const* widths    = nullptr;  ///< Bit widths of the current block, when not `packed`
  uint8_t const* staged_at = nullptr;  ///< Block start described by the staged header words
  cuda::pipeline<cuda::thread_scope_thread> pipe = cuda::make_pipeline();
  zigzag128_t min_delta                          = 0;
  uint64_t packed_widths = 0;  ///< Bit widths of the current block, one byte each, when `packed`
  uint32_t mb            = 0;
  uint32_t pass          = 0;
  uint32_t bits          = 0;
  bool packed            = false;
  bool error             = false;

  [[nodiscard]] __device__ __forceinline__ uint32_t width(uint32_t m) const
  {
    if (packed) { return static_cast<uint32_t>(packed_widths >> (byte_bits * m)) & 0xffu; }
    return widths + m < end ? widths[m] : 0u;
  }

  __device__ __forceinline__ void set_bits(uint32_t m)
  {
    bits = width(m);
    if (bits > word_bits) {
      error = true;
      bits  = 0;
    }
  }

  /**
   * @brief Waits for the staged header copy, if any, to land in shared memory.
   */
  __device__ __forceinline__ void wait_staged() { cuda::pipeline_consumer_wait_prior<0>(pipe); }

  /**
   * @brief Starts copying the aligned words covering [at, at + header_bytes) into `staged`.
   *
   * Words wholly past the end of the page are not copied; `enter_block` reads them as zero.
   */
  __device__ __forceinline__ void stage_header(uint8_t const* at, warp_tile const& warp)
  {
    wait_staged();  // no copy may still target `staged`
    warp.sync();    // every lane has read the previous header
    staged_at       = at;
    auto const lane = warp.thread_rank();
    auto const word = aligned_word(at) + word_bytes * lane;
    pipe.producer_acquire();
    if (lane < header_words && word < reinterpret_cast<uintptr_t>(end)) {
      // Lets `memcpy_async` select the global-to-shared copy at compile time instead of checking
      // both address spaces at run time.
      __builtin_assume(__isShared(staged + lane));
      __builtin_assume(__isGlobal(reinterpret_cast<void const*>(word)));
      cuda::memcpy_async(staged + lane,
                         reinterpret_cast<uint64_t const*>(word),
                         cuda::aligned_size_t<word_bytes>{word_bytes},
                         pipe);
    }
    pipe.producer_commit();
  }

  /**
   * @brief Parses the header of the block starting at `p` and stages the next block's header.
   */
  __device__ __forceinline__ void enter_block(uint8_t const* p, warp_tile const& warp)
  {
    bool const from_staged = staged_at == p;
    uint32_t len           = 0;
    uint64_t lo            = 0;
    uint64_t hi            = 0;
    if (from_staged) {
      wait_staged();
      warp.sync();
      auto const aligned  = aligned_word(p);
      auto const page_end = reinterpret_cast<uintptr_t>(end);
      uint64_t w[header_words];
#pragma unroll
      for (uint32_t i = 0; i < header_words; ++i) {
        auto const word = aligned + word_bytes * i;
        w[i]            = word < page_end ? clear_past_end(staged[i], word, page_end) : 0;
      }
      // The `header_bytes` bytes starting at p; bytes past the end of the page read as zero, like
      // the bounded byte loads of the fallback.
      auto const shift =
        static_cast<uint32_t>(reinterpret_cast<uintptr_t>(p) - aligned) * byte_bits;
      lo = extract64(w[0], w[1], shift);
      hi = extract64(w[1], w[2], shift);
      // The varint ends at the first byte without the continuation bit.
      constexpr uint64_t continuation_bits = 0x8080'8080'8080'8080ull;
      uint64_t const stop_lo               = ~lo & continuation_bits;
      uint64_t const stop_hi               = ~hi & continuation_bits;
      len                                  = stop_lo != 0
                                               ? stop_byte_count(stop_lo)
                                               : (stop_hi != 0 ? word_bytes + stop_byte_count(stop_hi) : header_bytes + 1);
    }
    bool const fits = from_staged && mini_block_count <= max_packed_widths &&
                      len <= max_varint_bytes && len + mini_block_count <= header_bytes;
    if (fits) {
      uint64_t v = 0;
#pragma unroll
      for (uint32_t i = 0; i < max_varint_bytes; ++i) {
        uint64_t const byte =
          (i < word_bytes ? lo >> (byte_bits * i) : hi >> (byte_bits * (i - word_bytes))) & 0x7f;
        if (i < len) { v |= byte << (7 * i); }
      }
      min_delta = static_cast<zigzag128_t>((v >> 1u) ^ -static_cast<zigzag128_t>(v & 1));
      // The bit widths are the bytes following the varint; 1 <= len <= 10.
      uint64_t const widths_lo = len < word_bytes ? lo >> (byte_bits * len) : 0;
      uint64_t const widths_hi = len < word_bytes ? hi << (word_bits - byte_bits * len)
                                                  : hi >> (byte_bits * (len - word_bytes));
      packed_widths            = widths_lo | widths_hi;
      if (mini_block_count < max_packed_widths) {
        packed_widths &= (uint64_t{1} << (byte_bits * mini_block_count)) - 1;
      }
      packed   = true;
      mb_start = p + len + mini_block_count;
    } else {
      min_delta = get_zz128(p, end);
      widths    = p;
      packed    = false;
      mb_start  = p + mini_block_count;
    }
    mb   = 0;
    pass = 0;
    set_bits(0);

    // Stage the header of the following block.
    uint32_t width_sum = 0;
    if (packed) {
      // Byte-wise sums of the two halves of the packed widths
      width_sum = __dp4a(static_cast<uint32_t>(packed_widths), 0x0101'0101u, 0u);
      width_sum = __dp4a(static_cast<uint32_t>(packed_widths >> 32), 0x0101'0101u, width_sum);
    } else {
      for (uint32_t m = 0; m < mini_block_count; ++m) {
        width_sum += width(m);
      }
    }
    auto const* next =
      mb_start + static_cast<size_t>(width_sum) * passes_per_mb * pass_bytes_per_bit;
    if (next >= mb_start && next < end) {
      stage_header(next, warp);
    } else {
      staged_at = nullptr;
    }
  }

  /**
   * @brief Moves to the next 32-value pass.
   */
  __device__ __forceinline__ void advance(warp_tile const& warp)
  {
    if (++pass == passes_per_mb) {
      pass = 0;
      mb_start += static_cast<size_t>(bits) * passes_per_mb * pass_bytes_per_bit;
      if (++mb == mini_block_count) {
        enter_block(mb_start, warp);
      } else {
        set_bits(mb);
      }
    }
  }

  /**
   * @brief Returns the packed data of the current pass.
   */
  [[nodiscard]] __device__ __forceinline__ uint8_t const* pass_start() const
  {
    return mb_start + pass * bits * pass_bytes_per_bit;
  }
};

/**
 * @brief Packed input of one lane for one tile.
 */
struct tile_input {
  uint64_t words[words_per_lane];  ///< Aligned words covering this lane's bits
  zigzag128_t min_delta;
  uint32_t bits;
  uint32_t shift;  ///< Bit offset of the lane's first value within `words[0]`
};

/**
 * @brief Walks the (up to) four passes of the tile at the walker's position and loads this lane's
 * packed input.
 *
 * Leaves the walker at the first pass of the following tile when there is one.
 */
__device__ __forceinline__ tile_input load_tile(walker& wk,
                                                uint32_t remaining,
                                                warp_tile const& warp)
{
  uint32_t const lane    = warp.thread_rank();
  uint32_t const my_pass = lane / lanes_per_pass;
  uint8_t const* start   = wk.pass_start();
  uint32_t bits          = wk.bits;
  zigzag128_t min_delta  = wk.min_delta;
#pragma unroll
  for (uint32_t i = 1; i < items_per_lane; ++i) {
    if (remaining > i * warp_lanes) {
      wk.advance(warp);
      if (my_pass == i) {
        start     = wk.pass_start();
        bits      = wk.bits;
        min_delta = wk.min_delta;
      }
    }
  }
  if (remaining > tile_size) { wk.advance(warp); }

  tile_input result{};
  if (remaining > my_pass * warp_lanes) {
    result.min_delta         = min_delta;
    result.bits              = bits;
    uint32_t const first_bit = (lane % lanes_per_pass) * items_per_lane * bits;
    auto const* const first  = start + first_bit / byte_bits;
    auto const aligned       = aligned_word(first);
    result.shift = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(first) - aligned) * byte_bits +
                   first_bit % byte_bits;
    uint32_t const used_bits = result.shift + items_per_lane * bits;
    auto const page_end      = reinterpret_cast<uintptr_t>(wk.end);
    auto const* const src    = reinterpret_cast<uint64_t const*>(aligned);
#pragma unroll
    for (uint32_t i = 0; i < words_per_lane; ++i) {
      bool const needed =
        bits != 0 && i * word_bits < used_bits && aligned + word_bytes * i < page_end;
      result.words[i] = needed ? src[i] : 0;
    }
    if (aligned + word_bytes * words_per_lane > page_end) {
#pragma unroll
      for (uint32_t i = 0; i < words_per_lane; ++i) {
        result.words[i] = clear_past_end(result.words[i], aligned + word_bytes * i, page_end);
      }
    }
  }
  return result;
}

/**
 * @brief Unpacks the `Item`-th value of a lane from its tile input.
 */
template <uint32_t Item>
__device__ __forceinline__ uint64_t unpack(tile_input const& in)
{
  // shift < 64 and bits <= 64, so item k starts in word k or earlier.
  uint32_t const bit = in.shift + Item * in.bits;
  uint32_t const idx = bit / word_bits;
  uint64_t const lo  = select_word<Item + 1>(in.words, idx);
  uint64_t hi;
  if constexpr (Item == 0) {
    hi = in.words[1];
  } else {
    uint64_t const upper[words_per_lane - 1] = {in.words[1], in.words[2], in.words[3], in.words[4]};
    hi                                       = select_word<Item + 1>(upper, idx);
  }
  uint64_t const mask = in.bits < word_bits ? (uint64_t{1} << in.bits) - 1 : ~uint64_t{0};
  return extract64(lo, hi, bit % word_bits) & mask;
}

/**
 * @brief Decodes the values of one DELTA_BINARY_PACKED page.
 *
 * Called by every lane of one warp. The values go to `consumer`, whose members are also called by
 * every lane together, so they may use warp collectives. `value_limit` below is the header's value
 * count capped at `nz_count`; ordinal 0 is the page's first value, and the remaining ordinals
 * arrive in tiles of `tile_size` starting at ordinal 1.
 *
 * - `place_first(uint64_t value, uint32_t value_limit)`: ordinal 0, when `value_limit > 0`
 * - `prefetch(uint32_t tile_start, uint32_t value_limit)`: the tile after the one about to be
 *   placed, called while that tile's packed input is loaded
 * - `next_tile()`: the most recently prefetched tile becomes the one to place
 * - `place(uint32_t tile_start, uint64_t const (&values)[items_per_lane], uint32_t value_limit)`:
 *   `values[item]` holds ordinal `tile_start + item * 32 + lane`, valid below `value_limit`
 *
 * @tparam Consumer Type of `consumer`
 *
 * @param warp The decoding warp
 * @param data Start of the DELTA_BINARY_PACKED stream
 * @param end End of the page data
 * @param nz_count Number of values to decode
 * @param storage Shared memory for this warp
 * @param consumer Receives the decoded values
 * @return 0, `DELTA_PARAMS_UNSUPPORTED` if the page header or a mini-block bit width is invalid,
 * or `DATA_STREAM_OVERRUN` if the page holds fewer than `nz_count` values
 */
template <typename Consumer>
__device__ kernel_error::value_type decode_page(warp_tile const& warp,
                                                uint8_t const* data,
                                                uint8_t const* end,
                                                uint32_t nz_count,
                                                shared_storage& storage,
                                                Consumer& consumer)
{
  uint32_t const lane = warp.thread_rank();

  auto const* p                 = data;
  auto const block_size         = get_uleb128(p, end);
  auto const mini_block_count   = get_uleb128(p, end);
  auto const value_count        = get_uleb128(p, end);
  zigzag128_t const first_value = get_zz128(p, end);
  if (mini_block_count == 0 or block_size == 0 or (block_size % mini_block_count) != 0 or
      ((block_size / mini_block_count) % delta_mini_block_size_multiple) != 0) {
    return static_cast<kernel_error::value_type>(decode_error::DELTA_PARAMS_UNSUPPORTED);
  }
  walker wk{end,
            storage.header,
            static_cast<uint32_t>(mini_block_count),
            static_cast<uint32_t>((block_size / mini_block_count) / warp_lanes)};
  uint32_t const value_limit = cuda::std::min(nz_count, static_cast<uint32_t>(value_count));
  consumer.place_first(static_cast<uint64_t>(first_value), value_limit);
  uint32_t position = cuda::std::min(value_limit, uint32_t{1});
  uint64_t carry    = static_cast<uint64_t>(first_value);

  // Prefetch the page into L2 in bursts ahead of the read position.
  uintptr_t prefetched = reinterpret_cast<uintptr_t>(p) & ~uintptr_t{cache_line_bytes - 1};
  auto const page_end  = reinterpret_cast<uintptr_t>(end);
  auto prefetch_burst  = [&]() {
#pragma unroll
    for (uint32_t k = 0; k < prefetch_window / (cache_line_bytes * warp_lanes); ++k) {
      auto const line = prefetched + (k * warp_lanes + lane) * cache_line_bytes;
      if (line < page_end) {
        // Page data is in global memory; this drops the run-time address space check.
        __builtin_assume(__isGlobal(reinterpret_cast<void const*>(line)));
        cuda::apply_access_property(
          reinterpret_cast<void const*>(line), cache_line_bytes, cuda::access_property::normal{});
      }
    }
    prefetched += prefetch_window;
  };

  tile_input current{};
  if (position < value_limit) {
    prefetch_burst();
    wk.stage_header(p, warp);
    wk.enter_block(p, warp);
    current = load_tile(wk, value_limit - position, warp);
    consumer.prefetch(position, value_limit);
  }
  while (position < value_limit) {
    if (reinterpret_cast<uintptr_t>(wk.mb_start) + prefetch_window / 2 > prefetched &&
        prefetched < page_end) {
      prefetch_burst();
    }
    uint32_t const remaining     = value_limit - position;
    uint32_t const next_position = position + cuda::std::min(uint32_t{tile_size}, remaining);
    consumer.next_tile();
    tile_input next{};
    if (next_position < value_limit) {
      next = load_tile(wk, value_limit - next_position, warp);
      consumer.prefetch(next_position, value_limit);
    }

    uint64_t values[items_per_lane];
    values[0] = unpack<0>(current);
    values[1] = unpack<1>(current);
    values[2] = unpack<2>(current);
    values[3] = unpack<3>(current);
#pragma unroll
    for (uint32_t item = 0; item < items_per_lane; ++item) {
      // Inactive entries must be zero, including their minimum-delta contribution.
      values[item] = lane * items_per_lane + item < remaining
                       ? values[item] + static_cast<uint64_t>(current.min_delta)
                       : uint64_t{0};
    }
#pragma unroll
    for (uint32_t item = 1; item < items_per_lane; ++item) {
      values[item] += values[item - 1];
    }
    uint64_t const lane_total = values[items_per_lane - 1];
    uint64_t const prefix     = cg::inclusive_scan(warp, lane_total, cg::plus<uint64_t>{});
    uint64_t const tile_total = warp.shfl(prefix, warp_lanes - 1);
    uint64_t const offset     = carry + prefix - lane_total;
    carry += tile_total;
#pragma unroll
    for (uint32_t item = 0; item < items_per_lane; ++item) {
      values[item] += offset;
    }
    // The values are blocked; transpose them once for coalesced output stores.
    exchange_type(storage.exchange).BlockedToStriped(values, values);
    consumer.place(position, values, value_limit);
    // The next tile's exchange reuses the storage.
    warp.sync();
    position = next_position;
    current  = next;
  }
  // An in-flight header copy must land before the block's shared memory is released.
  wk.wait_staged();
  if (wk.error) {
    return static_cast<kernel_error::value_type>(decode_error::DELTA_PARAMS_UNSUPPORTED);
  }
  if (position < nz_count) {
    return static_cast<kernel_error::value_type>(decode_error::DATA_STREAM_OVERRUN);
  }
  return 0;
}

/**
 * @brief `decode_page` consumer that writes ordinal `i` to `output[i - skip]`.
 */
struct direct_output {
  int64_t* output;  ///< Output of ordinal `skip`
  uint32_t skip;    ///< Leading ordinals to drop
  uint32_t lane;

  __device__ __forceinline__ void place_first(uint64_t value, uint32_t value_limit) const
  {
    if (lane == 0 && value_limit > 0 && skip == 0) { output[0] = static_cast<int64_t>(value); }
  }

  __device__ __forceinline__ void prefetch(uint32_t, uint32_t) const {}

  __device__ __forceinline__ void next_tile() const {}

  __device__ __forceinline__ void place(uint32_t tile_start,
                                        uint64_t const (&values)[items_per_lane],
                                        uint32_t value_limit) const
  {
#pragma unroll
    for (uint32_t item = 0; item < items_per_lane; ++item) {
      uint32_t const ordinal = tile_start + item * warp_lanes + lane;
      if (ordinal < value_limit && ordinal >= skip) {
        output[ordinal - skip] = static_cast<int64_t>(values[item]);
      }
    }
  }
};

/**
 * @brief Stores the low `Width` bytes of `value` at `dst`; a `Width` of 0 takes the width from
 * `dtype_len` at run time.
 */
template <uint32_t Width>
__device__ __forceinline__ void store_value(uint8_t* dst, uint32_t dtype_len, uint64_t value)
{
  switch (Width != 0 ? Width : dtype_len) {
    case 1: *reinterpret_cast<int8_t*>(dst) = static_cast<int8_t>(value); break;
    case 2: *reinterpret_cast<int16_t*>(dst) = static_cast<int16_t>(value); break;
    case 4: *reinterpret_cast<int32_t*>(dst) = static_cast<int32_t>(value); break;
    case 8: *reinterpret_cast<int64_t*>(dst) = static_cast<int64_t>(value); break;
    default: break;
  }
}

/**
 * @brief `decode_page` consumer that places ordinals through the nz map's rank map and, when
 * `fill_nulls` is set, zero-fills the null rows between them.
 *
 * The rank map entries of a tile are loaded one tile ahead, while the decoder loads that tile's
 * packed input, so the stores do not wait on them. The first row of the null gap before an
 * ordinal follows the previous ordinal's row, which a neighboring lane already holds.
 *
 * @tparam Width Output width in bytes, or 0 to switch on `dtype_len` at run time
 */
template <uint32_t Width>
struct rank_map_output {
  warp_tile warp;
  uint8_t* data_out;       ///< Output of row `first_row`
  uint32_t const* nz_idx;  ///< Rank map, or null when each ordinal is its own row
  uint32_t dtype_len;
  int32_t first_row;  ///< First row the page outputs
  int32_t row_end;    ///< One past the last row the page outputs
  bool fill_nulls;
  int32_t rows[items_per_lane]{};       ///< Rows of the tile being placed, striped like the values
  int32_t next_rows[items_per_lane]{};  ///< Rows of the prefetched tile
  int32_t last_row{};                   ///< Row of the ordinal before the tile being placed

  [[nodiscard]] __device__ __forceinline__ int32_t row_of(uint32_t ordinal) const
  {
    return nz_idx != nullptr ? static_cast<int32_t>(nz_idx[ordinal])
                             : static_cast<int32_t>(ordinal);
  }

  __device__ __forceinline__ void store(int32_t row, uint64_t value) const
  {
    store_value<Width>(
      data_out + static_cast<int64_t>(row - first_row) * (Width != 0 ? Width : dtype_len),
      dtype_len,
      value);
  }

  /**
   * @brief Places one ordinal per active lane and, when `fill_nulls` is set, fills the null rows
   * between the previous value and it.
   *
   * Short gaps are filled by the lane that owns the ordinal, long ones by the whole warp.
   */
  __device__ __forceinline__ void place_one(uint64_t value,
                                            bool active,
                                            int32_t row,
                                            int32_t gap_lo) const
  {
    if (active && row >= first_row) { store(row, value); }
    if (!fill_nulls) { return; }
    constexpr int32_t short_gap = 8;
    // The gap's rows that this page outputs
    int32_t lo = active ? cuda::std::max(gap_lo, first_row) : 0;
    int32_t hi = active ? cuda::std::min(row, row_end) : 0;
    if (hi - lo <= short_gap) {
      for (auto r = lo; r < hi; ++r) {
        store(r, 0);
      }
      lo = hi;
    }
    auto const lane = static_cast<int32_t>(warp.thread_rank());
    for (uint32_t pending = warp.ballot(hi > lo); pending != 0; pending &= pending - 1) {
      auto const owner     = static_cast<uint32_t>(cuda::std::countr_zero(pending));
      auto const gap_start = warp.shfl(lo, owner);
      auto const gap_end   = warp.shfl(hi, owner);
      for (auto r = gap_start + lane; r < gap_end; r += warp_tile::num_threads()) {
        store(r, 0);
      }
    }
  }

  __device__ __forceinline__ void place_first(uint64_t value, uint32_t value_limit)
  {
    last_row = value_limit > 0 ? row_of(0) : 0;
    place_one(value, warp.thread_rank() == 0 && value_limit > 0, last_row, 0);
  }

  __device__ __forceinline__ void prefetch(uint32_t tile_start, uint32_t value_limit)
  {
#pragma unroll
    for (uint32_t item = 0; item < items_per_lane; ++item) {
      uint32_t const ordinal = tile_start + item * warp_lanes + warp.thread_rank();
      next_rows[item]        = ordinal < value_limit ? row_of(ordinal) : 0;
    }
  }

  __device__ __forceinline__ void next_tile()
  {
#pragma unroll
    for (uint32_t item = 0; item < items_per_lane; ++item) {
      rows[item] = next_rows[item];
    }
  }

  __device__ __forceinline__ void place(uint32_t tile_start,
                                        uint64_t const (&values)[items_per_lane],
                                        uint32_t value_limit)
  {
    uint32_t const lane = warp.thread_rank();
    // Row of each ordinal's predecessor: the same item on the previous lane, the previous item on
    // the last lane, or, for the tile's first ordinal, the last row of the previous (full) tile.
    // Only the null fill reads them.
    int32_t previous[items_per_lane]{};
    if (fill_nulls) {
#pragma unroll
      for (uint32_t item = 0; item < items_per_lane; ++item) {
        auto const from_lane_below = warp.shfl_up(rows[item], 1);
        auto const from_last_lane =
          item == 0 ? last_row : warp.shfl(rows[item - 1], warp_lanes - 1);
        previous[item] = lane == 0 ? from_last_lane : from_lane_below;
      }
      last_row = warp.shfl(rows[items_per_lane - 1], warp_lanes - 1);
    }
#pragma unroll
    for (uint32_t item = 0; item < items_per_lane; ++item) {
      uint32_t const ordinal = tile_start + item * warp_lanes + lane;
      place_one(values[item], ordinal < value_limit, rows[item], previous[item] + 1);
    }
  }

  /**
   * @brief When `fill_nulls` is set, zero-fills the null rows after the page's last value.
   *
   * @param nz_count Number of valid values in the page, all of them placed
   */
  __device__ __forceinline__ void fill_tail(uint32_t nz_count) const
  {
    if (!fill_nulls) { return; }
    auto const start =
      nz_count > 0 ? cuda::std::max(row_of(nz_count - 1) + 1, first_row) : first_row;
    for (auto row = start + static_cast<int32_t>(warp.thread_rank()); row < row_end;
         row += warp_tile::num_threads()) {
      store(row, 0);
    }
  }
};

}  // namespace nz_register_delta

/**
 * @brief Returns whether a mapped page is decoded by the no-null path: required, flat INT64
 * columns, whose rank map is the identity and whose output is eight bytes wide.
 *
 * Optional columns and all other physical types take the general path.
 */
CUDF_HOST_DEVICE inline bool takes_no_null_path(PageInfo const& page, ColumnChunkDesc const& chunk)
{
  return BitAnd(page.kernel_mask, decode_kernel_mask::DELTA_BINARY) != 0 &&
         chunk.physical_type == Type::INT64 && chunk.max_level[level_type::DEFINITION] == 0 &&
         chunk.max_level[level_type::REPETITION] == 0 && chunk.max_nesting_depth == 1;
}

/**
 * @brief Decodes a page `takes_no_null_path` accepts, writing its values straight to the output.
 */
__device__ __forceinline__ void decode_required_page(nz_register_delta::warp_tile const& warp,
                                                     PageInfo& page,
                                                     ColumnChunkDesc const& chunk,
                                                     size_t min_row,
                                                     size_t num_rows,
                                                     nz_register_delta::shared_storage& storage,
                                                     kernel_error::pointer error_code)
{
  cg::invoke_one(warp, [&] {
    page.nesting_decode[0].null_count  = 0;
    page.nesting_decode[0].valid_count = 0;
    page.nesting_decode[0].value_count = 0;
  });
  if (!page_has_rows_to_process(page, chunk.start_row, min_row, num_rows, false)) { return; }
  size_t const page_start = chunk.start_row + page.chunk_row;
  auto const skip         = static_cast<uint32_t>(page_start < min_row ? min_row - page_start : 0);
  auto* const output      = static_cast<int64_t*>(chunk.column_data_base[0]) +
                       (page_start >= min_row ? page_start - min_row : 0);
  auto const nz_count       = page.nz_map->nz_count;
  int64_t const level_bytes = (page.flags & PAGEINFO_FLAGS_V2)
                                ? static_cast<int64_t>(page.lvl_bytes[level_type::DEFINITION]) +
                                    page.lvl_bytes[level_type::REPETITION]
                                : 0;
  auto error = static_cast<kernel_error::value_type>(decode_error::DATA_STREAM_OVERRUN);
  if (nz_count >= 0 && nz_count <= page.num_input_values && level_bytes >= 0 &&
      level_bytes <= page.uncompressed_page_size) {
    nz_register_delta::direct_output consumer{output, skip, warp.thread_rank()};
    error = nz_register_delta::decode_page(warp,
                                           page.page_data + level_bytes,
                                           page.page_data + page.uncompressed_page_size,
                                           static_cast<uint32_t>(nz_count),
                                           storage,
                                           consumer);
  }
  if (error != 0) {
    cg::invoke_one(warp, [&] { set_error(error, error_code); });
  }
}

/**
 * @brief Decodes a mapped page through its rank map.
 */
__device__ __forceinline__ void decode_mapped_page(nz_register_delta::warp_tile const& warp,
                                                   PageInfo* page,
                                                   device_span<ColumnChunkDesc const> chunks,
                                                   size_t min_row,
                                                   size_t num_rows,
                                                   nz_register_delta::shared_storage& storage,
                                                   kernel_error::pointer error_code)
{
  __shared__ __align__(16) full_page_decode_state state_g;
  auto* const s    = &state_g;
  auto const block = cg::this_thread_block();

  [[maybe_unused]] null_count_back_copier _{s, static_cast<int32_t>(block.thread_rank())};
  if (!setup_local_page_info(s,
                             page,
                             chunks,
                             min_row,
                             num_rows,
                             mask_filter{decode_kernel_mask::DELTA_BINARY},
                             page_processing_stage::DECODE)) {
    return;
  }

  bool const process_nulls = should_process_nulls(s);
  auto const* const nz_idx = page->nz_map->nz_idx;
  auto const nz_count      = page->nz_map->nz_count;

  auto& ni = s->nesting.nesting_info[s->setup.col.max_nesting_depth - 1];
  cg::invoke_one(block, [&] {
    ni.null_count = process_nulls
                      ? nz_map_null_count(nz_idx, nz_count, s->setup.first_row, s->setup.num_rows)
                      : 0;
  });
  block.sync();

  auto error = static_cast<kernel_error::value_type>(s->setup.error);
  if (error == 0) {
    auto const dtype_len = static_cast<uint32_t>(s->output_cvt.dtype_len);
    auto const first_row = s->setup.first_row;
    // Zero the null slots as the values are placed, like the fixed-width decoder does after
    // decoding. Writing every row of the range turns the partial-sector stores of a sparse rank
    // map into full-sector ones. On mostly-null pages the extra zero stores cost more than they
    // save, so those leave the null slots untouched.
    bool const fill_nulls = process_nulls && 4 * static_cast<int64_t>(nz_count) >= page->num_rows;
    // Resolve the common output widths at compile time.
    auto decode = [&]<uint32_t Width>() {
      nz_register_delta::rank_map_output<Width> consumer{warp,
                                                         ni.data_out,
                                                         process_nulls ? nz_idx : nullptr,
                                                         dtype_len,
                                                         first_row,
                                                         first_row + s->setup.num_rows,
                                                         fill_nulls};
      auto const result = nz_register_delta::decode_page(warp,
                                                         s->stream.data_start,
                                                         s->stream.data_end,
                                                         static_cast<uint32_t>(nz_count),
                                                         storage,
                                                         consumer);
      if (result == 0) { consumer.fill_tail(static_cast<uint32_t>(nz_count)); }
      return result;
    };
    switch (dtype_len) {
      case 8: error = decode.template operator()<8>(); break;
      case 4: error = decode.template operator()<4>(); break;
      default: error = decode.template operator()<0>(); break;
    }
  }
  if (error != 0) {
    cg::invoke_one(warp, [&] { set_error(error, error_code); });
  }
}

/**
 * @brief Decodes flat DELTA_BINARY_PACKED pages, like `decode_delta_binary_kernel`, but places
 * values using the nz map's rank map instead of decoding definition levels itself.
 *
 * One warp decodes each page with `nz_register_delta::decode_page`. The launch bounds ask for
 * only one resident block per SM, so the compiler does not limit registers to raise occupancy.
 *
 * @tparam NoNulls Whether to decode the pages `takes_no_null_path` accepts, or the remaining mapped
 * pages
 */
template <bool NoNulls>
CUDF_KERNEL void __launch_bounds__(decode_delta_binary_with_nz_map_block_size, 1)
  decode_delta_binary_kernel_with_nz_map(PageInfo* pages,
                                         device_span<ColumnChunkDesc const> chunks,
                                         size_t min_row,
                                         size_t num_rows,
                                         cudf::device_span<bool const> page_mask,
                                         kernel_error::pointer error_code)
{
  __shared__ nz_register_delta::shared_storage decode_storage;

  auto const warp     = cg::tiled_partition<cudf::detail::warp_size>(cg::this_thread_block());
  auto const page_idx = cg::this_grid().block_rank();

  if (page_mask.size() > 0 and not page_mask[page_idx]) { return; }
  auto& page = pages[page_idx];
  // Pages with a map are decoded here; the level-decoding kernel takes the rest.
  if (!page.has_nz_map(nz_map_kind::DELTA_FLAT)) { return; }
  auto const& chunk = chunks[page.chunk_idx];
  if (takes_no_null_path(page, chunk) != NoNulls) { return; }

  if constexpr (NoNulls) {
    decode_required_page(warp, page, chunk, min_row, num_rows, decode_storage, error_code);
  } else {
    decode_mapped_page(warp, &page, chunks, min_row, num_rows, decode_storage, error_code);
  }
}

// Decode page data that is DELTA_BYTE_ARRAY packed. This encoding consists of a DELTA_BINARY_PACKED
// array of prefix lengths, followed by a DELTA_BINARY_PACKED array of suffix lengths, followed by
// the suffixes (technically the suffixes are DELTA_LENGTH_BYTE_ARRAY encoded). The latter two can
// be used to create an offsets array for the suffix data, but then this needs to be combined with
// the prefix lengths to do the final decode for each value. Because the lengths of the prefixes and
// suffixes are not encoded in the header, we're going to have to first do a quick pass through them
// to find the start/end of each structure.
template <typename level_t>
CUDF_KERNEL void __launch_bounds__(decode_block_size)
  decode_delta_byte_array_kernel(PageInfo* pages,
                                 device_span<ColumnChunkDesc const> chunks,
                                 size_t min_row,
                                 size_t num_rows,
                                 cudf::device_span<bool const> page_mask,
                                 cudf::device_span<size_t> initial_str_offsets,
                                 kernel_error::pointer error_code)
{
  __shared__ __align__(16) delta_byte_array_decoder db_state;
  __shared__ __align__(16) full_page_decode_state state_g;
  __shared__ __align__(16) page_state_buffers_s<delta_nz_buf_size, 1, 1> state_buffers;

  auto* const s         = &state_g;
  auto* const sb        = &state_buffers;
  int const page_idx    = cg::this_grid().block_rank();
  auto const block      = cg::this_thread_block();
  auto const warp       = cg::tiled_partition<cudf::detail::warp_size>(block);
  auto* const prefix_db = &db_state.prefixes;
  auto* const suffix_db = &db_state.suffixes;
  auto* const dba       = &db_state;
  if (page_mask.size() > 0 and not page_mask[page_idx]) { return; }
  [[maybe_unused]] null_count_back_copier _{s, static_cast<int>(block.thread_rank())};

  if (!setup_local_page_info(s,
                             &pages[page_idx],
                             chunks,
                             min_row,
                             num_rows,
                             mask_filter{decode_kernel_mask::DELTA_BYTE_ARRAY},
                             page_processing_stage::DECODE)) {
    return;
  }

  if (s->setup.col.logical_type.has_value() &&
      s->setup.col.logical_type->type == LogicalType::DECIMAL) {
    // we cannot read decimal encoded with DELTA_BYTE_ARRAY yet
    if (block.thread_rank() == 0) {
      set_error(static_cast<kernel_error::value_type>(decode_error::INVALID_DATA_TYPE), error_code);
    }
    return;
  }

  bool const has_repetition = s->setup.col.max_level[level_type::REPETITION] > 0;
  bool const process_nulls  = should_process_nulls(s);

  // Capture initial valid_map_offset before any processing that might modify it
  int const init_valid_map_offset =
    s->nesting.nesting_info[s->setup.col.max_nesting_depth - 1].valid_map_offset;

  // choose a character parallel string copy when the average string is longer than a warp
  auto const use_char_ll =
    s->setup.page.num_valids > 0 &&
    (s->setup.page.str_bytes / s->setup.page.num_valids) > cudf::detail::warp_size;

  // copying logic from decode_page_data.
  PageNestingDecodeInfo const* nesting_info_base = s->nesting.nesting_info;

  // Get the level decode buffers for this page
  PageInfo* pp       = &pages[page_idx];
  level_t* const def = !process_nulls
                         ? nullptr
                         : reinterpret_cast<level_t*>(pp->lvl_decode_buf[level_type::DEFINITION]);
  auto* const rep    = reinterpret_cast<level_t*>(pp->lvl_decode_buf[level_type::REPETITION]);

  // skipped_leaf_values will always be 0 for flat hierarchies.
  uint32_t const skipped_leaf_values = s->setup.page.skipped_leaf_values;

  if (block.thread_rank() == 0) {
    // initialize the prefixes and suffixes blocks
    dba->init(s->stream.data_start,
              s->stream.data_end,
              s->setup.page.start_val,
              s->setup.page.temp_string_buf,
              s->setup.page.temp_string_size);
  }
  block.sync();

  // Propagate malformed-header errors from either underlying DELTA_BINARY_PACKED decoder.
  if (prefix_db->error or suffix_db->error) {
    if (block.thread_rank() == 0) {
      set_error(static_cast<kernel_error::value_type>(decode_error::DELTA_PARAMS_UNSUPPORTED),
                error_code);
    }
    return;
  }

  // assert that prefix and suffix have same mini-block size
  if (prefix_db->values_per_mb != suffix_db->values_per_mb or
      prefix_db->block_size != suffix_db->block_size or
      prefix_db->value_count != suffix_db->value_count) {
    set_error(static_cast<kernel_error::value_type>(decode_error::DELTA_PARAM_MISMATCH),
              error_code);
    return;
  }

  // pointer to location to output final strings
  int const leaf_level_index = s->setup.col.max_nesting_depth - 1;
  auto strings_data          = nesting_info_base[leaf_level_index].string_out;

  // if this is a bounds page and nested, then we need to skip up front. non-nested will work
  // its way through the page.
  int string_pos = has_repetition ? s->setup.page.start_val : 0;
  auto const is_bounds_pg =
    is_bounds_page(s->setup.page, s->setup.col.start_row, min_row, num_rows, has_repetition);
  bool const is_skip_resume = is_bounds_pg and string_pos > 0;

  // Number of values produced per main-loop iteration (see decode_delta_binary_kernel for why
  // skip-resume pages must produce a single warp_size pass per iteration).
  uint32_t const batch_size =
    is_skip_resume ? cudf::detail::warp_size
                   : min(prefix_db->values_per_mb, static_cast<uint32_t>(delta_max_batch_size));
  uint32_t const passes_per_batch = batch_size / cudf::detail::warp_size;

  if (is_skip_resume) { dba->skip(use_char_ll, block, warp); }

  while (!s->setup.error && (s->progress.input_value_count < s->setup.num_input_values ||
                             s->progress.src_pos < s->progress.nz_count)) {
    uint32_t target_pos;
    uint32_t const src_pos = s->progress.src_pos;

    if (warp.meta_group_rank() < 3) {  // warp 0..2
      target_pos =
        min(src_pos + 2 * batch_size, s->progress.nz_count + s->setup.first_row + batch_size);
    } else {  // warp 3
      target_pos = min(s->progress.nz_count, src_pos + batch_size);
    }
    // this needs to be here to prevent warp 3 modifying src_pos before all threads have read it
    block.sync();

    // warp0 will decode the rep/def levels, warp1 will unpack a mini-batch of prefixes, warp 2 will
    // unpack a mini-batch of suffixes. warp3 waits one cycle for warps 0-2 to produce a batch, and
    // then stuffs values into the proper location in the output.
    if (warp.meta_group_rank() == 0) {
      // decode repetition and definition levels.
      // - update validity vectors
      // - updates offsets (for nested columns)
      // - produces non-NULL value indices in s->nz_idx for subsequent decoding
      gpuDecodeLevels<delta_nz_buf_size, level_t>(s, sb, target_pos, rep, def, warp);
    } else if (warp.meta_group_rank() == 1) {
      // warp 1
      for (uint32_t i = 0; i < passes_per_batch; i++) {
        // make lane 0's state updates from the previous pass visible to the whole warp; the
        // block-wide sync below covers the last pass of the iteration
        if (i > 0) { warp.sync(); }
        prefix_db->decode_next_pass(warp);
      }
    } else if (warp.meta_group_rank() == 2) {
      // warp 2
      for (uint32_t i = 0; i < passes_per_batch; i++) {
        if (i > 0) { warp.sync(); }
        suffix_db->decode_next_pass(warp);
      }
    } else if (warp.meta_group_rank() == 3 and src_pos < target_pos) {
      // warp 3
      int const nproc = min(batch_size, s->setup.page.end_val - string_pos);
      strings_data +=
        use_char_ll
          ? dba->calculate_string_values_cp(strings_data, string_pos, nproc, warp.thread_rank())
          : dba->calculate_string_values(strings_data, string_pos, nproc, warp.thread_rank());
      string_pos += nproc;

      // Process the mini-block using warp 3
      for (uint32_t sp = src_pos + warp.thread_rank(); sp < src_pos + batch_size;
           sp += warp.size()) {
        // the position in the output column/buffer
        int dst_pos = sb->nz_idx[rolling_index<delta_nz_buf_size>(sp)];

        // handle skip_rows here. flat hierarchies can just skip up to first_row.
        if (!has_repetition) { dst_pos -= s->setup.first_row; }

        if (dst_pos >= 0 && sp < target_pos) {
          auto const offptr =
            reinterpret_cast<size_type*>(nesting_info_base[leaf_level_index].data_out) + dst_pos;
          auto const src_idx = sp + skipped_leaf_values;
          *offptr            = prefix_db->value_at(src_idx) + suffix_db->value_at(src_idx);
        }
        warp.sync();
      }

      if (warp.thread_rank() == 0) { s->progress.src_pos = src_pos + batch_size; }
    }

    block.sync();
  }

  // Zero-fill null positions after decoding valid values
  auto const& ni = s->nesting.nesting_info[leaf_level_index];
  if (ni.valid_map != nullptr) {
    int const num_values = ni.valid_map_offset - init_valid_map_offset;
    zero_fill_null_positions_shared<decode_block_size>(s,
                                                       sizeof(size_type),
                                                       init_valid_map_offset,
                                                       num_values,
                                                       static_cast<int>(block.thread_rank()));
  }

  // For large strings, update the initial string buffer offset to be used during large string
  // column construction. Otherwise, convert string sizes to final offsets.
  if (s->setup.col.is_large_string_col) {
    // page.chunk_idx are ordered by input_col_idx and row_group_idx respectively.
    auto const chunks_per_rowgroup = initial_str_offsets.size();
    auto const input_col_idx       = pages[page_idx].chunk_idx % chunks_per_rowgroup;
    if (has_repetition) {
      compute_initial_large_strings_offset<true>(s, initial_str_offsets[input_col_idx]);
    } else {
      compute_initial_large_strings_offset<false>(s, initial_str_offsets[input_col_idx]);
    }
  } else {
    if (has_repetition) {
      convert_small_string_lengths_to_offsets<decode_block_size, true>(s);
    } else {
      convert_small_string_lengths_to_offsets<decode_block_size, false>(s);
    }
  }

  if (block.thread_rank() == 0 and s->setup.error != 0) { set_error(s->setup.error, error_code); }
}

// Decode page data that is DELTA_LENGTH_BYTE_ARRAY packed. This encoding consists of a
// DELTA_BINARY_PACKED array of string lengths, followed by the string data.
template <typename level_t>
CUDF_KERNEL void __launch_bounds__(decode_block_size)
  decode_delta_length_byte_array_kernel(PageInfo* pages,
                                        device_span<ColumnChunkDesc const> chunks,
                                        size_t min_row,
                                        size_t num_rows,
                                        cudf::device_span<bool const> page_mask,
                                        bool has_flat_nz_map,
                                        cudf::device_span<size_t> initial_str_offsets,
                                        kernel_error::pointer error_code)
{
  __shared__ __align__(16) delta_binary_decoder db_state;
  __shared__ __align__(16) full_page_decode_state state_g;
  __shared__ __align__(16) page_state_buffers_s<delta_nz_buf_size, 1, 1> state_buffers;
  __shared__ __align__(8) uint8_t const* page_string_data;
  __shared__ size_t string_offset;

  auto* const s      = &state_g;
  auto* const sb     = &state_buffers;
  int const page_idx = cg::this_grid().block_rank();
  auto const block   = cg::this_thread_block();
  auto const warp    = cg::tiled_partition<cudf::detail::warp_size>(block);
  auto* const db     = &db_state;
  if (page_mask.size() > 0 and not page_mask[page_idx]) { return; }
  // Skip pages that `decode_delta_length_byte_array_kernel_with_nz_map` decodes. Gated on the same
  // flag that launches that kernel, so the two always agree on which pages it takes.
  if (has_flat_nz_map and pages[page_idx].has_nz_map(nz_map_kind::DELTA_FLAT)) { return; }
  [[maybe_unused]] null_count_back_copier _{s, static_cast<int>(block.thread_rank())};

  auto const mask = decode_kernel_mask::DELTA_LENGTH_BA;
  if (!setup_local_page_info(s,
                             &pages[page_idx],
                             chunks,
                             min_row,
                             num_rows,
                             mask_filter{mask},
                             page_processing_stage::DECODE)) {
    return;
  }

  if (s->setup.col.logical_type.has_value() &&
      s->setup.col.logical_type->type == LogicalType::DECIMAL) {
    // we cannot read decimal encoded with DELTA_LENGTH_BYTE_ARRAY yet
    if (block.thread_rank() == 0) {
      set_error(static_cast<kernel_error::value_type>(decode_error::INVALID_DATA_TYPE), error_code);
    }
    return;
  }

  bool const has_repetition = s->setup.col.max_level[level_type::REPETITION] > 0;
  bool const process_nulls  = should_process_nulls(s);

  // Capture initial valid_map_offset before any processing that might modify it
  int const init_valid_map_offset =
    s->nesting.nesting_info[s->setup.col.max_nesting_depth - 1].valid_map_offset;

  // copying logic from gpuDecodePageData.
  PageNestingDecodeInfo const* nesting_info_base = s->nesting.nesting_info;

  // Get the level decode buffers for this page
  PageInfo* pp       = &pages[page_idx];
  level_t* const def = !process_nulls
                         ? nullptr
                         : reinterpret_cast<level_t*>(pp->lvl_decode_buf[level_type::DEFINITION]);
  auto* const rep    = reinterpret_cast<level_t*>(pp->lvl_decode_buf[level_type::REPETITION]);

  // skipped_leaf_values will always be 0 for flat hierarchies.
  uint32_t const skipped_leaf_values = s->setup.page.skipped_leaf_values;

  // initialize delta state
  if (block.thread_rank() == 0) {
    string_offset    = 0;
    page_string_data = db->find_end_of_block(s->stream.data_start, s->stream.data_end);
  }
  block.sync();

  // The decode loop below sub-batches each mini-block into warp_size-wide passes, so any mini-block
  // size is supported (see decode_next_pass).
  if (db->error) {
    if (block.thread_rank() == 0) {
      set_error(static_cast<kernel_error::value_type>(decode_error::DELTA_PARAMS_UNSUPPORTED),
                error_code);
    }
    return;
  }

  int const leaf_level_index = s->setup.col.max_nesting_depth - 1;

  // if this is a bounds page, then we need to decode up to the first mini-block
  // that has a value we need, and set string_offset to the position of the first value in the
  // string data block.
  auto const is_bounds_pg =
    is_bounds_page(s->setup.page, s->setup.col.start_row, min_row, num_rows, has_repetition);
  bool const is_skip_resume = is_bounds_pg and s->setup.page.start_val > 0;

  // Only nested pages resume the decoder mid-page; flat pages re-init it below and can keep the
  // full batch. Mid-page resumption must produce a single warp_size pass per iteration (see
  // decode_delta_binary_kernel for why).
  bool const resumes_mid_page = is_skip_resume and has_repetition;
  uint32_t const batch_size =
    resumes_mid_page ? cudf::detail::warp_size
                     : min(db->values_per_mb, static_cast<uint32_t>(delta_max_batch_size));
  uint32_t const passes_per_batch = batch_size / cudf::detail::warp_size;

  // db->init_binary_block below resets db->values_per_mb, so make sure every thread has read it
  // for batch_size above before warp 0 re-initializes the decoder
  block.sync();

  if (is_skip_resume) {
    if (warp.meta_group_rank() == 0) {
      // string_off is only valid on thread 0
      auto const string_off = db->skip_values_and_sum(s->setup.page.start_val, warp);
      // Threads in the warp might diverge and read in skip_values_and_sum
      // after lane 0 reinits below.
      warp.sync();
      if (warp.thread_rank() == 0) {
        string_offset = string_off;

        // if there is no repetition, then we need to work through the whole page, so reset the
        // delta decoder to the beginning of the page
        if (not has_repetition) { db->init_binary_block(s->stream.data_start, s->stream.data_end); }
      }
    }
    block.sync();
  }

  int string_pos = has_repetition ? s->setup.page.start_val : 0;

  while (!s->setup.error && (s->progress.input_value_count < s->setup.num_input_values ||
                             s->progress.src_pos < s->progress.nz_count)) {
    uint32_t target_pos;
    uint32_t const src_pos = s->progress.src_pos;

    if (warp.meta_group_rank() < 2) {  // warp0..1
      target_pos = min(src_pos + 2 * batch_size, s->progress.nz_count + batch_size);
    } else {  // warp2
      target_pos = min(s->progress.nz_count, src_pos + batch_size);
    }
    // this needs to be here to prevent warp 2 modifying src_pos before all threads have read it
    __syncthreads();

    // warp0 will decode the rep/def levels, warp1 will unpack a mini-batch of deltas.
    // warp2 waits one cycle for warps 0/1 to produce a batch, and then stuffs string sizes
    // into the proper location in the output. warp 3 does nothing until it's time to copy
    // string data.
    if (warp.meta_group_rank() == 0) {
      // warp 0
      // decode repetition and definition levels.
      // - update validity vectors
      // - updates offsets (for nested columns)
      // - produces non-NULL value indices in s->nz_idx for subsequent decoding
      gpuDecodeLevels<delta_nz_buf_size, level_t>(s, sb, target_pos, rep, def, warp);
    } else if (warp.meta_group_rank() == 1) {
      // warp 1
      for (uint32_t i = 0; i < passes_per_batch; i++) {
        // make lane 0's state updates from the previous pass visible to the whole warp; the
        // block-wide sync below covers the last pass of the iteration
        if (i > 0) { warp.sync(); }
        db->decode_next_pass(warp);
      }
    } else if (warp.meta_group_rank() == 2 && src_pos < target_pos) {
      // warp 2
      int const nproc = min(batch_size, s->setup.page.end_val - string_pos);
      string_pos += nproc;

      // process the mini-block in batches of 32
      for (uint32_t sp = src_pos + warp.thread_rank(); sp < src_pos + batch_size;
           sp += warp.size()) {
        // the position in the output column/buffer
        int dst_pos = sb->nz_idx[rolling_index<delta_nz_buf_size>(sp)];

        // handle skip_rows here. flat hierarchies can just skip up to first_row.
        if (!has_repetition) { dst_pos -= s->setup.first_row; }

        // fill in offsets array
        if (dst_pos >= 0 && sp < target_pos) {
          auto const offptr =
            reinterpret_cast<size_type*>(nesting_info_base[leaf_level_index].data_out) + dst_pos;
          *offptr = db->value_at(sp + skipped_leaf_values);
        }
        warp.sync();
      }

      if (warp.thread_rank() == 0) { s->progress.src_pos = src_pos + batch_size; }
    }
    block.sync();
  }

  // Zero-fill null positions after decoding valid values
  auto const& ni = nesting_info_base[leaf_level_index];
  if (ni.valid_map != nullptr) {
    int const num_values = ni.valid_map_offset - init_valid_map_offset;
    zero_fill_null_positions_shared<decode_block_size>(s,
                                                       sizeof(size_type),
                                                       init_valid_map_offset,
                                                       num_values,
                                                       static_cast<int>(block.thread_rank()));
  }

  // For large strings, update the initial string buffer offset to be used during large string
  // column construction. Otherwise, convert string sizes to final offsets.
  if (s->setup.col.is_large_string_col) {
    // page.chunk_idx are ordered by input_col_idx and row_group_idx respectively.
    auto const chunks_per_rowgroup = initial_str_offsets.size();
    auto const input_col_idx       = pages[page_idx].chunk_idx % chunks_per_rowgroup;
    if (has_repetition) {
      compute_initial_large_strings_offset<true>(s, initial_str_offsets[input_col_idx]);
    } else {
      compute_initial_large_strings_offset<false>(s, initial_str_offsets[input_col_idx]);
    }
  } else {
    // convert string sizes to offsets
    if (has_repetition) {
      convert_small_string_lengths_to_offsets<decode_block_size, true>(s);
    } else {
      convert_small_string_lengths_to_offsets<decode_block_size, false>(s);
    }
  }

  // finally, copy the string data into place
  auto const dst = nesting_info_base[leaf_level_index].string_out;
  auto const src = page_string_data + string_offset;
  memcpy_block<decode_block_size, true>(dst, src, s->setup.page.str_bytes, block);

  if (block.thread_rank() == 0 and s->setup.error != 0) { set_error(s->setup.error, error_code); }
}

// Decodes flat DELTA_LENGTH_BYTE_ARRAY pages like `decode_delta_length_byte_array_kernel`, but
// places values using the nz map instead of decoding definition levels.
CUDF_KERNEL void __launch_bounds__(decode_block_size)
  decode_delta_length_byte_array_kernel_with_nz_map(PageInfo* pages,
                                                    device_span<ColumnChunkDesc const> chunks,
                                                    size_t min_row,
                                                    size_t num_rows,
                                                    cudf::device_span<bool const> page_mask,
                                                    cudf::device_span<size_t> initial_str_offsets,
                                                    kernel_error::pointer error_code)
{
  __shared__ __align__(16) delta_binary_decoder db_state;
  __shared__ __align__(16) full_page_decode_state state_g;
  __shared__ __align__(8) uint8_t const* page_string_data;
  __shared__ size_t string_offset;

  auto* const s      = &state_g;
  int const page_idx = cg::this_grid().block_rank();
  auto const block   = cg::this_thread_block();
  auto const warp    = cg::tiled_partition<cudf::detail::warp_size>(block);
  auto* const db     = &db_state;
  if (page_mask.size() > 0 and not page_mask[page_idx]) { return; }
  // Pages with a map are decoded here; the level-decoding kernel takes the rest.
  if (not pages[page_idx].has_nz_map(nz_map_kind::DELTA_FLAT)) { return; }

  [[maybe_unused]] null_count_back_copier _{s, static_cast<int>(block.thread_rank())};

  if (!setup_local_page_info(s,
                             &pages[page_idx],
                             chunks,
                             min_row,
                             num_rows,
                             mask_filter{decode_kernel_mask::DELTA_LENGTH_BA},
                             page_processing_stage::DECODE)) {
    return;
  }
  PageInfo* const pp = &pages[page_idx];

  if (s->setup.col.logical_type.has_value() &&
      s->setup.col.logical_type->type == LogicalType::DECIMAL) {
    cg::invoke_one(block, [&] {
      set_error(static_cast<kernel_error::value_type>(decode_error::INVALID_DATA_TYPE), error_code);
    });
    return;
  }

  bool const process_nulls = should_process_nulls(s);
  auto const* const nz_idx = pp->nz_map->nz_idx;
  int const nz_count_in    = pp->nz_map->nz_count;

  auto& ni                        = s->nesting.nesting_info[s->setup.col.max_nesting_depth - 1];
  int const init_valid_map_offset = ni.valid_map_offset;

  cg::invoke_one(block, [&] {
    ni.null_count =
      process_nulls ? nz_map_null_count(nz_idx, nz_count_in, s->setup.first_row, s->setup.num_rows)
                    : 0;
    s->progress.nz_count = nz_count_in;
    string_offset        = 0;
    page_string_data     = db->find_end_of_block(s->stream.data_start, s->stream.data_end);
  });
  block.sync();

  if (db->error) {
    cg::invoke_one(block, [&] {
      set_error(static_cast<kernel_error::value_type>(decode_error::DELTA_PARAMS_UNSUPPORTED),
                error_code);
    });
    return;
  }

  // Reject the page if it needs more lengths than the stream declares. The decoder's cursor moves a
  // mini-block at a time and can pass the end of a short stream, so the writer below would read
  // buffer slots this page never wrote. `nz_count` counts from the start of the page, so it is the
  // number of lengths the read needs.
  if (static_cast<uleb128_t>(nz_count_in) > db->value_count) {
    cg::invoke_one(block, [&] {
      set_error(static_cast<kernel_error::value_type>(decode_error::DATA_STREAM_OVERRUN),
                error_code);
    });
    return;
  }

  auto const is_bounds_pg =
    is_bounds_page(s->setup.page, s->setup.col.start_row, min_row, num_rows, false);
  bool const is_skip_resume = is_bounds_pg && s->setup.page.start_val > 0;
  if (is_skip_resume) {
    // Sync so that every thread has read db->error above before warp 0 resets it in
    // init_binary_block below.
    block.sync();
    if (warp.meta_group_rank() == 0) {
      auto const string_off = db->skip_values_and_sum(s->setup.page.start_val, warp);
      warp.sync();
      if (warp.thread_rank() == 0) {
        string_offset = string_off;
        // Restart the length stream at index 0. The skip above was only needed for the starting
        // byte offset; a flat page still decodes every length from the start of the page.
        db->init_binary_block(s->stream.data_start, s->stream.data_end);
      }
    }
    block.sync();
  }

  // Computed after the resume above, which re-initializes the decoder.
  uint32_t const batch_size = min(db->values_per_mb, static_cast<uint32_t>(delta_max_batch_size));
  uint32_t const passes_per_batch = batch_size / cudf::detail::warp_size;

  {
    // A producer/writer pipeline: warp 0 decodes a batch of lengths into the decoder's rolling
    // buffer while warp 1 writes the batch before it. `next_pass_start_idx()` is the first index
    // the producer has not decoded yet; every index below it is still resident because the rolling
    // buffer holds two batches, the one in flight and the one trailing. Reading it only at the
    // barrier keeps the concurrent region free of the decoder's mutable scalars. The other warps
    // only join the barriers here; they take part in the copy below.
    //
    // `offptr` and `first_row` are read into locals once, before the loop, rather than from `ni`
    // and `s->setup` inside it. The compiler cannot rule out the store through `offptr` aliasing
    // that shared state, so it would otherwise reload both for every value.
    auto* const offptr   = reinterpret_cast<size_type*>(ni.data_out);
    auto const first_row = s->setup.first_row;
    uint32_t produced    = db->next_pass_start_idx();
    while (s->setup.error == 0 && s->progress.src_pos < s->progress.nz_count) {
      uint32_t const src_pos = s->progress.src_pos;
      // Capped at one batch past `src_pos`: the rolling buffer holds two batches, so a writer more
      // than one batch behind would read slots the producer has overwritten.
      uint32_t const target_pos =
        min(static_cast<uint32_t>(s->progress.nz_count), min(produced, src_pos + batch_size));
      block.sync();

      if (warp.meta_group_rank() == 0) {
        for (uint32_t i = 0; i < passes_per_batch; ++i) {
          if (i > 0) { warp.sync(); }
          db->decode_next_pass(warp);
        }
      } else if (warp.meta_group_rank() == 1 && src_pos < target_pos) {
        for (uint32_t sp = src_pos + warp.thread_rank(); sp < target_pos; sp += warp.size()) {
          auto dst_pos =
            process_nulls ? static_cast<int32_t>(nz_idx[sp]) : static_cast<int32_t>(sp);
          dst_pos -= first_row;
          if (dst_pos >= 0) { offptr[dst_pos] = db->value_at(sp); }
        }
        if (warp.thread_rank() == 0) { s->progress.src_pos = target_pos; }
      }
      block.sync();

      // Stop once neither warp advances, which only a malformed page can cause.
      uint32_t const prev_produced = produced;
      produced                     = db->next_pass_start_idx();
      if (produced == prev_produced && s->progress.src_pos <= src_pos) { break; }
    }

    // Report DATA_STREAM_OVERRUN and return before the offsets scan if the loop stopped short:
    // those rows are marked valid but their lengths were never written. Every thread reaches the
    // same decision from state that is read-only after the loop, so no barrier is needed.
    if (s->setup.error != 0 or s->progress.src_pos < s->progress.nz_count) {
      cg::invoke_one(block, [&] {
        set_error(s->setup.error != 0
                    ? s->setup.error
                    : static_cast<kernel_error::value_type>(decode_error::DATA_STREAM_OVERRUN),
                  error_code);
      });
      return;
    }
  }

  // Set the counts that `gpuDecodeLevels` sets in the level-decoding kernel. The offsets scan below
  // reads `ni.value_count`.
  cg::invoke_one(block, [&] {
    auto const page_rows =
      min(s->setup.page.num_input_values, s->setup.first_row + s->setup.num_rows);
    ni.value_count                = page_rows;
    s->progress.input_value_count = page_rows;
    s->progress.input_row_count   = page_rows;
  });
  block.sync();

  if (ni.valid_map != nullptr) {
    zero_fill_null_positions_shared<decode_block_size>(s,
                                                       sizeof(size_type),
                                                       init_valid_map_offset,
                                                       s->setup.num_rows,
                                                       static_cast<int>(block.thread_rank()));
  }
  if (s->setup.col.is_large_string_col) {
    auto const chunks_per_rowgroup = initial_str_offsets.size();
    auto const input_col_idx       = pp->chunk_idx % chunks_per_rowgroup;
    compute_initial_large_strings_offset<false>(s, initial_str_offsets[input_col_idx]);
  } else {
    convert_small_string_lengths_to_offsets<decode_block_size, false>(s);
  }
  // TODO: Copy 16 bytes per thread in `memcpy_block` to cut its barriers by four. The Parquet
  // writer shares it, so do that in a separate PR.
  auto const dst = ni.string_out;
  auto const src = page_string_data + string_offset;
  memcpy_block<decode_block_size, true>(dst, src, s->setup.page.str_bytes, block);
}

}  // anonymous namespace

/**
 * @copydoc cudf::io::parquet::detail::decode_delta_binary
 */
void decode_delta_binary(cudf::detail::hostdevice_span<PageInfo> pages,
                         cudf::detail::hostdevice_span<ColumnChunkDesc const> chunks,
                         size_t num_rows,
                         size_t min_row,
                         int level_type_size,
                         cudf::device_span<bool const> page_mask,
                         bool has_flat_nz_map,
                         bool needs_level_decoder,
                         kernel_error::pointer error_code,
                         cuda::stream_ref stream,
                         cuda::stream_ref consumer_stream)
{
  CUDF_EXPECTS(pages.size() > 0, "There is no page to decode");

  dim3 dim_block(decode_delta_binary_block_size, 1);
  dim3 dim_grid(pages.size(), 1);  // 1 threadblock per page
  dim3 dim_block_nz_map(decode_delta_binary_with_nz_map_block_size, 1);

  // `consumer_stream` is ordered after the producer by the caller. The level-decoding kernel below
  // needs no such ordering because it reads only fields set on the host.
  if (has_flat_nz_map) {
    auto const has_mapped_page = [&](bool no_null_path) {
      return std::any_of(pages.host_begin(), pages.host_end(), [&](PageInfo const& page) {
        return page.has_nz_map(nz_map_kind::DELTA_FLAT) &&
               takes_no_null_path(page, chunks[page.chunk_idx]) == no_null_path;
      });
    };
    auto launch = [&]<bool NoNulls>() {
      decode_delta_binary_kernel_with_nz_map<NoNulls>
        <<<dim_grid, dim_block_nz_map, 0, consumer_stream.get()>>>(
          pages.device_ptr(), chunks, min_row, num_rows, page_mask, error_code);
      CUDF_CUDA_TRY(cudaGetLastError());
    };
    if (has_mapped_page(true)) { launch.template operator()<true>(); }
    if (has_mapped_page(false)) { launch.template operator()<false>(); }
  }

  // Skip the launch when every DELTA_BINARY page has a map, since every block would just return.
  if (not needs_level_decoder) { return; }

  if (level_type_size == 1) {
    decode_delta_binary_kernel<uint8_t><<<dim_grid, dim_block, 0, stream.get()>>>(
      pages.device_ptr(), chunks, min_row, num_rows, page_mask, has_flat_nz_map, error_code);
    CUDF_CUDA_TRY(cudaGetLastError());
  } else {
    decode_delta_binary_kernel<uint16_t><<<dim_grid, dim_block, 0, stream.get()>>>(
      pages.device_ptr(), chunks, min_row, num_rows, page_mask, has_flat_nz_map, error_code);
    CUDF_CUDA_TRY(cudaGetLastError());
  }
}

/**
 * @copydoc cudf::io::parquet::gpu::decode_delta_byte_array
 */
void decode_delta_byte_array(cudf::detail::hostdevice_span<PageInfo> pages,
                             cudf::detail::hostdevice_span<ColumnChunkDesc const> chunks,
                             size_t num_rows,
                             size_t min_row,
                             int level_type_size,
                             cudf::device_span<bool const> page_mask,
                             cudf::device_span<size_t> initial_str_offsets,
                             kernel_error::pointer error_code,
                             cuda::stream_ref stream)
{
  CUDF_EXPECTS(pages.size() > 0, "There is no page to decode");

  dim3 const dim_block(decode_block_size, 1);
  dim3 const dim_grid(pages.size(), 1);  // 1 threadblock per page

  if (level_type_size == 1) {
    decode_delta_byte_array_kernel<uint8_t><<<dim_grid, dim_block, 0, stream.get()>>>(
      pages.device_ptr(), chunks, min_row, num_rows, page_mask, initial_str_offsets, error_code);
    CUDF_CUDA_TRY(cudaGetLastError());
  } else {
    decode_delta_byte_array_kernel<uint16_t><<<dim_grid, dim_block, 0, stream.get()>>>(
      pages.device_ptr(), chunks, min_row, num_rows, page_mask, initial_str_offsets, error_code);
    CUDF_CUDA_TRY(cudaGetLastError());
  }
}

/**
 * @copydoc cudf::io::parquet::gpu::decode_delta_length_byte_array
 */
void decode_delta_length_byte_array(cudf::detail::hostdevice_span<PageInfo> pages,
                                    cudf::detail::hostdevice_span<ColumnChunkDesc const> chunks,
                                    size_t num_rows,
                                    size_t min_row,
                                    int level_type_size,
                                    cudf::device_span<bool const> page_mask,
                                    bool has_flat_nz_map,
                                    bool needs_level_decoder,
                                    cudf::device_span<size_t> initial_str_offsets,
                                    kernel_error::pointer error_code,
                                    cuda::stream_ref stream,
                                    cuda::stream_ref consumer_stream)
{
  CUDF_EXPECTS(pages.size() > 0, "There is no page to decode");

  dim3 const dim_block(decode_block_size, 1);
  dim3 const dim_grid(pages.size(), 1);  // 1 threadblock per page

  static_assert(delta_rolling_buf_size >= 2 * delta_max_batch_size + 1,
                "The writer trails the producer by one batch plus the header slot");

  // `consumer_stream` is ordered after the producer by the caller. The level-decoding kernel below
  // needs no such ordering because it reads only fields set on the host.
  if (has_flat_nz_map) {
    decode_delta_length_byte_array_kernel_with_nz_map<<<dim_grid,
                                                        dim_block,
                                                        0,
                                                        consumer_stream.get()>>>(
      pages.device_ptr(), chunks, min_row, num_rows, page_mask, initial_str_offsets, error_code);
    CUDF_CUDA_TRY(cudaGetLastError());
  }

  // Skip the launch when every DELTA_LENGTH_BYTE_ARRAY page has a map, since every block would just
  // return.
  if (not needs_level_decoder) { return; }

  if (level_type_size == 1) {
    decode_delta_length_byte_array_kernel<uint8_t>
      <<<dim_grid, dim_block, 0, stream.get()>>>(pages.device_ptr(),
                                                 chunks,
                                                 min_row,
                                                 num_rows,
                                                 page_mask,
                                                 has_flat_nz_map,
                                                 initial_str_offsets,
                                                 error_code);
    CUDF_CUDA_TRY(cudaGetLastError());
  } else {
    decode_delta_length_byte_array_kernel<uint16_t>
      <<<dim_grid, dim_block, 0, stream.get()>>>(pages.device_ptr(),
                                                 chunks,
                                                 min_row,
                                                 num_rows,
                                                 page_mask,
                                                 has_flat_nz_map,
                                                 initial_str_offsets,
                                                 error_code);
    CUDF_CUDA_TRY(cudaGetLastError());
  }
}

}  // namespace cudf::io::parquet::detail
