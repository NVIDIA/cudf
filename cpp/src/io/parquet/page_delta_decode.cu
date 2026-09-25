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

#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/transform_scan.h>

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

// The prepass consumer does not decode levels, so the warp the legacy kernel gives that job to has
// no work in it. Sizing the block to the warps it actually uses stops it paying registers for a
// warp that never runs, which is what bounds occupancy for these kernels.
constexpr int decode_delta_binary_prepass_block_size     = 64;  // producer + writer
constexpr int decode_delta_byte_array_prepass_block_size = 96;  // prefix + suffix + writer

// The flat prepass publishes a sorted valid-rank -> input-position map for the whole page. A
// sliced page still needs the number of valid values preceding `first_row`, to report its null
// count to the legacy output-state plumbing.
__device__ int flat_prepass_valid_count_before(uint32_t const* nz_idx, int count, int input_row)
{
  int first = 0;
  int last  = count;
  while (first < last) {
    int const mid = first + (last - first) / 2;
    if (nz_idx[mid] < static_cast<uint32_t>(input_row)) {
      first = mid + 1;
    } else {
      last = mid;
    }
  }
  return first;
}

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
    // the temp allocation holds one leading string slot (see the string-size prepass) followed by
    // delta_max_batch_size scratch slots. reserve the leading slot for the last decoded string so
    // it stays clear of the scratch, which each round overwrites from its start.
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
  decode_delta_binary_kernel_legacy(PageInfo* pages,
                                    device_span<ColumnChunkDesc const> chunks,
                                    size_t min_row,
                                    size_t num_rows,
                                    cudf::device_span<bool const> page_mask,
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

  if (block.thread_rank() == 0 and s->setup.error != 0) { set_error(s->setup.error, error_code); }
}

// Flat DELTA_BINARY_PACKED consumer for the opt-in page-global level prepass.
//
// This is deliberately a second kernel rather than a branch inside the legacy one above: while the
// feature is opt-in, the legacy decoder is the rollback path and must stay independent of it.
template <typename level_t, bool use_list_prepass_t = false>
CUDF_KERNEL void __launch_bounds__(decode_delta_binary_prepass_block_size)
  decode_delta_binary_kernel_prepass(PageInfo* pages,
                                     device_span<ColumnChunkDesc const> chunks,
                                     size_t min_row,
                                     size_t num_rows,
                                     cudf::device_span<bool const> page_mask,
                                     kernel_error::pointer error_code)
{
  __shared__ __align__(16) delta_binary_decoder db_state;
  __shared__ __align__(16) full_page_decode_state state_g;

  auto* const s      = &state_g;
  int const page_idx = cg::this_grid().block_rank();
  auto const block   = cg::this_thread_block();
  auto const warp    = cg::tiled_partition<cudf::detail::warp_size>(block);
  auto* const db     = &db_state;

  if (page_mask.size() > 0 and not page_mask[page_idx]) { return; }
  // Selection is recorded entirely by the host, as a non-null `prepass_state` on the claimed
  // family. `filter_delta_legacy_pages` below applies the complement of exactly this test, so
  // every page is decoded by one kernel and no page by both.
  if constexpr (use_list_prepass_t) {
    if (!pages[page_idx].prepass_is(level_prepass_family::DELTA_LIST)) { return; }
  } else {
    if (!pages[page_idx].prepass_is(level_prepass_family::DELTA_FLAT)) { return; }
  }

  [[maybe_unused]] null_count_back_copier _{s, static_cast<int>(block.thread_rank())};
  if (!setup_local_page_info(s,
                             &pages[page_idx],
                             chunks,
                             min_row,
                             num_rows,
                             mask_filter{decode_kernel_mask::DELTA_BINARY},
                             page_processing_stage::DECODE)) {
    return;
  }

  // The flat instantiation indexes the map by output position, so a page that somehow carried
  // repetition levels would place values wrongly; refuse it rather than trust the classifier.
  bool const has_repetition = s->setup.col.max_level[level_type::REPETITION] > 0;
  if (!use_list_prepass_t && has_repetition) { return; }

  bool const process_nulls         = should_process_nulls(s);
  auto* const pp                   = &pages[page_idx];
  auto const* const prepass_nz_idx = pp->prepass_state->nz_idx;
  int const prepass_nz_count       = pp->prepass_state->nz_count;
  // Negative means the producer never ran for this page; a nullable page with no map likewise has
  // nothing to consume. Either way, leave the page to the legacy kernel.
  if (prepass_nz_count < 0 || (process_nulls && prepass_nz_idx == nullptr)) { return; }

  auto& ni = s->nesting.nesting_info[s->setup.col.max_nesting_depth - 1];
  if (block.thread_rank() == 0) {
    // A list page's null counts are per depth and are restored from the prepass state in the
    // epilogue instead.
    if constexpr (!use_list_prepass_t) {
      auto const prefix_valid_count =
        process_nulls
          ? flat_prepass_valid_count_before(prepass_nz_idx, prepass_nz_count, s->setup.first_row)
          : static_cast<int>(s->setup.first_row);
      ni.null_count =
        process_nulls ? s->setup.num_rows - (prepass_nz_count - prefix_valid_count) : 0;
    }
    s->progress.input_value_count = s->setup.num_input_values;
    s->progress.nz_count          = prepass_nz_count;
  }
  block.sync();

  if (block.thread_rank() == 0) { db->init_binary_block(s->stream.data_start, s->stream.data_end); }
  block.sync();
  if (db->error) {
    if (block.thread_rank() == 0) {
      set_error(static_cast<kernel_error::value_type>(decode_error::DELTA_PARAMS_UNSUPPORTED),
                error_code);
    }
    return;
  }

  uint32_t const skipped_leaf_values = s->setup.page.skipped_leaf_values;
  bool const is_skip_resume          = skipped_leaf_values > 0;
  uint32_t const batch_size =
    is_skip_resume ? cudf::detail::warp_size
                   : min(db->values_per_mb, static_cast<uint32_t>(delta_max_batch_size));
  uint32_t const passes_per_batch = batch_size / cudf::detail::warp_size;
  if (is_skip_resume) { db->skip_values(skipped_leaf_values, block, warp); }

  // Two warps, two jobs: warp 0 is the delta producer and warp 1 the value writer. The legacy
  // kernel's third warp decoded levels, which the prepass has already done, so this block is one
  // warp narrower than its legacy counterpart rather than carrying an idle one.
  //
  // The writer trails the producer by one batch, which is what lets the two run concurrently in a
  // single phase rather than in two phases separated by a barrier. The legacy kernel gets the same
  // lag for free: its `nz_count` starts at zero and is grown by warp 0's level decode, so its
  // writer fails `src_pos < target_pos` on the first iteration and idles through it. The prepass
  // publishes the final `nz_count` up front, so the lag has to be stated explicitly instead.
  //
  // `next_pass_start_idx()` is the producer's watermark: every stream index below it has been
  // decoded, and is still resident because `delta_rolling_buf_size` holds two batches, the one in
  // flight plus the one trailing. Reading it only at the barrier keeps the concurrent region free
  // of any access to the decoder's mutable scalars -- `value_at()` touches `value[]` and
  // `first_value`, neither of which the producer writes below the watermark.
  uint32_t produced = db->next_pass_start_idx();
  while (s->setup.error == 0 && s->progress.src_pos < s->progress.nz_count) {
    uint32_t const src_pos = s->progress.src_pos;
    // Decoder indices run ahead of output positions by `skipped_leaf_values` on a resume page.
    // Capped at one batch so the writer trails the producer by exactly one, never more: the
    // rolling buffer holds two batches, so a writer that fell two behind would read values the
    // producer had already overwritten.
    uint32_t const target_pos =
      min(s->progress.nz_count,
          min(produced > skipped_leaf_values ? produced - skipped_leaf_values : 0,
              src_pos + batch_size));
    block.sync();

    if (warp.meta_group_rank() == 0) {
      for (uint32_t i = 0; i < passes_per_batch; ++i) {
        if (i > 0) { warp.sync(); }
        db->decode_next_pass(warp);
      }
    } else if (warp.meta_group_rank() == 1 && src_pos < target_pos) {
      for (uint32_t sp = src_pos + warp.thread_rank(); sp < target_pos; sp += warp.size()) {
        auto dst_pos =
          process_nulls ? static_cast<int32_t>(prepass_nz_idx[sp]) : static_cast<int32_t>(sp);
        // A list page's map already holds absolute output positions.
        if (!has_repetition) { dst_pos -= s->setup.first_row; }
        if (dst_pos >= 0) {
          void* const dst = ni.data_out + dst_pos * s->output_cvt.dtype_len;
          auto const val  = db->value_at(sp + skipped_leaf_values);
          switch (s->output_cvt.dtype_len) {
            case 1: *static_cast<int8_t*>(dst) = val; break;
            case 2: *static_cast<int16_t*>(dst) = val; break;
            case 4: *static_cast<int32_t*>(dst) = val; break;
            case 8: *static_cast<int64_t*>(dst) = val; break;
          }
        }
      }
      if (warp.thread_rank() == 0) { s->progress.src_pos = target_pos; }
    }
    block.sync();

    // Gating the writer on the producer's watermark means the loop's exit condition is no longer
    // reached by construction: a malformed page whose delta stream holds fewer values than its
    // levels imply leaves `nz_count` permanently above anything the producer can publish. The old
    // unconditional `src_pos + batch_size` target ran off the end of the stream and wrote garbage,
    // but it did terminate. Stop only once *neither* side can advance.
    //
    // Both halves of that are required. "The writer did not advance" is true on the first
    // iteration by construction -- that is the lag -- so pairing it with a static exhaustion test
    // like `produced >= value_count` breaks out before the writer has run at all on any page whose
    // values fit in a single batch.
    uint32_t const prev_produced = produced;
    produced                     = db->next_pass_start_idx();
    if (produced == prev_produced && s->progress.src_pos <= src_pos) { break; }
  }

  // The loop's barrier is inside its body, so a straggler may still be evaluating the loop
  // condition -- which reads s->progress.nz_count -- when a later write lands on that field.
  block.sync();

  if constexpr (use_list_prepass_t) {
    // The list walker's per-depth counters are what the output-column plumbing reads after this
    // kernel; the prepass computed them, so restore them verbatim rather than recomputing.
    if (block.thread_rank() == 0) {
      for (int depth = 0; depth < s->setup.page.nesting_info_size; ++depth) {
        auto const& source                              = pp->prepass_state->nesting[depth];
        s->nesting.nesting_info[depth].null_count       = source.null_count;
        s->nesting.nesting_info[depth].valid_map_offset = source.valid_map_offset;
        s->nesting.nesting_info[depth].valid_count      = source.valid_count;
        s->nesting.nesting_info[depth].value_count      = source.value_count;
      }
      s->progress.nz_count          = pp->prepass_state->nz_count;
      s->progress.input_value_count = pp->prepass_state->aux_count;
      s->progress.input_row_count   = pp->prepass_state->input_row_count;
    }
    block.sync();
  } else if (block.thread_rank() == 0 and s->setup.error != 0) {
    set_error(s->setup.error, error_code);
  }
}

// Build a launch mask for the unchanged legacy delta kernels. Keeping this filtering outside those
// kernels is what makes legacy mode a true rollback path: their device bodies and symbols stay
// independent of the selector.
CUDF_KERNEL void filter_delta_legacy_pages(PageInfo const* pages,
                                           cudf::device_span<bool const> page_mask,
                                           bool* legacy_page_mask,
                                           size_t num_pages)
{
  auto const page_idx = static_cast<size_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (page_idx < num_pages) {
    // Must be the exact complement of the consumer's own test: if the two disagree a page is
    // decoded twice or not at all.
    bool const claimed = (pages[page_idx].prepass_is(level_prepass_family::DELTA_FLAT) ||
                          pages[page_idx].prepass_is(level_prepass_family::DELTA_LIST)) &&
                         delta_prepass_pays_for_itself(pages[page_idx]);
    legacy_page_mask[page_idx] = (page_mask.empty() || page_mask[page_idx]) && !claimed;
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
  decode_delta_byte_array_kernel_legacy(PageInfo* pages,
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
  decode_delta_length_byte_array_kernel_legacy(PageInfo* pages,
                                               device_span<ColumnChunkDesc const> chunks,
                                               size_t min_row,
                                               size_t num_rows,
                                               cudf::device_span<bool const> page_mask,
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

// Flat DELTA_LENGTH_BYTE_ARRAY consumer for the opt-in page-global level prepass. As with the
// DELTA_BINARY_PACKED consumer, this is deliberately a second kernel: the legacy decoder above is
// the rollback path and stays independent of the selector.
template <bool use_list_prepass_t = false>
CUDF_KERNEL void __launch_bounds__(decode_block_size)
  decode_delta_length_byte_array_kernel_prepass(PageInfo* pages,
                                                device_span<ColumnChunkDesc const> chunks,
                                                size_t min_row,
                                                size_t num_rows,
                                                cudf::device_span<bool const> page_mask,
                                                cudf::device_span<size_t> initial_str_offsets,
                                                kernel_error::pointer error_code)
{
  // The block-wide value loop needs a full-width block: one warp per DELTA pass, bounded by
  // delta_max_decode_warps. Both hold for decode_block_size.
  static_assert(decode_block_size / cudf::detail::warp_size <= delta_max_decode_warps,
                "delta_max_decode_warps bounds the block-wide value loop's warp count");

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
  if constexpr (use_list_prepass_t) {
    if (!pages[page_idx].prepass_is(level_prepass_family::DELTA_LIST)) { return; }
  } else {
    if (!pages[page_idx].prepass_is(level_prepass_family::DELTA_FLAT)) { return; }
  }

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
    if (block.thread_rank() == 0) {
      set_error(static_cast<kernel_error::value_type>(decode_error::INVALID_DATA_TYPE), error_code);
    }
    return;
  }

  bool const has_repetition        = s->setup.col.max_level[level_type::REPETITION] > 0;
  bool const process_nulls         = should_process_nulls(s);
  auto const* const prepass_nz_idx = pp->prepass_state->nz_idx;
  int const prepass_nz_count       = pp->prepass_state->nz_count;
  if ((!use_list_prepass_t && has_repetition) || prepass_nz_count < 0 ||
      (process_nulls && prepass_nz_idx == nullptr)) {
    return;
  }

  int const leaf_level_index      = s->setup.col.max_nesting_depth - 1;
  int const init_valid_map_offset = s->nesting.nesting_info[leaf_level_index].valid_map_offset;
  PageNestingDecodeInfo const* nesting_info_base = s->nesting.nesting_info;
  uint32_t const skipped_leaf_values             = s->setup.page.skipped_leaf_values;

  if (block.thread_rank() == 0) {
    // A list page's null counts are per depth and are restored from the prepass state in the
    // epilogue instead.
    if constexpr (!use_list_prepass_t) {
      auto& ni = s->nesting.nesting_info[leaf_level_index];
      auto const prefix_valid =
        process_nulls
          ? flat_prepass_valid_count_before(prepass_nz_idx, prepass_nz_count, s->setup.first_row)
          : static_cast<int>(s->setup.first_row);
      ni.null_count = process_nulls ? s->setup.num_rows - (prepass_nz_count - prefix_valid) : 0;
    }
    s->progress.input_value_count = s->setup.num_input_values;
    s->progress.nz_count          = prepass_nz_count;
    string_offset                 = 0;
    page_string_data              = db->find_end_of_block(s->stream.data_start, s->stream.data_end);
  }
  block.sync();

  if (db->error) {
    if (block.thread_rank() == 0) {
      set_error(static_cast<kernel_error::value_type>(decode_error::DELTA_PARAMS_UNSUPPORTED),
                error_code);
    }
    return;
  }

  auto const is_bounds_pg =
    is_bounds_page(s->setup.page, s->setup.col.start_row, min_row, num_rows, has_repetition);
  bool const is_skip_resume   = is_bounds_pg && s->setup.page.start_val > 0;
  bool const resumes_mid_page = is_skip_resume && has_repetition;
  uint32_t const batch_size =
    resumes_mid_page ? cudf::detail::warp_size
                     : min(db->values_per_mb, static_cast<uint32_t>(delta_max_batch_size));
  uint32_t const passes_per_batch = batch_size / cudf::detail::warp_size;

  // db->init_binary_block below resets db->values_per_mb and db->error, so make sure every thread
  // has read them (for batch_size above and the error check before it) before warp 0
  // re-initializes the decoder.
  block.sync();

  if (is_skip_resume) {
    if (warp.meta_group_rank() == 0) {
      auto const string_off = db->skip_values_and_sum(s->setup.page.start_val, warp);
      warp.sync();
      if (warp.thread_rank() == 0) {
        string_offset = string_off;
        // A flat page still decodes the whole page after deriving the byte offset, whereas a
        // repeated one resumes the length stream at start_val. Reinitializing the latter would
        // make the first selected list element consume the first page length, truncating every
        // selected string after a slice.
        if (!has_repetition) { db->init_binary_block(s->stream.data_start, s->stream.data_end); }
      }
    }
    block.sync();
  }

  // The block-wide value loop needs every pass of a DELTA block addressable without a serial
  // header walk, which holds only when the block is at least as wide as the thread block. That is
  // parsed from the page's own data stream on the device, so it stays a run-time test; pages that
  // fail it fall through to the warp-fused loop below.
  //
  // The repetition half is decided by the instantiation, not at run time: this kernel's page gate
  // admits DELTA_LIST only when `use_list_prepass_t`, and `classify_prepass_family` keys `is_list`
  // off `max_level[REPETITION] > 0`. So a page reaching the flat instantiation has no repetition
  // levels by construction. Spelling that as a constexpr keeps `decode_next_passes_wide` out of
  // the list instantiation entirely rather than carrying it as dead code.
  bool const use_wide_loop =
    !use_list_prepass_t && db->block_size >= static_cast<uleb128_t>(decode_block_size);
  if constexpr (!use_list_prepass_t) {
    if (use_wide_loop) {
      constexpr int num_warps = decode_block_size / cudf::detail::warp_size;
      auto* const offptr =
        reinterpret_cast<size_type*>(nesting_info_base[leaf_level_index].data_out);
      auto const idx_end     = static_cast<uint32_t>(s->progress.nz_count);
      auto const write_value = [&](uint32_t idx, zigzag128_t val) {
        int dst_pos = process_nulls ? static_cast<int>(prepass_nz_idx[idx]) : static_cast<int>(idx);
        dst_pos -= s->setup.first_row;
        if (dst_pos >= 0) { offptr[dst_pos] = val; }
      };

      // Stream index 0 is the block header's first value and belongs to no pass.
      if (block.thread_rank() == 0 && idx_end > 0) { write_value(0, db->value_at(0)); }

      // Values are consumed in the iteration that produces them, so they never touch the rolling
      // buffer and the loop does no shared-memory traffic at all.
      zigzag128_t val{};
      uint32_t idx{}, produced{};
      while (!s->setup.error && db->next_pass_start_idx() < idx_end) {
        if (!db->decode_next_passes_wide<num_warps>(block, warp, val, idx, produced)) { break; }
        if (idx < idx_end) { write_value(idx, val); }
      }
      if (block.thread_rank() == 0) { s->progress.src_pos = idx_end; }
      block.sync();
    }
  }
  // Warp-synchronous fallback for every page the wide loop cannot take: list pages (whose stream
  // index is offset by skipped_leaf_values) and pages whose DELTA block is narrower than the
  // thread block.
  if (!use_wide_loop) {
    // Warp-synchronous fallback. Because the prepass already materialized the rank->position map
    // in global memory, the writer's position input no longer depends on any warp in this block,
    // leaving only the delta decode's own serial chain. Folding the write into the decoding warp
    // therefore removes every block barrier from the value loop; the other warps drop straight to
    // the epilogue barrier instead of round-tripping through barriers they cannot help with.
    if (warp.meta_group_rank() == 0) {
      while (!s->setup.error && s->progress.src_pos < s->progress.nz_count) {
        uint32_t const src_pos    = s->progress.src_pos;
        uint32_t const target_pos = min(s->progress.nz_count, src_pos + batch_size);

        for (uint32_t i = 0; i < passes_per_batch; ++i) {
          if (i > 0) { warp.sync(); }
          db->decode_next_pass(warp);
        }
        // decode_next_pass leaves lane 0's decoder-state updates unpublished on exit.
        warp.sync();

        for (uint32_t sp = src_pos + warp.thread_rank(); sp < target_pos; sp += warp.size()) {
          int dst_pos = process_nulls ? static_cast<int>(prepass_nz_idx[sp]) : static_cast<int>(sp);
          // A list page's map already holds absolute output positions.
          if (!has_repetition) { dst_pos -= s->setup.first_row; }
          if (dst_pos >= 0) {
            auto const offptr =
              reinterpret_cast<size_type*>(nesting_info_base[leaf_level_index].data_out) + dst_pos;
            *offptr = db->value_at(sp + skipped_leaf_values);
          }
        }
        // src_pos is read by every lane at the top of the loop, so lane 0's update has to be
        // ordered against both the reads above it and the reads of the next iteration.
        warp.sync();
        if (warp.thread_rank() == 0) { s->progress.src_pos = target_pos; }
        warp.sync();
      }
    }
    block.sync();
  }

  if constexpr (use_list_prepass_t) {
    // The list walker's per-depth counters are what the output-column plumbing reads after this
    // kernel; the prepass computed them, so restore them verbatim rather than recomputing.
    if (block.thread_rank() == 0) {
      for (int depth = 0; depth < s->setup.page.nesting_info_size; ++depth) {
        auto const& source                              = pp->prepass_state->nesting[depth];
        s->nesting.nesting_info[depth].null_count       = source.null_count;
        s->nesting.nesting_info[depth].valid_map_offset = source.valid_map_offset;
        s->nesting.nesting_info[depth].valid_count      = source.valid_count;
        s->nesting.nesting_info[depth].value_count      = source.value_count;
      }
      s->progress.nz_count          = pp->prepass_state->nz_count;
      s->progress.input_value_count = pp->prepass_state->aux_count;
      s->progress.input_row_count   = pp->prepass_state->input_row_count;
    }
    block.sync();
  } else if (block.thread_rank() == 0) {
    auto& ni = s->nesting.nesting_info[leaf_level_index];
    auto const page_rows =
      min(s->setup.page.num_input_values, s->setup.first_row + s->setup.num_rows);
    ni.valid_count                = pp->prepass_state->nz_count;
    ni.value_count                = page_rows;
    s->progress.input_value_count = page_rows;
    s->progress.input_row_count   = page_rows;
  }
  block.sync();

  auto const& ni = nesting_info_base[leaf_level_index];
  if (ni.valid_map != nullptr) {
    int const num_values =
      use_list_prepass_t ? ni.valid_map_offset - init_valid_map_offset : s->setup.num_rows;
    zero_fill_null_positions_shared<decode_block_size>(s,
                                                       sizeof(size_type),
                                                       init_valid_map_offset,
                                                       num_values,
                                                       static_cast<int>(block.thread_rank()));
  }
  if (s->setup.col.is_large_string_col) {
    auto const chunks_per_rowgroup = initial_str_offsets.size();
    auto const input_col_idx       = pages[page_idx].chunk_idx % chunks_per_rowgroup;
    compute_initial_large_strings_offset<use_list_prepass_t>(s, initial_str_offsets[input_col_idx]);
  } else {
    convert_small_string_lengths_to_offsets<decode_block_size, use_list_prepass_t>(s);
  }
  auto const dst = nesting_info_base[leaf_level_index].string_out;
  auto const src = page_string_data + string_offset;
  memcpy_block<decode_block_size, true>(dst, src, s->setup.page.str_bytes, block);
  if (block.thread_rank() == 0 and s->setup.error != 0) { set_error(s->setup.error, error_code); }
}

// Flat DELTA_BYTE_ARRAY consumer for the opt-in page-global level prepass. As with the other two
// consumers, this is deliberately a second kernel: the legacy decoder is the rollback path and
// stays independent of the selector. The page-global map replaces only the level-decoding warp;
// the prefix, suffix and writer roles keep their legacy arrangement.
template <bool use_list_prepass_t = false>
CUDF_KERNEL void __launch_bounds__(decode_delta_byte_array_prepass_block_size)
  decode_delta_byte_array_kernel_prepass(PageInfo* pages,
                                         device_span<ColumnChunkDesc const> chunks,
                                         size_t min_row,
                                         size_t num_rows,
                                         cudf::device_span<bool const> page_mask,
                                         cudf::device_span<size_t> initial_str_offsets,
                                         kernel_error::pointer error_code)
{
  __shared__ __align__(16) delta_byte_array_decoder db_state;
  __shared__ __align__(16) full_page_decode_state state_g;

  auto* const s         = &state_g;
  int const page_idx    = cg::this_grid().block_rank();
  auto const block      = cg::this_thread_block();
  auto const warp       = cg::tiled_partition<cudf::detail::warp_size>(block);
  auto* const prefix_db = &db_state.prefixes;
  auto* const suffix_db = &db_state.suffixes;
  auto* const dba       = &db_state;
  if (page_mask.size() > 0 and not page_mask[page_idx]) { return; }
  if constexpr (use_list_prepass_t) {
    if (!pages[page_idx].prepass_is(level_prepass_family::DELTA_LIST)) { return; }
  } else {
    if (!pages[page_idx].prepass_is(level_prepass_family::DELTA_FLAT)) { return; }
    // Too few nulls for the prepass to repay its map: the producer skipped this page, and
    // filter_delta_legacy_pages hands it to the legacy kernel instead. The gate is flat-only --
    // the list shapes are ahead of the legacy decoders even at 1% nulls.
    if (!delta_prepass_pays_for_itself(pages[page_idx])) { return; }
  }

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
  PageInfo* const pp = &pages[page_idx];

  if (s->setup.col.logical_type.has_value() &&
      s->setup.col.logical_type->type == LogicalType::DECIMAL) {
    if (block.thread_rank() == 0) {
      set_error(static_cast<kernel_error::value_type>(decode_error::INVALID_DATA_TYPE), error_code);
    }
    return;
  }

  bool const has_repetition        = s->setup.col.max_level[level_type::REPETITION] > 0;
  bool const process_nulls         = should_process_nulls(s);
  auto const* const prepass_nz_idx = pp->prepass_state->nz_idx;
  int const prepass_nz_count       = pp->prepass_state->nz_count;
  if ((!use_list_prepass_t && has_repetition) || prepass_nz_count < 0 ||
      (process_nulls && prepass_nz_idx == nullptr)) {
    return;
  }

  int const leaf_level_index      = s->setup.col.max_nesting_depth - 1;
  int const init_valid_map_offset = s->nesting.nesting_info[leaf_level_index].valid_map_offset;
  PageNestingDecodeInfo const* nesting_info_base = s->nesting.nesting_info;
  // A bounds page in a repeated hierarchy can resume after leaf values that precede the selected
  // rows. The delta streams retain those values in their decode position, so the legacy
  // adjustment is retained when looking up the unpacked prefix and suffix lengths.
  uint32_t const skipped_leaf_values = s->setup.page.skipped_leaf_values;
  auto const use_char_ll =
    s->setup.page.num_valids > 0 &&
    (s->setup.page.str_bytes / s->setup.page.num_valids) > cudf::detail::warp_size;

  if (block.thread_rank() == 0) {
    // A list page's null counts are per depth and are restored from the prepass state in the
    // epilogue instead.
    if constexpr (!use_list_prepass_t) {
      auto& ni = s->nesting.nesting_info[leaf_level_index];
      auto const prefix_valid =
        process_nulls
          ? flat_prepass_valid_count_before(prepass_nz_idx, prepass_nz_count, s->setup.first_row)
          : static_cast<int>(s->setup.first_row);
      ni.null_count = process_nulls ? s->setup.num_rows - (prepass_nz_count - prefix_valid) : 0;
    }
    s->progress.input_value_count = s->setup.num_input_values;
    s->progress.nz_count          = prepass_nz_count;
    dba->init(s->stream.data_start,
              s->stream.data_end,
              s->setup.page.start_val,
              s->setup.page.temp_string_buf,
              s->setup.page.temp_string_size);
  }
  block.sync();

  if (prefix_db->error or suffix_db->error) {
    if (block.thread_rank() == 0) {
      set_error(static_cast<kernel_error::value_type>(decode_error::DELTA_PARAMS_UNSUPPORTED),
                error_code);
    }
    return;
  }
  if (prefix_db->values_per_mb != suffix_db->values_per_mb ||
      prefix_db->block_size != suffix_db->block_size ||
      prefix_db->value_count != suffix_db->value_count) {
    if (block.thread_rank() == 0) {
      set_error(static_cast<kernel_error::value_type>(decode_error::DELTA_PARAM_MISMATCH),
                error_code);
    }
    return;
  }

  auto strings_data = nesting_info_base[leaf_level_index].string_out;
  int string_pos    = has_repetition ? s->setup.page.start_val : 0;
  auto const is_bounds_pg =
    is_bounds_page(s->setup.page, s->setup.col.start_row, min_row, num_rows, has_repetition);
  bool const is_skip_resume = is_bounds_pg && string_pos > 0;
  uint32_t const batch_size =
    is_skip_resume ? cudf::detail::warp_size
                   : min(prefix_db->values_per_mb, static_cast<uint32_t>(delta_max_batch_size));
  uint32_t const passes_per_batch = batch_size / cudf::detail::warp_size;

  if (is_skip_resume) { dba->skip(use_char_ll, block, warp); }

  // The writer trails the two unpackers by one batch so that all three warps run in a single
  // concurrent phase. See decode_delta_binary_kernel_prepass for why the lag has to be explicit
  // here when the legacy kernel gets it for free from a `nz_count` that grows as levels decode.
  // Both streams must have published a value before the writer may read it, so the watermark is
  // the smaller of the two.
  uint32_t produced = min(prefix_db->next_pass_start_idx(), suffix_db->next_pass_start_idx());
  while (!s->setup.error && s->progress.src_pos < s->progress.nz_count) {
    uint32_t const src_pos = s->progress.src_pos;
    // The one-batch cap is load-bearing twice over: it keeps the writer from falling two batches
    // behind, which would read values the producer had already wrapped over in the rolling buffer,
    // and it keeps `nproc` within the `values_per_mb` limit that calculate_string_values() applies
    // internally -- advancing `string_pos` past what that call actually converted would silently
    // drop strings.
    // Decoder indices run ahead of output positions by `skipped_leaf_values` on a resumed list
    // page, so the watermark is compared in decoder space.
    uint32_t const target_pos =
      min(s->progress.nz_count,
          min(produced > skipped_leaf_values ? produced - skipped_leaf_values : 0,
              src_pos + batch_size));
    block.sync();

    // Three warps, three jobs: the prepass owns level decoding, so there is no fourth. The roles
    // match skip()'s, which already unpacks prefixes on warp 0 and suffixes on warp 1.
    if (warp.meta_group_rank() == 0) {
      for (uint32_t i = 0; i < passes_per_batch; ++i) {
        if (i > 0) { warp.sync(); }
        prefix_db->decode_next_pass(warp);
      }
    } else if (warp.meta_group_rank() == 1) {
      for (uint32_t i = 0; i < passes_per_batch; ++i) {
        if (i > 0) { warp.sync(); }
        suffix_db->decode_next_pass(warp);
      }
    } else if (warp.meta_group_rank() == 2 && src_pos < target_pos) {
      int const nproc =
        min(static_cast<int>(target_pos - src_pos), s->setup.page.end_val - string_pos);
      strings_data +=
        use_char_ll
          ? dba->calculate_string_values_cp(strings_data, string_pos, nproc, warp.thread_rank())
          : dba->calculate_string_values(strings_data, string_pos, nproc, warp.thread_rank());
      string_pos += nproc;

      // The trip count must stay lane-uniform because of the `warp.sync()` below, so the loop
      // bound is the batch and `target_pos` gates the body -- `target_pos` is no longer a multiple
      // of the warp size ahead of `src_pos` now that the producers' watermark clamps it.
      for (uint32_t sp = src_pos + warp.thread_rank(); sp < src_pos + batch_size;
           sp += warp.size()) {
        if (sp < target_pos) {
          int dst_pos = process_nulls ? static_cast<int>(prepass_nz_idx[sp]) : static_cast<int>(sp);
          // A list page's map already holds absolute output positions.
          if (!has_repetition) { dst_pos -= s->setup.first_row; }
          if (dst_pos >= 0) {
            auto const offptr =
              reinterpret_cast<size_type*>(nesting_info_base[leaf_level_index].data_out) + dst_pos;
            auto const src_idx = sp + skipped_leaf_values;
            *offptr            = prefix_db->value_at(src_idx) + suffix_db->value_at(src_idx);
          }
        }
        warp.sync();
      }
      if (warp.thread_rank() == 0) { s->progress.src_pos = target_pos; }
    }
    block.sync();

    // Liveness: gating on the producers' watermark means a malformed page whose delta streams hold
    // fewer values than its levels imply can no longer reach the loop's exit condition. See
    // decode_delta_binary_kernel_prepass for why the producer's watermark must be compared against
    // its own previous value rather than against a static end-of-stream test.
    uint32_t const prev_produced = produced;
    produced = min(prefix_db->next_pass_start_idx(), suffix_db->next_pass_start_idx());
    if (produced == prev_produced && s->progress.src_pos <= src_pos) { break; }
  }

  // The loop's barrier is inside its body, so a straggler may still be evaluating the loop
  // condition -- which reads s->progress.nz_count -- when thread 0 overwrites that field below.
  block.sync();

  if constexpr (use_list_prepass_t) {
    // The list walker's per-depth counters are what the output-column plumbing reads after this
    // kernel; the prepass computed them, so restore them verbatim rather than recomputing.
    if (block.thread_rank() == 0) {
      for (int depth = 0; depth < s->setup.page.nesting_info_size; ++depth) {
        auto const& source                              = pp->prepass_state->nesting[depth];
        s->nesting.nesting_info[depth].null_count       = source.null_count;
        s->nesting.nesting_info[depth].valid_map_offset = source.valid_map_offset;
        s->nesting.nesting_info[depth].valid_count      = source.valid_count;
        s->nesting.nesting_info[depth].value_count      = source.value_count;
      }
      s->progress.nz_count          = pp->prepass_state->nz_count;
      s->progress.input_value_count = pp->prepass_state->aux_count;
      s->progress.input_row_count   = pp->prepass_state->input_row_count;
    }
    block.sync();
  } else if (block.thread_rank() == 0) {
    auto& ni = s->nesting.nesting_info[leaf_level_index];
    auto const page_rows =
      min(s->setup.page.num_input_values, s->setup.first_row + s->setup.num_rows);
    ni.valid_count                = pp->prepass_state->nz_count;
    ni.value_count                = page_rows;
    s->progress.input_value_count = page_rows;
    s->progress.input_row_count   = page_rows;
  }
  block.sync();

  auto const& ni = s->nesting.nesting_info[leaf_level_index];
  if (ni.valid_map != nullptr) {
    int const num_values =
      use_list_prepass_t ? ni.valid_map_offset - init_valid_map_offset : s->setup.num_rows;
    zero_fill_null_positions_shared<decode_delta_byte_array_prepass_block_size>(
      s,
      sizeof(size_type),
      init_valid_map_offset,
      num_values,
      static_cast<int>(block.thread_rank()));
  }
  if (s->setup.col.is_large_string_col) {
    auto const chunks_per_rowgroup = initial_str_offsets.size();
    auto const input_col_idx       = pages[page_idx].chunk_idx % chunks_per_rowgroup;
    compute_initial_large_strings_offset<use_list_prepass_t>(s, initial_str_offsets[input_col_idx]);
  } else {
    convert_small_string_lengths_to_offsets<decode_delta_byte_array_prepass_block_size,
                                            use_list_prepass_t>(s);
  }
  if (block.thread_rank() == 0 and s->setup.error != 0) { set_error(s->setup.error, error_code); }
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
                         kernel_error::pointer error_code,
                         cuda::stream_ref stream,
                         bool use_flat_prepass,
                         bool use_list_prepass)
{
  CUDF_EXPECTS(pages.size() > 0, "There is no page to decode");

  dim3 dim_block(decode_delta_binary_block_size, 1);
  dim3 dim_grid(pages.size(), 1);  // 1 threadblock per page
  // The prepass consumer drops the legacy kernel's level-decoding warp rather than idling it.
  dim3 dim_block_prepass(decode_delta_binary_prepass_block_size, 1);

  auto const use_prepass = use_flat_prepass || use_list_prepass;
  rmm::device_uvector<bool> legacy_page_mask(use_prepass ? pages.size() : 0, stream);
  auto launch_prepass = [&]<bool list_t>() {
    if (level_type_size == 1) {
      decode_delta_binary_kernel_prepass<uint8_t, list_t>
        <<<dim_grid, dim_block_prepass, 0, stream.get()>>>(
          pages.device_ptr(), chunks, min_row, num_rows, page_mask, error_code);
    } else {
      decode_delta_binary_kernel_prepass<uint16_t, list_t>
        <<<dim_grid, dim_block_prepass, 0, stream.get()>>>(
          pages.device_ptr(), chunks, min_row, num_rows, page_mask, error_code);
    }
    CUDF_CUDA_TRY(cudaGetLastError());
  };
  if (use_flat_prepass) { launch_prepass.template operator()<false>(); }
  if (use_list_prepass) { launch_prepass.template operator()<true>(); }
  if (use_prepass) {
    // Hand the legacy kernel everything the consumers above did not claim.
    constexpr int filter_block_size = 256;
    auto const filter_grid =
      cudf::util::div_rounding_up_safe<size_t>(pages.size(), filter_block_size);
    filter_delta_legacy_pages<<<filter_grid, filter_block_size, 0, stream.get()>>>(
      pages.device_ptr(), page_mask, legacy_page_mask.data(), pages.size());
    CUDF_CUDA_TRY(cudaGetLastError());
    page_mask = cudf::device_span<bool const>{legacy_page_mask.data(), legacy_page_mask.size()};
  }

  if (level_type_size == 1) {
    decode_delta_binary_kernel_legacy<uint8_t><<<dim_grid, dim_block, 0, stream.get()>>>(
      pages.device_ptr(), chunks, min_row, num_rows, page_mask, error_code);
    CUDF_CUDA_TRY(cudaGetLastError());
  } else {
    decode_delta_binary_kernel_legacy<uint16_t><<<dim_grid, dim_block, 0, stream.get()>>>(
      pages.device_ptr(), chunks, min_row, num_rows, page_mask, error_code);
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
                             cuda::stream_ref stream,
                             bool use_flat_prepass,
                             bool use_list_prepass)
{
  CUDF_EXPECTS(pages.size() > 0, "There is no page to decode");

  dim3 const dim_block(decode_block_size, 1);
  dim3 const dim_grid(pages.size(), 1);  // 1 threadblock per page
  // The prepass consumer runs one warp narrower than the legacy kernel below: it has no level
  // decoding, so it has no use for its fourth warp. That is what pays for the extra register the
  // writer lag costs -- at 96 threads it fits 9 blocks per SM with every warp working, against 8
  // blocks with one idle warp each.
  dim3 const dim_block_prepass(decode_delta_byte_array_prepass_block_size, 1);

  auto const use_prepass = use_flat_prepass || use_list_prepass;
  rmm::device_uvector<bool> legacy_page_mask(use_prepass ? pages.size() : 0, stream);
  auto launch_prepass = [&]<bool list_t>() {
    decode_delta_byte_array_kernel_prepass<list_t>
      <<<dim_grid, dim_block_prepass, 0, stream.get()>>>(
        pages.device_ptr(), chunks, min_row, num_rows, page_mask, initial_str_offsets, error_code);
    CUDF_CUDA_TRY(cudaGetLastError());
  };
  if (use_flat_prepass) { launch_prepass.template operator()<false>(); }
  if (use_list_prepass) { launch_prepass.template operator()<true>(); }
  if (use_prepass) {
    // Hand the legacy kernel everything the consumers above did not claim.
    constexpr int filter_block_size = 256;
    auto const filter_grid =
      cudf::util::div_rounding_up_safe<size_t>(pages.size(), filter_block_size);
    filter_delta_legacy_pages<<<filter_grid, filter_block_size, 0, stream.get()>>>(
      pages.device_ptr(), page_mask, legacy_page_mask.data(), pages.size());
    CUDF_CUDA_TRY(cudaGetLastError());
    page_mask = cudf::device_span<bool const>{legacy_page_mask.data(), legacy_page_mask.size()};
  }

  if (level_type_size == 1) {
    decode_delta_byte_array_kernel_legacy<uint8_t><<<dim_grid, dim_block, 0, stream.get()>>>(
      pages.device_ptr(), chunks, min_row, num_rows, page_mask, initial_str_offsets, error_code);
    CUDF_CUDA_TRY(cudaGetLastError());
  } else {
    decode_delta_byte_array_kernel_legacy<uint16_t><<<dim_grid, dim_block, 0, stream.get()>>>(
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
                                    cudf::device_span<size_t> initial_str_offsets,
                                    kernel_error::pointer error_code,
                                    cuda::stream_ref stream,
                                    bool use_flat_prepass,
                                    bool use_list_prepass)
{
  CUDF_EXPECTS(pages.size() > 0, "There is no page to decode");

  dim3 const dim_block(decode_block_size, 1);
  dim3 const dim_grid(pages.size(), 1);  // 1 threadblock per page

  auto const use_prepass = use_flat_prepass || use_list_prepass;
  rmm::device_uvector<bool> legacy_page_mask(use_prepass ? pages.size() : 0, stream);
  auto launch_prepass = [&]<bool list_t>() {
    decode_delta_length_byte_array_kernel_prepass<list_t><<<dim_grid, dim_block, 0, stream.get()>>>(
      pages.device_ptr(), chunks, min_row, num_rows, page_mask, initial_str_offsets, error_code);
    CUDF_CUDA_TRY(cudaGetLastError());
  };
  if (use_flat_prepass) { launch_prepass.template operator()<false>(); }
  if (use_list_prepass) { launch_prepass.template operator()<true>(); }
  if (use_prepass) {
    // Hand the legacy kernel everything the consumers above did not claim.
    constexpr int filter_block_size = 256;
    auto const filter_grid =
      cudf::util::div_rounding_up_safe<size_t>(pages.size(), filter_block_size);
    filter_delta_legacy_pages<<<filter_grid, filter_block_size, 0, stream.get()>>>(
      pages.device_ptr(), page_mask, legacy_page_mask.data(), pages.size());
    CUDF_CUDA_TRY(cudaGetLastError());
    page_mask = cudf::device_span<bool const>{legacy_page_mask.data(), legacy_page_mask.size()};
  }

  if (level_type_size == 1) {
    decode_delta_length_byte_array_kernel_legacy<uint8_t><<<dim_grid, dim_block, 0, stream.get()>>>(
      pages.device_ptr(), chunks, min_row, num_rows, page_mask, initial_str_offsets, error_code);
    CUDF_CUDA_TRY(cudaGetLastError());
  } else {
    decode_delta_length_byte_array_kernel_legacy<uint16_t>
      <<<dim_grid, dim_block, 0, stream.get()>>>(
        pages.device_ptr(), chunks, min_row, num_rows, page_mask, initial_str_offsets, error_code);
    CUDF_CUDA_TRY(cudaGetLastError());
  }
}

}  // namespace cudf::io::parquet::detail
