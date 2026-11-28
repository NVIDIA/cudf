/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "decimal_decode_utils.cuh"
#include "delta_binary.cuh"
#include "io/utilities/block_utils.cuh"
#include "page_state_composed.cuh"
#include "page_string_utils.cuh"
#include "parquet_gpu.hpp"

#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/detail/utilities/integer_utils.hpp>

#include <cuda/atomic>
#include <cuda/std/algorithm.lower_bound.h>
#include <cuda/std/bit>
#include <cuda/std/limits>

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

// One producer warp and one writer warp
constexpr int decode_delta_binary_with_nz_map_block_size = 64;
// Two warps decode the prefix and suffix lengths and six build strings. More string-building warps
// gain little because each batch's strings are built only after the previous batch's.
constexpr int decode_delta_byte_array_with_nz_map_block_size = 256;
// Blocks of the DELTA_BYTE_ARRAY consumer per SM, the most that fit in the 1024 threads of an
// sm_75 SM. Uncapped, the kernel's registers leave room for only two, which made large reads with
// few nulls about 20% slower than the level-decoding kernel.
constexpr int decode_delta_byte_array_with_nz_map_min_blocks = 4;

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

/**
 * @brief Store a DELTA_BINARY_PACKED decoded value at the output width
 *
 * Deltas wrap around at the physical width, so decimals are first narrowed to their signed physical
 * type, then sign extended or truncated to the output decimal width.
 *
 * @param dst Pointer to the output value
 * @param val Decoded value
 * @param dtype_len Size in bytes of the output type
 * @param dtype_len_in Size in bytes of the physical type
 * @param is_decimal Whether the column is a decimal
 */
__device__ inline void store_delta_binary_value(
  void* dst, zigzag128_t val, uint32_t dtype_len, int32_t dtype_len_in, bool is_decimal)
{
  if (is_decimal) {
    auto const value = dtype_len_in == sizeof(int32_t)
                         ? static_cast<int64_t>(static_cast<int32_t>(val))
                         : static_cast<int64_t>(val);
    store_decimal(dst, dtype_len, [value]<typename T>() { return static_cast<T>(value); });
    return;
  }
  switch (dtype_len) {
    case 1: *static_cast<int8_t*>(dst) = val; break;
    case 2: *static_cast<int16_t*>(dst) = val; break;
    case 4: *static_cast<int32_t*>(dst) = val; break;
    case 8: *static_cast<int64_t*>(dst) = val; break;
  }
}

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
  bool const is_decimal     = s->setup.col.logical_type.has_value() and
                          s->setup.col.logical_type->type == LogicalType::DECIMAL;

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
          void* const dst = nesting_info_base[leaf_level_index].data_out +
                            static_cast<size_t>(dst_pos) * s->output_cvt.dtype_len;
          store_delta_binary_value(dst,
                                   db->value_at(sp + skipped_leaf_values),
                                   s->output_cvt.dtype_len,
                                   s->output_cvt.dtype_len_in,
                                   is_decimal);
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

// Decodes flat DELTA_BINARY_PACKED pages, like `decode_delta_binary_kernel`, but places values
// using the nz map's rank map instead of decoding definition levels itself.
CUDF_KERNEL void __launch_bounds__(decode_delta_binary_with_nz_map_block_size)
  decode_delta_binary_kernel_with_nz_map(PageInfo* pages,
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
  // Pages with a map are decoded here; the level-decoding kernel takes the rest.
  if (not pages[page_idx].has_nz_map(nz_map_kind::DELTA_FLAT)) { return; }

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

  bool const process_nulls = should_process_nulls(s);
  auto* const pp           = &pages[page_idx];
  auto const* const nz_idx = pp->nz_map->nz_idx;
  int const nz_count_in    = pp->nz_map->nz_count;

  auto& ni = s->nesting.nesting_info[s->setup.col.max_nesting_depth - 1];
  cg::invoke_one(block, [&] {
    ni.null_count =
      process_nulls ? nz_map_null_count(nz_idx, nz_count_in, s->setup.first_row, s->setup.num_rows)
                    : 0;
    s->progress.nz_count = nz_count_in;
  });
  block.sync();

  cg::invoke_one(block, [&] { db->init_binary_block(s->stream.data_start, s->stream.data_end); });
  block.sync();
  if (db->error) {
    cg::invoke_one(block, [&] {
      set_error(static_cast<kernel_error::value_type>(decode_error::DELTA_PARAMS_UNSUPPORTED),
                error_code);
    });
    return;
  }

  // Reject the page if it needs more values than the stream declares. The decoder's cursor moves a
  // mini-block at a time and can pass the end of a short stream, so the writer below would read
  // buffer slots this page never wrote. `nz_count` counts from the start of the page, so it is the
  // number of values the read needs.
  if (static_cast<uleb128_t>(nz_count_in) > db->value_count) {
    cg::invoke_one(block, [&] {
      set_error(static_cast<kernel_error::value_type>(decode_error::DATA_STREAM_OVERRUN),
                error_code);
    });
    return;
  }

  // `setup_local_page_info` zeroes `skipped_leaf_values` for a page without repetition, and the
  // classifier selects only such pages, so there is no resume to handle here. The decode index is
  // the rank itself: `nz_count` counts from page position 0, so it already spans any prefix below
  // `first_row`.
  uint32_t const batch_size = min(db->values_per_mb, static_cast<uint32_t>(delta_max_batch_size));
  uint32_t const passes_per_batch = batch_size / cudf::detail::warp_size;

  // This is a two-warp version of delta binary decoding using a delta producer warp and a value
  // writer warp. The writer trails the producer by one batch to allow the two to run concurrently
  // in a single phase. `next_pass_start_idx()` is the first stream index the producer has not
  // decoded yet. Every index below it is decoded, and is still resident because
  // `delta_rolling_buf_size` holds two batches, the one in flight plus the one trailing. Reading
  // `next_pass_start_idx()` only at the barrier keeps the concurrent region free of any access to
  // the decoder's mutable scalars. `value_at()` touches `value[]` and `first_value`, and the
  // producer writes neither below that index.
  //
  // `data_out`, the dtype lengths and `first_row` are read into locals once, before the loop,
  // rather than from `ni`, `s->output_cvt` and `s->setup` inside it. The compiler cannot rule out
  // the store through `data_out` aliasing that shared state, so it would otherwise reload them for
  // every value.
  auto* const data_out    = ni.data_out;
  auto const dtype_len    = s->output_cvt.dtype_len;
  auto const dtype_len_in = s->output_cvt.dtype_len_in;
  auto const is_decimal   = s->setup.col.logical_type.has_value() and
                          s->setup.col.logical_type->type == LogicalType::DECIMAL;
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
        auto dst_pos = process_nulls ? static_cast<int32_t>(nz_idx[sp]) : static_cast<int32_t>(sp);
        dst_pos -= first_row;
        if (dst_pos >= 0) {
          void* const dst = data_out + static_cast<size_t>(dst_pos) * dtype_len;
          store_delta_binary_value(dst, db->value_at(sp), dtype_len, dtype_len_in, is_decimal);
        }
      }
      if (warp.thread_rank() == 0) { s->progress.src_pos = target_pos; }
    }
    block.sync();

    // Stop once neither warp advances: a malformed page whose delta stream is shorter than its
    // definition levels imply never reaches `nz_count`.
    //
    // Both halves are needed: the producer finishes first and the writer still has a batch to
    // drain, and on the first iteration the writer has not run at all.
    uint32_t const prev_produced = produced;
    produced                     = db->next_pass_start_idx();
    if (produced == prev_produced && s->progress.src_pos <= src_pos) { break; }
  }

  // The loop's barrier is inside its body, so a straggler may still be evaluating the loop
  // condition -- which reads s->progress.nz_count -- when a later write lands on that field.
  block.sync();

  // Report DELTA_PARAMS_UNSUPPORTED if `check_mini_block` rejected a mini-block mid-decode.
  // Otherwise, values still outstanding means the loop stopped above rather than finishing: the
  // delta stream held fewer values than the definition levels call for. Those rows are already
  // marked valid, so without an error the read would return whatever their output slots happened
  // to contain.
  //
  // The error is reported by the same thread that updates `s->setup.error`, so no other thread
  // reads the field while it is being written.
  if (block.thread_rank() == 0) {
    if (s->setup.error == 0 and db->error) {
      s->setup.error |=
        static_cast<kernel_error::value_type>(decode_error::DELTA_PARAMS_UNSUPPORTED);
    } else if (s->setup.error == 0 and s->progress.src_pos < s->progress.nz_count) {
      s->setup.error |= static_cast<kernel_error::value_type>(decode_error::DATA_STREAM_OVERRUN);
    }
    if (s->setup.error != 0) { set_error(s->setup.error, error_code); }
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
                                 bool has_flat_nz_map,
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
  // Skip pages that `decode_delta_byte_array_kernel_with_nz_map` decodes. Gated on the same flag
  // that launches that kernel, so the two always agree on which pages it takes.
  if (has_flat_nz_map and pages[page_idx].has_nz_map(nz_map_kind::DELTA_FLAT)) { return; }
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

// Decodes flat DELTA_BYTE_ARRAY pages like `decode_delta_byte_array_kernel`, but places values
// using the nz map instead of decoding definition levels. Warps 0 and 1 decode the prefix and
// suffix lengths of batch b + 1 into one half of a double buffer while the other six warps build
// the strings of batch b from the other half.
//
// This kernel also avoids some of the decoder's functionality like `calculate_string_values`,
// `string_scan` or `skip` that implicitly encode a dependency between groups of strings that leads
// to unnecessary serialization. With the nz map the stalls from the warps impacted by that
// serialization are a bigger limiting factor, so this kernel takes a different approach. Byte j of
// string i's prefix is byte j of string i - 1. Following that back, it is byte `j - p_k` of the
// suffix of the latest string k <= i with prefix length `p_k <= j`, or byte j of the last string of
// the previous batch if no string in the batch qualifies. With this approach each string can be
// built from the suffix data alone, independently of the other strings in its batch. With this
// change, within a batch no string waits for another, and only the previous batch's last string is
// carried over. A skipped batch builds only that string, not all of its strings as `skip` does.
CUDF_KERNEL void __launch_bounds__(decode_delta_byte_array_with_nz_map_block_size,
                                   decode_delta_byte_array_with_nz_map_min_blocks)
  decode_delta_byte_array_kernel_with_nz_map(PageInfo* pages,
                                             device_span<ColumnChunkDesc const> chunks,
                                             size_t min_row,
                                             size_t num_rows,
                                             cudf::device_span<bool const> page_mask,
                                             cudf::device_span<size_t> initial_str_offsets,
                                             kernel_error::pointer error_code)
{
  using cudf::detail::warp_size;
  constexpr int block_size        = decode_delta_byte_array_with_nz_map_block_size;
  constexpr int num_length_warps  = 2;  // one decodes prefix lengths, one suffix lengths
  constexpr int num_builders      = block_size - num_length_warps * warp_size;
  constexpr int num_builder_warps = num_builders / warp_size;
  // Strings each builder thread scans per batch. 3 keeps shared memory usage low so GPUs with less
  // of it per SM fit more blocks. On H100 larger values were not meaningfully faster.
  constexpr int strings_per_builder = 3;
  constexpr int batch_size          = num_builders * strings_per_builder;
  // Rounds of pointer jumping per batch: enough for pointers that double every round to cross the
  // batch. `build` follows whatever pointers remain.
  constexpr int max_jump_rounds = cuda::std::bit_width(static_cast<unsigned>(batch_size)) + 1;
  // Bits of `batch_flags`. They share one `int` rather than being separate `bool`s so that one warp
  // reduction and one `fetch_or` per warp combine both.
  constexpr int needs_jump   = 1;
  constexpr int is_malformed = 2;

  __shared__ __align__(16) delta_byte_array_decoder db_state;
  __shared__ __align__(16) full_page_decode_state state_g;
  __shared__ int prefix_lens[2][batch_size];         // double buffered
  __shared__ int suffix_lens[2][batch_size];         // double buffered
  __shared__ int64_t string_offsets[batch_size];     // output offset within the batch
  __shared__ int suffix_offsets[batch_size];         // suffix data offset within the batch
  __shared__ int shorter_prefix_idx[2][batch_size];  // see the pointer jumping below
  __shared__ int64_t warp_string_bytes[num_builder_warps];
  __shared__ int64_t warp_suffix_bytes[num_builder_warps];
  // Relaxed is enough for `batch_flags`: the builder barriers order its accesses.
  __shared__ cuda::atomic<int, cuda::thread_scope_block> batch_flags;
  // Set with `atomicOr` rather than being a `cuda::atomic<int, cuda::thread_scope_block>`. libcu++
  // implements `fetch_or` as inline PTX on a generic address, which compiles to the generic `ATOM`
  // instead of the shared memory `ATOMS` (an `__isShared` assume does not change that).
  // `batch_flags` pays that once per warp per batch, which measured neutral; these flags pay it
  // once per warp per pointer-jumping round, which measured 0.2-0.5% slower (H100, CTK 12.9).
  // TODO: Make these a `cuda::atomic` too once builds use CTK 13.5 or newer, where CCCL's
  // `__nv_atomic` backend (NVIDIA/cccl#11075) should compile `fetch_or` to `ATOMS`.
  __shared__ int jump_changed[3];
  // State the builders carry from one batch to the next, and the page's rank map. They live in
  // shared memory and are read where they are used, as are the page constants in `state_g`:
  // holding them all in registers made the kernel spill under its register cap.
  struct builder_state {
    uint8_t* string_out;
    uint8_t const* suffix_data;
    uint8_t const* prev_str;
    int prev_len;
  };
  __shared__ builder_state bs;
  __shared__ uint32_t const* nz_idx_sh;

  auto* const s         = &state_g;
  int const page_idx    = cg::this_grid().block_rank();
  auto const block      = cg::this_thread_block();
  auto const warp       = cg::tiled_partition<warp_size>(block);
  auto* const prefix_db = &db_state.prefixes;
  auto* const suffix_db = &db_state.suffixes;
  auto* const dba       = &db_state;
  if (page_mask.size() > 0 and not page_mask[page_idx]) { return; }
  // Decode only the pages with a map. `decode_delta_byte_array_kernel` skips them.
  if (not pages[page_idx].has_nz_map(nz_map_kind::DELTA_FLAT)) { return; }
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

  bool const process_nulls = should_process_nulls(s);

  // Capture initial valid_map_offset before any processing that might modify it
  int const init_valid_map_offset =
    s->nesting.nesting_info[s->setup.col.max_nesting_depth - 1].valid_map_offset;

  // choose a character parallel string copy when the average string is longer than a warp
  auto const use_char_ll = s->setup.page.num_valids > 0 &&
                           (s->setup.page.str_bytes / s->setup.page.num_valids) > warp_size;

  // Get the nz map for this page, which takes the place of its level decode buffers
  PageInfo* pp             = &pages[page_idx];
  auto const* const nz_idx = pp->nz_map->nz_idx;
  int const nz_count_in    = pp->nz_map->nz_count;
  auto& ni                 = s->nesting.nesting_info[s->setup.col.max_nesting_depth - 1];

  if (block.thread_rank() == 0) {
    // initialize the prefixes and suffixes blocks
    dba->init(s->stream.data_start,
              s->stream.data_end,
              s->setup.page.start_val,
              s->setup.page.temp_string_buf,
              s->setup.page.temp_string_size);
    // the null count and progress that the level decoder would set
    ni.null_count =
      process_nulls ? nz_map_null_count(nz_idx, nz_count_in, s->setup.first_row, s->setup.num_rows)
                    : 0;
    s->progress.nz_count = nz_count_in;
    bs                   = {ni.string_out, dba->suffix_char_data, nullptr, 0};
    nz_idx_sh            = nz_idx;
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
    if (block.thread_rank() == 0) {
      set_error(static_cast<kernel_error::value_type>(decode_error::DELTA_PARAM_MISMATCH),
                error_code);
    }
    return;
  }

  // Reject the page if it needs more values than the streams declare. `nz_count` counts from the
  // start of the page, so it is the number of values the read needs.
  if (static_cast<uleb128_t>(nz_count_in) > prefix_db->value_count) {
    if (block.thread_rank() == 0) {
      set_error(static_cast<kernel_error::value_type>(decode_error::DATA_STREAM_OVERRUN),
                error_code);
    }
    return;
  }

  // Writes strings [start_val, end_val) to the output. Earlier strings are only copied from, so
  // they are not written, except the last string of a batch that ends before `start_val` when a
  // later batch has output strings. It goes to the page's `temp_string_buf`, which a page with
  // `start_val > 0` has, alternating between two slots so that building batch b's last string never
  // overwrites batch b - 1's while it is being read. Each slot fits any string but the page's
  // first, which is never stored: a batch ending with it is the page's only batch.
  int const lane        = warp.thread_rank();
  int const wid         = warp.meta_group_rank();
  int const num_batches = cudf::detail::div_rounding_up_safe(nz_count_in, batch_size);

  // Synchronizes the string-building warps only, on barrier 1 (`__syncthreads` uses 0), so the
  // length-decoding warps never wait on it. Do not replace it with a `cuda::barrier`, which made
  // reads of long strings about 1.3% slower on H100: it compiles to a shared memory arrive by every
  // thread and a polling wait (`SYNCS.ARRIVE`, then a `SYNCS.PHASECHK` loop), where this is a
  // single `BAR.SYNC` per warp.
  // TODO: Use CCCL's `interwarp_synchronizer`, which wraps `__barrier_sync_count`, once it moves
  // from cudax into libcu++ (NVIDIA/cccl#11211).
  auto const builder_sync = [] { __barrier_sync_count(1, num_builders); };

  enum class length_status { ok, stream_ended, malformed };
  // Copies lengths [first, last) of `db` into `dst`, decoding passes as needed. Called by one warp.
  // A length is malformed if it is negative or does not fit in an `int`. It is called below once
  // per decoder rather than inlined with `db` chosen by warp: with a fixed decoder each call
  // addresses its decoder's shared state directly, and choosing it at run time quadrupled spills.
  auto const decode_lengths = [](cg::thread_block_tile<warp_size, cg::thread_block> const& warp,
                                 int lane,
                                 delta_binary_decoder& db,
                                 int* dst,
                                 int first,
                                 int last) -> length_status {
    int copied = first;
    while (true) {
      // Value 0 comes from the header, so it is available before the first pass. The decoder's
      // position stays a `uint32_t` until it is clamped to `last`, since it can round up past
      // INT32_MAX at the end of a stream. Indices are relative to `first` so none can overflow.
      auto const avail =
        max(db.next_pass_start_idx(), db.value_count > 0 ? uint32_t{1} : uint32_t{0});
      int const upto = static_cast<int>(min(avail, static_cast<uint32_t>(last)));
      bool bad       = false;
      for (int i = copied - first + lane; i < upto - first; i += warp_size) {
        auto const v = db.value_at(first + i);
        bad          = bad or v < 0 or v > cuda::std::numeric_limits<int>::max();
        dst[i]       = static_cast<int>(v);
      }
      if (warp.any(bad)) { return length_status::malformed; }
      copied = max(copied, upto);
      if (copied >= last) { return length_status::ok; }
      warp.sync();
      db.decode_next_pass(warp);
      warp.sync();
      if (db.error) { return length_status::malformed; }
      if (db.next_pass_start_idx() <= avail) { return length_status::stream_ended; }
    }
  };

  // Iteration `it` decodes the lengths of batch `it` and builds batch `it - 1`, so there is one
  // more iteration than batches.
  for (int it = 0; it <= num_batches; ++it) {
    bool stream_ended = false;  // this thread's length decode ran out of values
    bool malformed    = false;  // this thread found a malformed length or string
    if (wid < num_length_warps) {
      if (it < num_batches) {
        int const first = it * batch_size;
        int const last  = first + min(batch_size, nz_count_in - first);
        auto const status =
          wid == 0 ? decode_lengths(warp, lane, *prefix_db, prefix_lens[it % 2], first, last)
                   : decode_lengths(warp, lane, *suffix_db, suffix_lens[it % 2], first, last);
        stream_ended = status == length_status::stream_ended;
        malformed    = status == length_status::malformed;
      }
    } else if (it > 0) {
      int const batch_idx    = it - 1;
      int const batch_start  = batch_idx * batch_size;
      int const n            = min(batch_size, nz_count_in - batch_start);
      int const* const p     = prefix_lens[batch_idx % 2];
      int const* const l     = suffix_lens[batch_idx % 2];
      int const builder_rank = static_cast<int>(block.thread_rank()) - num_length_warps * warp_size;
      int const builder_warp = wid - num_length_warps;

      // Computes each string's output offset and suffix data offset within the batch. Each builder
      // owns `strings_per_builder` consecutive strings, whose sums are scanned within the warp and
      // then across the builder warps. Only strings in [start_val, end_val) take output space. The
      // sums are 64-bit because one batch can decode to more than 2 GiB even from a small page: a
      // string that repeats the previous one is stored as a prefix length with no new bytes, so a
      // string of a few MB followed by a batch of copies of it decodes to over 2 GiB.
      int64_t string_bytes[strings_per_builder];
      int64_t suffix_bytes[strings_per_builder];
      int64_t thread_string_bytes = 0;
      int64_t thread_suffix_bytes = 0;
      for (int i = 0; i < strings_per_builder; ++i) {
        int const idx = builder_rank * strings_per_builder + i;
        bool const in = idx < n;
        // Padding slots past the end of the last batch get no rank: near INT32_MAX values,
        // `batch_start + idx` would overflow for them.
        int const sp      = in ? batch_start + idx : 0;
        bool const is_out = in and sp >= s->setup.page.start_val and sp < s->setup.page.end_val;
        string_bytes[i]   = is_out ? int64_t{p[idx]} + l[idx] : 0;
        suffix_bytes[i]   = in ? l[idx] : 0;
        thread_string_bytes += string_bytes[i];
        thread_suffix_bytes += suffix_bytes[i];
      }
      auto string_off = cg::exclusive_scan(warp, thread_string_bytes, cg::plus<int64_t>{});
      auto suffix_off = cg::exclusive_scan(warp, thread_suffix_bytes, cg::plus<int64_t>{});
      if (lane == warp_size - 1) {
        warp_string_bytes[builder_warp] = string_off + thread_string_bytes;
        warp_suffix_bytes[builder_warp] = suffix_off + thread_suffix_bytes;
      }
      if (builder_rank == 0) { batch_flags.store(0, cuda::std::memory_order_relaxed); }
      builder_sync();
      for (int w = 0; w < builder_warp; ++w) {
        string_off += warp_string_bytes[w];
        suffix_off += warp_suffix_bytes[w];
      }

      // Checks every string before any is built, since building a string reads the strings its
      // prefix comes from. A prefix longer than the string before it, or a suffix past the end of
      // the page, is malformed.
      auto const suffix_bytes_left = static_cast<int64_t>(s->stream.data_end - bs.suffix_data);
      int flags                    = 0;
      for (int i = 0; i < strings_per_builder; ++i) {
        int const idx = builder_rank * strings_per_builder + i;
        if (idx < n) {
          string_offsets[idx]  = string_off;
          suffix_offsets[idx]  = static_cast<int>(suffix_off);
          auto const prev_size = idx > 0 ? int64_t{p[idx - 1]} + l[idx - 1] : int64_t{bs.prev_len};
          if (p[idx] > prev_size or suffix_off + l[idx] > suffix_bytes_left) {
            flags |= is_malformed;
          }
          if (idx > 0 and p[idx] > 0 and p[idx - 1] >= p[idx]) { flags |= needs_jump; }
        }
        string_off += string_bytes[i];
        suffix_off += suffix_bytes[i];
      }
      flags = cg::reduce(warp, flags, cg::bit_or<int>{});
      if (lane == 0 and flags != 0) {
        batch_flags.fetch_or(flags, cuda::std::memory_order_relaxed);
      }
      for (int idx = builder_rank; idx < n; idx += num_builders) {
        shorter_prefix_idx[0][idx] = idx - 1;
      }
      if (builder_rank < 3) { jump_changed[builder_rank] = 0; }
      builder_sync();
      int const all_flags = batch_flags.load(cuda::std::memory_order_relaxed);

      if (all_flags & is_malformed) {
        malformed = true;
      } else {
        // Finds, for each string, the latest earlier string in the batch with a shorter prefix.
        // Each round, a string whose target's prefix is at least as long as its own takes the
        // target's pointer, until none changes or the rounds run out. A pointer only skips strings
        // whose prefix is at least as long, so `build` can follow a pointer at any round. Round r
        // sets jump_changed[r % 3] and clears jump_changed[(r + 2) % 3], which nobody reads after
        // round r's barrier and nobody writes before round r + 1's.
        //
        // Jumping only shortens `build`'s walk by skipping strings that supply none of the missing
        // bytes, and the walk only meets such a string k where p_k >= p_k+1 > 0. Without such a
        // pair in the batch, every step of every walk copies bytes, so the rounds are skipped. That
        // is the common case for random strings, where the rounds have a small cost, but runs of
        // repeated strings need them to avoid a ~2x slowdown.
        int cur = 0;
        if (all_flags & needs_jump) {
          for (int round = 0; round < max_jump_rounds; ++round) {
            bool changed = false;
            for (int idx = builder_rank; idx < n; idx += num_builders) {
              int const q = shorter_prefix_idx[cur][idx];
              int nq      = q;
              if (q >= 0 && p[q] >= p[idx]) {
                nq      = shorter_prefix_idx[cur][q];
                changed = true;
              }
              shorter_prefix_idx[cur ^ 1][idx] = nq;
            }
            if (warp.any(changed) and lane == 0) { atomicOr(&jump_changed[round % 3], 1); }
            cur ^= 1;
            builder_sync();
            int const any_changed = jump_changed[round % 3];
            if (builder_rank == 0) { jump_changed[(round + 2) % 3] = 0; }
            if (not any_changed) { break; }
          }
        }

        // Read here, after the last barrier before the build, so they are not held across barriers.
        // The pointers are read from `bs` where they are used: holding them here adds spills.
        int const start_val = s->setup.page.start_val;
        int const end_val   = s->setup.page.end_val;

        // Writes string `idx` to `out`. The suffix comes from the suffix data. The prefix comes
        // from the suffixes of earlier strings in the batch, following `shorter_prefix_idx` back
        // from `idx - 1`: string k supplies prefix bytes [p_k, hi), where hi is the lowest byte
        // filled so far. Bytes still missing at the start of the batch come from `bs.prev_str`.
        // `copy` copies one byte range, on one thread or on every lane of a warp.
        auto const build = [&](int idx, uint8_t* out, auto const& copy) {
          int const pl = p[idx];
          copy(out + pl, bs.suffix_data + suffix_offsets[idx], l[idx]);
          int hi = pl;
          int k  = idx - 1;
          while (hi > 0) {
            if (k < 0) {
              copy(out, bs.prev_str, hi);
              break;
            }
            int const pk = p[k];
            if (pk < hi) {
              copy(out + pk, bs.suffix_data + suffix_offsets[k], hi - pk);
              hi = pk;
            }
            k = shorter_prefix_idx[cur][k];
          }
        };
        auto const thread_copy = [](uint8_t* dst, uint8_t const* src, int len) {
          memcpy(dst, src, len);
        };
        // Copies one byte range with every lane, using `wideStrcpy` for ranges over 64 bytes.
        auto const warp_copy = [&](uint8_t* dst, uint8_t const* src, int len) {
          if (len > 64) {
            wideStrcpy(dst, src, len, lane);
          } else {
            for (int i = lane; i < len; i += warp_size) {
              dst[i] = src[i];
            }
          }
        };

        // Strings averaging more than a warp's width are built by a whole warp each, as in
        // `calculate_string_values_cp`; shorter ones by one thread each. The first loop gives each
        // builder thread its own strings: it writes every string's length to the output and, for
        // short strings, builds the string too. A warp builds each long string, which needs a
        // different split, so those builds run in the second loop. One loop per mode with separate
        // length writes was measured, and was slower.
        for (int idx = builder_rank; idx < n; idx += num_builders) {
          int const sp = batch_start + idx;
          if (not use_char_ll and sp >= start_val and sp < end_val) {
            build(idx, bs.string_out + string_offsets[idx], thread_copy);
          }
          auto dst_pos =
            process_nulls ? static_cast<int32_t>(nz_idx_sh[sp]) : static_cast<int32_t>(sp);
          dst_pos -= s->setup.first_row;
          if (dst_pos >= 0) {
            reinterpret_cast<size_type*>(ni.data_out)[dst_pos] = p[idx] + l[idx];
          }
        }
        if (use_char_ll) {
          for (int idx = builder_warp; idx < n; idx += num_builder_warps) {
            int const sp = batch_start + idx;
            if (sp >= start_val and sp < end_val) {
              build(idx, bs.string_out + string_offsets[idx], warp_copy);
            }
          }
        }

        // Keep the batch's last string for the next batch to copy from. A batch entirely before
        // `start_val` writes nothing to the output, so its last string is built here into a slot
        // of `temp_string_buf`. Building it inside the loops above instead, by choosing each
        // string's destination there, was measured about 1.7% slower on long strings (H100).
        uint8_t const* next_prev_str = bs.prev_str;
        int const last               = n - 1;
        int const batch_end          = batch_start + n;
        if (batch_end <= start_val) {
          if (batch_end < end_val) {
            auto* const buf = batch_idx % 2 == 0 ? dba->prefix_seed : dba->temp_buf;
            if (use_char_ll) {
              if (builder_warp == 0) { build(last, buf, warp_copy); }
            } else if (builder_rank == 0) {
              build(last, buf, thread_copy);
            }
            next_prev_str = buf;
          }
        } else {
          next_prev_str = bs.string_out + string_offsets[last];
        }
        // Builder 0 advances the carried state once every builder is done reading it.
        builder_sync();
        if (builder_rank == 0) {
          int64_t batch_string_bytes = 0;
          int64_t batch_suffix_bytes = 0;
          for (int w = 0; w < num_builder_warps; ++w) {
            batch_string_bytes += warp_string_bytes[w];
            batch_suffix_bytes += warp_suffix_bytes[w];
          }
          bs.prev_str = next_prev_str;
          bs.prev_len = p[last] + l[last];
          bs.string_out += batch_string_bytes;
          bs.suffix_data += batch_suffix_bytes;
        }
      }
    }
    // Also the barrier that hands this iteration's lengths to the next iteration's builders.
    bool const any_stream_ended = __syncthreads_or(stream_ended);
    bool const any_malformed    = __syncthreads_or(malformed);
    if (any_stream_ended or any_malformed) {
      if (block.thread_rank() == 0) {
        set_error(static_cast<kernel_error::value_type>(any_malformed
                                                          ? decode_error::DELTA_PARAMS_UNSUPPORTED
                                                          : decode_error::DATA_STREAM_OVERRUN),
                  error_code);
      }
      return;
    }
  }

  // Set the counts that `gpuDecodeLevels` sets in the level-decoding kernel. The offsets scan below
  // reads `ni.value_count`.
  if (block.thread_rank() == 0) {
    auto const page_rows =
      min(s->setup.page.num_input_values, s->setup.first_row + s->setup.num_rows);
    ni.value_count                = page_rows;
    s->progress.input_value_count = page_rows;
    s->progress.input_row_count   = page_rows;
  }
  block.sync();

  // Zero-fill null positions after decoding valid values
  if (ni.valid_map != nullptr) {
    // the level decoder would have advanced valid_map_offset past the page's rows
    int const num_values = s->setup.num_rows;
    zero_fill_null_positions_shared<block_size>(s,
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
    compute_initial_large_strings_offset<false>(s, initial_str_offsets[input_col_idx]);
  } else {
    convert_small_string_lengths_to_offsets<block_size, false>(s);
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
    // The same producer/writer pipeline as `decode_delta_binary_kernel_with_nz_map`: warp 0 decodes
    // a batch of lengths into the decoder's rolling buffer while warp 1 writes the batch before it.
    // The other warps only join the barriers here; they take part in the copy below.
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
  static_assert(decode_delta_binary_with_nz_map_block_size == 2 * cudf::detail::warp_size,
                "The nz map consumer runs one producer warp and one writer warp");

  // `consumer_stream` is ordered after the producer by the caller. The level-decoding kernel below
  // needs no such ordering because it reads only fields set on the host.
  if (has_flat_nz_map) {
    // Not templated on `level_t`: this kernel never reads levels.
    decode_delta_binary_kernel_with_nz_map<<<dim_grid,
                                             dim_block_nz_map,
                                             0,
                                             consumer_stream.get()>>>(
      pages.device_ptr(), chunks, min_row, num_rows, page_mask, error_code);
    CUDF_CUDA_TRY(cudaGetLastError());
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
  dim3 const dim_block_nz_map(decode_delta_byte_array_with_nz_map_block_size, 1);

  // `consumer_stream` is ordered after the producer by the caller. The level-decoding kernel below
  // needs no such ordering because it reads only fields set on the host.
  if (has_flat_nz_map) {
    decode_delta_byte_array_kernel_with_nz_map<<<dim_grid,
                                                 dim_block_nz_map,
                                                 0,
                                                 consumer_stream.get()>>>(
      pages.device_ptr(), chunks, min_row, num_rows, page_mask, initial_str_offsets, error_code);
    CUDF_CUDA_TRY(cudaGetLastError());
  }

  // Skip the launch when every DELTA_BYTE_ARRAY page has a map, since every block would just
  // return.
  if (not needs_level_decoder) { return; }

  if (level_type_size == 1) {
    decode_delta_byte_array_kernel<uint8_t>
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
    decode_delta_byte_array_kernel<uint16_t>
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
