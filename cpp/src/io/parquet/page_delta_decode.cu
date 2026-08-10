/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "delta_binary.cuh"
#include "io/utilities/block_utils.cuh"
#include "page_string_utils.cuh"
#include "parquet_gpu.hpp"

#include <cudf/detail/utilities/cuda.cuh>

#include <rmm/exec_policy.hpp>

#include <thrust/transform_scan.h>

namespace cudf::io::parquet::detail {

namespace {

namespace cg = cooperative_groups;

constexpr int decode_block_size              = 128;
constexpr int decode_delta_binary_block_size = 96;

// DELTA_BYTE_ARRAY encoding (incremental encoding or front compression), is used for BYTE_ARRAY
// columns. For each element in a sequence of strings, a prefix length from the preceding string
// and a suffix is stored. The prefix lengths are DELTA_BINARY_PACKED encoded. The suffixes are
// encoded with DELTA_LENGTH_BYTE_ARRAY encoding, which is a DELTA_BINARY_PACKED list of suffix
// lengths, followed by the concatenated suffix data.
struct delta_byte_array_decoder {
  uint8_t const* last_string;       // pointer to last decoded string...needed for its prefix
  uint8_t const* suffix_char_data;  // pointer to the start of character data

  uint8_t* temp_buf;         // buffer used when skipping values
  uint32_t start_val;        // decoded strings up to this index will be dumped to temp_buf
  uint32_t last_string_len;  // length of the last decoded string

  delta_binary_decoder prefixes;  // state of decoder for prefix lengths
  delta_binary_decoder suffixes;  // state of decoder for suffix lengths

  // initialize the prefixes and suffixes blocks
  __device__ void init(uint8_t const* start, uint8_t const* end, uint32_t start_idx, uint8_t* temp)
  {
    auto const* suffix_start = prefixes.find_end_of_block(start, end);
    suffix_char_data         = suffixes.find_end_of_block(suffix_start, end);
    last_string              = nullptr;
    temp_buf                 = temp;
    start_val                = start_idx;
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
      if (prefix_len != 0 && prefix_lens[blocker] == 0 && lane_out != nullptr) {
        memcpy(lane_out, offsets[blocker], prefix_len);
        prefix_lens[lane_id] = prefix_len = 0;
      }

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

    return string_total;
  }

  // dump strings before start_val to temp buf
  __device__ void skip(bool use_char_ll)
  {
    using cudf::detail::warp_size;
    int const t       = threadIdx.x;
    int const lane_id = t % warp_size;

    // is this even necessary? return if asking to skip the whole block.
    if (start_val >= prefixes.num_encoded_values(true)) { return; }

    // prefixes and suffixes will have the same parameters (it's checked earlier)
    auto const batch_size = prefixes.values_per_mb;

    uint32_t skip_pos = 0;
    while (prefixes.current_value_idx < start_val) {
      // warp 0 gets prefixes and warp 1 gets suffixes
      auto* const db = t < 32 ? &prefixes : &suffixes;

      // this will potentially decode past start_val, but that's ok
      if (t < 64) { db->decode_batch(); }
      __syncthreads();

      // warp 0 decodes the batch.
      if (t < 32) {
        auto const num_to_decode = min(batch_size, start_val - skip_pos);
        auto const bytes_written =
          use_char_ll ? calculate_string_values_cp(temp_buf, skip_pos, num_to_decode, lane_id)
                      : calculate_string_values(temp_buf, skip_pos, num_to_decode, lane_id);
        // store last_string someplace safe in temp buffer
        if (t == 0) {
          memcpy(temp_buf + bytes_written, last_string, last_string_len);
          last_string = temp_buf + bytes_written;
        }
      }
      skip_pos += prefixes.values_per_mb;
      __syncthreads();
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
                             kernel_error::pointer error_code)
{
  __shared__ __align__(16) delta_binary_decoder db_state;
  __shared__ __align__(16) page_state_s state_g;
  __shared__ __align__(16) page_state_buffers_s<delta_rolling_buf_size, 1, 1> state_buffers;

  page_state_s* const s = &state_g;
  auto* const sb        = &state_buffers;
  int const page_idx    = cg::this_grid().block_rank();
  auto const block      = cg::this_thread_block();
  auto const warp       = cg::tiled_partition<cudf::detail::warp_size>(block);
  auto* const db        = &db_state;

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
    s->nesting_info[s->setup.col.max_nesting_depth - 1].valid_map_offset;

  // copying logic from gpuDecodePageData.
  PageNestingDecodeInfo const* nesting_info_base = s->nesting_info;

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

  auto const batch_size = db->values_per_mb;
  if (db->error or batch_size > max_delta_mini_block_size) {
    if (block.thread_rank() == 0) {
      set_error(static_cast<kernel_error::value_type>(decode_error::DELTA_PARAMS_UNSUPPORTED),
                error_code);
    }
    return;
  }

  // if skipped_leaf_values is non-zero, then we need to decode up to the first mini-block
  // that has a value we need.
  if (skipped_leaf_values > 0) { db->skip_values(skipped_leaf_values); }

  while (s->setup.error == 0 &&
         (s->input_value_count < s->setup.num_input_values || s->src_pos < s->nz_count)) {
    uint32_t target_pos;
    uint32_t const src_pos = s->src_pos;

    if (warp.meta_group_rank() < 2) {  // warp0..1
      target_pos = min(src_pos + 2 * batch_size, s->nz_count + batch_size);
    } else {  // warp2
      target_pos = min(s->nz_count, src_pos + batch_size);
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
      gpuDecodeLevels<delta_rolling_buf_size, level_t>(s, sb, target_pos, rep, def, warp);
    } else if (warp.meta_group_rank() == 1) {
      // warp 1
      db->decode_batch();
    } else if (src_pos < target_pos) {
      // warp 2
      // nesting level that is storing actual leaf values
      int const leaf_level_index = s->setup.col.max_nesting_depth - 1;

      // process the mini-block using warps
      for (uint32_t sp = src_pos + warp.thread_rank(); sp < src_pos + batch_size;
           sp += warp.size()) {
        // the position in the output column/buffer
        int32_t dst_pos = sb->nz_idx[rolling_index<delta_rolling_buf_size>(sp)];

        // handle skip_rows here. flat hierarchies can just skip up to first_row.
        if (!has_repetition) { dst_pos -= s->setup.first_row; }

        // place value for this thread
        if (dst_pos >= 0 && sp < target_pos) {
          void* const dst = nesting_info_base[leaf_level_index].data_out + dst_pos * s->dtype_len;
          auto const val  = db->value_at(sp + skipped_leaf_values);
          switch (s->dtype_len) {
            case 1: *static_cast<int8_t*>(dst) = val; break;
            case 2: *static_cast<int16_t*>(dst) = val; break;
            case 4: *static_cast<int32_t*>(dst) = val; break;
            case 8: *static_cast<int64_t*>(dst) = val; break;
          }
        }
      }
      if (warp.thread_rank() == 0) { s->src_pos = src_pos + batch_size; }
    }

    block.sync();
  }

  if (has_repetition) {
    // Zero-fill null positions after decoding valid values
    auto const& ni = s->nesting_info[s->setup.col.max_nesting_depth - 1];
    if (ni.valid_map != nullptr) {
      int const num_values = ni.valid_map_offset - init_valid_map_offset;
      zero_fill_null_positions_shared<decode_block_size>(
        s, s->dtype_len, init_valid_map_offset, num_values, static_cast<int>(block.thread_rank()));
    }
  }

  if (block.thread_rank() == 0 and s->setup.error != 0) { set_error(s->setup.error, error_code); }
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
  __shared__ __align__(16) page_state_s state_g;
  __shared__ __align__(16) page_state_buffers_s<delta_rolling_buf_size, 1, 1> state_buffers;

  page_state_s* const s = &state_g;
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
    s->nesting_info[s->setup.col.max_nesting_depth - 1].valid_map_offset;

  // choose a character parallel string copy when the average string is longer than a warp
  auto const use_char_ll =
    s->setup.page.num_valids > 0 &&
    (s->setup.page.str_bytes / s->setup.page.num_valids) > cudf::detail::warp_size;

  // copying logic from decode_page_data.
  PageNestingDecodeInfo const* nesting_info_base = s->nesting_info;

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
              s->setup.page.temp_string_buf);
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

  // sanity check to make sure we can process this page
  auto const batch_size = prefix_db->values_per_mb;
  if (batch_size > max_delta_mini_block_size) {
    if (block.thread_rank() == 0) {
      set_error(static_cast<kernel_error::value_type>(decode_error::DELTA_PARAMS_UNSUPPORTED),
                error_code);
    }
    return;
  }

  // if this is a bounds page and nested, then we need to skip up front. non-nested will work
  // its way through the page.
  int string_pos = has_repetition ? s->setup.page.start_val : 0;
  auto const is_bounds_pg =
    is_bounds_page(s->setup.page, s->setup.col.start_row, min_row, num_rows, has_repetition);
  if (is_bounds_pg && string_pos > 0) { dba->skip(use_char_ll); }

  while (!s->setup.error &&
         (s->input_value_count < s->setup.num_input_values || s->src_pos < s->nz_count)) {
    uint32_t target_pos;
    uint32_t const src_pos = s->src_pos;

    if (warp.meta_group_rank() < 3) {  // warp 0..2
      target_pos = min(src_pos + 2 * batch_size, s->nz_count + s->setup.first_row + batch_size);
    } else {  // warp 3
      target_pos = min(s->nz_count, src_pos + batch_size);
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
      gpuDecodeLevels<delta_rolling_buf_size, level_t>(s, sb, target_pos, rep, def, warp);
    } else if (warp.meta_group_rank() == 1) {
      // warp 1
      prefix_db->decode_batch();
    } else if (warp.meta_group_rank() == 2) {
      // warp 2
      suffix_db->decode_batch();
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
        int dst_pos = sb->nz_idx[rolling_index<delta_rolling_buf_size>(sp)];

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

      if (warp.thread_rank() == 0) { s->src_pos = src_pos + batch_size; }
    }

    block.sync();
  }

  // Zero-fill null positions after decoding valid values
  auto const& ni = s->nesting_info[leaf_level_index];
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
                                        cudf::device_span<size_t> initial_str_offsets,
                                        kernel_error::pointer error_code)
{
  __shared__ __align__(16) delta_binary_decoder db_state;
  __shared__ __align__(16) page_state_s state_g;
  __shared__ __align__(16) page_state_buffers_s<delta_rolling_buf_size, 1, 1> state_buffers;
  __shared__ __align__(8) uint8_t const* page_string_data;
  __shared__ size_t string_offset;

  page_state_s* const s = &state_g;
  auto* const sb        = &state_buffers;
  int const page_idx    = cg::this_grid().block_rank();
  auto const block      = cg::this_thread_block();
  auto const warp       = cg::tiled_partition<cudf::detail::warp_size>(block);
  auto* const db        = &db_state;
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
    s->nesting_info[s->setup.col.max_nesting_depth - 1].valid_map_offset;

  // copying logic from gpuDecodePageData.
  PageNestingDecodeInfo const* nesting_info_base = s->nesting_info;

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

  // sanity check to make sure we can process this page
  auto const batch_size = db->values_per_mb;
  if (db->error or batch_size > max_delta_mini_block_size) {
    if (block.thread_rank() == 0) {
      set_error(static_cast<kernel_error::value_type>(decode_error::DELTA_PARAMS_UNSUPPORTED),
                error_code);
    }
    return;
  }

  int const leaf_level_index = s->setup.col.max_nesting_depth - 1;

  // db->init_binary_block below resets db->values_per_mb
  block.sync();
  // if this is a bounds page, then we need to decode up to the first mini-block
  // that has a value we need, and set string_offset to the position of the first value in the
  // string data block.
  auto const is_bounds_pg =
    is_bounds_page(s->setup.page, s->setup.col.start_row, min_row, num_rows, has_repetition);
  if (is_bounds_pg && s->setup.page.start_val > 0) {
    if (warp.meta_group_rank() == 0) {
      // string_off is only valid on thread 0
      auto const string_off = db->skip_values_and_sum(s->setup.page.start_val);
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

  while (!s->setup.error &&
         (s->input_value_count < s->setup.num_input_values || s->src_pos < s->nz_count)) {
    uint32_t target_pos;
    uint32_t const src_pos = s->src_pos;

    if (warp.meta_group_rank() < 2) {  // warp0..1
      target_pos = min(src_pos + 2 * batch_size, s->nz_count + batch_size);
    } else {  // warp2
      target_pos = min(s->nz_count, src_pos + batch_size);
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
      gpuDecodeLevels<delta_rolling_buf_size, level_t>(s, sb, target_pos, rep, def, warp);
    } else if (warp.meta_group_rank() == 1) {
      // warp 1
      db->decode_batch();
    } else if (warp.meta_group_rank() == 2 && src_pos < target_pos) {
      // warp 2
      int const nproc = min(batch_size, s->setup.page.end_val - string_pos);
      string_pos += nproc;

      // process the mini-block in batches of 32
      for (uint32_t sp = src_pos + warp.thread_rank(); sp < src_pos + batch_size;
           sp += warp.size()) {
        // the position in the output column/buffer
        int dst_pos = sb->nz_idx[rolling_index<delta_rolling_buf_size>(sp)];

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

      if (warp.thread_rank() == 0) { s->src_pos = src_pos + batch_size; }
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
                         rmm::cuda_stream_view stream)
{
  CUDF_EXPECTS(pages.size() > 0, "There is no page to decode");

  dim3 dim_block(decode_delta_binary_block_size, 1);
  dim3 dim_grid(pages.size(), 1);  // 1 threadblock per page

  if (level_type_size == 1) {
    decode_delta_binary_kernel<uint8_t><<<dim_grid, dim_block, 0, stream.value()>>>(
      pages.device_ptr(), chunks, min_row, num_rows, page_mask, error_code);
    CUDF_CUDA_TRY(cudaGetLastError());
  } else {
    decode_delta_binary_kernel<uint16_t><<<dim_grid, dim_block, 0, stream.value()>>>(
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
                             rmm::cuda_stream_view stream)
{
  CUDF_EXPECTS(pages.size() > 0, "There is no page to decode");

  dim3 const dim_block(decode_block_size, 1);
  dim3 const dim_grid(pages.size(), 1);  // 1 threadblock per page

  if (level_type_size == 1) {
    decode_delta_byte_array_kernel<uint8_t><<<dim_grid, dim_block, 0, stream.value()>>>(
      pages.device_ptr(), chunks, min_row, num_rows, page_mask, initial_str_offsets, error_code);
    CUDF_CUDA_TRY(cudaGetLastError());
  } else {
    decode_delta_byte_array_kernel<uint16_t><<<dim_grid, dim_block, 0, stream.value()>>>(
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
                                    rmm::cuda_stream_view stream)
{
  CUDF_EXPECTS(pages.size() > 0, "There is no page to decode");

  dim3 const dim_block(decode_block_size, 1);
  dim3 const dim_grid(pages.size(), 1);  // 1 threadblock per page

  if (level_type_size == 1) {
    decode_delta_length_byte_array_kernel<uint8_t><<<dim_grid, dim_block, 0, stream.value()>>>(
      pages.device_ptr(), chunks, min_row, num_rows, page_mask, initial_str_offsets, error_code);
    CUDF_CUDA_TRY(cudaGetLastError());
  } else {
    decode_delta_length_byte_array_kernel<uint16_t><<<dim_grid, dim_block, 0, stream.value()>>>(
      pages.device_ptr(), chunks, min_row, num_rows, page_mask, initial_str_offsets, error_code);
    CUDF_CUDA_TRY(cudaGetLastError());
  }
}

}  // namespace cudf::io::parquet::detail
