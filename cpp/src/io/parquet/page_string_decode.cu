/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "delta_binary.cuh"
#include "error.hpp"
#include "page_decode.cuh"
#include "page_state_composed.cuh"
#include "page_string_utils.cuh"

#include <cudf/detail/algorithms/reduce.cuh>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/detail/utilities/stream_pool.hpp>
#include <cudf/strings/detail/gather.cuh>
#include <cudf/utilities/memory_resource.hpp>

#include <cooperative_groups/reduce.h>
#include <cuda/buffer>
#include <cuda/cmath>
#include <cuda/functional>
#include <cuda/std/utility>
#include <thrust/transform_scan.h>

#include <bitset>

namespace cudf::io::parquet::detail {

namespace cg = cooperative_groups;

namespace {

constexpr int preprocess_block_size    = 512;
constexpr int delta_preproc_block_size = 64;
constexpr int delta_length_block_size  = 32;

/**
 * @brief Compute the start and end page value bounds for this page
 *
 * This uses definition and repetition level info to determine the number of valid and null
 * values for the page, taking into account skip_rows/num_rows (if set).
 *
 * @param s The local page info
 * @param min_row Row index to start reading at
 * @param num_rows Maximum number of rows to read
 * @param is_bounds_pg True if this page is clipped
 * @param has_repetition True if the schema is nested
 * @return pair containing start and end value indexes
 * @tparam level_t Type used to store decoded repetition and definition levels
 */
template <typename level_t>
__device__ cuda::std::pair<int, int> page_bounds(
  auto* const s, size_t min_row, size_t num_rows, bool is_bounds_pg, bool has_repetition)
{
  using block_reduce = cub::BlockReduce<int, preprocess_block_size>;
  using block_scan   = cub::BlockScan<int, preprocess_block_size>;
  __shared__ union {
    typename block_reduce::TempStorage reduce_storage;
    typename block_scan::TempStorage scan_storage;
  } temp_storage;

  auto const block = cg::this_thread_block();
  auto const t     = block.thread_rank();

  int const max_depth = s->setup.col.max_nesting_depth;
  int const max_def   = s->nesting.nesting_info[max_depth - 1].max_def_level;

  auto const pp = &s->setup.page;
  // Clamp to decoded level count; for chunked reads we may have decoded fewer values than
  // num_input_values (skip_rows/num_rows), and the level buffers are only that large.
  int const actual_num_values =
    (pp->num_decoded_level_values > 0)
      ? cuda::std::min(pp->num_input_values, pp->num_decoded_level_values)
      : pp->num_input_values;

  // can skip all this if we know there are no nulls
  if (max_def == 0 && !is_bounds_pg) {
    if (t == 0) {
      s->setup.page.num_valids = actual_num_values;
      s->setup.page.num_nulls  = 0;
    }
    return {0, actual_num_values};
  }

  int start_value = 0;
  int end_value   = actual_num_values;

  // get the level data
  bool const process_nulls = should_process_nulls(s);
  auto const def_decode    = reinterpret_cast<level_t*>(pp->lvl_decode_buf[level_type::DEFINITION]);
  auto const rep_decode    = reinterpret_cast<level_t*>(pp->lvl_decode_buf[level_type::REPETITION]);

  // if this is a bounds page, we need to do extra work to find the start and/or end value index
  if (is_bounds_pg) {
    __shared__ int skipped_values;
    __shared__ int skipped_leaf_values;
    __shared__ int last_input_value;
    __shared__ int end_val_idx;

    // need these for skip_rows case
    auto const page_start_row = s->setup.col.start_row + pp->chunk_row;
    auto const max_row        = min_row + num_rows;
    auto const begin_row      = page_start_row >= min_row ? 0 : min_row - page_start_row;
    auto const max_page_rows  = pp->num_rows - begin_row;
    auto const page_rows      = page_start_row + begin_row + max_page_rows <= max_row
                                  ? max_page_rows
                                  : max_row - (page_start_row + begin_row);
    auto end_row              = begin_row + page_rows;
    int row_fudge             = -1;

    // short circuit for no nulls
    if (max_def == 0 && !has_repetition) {
      if (t == 0) {
        pp->num_nulls  = 0;
        pp->num_valids = end_row - begin_row;
      }
      return {begin_row, end_row};
    }

    int row_count           = 0;
    int leaf_count          = 0;
    bool skipped_values_set = false;
    bool end_value_set      = false;

    // If page_start_row >= min_row, then skipped_values is 0 and we don't have to search for
    // start_value. If there's repetition then we've already calculated
    // skipped_values/skipped_leaf_values.
    // TODO(ets): If we hit this condition, and end_row > last row in page, then we can skip
    // more of the processing below.
    if (has_repetition or page_start_row >= min_row) {
      if (t == 0) {
        if (has_repetition) {
          skipped_values      = pp->skipped_values;
          skipped_leaf_values = pp->skipped_leaf_values;
        } else {
          skipped_values      = 0;
          skipped_leaf_values = 0;
        }
      }
      skipped_values_set = true;
      __syncthreads();
    }

    int processed = 0;
    while (processed < actual_num_values) {
      if (has_repetition && (processed == 0) && (rep_decode[0] != 0)) {
        // special case where page does not begin at a row boundary
        end_row++;  // need to finish off the previous row
        row_fudge = 0;
      }

      // get absolute thread row index
      int const idx_t = processed + t;
      int const is_new_row =
        idx_t < actual_num_values && (!has_repetition || rep_decode[idx_t] == 0);
      int thread_row_count, block_row_count;
      block_scan(temp_storage.scan_storage)
        .InclusiveSum(is_new_row, thread_row_count, block_row_count);
      __syncthreads();

      // get absolute thread leaf index
      int const is_new_leaf = process_nulls
                                ? idx_t < actual_num_values && (def_decode[idx_t] >= max_def)
                                : idx_t < actual_num_values;
      int thread_leaf_count, block_leaf_count;
      block_scan(temp_storage.scan_storage)
        .InclusiveSum(is_new_leaf, thread_leaf_count, block_leaf_count);
      __syncthreads();

      // if we have not set skipped values yet, see if we found the first in-bounds row
      if (!skipped_values_set && row_count + block_row_count > begin_row) {
        // if this thread is in row bounds
        int const row_index = thread_row_count + row_count - 1;
        int const in_row_bounds =
          idx_t < actual_num_values && (row_index >= begin_row) && (row_index < end_row);

        int local_count, global_count;
        block_scan(temp_storage.scan_storage)
          .InclusiveSum(in_row_bounds, local_count, global_count);
        __syncthreads();

        // we found it
        if (global_count > 0) {
          // this is the thread that represents the first row. need to test in_row_bounds for
          // the case where we only want one row and local_count == 1 for many threads.
          if (local_count == 1 && in_row_bounds) {
            skipped_values = idx_t;
            skipped_leaf_values =
              leaf_count + (is_new_leaf ? thread_leaf_count - 1 : thread_leaf_count);
          }
          skipped_values_set = true;
        }
      }

      // test if row_count will exceed end_row in this batch
      if (row_count + block_row_count >= end_row) {
        // if this thread exceeds row bounds. row_fudge change depending on whether we've faked
        // the end row to account for starting a page in the middle of a row.
        int const row_index          = thread_row_count + row_count + row_fudge;
        int const exceeds_row_bounds = row_index >= end_row;

        int local_count, global_count;
        block_scan(temp_storage.scan_storage)
          .InclusiveSum(exceeds_row_bounds, local_count, global_count);
        __syncthreads();

        // we found it
        if (global_count > 0) {
          // this is the thread that represents the end row.
          if (local_count == 1) {
            last_input_value = idx_t;
            end_val_idx = leaf_count + (is_new_leaf ? thread_leaf_count - 1 : thread_leaf_count);
          }
          end_value_set = true;
          break;
        }
      }

      row_count += block_row_count;
      leaf_count += block_leaf_count;
      processed += preprocess_block_size;
    }

    // Sync shared variables like end_val_idx, skipped_leaf_values, etc.
    __syncthreads();
    start_value = skipped_values_set ? skipped_leaf_values : 0;
    end_value   = end_value_set ? end_val_idx : leaf_count;

    if (t == 0) {
      int const v0                = skipped_values_set ? skipped_values : 0;
      int const vn                = end_value_set ? last_input_value : actual_num_values;
      int const total_values      = vn - v0;
      int const total_leaf_values = end_value - start_value;
      int const num_nulls         = total_values - total_leaf_values;
      pp->num_nulls               = num_nulls;
      pp->num_valids              = total_leaf_values;
    }
  }
  // already filtered out unwanted pages, so need to count all non-null values in this page
  else if (process_nulls) {
    int num_nulls = 0;
    int idx_t     = t;
    while (idx_t < actual_num_values) {
      if (def_decode[idx_t] < max_def) { num_nulls++; }
      idx_t += preprocess_block_size;
    }

    int const null_count = block_reduce(temp_storage.reduce_storage).Sum(num_nulls);
    __syncthreads();

    if (t == 0) {
      pp->num_nulls  = null_count;
      pp->num_valids = actual_num_values - null_count;
    }
    end_value -= null_count;
  } else {
    if (t == 0) {
      pp->num_nulls  = 0;
      pp->num_valids = actual_num_values;
    }
  }

  return {start_value, end_value};
}

/**
 * @brief Compute string size information for dictionary encoded strings.
 *
 * @param data Pointer to the start of the page data stream
 * @param dict_base Pointer to the start of the dictionary
 * @param dict_bits The number of bits used to in the dictionary bit packing
 * @param dict_size Size of the dictionary in bytes
 * @param data_size Size of the page data in bytes
 * @param start_value Do not count values that occur before this index
 * @param end_value Do not count values that occur after this index
 */
__device__ size_t totalDictEntriesSize(uint8_t const* data,
                                       uint8_t const* dict_base,
                                       int dict_bits,
                                       int dict_size,
                                       int data_size,
                                       int start_value,
                                       int end_value)
{
  int const t              = threadIdx.x;
  uint8_t const* ptr       = data;
  uint8_t const* const end = data + data_size;
  int const bytecnt        = (dict_bits + 7) >> 3;
  size_t l_str_len         = 0;  // partial sums across threads
  int pos                  = 0;  // current value index in the data stream
  int t0                   = 0;  // thread 0 for this batch

  int dict_run = 0;
  int dict_val = 0;

  while (pos < end_value && ptr <= end) {
    if (dict_run <= 1) {
      dict_run = (ptr < end) ? get_vlq32(ptr, end) : 0;
      if (!(dict_run & 1)) {
        // Repeated value
        if (ptr + bytecnt <= end) {
          int32_t run_val = ptr[0];
          if (bytecnt > 1) {
            run_val |= ptr[1] << 8;
            if (bytecnt > 2) {
              run_val |= ptr[2] << 16;
              if (bytecnt > 3) { run_val |= ptr[3] << 24; }
            }
          }
          dict_val = run_val & ((1 << dict_bits) - 1);
        }
        ptr += bytecnt;
      }
    }

    int batch_len;
    if (dict_run & 1) {
      // Literal batch: must output a multiple of 8, except for the last batch
      int batch_len_div8;
      batch_len      = max(min(preprocess_block_size, (int)(dict_run >> 1) * 8), 1);
      batch_len_div8 = (batch_len + 7) >> 3;
      dict_run -= batch_len_div8 * 2;
      ptr += batch_len_div8 * dict_bits;
    } else {
      batch_len = dict_run >> 1;
      dict_run  = 0;
    }

    int const is_literal = dict_run & 1;

    // calculate my thread id for this batch.  way to round-robin the work.
    int mytid = t - t0;
    if (mytid < 0) mytid += preprocess_block_size;

    // compute dictionary index.
    if (is_literal) {
      int dict_idx = 0;
      if (mytid < batch_len) {
        dict_idx         = dict_val;
        int32_t ofs      = (mytid - ((batch_len + 7) & ~7)) * dict_bits;
        uint8_t const* p = ptr + (ofs >> 3);
        ofs &= 7;
        if (p < end) {
          uint32_t c = 8 - ofs;
          dict_idx   = (*p++) >> ofs;
          if (c < dict_bits && p < end) {
            dict_idx |= (*p++) << c;
            c += 8;
            if (c < dict_bits && p < end) {
              dict_idx |= (*p++) << c;
              c += 8;
              if (c < dict_bits && p < end) { dict_idx |= (*p++) << c; }
            }
          }
          dict_idx &= (1 << dict_bits) - 1;
        }

        if (pos + mytid < end_value) {
          uint32_t const dict_pos = (dict_bits > 0) ? dict_idx * sizeof(string_index_pair) : 0;
          if (pos + mytid >= start_value && dict_pos < (uint32_t)dict_size) {
            auto const* src = reinterpret_cast<string_index_pair const*>(dict_base + dict_pos);
            l_str_len += src->second;
          }
        }
      }

      t0 += batch_len;
    } else {
      int const start_off =
        (pos < start_value && pos + batch_len > start_value) ? start_value - pos : 0;
      batch_len = min(batch_len, end_value - pos);
      if (mytid == 0) {
        uint32_t const dict_pos = (dict_bits > 0) ? dict_val * sizeof(string_index_pair) : 0;
        if (pos + batch_len > start_value && dict_pos < (uint32_t)dict_size) {
          auto const* src = reinterpret_cast<string_index_pair const*>(dict_base + dict_pos);
          l_str_len += (batch_len - start_off) * src->second;
        }
      }

      t0 += 1;
    }

    t0 = t0 % preprocess_block_size;
    pos += batch_len;
  }
  __syncthreads();

  using block_reduce = cub::BlockReduce<size_t, preprocess_block_size>;
  __shared__ typename block_reduce::TempStorage reduce_storage;
  size_t sum_l = block_reduce(reduce_storage).Sum(l_str_len);

  return sum_l;
}

/**
 * @brief Compute string size information for DELTA_BYTE_ARRAY encoded strings.
 *
 * This traverses the packed prefix and suffix lengths, summing them to obtain the total
 * number of bytes needed for the decoded string data. It also calculates an upper bound
 * for the largest string length to obtain an upper bound on temporary space needed if
 * rows will be skipped.
 *
 * Called with 64 threads.
 *
 * @param data Pointer to the start of the page data stream
 * @param end Pointer to the end of the page data stream
 * @param start_value Do not count values that occur before this index
 * @param end_value Do not count values that occur after this index
 * @return A pair of `size_t` values representing the total string size and temp buffer size
 * required for decoding
 */
__device__ cuda::std::pair<size_t, size_t> totalDeltaByteArraySize(uint8_t const* data,
                                                                   uint8_t const* end,
                                                                   int start_value,
                                                                   int end_value)
{
  using cudf::detail::warp_size;

  __shared__ __align__(16) delta_binary_decoder prefixes;
  __shared__ __align__(16) delta_binary_decoder suffixes;

  auto const block  = cg::this_thread_block();
  auto const warp   = cg::tiled_partition<cudf::detail::warp_size>(block);
  int const t       = block.thread_rank();
  int const lane_id = warp.thread_rank();

  if (t == 0) {
    auto const* suffix_start = prefixes.find_end_of_block(data, end);
    suffixes.init_binary_block(suffix_start, end);
  }
  block.sync();

  // two warps will traverse the prefixes and suffixes and sum them up
  auto const warp_id = warp.meta_group_rank();

  size_t total_bytes = 0;
  uleb128_t max_len  = 0;

  // traverse one decoder's values, accumulating into total_bytes/max_len. dispatched below on a
  // compile-time-constant decoder per warp rather than a runtime-selected `db` pointer: selecting
  // the object at runtime makes the compiler speculatively load both prefixes and suffixes on
  // every `db->` access, so the suffix warp reads prefix state (and vice versa) while the other
  // warp writes it, which compute-sanitizer racecheck reports as a cross-warp hazard.
  auto const traverse = [&](delta_binary_decoder& db) {
    // initialize with first value (which is stored in last_value)
    if (lane_id == 0 && start_value == 0) { total_bytes = db.last_value; }

    uleb128_t lane_sum = 0;
    uleb128_t lane_max = 0;
    while (db.current_value_idx < end_value && db.current_value_idx < db.num_encoded_values(true)) {
      if (not db.advance_past_first_value(warp)) { break; }

      // decode one warp_size-wide pass at a time and read it back immediately: a whole
      // mini-block is only fully resident in the rolling buffer while it fits, but a single
      // pass always is
      uint32_t const num_pass = db.values_per_mb / warp_size;
      for (uint32_t p = 0; p < num_pass; p++) {
        db.calc_mini_block_pass(p, warp);

        // get per lane sum for this pass
        uint32_t const idx = db.current_value_idx + p * warp_size + lane_id;
        if (idx >= start_value && idx < end_value && idx < db.value_count) {
          lane_sum += db.value[rolling_index<delta_rolling_buf_size>(idx)];
        }
        // need lane_max over all values, not just in bounds
        if (idx < db.value_count) {
          lane_max = max(lane_max, db.value[rolling_index<delta_rolling_buf_size>(idx)]);
        }
      }

      warp.sync();
      if (lane_id == 0) { db.setup_next_mini_block(true); }
      warp.sync();
    }

    // get sum and max for warp (valid on all lanes; only lane 0 is consumed below).
    auto const warp_sum = cg::reduce(warp, lane_sum, cg::plus<uleb128_t>{});
    auto const warp_max = cg::reduce(warp, lane_max, cg::greater<uleb128_t>{});

    if (lane_id == 0) {
      total_bytes += warp_sum;
      max_len = warp_max;
    }
  };

  if (warp_id == 0) {
    traverse(prefixes);
  } else if (warp_id == 1) {
    traverse(suffixes);
  }
  block.sync();

  // now sum up total_bytes from the two warps
  auto const final_bytes =
    cudf::detail::single_lane_block_sum_reduce<delta_preproc_block_size, 0>(total_bytes);

  // Sum up prefix and suffix max lengths to get a max possible string length. Multiply that by
  // the largest number of strings a decode-loop iteration can stage, plus one to save the last
  // string.
  auto const temp_bytes =
    cudf::detail::single_lane_block_sum_reduce<delta_preproc_block_size, 0>(max_len) *
    (delta_max_batch_size + 1);

  return {final_bytes, temp_bytes};
}

/**
 * @brief Kernel for computing string page bounds information.
 *
 * This kernel traverses the repetition and definition level data to determine start and end values
 * for pages with string-like data. Also calculates the number of null and valid values in the
 * page. Does nothing if the page mask is neither `STRING` nor `DELTA_BYTE_ARRAY`. On exit the
 * `num_nulls`, `num_valids`, `start_val` and `end_val` fields of the `PageInfo` struct will be
 * populated.
 *
 * @param pages All pages to be decoded
 * @param chunks All chunks to be decoded
 * @param min_rows crop all rows below min_row
 * @param num_rows Maximum number of rows to read
 * @param all_rows If true, all rows will be read, regardless of `min_row` and `num_rows`
 * @tparam level_t Type used to store decoded repetition and definition levels
 */
template <typename level_t>
CUDF_KERNEL void __launch_bounds__(preprocess_block_size)
  compute_string_page_bounds_kernel(PageInfo* pages,
                                    device_span<ColumnChunkDesc const> chunks,
                                    device_span<bool const> page_mask,
                                    size_t min_row,
                                    size_t num_rows,
                                    bool all_rows)
{
  __shared__ __align__(16) full_page_decode_state state_g;

  auto* const s      = &state_g;
  int const page_idx = blockIdx.x;
  int const t        = threadIdx.x;
  PageInfo* const pp = &pages[page_idx];

  if (t == 0) {
    // don't clobber these if they're already computed from the index
    if (!pp->has_value_info) {
      pp->num_nulls  = 0;
      pp->num_valids = 0;
    }
    // reset str_bytes to 0 in case it's already been calculated (esp needed for chunked reads).
    pp->str_bytes = 0;
    pp->start_val = 0;
    pp->end_val   = 0;
  }

  if (all_rows) {
    min_row  = chunks[pp->chunk_idx].start_row + pp->chunk_row;
    num_rows = pp->num_rows;
  }

  // whether or not we have repetition levels (lists)
  bool const has_repetition = chunks[pp->chunk_idx].max_level[level_type::REPETITION] > 0;

  // setup page info
  if (!setup_local_page_info(s,
                             pp,
                             chunks,
                             min_row,
                             num_rows,
                             mask_filter{STRINGS_MASK},
                             page_processing_stage::STRING_BOUNDS)) {
    return;
  }

  bool const is_bounds_pg =
    is_bounds_page(s->setup.page, s->setup.col.start_row, min_row, num_rows, has_repetition);

  // if we have size info, then we only need to do this for bounds pages
  if (pp->has_value_info && !is_bounds_pg) { return; }

  // Zero out everything and return early if the page is pruned
  if (not page_mask.empty() and not page_mask[page_idx]) {
    if (t == 0) {
      pp->num_nulls  = 0;
      pp->num_valids = 0;
      pp->start_val  = 0;
      pp->end_val    = 0;
    }
    return;
  }

  // find start/end value indices
  auto const [start_value, end_value] =
    page_bounds<level_t>(s, min_row, num_rows, is_bounds_pg, has_repetition);

  // need to save num_nulls and num_valids calculated in page_bounds in this page
  if (t == 0) {
    pp->num_nulls  = s->setup.page.num_nulls;
    pp->num_valids = s->setup.page.num_valids;
    pp->start_val  = start_value;
    pp->end_val    = end_value;
  }
}

/**
 * @brief Kernel for computing string page output size information for delta_byte_array encoding.
 *
 * This call ignores columns that are not DELTA_BYTE_ARRAY encoded. On exit the `str_bytes` field
 * of the `PageInfo` struct will be populated. Also fills in the `temp_string_size` field if rows
 * are to be skipped.
 *
 * @param pages All pages to be decoded
 * @param chunks All chunks to be decoded
 * @param page_mask Page mask indicating if this column needs to be decoded
 * @param min_rows crop all rows below min_row
 * @param num_rows Maximum number of rows to read
 */
CUDF_KERNEL void __launch_bounds__(delta_preproc_block_size)
  compute_delta_page_string_sizes_kernel(PageInfo* pages,
                                         device_span<ColumnChunkDesc const> chunks,
                                         device_span<bool const> page_mask,
                                         size_t min_row,
                                         size_t num_rows)
{
  __shared__ __align__(16) string_size_scan_state state_g;

  auto* const s      = &state_g;
  int const page_idx = blockIdx.x;
  int const t        = threadIdx.x;
  PageInfo* const pp = &pages[page_idx];

  // whether or not we have repetition levels (lists)
  bool const has_repetition = chunks[pp->chunk_idx].max_level[level_type::REPETITION] > 0;

  // setup page info
  if (!setup_local_page_info(s,
                             pp,
                             chunks,
                             min_row,
                             num_rows,
                             mask_filter{decode_kernel_mask::DELTA_BYTE_ARRAY},
                             page_processing_stage::STRING_BOUNDS)) {
    return;
  }
  // Return early if the page is pruned
  if (page_mask.size() > 0 && not page_mask[page_idx]) {
    pp->str_bytes = 0;
    return;
  }

  auto const start_value = pp->start_val;

  // if data size is known, can short circuit here
  if (chunks[pp->chunk_idx].physical_type == Type::FIXED_LEN_BYTE_ARRAY) {
    if (t == 0) {
      pp->str_bytes = pp->num_valids * s->output_cvt.dtype_len_in;

      // only need temp space if we're skipping values
      if (start_value > 0) {
        // save enough for one decode batch plus some extra to save the last_string
        pp->temp_string_size = s->output_cvt.dtype_len_in * (delta_max_batch_size + 1);
      }
    }
  } else {
    bool const is_bounds_pg =
      is_bounds_page(s->setup.page, s->setup.col.start_row, min_row, num_rows, has_repetition);

    // if we have size info, then we only need to do this for bounds pages
    if (pp->has_value_info && !is_bounds_pg) {
      // check if we need to store values from the index
      if (t == 0 && is_page_contained(s->setup.page, s->setup.col.start_row, min_row, num_rows)) {
        pp->str_bytes = pp->str_bytes_from_index;
      }
      return;
    }

    // now process string info in the range [start_value, end_value)
    // set up for decoding strings...can be either plain or dictionary
    uint8_t const* data      = s->stream.data_start;
    uint8_t const* const end = s->stream.data_end;
    auto const end_value     = pp->end_val;

    auto const [len, temp_bytes] = totalDeltaByteArraySize(data, end, start_value, end_value);

    if (t == 0) {
      pp->str_bytes = len;

      // only need temp space if we're skipping values
      if (start_value > 0) { pp->temp_string_size = temp_bytes; }
    }
  }
}

/**
 * @brief Kernel for computing string page output size information for DELTA_LENGTH_BYTE_ARRAY
 * encoding.
 *
 * This call ignores columns that are not DELTA_LENGTH_BYTE_ARRAY encoded. On exit the `str_bytes`
 * field of the `PageInfo` struct will be populated.
 *
 * Currently this function only supports being called by a single warp.
 *
 * @param pages All pages to be decoded
 * @param chunks All chunks to be decoded
 * @param page_mask Page mask indicating if this column needs to be decoded
 * @param min_rows crop all rows below min_row
 * @param num_rows Maximum number of rows to read
 */
CUDF_KERNEL void __launch_bounds__(delta_length_block_size)
  compute_delta_length_page_string_sizes_kernel(PageInfo* pages,
                                                device_span<ColumnChunkDesc const> chunks,
                                                device_span<bool const> page_mask,
                                                size_t min_row,
                                                size_t num_rows)
{
  using cudf::detail::warp_size;
  __shared__ __align__(16) string_size_scan_state state_g;
  __shared__ __align__(16) delta_binary_decoder string_lengths;

  auto const warp    = cg::tiled_partition<cudf::detail::warp_size>(cg::this_thread_block());
  auto* const s      = &state_g;
  int const page_idx = blockIdx.x;
  int const t        = threadIdx.x;
  PageInfo* const pp = &pages[page_idx];

  // whether or not we have repetition levels (lists)
  bool const has_repetition = chunks[pp->chunk_idx].max_level[level_type::REPETITION] > 0;

  // setup page info
  if (!setup_local_page_info(s,
                             pp,
                             chunks,
                             min_row,
                             num_rows,
                             mask_filter{decode_kernel_mask::DELTA_LENGTH_BA},
                             page_processing_stage::STRING_BOUNDS)) {
    return;
  }

  // Return early if the page is pruned
  if (page_mask.size() > 0 && not page_mask[page_idx]) {
    pp->str_bytes = 0;
    return;
  }

  bool const is_bounds_pg =
    is_bounds_page(s->setup.page, s->setup.col.start_row, min_row, num_rows, has_repetition);

  // if we have size info, then we only need to do this for bounds pages
  if (pp->has_value_info && !is_bounds_pg) {
    // check if we need to store values from the index
    if (t == 0 && is_page_contained(s->setup.page, s->setup.col.start_row, min_row, num_rows)) {
      pp->str_bytes = pp->str_bytes_from_index;
    }
    return;
  }

  // for DELTA_LENGTH_BYTE_ARRAY, string size is page_data_size - size_of_delta_binary_block.
  // so all we need to do is skip the encoded string size info and then do pointer arithmetic,
  // if this isn't a bounds page.
  if (not is_bounds_pg) {
    if (t == 0) {
      auto const* string_start =
        string_lengths.find_end_of_block(s->stream.data_start, s->stream.data_end);
      size_t len    = static_cast<size_t>(s->stream.data_end - string_start);
      pp->str_bytes = len;
    }
  } else {
    // now process string info in the range [start_value, end_value)
    // set up for decoding strings...can be either plain or dictionary
    auto const start_value = pp->start_val;
    auto const end_value   = pp->end_val;

    if (t == 0) { string_lengths.init_binary_block(s->stream.data_start, s->stream.data_end); }
    warp.sync();

    size_t total_bytes = 0;

    // initialize with first value (unless there are no values)
    if (t == 0 && start_value == 0 && start_value < end_value) {
      total_bytes = string_lengths.value_at(0);
    }

    uleb128_t lane_sum = 0;
    while (string_lengths.current_value_idx < end_value &&
           string_lengths.current_value_idx < string_lengths.num_encoded_values(true)) {
      if (not string_lengths.advance_past_first_value(warp)) { break; }

      // decode one warp_size-wide pass at a time and read it back immediately: a whole
      // mini-block is only fully resident in the rolling buffer while it fits, but a single
      // pass always is
      uint32_t const num_pass = string_lengths.values_per_mb / warp_size;
      for (uint32_t p = 0; p < num_pass; p++) {
        string_lengths.calc_mini_block_pass(p, warp);

        // get per lane sum for this pass
        uint32_t const idx = string_lengths.current_value_idx + p * warp_size + t;
        if (idx >= start_value && idx < end_value && idx < string_lengths.value_count) {
          lane_sum += string_lengths.value[rolling_index<delta_rolling_buf_size>(idx)];
        }
      }

      warp.sync();
      if (t == 0) { string_lengths.setup_next_mini_block(true); }
      warp.sync();
    }

    // get sum for warp.
    // note: warp_sum will only be valid on lane 0.
    auto const warp_sum = cg::reduce(warp, lane_sum, cg::plus<uleb128_t>{});

    if (t == 0) {
      total_bytes += warp_sum;
      pp->str_bytes = total_bytes;
    }
  }
}

/**
 * @brief Kernel for computing string page output size information.
 *
 * This call ignores non-string columns. On exit the `str_bytes` field of the `PageInfo` struct will
 * be populated.
 *
 * @param pages All pages to be decoded
 * @param chunks All chunks to be decoded
 * @param page_mask Page mask indicating if this column needs to be decoded
 * @param page_string_offset_indices Per-page indices into the string offset buffer
 * @param min_rows crop all rows below min_row
 * @param num_rows Maximum number of rows to read
 */
CUDF_KERNEL void __launch_bounds__(preprocess_block_size)
  compute_page_string_sizes_kernel(PageInfo* pages,
                                   device_span<ColumnChunkDesc const> chunks,
                                   device_span<bool const> page_mask,
                                   device_span<size_t const> page_string_offset_indices,
                                   size_t min_row,
                                   size_t num_rows)
{
  __shared__ __align__(16) string_size_scan_state state_g;

  auto* const s      = &state_g;
  int const page_idx = blockIdx.x;
  int const t        = threadIdx.x;
  PageInfo* const pp = &pages[page_idx];

  // whether or not we have repetition levels (lists)
  bool const has_repetition = chunks[pp->chunk_idx].max_level[level_type::REPETITION] > 0;

  // setup page info
  if (!setup_local_page_info(s,
                             pp,
                             chunks,
                             min_row,
                             num_rows,
                             mask_filter{STRINGS_MASK_NON_DELTA},
                             page_processing_stage::STRING_BOUNDS)) {
    return;
  }

  // Return early if the page is pruned
  if (page_mask.size() > 0 && not page_mask[page_idx]) {
    pp->str_bytes = 0;
    return;
  }

  bool const is_bounds_pg =
    is_bounds_page(s->setup.page, s->setup.col.start_row, min_row, num_rows, has_repetition);

  // if we have size info, then we only need to do this for bounds pages
  if (pp->has_value_info && !is_bounds_pg) {
    // check if we need to store values from the index
    if (t == 0 && is_page_contained(s->setup.page, s->setup.col.start_row, min_row, num_rows)) {
      pp->str_bytes = pp->str_bytes_from_index;
    }
    return;
  }

  auto const& col  = s->setup.col;
  size_t str_bytes = 0;
  // short circuit for FIXED_LEN_BYTE_ARRAY
  if (col.physical_type == Type::FIXED_LEN_BYTE_ARRAY) {
    str_bytes = pp->num_valids * s->output_cvt.dtype_len_in;
  } else {
    // now process string info in the range [start_value, end_value)
    // set up for decoding strings...can be either plain or dictionary
    uint8_t const* data      = s->stream.data_start;
    uint8_t const* const end = s->stream.data_end;
    uint8_t const* dict_base = nullptr;
    int dict_size            = 0;
    auto const start_value   = pp->start_val;
    auto const end_value     = pp->end_val;

    switch (pp->encoding) {
      case Encoding::PLAIN_DICTIONARY:
      case Encoding::RLE_DICTIONARY:
        // RLE-packed dictionary indices, first byte indicates index length in bits
        if (col.str_dict_index) {
          // String dictionary: use index
          dict_base = reinterpret_cast<uint8_t const*>(col.str_dict_index);
          dict_size = col.dict_page->num_input_values * sizeof(string_index_pair);
        } else {
          dict_base = col.dict_page->page_data;
          dict_size = col.dict_page->uncompressed_page_size;
        }

        // FIXME: need to return an error condition...this won't actually do anything
        if (s->stream.dict_bits > 32 || (!dict_base && col.dict_page->num_input_values > 0)) {
          CUDF_UNREACHABLE("invalid dictionary bit size");
        }

        str_bytes = totalDictEntriesSize(
          data, dict_base, s->stream.dict_bits, dict_size, (end - data), start_value, end_value);
        break;
      case Encoding::PLAIN:
        if (start_value >= end_value) {
          // No values, so no offsets are needed (the offset buffer may be empty)
          str_bytes = 0;
        } else {
          // Check if we have precomputed offsets available
          if (col.column_string_offset_base == nullptr || page_string_offset_indices.empty()) {
            CUDF_UNREACHABLE("string offsets should have been preprocessed already!");
          }
          // The span from str_offsets[start] to str_offsets[end] includes the length prefixes
          // of strings [start+1, end), but we only want the string data bytes.
          // So we need to subtract 4 bytes per string for those embedded length prefixes.
          uint32_t const* str_offsets =
            col.column_string_offset_base + page_string_offset_indices[page_idx];
          int const num_values = end_value - start_value;
          size_t const span    = str_offsets[end_value] - str_offsets[start_value];
          str_bytes            = span - sizeof(int32_t) * num_values;
        }
        break;
    }
  }

  if (t == 0) {
    // TODO check for overflow
    pp->str_bytes = str_bytes;

    // only need temp space for delta
    pp->temp_string_size = 0;
  }
}

// Functor used to set the `temp_string_buf` pointer for each page. `data` points to a buffer
// to be used when skipping rows in the delta_byte_array decoder. Given a page and an offset,
// set the page's `temp_string_buf` to be `data + offset`.
struct page_tform_functor {
  uint8_t* const data;

  __device__ PageInfo operator()(PageInfo& page, int64_t offset)
  {
    if (page.temp_string_size != 0) { page.temp_string_buf = data + offset; }
    return page;
  }
};

}  // anonymous namespace

/**
 * @copydoc cudf::io::parquet::detail::compute_page_string_sizes_pass1
 */
void compute_page_string_sizes_pass1(cudf::detail::hostdevice_span<PageInfo> pages,
                                     cudf::detail::hostdevice_span<ColumnChunkDesc const> chunks,
                                     cudf::device_span<bool const> page_mask,
                                     cudf::device_span<size_t const> page_string_offset_indices,
                                     size_t min_row,
                                     size_t num_rows,
                                     uint32_t kernel_mask,
                                     bool all_rows,
                                     int level_type_size,
                                     cuda::stream_ref stream)
{
  dim3 const dim_block(preprocess_block_size, 1);
  dim3 const dim_grid(pages.size(), 1);  // 1 threadblock per page

  if (level_type_size == 1) {
    compute_string_page_bounds_kernel<uint8_t><<<dim_grid, dim_block, 0, stream.get()>>>(
      pages.device_ptr(), chunks, page_mask, min_row, num_rows, all_rows);
    CUDF_CUDA_TRY(cudaGetLastError());
  } else {
    compute_string_page_bounds_kernel<uint16_t><<<dim_grid, dim_block, 0, stream.get()>>>(
      pages.device_ptr(), chunks, page_mask, min_row, num_rows, all_rows);
    CUDF_CUDA_TRY(cudaGetLastError());
  }

  // kernel mask may contain other kernels we don't need to count
  int const count_mask = kernel_mask & STRINGS_MASK;
  int const nkernels   = std::bitset<32>(count_mask).count();
  auto const streams   = cudf::detail::fork_streams(stream, nkernels);

  int s_idx = 0;
  if (BitAnd(kernel_mask, decode_kernel_mask::DELTA_BYTE_ARRAY) != 0) {
    dim3 dim_delta(delta_preproc_block_size, 1);
    compute_delta_page_string_sizes_kernel<<<dim_grid, dim_delta, 0, streams[s_idx++].get()>>>(
      pages.device_ptr(), chunks, page_mask, min_row, num_rows);
    CUDF_CUDA_TRY(cudaGetLastError());
  }
  if (BitAnd(kernel_mask, decode_kernel_mask::DELTA_LENGTH_BA) != 0) {
    dim3 dim_delta(delta_length_block_size, 1);
    compute_delta_length_page_string_sizes_kernel<<<dim_grid,
                                                    dim_delta,
                                                    0,
                                                    streams[s_idx++].get()>>>(
      pages.device_ptr(), chunks, page_mask, min_row, num_rows);
    CUDF_CUDA_TRY(cudaGetLastError());
  }
  if (BitAnd(kernel_mask, STRINGS_MASK_NON_DELTA) != 0) {
    compute_page_string_sizes_kernel<<<dim_grid, dim_block, 0, streams[s_idx++].get()>>>(
      pages.device_ptr(), chunks, page_mask, page_string_offset_indices, min_row, num_rows);
    CUDF_CUDA_TRY(cudaGetLastError());
  }

  // synchronize the streams
  cudf::detail::join_streams(streams, stream);
}

/**
 * @copydoc cudf::io::parquet::detail::compute_page_string_sizes_pass2
 */
void compute_page_string_sizes_pass2(cudf::detail::hostdevice_span<PageInfo> pages,
                                     cudf::detail::hostdevice_span<ColumnChunkDesc const> chunks,
                                     cuda::device_buffer<uint8_t>& temp_string_buf,
                                     cuda::stream_ref stream)
{
  // check for needed temp space for DELTA_BYTE_ARRAY
  auto const need_sizes =
    cudf::detail::any_of(pages.device_begin(),
                         pages.device_end(),
                         cuda::proclaim_return_type<bool>(
                           [] __device__(auto const& page) { return page.temp_string_size != 0; }),
                         stream);

  if (need_sizes) {
    // sum up all of the temp_string_sizes
    auto const page_sizes = cuda::proclaim_return_type<int64_t>(
      [] __device__(PageInfo const& page) { return page.temp_string_size; });
    auto const total_size = cudf::detail::transform_reduce(pages.device_begin(),
                                                           pages.device_end(),
                                                           page_sizes,
                                                           int64_t{0},
                                                           cuda::std::plus<int64_t>{},
                                                           stream);

    // now do an exclusive scan over the temp_string_sizes to get offsets for each
    // page's chunk of the temp buffer
    cuda::device_buffer<int64_t> page_string_offsets(
      stream, cudf::get_current_device_resource_ref(), pages.size(), cuda::no_init);
    thrust::transform_exclusive_scan(
      rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
      pages.device_begin(),
      pages.device_end(),
      page_string_offsets.begin(),
      page_sizes,
      0L,
      cuda::std::plus<int64_t>{});

    // allocate the temp space
    temp_string_buf = cuda::device_buffer<uint8_t>(
      stream, temp_string_buf.memory_resource(), total_size, cuda::no_init);

    // now use the offsets array to set each page's temp_string_buf pointers
    thrust::transform(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                      pages.device_begin(),
                      pages.device_end(),
                      page_string_offsets.begin(),
                      pages.device_begin(),
                      page_tform_functor{temp_string_buf.data()});
  }
}

namespace {

constexpr int string_offset_warps          = 8;
constexpr int string_offset_window_bytes   = 1024;
constexpr unsigned string_offset_warp_mask = 0xffffffffu;

// Snapshot each page's scan inputs before reusing the CTA-wide setup state.
struct string_offset_page {
  uint8_t const* data;
  uint32_t* offsets;
  int32_t bytes;
  int32_t num_input_values;
  uint32_t num_values;
  bool active;
};

struct alignas(128) string_offset_window {
  // The extra word makes the second word of an unaligned funnel load safe at the end.
  uint32_t data[string_offset_window_bytes / sizeof(uint32_t) + 1];
};

/**
 * @brief Refill one warp's window, aligned to a 128-byte global-memory boundary.
 *
 * Only complete, aligned 16-byte vectors use asynchronous copies. The first partial
 * alignment unit and the final partial vector use bounded scalar loads, so neither
 * an unaligned page start nor a truncated page can cause an out-of-page access.
 * All lanes must participate; the caller must finish reading the previous window first.
 */
__device__ void prefetch_string_data(string_offset_page const& page,
                                     int lane,
                                     uint32_t next,
                                     string_offset_window& window,
                                     int32_t& base,
                                     int32_t& end)
{
  int const misalignment = (reinterpret_cast<uintptr_t>(page.data) + next) & 127;
  base                   = static_cast<int32_t>(next) - misalignment;
  if (base >= 0 && static_cast<uint32_t>(page.bytes) - base >= string_offset_window_bytes) {
    // Interior windows need exactly two aligned vectors per lane, with no scalar edges.
    constexpr int stripe_bytes = cudf::detail::warp_size * 16;
    static_assert(string_offset_window_bytes == 2 * stripe_bytes);
    end             = base + string_offset_window_bytes;
    auto* dst       = window.data + lane * 4;
    auto const* src = page.data + base + lane * 16;
#if __CUDA_ARCH__ >= 800
    asm volatile(
      "cp.async.cg.shared.global [%0], [%1], 16;\n"
      "cp.async.cg.shared.global [%0+%2], [%1+%2], 16;" ::"r"(
        static_cast<uint32_t>(__cvta_generic_to_shared(dst))),
      "l"(src),
      "n"(stripe_bytes)
      : "memory");
    asm volatile("cp.async.commit_group;" ::: "memory");
#else
    *reinterpret_cast<uint4*>(dst) = *reinterpret_cast<uint4 const*>(src);
    *reinterpret_cast<uint4*>(dst + stripe_bytes / sizeof(uint32_t)) =
      *reinterpret_cast<uint4 const*>(src + stripe_bytes);
#endif
    if (lane == 0) { window.data[string_offset_window_bytes / sizeof(uint32_t)] = 0; }
  } else {
    int const count = static_cast<int>(
      cuda::std::min<int64_t>(string_offset_window_bytes, int64_t{page.bytes} - base));
    end                = static_cast<int32_t>(int64_t{base} + count);
    int const first    = base < 0 ? 128 : 0;
    int const bulk_end = count & ~15;

    for (int offset = first + lane * 16; offset < bulk_end;
         offset += cudf::detail::warp_size * 16) {
      auto const* src = page.data + base + offset;
      auto* dst       = window.data + offset / sizeof(uint32_t);
#if __CUDA_ARCH__ >= 800
      asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(
                     static_cast<uint32_t>(__cvta_generic_to_shared(dst))),
                   "l"(src)
                   : "memory");
#else
      *reinterpret_cast<uint4*>(dst) = *reinterpret_cast<uint4 const*>(src);
#endif
    }
#if __CUDA_ARCH__ >= 800
    asm volatile("cp.async.commit_group;" ::: "memory");
#endif

    // Scalar stores never overlap the asynchronous copies.
    for (int word_index = lane; word_index < (count + 3) / 4;
         word_index += cudf::detail::warp_size) {
      int const offset = word_index * 4;
      if (offset >= first && offset < bulk_end) { continue; }
      uint32_t word = 0;
#pragma unroll
      for (int byte = 0; byte < 4; ++byte) {
        int64_t const index = int64_t{base} + offset + byte;
        if (index >= 0 && index < page.bytes) { word |= uint32_t{page.data[index]} << (8 * byte); }
      }
      window.data[word_index] = word;
    }
    if (lane == 0) { window.data[(count + 3) / 4] = 0; }
  }
#if __CUDA_ARCH__ >= 800
  asm volatile("cp.async.wait_group 0;" ::: "memory");
#endif
  __syncwarp();
}

/** @brief Fill unwritten offsets, including the terminal offset used to recover string lengths. */
__device__ void fill_string_offset_tail(string_offset_page const& page,
                                        int lane,
                                        uint32_t written,
                                        uint32_t next)
{
  __syncwarp();
  for (uint64_t pos = uint64_t{written} + lane; pos <= page.num_values;
       pos += cudf::detail::warp_size) {
    page.offsets[pos] = next + uint32_t{4};
  }
}

/** @brief Cooperatively scan length prefixes using one warp's shared-memory window. */
__device__ void read_string_offsets_buffered(string_offset_page const& page,
                                             int lane,
                                             string_offset_window& window,
                                             kernel_error::pointer error_code)
{
  if (page.bytes <= 0) { return; }
  uint32_t pos = 0, next = 0;
  if (page.bytes < 4 || page.num_values == 0) {
    fill_string_offset_tail(page, lane, pos, next);
    return;
  }
  int32_t base = 0, end = 0;
  prefetch_string_data(page, lane, next, window, base, end);

  while (true) {
    // A complete prefix inside this window is also inside the page. Keep page-end
    // and refill handling outside the common loop, using a window-relative cursor.
    // Unsigned subtraction also yields the correct index when the initial base is negative.
    uint32_t index           = next - static_cast<uint32_t>(base);
    uint32_t const limit     = static_cast<uint32_t>(end) - static_cast<uint32_t>(base) - 4;
    uint32_t const remaining = static_cast<uint32_t>(page.bytes) - static_cast<uint32_t>(base) - 4;
    uint32_t const bias      = static_cast<uint32_t>(base) + 4;
    do {
      uint32_t const len =
        __funnelshift_r(window.data[index / 4], window.data[index / 4 + 1], (index & 3) * 8);
      // Remaining page bytes fit in INT32_MAX, so this also rejects negative lengths.
      if (len > remaining - index) {
        next = static_cast<uint32_t>(base) + index;
        if (lane == 0) {
          set_error(static_cast<kernel_error::value_type>(decode_error::STRING_DATA_OVERRUN),
                    error_code);
        }
        goto done;
      }
      uint32_t const offset = bias + index;
      index += len + 4;
      ++pos;
      if (lane == 0) { page.offsets[pos - 1] = offset; }
    } while (index <= limit && pos < page.num_values);

    next = static_cast<uint32_t>(base) + index;
    // Accepted lengths keep next <= page.bytes <= INT32_MAX; next + 4 cannot overflow.
    if (pos == page.num_values || next + 4 > static_cast<uint32_t>(page.bytes)) { break; }
    __syncwarp();  // All consumers finish before this warp reuses its window.
    prefetch_string_data(page, lane, next, window, base, end);
  }
done:
  fill_string_offset_tail(page, lane, pos, next);
}

/** @brief Read long strings' length prefixes directly, avoiding unused payload copies. */
__device__ void read_string_offsets_sequential(string_offset_page const& page,
                                               int lane,
                                               kernel_error::pointer error_code)
{
  uint32_t next = 0, written = page.num_values;
  if (lane == 0) {
    for (uint32_t pos = 0; pos < page.num_values; ++pos) {
      auto const string_offset = next + sizeof(int32_t);
      if (string_offset > static_cast<uint32_t>(page.bytes)) {
        written = pos;
        break;
      }
      int32_t len;
      cuda::std::memcpy(&len, page.data + next, sizeof(len));
      if (len < 0 || static_cast<int64_t>(string_offset) + len > page.bytes) {
        written = pos;
        set_error(static_cast<kernel_error::value_type>(decode_error::STRING_DATA_OVERRUN),
                  error_code);
        break;
      }
      page.offsets[pos] = string_offset;
      next              = string_offset + len;
    }
  }
  written = __shfl_sync(string_offset_warp_mask, written, 0);
  next    = __shfl_sync(string_offset_warp_mask, next, 0);
  for (uint64_t pos = uint64_t{written} + lane; pos <= page.num_values;
       pos += cudf::detail::warp_size) {
    page.offsets[pos] = next + sizeof(int32_t);
  }
}

/**
 * @brief Precompute PLAIN string offsets, with one page per warp and eight warps per CTA.
 *
 * Page setup uses CTA-wide barriers, so every thread sets up the pages collectively
 * before the independent warp scans start. Inactive pages and a partial final group
 * must still participate in all setup barriers.
 */
CUDF_KERNEL void preprocess_string_offsets_kernel(
  PageInfo* pages,
  device_span<ColumnChunkDesc const> chunks,
  device_span<size_t const> page_string_offset_indices,
  device_span<bool const> page_mask,
  int num_pages,
  size_t min_row,
  size_t num_rows,
  kernel_error::pointer error_code)
{
  constexpr uint32_t string_mask = BitOr(decode_kernel_mask::STRING,
                                         decode_kernel_mask::STRING_NESTED,
                                         decode_kernel_mask::STRING_LIST,
                                         decode_kernel_mask::STRING_STREAM_SPLIT,
                                         decode_kernel_mask::STRING_STREAM_SPLIT_NESTED,
                                         decode_kernel_mask::STRING_STREAM_SPLIT_LIST);
  __shared__ __align__(16) string_offset_scan_state state;
  __shared__ string_offset_page work[string_offset_warps];
  __shared__ string_offset_window windows[string_offset_warps];

  for (int slot = 0; slot < string_offset_warps; ++slot) {
    int const page_idx = blockIdx.x * string_offset_warps + slot;
    if (threadIdx.x == 0) { work[slot] = {nullptr, nullptr, 0, 0, 0, false}; }
    if (page_idx < num_pages && (page_mask.empty() || page_mask[page_idx])) {
      auto* const pp    = pages + page_idx;
      auto const& chunk = chunks[pp->chunk_idx];
      if (chunk.physical_type != Type::FIXED_LEN_BYTE_ARRAY &&
          setup_local_page_info(&state,
                                pp,
                                chunks,
                                min_row,
                                num_rows,
                                mask_filter{string_mask},
                                page_processing_stage::STRING_BOUNDS)) {
        if (threadIdx.x == 0) {
          // Nulls cannot be skipped yet. List pages count leaf values, including
          // continuations from the preceding page; flat pages include skipped rows.
          bool const is_list = chunk.max_level[level_type::REPETITION] != 0;
          size_t const count = is_list ? pp->nesting[chunk.max_nesting_depth - 1].batch_size
                                       : state.setup.num_rows + state.setup.first_row;
          if (count != 0) {
            work[slot] = {state.stream.data_start,
                          chunk.column_string_offset_base + page_string_offset_indices[page_idx],
                          state.stream.dict_size,
                          pp->num_input_values,
                          static_cast<uint32_t>(count),
                          true};
          }
        }
      }
    }
    // Finish consuming setup state before the next page can overwrite it.
    __syncthreads();
  }

  int const warp  = threadIdx.x / cudf::detail::warp_size;
  int const lane  = threadIdx.x % cudf::detail::warp_size;
  auto const page = work[warp];
  if (!page.active) { return; }

  // Preserve the existing cutoff, including unsigned subtraction for null-heavy pages.
  auto const avg_string_length = page.bytes / page.num_input_values - sizeof(int32_t);
  if (avg_string_length > string_offset_window_bytes / 16) {
    read_string_offsets_sequential(page, lane, error_code);
  } else {
    read_string_offsets_buffered(page, lane, windows[warp], error_code);
  }
}

}  // namespace

/**
 * @brief Launches the pre-processing kernel to fill string offsets for non-dictionary, non-FLBA
 * string columns
 *
 * @param pages List of pages
 * @param chunks List of column chunks
 * @param page_string_offset_indices Device span of offsets, indexed per-page, into the column's
 * string offset buffer
 * @param page_mask Boolean vector indicating which pages need to be processed
 * @param min_row Minimum row index to read
 * @param num_rows Number of rows to read starting from min_row
 * @param error_code Error code to set if string data is corrupted
 * @param stream CUDA stream to use
 */
void preprocess_string_offsets(cudf::detail::hostdevice_span<PageInfo> pages,
                               cudf::detail::hostdevice_span<ColumnChunkDesc const> chunks,
                               cudf::device_span<size_t const> page_string_offset_indices,
                               cudf::device_span<bool const> page_mask,
                               size_t min_row,
                               size_t num_rows,
                               kernel_error::pointer error_code,
                               cuda::stream_ref stream)
{
  if (pages.size() == 0) { return; }

  constexpr int block_size = string_offset_warps * cudf::detail::warp_size;
  auto const num_blocks    = cuda::ceil_div(pages.size(), size_t{string_offset_warps});
  preprocess_string_offsets_kernel<<<num_blocks, block_size, 0, stream.get()>>>(
    pages.device_ptr(),
    chunks,
    page_string_offset_indices,
    page_mask,
    static_cast<int>(pages.size()),
    min_row,
    num_rows,
    error_code);
  CUDF_CUDA_TRY(cudaGetLastError());
}

}  // namespace cudf::io::parquet::detail
