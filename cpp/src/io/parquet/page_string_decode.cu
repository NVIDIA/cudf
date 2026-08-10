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
  page_state_s* const s, size_t min_row, size_t num_rows, bool is_bounds_pg, bool has_repetition)
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
  int const max_def   = s->nesting_info[max_depth - 1].max_def_level;

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
  using WarpReduce = cub::WarpReduce<uleb128_t>;

  __shared__ typename WarpReduce::TempStorage temp_storage[2];

  __shared__ __align__(16) delta_binary_decoder prefixes;
  __shared__ __align__(16) delta_binary_decoder suffixes;

  int const t       = threadIdx.x;
  int const lane_id = t % warp_size;
  int const warp_id = t / warp_size;

  if (t == 0) {
    auto const* suffix_start = prefixes.find_end_of_block(data, end);
    suffixes.init_binary_block(suffix_start, end);
  }
  __syncthreads();

  // two warps will traverse the prefixes and suffixes and sum them up
  auto const db = t < warp_size ? &prefixes : t < 2 * warp_size ? &suffixes : nullptr;

  size_t total_bytes = 0;
  uleb128_t max_len  = 0;

  if (db != nullptr) {
    // initialize with first value (which is stored in last_value)
    if (lane_id == 0 && start_value == 0) { total_bytes = db->last_value; }

    uleb128_t lane_sum = 0;
    uleb128_t lane_max = 0;
    while (db->current_value_idx < end_value &&
           db->current_value_idx < db->num_encoded_values(true)) {
      // calculate values for current mini-block
      db->calc_mini_block_values(lane_id);

      // get per lane sum for mini-block
      for (uint32_t i = 0; i < db->values_per_mb; i += 32) {
        uint32_t const idx = db->current_value_idx + i + lane_id;
        if (idx >= start_value && idx < end_value && idx < db->value_count) {
          lane_sum += db->value[rolling_index<delta_rolling_buf_size>(idx)];
        }
        // need lane_max over all values, not just in bounds
        if (idx < db->value_count) {
          lane_max = max(lane_max, db->value[rolling_index<delta_rolling_buf_size>(idx)]);
        }
      }

      if (lane_id == 0) { db->setup_next_mini_block(true); }
      __syncwarp();
    }

    // get sum for warp.
    // note: warp_sum will only be valid on lane 0.
    auto const warp_sum = WarpReduce(temp_storage[warp_id]).Sum(lane_sum);
    __syncwarp();
    auto const warp_max = WarpReduce(temp_storage[warp_id]).Reduce(lane_max, cuda::maximum{});

    if (lane_id == 0) {
      total_bytes += warp_sum;
      max_len = warp_max;
    }
  }
  __syncthreads();

  // now sum up total_bytes from the two warps
  auto const final_bytes =
    cudf::detail::single_lane_block_sum_reduce<delta_preproc_block_size, 0>(total_bytes);

  // Sum up prefix and suffix max lengths to get a max possible string length. Multiply that
  // by the number of strings in a mini-block, plus one to save the last string.
  auto const temp_bytes =
    cudf::detail::single_lane_block_sum_reduce<delta_preproc_block_size, 0>(max_len) *
    (db->values_per_mb + 1);

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
  __shared__ __align__(16) page_state_s state_g;

  page_state_s* const s = &state_g;
  int const page_idx    = blockIdx.x;
  int const t           = threadIdx.x;
  PageInfo* const pp    = &pages[page_idx];

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

  bool const is_bounds_pg = is_bounds_page(s, min_row, num_rows, has_repetition);

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
  __shared__ __align__(16) page_state_s state_g;

  page_state_s* const s = &state_g;
  int const page_idx    = blockIdx.x;
  int const t           = threadIdx.x;
  PageInfo* const pp    = &pages[page_idx];

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
      pp->str_bytes = pp->num_valids * s->dtype_len_in;

      // only need temp space if we're skipping values
      if (start_value > 0) {
        // just need to parse the header of the first delta binary block to get values_per_mb
        delta_binary_decoder db;
        db.init_binary_block(s->stream.data_start, s->stream.data_end);
        // save enough for one mini-block plus some extra to save the last_string
        pp->temp_string_size = s->dtype_len_in * (db.values_per_mb + 1);
      }
    }
  } else {
    bool const is_bounds_pg = is_bounds_page(s, min_row, num_rows, has_repetition);

    // if we have size info, then we only need to do this for bounds pages
    if (pp->has_value_info && !is_bounds_pg) {
      // check if we need to store values from the index
      if (t == 0 && is_page_contained(s, min_row, num_rows)) {
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
  using WarpReduce = cub::WarpReduce<uleb128_t>;
  __shared__ typename WarpReduce::TempStorage temp_storage;
  __shared__ __align__(16) page_state_s state_g;
  __shared__ __align__(16) delta_binary_decoder string_lengths;

  page_state_s* const s = &state_g;
  int const page_idx    = blockIdx.x;
  int const t           = threadIdx.x;
  PageInfo* const pp    = &pages[page_idx];

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

  bool const is_bounds_pg = is_bounds_page(s, min_row, num_rows, has_repetition);

  // if we have size info, then we only need to do this for bounds pages
  if (pp->has_value_info && !is_bounds_pg) {
    // check if we need to store values from the index
    if (t == 0 && is_page_contained(s, min_row, num_rows)) {
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
    __syncwarp();

    size_t total_bytes = 0;

    // initialize with first value (unless there are no values)
    if (t == 0 && start_value == 0 && start_value < end_value) {
      total_bytes = string_lengths.value_at(0);
    }

    uleb128_t lane_sum = 0;
    while (string_lengths.current_value_idx < end_value &&
           string_lengths.current_value_idx < string_lengths.num_encoded_values(true)) {
      // calculate values for current mini-block
      string_lengths.calc_mini_block_values(t);

      // get per lane sum for mini-block
      for (uint32_t i = 0; i < string_lengths.values_per_mb; i += warp_size) {
        uint32_t const idx = string_lengths.current_value_idx + i + t;
        if (idx >= start_value && idx < end_value && idx < string_lengths.value_count) {
          lane_sum += string_lengths.value[rolling_index<delta_rolling_buf_size>(idx)];
        }
      }

      if (t == 0) { string_lengths.setup_next_mini_block(true); }
      __syncwarp();
    }

    // get sum for warp.
    // note: warp_sum will only be valid on lane 0.
    auto const warp_sum = WarpReduce(temp_storage).Sum(lane_sum);

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
  __shared__ __align__(16) page_state_s state_g;

  page_state_s* const s = &state_g;
  int const page_idx    = blockIdx.x;
  int const t           = threadIdx.x;
  PageInfo* const pp    = &pages[page_idx];

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

  bool const is_bounds_pg = is_bounds_page(s, min_row, num_rows, has_repetition);

  // if we have size info, then we only need to do this for bounds pages
  if (pp->has_value_info && !is_bounds_pg) {
    // check if we need to store values from the index
    if (t == 0 && is_page_contained(s, min_row, num_rows)) {
      pp->str_bytes = pp->str_bytes_from_index;
    }
    return;
  }

  auto const& col  = s->setup.col;
  size_t str_bytes = 0;
  // short circuit for FIXED_LEN_BYTE_ARRAY
  if (col.physical_type == Type::FIXED_LEN_BYTE_ARRAY) {
    str_bytes = pp->num_valids * s->dtype_len_in;
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
        // Check if we have precomputed offsets available
        if (col.column_string_offset_base == nullptr || page_string_offset_indices.empty()) {
          CUDF_UNREACHABLE("string offsets should have been preprocessed already!");
        }

        if (start_value >= end_value) {
          str_bytes = 0;
        } else {
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
                                     rmm::cuda_stream_view stream)
{
  dim3 const dim_block(preprocess_block_size, 1);
  dim3 const dim_grid(pages.size(), 1);  // 1 threadblock per page

  if (level_type_size == 1) {
    compute_string_page_bounds_kernel<uint8_t><<<dim_grid, dim_block, 0, stream.value()>>>(
      pages.device_ptr(), chunks, page_mask, min_row, num_rows, all_rows);
    CUDF_CUDA_TRY(cudaGetLastError());
  } else {
    compute_string_page_bounds_kernel<uint16_t><<<dim_grid, dim_block, 0, stream.value()>>>(
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
    compute_delta_page_string_sizes_kernel<<<dim_grid, dim_delta, 0, streams[s_idx++].value()>>>(
      pages.device_ptr(), chunks, page_mask, min_row, num_rows);
    CUDF_CUDA_TRY(cudaGetLastError());
  }
  if (BitAnd(kernel_mask, decode_kernel_mask::DELTA_LENGTH_BA) != 0) {
    dim3 dim_delta(delta_length_block_size, 1);
    compute_delta_length_page_string_sizes_kernel<<<dim_grid,
                                                    dim_delta,
                                                    0,
                                                    streams[s_idx++].value()>>>(
      pages.device_ptr(), chunks, page_mask, min_row, num_rows);
    CUDF_CUDA_TRY(cudaGetLastError());
  }
  if (BitAnd(kernel_mask, STRINGS_MASK_NON_DELTA) != 0) {
    compute_page_string_sizes_kernel<<<dim_grid, dim_block, 0, streams[s_idx++].value()>>>(
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
                                     rmm::device_uvector<uint8_t>& temp_string_buf,
                                     rmm::cuda_stream_view stream)
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
    rmm::device_uvector<int64_t> page_string_offsets(pages.size(), stream);
    thrust::transform_exclusive_scan(
      rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
      pages.device_begin(),
      pages.device_end(),
      page_string_offsets.begin(),
      page_sizes,
      0L,
      cuda::std::plus<int64_t>{});

    // allocate the temp space
    temp_string_buf.resize(total_size, stream);

    // now use the offsets array to set each page's temp_string_buf pointers
    thrust::transform(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                      pages.device_begin(),
                      pages.device_end(),
                      page_string_offsets.begin(),
                      pages.device_begin(),
                      page_tform_functor{temp_string_buf.data()});
  }
}

/**
 * @brief Helper function to prefetch next chunk of data into shared memory using multiple threads
 *
 * @param t Thread index within block
 * @param next_length_offset Current offset in the data stream
 * @param dict_size Total size of the dictionary/data
 * @param cur Pointer to the start of the data
 * @param prefetch_buffer Shared memory buffer for prefetching
 * @param[in,out] buffer_base Offset corresponding to the start of the prefetched buffer
 * @param[in,out] buffer_end Offset corresponding to the end of the prefetched buffer
 * @return Whether the buffer contains valid data
 */
template <int32_t prefetch_size, int32_t block_size>
inline __device__ bool prefetch_string_data(int t,
                                            int32_t next_length_offset,
                                            int32_t dict_size,
                                            uint8_t const* cur,
                                            uint8_t* prefetch_buffer,
                                            int32_t& buffer_base,
                                            int32_t& buffer_end)
{
  // The start of the prefetched buffer is the offset to the next length in the data stream
  buffer_base = next_length_offset;

  int32_t const total_bytes_to_copy = cuda::std::min(prefetch_size, dict_size - buffer_base);
  if (total_bytes_to_copy <= 0) { return false; }  // No data left to copy
  buffer_end = buffer_base + total_bytes_to_copy;

  // Nominally, each thread will copy an equal number of bytes; this rounds up.
  auto const nominal_thread_bytes_to_copy =
    cudf::util::div_rounding_up_unsafe<int32_t>(total_bytes_to_copy, block_size);
  int32_t const thread_offset = nominal_thread_bytes_to_copy * t;

  if (thread_offset < total_bytes_to_copy) {
    // Guard against the end of the data stream
    int32_t const thread_bytes_to_copy =
      cuda::std::min(nominal_thread_bytes_to_copy, total_bytes_to_copy - thread_offset);

    if (thread_bytes_to_copy > 0) {
      int32_t const thread_copy_from_index = buffer_base + thread_offset;
      cuda::std::memcpy(reinterpret_cast<void*>(&prefetch_buffer[thread_offset]),
                        reinterpret_cast<void const*>(&cur[thread_copy_from_index]),
                        thread_bytes_to_copy);
    }
  }
  return true;
}

/**
 * @brief Read string offsets with buffering and prefetching
 *
 * This function uses a prefetch buffer to efficiently process string offsets by
 * reading data in chunks. Thread 0 reads string lengths, and all threads cooperate
 * on prefetching data and filling the remaining entries.
 *
 * @param s Page state containing data_start, dict_size and other info
 * @param num_values_to_process Number of values to process
 * @param str_offsets Output buffer for string offsets
 * @param error_code Error code to set if a string length overruns the page
 */
template <int32_t block_size, size_t prefetch_size>
inline __device__ void read_string_offsets_buffered(auto* s,
                                                    size_t num_values_to_process,
                                                    uint32_t* str_offsets,
                                                    kernel_error::pointer error_code)
{
  auto const block     = cg::this_thread_block();
  int const t          = block.thread_rank();
  uint8_t const* cur   = s->stream.data_start;
  auto const dict_size = s->stream.dict_size;

  int32_t buffer_base        = 0;
  int32_t buffer_end         = 0;
  int32_t next_length_offset = 0;
  __shared__ __align__(128) uint8_t prefetch_buffer[prefetch_size];

  // Initial prefetch
  if (!prefetch_string_data<prefetch_size, block_size>(
        t, next_length_offset, dict_size, cur, prefetch_buffer, buffer_base, buffer_end)) {
    return;  // No data to process
  }
  block.sync();  // Sync all of the prefetched data for all of the threads

  // Parquet data is: 4-byte length, string, 4-byte length, string, ...
  size_t num_values_written = num_values_to_process;  // Will update below if run out of data
  for (size_t pos = 0; pos < num_values_to_process; pos++) {
    int32_t const string_offset = next_length_offset + sizeof(int32_t);
    // Natural end-of-data exit: we've consumed all the value bytes, so the next length prefix
    // would begin at or past the end of the data -- it doesn't exist. Nothing left to read.
    if (string_offset > dict_size) {
      num_values_written = pos;  // Reached the end of the data
      break;
    }

    // Check if we need to prefetch more data
    if ((next_length_offset + sizeof(int32_t)) > buffer_end) {
      block.sync();  // Make sure all of the threads have finished reading the previous data
      if (!prefetch_string_data<prefetch_size, block_size>(
            t, next_length_offset, dict_size, cur, prefetch_buffer, buffer_base, buffer_end)) {
        num_values_written = pos;  // End of the data
        break;
      }
      block.sync();  // Sync all of the prefetched data for all of the threads
    }

    // Read the length of the string from the prefetched buffer
    int32_t const prefetch_read_index = next_length_offset - buffer_base;
    int32_t len;
    cuda::std::memcpy(reinterpret_cast<void*>(&len),
                      reinterpret_cast<void const*>(&prefetch_buffer[prefetch_read_index]),
                      sizeof(int32_t));

    // Genuine corruption: the length prefix is not a valid in-page size. This covers a length
    // that claims more bytes than remain in the page, and a negative length (never valid in PLAIN,
    // e.g. from a stray byte shifting the length window). The sum is widened to 64 bits so a
    // near-INT_MAX corrupt length cannot overflow the comparison itself (int32 overflow is UB).
    if (len < 0 || static_cast<int64_t>(string_offset) + len > dict_size) {
      num_values_written = pos;  // Data is corrupted or incomplete
      cg::invoke_one(block, [&]() {
        set_error(static_cast<kernel_error::value_type>(decode_error::STRING_DATA_OVERRUN),
                  error_code);
      });
      break;
    }
    next_length_offset = string_offset + len;

    cg::invoke_one(block, [&]() { str_offsets[pos] = string_offset; });
  }

  // +4 for "stored" length of "next" string that we'll subtract off during decode
  // Easier/faster than branching in the decode loop
  int32_t const last_string_offset = next_length_offset + sizeof(int32_t);

  // Use all threads in the block to fill remaining entries with the last offset
  // This fills in the rest if we break early above because the dictionary wasn't large enough
  // But it also fills in the last offset for the page, hence the +1 in the loop limit.
  for (size_t pos = num_values_written + t; pos < num_values_to_process + 1; pos += block_size) {
    str_offsets[pos] = last_string_offset;
  }
}

/**
 * @brief Read string offsets without buffering for large average string lengths
 *
 * This function handles the case where the average string length is too large
 * to use the buffering approach. Thread 0 sequentially reads all string lengths,
 * then all threads cooperatively fill the remaining entries.
 *
 * @param s Page state containing data_start, dict_size and other info
 * @param num_values_to_process Number of values to process
 * @param str_offsets Output buffer for string offsets
 * @param error_code Error code to set if a string length overruns the page
 */
template <int32_t block_size>
inline __device__ void read_string_offsets_sequential(auto* s,
                                                      size_t num_values_to_process,
                                                      uint32_t* str_offsets,
                                                      kernel_error::pointer error_code)
{
  auto const block = cg::this_thread_block();
  int const t      = block.thread_rank();

  __shared__ size_t num_values_written;
  __shared__ uint32_t last_offset;  // offset into data_start

  cg::invoke_one(block, [&]() {
    uint8_t const* cur     = s->stream.data_start;
    uint32_t length_offset = 0;
    auto const dict_size   = s->stream.dict_size;

    // Process the data
    // Parquet data is: 4-byte length, string, 4-byte length, string, ...
    num_values_written = num_values_to_process;  // Will update below if run out of data
    for (size_t pos = 0; pos < num_values_to_process; pos++) {
      uint32_t const string_offset = length_offset + sizeof(int32_t);
      // Natural end-of-data exit: we've consumed all the value bytes, so the next length prefix
      // would begin at or past the end of the data -- it doesn't exist. Nothing left to read.
      if (string_offset > dict_size) {
        num_values_written = pos;  // Reached the end of the data
        break;
      }

      // Read the length of the string from the data stream
      int32_t len;
      cuda::std::memcpy(reinterpret_cast<void*>(&len),
                        reinterpret_cast<void const*>(&cur[length_offset]),
                        sizeof(int32_t));

      // Genuine corruption: the length prefix is not a valid in-page size. This covers a length
      // that claims more bytes than remain in the page, and a negative length (never valid in
      // PLAIN, e.g. from a stray byte shifting the length window). The sum is widened to 64 bits so
      // a near-INT_MAX corrupt length cannot overflow the comparison itself (int32 overflow is UB).
      if (len < 0 || static_cast<int64_t>(string_offset) + len > dict_size) {
        num_values_written = pos;  // Data is corrupted or incomplete
        set_error(static_cast<kernel_error::value_type>(decode_error::STRING_DATA_OVERRUN),
                  error_code);
        break;
      }
      str_offsets[pos] = string_offset;
      length_offset    = string_offset + len;
    }

    // +4 for "stored" length of "next" string that we'll subtract off during decode
    last_offset = length_offset + sizeof(int32_t);
  });

  block.sync();  // Ensure all threads see num_values_written

  // Use all threads in the block to fill remaining entries with the last offset
  // This fills in the rest if we break early above because the dictionary wasn't large enough
  // But it also fills in the last offset for the page, hence the +1 in the loop limit.
  for (size_t pos = num_values_written + t; pos < num_values_to_process + 1; pos += block_size) {
    str_offsets[pos] = last_offset;
  }
}

/**
 * @brief Pre-processing kernel to fill string offsets for non-dictionary string columns
 *
 * This kernel runs before the main decode kernel to pre-compute string offsets
 * for columns that use plain encoding without dictionaries.
 *
 * @param pages List of pages
 * @param chunks List of column chunks
 * @param page_string_offset_indices Device span of offsets, indexed per-page, into the column's
 * string offset buffer
 * @param page_mask Boolean vector indicating which pages need to be processed
 * @param min_row Minimum row index to read
 * @param num_rows Number of rows to read starting from min_row
 * @param error_code Error code to set if string data is corrupted
 */
template <int decode_block_size, size_t prefetch_size>
CUDF_KERNEL void preprocess_string_offsets_kernel(
  PageInfo* pages,
  device_span<ColumnChunkDesc const> chunks,
  device_span<size_t const> page_string_offset_indices,
  cudf::device_span<bool const> page_mask,
  size_t min_row,
  size_t num_rows,
  kernel_error::pointer error_code)
{
  int const page_idx = cg::this_grid().block_rank();
  PageInfo* const pp = &pages[page_idx];

  // Don't process pages that don't need to be decoded
  if (not page_mask.empty() and not page_mask[page_idx]) { return; }

  auto const& chunk = chunks[pp->chunk_idx];
  if (chunk.physical_type == Type::FIXED_LEN_BYTE_ARRAY) {
    return;  // String lengths, and thus offsets are fixed, no need to preprocess
  }

  // Check if this is a string column without dictionary using kernel mask
  constexpr uint32_t STRINGS_MASK_NON_DELTA_NON_DICT =
    BitOr(decode_kernel_mask::STRING,
          decode_kernel_mask::STRING_NESTED,
          decode_kernel_mask::STRING_LIST,
          decode_kernel_mask::STRING_STREAM_SPLIT,
          decode_kernel_mask::STRING_STREAM_SPLIT_NESTED,
          decode_kernel_mask::STRING_STREAM_SPLIT_LIST);

  __shared__ __align__(16) string_offset_scan_state state_g;
  auto* const s = &state_g;
  if (!setup_local_page_info(s,
                             pp,
                             chunks,
                             min_row,
                             num_rows,
                             mask_filter{STRINGS_MASK_NON_DELTA_NON_DICT},
                             page_processing_stage::STRING_BOUNDS)) {
    return;
  }

  // Determine if this is a list column and how many values to process
  // We don't know how many values we'll need to read, because we don't know
  // how many nulls we'll skip. So we have to read through the skipped rows.
  // This runs before we know skipped_leaf_values so can't skip for lists either.
  bool const is_list                 = (chunk.max_level[level_type::REPETITION] != 0);
  size_t const num_values_to_process = is_list ? pp->nesting[chunk.max_nesting_depth - 1].batch_size
                                               : s->setup.num_rows + s->setup.first_row;

  if (num_values_to_process == 0) { return; }

  uint32_t* const str_offsets =
    chunk.column_string_offset_base + page_string_offset_indices[page_idx];

  // If the average string length is small, all of the threads in the warp can iteratively prefetch
  // blocks of data containing many string lengths into shared memory.
  // However, if the average string length is large, we'll spend too much time copying raw string
  // data we don't need: have a single thread read the string lengths sequentially.
  constexpr int max_avg_string_length_for_buffer = prefetch_size / 16;  // for 1024 buffer, is 64
  auto const avg_string_length = s->stream.dict_size / pp->num_input_values - sizeof(int32_t);

  if (avg_string_length > max_avg_string_length_for_buffer) {
    // Use sequential processing for large average string lengths
    read_string_offsets_sequential<decode_block_size>(
      s, num_values_to_process, str_offsets, error_code);
  } else {
    // Use buffered processing for typical string lengths
    read_string_offsets_buffered<decode_block_size, prefetch_size>(
      s, num_values_to_process, str_offsets, error_code);
  }
}

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
                               rmm::cuda_stream_view stream)
{
  if (pages.size() == 0) { return; }

  constexpr int preprocess_block_size = cudf::detail::warp_size;
  constexpr int prefetch_size         = 1024;

  dim3 dim_block(preprocess_block_size, 1);
  dim3 dim_grid(pages.size(), 1);  // 1 threadblock per page

  preprocess_string_offsets_kernel<preprocess_block_size, prefetch_size>
    <<<dim_grid, dim_block, 0, stream.value()>>>(pages.device_ptr(),
                                                 chunks,
                                                 page_string_offset_indices,
                                                 page_mask,
                                                 min_row,
                                                 num_rows,
                                                 error_code);
  CUDF_CUDA_TRY(cudaGetLastError());
}

}  // namespace cudf::io::parquet::detail
