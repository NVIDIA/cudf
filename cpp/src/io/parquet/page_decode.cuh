/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "error.hpp"
#include "io/parquet/timestamp_utils.cuh"
#include "io/utilities/block_utils.cuh"
#include "parquet_gpu.hpp"
#include "rle_stream.cuh"

#include <cooperative_groups.h>
#include <cuda/atomic>
#include <cuda/std/tuple>

namespace cudf::io::parquet::detail {

namespace cg = cooperative_groups;

enum class copy_mode : bool { INDIRECT, DIRECT };

struct page_decode_setup_state {
  PageInfo page{};
  ColumnChunkDesc col{};
  int32_t first_row{};         // First row in page to output
  int32_t num_rows{};          // Rows in page to decode (including rows to be skipped)
  int32_t num_input_values{};  // total # of input/level values in the page
  kernel_error::value_type error{};
};

// Data-source + dictionary + level byte-range state used while consuming a page's
// encoded byte streams. Grouped so passes that only need to advance streams can
// share exactly this subset (see page_state_composed.cuh).
struct page_decode_stream_state {
  uint8_t const* data_start{};
  uint8_t const* data_end{};
  uint8_t const* abs_lvl_start[NUM_LEVEL_TYPES]{};
  uint8_t const* abs_lvl_end[NUM_LEVEL_TYPES]{};
  uint32_t initial_rle_run[NUM_LEVEL_TYPES]{};
  int32_t initial_rle_value[NUM_LEVEL_TYPES]{};
  uint8_t const* dict_base{};
  int32_t dict_size{};
  int32_t dict_bits{};
  uint32_t dict_run{};
  int32_t dict_val{};
};

struct page_state_s {
  CUDF_HOST_DEVICE constexpr page_state_s() noexcept {}
  page_decode_setup_state setup{};
  page_decode_stream_state stream{};
  int32_t dtype_len{};     // Output data type length
  int32_t dtype_len_in{};  // Can be larger than dtype_len if truncating 32-bit into 8-bit

  // (leaf) value decoding
  int32_t nz_count{};  // number of valid entries in nz_idx (write position in circular buffer)
  int32_t dict_pos{};  // write position of dictionary indices
  int32_t src_pos{};   // input read position of final output value
  int32_t ts_scale{};  // timestamp scale: <0: divide by -ts_scale, >0: multiply by ts_scale

  // repetition/definition level decoding
  int32_t input_value_count{};      // how many values of the input we've processed
  int32_t input_row_count{};        // how many rows of the input we've processed
  int32_t input_leaf_count{};       // how many leaf values of the input we've processed
  int32_t row_index_lower_bound{};  // lower bound of row indices we should process

  // a shared-memory cache of frequently used data when decoding. The source of this data is
  // normally stored in global memory which can yield poor performance. So, when possible
  // we copy that info here prior to decoding
  PageNestingDecodeInfo nesting_decode_cache[max_cacheable_nesting_decode_info]{};
  // points to either nesting_decode_cache above when possible, or to the global source otherwise
  PageNestingDecodeInfo* nesting_info{};

  inline __device__ void set_error_code(decode_error err)
  {
    cuda::atomic_ref<kernel_error::value_type, cuda::thread_scope_block> ref{setup.error};
    ref.fetch_or(static_cast<kernel_error::value_type>(err), cuda::std::memory_order_relaxed);
  }

  inline __device__ void reset_error_code()
  {
    cuda::atomic_ref<kernel_error::value_type, cuda::thread_scope_block> ref{setup.error};
    ref.store(0, cuda::std::memory_order_release);
  }
};

// buffers only used in the decode kernel.  separated from page_state_s to keep
// shared memory usage in other kernels (eg, compute_page_sizes_kernel) down.
template <int _nz_buf_size, int _dict_buf_size, int _str_buf_size>
struct page_state_buffers_s {
  static constexpr int nz_buf_size   = _nz_buf_size;
  static constexpr int dict_buf_size = _dict_buf_size;
  static constexpr int str_buf_size  = _str_buf_size;

  uint32_t nz_idx[nz_buf_size];      // circular buffer of non-null value positions
  uint32_t dict_idx[dict_buf_size];  // Dictionary index, boolean, or string offset values
  uint32_t str_len[str_buf_size];    // String length for plain encoding of strings
};

// Copies null counts back to `nesting_decode` at the end of scope
struct null_count_back_copier {
  page_state_s* s;
  int t;
  __device__ ~null_count_back_copier()
  {
    if (s->nesting_info != nullptr and s->nesting_info == s->nesting_decode_cache) {
      int depth = 0;
      while (depth < s->setup.page.num_output_nesting_levels) {
        int const thread_depth = depth + t;
        if (thread_depth < s->setup.page.num_output_nesting_levels) {
          s->setup.page.nesting_decode[thread_depth].null_count =
            s->nesting_decode_cache[thread_depth].null_count;
        }
        depth += blockDim.x;
      }
    }
  }
};

/**
 * @brief Check if the page is marked nullable or not
 *
 * @param s Page state
 * @return True if the page is nullable (max definition level > 0)
 */
__device__ inline bool is_nullable(auto* s)
{
  auto const lvl           = level_type::DEFINITION;
  auto const max_def_level = s->setup.col.max_level[lvl];
  return max_def_level > 0;
}

/**
 * @brief For a nullable page, check if it could have nulls
 *
 * This function performs a quick check based on the initial RLE run to determine
 * if the page could potentially contain null values. It returns true if:
 * - The initial run is a literal run (could contain mixed values)
 * - The repeated run count doesn't match the page's input value count
 * - The repeated run value indicates nulls (not equal to max definition level)
 *
 * @param s Page state
 * @return True if the page could contain null values
 */
__device__ inline bool maybe_has_nulls(auto* s)
{
  auto const lvl      = level_type::DEFINITION;
  auto const init_run = s->stream.initial_rle_run[lvl];
  // literal runs, lets assume they could hold nulls
  if (is_literal_run(init_run)) { return true; }

  // repeated run with number of items in the run not equal
  // to the rows in the page, assume that means we could have nulls
  if (s->setup.page.num_input_values != (init_run >> 1)) { return true; }

  auto const lvl_bits = s->setup.col.level_bits[lvl];
  auto const run_val  = lvl_bits == 0 ? 0 : s->stream.initial_rle_value[lvl];

  // the encoded repeated value isn't valid, we have (all) nulls
  return run_val != s->setup.col.max_level[lvl];
}

/**
 * @brief Check if the page should process nulls
 *
 * @param s Page state
 * @return True if the page should process nulls
 */
__device__ inline bool should_process_nulls(auto* s)
{
  return is_nullable(s) && maybe_has_nulls(s);
}

/**
 * @brief Test if the given page is in a string column
 */
__device__ constexpr bool is_string_col(PageInfo const& page,
                                        device_span<ColumnChunkDesc const> chunks)
{
  if ((page.flags & PAGEINFO_FLAGS_DICTIONARY) != 0) { return false; }
  auto const& col = chunks[page.chunk_idx];
  return is_string_col(col);
}

/**
 * @brief Returns whether or not a page spans either the beginning or the end of the
 * specified row bounds
 *
 * @param page The page to be checked
 * @param chunk_start_row Absolute row index of the first row in the page's column chunk
 * @param start_row The starting row index
 * @param num_rows The number of rows
 * @param has_repetition True if the schema has nesting
 *
 * @return True if the page spans the beginning or the end of the row bounds
 */
inline __device__ bool is_bounds_page(PageInfo const& page,
                                      size_t chunk_start_row,
                                      size_t start_row,
                                      size_t num_rows,
                                      bool has_repetition)
{
  size_t const page_begin = chunk_start_row + page.chunk_row;
  size_t const page_end   = page_begin + page.num_rows;
  size_t const begin      = start_row;
  size_t const end        = start_row + num_rows;

  // Test for list schemas.
  auto const is_bounds_page_lists =
    ((page_begin <= begin and page_end >= begin) or (page_begin <= end and page_end >= end));

  // For non-list schemas, rows cannot span pages, so use a more restrictive test. Make sure to
  // relax the test for `page_end` if we adjusted the `num_rows` for the last page to compensate
  // for list row size estimates in `generate_list_column_row_count_estimates()` when chunked
  // read mode.
  auto const test_page_end_nonlists = page.is_num_rows_adjusted ? page_end >= end : page_end > end;

  auto const is_bounds_page_nonlists =
    (page_begin < begin and page_end > begin) or (page_begin < end and test_page_end_nonlists);

  return has_repetition ? is_bounds_page_lists : is_bounds_page_nonlists;
}

/**
 * @brief Returns whether or not a page is completely contained within the specified
 * row bounds
 *
 * @param page The page to be checked
 * @param chunk_start_row Absolute row index of the first row in the page's column chunk
 * @param start_row The starting row index
 * @param num_rows The number of rows
 *
 * @return True if the page is completely contained within the row bounds
 */
inline __device__ bool is_page_contained(PageInfo const& page,
                                         size_t chunk_start_row,
                                         size_t start_row,
                                         size_t num_rows)
{
  size_t const page_begin = chunk_start_row + page.chunk_row;
  size_t const page_end   = page_begin + page.num_rows;
  size_t const begin      = start_row;
  size_t const end        = start_row + num_rows;

  return page_begin >= begin && page_end <= end;
}

/**
 * @brief Determine whether a page contains work to do for the requested row bounds.
 *
 * A page normally has work to do when its row range [page_start_row, page_start_row +
 * page_num_rows) intersects the requested range [min_row, min_row + num_rows).
 *
 * For list schemas a single row can span multiple pages, so a page may legitimately
 * carry values while containing zero of its own rows. Such a page must still be processed when
 * it spans (is a "bounds" page for) or is fully contained within the requested range.
 *
 * @param page The page to be checked
 * @param chunk_start_row Absolute row index of the first row in the page's column chunk
 * @param min_row Absolute index of the first requested row
 * @param num_rows Number of requested rows
 * @param has_repetition True if the schema has nesting (list) columns
 *
 * @return True if the page has rows/values to process for the requested range
 */
inline __device__ bool page_has_rows_to_process(PageInfo const& page,
                                                size_t chunk_start_row,
                                                size_t min_row,
                                                size_t num_rows,
                                                bool has_repetition)
{
  size_t const page_start_row = chunk_start_row + page.chunk_row;
  size_t const page_end_row   = page_start_row + page.num_rows;
  size_t const end_row        = min_row + num_rows;

  // A page has rows to read when its row range intersects the requested range.
  bool const has_rows =
    (page.num_rows > 0) && (page_start_row < end_row) && (page_end_row > min_row);
  if (has_rows || !has_repetition) { return has_rows; }

  // A single list row can span pages, so a list page can carry values (and offsets) with 0 rows;
  // such a page carries no rows of its own but must still be processed.
  return is_bounds_page(page, chunk_start_row, min_row, num_rows, has_repetition) ||
         is_page_contained(page, chunk_start_row, min_row, num_rows);
}

/**
 * @brief Retrieves string information for a string at the specified source position
 *
 * @param[in] s Page state input
 * @param[out] sb Page state buffer output
 * @param[in] src_pos Source position
 * @tparam state_buf Typename of the `state_buf` (usually inferred)
 *
 * @return A pair containing a pointer to the string and its length
 */
template <typename state_buf>
inline __device__ string_index_pair gpuGetStringData(page_state_s* s, state_buf* sb, int src_pos)
{
  char const* ptr = nullptr;
  using len_type  = cuda::std::tuple_element<1, string_index_pair>::type;
  len_type len    = 0;

  if (s->stream.dict_base) {
    // String dictionary
    uint32_t dict_pos =
      (s->stream.dict_bits > 0)
        ? sb->dict_idx[rolling_index<state_buf::dict_buf_size>(src_pos)] * sizeof(string_index_pair)
        : 0;
    if (dict_pos < (uint32_t)s->stream.dict_size) {
      return *reinterpret_cast<string_index_pair const*>(s->stream.dict_base + dict_pos);
    }
  } else {
    // Plain encoding
    uint32_t dict_pos = sb->dict_idx[rolling_index<state_buf::dict_buf_size>(src_pos)];
    if (dict_pos <= (uint32_t)s->stream.dict_size) {
      ptr = reinterpret_cast<char const*>(s->stream.data_start + dict_pos);
      len = sb->str_len[rolling_index<state_buf::str_buf_size>(src_pos)];
    }
  }

  return {ptr, len};
}

/**
 * @brief Indicates if the string descriptor initializer kernel is only calculating sizes
 */
enum class is_calc_sizes_only : bool { NO = false, YES = true };

/**
 * @brief Performs RLE decoding of dictionary indexes
 *
 * @param[in,out] s Page state input/output
 * @param[out] sb Page state buffer output
 * @param[in] target_pos Target index position in dict_idx buffer (may exceed this value by up to
 * 31)
 * @param[in] warp Warp cooperative group
 * @tparam sizes_only Indicates if only sizes are to be calculated
 * @tparam state_buf Typename of the `state_buf` (usually inferred)
 *
 * @return A pair containing the new output position, and the total length of strings decoded (this
 * will only be valid on thread 0 and if sizes_only is true). In the event that this function
 * decodes strings beyond target_pos, the total length of strings returned will include these
 * additional values.
 */
template <is_calc_sizes_only sizes_only, typename state_buf>
__device__ cuda::std::pair<int, int> decode_dictionary_indices(
  page_state_s* s,
  [[maybe_unused]] state_buf* sb,
  int target_pos,
  cg::thread_block_tile<cudf::detail::warp_size, cg::thread_block> const& warp)
{
  uint8_t const* end = s->stream.data_end;
  int dict_bits      = s->stream.dict_bits;
  int pos            = s->dict_pos;
  int str_len        = 0;
  int const t        = warp.thread_rank();

  // NOTE: racecheck warns about a RAW involving s->dict_pos, which is likely a false
  // positive because the only path that does not include a sync will lead to
  // s->dict_pos being overwritten with the same value

  while (pos < target_pos) {
    int is_literal, batch_len;
    if (t == 0) {
      uint32_t run       = s->stream.dict_run;
      uint8_t const* cur = s->stream.data_start;
      if (run <= 1) {
        run = (cur < end) ? get_vlq32(cur, end) : 0;
        if (!(run & 1)) {
          // Repeated value
          int bytecnt = (dict_bits + 7) >> 3;
          if (cur + bytecnt <= end) {
            int32_t run_val = cur[0];
            if (bytecnt > 1) {
              run_val |= cur[1] << 8;
              if (bytecnt > 2) {
                run_val |= cur[2] << 16;
                if (bytecnt > 3) { run_val |= cur[3] << 24; }
              }
            }
            s->stream.dict_val = run_val & ((1 << dict_bits) - 1);
          }
          cur += bytecnt;
        }
      }
      if (run & 1) {
        // Literal batch: must output a multiple of 8, except for the last batch
        int batch_len_div8;
        batch_len      = max(min(32, (int)(run >> 1) * 8), 1);
        batch_len_div8 = (batch_len + 7) >> 3;
        run -= batch_len_div8 * 2;
        cur += batch_len_div8 * dict_bits;
      } else {
        batch_len = max(min(32, (int)(run >> 1)), 1);
        run -= batch_len * 2;
      }
      s->stream.dict_run   = run;
      s->stream.data_start = cur;
      is_literal           = run & 1;
      __threadfence_block();
    }
    warp.sync();
    is_literal = shuffle(is_literal);
    batch_len  = shuffle(batch_len);

    // compute dictionary index.
    int dict_idx = 0;
    if (t < batch_len) {
      dict_idx = s->stream.dict_val;
      if (is_literal) {
        int32_t ofs      = (t - ((batch_len + 7) & ~7)) * dict_bits;
        uint8_t const* p = s->stream.data_start + (ofs >> 3);
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
      }

      // if we're not computing sizes, store off the dictionary index
      if constexpr (sizes_only == is_calc_sizes_only::NO) {
        sb->dict_idx[rolling_index<state_buf::dict_buf_size>(pos + t)] = dict_idx;
      }
    }

    // if we're computing sizes, add the length(s)
    if constexpr (sizes_only == is_calc_sizes_only::YES) {
      int const len = [&]() {
        if (t >= batch_len || (pos + t >= target_pos)) { return 0; }
        uint32_t const dict_pos =
          (s->stream.dict_bits > 0) ? dict_idx * sizeof(string_index_pair) : 0;
        if (dict_pos < (uint32_t)s->stream.dict_size) {
          auto const* src =
            reinterpret_cast<string_index_pair const*>(s->stream.dict_base + dict_pos);
          return src->second;
        }
        return 0;
      }();

      using WarpReduce = cub::WarpReduce<size_type>;
      __shared__ typename WarpReduce::TempStorage temp_storage;
      // note: str_len will only be valid on thread 0.
      str_len += WarpReduce(temp_storage).Sum(len);
    }

    pos += batch_len;
  }
  return {pos, str_len};
}

/**
 * @brief Performs RLE decoding of dictionary indexes, for when dict_size=1
 *
 * @param[in,out] s Page state input/output
 * @param[out] sb Page state buffer output
 * @param[in] target_pos Target write position
 * @param[in] warp Warp cooperative group
 * @tparam state_buf Typename of the `state_buf` (usually inferred)
 *
 * @return The new output position
 */
template <typename state_buf>
inline __device__ int decode_rle_booleans(
  page_state_s* s,
  state_buf* sb,
  int target_pos,
  cg::thread_block_tile<cudf::detail::warp_size, cg::thread_block> const& warp)
{
  uint8_t const* end = s->stream.data_end;
  int64_t pos        = s->dict_pos;
  int const t        = warp.thread_rank();

  // NOTE: racecheck warns about a RAW involving s->dict_pos, which is likely a false positive
  // because the only path that does not include a sync will lead to s->dict_pos being overwritten
  // with the same value

  while (pos < target_pos) {
    int is_literal, batch_len;
    if (t == 0) {
      uint32_t run       = s->stream.dict_run;
      uint8_t const* cur = s->stream.data_start;
      if (run <= 1) {
        run = (cur < end) ? get_vlq32(cur, end) : 0;
        if (!(run & 1)) {
          // Repeated value
          s->stream.dict_val = (cur < end) ? cur[0] & 1 : 0;
          cur++;
        }
      }
      if (run & 1) {
        // Literal batch: must output a multiple of 8, except for the last batch
        int batch_len_div8;
        batch_len = max(min(32, (int)(run >> 1) * 8), 1);
        if (batch_len >= 8) { batch_len &= ~7; }
        batch_len_div8 = (batch_len + 7) >> 3;
        run -= batch_len_div8 * 2;
        cur += batch_len_div8;
      } else {
        batch_len = max(min(32, (int)(run >> 1)), 1);
        run -= batch_len * 2;
      }
      s->stream.dict_run   = run;
      s->stream.data_start = cur;
      is_literal           = run & 1;
      __threadfence_block();
    }

    warp.sync();
    is_literal = shuffle(is_literal);
    batch_len  = shuffle(batch_len);

    if (t < batch_len) {
      int dict_idx;
      if (is_literal) {
        int32_t ofs      = t - ((batch_len + 7) & ~7);
        uint8_t const* p = s->stream.data_start + (ofs >> 3);
        dict_idx         = (p < end) ? (p[0] >> (ofs & 7u)) & 1 : 0;
      } else {
        dict_idx = s->stream.dict_val;
      }
      sb->dict_idx[rolling_index<state_buf::dict_buf_size>(pos + t)] = dict_idx;
    }
    pos += batch_len;
  }
  return pos;
}

/**
 * @brief Parses the length and position of strings and returns total length of all strings
 * processed
 *
 * @param[in,out] s Page state input/output
 * @param[out] sb Page state buffer output
 * @param[in] target_pos Target output position
 * @param[in] group Cooperative group (thread block or tile)
 * @tparam sizes_only Indicates if only sizes are to be calculated
 * @tparam state_buf Typename of the `state_buf` (usually inferred)
 * @tparam thread_group Typename of the cooperative group (inferred)
 *
 * @return Total length of strings processed
 */
template <is_calc_sizes_only sizes_only, typename state_buf, typename thread_group>
__device__ size_type initialize_string_descriptors(page_state_s* s,
                                                   [[maybe_unused]] state_buf* sb,
                                                   int target_pos,
                                                   thread_group const& group)
{
  int const t         = group.thread_rank();
  int const dict_size = s->stream.dict_size;
  int k               = s->stream.dict_val;
  int pos             = s->dict_pos;
  int total_len       = 0;

  // All group threads can participate for fixed len byte arrays.
  if (s->setup.col.physical_type == Type::FIXED_LEN_BYTE_ARRAY) {
    int const dtype_len_in = s->dtype_len_in;
    total_len              = min((target_pos - pos) * dtype_len_in, dict_size - s->stream.dict_val);
    if constexpr (sizes_only == is_calc_sizes_only::NO) {
      for (pos += t, k += t * dtype_len_in; pos < target_pos; pos += group.size()) {
        sb->str_len[rolling_index<state_buf::str_buf_size>(pos)] =
          (k < dict_size) ? dtype_len_in : 0;
        // dict_idx is upperbounded by dict_size.
        sb->dict_idx[rolling_index<state_buf::dict_buf_size>(pos)] = k;
        // Increment k if needed.
        if (k < dict_size) { k = min(k + (group.size() * dtype_len_in), dict_size); }
      }
    }
    // Only thread_rank = 0 updates the s->stream.dict_val
    if (t == 0) { s->stream.dict_val += total_len; }
  }
  // This step is purely serial for byte arrays
  else {
    if (t == 0) {
      uint8_t const* cur = s->stream.data_start;

      for (int len = 0; pos < target_pos; pos++, len = 0) {
        if (k + 4 <= dict_size) {
          len = (cur[k]) | (cur[k + 1] << 8) | (cur[k + 2] << 16) | (cur[k + 3] << 24);
          k += 4;
          if (k + len > dict_size) { len = 0; }
        }
        if constexpr (sizes_only == is_calc_sizes_only::NO) {
          sb->dict_idx[rolling_index<state_buf::dict_buf_size>(pos)] = k;
          sb->str_len[rolling_index<state_buf::str_buf_size>(pos)]   = len;
        }
        k += len;
        total_len += len;
      }
      s->stream.dict_val = k;
    }
  }

  return total_len;
}

/**
 * @brief Store a validity mask containing value_count bits into the output validity buffer of the
 * page.
 *
 * @param[in] valid_map_offset The offset into the validity map at which to store bits
 * @param[in,out] valid_map Pointer to bitmask to store validity information to
 * @param[in] valid_mask The validity mask to be stored
 * @param[in] value_count # of bits in the validity mask
 */
inline __device__ void store_validity(int valid_map_offset,
                                      bitmask_type* valid_map,
                                      uint32_t valid_mask,
                                      int32_t value_count)
{
  int word_offset = valid_map_offset / 32;
  int bit_offset  = valid_map_offset % 32;
  // if we fit entirely in the output word
  if (bit_offset + value_count <= 32) {
    auto relevant_mask = static_cast<uint32_t>((static_cast<uint64_t>(1) << value_count) - 1);

    if (relevant_mask == ~0) {
      valid_map[word_offset] = valid_mask;
    } else {
      auto validity_map_word_ref =
        cuda::atomic_ref<bitmask_type, cuda::thread_scope_device>(valid_map[word_offset]);
      // It's okay to fetch_and and fetch_or in separate transactions here as each thread modifies a
      // different set of bits within the word
      validity_map_word_ref.fetch_and(~(relevant_mask << bit_offset),
                                      cuda::std::memory_order_relaxed);
      validity_map_word_ref.fetch_or((valid_mask & relevant_mask) << bit_offset,
                                     cuda::std::memory_order_relaxed);
    }
  }
  // we're going to spill over into the next word.
  // note : writing both values here is the lazy/slow way.  we could be writing just
  // the first word and rolling the remaining bits over into the next call.
  // however, some basic performance tests shows almost no difference between these two
  // methods. More detailed performance testing might be worthwhile here.
  else {
    uint32_t bits_left = 32 - bit_offset;

    // first word. strip bits_left bits off the beginning and store that
    uint32_t relevant_mask = ((1 << bits_left) - 1);
    uint32_t mask_word0    = valid_mask & relevant_mask;
    auto validity_map_first_word_ref =
      cuda::atomic_ref<bitmask_type, cuda::thread_scope_device>(valid_map[word_offset]);
    // It's okay to fetch_and and fetch_or in separate transactions here as each thread modifies a
    // different set of bits within the word
    validity_map_first_word_ref.fetch_and(~(relevant_mask << bit_offset),
                                          cuda::std::memory_order_relaxed);
    validity_map_first_word_ref.fetch_or(mask_word0 << bit_offset, cuda::std::memory_order_relaxed);

    // second word. strip the remainder of the bits off the end and store that
    auto validity_map_second_word_ref =
      cuda::atomic_ref<bitmask_type, cuda::thread_scope_device>(valid_map[word_offset + 1]);
    relevant_mask       = ((1 << (value_count - bits_left)) - 1);
    uint32_t mask_word1 = valid_mask & (relevant_mask << bits_left);
    // It's okay to fetch_and and fetch_or in separate transactions here as each thread modifies a
    // different set of bits within the word
    validity_map_second_word_ref.fetch_and(~(relevant_mask), cuda::std::memory_order_relaxed);
    validity_map_second_word_ref.fetch_or(mask_word1 >> bits_left, cuda::std::memory_order_relaxed);
  }
}

/**
 * @brief Compute the nesting bounds within the hierarchy to add values to, and the definition level
 * D to which we should considered them null or not.
 *
 * @param[out] start_depth The start nesting depth
 * @param[out] end_depth The end nesting depth (inclusive)
 * @param[out] d The definition level up to which added values are not-null. if t is out of bounds,
 * d will be -1
 * @param[in] s Local page information
 * @param[in] rep Repetition level buffer
 * @param[in] def Definition level buffer
 * @param[in] input_value_count The current count of input level values we have processed
 * @param[in] target_input_value_count The desired # of input level values we want to process
 * @param[in] t Thread index
 * @tparam level_t Type used to store decoded repetition and definition levels
 */
template <typename level_t>
inline __device__ void get_nesting_bounds(int& start_depth,
                                          int& end_depth,
                                          int& d,
                                          page_state_s* s,
                                          level_t const* const rep,
                                          level_t const* const def,
                                          int input_value_count,
                                          int32_t target_input_value_count,
                                          int t)
{
  start_depth = -1;
  end_depth   = -1;
  d           = -1;
  // Clamp to decoded level count; for chunked reads the level buffers may be
  // smaller than num_input_values when skip_rows/num_rows reduces the range.
  auto const actual_num_values =
    (s->setup.page.num_decoded_level_values > 0)
      ? cuda::std::min(s->setup.page.num_input_values, s->setup.page.num_decoded_level_values)
      : s->setup.page.num_input_values;
  auto const max_idx = cuda::std::min(target_input_value_count, actual_num_values);
  if (input_value_count + t < max_idx) {
    int const index = input_value_count + t;
    d = (def != nullptr) ? def[index] : s->setup.col.max_level[level_type::DEFINITION];

    // if we have repetition (there are list columns involved) we have to
    // bound what nesting levels we apply values to
    if (s->setup.col.max_level[level_type::REPETITION] > 0) {
      int const r = rep[index];
      start_depth = s->nesting_info[r].start_depth;
      end_depth   = s->nesting_info[d].end_depth;
    }
    // for columns without repetition (even ones involving structs) we always
    // traverse the entire hierarchy.
    else {
      start_depth = 0;
      end_depth   = s->setup.col.max_nesting_depth - 1;
    }
  }
}

/**
 * @brief Process a batch of incoming repetition/definition level values and generate
 *        validity, nested column offsets (where appropriate) and decoding indices.
 *
 * @param[in] target_input_value_count The # of repetition/definition levels to process up to
 * @param[in] s Local page information
 * @param[out] sb Page state buffer output
 * @param[in] rep Repetition level buffer
 * @param[in] def Definition level buffer
 * @param[in] t Thread index
 * @tparam level_t Type used to store decoded repetition and definition levels
 * @tparam state_buf Typename of the `state_buf` (usually inferred)
 * @tparam rolling_buf_size Size of the cyclic buffer used to store value data
 */
template <typename level_t, typename state_buf, int rolling_buf_size>
__device__ void gpuUpdateValidityOffsetsAndRowIndices(int32_t target_input_value_count,
                                                      page_state_s* s,
                                                      state_buf* sb,
                                                      level_t const* const rep,
                                                      level_t const* const def,
                                                      int t)
{
  // exit early if there's no work to do
  if (s->input_value_count >= target_input_value_count) { return; }

  // max nesting depth of the column
  int const max_depth       = s->setup.col.max_nesting_depth;
  bool const has_repetition = s->setup.col.max_level[level_type::REPETITION] > 0;
  // how many (input) values we've processed in the page so far
  int input_value_count = s->input_value_count;
  // how many rows we've processed in the page so far
  int input_row_count = s->input_row_count;

  PageNestingDecodeInfo* nesting_info_base = s->nesting_info;

  // process until we've reached the target
  while (input_value_count < target_input_value_count) {
    // determine the nesting bounds for this thread (the range of nesting depths we
    // will generate new value indices and validity bits for)
    int start_depth, end_depth, d;
    get_nesting_bounds<level_t>(
      start_depth, end_depth, d, s, rep, def, input_value_count, target_input_value_count, t);

    // 4 interesting things to track:
    // thread_value_count : # of output values from the view of this thread
    // warp_value_count   : # of output values for the whole warp
    //
    // thread_valid_count : # of valid values from the view of this thread
    // warp_valid_count   : # of valid values for the whole warp
    uint32_t thread_value_count, warp_value_count;
    uint32_t thread_valid_count, warp_valid_count;

    // track (page-relative) row index for the thread so we can compare against input bounds
    // keep track of overall # of rows we've read.
    int const is_new_row               = start_depth == 0 ? 1 : 0;
    uint32_t const warp_row_count_mask = ballot(is_new_row);
    int32_t const thread_row_index =
      input_row_count + ((__popc(warp_row_count_mask & ((1 << t) - 1)) + is_new_row) - 1);
    input_row_count += __popc(warp_row_count_mask);
    // is this thread within read row bounds?
    int const in_row_bounds = thread_row_index >= s->row_index_lower_bound &&
                                  thread_row_index < (s->setup.first_row + s->setup.num_rows)
                                ? 1
                                : 0;

    // compute warp and thread value counts
    uint32_t const warp_count_mask =
      ballot((0 >= start_depth && 0 <= end_depth) && in_row_bounds ? 1 : 0);

    warp_value_count = __popc(warp_count_mask);
    // Note : ((1 << t) - 1) implies "for all threads before me"
    thread_value_count = __popc(warp_count_mask & ((1 << t) - 1));

    // walk from 0 to max_depth
    uint32_t next_thread_value_count, next_warp_value_count;
    for (int s_idx = 0; s_idx < max_depth; s_idx++) {
      PageNestingDecodeInfo* nesting_info = &nesting_info_base[s_idx];

      // if we are within the range of nesting levels we should be adding value indices for
      int const in_nesting_bounds =
        ((s_idx >= start_depth && s_idx <= end_depth) && in_row_bounds) ? 1 : 0;

      // everything up to the max_def_level is a non-null value
      uint32_t const is_valid = d >= nesting_info->max_def_level && in_nesting_bounds ? 1 : 0;

      // compute warp and thread valid counts
      uint32_t const warp_valid_mask =
        // for flat schemas, a simple ballot_sync gives us the correct count and bit positions
        // because every value in the input matches to a value in the output
        !has_repetition
          ? ballot(is_valid)
          :
          // for nested schemas, it's more complicated.  This warp will visit 32 incoming values,
          // however not all of them will necessarily represent a value at this nesting level. so
          // the validity bit for thread t might actually represent output value t-6. the correct
          // position for thread t's bit is thread_value_count. for cuda 11 we could use
          // __reduce_or_sync(), but until then we have to do a warp reduce.
          warp_reduce_or<cudf::detail::warp_size>(is_valid << thread_value_count);

      thread_valid_count = __popc(warp_valid_mask & ((1 << thread_value_count) - 1));
      warp_valid_count   = __popc(warp_valid_mask);

      // if this is the value column emit an index for value decoding
      if (is_valid && s_idx == max_depth - 1) {
        int const src_pos = nesting_info->valid_count + thread_valid_count;
        int const dst_pos = nesting_info->value_count + thread_value_count;
        // nz_idx is a mapping of src buffer indices to destination buffer indices
        sb->nz_idx[rolling_index<rolling_buf_size>(src_pos)] = dst_pos;
      }

      // compute warp and thread value counts for the -next- nesting level. we need to
      // do this for nested schemas so that we can emit an offset for the -current- nesting
      // level. more concretely : the offset for the current nesting level == current length of the
      // next nesting level
      if (s_idx < max_depth - 1) {
        uint32_t const next_warp_count_mask =
          ballot((s_idx + 1 >= start_depth && s_idx + 1 <= end_depth && in_row_bounds) ? 1 : 0);
        next_warp_value_count   = __popc(next_warp_count_mask);
        next_thread_value_count = __popc(next_warp_count_mask & ((1 << t) - 1));

        // if we're -not- at a leaf column and we're within nesting/row bounds
        // and we have a valid data_out pointer, it implies this is a list column, so
        // emit an offset.
        if (in_nesting_bounds && nesting_info->data_out != nullptr) {
          int const idx             = nesting_info->value_count + thread_value_count;
          cudf::size_type const ofs = nesting_info_base[s_idx + 1].value_count +
                                      next_thread_value_count +
                                      nesting_info_base[s_idx + 1].page_start_value;
          (reinterpret_cast<cudf::size_type*>(nesting_info->data_out))[idx] = ofs;
        }
      }

      // nested schemas always read and write to the same bounds (that is, read and write positions
      // are already pre-bounded by first_row/num_rows). flat schemas will start reading at the
      // first value, even if that is before first_row, because we cannot trivially jump to
      // the correct position to start reading. since we are about to write the validity vector here
      // we need to adjust our computed mask to take into account the write row bounds.
      int const in_write_row_bounds =
        !has_repetition ? thread_row_index >= s->setup.first_row &&
                            thread_row_index < (s->setup.first_row + s->setup.num_rows)
                        : in_row_bounds;
      int const first_thread_in_write_range =
        !has_repetition ? __ffs(ballot(in_write_row_bounds)) - 1 : 0;

      // # of bits to of the validity mask to write out
      int const warp_valid_mask_bit_count =
        first_thread_in_write_range < 0 ? 0 : warp_value_count - first_thread_in_write_range;

      // increment count of valid values, count of total values, and update validity mask
      if (!t) {
        if (nesting_info->valid_map != nullptr && warp_valid_mask_bit_count > 0) {
          uint32_t const warp_output_valid_mask = warp_valid_mask >> first_thread_in_write_range;
          store_validity(nesting_info->valid_map_offset,
                         nesting_info->valid_map,
                         warp_output_valid_mask,
                         warp_valid_mask_bit_count);
          nesting_info->valid_map_offset += warp_valid_mask_bit_count;
          nesting_info->null_count += warp_valid_mask_bit_count - __popc(warp_output_valid_mask);
        }
        nesting_info->valid_count += warp_valid_count;
        nesting_info->value_count += warp_value_count;
      }

      // propagate value counts for the next level
      warp_value_count   = next_warp_value_count;
      thread_value_count = next_thread_value_count;
    }

    input_value_count += min(32, (target_input_value_count - input_value_count));
    __syncwarp();
  }

  // update
  if (!t) {
    // update valid value count for decoding and total # of values we've processed
    s->nz_count          = nesting_info_base[max_depth - 1].valid_count;
    s->input_value_count = input_value_count;
    s->input_row_count   = input_row_count;
  }
}

/**
 * @brief Process repetition and definition levels up to the target count of leaf values.
 *
 * In order to decode actual leaf values from the input stream, we need to generate the
 * list of non-null value positions (page_state_s::nz_idx). We do this by processing
 * the repetition and definition level streams.  This process also generates validity information,
 * and offset column values in the case of nested schemas. Because of the way the streams
 * are encoded, this function may generate slightly more than target_leaf_count.
 *
 * Only runs on 1 warp.
 *
 * @param[in] s The local page state
 * @param[out] sb Page state buffer output
 * @param[in] target_leaf_count Target count of non-null leaf values to generate indices for
 * @param[in] rep Repetition level buffer
 * @param[in] def Definition level buffer
 * @param[in] warp Warp cooperative group
 * @tparam rolling_buf_size Size of the cyclic buffer used to store value data
 * @tparam level_t Type used to store decoded repetition and definition levels
 * @tparam state_buf Typename of the `state_buf` (usually inferred)
 */
template <int rolling_buf_size, typename level_t, typename state_buf>
__device__ void gpuDecodeLevels(
  page_state_s* s,
  state_buf* sb,
  int32_t target_leaf_count,
  level_t* const rep,
  level_t* const def,
  cg::thread_block_tile<cudf::detail::warp_size, cg::thread_block> const& warp)
{
  auto cur_leaf_count = target_leaf_count;
  while (s->setup.error == 0 && s->nz_count < target_leaf_count &&
         s->input_value_count < s->setup.num_input_values) {
    // because the rep and def streams are encoded separately, we cannot request an exact
    // # of values to be decoded at once. we can only process the lowest # of decoded rep/def
    // levels we get.
    auto const actual_leaf_count = cuda::std::min(cur_leaf_count, s->setup.num_input_values);

    // process what we got back
    gpuUpdateValidityOffsetsAndRowIndices<level_t, state_buf, rolling_buf_size>(
      actual_leaf_count, s, sb, rep, def, warp.thread_rank());
    cur_leaf_count = actual_leaf_count + warp.size();
    warp.sync();
  }
}

/**
 * @brief Parse the beginning of the level section (definition or repetition),
 * initializes the initial RLE run & value, and returns the section length
 *
 * @param[in,out] s The page state
 * @param[in] cur The current data position
 * @param[in] end The end of the data
 * @param[in] lvl Enum indicating whether this is to initialize repetition or definition level data
 *
 * @return The length of the section
 */
inline __device__ uint32_t InitLevelSection(auto* s,
                                            uint8_t const* cur,
                                            uint8_t const* end,
                                            level_type lvl)
{
  int32_t len;
  int const level_bits = s->setup.col.level_bits[lvl];
  auto const encoding  = lvl == level_type::DEFINITION ? s->setup.page.definition_level_encoding
                                                       : s->setup.page.repetition_level_encoding;

  auto start = cur;

  auto init_rle = [s, lvl, level_bits](uint8_t const* cur, uint8_t const* end) {
    uint32_t const run             = get_vlq32(cur, end);
    s->stream.initial_rle_run[lvl] = run;
    if (!(run & 1)) {
      if (cur < end) {
        int v = cur[0];
        cur++;
        if (level_bits > 8) {
          v |= ((cur < end) ? cur[0] : 0) << 8;
          cur++;
        }
        s->stream.initial_rle_value[lvl] = v;
      } else {
        s->stream.initial_rle_value[lvl] = 0;
      }
    }

    if (cur > end) { s->set_error_code(decode_error::LEVEL_STREAM_OVERRUN); }
  };

  // this is a little redundant. if level_bits == 0, then nothing should be encoded
  // for the level, but some V2 files in the wild violate this and encode the data anyway.
  // thus we will handle V2 headers separately.
  if ((s->setup.page.flags & PAGEINFO_FLAGS_V2) != 0 && (len = s->setup.page.lvl_bytes[lvl]) != 0) {
    // V2 only uses RLE encoding so no need to check encoding
    s->stream.abs_lvl_start[lvl] = cur;
    init_rle(cur, cur + len);
  } else if (level_bits == 0) {
    len                              = 0;
    s->stream.initial_rle_run[lvl]   = s->setup.page.num_input_values * 2;  // repeated value
    s->stream.initial_rle_value[lvl] = 0;
    s->stream.abs_lvl_start[lvl]     = cur;
  } else if (encoding == Encoding::RLE) {  // V1 header with RLE encoding
    if (cur + 4 < end) {
      len = (cur[0]) + (cur[1] << 8) + (cur[2] << 16) + (cur[3] << 24);
      cur += 4;
      s->stream.abs_lvl_start[lvl] = cur;
      init_rle(cur, cur + len);
      // add back the 4 bytes for the length
      len += 4;
    } else {
      len = 0;
      s->set_error_code(decode_error::LEVEL_STREAM_OVERRUN);
    }
  } else if (encoding == Encoding::BIT_PACKED) {
    len = (s->setup.page.num_input_values * level_bits + 7) >> 3;
    s->stream.initial_rle_run[lvl] =
      ((s->setup.page.num_input_values + 7) >> 3) * 2 + 1;  // literal run
    s->stream.initial_rle_value[lvl] = 0;
    s->stream.abs_lvl_start[lvl]     = cur;
  } else {
    len = 0;
    s->set_error_code(decode_error::UNSUPPORTED_ENCODING);
  }

  s->stream.abs_lvl_end[lvl] = start + len;

  return static_cast<uint32_t>(len);
}

/**
 * @brief Functor for setup_local_page_info that always returns true.
 */
struct all_types_filter {
  __device__ inline bool operator()(PageInfo const& page) { return true; }
};

/**
 * @brief Functor for setup_local_page_info that takes a mask of allowed types.
 */
struct mask_filter {
  uint32_t mask;

  __device__ mask_filter(uint32_t m) : mask(m) {}
  __device__ mask_filter(decode_kernel_mask m) : mask(static_cast<uint32_t>(m)) {}

  __device__ inline bool operator()(PageInfo const& page)
  {
    return BitAnd(mask, page.kernel_mask) != 0;
  }
};

enum class page_processing_stage {
  PREPROCESS,
  STRING_BOUNDS,
  DECODE,
};

/**
 * @brief Sets up block-local page state information from the global pages.
 *
 * @param[in, out] s The local page state to be filled in
 * @param[in] p The global page to be copied from
 * @param[in] chunks The global list of chunks
 * @param[in] min_row Crop all rows below min_row
 * @param[in] num_rows Maximum number of rows to read
 * @param[in] filter Filtering function used to decide which pages to operate on
 * @param[in] stage What stage of the decoding process is this being called from
 * @tparam Filter Function that takes a PageInfo reference and returns true if the given page should
 * be operated on Currently only used by compute_page_sizes_kernel step)
 * @return True if this page should be processed further
 */
template <typename Filter>
inline __device__ bool setup_local_page_info(auto* const s,
                                             PageInfo const* p,
                                             device_span<ColumnChunkDesc const> chunks,
                                             size_t min_row,
                                             size_t num_rows,
                                             Filter filter,
                                             page_processing_stage stage)
{
  int t = threadIdx.x;

  // Fetch page info
  if (!t) {
    s->setup.page = *p;
    if constexpr (requires { s->nesting_info; }) { s->nesting_info = nullptr; }
    s->setup.col = chunks[s->setup.page.chunk_idx];
  }
  __syncthreads();

  // return false if this is a dictionary page or it does not pass the filter condition
  if ((s->setup.page.flags & PAGEINFO_FLAGS_DICTIONARY) != 0 || !filter(s->setup.page)) {
    return false;
  }

  // our starting row (absolute index) is
  // col.start_row == absolute row index
  // page.chunk-row == relative row index within the chunk
  size_t const page_start_row = s->setup.col.start_row + s->setup.page.chunk_row;

  // if we can use the nesting decode cache, set it up now
  auto const can_use_decode_cache =
    s->setup.page.nesting_info_size <= max_cacheable_nesting_decode_info;
  if constexpr (requires { s->nesting_decode_cache; }) {
    if (can_use_decode_cache) {
      int depth = 0;
      while (depth < s->setup.page.nesting_info_size) {
        int const thread_depth = depth + t;
        if (thread_depth < s->setup.page.nesting_info_size) {
          // these values need to be copied over from global
          s->nesting_decode_cache[thread_depth].max_def_level =
            s->setup.page.nesting_decode[thread_depth].max_def_level;
          s->nesting_decode_cache[thread_depth].page_start_value =
            s->setup.page.nesting_decode[thread_depth].page_start_value;
          s->nesting_decode_cache[thread_depth].start_depth =
            s->setup.page.nesting_decode[thread_depth].start_depth;
          s->nesting_decode_cache[thread_depth].end_depth =
            s->setup.page.nesting_decode[thread_depth].end_depth;
        }
        depth += blockDim.x;
      }
    }
  }

  if (!t) {
    if constexpr (requires { s->nesting_info; }) {
      s->nesting_info =
        can_use_decode_cache ? s->nesting_decode_cache : s->setup.page.nesting_decode;
    }

    // NOTE: s->setup.page.num_rows, s->setup.col.chunk_row, s->setup.first_row and
    // s->setup.num_rows will be invalid/bogus during first pass of the preprocess step for nested
    // types. this is ok because we ignore these values in that stage.
    auto const end_row = min_row + num_rows;

    // if we are totally outside the range of the input, do nothing
    auto const page_end_row = page_start_row + s->setup.page.num_rows;
    if ((page_start_row >= end_row) || (page_end_row <= min_row)) {
      s->setup.first_row = 0;
      s->setup.num_rows  = 0;
    }
    // otherwise
    else {
      s->setup.first_row       = page_start_row >= min_row ? 0 : min_row - page_start_row;
      auto const max_page_rows = s->setup.page.num_rows - s->setup.first_row;
      s->setup.num_rows        = (page_start_row + s->setup.first_row) + max_page_rows <= end_row
                                   ? max_page_rows
                                   : end_row - (page_start_row + s->setup.first_row);
    }
  }
  __syncthreads();

  // zero counts
  if constexpr (requires { s->nesting_info; }) {
    int depth = 0;
    while (depth < s->setup.page.num_output_nesting_levels) {
      int const thread_depth = depth + t;
      if (thread_depth < s->setup.page.num_output_nesting_levels) {
        s->nesting_info[thread_depth].valid_count = 0;
        s->nesting_info[thread_depth].value_count = 0;
        s->nesting_info[thread_depth].null_count  = 0;
      }
      depth += blockDim.x;
    }
  }
  __syncthreads();

  // if we have no work to do (eg, in a skip_rows/num_rows case) in this page.
  //
  // corner case: in the case of lists, we can have pages that contain "0" rows if the current row
  // starts before this page and ends after this page:
  //       P0        P1        P2
  //  |---------|---------|----------|
  //        ^------------------^
  //      row start           row end
  // P1 will contain 0 rows
  //
  // NOTE: this check needs to be done after the null counts have been zeroed out
  bool const has_repetition = s->setup.col.max_level[level_type::REPETITION] > 0;
  if ((stage == page_processing_stage::STRING_BOUNDS || stage == page_processing_stage::DECODE) &&
      !page_has_rows_to_process(
        s->setup.page, s->setup.col.start_row, min_row, num_rows, has_repetition)) {
    return false;
  }

  if (!t) {
    s->reset_error_code();

    // IMPORTANT : nested schemas can have 0 rows in a page but still have
    // values. The case is:
    // - On page N-1, the last row starts, with 2/6 values encoded
    // - On page N, the remaining 4/6 values are encoded, but there are no new rows.
    // if (s->setup.page.num_input_values > 0 && s->setup.page.num_rows > 0) {
    if (s->setup.page.num_input_values > 0) {
      uint8_t* cur = s->setup.page.page_data;
      uint8_t* end = cur + s->setup.page.uncompressed_page_size;
      if constexpr (requires { s->dtype_len; }) {
        s->ts_scale = 0;
        // Validate data type
        auto const data_type  = s->setup.col.physical_type;
        auto const is_decimal = s->setup.col.logical_type.has_value() and
                                s->setup.col.logical_type->type == LogicalType::DECIMAL;
        switch (data_type) {
          case Type::BOOLEAN:
            s->dtype_len = 1;  // Boolean are stored as 1 byte on the output
            break;
          case Type::INT32: [[fallthrough]];
          case Type::FLOAT: s->dtype_len = 4; break;
          case Type::INT64:
            if (s->setup.col.ts_clock_rate) {
              s->ts_scale =
                calc_timestamp_scale(s->setup.col.logical_type, s->setup.col.ts_clock_rate);
            }
            [[fallthrough]];
          case Type::DOUBLE: s->dtype_len = 8; break;
          case Type::INT96: s->dtype_len = 12; break;
          case Type::BYTE_ARRAY:
            if (is_decimal) {
              auto const decimal_precision = s->setup.col.logical_type->precision();
              s->dtype_len                 = [decimal_precision]() {
                if (decimal_precision <= MAX_DECIMAL32_PRECISION) {
                  return sizeof(int32_t);
                } else if (decimal_precision <= MAX_DECIMAL64_PRECISION) {
                  return sizeof(int64_t);
                } else {
                  return sizeof(__int128_t);
                }
              }();
            } else {
              s->dtype_len = sizeof(string_index_pair);
            }
            break;
          default:  // FIXED_LEN_BYTE_ARRAY:
            s->dtype_len = s->setup.col.type_length;
            if (s->dtype_len <= 0) { s->set_error_code(decode_error::INVALID_DATA_TYPE); }
            break;
        }
        // Special check for downconversions
        s->dtype_len_in = s->dtype_len;
        if (data_type == Type::FIXED_LEN_BYTE_ARRAY) {
          if (is_decimal) {
            s->dtype_len = [dtype_len = s->dtype_len]() {
              if (dtype_len <= sizeof(int32_t)) {
                return sizeof(int32_t);
              } else if (dtype_len <= sizeof(int64_t)) {
                return sizeof(int64_t);
              } else {
                return sizeof(__int128_t);
              }
            }();
          } else {
            s->dtype_len = sizeof(string_index_pair);
          }
        } else if (data_type == Type::INT32) {
          // check for smaller bitwidths
          if (s->setup.col.logical_type.has_value()) {
            auto const& lt = *s->setup.col.logical_type;
            if (lt.type == LogicalType::INTEGER) {
              s->dtype_len = lt.bit_width() / 8;
            } else if (lt.is_time_millis()) {
              // cudf outputs as INT64
              s->dtype_len = 8;
            }
          }
        } else if (data_type == Type::BYTE_ARRAY && s->setup.col.is_strings_to_cat) {
          s->dtype_len = 4;  // HASH32 output
        } else if (data_type == Type::INT96) {
          s->dtype_len = 8;  // Convert to 64-bit timestamp
        }

        // during the decoding step we need to offset the global output buffers
        // for each level of nesting so that we write to the section this page
        // is responsible for.
        // - for flat schemas, we can do this directly by using row counts
        // - for nested schemas, these offsets are computed during the preprocess step
        //
        // NOTE: in a chunked read situation, s->setup.col.column_data_base and
        // s->setup.col.valid_map_base will be aliased to memory that has been freed when we get
        // here in the non-decode step, so we cannot check against nullptr.  we'll just check a flag
        // directly.
        if (stage == page_processing_stage::DECODE) {
          int max_depth = s->setup.col.max_nesting_depth;
          for (int idx = 0; idx < max_depth; idx++) {
            PageNestingDecodeInfo* nesting_info = &s->nesting_info[idx];

            size_t output_offset;
            // schemas without lists
            if (s->setup.col.max_level[level_type::REPETITION] == 0) {
              output_offset = page_start_row >= min_row ? page_start_row - min_row : 0;
            }
            // for schemas with lists, we've already got the exact value precomputed
            else {
              output_offset = nesting_info->page_start_value;
            }

            if (s->setup.col.column_data_base != nullptr) {
              nesting_info->data_out = static_cast<uint8_t*>(s->setup.col.column_data_base[idx]);
              if (s->setup.col.column_string_base != nullptr) {
                nesting_info->string_out =
                  static_cast<uint8_t*>(s->setup.col.column_string_base[idx]);
              }

              nesting_info->data_out = static_cast<uint8_t*>(s->setup.col.column_data_base[idx]);

              if (nesting_info->data_out != nullptr) {
                // anything below max depth with a valid data pointer must be a list, so the
                // element size is the size of the offset type.
                uint32_t len = idx < max_depth - 1 ? sizeof(cudf::size_type) : s->dtype_len;
                // if this is a string column, then dtype_len is a lie. data will be offsets rather
                // than (ptr,len) tuples.
                if (is_string_col(s->setup.col)) { len = sizeof(cudf::size_type); }
                nesting_info->data_out += (output_offset * len);
              }
              if (nesting_info->string_out != nullptr) {
                nesting_info->string_out += s->setup.page.str_offset;
              }
              nesting_info->valid_map = s->setup.col.valid_map_base[idx];
              if (nesting_info->valid_map != nullptr) {
                nesting_info->valid_map += output_offset >> 5;
                nesting_info->valid_map_offset = (int32_t)(output_offset & 0x1f);
              }
            }
          }
        }
      }

      // Find the compressed size of repetition levels
      cur += InitLevelSection(s, cur, end, level_type::REPETITION);
      // Find the compressed size of definition levels
      cur += InitLevelSection(s, cur, end, level_type::DEFINITION);

      s->stream.dict_bits = 0;
      s->stream.dict_base = nullptr;
      s->stream.dict_size = 0;
      s->stream.dict_val  = 0;
      // NOTE:  if additional encodings are supported in the future, modifications must
      // be made to is_supported_encoding() in reader_impl_preprocess.cu
      switch (s->setup.page.encoding) {
        case Encoding::PLAIN_DICTIONARY:
        case Encoding::RLE_DICTIONARY: {
          // RLE-packed dictionary indices, first byte indicates index length in bits
          auto const is_decimal = s->setup.col.logical_type.has_value() and
                                  s->setup.col.logical_type->type == LogicalType::DECIMAL;
          if ((s->setup.col.physical_type == Type::BYTE_ARRAY or
               s->setup.col.physical_type == Type::FIXED_LEN_BYTE_ARRAY) and
              not is_decimal and s->setup.col.str_dict_index != nullptr) {
            // String dictionary: use index
            s->stream.dict_base = reinterpret_cast<uint8_t const*>(s->setup.col.str_dict_index);
            s->stream.dict_size =
              s->setup.col.dict_page->num_input_values * sizeof(string_index_pair);
          } else {
            s->stream.dict_base = s->setup.col.dict_page->page_data;
            s->stream.dict_size = s->setup.col.dict_page->uncompressed_page_size;
          }
          s->stream.dict_run  = 0;
          s->stream.dict_val  = 0;
          s->stream.dict_bits = (cur < end) ? *cur++ : 0;
          if (s->stream.dict_bits > 32 ||
              (!s->stream.dict_base && s->setup.col.dict_page->num_input_values > 0)) {
            s->set_error_code(decode_error::INVALID_DICT_WIDTH);
          }
        } break;
        case Encoding::PLAIN:
        case Encoding::BYTE_STREAM_SPLIT:
          s->stream.dict_size = static_cast<int32_t>(end - cur);
          s->stream.dict_val  = 0;
          if (s->setup.col.physical_type == Type::BOOLEAN) {
            s->stream.dict_run = s->stream.dict_size * 2 + 1;
          }
          break;
        case Encoding::RLE: {
          // first 4 bytes are length of RLE data
          int const len = (cur[0]) + (cur[1] << 8) + (cur[2] << 16) + (cur[3] << 24);
          cur += 4;
          if (cur + len > end) { s->set_error_code(decode_error::DATA_STREAM_OVERRUN); }
          s->stream.dict_run = 0;
        } break;
        case Encoding::DELTA_BINARY_PACKED:
        case Encoding::DELTA_LENGTH_BYTE_ARRAY:
        case Encoding::DELTA_BYTE_ARRAY:
          // nothing to do, just don't error
          break;
        default: {
          s->set_error_code(decode_error::UNSUPPORTED_ENCODING);
          break;
        }
      }
      if (cur > end) { s->set_error_code(decode_error::DATA_STREAM_OVERRUN); }
      s->stream.data_start = cur;
      s->stream.data_end   = end;
    } else {
      s->set_error_code(decode_error::EMPTY_PAGE);
    }

    s->setup.num_input_values = s->setup.page.num_input_values;
    if constexpr (requires { s->input_value_count; }) {
      s->nz_count = 0;
      s->dict_pos = 0;
      s->src_pos  = 0;

      // for flat hierarchies, we can't know how many leaf values to skip unless we do a full
      // preprocess of the definition levels (since nulls will have no actual decodable value, there
      // is no direct correlation between # of rows and # of decodable values).  so we will start
      // processing at the beginning of the value stream and disregard any indices that start
      // before the first row.
      if (s->setup.col.max_level[level_type::REPETITION] == 0) {
        s->setup.page.skipped_values      = 0;
        s->setup.page.skipped_leaf_values = 0;
        s->input_value_count              = 0;
        s->input_row_count                = 0;
        s->input_leaf_count               = 0;

        // The fixed-width decode kernel ASSUMES this is always -1 for non-lists!
        s->row_index_lower_bound = -1;
      }
      // for nested hierarchies, we have run a preprocess that lets us skip directly to the values
      // we need to start decoding at
      else {
        // input_row_count translates to "how many rows we have processed so far", so since we are
        // skipping directly to where we want to start decoding, set it to first_row
        s->input_row_count = s->setup.first_row;

        // return the lower bound to compare (page-relative) thread row index against. Explanation:
        // In the case of nested schemas, rows can span page boundaries.  That is to say,
        // we can encounter the first value for row X on page M, but the last value for page M
        // might not be the last value for row X. page M+1 (or further) may contain the last value.
        //
        // This means that the first values we encounter for a given page (M+1) may not belong to
        // the row indicated by chunk_row, but to the row before it that spanned page boundaries. If
        // that previous row is within the overall row bounds, include the values by allowing
        // relative row index -1
        int const max_row = (min_row + num_rows) - 1;
        if (min_row < page_start_row && max_row >= page_start_row - 1) {
          s->row_index_lower_bound = -1;
        } else {
          s->row_index_lower_bound = s->setup.first_row;
        }

        // if we're in the decoding step, jump directly to the first
        // value we care about
        if (stage == page_processing_stage::DECODE) {
          s->input_value_count =
            s->setup.page.skipped_values > -1 ? s->setup.page.skipped_values : 0;
        } else if (stage == page_processing_stage::PREPROCESS) {
          s->input_value_count = 0;
          s->input_leaf_count  = 0;
          // magic number to indicate it hasn't been set for use inside UpdatePageSizes
          s->setup.page.skipped_values      = -1;
          s->setup.page.skipped_leaf_values = 0;
        }
      }
    }  // if constexpr (requires { s->input_value_count; })

    __threadfence_block();
  }
  __syncthreads();

  return true;
}

/**
 * @brief Zero-fill null positions in output data using parallel per-validity-block processing
 *
 * This function processes the validity bitmap and zero-fills all positions in the output
 * data that correspond to null values. It uses a parallel approach where each thread
 * handles one 32-bit validity block at a time, looping only over the zero bits (null positions)
 * within that block.
 *
 * @tparam block_size CUDA block size for the kernel
 * @param s Page state containing all necessary information
 * @param dtype_len Size of each data element in bytes
 * @param valid_map_offset Starting bit offset in the validity map
 * @param num_values Number of values to process
 * @param t Thread index within the block
 */
template <int block_size>
__device__ void zero_fill_null_positions_shared(
  page_state_s* s, uint32_t dtype_len, int valid_map_offset, int num_values, int t)
{
  auto const block = cg::this_thread_block();
  auto const warp  = cg::tiled_partition<cudf::detail::warp_size>(block);

  // nesting level that is storing actual leaf values
  int const leaf_level_index = s->setup.col.max_nesting_depth - 1;
  auto const& ni             = s->nesting_info[leaf_level_index];

  // Check if we have nulls to fill
  if ((ni.valid_map == nullptr) || (num_values == 0)) { return; }

  auto const data_out = ni.data_out;

  constexpr int bits_per_mask = cudf::detail::size_in_bits<bitmask_type>();
  using cudf::detail::warp_size;
  constexpr int num_warps = block_size / warp_size;

  // Calculate the range of validity blocks we need to process
  int const start_bit_idx = valid_map_offset;
  int const end_bit_idx   = valid_map_offset + num_values;
  int const start_block   = start_bit_idx / bits_per_mask;
  int const end_block     = cudf::util::div_rounding_up_safe(end_bit_idx, bits_per_mask);
  int const num_blocks    = end_block - start_block;

  int const warp_id = t / warp_size;
  int const lane_id = warp.thread_rank();

  // Helper lambda for warp-parallel bit processing
  auto process_block_parallel = [&](int block_idx) {
    static_assert(bits_per_mask == warp_size, "if 64bit mask, use 2 warps per mask");

    cudf::bitmask_type validity_word = ni.valid_map[block_idx];
    int const block_start_bit        = block_idx * bits_per_mask;

    // Each thread in the warp processes one bit
    int const bit_idx = block_start_bit + lane_id;
    int const dst_pos = bit_idx - valid_map_offset;

    // Check if this bit is within our range
    bool in_range = (bit_idx >= start_bit_idx && bit_idx < end_bit_idx);

    // Check if this bit is null (0 in validity mask)
    bool const is_null = not cudf::bit_is_set(&validity_word, lane_id);

    if (in_range && is_null) {
      void* const dst = data_out + (static_cast<size_t>(dst_pos) * dtype_len);
      cuda::std::memset(dst, 0, dtype_len);
    }
  };

  // Helper lambda for sequential bit processing (fallback for remaining blocks)
  auto process_block_sequential = [&](int block_idx) {
    cudf::bitmask_type validity_word  = ni.valid_map[block_idx];
    cudf::bitmask_type null_positions = ~validity_word;
    int const dst_pos_first_bit       = block_idx * bits_per_mask - valid_map_offset;

    while (null_positions != 0) {
      int const bit_pos = __ffs(null_positions) - 1;
      int const dst_pos = dst_pos_first_bit + bit_pos;

      void* const dst = data_out + (static_cast<size_t>(dst_pos) * dtype_len);
      cuda::std::memset(dst, 0, dtype_len);

      null_positions &= (null_positions - 1);
    }
  };

  // Phase 1: Assign specific blocks to warps for warp-parallel processing
  if (warp_id == 0) {
    // Warp 0: Process first block
    process_block_parallel(start_block);
  } else if (warp_id == 1 && num_blocks > 1) {
    // Warp 1: Process last block (if different from first)
    process_block_parallel(end_block - 1);
  } else if (warp_id >= 2) {
    // Warps 2+: Process additional blocks from the beginning
    int const block_idx = start_block + (warp_id - 1);
    if (block_idx < (end_block - 1)) { process_block_parallel(block_idx); }
  }

  // Phase 2: All warps cooperatively process remaining middle blocks
  auto const last_block_processed = static_cast<int>(num_blocks > 1);
  int const remaining_start       = start_block + num_warps - last_block_processed;
  int const remaining_end         = end_block - last_block_processed;
  for (int block_idx = remaining_start + t; block_idx < remaining_end; block_idx += block_size) {
    process_block_sequential(block_idx);
  }

  __syncthreads();
}

}  // namespace cudf::io::parquet::detail
