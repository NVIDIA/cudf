/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "csv_common.hpp"
#include "csv_gpu.hpp"
#include "io/utilities/block_utils.cuh"
#include "io/utilities/parsing_utils.cuh"
#include "io/utilities/trie.cuh"

#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/detail/utilities/grid_1d.cuh>
#include <cudf/detail/utilities/integer_utils.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/fixed_point/fixed_point.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/strings/detail/convert/fixed_point.cuh>
#include <cudf/strings/string_view.cuh>
#include <cudf/utilities/bit.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/span.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <cuda/atomic>
#include <cuda/std/algorithm>
#include <thrust/count.h>
#include <thrust/detail/copy.h>
#include <thrust/remove.h>
#include <thrust/transform.h>

#include <type_traits>

using namespace ::cudf::io;

using cudf::device_span;
using cudf::detail::grid_1d;

namespace cudf {
namespace io {
namespace csv {
namespace gpu {

/// Block dimension for dtype detection and conversion kernels
constexpr uint32_t csvparse_block_dim = 128;

/*
 * @brief Returns true is the input character is a valid digit.
 * Supports both decimal and hexadecimal digits (uppercase and lowercase).
 *
 * @param c Character to check
 * @param is_hex Whether to check as a hexadecimal
 *
 * @return `true` if it is digit-like, `false` otherwise
 */
__device__ __inline__ bool is_digit(char c, bool is_hex = false)
{
  if (c >= '0' && c <= '9') return true;

  if (is_hex) {
    if (c >= 'A' && c <= 'F') return true;
    if (c >= 'a' && c <= 'f') return true;
  }

  return false;
}

/*
 * @brief Checks whether the given character counters indicate a potentially
 * valid date and/or time field.
 *
 * For performance and simplicity, we detect only the most common date
 * formats. Example formats that are detectable:
 *
 *    `2001/02/30`
 *    `2001-02-30 00:00:00`
 *    `2/30/2001 T04:05:60.7`
 *    `2 / 1 / 2011`
 *    `02/January`
 *
 * @param len Number of non special-symbol or numeric characters
 * @param decimal_count Number of '.' characters
 * @param colon_count Number of ':' characters
 * @param dash_count Number of '-' characters
 * @param slash_count Number of '/' characters
 *
 * @return `true` if it is date-like, `false` otherwise
 */
__device__ __inline__ bool is_datetime(
  long len, long decimal_count, long colon_count, long dash_count, long slash_count)
{
  // Must not exceed count of longest month (September) plus `T` time indicator
  if (len > 10) { return false; }
  // Must not exceed more than one decimals or more than two time separators
  if (decimal_count > 1 || colon_count > 2) { return false; }
  // Must have one or two '-' or '/' but not both as date separators
  if ((dash_count > 0 && dash_count < 3 && slash_count == 0) ||
      (dash_count == 0 && slash_count > 0 && slash_count < 3)) {
    return true;
  }

  return false;
}

/*
 * @brief Returns true if the counters indicate a potentially valid float.
 * False positives are possible because positions are not taken into account.
 * For example, field "e.123-" would match the pattern.
 *
 * @param len Number of non special-symbol or numeric characters
 * @param digit_count Number of digits characters
 * @param decimal_count Number of occurrences of the decimal point character
 * @param thousands_count Number of occurrences of the thousands separator character
 * @param dash_count Number of '-' characters
 * @param exponent_count Number of 'e or E' characters
 *
 * @return `true` if it is floating point-like, `false` otherwise
 */
__device__ __inline__ bool is_floatingpoint(long len,
                                            long digit_count,
                                            long decimal_count,
                                            long thousands_count,
                                            long dash_count,
                                            long exponent_count)
{
  // Can't have more than one exponent and one decimal point
  if (decimal_count > 1) return false;
  if (exponent_count > 1) return false;

  // Without the exponent or a decimal point, this is an integer, not a float
  if (decimal_count == 0 && exponent_count == 0) return false;

  // Can only have one '-' per component
  if (dash_count > 1 + exponent_count) return false;

  // If anything other than these characters is present, it's not a float
  if (digit_count + decimal_count + dash_count + exponent_count + thousands_count != len) {
    return false;
  }

  // Needs at least 1 digit, 2 if exponent is present
  if (digit_count < 1 + exponent_count) return false;

  return true;
}

/*
 * @brief CUDA kernel that parses and converts CSV data into cuDF column data.
 *
 * Data is processed in one row/record at a time, so the number of total
 * threads (tid) is equal to the number of rows.
 *
 * @param opts A set of parsing options
 * @param csv_text The entire CSV data to read
 * @param column_flags Per-column parsing behavior flags
 * @param row_offsets The start the CSV data of interest
 * @param d_column_data The count for each column data type
 */
CUDF_KERNEL void __launch_bounds__(csvparse_block_dim)
  data_type_detection(parse_options_view const opts,
                      device_span<char const> csv_text,
                      device_span<column_parse::flags const> const column_flags,
                      device_span<uint64_t const> const row_offsets,
                      device_span<column_type_histogram> d_column_data)
{
  auto const raw_csv = csv_text.data();

  // ThreadIds range per block, so also need the blockId
  // This is entry into the fields; threadId is an element within `num_records`
  auto const rec_id      = grid_1d::global_thread_id();
  auto const rec_id_next = rec_id + 1;

  // we can have more threads than data, make sure we are not past the end of the data
  if (rec_id_next >= row_offsets.size()) { return; }

  auto field_start   = raw_csv + row_offsets[rec_id];
  auto const row_end = raw_csv + row_offsets[rec_id_next];

  auto next_field = field_start;
  int col         = 0;
  int actual_col  = 0;

  // Going through all the columns of a given record
  while (col < column_flags.size() && field_start < row_end) {
    // In delim_whitespace mode, collapse leading delimiter runs so leading whitespace does
    // not produce empty fields (matches pandas behavior).
    field_start = cudf::io::gpu::skip_leading_delimiter_run(field_start, row_end, opts);
    if (field_start >= row_end) break;
    auto next_delimiter = cudf::io::gpu::seek_field_end(field_start, row_end, opts);

    // Checking if this is a column that the user wants --- user can filter columns
    if (column_flags[col] & column_parse::inferred) {
      // points to last character in the field
      auto const field_len = static_cast<size_t>(next_delimiter - field_start);
      if (serialized_trie_contains(opts.trie_na, {field_start, field_len})) {
        atomicAdd(&d_column_data[actual_col].null_count, 1);
      } else if (serialized_trie_contains(opts.trie_true, {field_start, field_len}) ||
                 serialized_trie_contains(opts.trie_false, {field_start, field_len})) {
        atomicAdd(&d_column_data[actual_col].bool_count, 1);
      } else if (cudf::io::is_infinity(field_start, next_delimiter)) {
        atomicAdd(&d_column_data[actual_col].float_count, 1);
      } else {
        long count_number    = 0;
        long count_decimal   = 0;
        long count_thousands = 0;
        long count_slash     = 0;
        long count_dash      = 0;
        long count_plus      = 0;
        long count_colon     = 0;
        long count_string    = 0;
        long count_exponent  = 0;

        // Modify field_start & end to ignore whitespace and quotechars
        // This could possibly result in additional empty fields
        auto const trimmed_field_range = trim_whitespaces_quotes(field_start, next_delimiter);
        auto const trimmed_field_len   = trimmed_field_range.second - trimmed_field_range.first;

        if (trimmed_field_len == 0) {
          cuda::atomic_ref<cudf::size_type, cuda::thread_scope_device> ref{
            d_column_data[actual_col].string_count};
          ref.fetch_add(1, cuda::memory_order_relaxed);
        } else {
          for (auto cur = trimmed_field_range.first; cur < trimmed_field_range.second; ++cur) {
            if (is_digit(*cur)) {
              count_number++;
              continue;
            }
            if (*cur == opts.decimal) {
              count_decimal++;
              continue;
            }
            if (*cur == opts.thousands) {
              count_thousands++;
              continue;
            }
            // Looking for unique characters that will help identify column types.
            switch (*cur) {
              case '-': count_dash++; break;
              case '+': count_plus++; break;
              case '/': count_slash++; break;
              case ':': count_colon++; break;
              case 'e':
              case 'E':
                if (cur > trimmed_field_range.first && cur < trimmed_field_range.second - 1)
                  count_exponent++;
                break;
              default: count_string++; break;
            }
          }

          // Integers have to have the length of the string
          // Off by one if they start with a minus sign
          auto const int_req_number_cnt =
            trimmed_field_len - count_thousands -
            ((*trimmed_field_range.first == '-' || *trimmed_field_range.first == '+') &&
             trimmed_field_len > 1);

          if (column_flags[col] & column_parse::as_datetime) {
            // PANDAS uses `object` dtype if the date is unparseable
            if (is_datetime(count_string, count_decimal, count_colon, count_dash, count_slash)) {
              atomicAdd(&d_column_data[actual_col].datetime_count, 1);
            } else {
              atomicAdd(&d_column_data[actual_col].string_count, 1);
            }
          } else if (count_number == int_req_number_cnt) {
            auto const is_negative = (*trimmed_field_range.first == '-');
            auto const data_begin =
              trimmed_field_range.first + (is_negative || (*trimmed_field_range.first == '+'));
            cudf::size_type* ptr = cudf::io::gpu::infer_integral_field_counter(
              data_begin, data_begin + count_number, is_negative, d_column_data[actual_col]);
            atomicAdd(ptr, 1);
          } else if (is_floatingpoint(trimmed_field_len,
                                      count_number,
                                      count_decimal,
                                      count_thousands,
                                      count_dash + count_plus,
                                      count_exponent)) {
            atomicAdd(&d_column_data[actual_col].float_count, 1);
          } else {
            atomicAdd(&d_column_data[actual_col].string_count, 1);
          }
        }
      }
      actual_col++;
    }
    next_field  = next_delimiter + 1;
    field_start = next_field;
    col++;
  }
}

/**
 * @brief CUDA kernel that parses and converts CSV data into cuDF column data.
 *
 * Data is processed one record at a time
 *
 * @param[in] options A set of parsing options
 * @param[in] data The entire CSV data to read
 * @param[in] column_flags Per-column parsing behavior flags
 * @param[in] row_offsets The start the CSV data of interest
 * @param[in] dtypes The data type of the column
 * @param[out] columns The output column data
 * @param[out] valids The bitmaps indicating whether column fields are valid
 * @param[out] valid_counts The number of valid fields in each column
 * @param[out] is_quoted_flags Per-column boolean arrays tracking which rows were quoted fields
 */
CUDF_KERNEL void __launch_bounds__(csvparse_block_dim)
  convert_csv_to_cudf(cudf::io::parse_options_view options,
                      device_span<char const> data,
                      device_span<column_parse::flags const> column_flags,
                      device_span<uint64_t const> row_offsets,
                      device_span<cudf::data_type const> dtypes,
                      device_span<void* const> columns,
                      device_span<cudf::bitmask_type* const> valids,
                      device_span<size_type> valid_counts,
                      device_span<bool* const> is_quoted_flags)
{
  auto const raw_csv = data.data();
  // thread IDs range per block, so also need the block id.
  // this is entry into the field array - tid is an elements within the num_entries array
  auto const rec_id      = grid_1d::global_thread_id();
  auto const rec_id_next = rec_id + 1;

  // we can have more threads than data, make sure we are not past the end of the data
  if (rec_id_next >= row_offsets.size()) return;

  auto field_start   = raw_csv + row_offsets[rec_id];
  auto const row_end = raw_csv + row_offsets[rec_id_next];

  auto next_field = field_start;
  int col         = 0;
  int actual_col  = 0;

  while (col < column_flags.size() && field_start < row_end) {
    // In delim_whitespace mode, collapse leading delimiter runs so leading whitespace does
    // not produce empty fields (matches pandas behavior).
    field_start = cudf::io::gpu::skip_leading_delimiter_run(field_start, row_end, options);
    if (field_start >= row_end) break;
    next_field          = field_start;
    auto next_delimiter = cudf::io::gpu::seek_field_end(field_start, row_end, options);

    if (column_flags[col] & column_parse::enabled) {
      // check if the entire field is a NaN string - consistent with pandas
      auto const is_valid = !serialized_trie_contains(
        options.trie_na, {field_start, static_cast<size_t>(next_delimiter - field_start)});

      // Modify field_start & end to ignore whitespace and quotechars
      auto field_end = next_delimiter;
      if (is_valid && dtypes[actual_col].id() != cudf::type_id::STRING) {
        auto const trimmed_field =
          trim_whitespaces_quotes(field_start, field_end, options.quotechar);
        field_start = trimmed_field.first;
        field_end   = trimmed_field.second;
      }
      bool* const is_quoted_output =
        is_quoted_flags.empty() ? nullptr : is_quoted_flags[actual_col];
      if (is_valid) {
        // Type dispatcher does not handle STRING
        if (dtypes[actual_col].id() == cudf::type_id::STRING) {
          auto end        = next_delimiter;
          bool was_quoted = false;
          if (not options.keepquotes) {
            if (not options.detect_whitespace_around_quotes) {
              if ((*field_start == options.quotechar) && (*(end - 1) == options.quotechar)) {
                ++field_start;
                --end;
                was_quoted = true;
              }
            } else {
              // If the string is quoted, whitespace around the quotes get removed as well
              auto const trimmed_field = trim_whitespaces(field_start, end);
              if ((*trimmed_field.first == options.quotechar) &&
                  (*(trimmed_field.second - 1) == options.quotechar)) {
                field_start = trimmed_field.first + 1;
                end         = trimmed_field.second - 1;
                was_quoted  = true;
              }
            }
          }
          // Track whether this field was quoted (for doublequote unescaping)
          if (is_quoted_output != nullptr) { is_quoted_output[rec_id] = was_quoted; }
          auto str_list = static_cast<std::pair<char const*, size_t>*>(columns[actual_col]);
          str_list[rec_id].first  = field_start;
          str_list[rec_id].second = end - field_start;
        } else {
          if (cudf::type_dispatcher(dtypes[actual_col],
                                    ConvertFunctor{},
                                    field_start,
                                    field_end,
                                    columns[actual_col],
                                    rec_id,
                                    dtypes[actual_col],
                                    options,
                                    column_flags[col] & column_parse::as_hexadecimal)) {
            // set the valid bitmap - all bits were set to 0 to start
            set_bit(valids[actual_col], rec_id);
            atomicAdd(&valid_counts[actual_col], 1);
          }
        }
      } else if (dtypes[actual_col].id() == cudf::type_id::STRING) {
        auto str_list           = static_cast<std::pair<char const*, size_t>*>(columns[actual_col]);
        str_list[rec_id].first  = nullptr;
        str_list[rec_id].second = 0;
        if (is_quoted_output != nullptr) { is_quoted_output[rec_id] = false; }
      }
      ++actual_col;
    }
    next_field  = next_delimiter + 1;
    field_start = next_field;
    ++col;
  }
}

/*
 * @brief Merge two packed row contexts (each corresponding to a block of characters)
 * and return the packed row context corresponding to the merged character block
 */
inline __device__ packed_rowctx_t merge_row_contexts(packed_rowctx_t first_ctx,
                                                     packed_rowctx_t second_ctx)
{
  uint32_t id0 = get_row_context(first_ctx, ROW_CTX_NONE) & 3;
  uint32_t id1 = get_row_context(first_ctx, ROW_CTX_QUOTE) & 3;
  uint32_t id2 = get_row_context(first_ctx, ROW_CTX_COMMENT) & 3;
  return (first_ctx & ~pack_row_contexts(3, 3, 3)) +
         pack_row_contexts(get_row_context(second_ctx, id0),
                           get_row_context(second_ctx, id1),
                           get_row_context(second_ctx, id2));
}

/*
 * @brief Per-character context:
 * 1-bit count (0 or 1) per context in the lower 4 bits
 * 2-bit output context id per input context in bits 8..15
 */
constexpr __device__ uint32_t make_char_context(uint32_t id0,
                                                uint32_t id1,
                                                uint32_t id2 = ROW_CTX_COMMENT,
                                                uint32_t c0  = 0,
                                                uint32_t c1  = 0,
                                                uint32_t c2  = 0)
{
  return (id0 << 8) | (id1 << 10) | (id2 << 12) | (ROW_CTX_EOF << 14) | (c0) | (c1 << 1) |
         (c2 << 2);
}

/*
 * @brief Merge a 1-character context to keep track of bitmasks where new rows occur
 * Merges a single-character "block" row context at position pos with the current
 * block's row context (the current block contains 32-pos characters)
 *
 * @param ctx Current block context and new rows bitmaps
 * @param char_ctx state transitions associated with new character
 * @param pos Position within the current 32-character block
 *
 * NOTE: This is probably the most performance-critical piece of the row gathering kernel.
 * The char_ctx value should be created via make_char_context, and its value should
 * have been evaluated at compile-time.
 */
inline __device__ void merge_char_context(uint4& ctx, uint32_t char_ctx, uint32_t pos)
{
  uint32_t id0 = (ctx.w >> 0) & 3;
  uint32_t id1 = (ctx.w >> 2) & 3;
  uint32_t id2 = (ctx.w >> 4) & 3;
  // Set the newrow bit in the bitmap at the corresponding position
  ctx.x |= ((char_ctx >> id0) & 1) << pos;
  ctx.y |= ((char_ctx >> id1) & 1) << pos;
  ctx.z |= ((char_ctx >> id2) & 1) << pos;
  // Update the output context ids
  ctx.w = ((char_ctx >> (8 + id0 * 2)) & 0x03) | ((char_ctx >> (6 + id1 * 2)) & 0x0c) |
          ((char_ctx >> (4 + id2 * 2)) & 0x30) | (ROW_CTX_EOF << 6);
}

/*
 * Convert the context-with-row-bitmaps version to a packed row context
 */
inline __device__ packed_rowctx_t pack_rowmaps(uint4 ctx_map)
{
  return pack_row_contexts(make_row_context(__popc(ctx_map.x), (ctx_map.w >> 0) & 3),
                           make_row_context(__popc(ctx_map.y), (ctx_map.w >> 2) & 3),
                           make_row_context(__popc(ctx_map.z), (ctx_map.w >> 4) & 3));
}

/*
 * Selects the row bitmap corresponding to the given parser state
 */
inline __device__ uint32_t select_rowmap(uint4 ctx_map, uint32_t ctxid)
{
  return (ctxid == ROW_CTX_NONE)      ? ctx_map.x
         : (ctxid == ROW_CTX_QUOTE)   ? ctx_map.y
         : (ctxid == ROW_CTX_COMMENT) ? ctx_map.z
                                      : 0;
}

/**
 * @brief Single pair-wise 512-wide row context merge transform
 *
 * Merge row context blocks and record the merge operation in a context
 * tree so that the transform is reversible.
 * The tree is organized such that the left and right children of node n
 * are located at indices n*2 and n*2+1, the root node starting at index 1
 *
 * @tparam lanemask mask to specify source of packed row context
 * @tparam tmask mask to specify principle thread for merging row context
 * @tparam base start location for writing into packed row context tree
 * @tparam level_scale level of the node in the tree
 * @param[out] ctxtree packed row context tree
 * @param[in] ctxb packed row context for the current character block
 * @param t thread id (leaf node id)
 */
template <uint32_t lanemask, uint32_t tmask, uint32_t base, uint32_t level_scale>
inline __device__ void ctx_merge(device_span<uint64_t> ctxtree, packed_rowctx_t* ctxb, uint32_t t)
{
  uint64_t tmp = shuffle_xor(*ctxb, lanemask);
  if (!(t & tmask)) {
    *ctxb                              = merge_row_contexts(*ctxb, tmp);
    ctxtree[base + (t >> level_scale)] = *ctxb;
  }
}

/**
 * @brief Single 512-wide row context inverse merge transform
 *
 * Walks the context tree starting from a root node
 *
 * @tparam rmask Mask to specify which threads write input row context
 * @param[in] base Start read location of the merge transform tree
 * @param[in] ctxtree Merge transform tree
 * @param[in] ctx Input context
 * @param[in] brow4 output row in block *4
 * @param[in] t thread id (leaf node id)
 */
template <uint32_t rmask>
inline __device__ void ctx_unmerge(
  uint32_t base, device_span<uint64_t const> ctxtree, uint32_t* ctx, uint32_t* brow4, uint32_t t)
{
  rowctx32_t ctxb_left, ctxb_right, ctxb_sum;
  ctxb_sum   = get_row_context(ctxtree[base], *ctx);
  ctxb_left  = get_row_context(ctxtree[(base) * 2 + 0], *ctx);
  ctxb_right = get_row_context(ctxtree[(base) * 2 + 1], ctxb_left & 3);
  if (t & (rmask)) {
    *brow4 += (ctxb_sum & ~3) - (ctxb_right & ~3);
    *ctx = ctxb_left & 3;
  }
}

/*
 * @brief 512-wide row context merge transform
 *
 * Repeatedly merge row context blocks, keeping track of each merge operation
 * in a context tree so that the transform is reversible
 * The tree is organized such that the left and right children of node n
 * are located at indices n*2 and n*2+1, the root node starting at index 1
 *
 * Each node contains the counts and output contexts corresponding to the
 * possible input contexts.
 * Each parent node's count is obtained by adding the corresponding counts
 * from the left child node with the right child node's count selected from
 * the left child node's output context:
 *   parent.count[k] = left.count[k] + right.count[left.outctx[k]]
 *   parent.outctx[k] = right.outctx[left.outctx[k]]
 *
 * @param[out] ctxtree packed row context tree
 * @param[in] ctxb packed row context for the current character block
 * @param t thread id (leaf node id)
 */
static inline __device__ void rowctx_merge_transform(device_span<uint64_t> ctxtree,
                                                     packed_rowctx_t ctxb,
                                                     uint32_t t)
{
  ctxtree[512 + t] = ctxb;
  ctx_merge<1, 0x1, 256, 1>(ctxtree, &ctxb, t);
  ctx_merge<2, 0x3, 128, 2>(ctxtree, &ctxb, t);
  ctx_merge<4, 0x7, 64, 3>(ctxtree, &ctxb, t);
  ctx_merge<8, 0xf, 32, 4>(ctxtree, &ctxb, t);
  __syncthreads();
  if (t < 32) {
    ctxb = ctxtree[32 + t];
    ctx_merge<1, 0x1, 16, 1>(ctxtree, &ctxb, t);
    ctx_merge<2, 0x3, 8, 2>(ctxtree, &ctxb, t);
    ctx_merge<4, 0x7, 4, 3>(ctxtree, &ctxb, t);
    ctx_merge<8, 0xf, 2, 4>(ctxtree, &ctxb, t);
    // Final stage
    uint64_t tmp = shuffle_xor(ctxb, 16);
    if (t == 0) { ctxtree[1] = merge_row_contexts(ctxb, tmp); }
  }
}

/*
 * @brief 512-wide row context inverse merge transform
 *
 * Walks the context tree starting from the root node (index 1) using
 * the starting context in node index 0.
 * The return value is the starting row and input context for the given leaf node
 *
 * @param[in] ctxtree Merge transform tree
 * @param[in] t thread id (leaf node id)
 *
 * @return Final row context and count (row_position*4 + context_id format)
 */
static inline __device__ rowctx32_t
rowctx_inverse_merge_transform(device_span<uint64_t const> ctxtree, uint32_t t)
{
  uint32_t ctx     = ctxtree[0] & 3;  // Starting input context
  rowctx32_t brow4 = 0;               // output row in block *4

  ctx_unmerge<256>(1, ctxtree, &ctx, &brow4, t);
  ctx_unmerge<128>(2 + (t >> 8), ctxtree, &ctx, &brow4, t);
  ctx_unmerge<64>(4 + (t >> 7), ctxtree, &ctx, &brow4, t);
  ctx_unmerge<32>(8 + (t >> 6), ctxtree, &ctx, &brow4, t);
  ctx_unmerge<16>(16 + (t >> 5), ctxtree, &ctx, &brow4, t);
  ctx_unmerge<8>(32 + (t >> 4), ctxtree, &ctx, &brow4, t);
  ctx_unmerge<4>(64 + (t >> 3), ctxtree, &ctx, &brow4, t);
  ctx_unmerge<2>(128 + (t >> 2), ctxtree, &ctx, &brow4, t);
  ctx_unmerge<1>(256 + (t >> 1), ctxtree, &ctx, &brow4, t);

  return brow4 + ctx;
}

constexpr auto bk_ctxtree_size = rowofs_block_dim * 2;

/**
 * @brief Gather row offsets from CSV character data split into 16KB chunks
 *
 * This is done in two phases: the first phase returns the possible row counts
 * per 16K character block for each possible parsing context at the start of the block,
 * along with the resulting parsing context at the end of the block.
 * The caller can then compute the actual parsing context at the beginning of each
 * individual block and total row count.
 * The second phase outputs the location of each row in the block, using the parsing
 * context and initial row counter accumulated from the results of the previous phase.
 * Row parsing context will be updated after phase 2 such that the value contains
 * the number of rows starting at byte_range_end or beyond.
 *
 * @param row_ctx Row parsing context (output of phase 1 or input to phase 2)
 * @param offsets_out Row offsets (nullptr for phase1, non-null indicates phase 2)
 * @param data Base pointer of character data (all row offsets are relative to this)
 * @param chunk_size Total number of characters to parse
 * @param parse_pos Current parsing position in the file
 * @param start_offset Position of the start of the character buffer in the file
 * @param data_size CSV file size
 * @param byte_range_start Ignore rows starting before this position in the file
 * @param byte_range_end In phase 2, store the number of rows beyond range in row_ctx
 * @param skip_rows Number of rows to skip (ignored in phase 1)
 * @param terminator Line terminator character
 * @param delimiter Column delimiter character
 * @param quotechar Quote character
 * @param escapechar Delimiter escape character
 * @param commentchar Comment line character (skip rows starting with this character)
 */
CUDF_KERNEL void __launch_bounds__(rowofs_block_dim)
  gather_row_offsets_gpu(uint64_t* row_ctx,
                         device_span<uint64_t> ctxtree,
                         device_span<uint64_t> offsets_out,
                         device_span<char const> const data,
                         size_t chunk_size,
                         size_t parse_pos,
                         size_t start_offset,
                         size_t data_size,
                         size_t byte_range_start,
                         size_t byte_range_end,
                         size_t skip_rows,
                         int terminator,
                         int delimiter,
                         int quotechar,
                         int escapechar,
                         int commentchar)
{
  auto start            = data.data();
  auto const bk_ctxtree = ctxtree.subspan(blockIdx.x * bk_ctxtree_size, bk_ctxtree_size);

  // file-level end position for this scan, clamped to the file size
  size_t const end_in_file = (parse_pos >= data_size || chunk_size > data_size - parse_pos)
                               ? data_size
                               : parse_pos + chunk_size;
  // offset into the local `data` window (which begins at `start_offset`), clamped to the buffer
  size_t const end_off      = end_in_file > start_offset ? end_in_file - start_offset : 0;
  size_t const data_end_off = data_size > start_offset ? data_size - start_offset : 0;
  auto const end            = start + cuda::std::min(end_off, data.size());
  auto const data_end       = start + cuda::std::min(data_end_off, data.size());
  // Offset of `parse_pos` inside the local `data` window, clamped to avoid underflow
  auto const parse_off = parse_pos > start_offset ? parse_pos - start_offset : 0;
  uint32_t const t     = threadIdx.x;
  size_t block_pos     = parse_off + blockIdx.x * static_cast<size_t>(rowofs_block_bytes) + t * 32;
  auto cur             = start + block_pos;

  // Initial state is neutral context (no state transitions), zero rows
  uint4 ctx_map = {
    .x = 0,
    .y = 0,
    .z = 0,
    .w = (ROW_CTX_NONE << 0) | (ROW_CTX_QUOTE << 2) | (ROW_CTX_COMMENT << 4) | (ROW_CTX_EOF << 6)};
  int c, c_prev = (cur > start && cur <= end) ? cur[-1] : terminator;
  // Loop through all 32 bytes and keep a bitmask of row starts for each possible input context
  for (uint32_t pos = 0; pos < 32; pos++, cur++, c_prev = c) {
    uint32_t ctx;
    if (cur < end) {
      c = cur[0];
      if (c_prev == terminator) {
        if (c == commentchar) {
          // Start of a new comment row
          ctx = make_char_context(ROW_CTX_COMMENT, ROW_CTX_QUOTE, ROW_CTX_COMMENT, 1, 0, 1);
        } else if (c == quotechar) {
          // Quoted string on newrow, or quoted string ending in terminator
          ctx = make_char_context(ROW_CTX_QUOTE, ROW_CTX_NONE, ROW_CTX_QUOTE, 1, 0, 1);
        } else {
          // Start of a new row unless within a quote
          ctx = make_char_context(ROW_CTX_NONE, ROW_CTX_QUOTE, ROW_CTX_NONE, 1, 0, 1);
        }
      } else if (c == quotechar) {
        // Quote handling uses ROW_CTX_COMMENT as a "pending exit" state to correctly handle
        // escaped quotes (""). When in QUOTE state and we see a quote, we can't immediately
        // exit because it might be the first quote of a "" escape sequence. We transition to
        // COMMENT (pending exit) and wait for the next character:
        //   - If next char is quote: it's a "" escape, return to QUOTE
        //   - If next char is anything else: exit confirmed, go to NONE
        // This doesn't conflict with actual comment handling because comments are only
        // detected at row boundaries (after newline), where COMMENT state is set with row
        // counting. Mid-row, COMMENT is purely used for this pending exit mechanism.
        if (c_prev == delimiter) {
          // Quote after delimiter: start field or pending exit
          ctx = make_char_context(ROW_CTX_QUOTE, ROW_CTX_COMMENT);
        } else if (c_prev == quotechar) {
          // Quote after quote: "" escape or stay NONE (Spark compatibility)
          ctx = make_char_context(ROW_CTX_NONE, ROW_CTX_COMMENT, ROW_CTX_QUOTE);
        } else {
          // Quote after regular char: pending exit or stay NONE
          ctx = make_char_context(ROW_CTX_NONE, ROW_CTX_COMMENT);
        }
      } else {
        // Non-quote char: stay in current state, or exit from pending
        ctx = make_char_context(ROW_CTX_NONE, ROW_CTX_QUOTE, ROW_CTX_NONE);
      }
    } else {
      bool const is_last_chunk = data_end_off <= data.size();
      if (is_last_chunk && cur <= end && cur == data_end) {
        // Add a newline at data end (need the extra row offset to infer length of previous row)
        ctx = make_char_context(ROW_CTX_EOF, ROW_CTX_EOF, ROW_CTX_EOF, 1, 1, 1);
      } else {
        // Pass-through context (beyond chunk_size or data_end)
        ctx = make_char_context(ROW_CTX_NONE, ROW_CTX_QUOTE, ROW_CTX_COMMENT);
      }
    }
    // Merge with current context, keeping track of where new rows occur
    merge_char_context(ctx_map, ctx, pos);
  }

  // Eliminate rows that start before byte_range_start
  if (start_offset + block_pos < byte_range_start) {
    uint32_t dist_minus1 =
      cuda::std::min(byte_range_start - (start_offset + block_pos) - 1, UINT64_C(31));
    uint32_t mask = 0xffff'fffe << dist_minus1;
    ctx_map.x &= mask;
    ctx_map.y &= mask;
    ctx_map.z &= mask;
  }

  // Convert the long-form {rowmap,outctx}[inctx] version into packed version
  // {rowcount,ouctx}[inctx], then merge the row contexts of the 32-character blocks into
  // a single 16K-character block context
  rowctx_merge_transform(bk_ctxtree, pack_rowmaps(ctx_map), t);

  // If this is the second phase, get the block's initial parser state and row counter
  if (offsets_out.data()) {
    if (t == 0) { bk_ctxtree[0] = row_ctx[blockIdx.x]; }
    __syncthreads();

    // Walk back the transform tree with the known initial parser state
    rowctx32_t ctx             = rowctx_inverse_merge_transform(bk_ctxtree, t);
    uint64_t row               = (bk_ctxtree[0] >> 2) + (ctx >> 2);
    uint32_t rows_out_of_range = 0;
    uint32_t rowmap            = select_rowmap(ctx_map, ctx & 3);
    // Output row positions
    while (rowmap != 0) {
      uint32_t pos = __ffs(rowmap);
      block_pos += pos;
      if (row >= skip_rows && row - skip_rows < offsets_out.size()) {
        // Output byte offsets are relative to the base of the input buffer
        offsets_out[row - skip_rows] = block_pos - 1;
        rows_out_of_range += (start_offset + block_pos - 1 >= byte_range_end);
      }
      row++;
      rowmap >>= pos;
    }
    __syncthreads();
    // Return the number of rows out of range

    using block_reduce = typename cub::BlockReduce<uint32_t, rowofs_block_dim>;
    __shared__ typename block_reduce::TempStorage bk_storage;
    rows_out_of_range = block_reduce(bk_storage).Sum(rows_out_of_range);
    if (t == 0) { row_ctx[blockIdx.x] = rows_out_of_range; }
  } else {
    // Just store the row counts and output contexts
    if (t == 0) { row_ctx[blockIdx.x] = bk_ctxtree[1]; }
  }
}

size_t __host__ count_blank_rows(cudf::io::parse_options_view const& opts,
                                 device_span<char const> data,
                                 device_span<uint64_t const> row_offsets,
                                 rmm::cuda_stream_view stream)
{
  auto const newline  = opts.skipblanklines ? opts.terminator : opts.comment;
  auto const comment  = opts.comment != '\0' ? opts.comment : newline;
  auto const carriage = (opts.skipblanklines && opts.terminator == '\n') ? '\r' : comment;
  return thrust::count_if(
    rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
    row_offsets.begin(),
    row_offsets.end(),
    [data = data, newline, comment, carriage] __device__(uint64_t const pos) {
      return ((pos != data.size()) &&
              (data[pos] == newline || data[pos] == comment || data[pos] == carriage));
    });
}

device_span<uint64_t> __host__ remove_blank_rows(cudf::io::parse_options_view const& options,
                                                 device_span<char const> data,
                                                 device_span<uint64_t> row_offsets,
                                                 rmm::cuda_stream_view stream)
{
  size_t d_size       = data.size();
  auto const newline  = options.skipblanklines ? options.terminator : options.comment;
  auto const comment  = options.comment != '\0' ? options.comment : newline;
  auto const carriage = (options.skipblanklines && options.terminator == '\n') ? '\r' : comment;
  auto new_end        = thrust::remove_if(
    rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
    row_offsets.begin(),
    row_offsets.end(),
    [data = data, d_size, newline, comment, carriage] __device__(uint64_t const pos) {
      return ((pos != d_size) &&
              (data[pos] == newline || data[pos] == comment || data[pos] == carriage));
    });
  return row_offsets.subspan(0, new_end - row_offsets.begin());
}

cudf::detail::host_vector<column_type_histogram> detect_column_types(
  cudf::io::parse_options_view const& options,
  device_span<char const> const data,
  device_span<column_parse::flags const> const column_flags,
  device_span<uint64_t const> const row_starts,
  size_t const num_active_columns,
  rmm::cuda_stream_view stream)
{
  // Calculate actual block count to use based on records count
  int const block_size = csvparse_block_dim;
  int const grid_size  = (row_starts.size() + block_size - 1) / block_size;

  auto d_stats = cudf::detail::make_zeroed_device_uvector_async<column_type_histogram>(
    num_active_columns, stream, cudf::get_current_device_resource_ref());

  data_type_detection<<<grid_size, block_size, 0, stream.value()>>>(
    options, data, column_flags, row_starts, d_stats);
  CUDF_CUDA_TRY(cudaGetLastError());

  return cudf::detail::make_host_vector(d_stats, stream);
}

void decode_row_column_data(cudf::io::parse_options_view const& options,
                            device_span<char const> data,
                            device_span<column_parse::flags const> column_flags,
                            device_span<uint64_t const> row_offsets,
                            device_span<cudf::data_type const> dtypes,
                            device_span<void* const> columns,
                            device_span<cudf::bitmask_type* const> valids,
                            device_span<size_type> valid_counts,
                            device_span<bool* const> is_quoted_flags,
                            rmm::cuda_stream_view stream)
{
  // Calculate actual block count to use based on records count
  auto const block_size = csvparse_block_dim;
  auto const num_rows   = row_offsets.size() - 1;
  auto const grid_size  = cudf::util::div_rounding_up_safe<size_t>(num_rows, block_size);

  convert_csv_to_cudf<<<grid_size, block_size, 0, stream.value()>>>(options,
                                                                    data,
                                                                    column_flags,
                                                                    row_offsets,
                                                                    dtypes,
                                                                    columns,
                                                                    valids,
                                                                    valid_counts,
                                                                    is_quoted_flags);
  CUDF_CUDA_TRY(cudaGetLastError());
}

uint32_t __host__ gather_row_offsets(parse_options_view const& options,
                                     uint64_t* row_ctx,
                                     device_span<uint64_t> const offsets_out,
                                     device_span<char const> const data,
                                     size_t chunk_size,
                                     size_t parse_pos,
                                     size_t start_offset,
                                     size_t data_size,
                                     size_t byte_range_start,
                                     size_t byte_range_end,
                                     size_t skip_rows,
                                     rmm::cuda_stream_view stream)
{
  uint32_t dim_grid = 1 + (chunk_size / rowofs_block_bytes);
  auto ctxtree      = rmm::device_uvector<packed_rowctx_t>(dim_grid * bk_ctxtree_size, stream);

  gather_row_offsets_gpu<<<dim_grid, rowofs_block_dim, 0, stream.value()>>>(
    row_ctx,
    ctxtree,
    offsets_out,
    data,
    chunk_size,
    parse_pos,
    start_offset,
    data_size,
    byte_range_start,
    byte_range_end,
    skip_rows,
    options.terminator,
    options.delimiter,
    (options.quotechar) ? options.quotechar : 0x100,
    /*(options.escapechar) ? options.escapechar :*/ 0x100,
    (options.comment) ? options.comment : 0x100);
  CUDF_CUDA_TRY(cudaGetLastError());

  return dim_grid;
}

}  // namespace gpu
}  // namespace csv
}  // namespace io
}  // namespace cudf
