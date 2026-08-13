/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "io/utilities/parsing_utils.cuh"

#include <cudf/column/column.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/device_scalar.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/offsets_iterator_factory.cuh>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/detail/utilities/integer_utils.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/strings/detail/strings_children.cuh>
#include <cudf/strings/detail/utf8.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/exec_policy.hpp>

#include <cub/cub.cuh>
#include <cuda/functional>
#include <cuda/iterator>
#include <cuda/std/iterator>
#include <cuda/std/utility>
#include <thrust/copy.h>
#include <thrust/transform_reduce.h>

#include <memory>
#include <type_traits>

namespace cudf::io::json::detail {

constexpr auto SINGLE_THREAD_THRESHOLD = 128;
constexpr auto WARP_THRESHOLD          = 128 * 128;  // 16K

// Unicode code point escape sequence
static constexpr char UNICODE_SEQ = 0x7F;

// Invalid escape sequence
static constexpr char NON_ESCAPE_CHAR = 0x7E;

// Unicode code point escape sequence prefix comprises '\' and 'u' characters
static constexpr size_type UNICODE_ESC_PREFIX = 2;

// Unicode code point escape sequence comprises four hex characters
static constexpr size_type UNICODE_HEX_DIGIT_COUNT = 4;

// A unicode code point escape sequence is \uXXXX
static auto constexpr NUM_UNICODE_ESC_SEQ_CHARS = UNICODE_ESC_PREFIX + UNICODE_HEX_DIGIT_COUNT;

static constexpr auto UTF16_HIGH_SURROGATE_BEGIN = 0xD800;
static constexpr auto UTF16_HIGH_SURROGATE_END   = 0xDC00;
static constexpr auto UTF16_LOW_SURROGATE_BEGIN  = 0xDC00;
static constexpr auto UTF16_LOW_SURROGATE_END    = 0xE000;

/**
 * @brief Describing whether data casting of a certain item succeed, the item was parsed to null, or
 * whether type casting failed.
 */
enum class data_casting_result { PARSING_SUCCESS, PARSED_TO_NULL, PARSING_FAILURE };

/**
 * @brief Providing additional information about the type casting result.
 */
struct data_casting_result_info {
  // Number of bytes written to output
  size_type bytes;
  // Whether parsing succeeded, item was parsed to null, or failed
  data_casting_result result;
};

/**
 * @brief Returns the character to output for a given escaped character that's following a
 * backslash.
 *
 * @param escaped_char The character following the backslash.
 * @return The character to output for a given character that's following a backslash
 */
__device__ __forceinline__ char get_escape_char(char escaped_char)
{
  switch (escaped_char) {
    case '"': return '"';
    case '\\': return '\\';
    case '/': return '/';
    case 'b': return '\b';
    case 'f': return '\f';
    case 'n': return '\n';
    case 'r': return '\r';
    case 't': return '\t';
    case 'u': return UNICODE_SEQ;
    default: return NON_ESCAPE_CHAR;
  }
}

/**
 * @brief Parses the hex value from the four hex digits of a unicode code point escape sequence
 * \uXXXX.
 *
 * @param str Pointer to the first (most-significant) hex digit
 * @return The parsed hex value if successful, -1 otherwise.
 */
__device__ __forceinline__ int32_t parse_unicode_hex(char const* str)
{
  // Prepare result
  int32_t result = 0, base = 1;
  constexpr int32_t hex_radix = 16;

  // Iterate over hex digits right-to-left
  size_type index = UNICODE_HEX_DIGIT_COUNT;
  while (index-- > 0) {
    char const ch = str[index];
    if (ch >= '0' && ch <= '9') {
      result += static_cast<int32_t>((ch - '0') + 0) * base;
      base *= hex_radix;
    } else if (ch >= 'A' && ch <= 'F') {
      result += static_cast<int32_t>((ch - 'A') + 10) * base;
      base *= hex_radix;
    } else if (ch >= 'a' && ch <= 'f') {
      result += static_cast<int32_t>((ch - 'a') + 10) * base;
      base *= hex_radix;
    } else {
      return -1;
    }
  }
  return result;
}

/**
 * @brief Writes the UTF-8 byte sequence to \p out_it and returns the number of bytes written to
 * \p out_it
 */
__device__ constexpr size_type write_utf8_char(char_utf8 character, char*& out_it)
{
  auto const bytes = (out_it == nullptr) ? strings::detail::bytes_in_char_utf8(character)
                                         : strings::detail::from_char_utf8(character, out_it);
  if (out_it) out_it += bytes;
  return bytes;
}

/**
 * @brief Processes a string, replaces escape sequences and optionally strips off the quote
 * characters.
 *
 * @tparam in_iterator_t A bidirectional input iterator type whose value_type is convertible to
 * char
 * @param in_begin Iterator to the first item to process
 * @param in_end Iterator to one past the last item to process
 * @param d_buffer Output character buffer to the first item to write
 * @param options Settings for controlling string processing behavior
 * @return A struct of (num_bytes_written, parsing_success_result), where num_bytes_written is
 * the number of bytes written to d_buffer, parsing_success_result is enum value indicating whether
 * parsing succeeded, item was parsed to null, or failed.
 */
template <typename in_iterator_t>
__device__ __forceinline__ data_casting_result_info
process_string(in_iterator_t in_begin,
               in_iterator_t in_end,
               char* d_buffer,
               cudf::io::parse_options_view const& options)
{
  int32_t bytes           = 0;
  auto const num_in_chars = cuda::std::distance(in_begin, in_end);
  // String values are indicated by keeping the quote character
  bool const is_string_value =
    num_in_chars >= 2LL &&
    (options.quotechar == '\0' ||
     (*in_begin == options.quotechar) && (*cuda::std::prev(in_end) == options.quotechar));

  // Copy literal/numeric value
  if (not is_string_value) {
    bytes += (in_end - in_begin);
    if (d_buffer) d_buffer = thrust::copy(thrust::seq, in_begin, in_end, d_buffer);
    return {bytes, data_casting_result::PARSING_SUCCESS};
  }
  char constexpr backslash_char = '\\';

  // Escape-flag, set after encountering a backslash character
  bool is_prev_char_escape = false;

  // Exclude beginning and ending quote chars from string range
  if (!options.keepquotes) {
    ++in_begin;
    --in_end;
  }

  // Iterate over the input
  while (in_begin != in_end) {
    // Copy single character to output
    if (!is_prev_char_escape) {
      is_prev_char_escape = (*in_begin == backslash_char);
      if (!is_prev_char_escape) {
        if (d_buffer) *d_buffer++ = *in_begin;
        ++bytes;
      }
      ++in_begin;
      continue;
    }

    // Previous char indicated beginning of escape sequence
    // Reset escape flag for next loop iteration
    is_prev_char_escape = false;

    // Check the character that is supposed to be escaped
    auto escaped_char = get_escape_char(*in_begin);

    // We escaped an invalid escape character -> "fail"/null for this item
    if (escaped_char == NON_ESCAPE_CHAR) { return {bytes, data_casting_result::PARSING_FAILURE}; }

    // Regular, single-character escape
    if (escaped_char != UNICODE_SEQ) {
      if (d_buffer) *d_buffer++ = escaped_char;
      ++bytes;
      ++in_begin;
      continue;
    }

    // This is an escape sequence of a unicode code point: \uXXXX,
    // where each X in XXXX represents a hex digit
    // Skip over the 'u' char from \uXXXX to the first hex digit
    ++in_begin;

    // Make sure that there's at least 4 characters left from the
    // input, which are expected to be hex digits
    if (cuda::std::distance(in_begin, in_end) < UNICODE_HEX_DIGIT_COUNT) {
      return {bytes, data_casting_result::PARSING_FAILURE};
    }

    auto hex_val = parse_unicode_hex(in_begin);

    // Couldn't parse hex values from the four-character sequence -> "fail"/null for this item
    if (hex_val < 0) { return {bytes, data_casting_result::PARSING_FAILURE}; }

    // Skip over the four hex digits
    cuda::std::advance(in_begin, UNICODE_HEX_DIGIT_COUNT);

    // If this may be a UTF-16 encoded surrogate pair:
    // we expect another \uXXXX sequence
    int32_t hex_low_val = 0;
    if (hex_val >= UTF16_HIGH_SURROGATE_BEGIN && hex_val < UTF16_HIGH_SURROGATE_END &&
        cuda::std::distance(in_begin, in_end) >= NUM_UNICODE_ESC_SEQ_CHARS &&
        *in_begin == backslash_char && *cuda::std::next(in_begin) == 'u') {
      // Try to parse hex value following the '\' and 'u' characters from what may be a UTF16 low
      // surrogate
      hex_low_val = parse_unicode_hex(cuda::std::next(in_begin, 2));
    }

    // This is indeed a UTF16 surrogate pair
    if (hex_val >= UTF16_HIGH_SURROGATE_BEGIN && hex_val < UTF16_HIGH_SURROGATE_END &&
        hex_low_val >= UTF16_LOW_SURROGATE_BEGIN && hex_low_val < UTF16_LOW_SURROGATE_END) {
      // Skip over the second \uXXXX sequence
      cuda::std::advance(in_begin, NUM_UNICODE_ESC_SEQ_CHARS);

      // Compute UTF16-encoded code point
      uint32_t unicode_code_point = 0x10000 + ((hex_val - UTF16_HIGH_SURROGATE_BEGIN) << 10) +
                                    (hex_low_val - UTF16_LOW_SURROGATE_BEGIN);
      auto utf8_chars = strings::detail::codepoint_to_utf8(unicode_code_point);
      bytes += write_utf8_char(utf8_chars, d_buffer);
    } else {
      // Just a single \uXXXX sequence
      auto utf8_chars = strings::detail::codepoint_to_utf8(hex_val);
      bytes += write_utf8_char(utf8_chars, d_buffer);
    }
  }

  // The last character of the input is a backslash -> "fail"/null for this item
  if (is_prev_char_escape) { return {bytes, data_casting_result::PARSING_FAILURE}; }
  return {bytes, data_casting_result::PARSING_SUCCESS};
}

/**
 * @brief Data structure to hold 1 bit per thread with previous `UNICODE_LOOK_BACK` bits stored in a
 * warp.
 *
 * @tparam num_warps number of warps in the block
 */
template <unsigned num_warps>
struct bitfield_warp {
  static constexpr auto UNICODE_LOOK_BACK{5};
  // 5 because for skipping unicode hex chars, look back up to 5 chars are needed.
  // 5+32 for each warp.
  bool is_slash[num_warps][UNICODE_LOOK_BACK + cudf::detail::warp_size];

  /// Sets all bits to 0
  __device__ void reset(unsigned warp_id)
  {
    if (threadIdx.x % cudf::detail::warp_size < UNICODE_LOOK_BACK) {
      is_slash[warp_id][threadIdx.x % cudf::detail::warp_size] = 0;
    }
    is_slash[warp_id][threadIdx.x % cudf::detail::warp_size + UNICODE_LOOK_BACK] = 0;
  }

  /// Shifts UNICODE_LOOK_BACK bits to the left to hold the previous UNICODE_LOOK_BACK bits
  __device__ void shift(unsigned warp_id)
  {
    if (threadIdx.x % cudf::detail::warp_size < UNICODE_LOOK_BACK)
      is_slash[warp_id][threadIdx.x % cudf::detail::warp_size] =
        is_slash[warp_id][cudf::detail::warp_size + threadIdx.x % cudf::detail::warp_size];
    __syncwarp();
  }

  /// Each thread in a warp sets its own bit.
  __device__ void set_bits(unsigned warp_id, bool is_escaping_backslash)
  {
    is_slash[warp_id][UNICODE_LOOK_BACK + threadIdx.x % cudf::detail::warp_size] =
      is_escaping_backslash;
    __syncwarp();
  }

  /// Each thread in a warp gets the requested bit.
  __device__ bool get_bit(unsigned warp_id, int bit_index)
  {
    return is_slash[warp_id][UNICODE_LOOK_BACK + bit_index];
  }
};

/**
 * @brief Data structure to hold 1 bit per thread with previous `UNICODE_LOOK_BACK` bits stored in a
 * block.
 *
 * @tparam num_warps number of warps in the block
 */
template <unsigned num_warps>
struct bitfield_block {
  static constexpr auto UNICODE_LOOK_BACK{5};
  // 5 because for skipping unicode hex chars, look back up to 5 chars are needed.
  // 5 + num_warps*32 for entire block
  bool is_slash[UNICODE_LOOK_BACK + num_warps * cudf::detail::warp_size];

  /// Sets all bits to 0
  __device__ void reset(unsigned warp_id)
  {
    if (threadIdx.x < UNICODE_LOOK_BACK) { is_slash[threadIdx.x] = 0; }
    is_slash[threadIdx.x + UNICODE_LOOK_BACK] = 0;
  }

  /// Shifts UNICODE_LOOK_BACK bits to the left to hold the previous UNICODE_LOOK_BACK bits
  __device__ void shift(unsigned warp_id)
  {
    if (threadIdx.x < UNICODE_LOOK_BACK)
      is_slash[threadIdx.x] = is_slash[num_warps * cudf::detail::warp_size + threadIdx.x];
    __syncthreads();
  }

  /// Each thread in a block sets its own bit.
  __device__ void set_bits(unsigned warp_id, bool is_escaping_backslash)
  {
    is_slash[UNICODE_LOOK_BACK + threadIdx.x] = is_escaping_backslash;
    __syncthreads();
  }

  /// Each thread in a block gets the requested bit.
  __device__ bool get_bit(unsigned warp_id, int bit_index)
  {
    return is_slash[UNICODE_LOOK_BACK + bit_index];
  }
};

// Algorithm: warp/block parallel version of string_parse and process_string()
// Decoding character classes (u8, u16, \*, *):
// character      count: input->output
// \uXXXX         6->2/3/4
// \uXXXX\uXXXX  12->2/3/4
// \"             2->1
// *              1->1
//
// ERROR conditions. (all collaborating threads quit)
// c=='\' & curr_idx == end_idx-1;
// [c-1]=='\' &  get_escape[c]==NEC
// [c-1]=='\' &  [c]=='u' & end_idx-curr_idx < UNICODE_HEX_DIGIT_COUNT
// [c-1]=='\' &  [c]=='u' & end_idx-curr_idx >= UNICODE_HEX_DIGIT_COUNT && non-hex
//
// skip conditions. (current thread skips this char, no output)
// c=='\' skip. (Escaping char only)
// [c-2]=='\' && [c-1]=='u' for [2,1], [3,2] [4,5], [5, 6], skip.
//
// write conditions. (write to d_buffer)
// [c-1]!='\' &  [c]!='\' write [c]
// [c-1]!='\' &  [c]=='\' skip (already covered in skip conditions)
// [c-1]=='\' &  [c]!=NEC && [c]!=UNICODE_SEQ, write [c]
// [c-1]=='\' &  [c]=='u' & end_idx-curr_idx >= UNICODE_HEX_DIGIT_COUNT && hex, DECODE
// [c+1:4]=curr_hex_val
//        // if [c+5]=='\' & [c+6]=='u' & end_idx-curr_idx >= UNICODE_HEX_DIGIT_COUNT &&
//        hex,DECODE [c+7:4]=next_hex_val
//        // if [c-7]=='\' & [c-6]=='u' & end_idx-curr_idx >= UNICODE_HEX_DIGIT_COUNT &&
//        hex,DECODE [c-5:4]=prev_hex_val prev_hex_val, curr_hex_val, next_hex_val
//        // if prev_hex_val in high, curr_hex_val in low, skip.
//        // if curr_hex_val in high, next_hex_val in low, write [u16]
// if curr_hex_val not in high, write [u8]
// before writing, find num of output characters per threads,
// then do intra-warp/intra-block scan for out_idx
// propagate offset from next iteration to carry forward.
// Uses 1 warp per string or 1 block per string

/**
 * @brief Warp/Block parallel version of string_parse functor
 *
 * @tparam is_warp True if 1 warp per string, False if 1 block per string
 * @tparam num_warps Number of warps per block
 * @tparam str_tuple_it Iterator type for tuple with string pointer and its length
 * @param str_tuples iterator of tuple with string pointer and its length
 * @param total_out_strings Number of string rows to be processed
 * @param str_counter Counter to keep track of processed number of strings
 * @param null_mask Null mask
 * @param null_count_data pointer to store null count
 * @param options Settings for controlling string processing behavior
 * @param d_sizes Output size of each row
 * @param d_offsets Offsets to identify where to store the results for each string
 * @param d_chars Character array to store the characters of strings
 */
template <bool is_warp, size_type num_warps, typename str_tuple_it>
CUDF_KERNEL void parse_fn_string_parallel(str_tuple_it str_tuples,
                                          size_type total_out_strings,
                                          size_type* str_counter,
                                          bitmask_type* null_mask,
                                          size_type* null_count_data,
                                          cudf::io::parse_options_view const options,
                                          size_type* d_sizes,
                                          cudf::detail::input_offsetalator d_offsets,
                                          char* d_chars)
{
  constexpr auto BLOCK_SIZE =
    is_warp ? cudf::detail::warp_size : cudf::detail::warp_size * num_warps;
  size_type lane = is_warp ? (threadIdx.x % BLOCK_SIZE) : threadIdx.x;

  // get 1-string index per warp/block
  auto get_next_string = [&]() {
    if constexpr (is_warp) {
      size_type istring;
      if (lane == 0) { istring = atomicAdd(str_counter, 1); }
      return __shfl_sync(0xffffffff, istring, 0);
    } else {
      // Ensure lane 0 doesn't update istring before all threads have read the previous iteration's
      // istring value
      __syncthreads();
      __shared__ size_type istring;
      if (lane == 0) { istring = atomicAdd(str_counter, 1); }
      __syncthreads();
      return istring;
    }
  };
  // grid-stride loop.
  for (size_type istring = get_next_string(); istring < total_out_strings;
       istring           = get_next_string()) {
    // skip nulls
    if (null_mask != nullptr && not bit_is_set(null_mask, istring)) {
      if (!d_chars && lane == 0) { d_sizes[istring] = 0; }
      continue;  // gride-stride return;
    }

    auto in_begin           = str_tuples[istring].first;
    auto in_end             = in_begin + str_tuples[istring].second;
    auto const num_in_chars = str_tuples[istring].second;
    if constexpr (is_warp) {
      if (num_in_chars <= SINGLE_THREAD_THRESHOLD or num_in_chars > WARP_THRESHOLD) continue;
    } else {
      if (num_in_chars <= WARP_THRESHOLD) continue;
    }

    // Check if the value corresponds to the null literal
    if (!d_chars) {
      auto const is_null_literal = serialized_trie_contains(
        options.trie_na, {in_begin, static_cast<std::size_t>(num_in_chars)});
      if (is_null_literal && null_mask != nullptr) {
        if (lane == 0) {
          clear_bit(null_mask, istring);
          atomicAdd(null_count_data, 1);
          if (!d_chars) { d_sizes[istring] = 0; }
        }
        continue;  // gride-stride return;
      }
    }
    // String values are indicated by keeping the quote character
    bool const is_string_value =
      num_in_chars >= 2LL &&
      (options.quotechar == '\0' ||
       (*in_begin == options.quotechar) && (*cuda::std::prev(in_end) == options.quotechar));
    char* d_buffer = d_chars ? d_chars + d_offsets[istring] : nullptr;

    // Copy literal/numeric value
    if (not is_string_value) {
      if (!d_chars) {
        if (lane == 0) { d_sizes[istring] = in_end - in_begin; }
      } else {
        for (thread_index_type char_index = lane; char_index < (in_end - in_begin);
             char_index += BLOCK_SIZE) {
          d_buffer[char_index] = in_begin[char_index];
        }
      }
      continue;  // gride-stride return;
    }

    // Exclude beginning and ending quote chars from string range
    if (!options.keepquotes) {
      ++in_begin;
      --in_end;
    }
    // warp-parallelized or block-parallelized process_string()

    auto is_hex = [](auto ch) {
      return (ch >= '0' && ch <= '9') || (ch >= 'A' && ch <= 'F') || (ch >= 'a' && ch <= 'f');
    };

    // for backslash scan calculation: is_previous_escaping_backslash
    [[maybe_unused]] auto warp_id = threadIdx.x / cudf::detail::warp_size;
    bool init_state_reg;
    __shared__ bool init_state_shared;
    size_type last_offset_reg;
    __shared__ size_type last_offset_shared;
    bool& init_state(is_warp ? init_state_reg : init_state_shared);
    size_type& last_offset(is_warp ? last_offset_reg : last_offset_shared);
    if (is_warp || lane == 0) {
      init_state  = false;
      last_offset = 0;
    }
    using bitfield =
      std::conditional_t<is_warp, bitfield_warp<num_warps>, bitfield_block<num_warps>>;
    __shared__ bitfield is_slash;
    is_slash.reset(warp_id);
    __syncthreads();
    // 0-31, 32-63, ... i*32-n.
    // entire warp executes but with mask.
    for (thread_index_type char_index = lane;
         char_index < cudf::util::round_up_safe(in_end - in_begin, static_cast<long>(BLOCK_SIZE));
         char_index += BLOCK_SIZE) {
      bool const is_within_bounds = char_index < (in_end - in_begin);
      auto const c                = is_within_bounds ? in_begin[char_index] : '\0';
      auto const prev_c = (char_index > 0 and is_within_bounds) ? in_begin[char_index - 1] : '\0';
      auto const escaped_char = get_escape_char(c);

      bool is_escaping_backslash{false};
      [[maybe_unused]] bool is_prev_escaping_backslash{false};
      // To check current is backslash by checking if previous is backslash.
      // curr = !prev & c=='\\'
      // So, scan is required from beginning of string.
      // State table approach (intra-warp FST) (intra-block FST)
      // 2 states: Not-Slash(NS), Slash(S).
      // prev  /   *
      // NS    S  NS
      //  S   NS  NS
      // After inclusive scan, all current S states translate to escaping backslash.
      // All escaping backslash should be skipped.

      struct state_table {
        // using bit fields instead of state[2]
        bool state0 : 1;
        bool state1 : 1;
        bool inline __device__ get(bool init_state) const { return init_state ? state1 : state0; }
      };
      state_table curr{is_within_bounds && c == '\\', false};  // state transition vector.
      auto composite_op = [](state_table op1, state_table op2) {
        // equivalent of state_table{op2.state[op1.state[0]], op2.state[op1.state[1]]};
        return state_table{op1.state0 ? op2.state1 : op2.state0,
                           op1.state1 ? op2.state1 : op2.state0};
      };
      state_table scanned;
      // inclusive scan of escaping backslashes
      if constexpr (is_warp) {
        using SlashScan = cub::WarpScan<state_table>;
        __shared__ typename SlashScan::TempStorage temp_slash[num_warps];
        SlashScan(temp_slash[warp_id]).InclusiveScan(curr, scanned, composite_op);
        is_escaping_backslash = scanned.get(init_state);
        init_state            = __shfl_sync(~0u, is_escaping_backslash, BLOCK_SIZE - 1);
        __syncwarp();
        is_slash.shift(warp_id);
        is_slash.set_bits(warp_id, is_escaping_backslash);
        is_prev_escaping_backslash = is_slash.get_bit(warp_id, lane - 1);
      } else {
        using SlashScan = cub::BlockScan<state_table, BLOCK_SIZE>;
        __shared__ typename SlashScan::TempStorage temp_slash;
        SlashScan(temp_slash).InclusiveScan(curr, scanned, composite_op);
        is_escaping_backslash = scanned.get(init_state);
        __syncthreads();
        if (threadIdx.x == BLOCK_SIZE - 1) init_state = is_escaping_backslash;
        __syncthreads();
        is_slash.shift(warp_id);
        is_slash.set_bits(warp_id, is_escaping_backslash);
        is_prev_escaping_backslash = is_slash.get_bit(warp_id, lane - 1);
        // There is another __syncthreads() at the end of for-loop.
      }

      // String with parsing errors are made as null
      bool error = false;
      if (is_within_bounds) {
        // curr=='\' and end, or prev=='\' and curr=='u' and end-curr < UNICODE_HEX_DIGIT_COUNT
        // or prev=='\' and curr=='u' and end-curr >= UNICODE_HEX_DIGIT_COUNT and any non-hex
        error |= (is_escaping_backslash /*c == '\\'*/ && char_index == (in_end - in_begin) - 1);
        error |= (is_prev_escaping_backslash && escaped_char == NON_ESCAPE_CHAR);
        error |= (is_prev_escaping_backslash && c == 'u' &&
                  ((in_begin + char_index + UNICODE_HEX_DIGIT_COUNT >= in_end) |
                   !is_hex(in_begin[char_index + 1]) | !is_hex(in_begin[char_index + 2]) |
                   !is_hex(in_begin[char_index + 3]) | !is_hex(in_begin[char_index + 4])));
      }
      // Make sure all threads have no errors before continuing
      if constexpr (is_warp) {
        error = __any_sync(~0u, error);
      } else {
        using ErrorReduce = cub::BlockReduce<bool, BLOCK_SIZE>;
        __shared__ typename ErrorReduce::TempStorage temp_storage_error;
        __shared__ bool error_reduced;
        auto berr = ErrorReduce(temp_storage_error).Sum(error);  // TODO use cub::LogicalOR.
        // only valid in thread0, so shared memory is used for broadcast.
        if (lane == 0) error_reduced = berr;
        __syncthreads();
        error = error_reduced;
      }
      // If any thread has an error, skip the rest of the string and make this string as null
      if (error) {
        if (!d_chars && lane == 0) {
          if (null_mask != nullptr) {
            clear_bit(null_mask, istring);
            atomicAdd(null_count_data, 1);
          }
          last_offset      = 0;
          d_sizes[istring] = 0;
        }
        if constexpr (!is_warp) { __syncthreads(); }
        break;  // gride-stride return;
      }

      // Skipping non-copied escaped characters
      bool skip = !is_within_bounds;  // false;
      // skip \ for \" \\ \/ \b \f \n \r \t \uXXXX
      skip |= is_escaping_backslash;
      if (is_within_bounds) {
        // skip X for each X in \uXXXX
        skip |=
          char_index >= 2 && is_slash.get_bit(warp_id, lane - 2) && in_begin[char_index - 1] == 'u';
        skip |=
          char_index >= 3 && is_slash.get_bit(warp_id, lane - 3) && in_begin[char_index - 2] == 'u';
        skip |=
          char_index >= 4 && is_slash.get_bit(warp_id, lane - 4) && in_begin[char_index - 3] == 'u';
        skip |=
          char_index >= 5 && is_slash.get_bit(warp_id, lane - 5) && in_begin[char_index - 4] == 'u';
      }
      int this_num_out = 0;
      cudf::char_utf8 write_char{};

      if (!skip) {
        // 1. Unescaped character
        if (!is_prev_escaping_backslash) {
          this_num_out = 1;
          // writes char directly for non-unicode
        } else {
          // 2. Escaped character
          if (escaped_char != UNICODE_SEQ) {
            this_num_out = 1;
            // writes char directly for non-unicode
          } else {
            // 3. Unicode
            // UTF8 \uXXXX
            auto hex_val     = parse_unicode_hex(in_begin + char_index + 1);
            auto hex_low_val = 0;
            // UTF16 \uXXXX\uXXXX
            // Note: no need for scanned_backslash below because we already know that
            // only '\u' check is enough.
            if (hex_val >= UTF16_HIGH_SURROGATE_BEGIN && hex_val < UTF16_HIGH_SURROGATE_END &&
                (in_begin + char_index + UNICODE_HEX_DIGIT_COUNT + NUM_UNICODE_ESC_SEQ_CHARS) <
                  in_end &&
                in_begin[char_index + NUM_UNICODE_ESC_SEQ_CHARS - 1] == '\\' &&
                in_begin[char_index + NUM_UNICODE_ESC_SEQ_CHARS] == 'u') {
              hex_low_val = parse_unicode_hex(in_begin + char_index + 1 + 6);
            }
            if (hex_val >= UTF16_HIGH_SURROGATE_BEGIN && hex_val < UTF16_HIGH_SURROGATE_END &&
                hex_low_val >= UTF16_LOW_SURROGATE_BEGIN && hex_low_val < UTF16_LOW_SURROGATE_END) {
              // Compute UTF16-encoded code point
              uint32_t unicode_code_point = 0x10000 +
                                            ((hex_val - UTF16_HIGH_SURROGATE_BEGIN) << 10) +
                                            (hex_low_val - UTF16_LOW_SURROGATE_BEGIN);
              write_char   = strings::detail::codepoint_to_utf8(unicode_code_point);
              this_num_out = strings::detail::bytes_in_char_utf8(write_char);
            } else {
              // if hex_val is high surrogate, ideally it should be parsing failure.
              // but skipping it as other parsers do this too.
              if (hex_val >= UTF16_LOW_SURROGATE_BEGIN && hex_val < UTF16_LOW_SURROGATE_END) {
                // Ideally this should be skipped if previous char is high surrogate.
                skip         = true;
                this_num_out = 0;
                write_char   = 0;
              } else {
                // if UTF8
                write_char   = strings::detail::codepoint_to_utf8(hex_val);
                this_num_out = strings::detail::bytes_in_char_utf8(write_char);
              }
            }
          }
        }
      }  // !skip end.
      {
        // compute offset to write output for each thread
        size_type offset;
        if constexpr (is_warp) {
          using OffsetScan = cub::WarpScan<size_type>;
          __shared__ typename OffsetScan::TempStorage temp_storage[num_warps];
          OffsetScan(temp_storage[warp_id]).ExclusiveSum(this_num_out, offset);
        } else {
          using OffsetScan = cub::BlockScan<size_type, BLOCK_SIZE>;
          __shared__ typename OffsetScan::TempStorage temp_storage;
          OffsetScan(temp_storage).ExclusiveSum(this_num_out, offset);
          __syncthreads();
        }
        offset += last_offset;
        // Write output
        if (d_chars && !skip) {
          auto const is_not_unicode = (!is_prev_escaping_backslash) || escaped_char != UNICODE_SEQ;
          if (is_not_unicode) {
            *(d_buffer + offset) = (!is_prev_escaping_backslash) ? c : escaped_char;
          } else {
            strings::detail::from_char_utf8(write_char, d_buffer + offset);
          }
        }
        offset += this_num_out;
        if constexpr (is_warp) {
          last_offset = __shfl_sync(0xffffffff, offset, BLOCK_SIZE - 1);
        } else {
          __syncthreads();
          if (threadIdx.x == BLOCK_SIZE - 1) last_offset = offset;
          __syncthreads();
        }
      }
    }  // char for-loop
    if (!d_chars && lane == 0) { d_sizes[istring] = last_offset; }
  }  // grid-stride for-loop
}

template <typename str_tuple_it>
struct string_parse {
  str_tuple_it str_tuples;
  bitmask_type* null_mask;
  size_type* null_count_data;
  cudf::io::parse_options_view const options;
  size_type* d_sizes{};
  cudf::detail::input_offsetalator d_offsets;
  char* d_chars{};

  __device__ void operator()(size_type idx)
  {
    if (null_mask != nullptr && not bit_is_set(null_mask, idx)) {
      if (!d_chars) { d_sizes[idx] = 0; }
      return;
    }
    auto const in_begin     = str_tuples[idx].first;
    auto const in_end       = in_begin + str_tuples[idx].second;
    auto const num_in_chars = str_tuples[idx].second;

    if (num_in_chars > SINGLE_THREAD_THRESHOLD) return;

    // Check if the value corresponds to the null literal
    if (!d_chars) {
      auto const is_null_literal = serialized_trie_contains(
        options.trie_na, {in_begin, static_cast<std::size_t>(num_in_chars)});
      if (is_null_literal && null_mask != nullptr) {
        clear_bit(null_mask, idx);
        atomicAdd(null_count_data, 1);
        if (!d_chars) { d_sizes[idx] = 0; }
        return;
      }
    }

    char* d_buffer        = d_chars ? d_chars + d_offsets[idx] : nullptr;
    auto str_process_info = process_string(in_begin, in_end, d_buffer, options);
    if (str_process_info.result != data_casting_result::PARSING_SUCCESS) {
      if (null_mask != nullptr) {
        clear_bit(null_mask, idx);
        atomicAdd(null_count_data, 1);
      }
      if (!d_chars) { d_sizes[idx] = 0; }
    } else {
      if (!d_chars) { d_sizes[idx] = str_process_info.bytes; }
    }
  }
};

template <typename SymbolT>
struct to_string_view_pair {
  SymbolT const* data;
  to_string_view_pair(SymbolT const* _data) : data(_data) {}
  __device__ cuda::std::pair<char const*, std::size_t> operator()(
    cuda::std::tuple<size_type, size_type> ip)
  {
    return cuda::std::pair<char const*, std::size_t>{
      data + cuda::std::get<0>(ip), static_cast<std::size_t>(cuda::std::get<1>(ip))};
  }
};

template <typename string_view_pair_it>
static std::unique_ptr<column> parse_string(string_view_pair_it str_tuples,
                                            size_type col_size,
                                            rmm::device_buffer&& null_mask,
                                            cudf::detail::device_scalar<size_type>& d_null_count,
                                            cudf::io::parse_options_view const& options,
                                            rmm::cuda_stream_view stream,
                                            rmm::device_async_resource_ref mr)
{
  //  CUDF_FUNC_RANGE();

  auto const max_length = thrust::transform_reduce(
    rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
    str_tuples,
    str_tuples + col_size,
    cuda::proclaim_return_type<std::size_t>([] __device__(auto t) { return t.second; }),
    size_type{0},
    cuda::maximum<size_type>{});

  auto sizes           = rmm::device_uvector<size_type>(col_size, stream);
  auto d_sizes         = sizes.data();
  auto null_count_data = d_null_count.data();

  auto single_thread_fn = string_parse<decltype(str_tuples)>{
    str_tuples, static_cast<bitmask_type*>(null_mask.data()), null_count_data, options, d_sizes};
  thrust::for_each_n(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                     cuda::counting_iterator<size_type>{0},
                     col_size,
                     single_thread_fn);

  constexpr auto warps_per_block  = 8;
  constexpr int threads_per_block = cudf::detail::warp_size * warps_per_block;
  auto num_blocks                 = cudf::util::div_rounding_up_safe(col_size, warps_per_block);
  auto str_counter =
    cudf::numeric_scalar(size_type{0}, true, stream, cudf::get_current_device_resource_ref());

  // TODO run these independent kernels in parallel streams.
  if (max_length > SINGLE_THREAD_THRESHOLD) {
    parse_fn_string_parallel<true, warps_per_block>
      <<<num_blocks, threads_per_block, 0, stream.value()>>>(
        str_tuples,
        col_size,
        str_counter.data(),
        static_cast<bitmask_type*>(null_mask.data()),
        null_count_data,
        options,
        d_sizes,
        cudf::detail::input_offsetalator{},
        nullptr);
    CUDF_CUDA_TRY(cudaGetLastError());
  }

  if (max_length > WARP_THRESHOLD) {
    // for strings longer than WARP_THRESHOLD, 1 block per string
    str_counter.set_value(0, stream);
    parse_fn_string_parallel<false, warps_per_block>
      <<<num_blocks, threads_per_block, 0, stream.value()>>>(
        str_tuples,
        col_size,
        str_counter.data(),
        static_cast<bitmask_type*>(null_mask.data()),
        null_count_data,
        options,
        d_sizes,
        cudf::detail::input_offsetalator{},
        nullptr);
    CUDF_CUDA_TRY(cudaGetLastError());
  }

  auto [offsets, bytes] =
    cudf::strings::detail::make_offsets_child_column(sizes.begin(), sizes.end(), stream, mr);
  auto d_offsets = cudf::detail::offsetalator_factory::make_input_iterator(offsets->view());

  // CHARS column
  rmm::device_uvector<char> chars(bytes, stream, mr);
  auto d_chars = chars.data();

  single_thread_fn.d_chars   = d_chars;
  single_thread_fn.d_offsets = d_offsets;

  thrust::for_each_n(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                     cuda::counting_iterator<size_type>{0},
                     col_size,
                     single_thread_fn);

  if (max_length > SINGLE_THREAD_THRESHOLD) {
    str_counter.set_value(0, stream);
    parse_fn_string_parallel<true, warps_per_block>
      <<<num_blocks, threads_per_block, 0, stream.value()>>>(
        str_tuples,
        col_size,
        str_counter.data(),
        static_cast<bitmask_type*>(null_mask.data()),
        null_count_data,
        options,
        d_sizes,
        d_offsets,
        d_chars);
    CUDF_CUDA_TRY(cudaGetLastError());
  }

  if (max_length > WARP_THRESHOLD) {
    str_counter.set_value(0, stream);
    // for strings longer than WARP_THRESHOLD, 1 block per string
    parse_fn_string_parallel<false, warps_per_block>
      <<<num_blocks, threads_per_block, 0, stream.value()>>>(
        str_tuples,
        col_size,
        str_counter.data(),
        static_cast<bitmask_type*>(null_mask.data()),
        null_count_data,
        options,
        d_sizes,
        d_offsets,
        d_chars);
    CUDF_CUDA_TRY(cudaGetLastError());
  }

  return make_strings_column(col_size,
                             std::move(offsets),
                             chars.release(),
                             d_null_count.value(stream),
                             std::move(null_mask));
}

std::unique_ptr<column> parse_data(
  char const* data,
  cuda::zip_iterator<size_type const*, size_type const*> offset_length_begin,
  size_type col_size,
  data_type col_type,
  rmm::device_buffer&& null_mask,
  size_type null_count,
  cudf::io::parse_options_view const& options,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();

  if (col_size == 0) { return make_empty_column(col_type); }
  auto d_null_count = cudf::detail::device_scalar<size_type>(
    null_count, stream, cudf::get_current_device_resource_ref());
  auto null_count_data = d_null_count.data();
  if (null_mask.is_empty()) {
    null_mask = cudf::create_null_mask(col_size, mask_state::ALL_VALID, stream, mr);
  }

  // Prepare iterator that returns (string_ptr, string_length)-pairs needed by type conversion
  auto str_tuples = thrust::make_transform_iterator(offset_length_begin, to_string_view_pair{data});

  if (col_type == cudf::data_type{cudf::type_id::STRING}) {
    return parse_string(
      str_tuples, col_size, std::move(null_mask), d_null_count, options, stream, mr);
  }

  auto out_col =
    make_fixed_width_column(col_type, col_size, std::move(null_mask), null_count, stream, mr);
  auto output_dv_ptr = mutable_column_device_view::create(*out_col, stream);

  // use `ConvertFunctor` to convert non-string values
  thrust::for_each_n(
    rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
    cuda::counting_iterator<size_type>{0},
    col_size,
    [str_tuples, col = *output_dv_ptr, options, col_type, null_count_data] __device__(
      size_type row) {
      if (col.is_null(row)) { return; }
      auto const in = str_tuples[row];

      auto const is_null_literal =
        serialized_trie_contains(options.trie_na, {in.first, static_cast<size_t>(in.second)});

      if (is_null_literal) {
        col.set_null(row);
        atomicAdd(null_count_data, 1);
        return;
      }

      // If this is a string value, remove quotes
      auto [in_begin, in_end] = trim_quotes(in.first, in.first + in.second, options.quotechar);

      auto const is_parsed = cudf::type_dispatcher(col_type,
                                                   ConvertFunctor{},
                                                   in_begin,
                                                   in_end,
                                                   col.data<char>(),
                                                   row,
                                                   col_type,
                                                   options,
                                                   false);
      if (not is_parsed) {
        col.set_null(row);
        atomicAdd(null_count_data, 1);
      }
    });

  out_col->set_null_count(d_null_count.value(stream));

  return out_col;
}

}  // namespace cudf::io::json::detail
