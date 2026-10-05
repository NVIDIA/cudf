/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "column_type_histogram.hpp"
#include "io/csv/datetime.cuh"
#include "string_parsing.hpp"
#include "trie.cuh"

#include <cudf/io/types.hpp>
#include <cudf/lists/list_view.hpp>
#include <cudf/strings/detail/convert/fixed_point.cuh>
#include <cudf/strings/string_view.cuh>
#include <cudf/structs/struct_view.hpp>
#include <cudf/utilities/span.hpp>
#include <cudf/utilities/traits.hpp>

#include <rmm/device_uvector.hpp>

#include <cuda/std/iterator>
#include <cuda/std/limits>
#include <cuda/std/optional>
#include <cuda/std/type_traits>
#include <cuda/std/utility>
#include <thrust/execution_policy.h>
#include <thrust/mismatch.h>

using cudf::device_span;

namespace cudf {
namespace io {

/**
 * @brief Returns the escaped characters for a given character.
 *
 * @param escaped_char The character to escape.
 * @return The escaped characters for a given character.
 */
__device__ __forceinline__ cuda::std::pair<char, char> get_escaped_char(char escaped_char)
{
  switch (escaped_char) {
    case '"': return {'\\', '"'};
    case '\\': return {'\\', '\\'};
    case '/': return {'\\', '/'};
    case '\b': return {'\\', 'b'};
    case '\f': return {'\\', 'f'};
    case '\n': return {'\\', 'n'};
    case '\r': return {'\\', 'r'};
    case '\t': return {'\\', 't'};
    // case 'u': return UNICODE_SEQ;
    default: return {'\0', escaped_char};
  }
}

/**
 * @brief Returns the numeric value of an ASCII/UTF-8 character.
 * Handles hexadecimal digits, both uppercase and lowercase
 * for integral types and only decimal digits for floating point types.
 * If the character is not a valid numeric digit then `0` is returned and
 * valid_flag is set to false.
 *
 * @param c ASCII or UTF-8 character
 * @param valid_flag Set to false if input is not valid. Unchanged otherwise.
 *
 * @return uint8_t Numeric value of the character, or `0`
 */
template <typename T, bool as_hex = false>
__device__ constexpr uint8_t decode_digit(char c, bool* valid_flag)
{
  if (c >= '0' && c <= '9') return c - '0';
  if constexpr (as_hex and std::is_integral_v<T>) {
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
  }

  *valid_flag = false;
  return 0;
}

// Converts character to lowercase.
CUDF_HOST_DEVICE constexpr char to_lower(char const c)
{ return c >= 'A' && c <= 'Z' ? c + ('a' - 'A') : c; }

/**
 * @brief Checks if string is infinity, case insensitive with/without sign
 * Valid infinity strings are inf, +inf, -inf, infinity, +infinity, -infinity
 * String comparison is case insensitive.
 *
 * @param begin Pointer to the first element of the string
 * @param end Pointer to the first element after the string
 * @return true if string is valid infinity, else false.
 */
CUDF_HOST_DEVICE constexpr bool is_infinity(char const* begin, char const* end)
{
  if (*begin == '-' || *begin == '+') begin++;
  char const* cinf = "infinity";
  auto index       = begin;
  while (index < end) {
    if (*cinf != to_lower(*index)) break;
    index++;
    cinf++;
  }
  return ((index == begin + 3 || index == begin + 8) && index >= end);
}

/**
 * @brief Parses a character string and returns its numeric value.
 *
 * @param begin Pointer to the first element of the string
 * @param end Pointer to the first element after the string
 * @param opts The global parsing behavior options
 * @tparam base Base (radix) to use for conversion
 *
 * @return The parsed and converted value
 */
template <typename T, int base = 10>
CUDF_HOST_DEVICE cuda::std::optional<T> parse_numeric(char const* begin,
                                                      char const* end,
                                                      parse_options_view const& opts)
{
  T value{};
  bool all_digits_valid = true;
  constexpr bool as_hex = (base == 16);

  // Handle negative values if necessary
  int32_t sign = (*begin == '-') ? -1 : 1;

  // Handle infinity
  if (cuda::std::is_floating_point_v<T> && is_infinity(begin, end)) {
    return sign * cuda::std::numeric_limits<T>::infinity();
  }
  if (*begin == '-' || *begin == '+') begin++;

  // Skip over the "0x" prefix for hex notation
  if (base == 16 && begin + 2 < end && *begin == '0' && *(begin + 1) == 'x') { begin += 2; }

  // Handle the whole part of the number
  // auto index = begin;
  while (begin < end) {
    if (*begin == opts.decimal) {
      ++begin;
      break;
    } else if (base == 10 && (*begin == 'e' || *begin == 'E')) {
      break;
    } else if (*begin != opts.thousands && *begin != '+') {
      value = (value * base) + decode_digit<T, as_hex>(*begin, &all_digits_valid);
    }
    ++begin;
  }

  if (cuda::std::is_floating_point_v<T>) {
    // Handle fractional part of the number if necessary
    double divisor = 1;
    while (begin < end) {
      if (*begin == 'e' || *begin == 'E') {
        ++begin;
        break;
      } else if (*begin != opts.thousands && *begin != '+') {
        divisor /= base;
        value += decode_digit<T, as_hex>(*begin, &all_digits_valid) * divisor;
      }
      ++begin;
    }

    // Handle exponential part of the number if necessary
    if (begin < end) {
      int32_t const exponent_sign = *begin == '-' ? -1 : 1;
      if (*begin == '-' || *begin == '+') { ++begin; }
      int32_t exponent = 0;
      while (begin < end) {
        exponent = (exponent * 10) + decode_digit<T, as_hex>(*(begin++), &all_digits_valid);
      }
      if (exponent != 0) { value *= exp10(double(exponent * exponent_sign)); }
    }
  }
  if (!all_digits_valid) { return cuda::std::optional<T>{}; }

  return value * sign;
}

namespace gpu {

/**
 * @brief Advances past a run of delimiter characters at the start of a field.
 */
__device__ __inline__ char const* skip_leading_delimiter_run(char const* begin,
                                                             char const* end,
                                                             parse_options_view const& opts)
{
  if (!opts.multi_delimiter) { return begin; }
  while (begin < end && *begin == opts.delimiter) {
    ++begin;
  }
  return begin;
}

/**
 * @brief CUDA kernel iterates over the data until the end of the current field
 *
 * Also iterates over (one or more) delimiter characters after the field.
 * Function applies to formats with field delimiters and line terminators.
 *
 * @param begin Pointer to the first element of the string
 * @param end Pointer to the first element after the string
 * @param opts A set of parsing options
 * @param escape_char A boolean value to signify whether to consider `\` as escape character or
 * just a character.
 *
 * @return Pointer to the last character in the field, including the
 *  delimiter(s) following the field data
 */
__device__ __inline__ char const* seek_field_end(char const* begin,
                                                 char const* end,
                                                 parse_options_view const& opts,
                                                 bool escape_char = false)
{
  bool quotation   = false;
  auto current     = begin;
  bool escape_next = false;

  auto const field_starts_with_quote = (begin < end && *begin == opts.quotechar);
  while (current < end) {
    // Use simple logic to ignore control chars between any quote seq
    // Handles nominal cases including doublequotes within quotes, but
    // may not output exact failures as PANDAS for malformed fields.
    // Check for instances such as "a2\"bc" and "\\" if `escape_char` is true.

    // Only process quotes if field started with a quote
    if (field_starts_with_quote && *current == opts.quotechar && !escape_next) {
      quotation = !quotation;
    } else if (!quotation) {
      if (*current == opts.delimiter) {
        while (opts.multi_delimiter && (current + 1 < end) && *(current + 1) == opts.delimiter) {
          ++current;
        }
        break;
      } else if (*current == opts.terminator) {
        break;
      } else if (*current == '\r' && (current + 1 < end && *(current + 1) == '\n')) {
        --end;
        break;
      }
    }

    if (escape_char) {
      // If a escape character is encountered, escape next character in next loop.
      if (not escape_next and *current == '\\') {
        escape_next = true;
      } else {
        escape_next = false;
      }
    }

    if (current < end) { current++; }
  }
  return current;
}

/**
 * @brief Lexicographically compare digits in input against string
 * representing an integer
 *
 * @param data The pointer to beginning of character string
 * @param golden The pointer to beginning of character string representing
 * the value to be compared against
 * @return bool True if integer represented by character string is less
 * than or equal to golden data
 */
template <int N>
__device__ __inline__ bool less_equal_than(char const* data, char const (&golden)[N])
{
  auto mismatch_pair = thrust::mismatch(thrust::seq, data, data + N - 1, golden);
  if (mismatch_pair.first != data + N - 1) {
    return *mismatch_pair.first <= *mismatch_pair.second;
  } else {
    // Exact match
    return true;
  }
}

/**
 * @brief Determine which counter to increment when a sequence of digits
 * and a parity sign is encountered.
 *
 * @param data_begin The pointer to beginning of character string
 * @param data_end The pointer to end of character string
 * @param is_negative Whether the number is negative
 * @param stats Reference to structure with counters
 * @return Pointer to appropriate counter that belong to
 * the interpreted data type
 */
__device__ __inline__ cudf::size_type* infer_integral_field_counter(char const* data_begin,
                                                                    char const* data_end,
                                                                    bool is_negative,
                                                                    column_type_histogram& stats)
{
  static constexpr char uint64_max_abs[] = "18446744073709551615";
  static constexpr char int64_min_abs[]  = "9223372036854775808";
  static constexpr char int64_max_abs[]  = "9223372036854775807";

  auto digit_count = data_end - data_begin;

  // Remove preceding zeros
  if (digit_count >= (sizeof(int64_max_abs) - 1)) {
    // Trim zeros at the beginning of raw_data
    while (*data_begin == '0' && (data_begin < data_end)) {
      data_begin++;
    }
  }
  digit_count = data_end - data_begin;

  // After trimming the number of digits could be less than maximum
  // int64 digit count
  if (digit_count < (sizeof(int64_max_abs) - 1)) {  // CASE 0 : Accept validity
    // If the length of the string representing the integer is smaller
    // than string length of Int64Max then count this as an integer
    // representable by int64
    // If digit_count is 0 then ignore - sign, i.e. -000..00 should
    // be treated as a positive small integer
    return is_negative && (digit_count != 0) ? &stats.negative_small_int_count
                                             : &stats.positive_small_int_count;
  } else if (digit_count > (sizeof(uint64_max_abs) - 1)) {  // CASE 1 : Reject validity
    // If the length of the string representing the integer is greater
    // than string length of UInt64Max then count this as a string
    // since it cannot be represented as an int64 or uint64
    return &stats.string_count;
  } else if (digit_count == (sizeof(uint64_max_abs) - 1) && is_negative) {
    // A negative integer of length UInt64Max digit count cannot be represented
    // as a 64 bit integer
    return &stats.string_count;
  }

  if (digit_count == (sizeof(int64_max_abs) - 1) && is_negative) {
    return less_equal_than(data_begin, int64_min_abs) ? &stats.negative_small_int_count
                                                      : &stats.string_count;
  } else if (digit_count == (sizeof(int64_max_abs) - 1) && !is_negative) {
    return less_equal_than(data_begin, int64_max_abs) ? &stats.positive_small_int_count
                                                      : &stats.big_int_count;
  } else if (digit_count == (sizeof(uint64_max_abs) - 1)) {
    return less_equal_than(data_begin, uint64_max_abs) ? &stats.big_int_count : &stats.string_count;
  }

  return &stats.string_count;
}

}  // namespace gpu

/**
 * @brief Checks whether the given character is a whitespace character.
 *
 * @param ch The character to check
 *
 * @return True if the input is whitespace, False otherwise
 */
__inline__ __device__ bool is_whitespace(char ch) { return ch == '\t' || ch == ' '; }

/**
 * @brief Skips past the current character if it matches the given value.
 */
template <typename It>
__inline__ __device__ It skip_character(It const& it, char ch)
{ return it + (*it == ch); }

/**
 * @brief Adjusts the range to ignore starting/trailing whitespace and quotation characters.
 *
 * @param begin Pointer to the first character in the parsing range
 * @param end Pointer to the first character after the parsing range
 * @param quotechar The character used to denote quotes; '\0' if none
 *
 * @return Trimmed range
 */
__inline__ __device__ cuda::std::pair<char const*, char const*> trim_whitespaces_quotes(
  char const* begin, char const* end, char quotechar = '\0')
{
  auto not_whitespace = [] __device__(auto c) { return !is_whitespace(c); };

  auto const trim_begin = thrust::find_if(thrust::seq, begin, end, not_whitespace);
  auto const trim_end   = thrust::find_if(thrust::seq,
                                          cuda::std::make_reverse_iterator(end),
                                          cuda::std::make_reverse_iterator(trim_begin),
                                          not_whitespace);

  return {skip_character(trim_begin, quotechar), skip_character(trim_end, quotechar).base()};
}

/**
 * @brief Adjusts the range to ignore starting/trailing whitespace characters.
 *
 * @param begin Pointer to the first character in the parsing range
 * @param end Pointer to the first character after the parsing range
 *
 * @return Trimmed range
 */
__inline__ __device__ cuda::std::pair<char const*, char const*> trim_whitespaces(char const* begin,
                                                                                 char const* end)
{
  auto not_whitespace = [] __device__(auto c) { return !is_whitespace(c); };

  auto const trim_begin = thrust::find_if(thrust::seq, begin, end, not_whitespace);
  auto const trim_end   = thrust::find_if(thrust::seq,
                                          cuda::std::make_reverse_iterator(end),
                                          cuda::std::make_reverse_iterator(trim_begin),
                                          not_whitespace);

  return {trim_begin, trim_end.base()};
}

/**
 * @brief Adjusts the range to ignore starting/trailing quotation characters.
 *
 * @param begin Pointer to the first character in the parsing range
 * @param end Pointer to the first character after the parsing range
 * @param quotechar The character used to denote quotes. Provide '\0' if no quotes should be
 * trimmed.
 *
 * @return Trimmed range
 */
__inline__ __device__ cuda::std::pair<char const*, char const*> trim_quotes(char const* begin,
                                                                            char const* end,
                                                                            char quotechar)
{
  if ((cuda::std::distance(begin, end) >= 2 && *begin == quotechar &&
       *cuda::std::prev(end) == quotechar)) {
    cuda::std::advance(begin, 1);
    cuda::std::advance(end, -1);
  }
  return {begin, end};
}

struct ConvertFunctor {
  /**
   * @brief Dispatch for numeric types whose values can be convertible to
   * 0 or 1 to represent boolean false/true, based upon checking against a
   * true/false values list.
   *
   * @return bool Whether the parsed value is valid.
   */
  template <typename T,
            CUDF_ENABLE_IF(std::is_integral_v<T> and !std::is_same_v<T, bool> and
                           !cudf::is_fixed_point<T>())>
  __device__ __forceinline__ bool operator()(char const* begin,
                                             char const* end,
                                             void* out_buffer,
                                             size_t row,
                                             data_type const output_type,
                                             parse_options_view const& opts,
                                             bool as_hex = false)
  {
    auto const value = [as_hex, &opts, begin, end]() -> cuda::std::optional<T> {
      // Check for user-specified true/false values
      auto const field_len = static_cast<size_t>(end - begin);
      if (serialized_trie_contains(opts.trie_true, {begin, field_len})) { return 1; }
      if (serialized_trie_contains(opts.trie_false, {begin, field_len})) { return 0; }
      return as_hex ? cudf::io::parse_numeric<T, 16>(begin, end, opts)
                    : cudf::io::parse_numeric<T>(begin, end, opts);
    }();
    if (value.has_value()) { static_cast<T*>(out_buffer)[row] = *value; }

    return value.has_value();
  }

  /**
   * @brief Dispatch for fixed point types.
   *
   * @return bool Whether the parsed value is valid.
   */
  template <typename T, CUDF_ENABLE_IF(cudf::is_fixed_point<T>())>
  __host__ __device__ __forceinline__ bool operator()(char const* begin,
                                                      char const* end,
                                                      void* out_buffer,
                                                      size_t row,
                                                      data_type const output_type,
                                                      parse_options_view const& opts,
                                                      bool as_hex)
  {
    // TODO decide what's invalid input and update parsing functions
    static_cast<device_storage_type_t<T>*>(out_buffer)[row] =
      [&opts, output_type, begin, end]() -> device_storage_type_t<T> {
      return strings::detail::parse_decimal<device_storage_type_t<T>>(
        begin, end, output_type.scale());
    }();

    return true;
  }

  /**
   * @brief Dispatch for boolean type types.
   */
  template <typename T, CUDF_ENABLE_IF(std::is_same_v<T, bool>)>
  __device__ __forceinline__ bool operator()(char const* begin,
                                             char const* end,
                                             void* out_buffer,
                                             size_t row,
                                             data_type const output_type,
                                             parse_options_view const& opts,
                                             bool as_hex)
  {
    auto const value = [&opts, begin, end]() -> cuda::std::optional<T> {
      // Check for user-specified true/false values
      auto const field_len = static_cast<size_t>(end - begin);
      if (serialized_trie_contains(opts.trie_true, {begin, field_len})) {
        return static_cast<T>(true);
      }
      if (serialized_trie_contains(opts.trie_false, {begin, field_len})) {
        return static_cast<T>(false);
      }
      return cudf::io::parse_numeric<T>(begin, end, opts);
    }();
    if (value.has_value()) { static_cast<T*>(out_buffer)[row] = *value; }

    return value.has_value();
  }

  /**
   * @brief Dispatch for floating points, which are set to NaN if the input
   * is not valid. In such case, the validity mask is set to zero too.
   */
  template <typename T, CUDF_ENABLE_IF(std::is_floating_point_v<T>)>
  __device__ __forceinline__ bool operator()(char const* begin,
                                             char const* end,
                                             void* out_buffer,
                                             size_t row,
                                             data_type const output_type,
                                             parse_options_view const& opts,
                                             bool as_hex)
  {
    auto const value = [&opts, begin, end]() -> cuda::std::optional<T> {
      // Check for user-specified true/false values
      auto const field_len = static_cast<size_t>(end - begin);
      if (serialized_trie_contains(opts.trie_true, {begin, field_len})) {
        return static_cast<T>(true);
      }
      if (serialized_trie_contains(opts.trie_false, {begin, field_len})) {
        return static_cast<T>(false);
      }
      return cudf::io::parse_numeric<T>(begin, end, opts);
    }();
    if (value.has_value()) { static_cast<T*>(out_buffer)[row] = *value; }

    return value.has_value() and !std::isnan(*value);
  }

  /**
   * @brief Dispatch for remaining supported types, i.e., timestamp and duration types.
   */
  template <typename T,
            CUDF_ENABLE_IF(!std::is_integral_v<T> and !std::is_floating_point_v<T> and
                           !cudf::is_fixed_point<T>())>
  __host__ __device__ __forceinline__ bool operator()(char const* begin,
                                                      char const* end,
                                                      void* out_buffer,
                                                      size_t row,
                                                      data_type const output_type,
                                                      parse_options_view const& opts,
                                                      bool as_hex)
  {
    // TODO decide what's invalid input and update parsing functions
    if constexpr (cudf::is_timestamp<T>()) {
      static_cast<T*>(out_buffer)[row] = to_timestamp<T>(begin, end, opts.dayfirst);
    } else if constexpr (cudf::is_duration<T>()) {
      static_cast<T*>(out_buffer)[row] = to_duration<T>(begin, end);
    } else {
      return false;
    }
    return true;
  }
};

}  // namespace io
}  // namespace cudf
