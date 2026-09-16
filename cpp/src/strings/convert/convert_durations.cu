/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "utilities/time_utils.cuh"

#include <cudf/column/column_device_view.cuh>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/strings/convert/convert_durations.hpp>
#include <cudf/strings/detail/convert/int_to_string.cuh>
#include <cudf/strings/detail/strings_children.cuh>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/device_uvector.hpp>

#include <cuda/iterator>
#include <cuda/std/algorithm>
#include <cuda/std/cmath>
#include <cuda/stream>
#include <thrust/execution_policy.h>
#include <thrust/transform.h>
#include <thrust/transform_reduce.h>

#include <map>
#include <vector>

namespace cudf {
namespace strings {
namespace detail {

namespace {

// duration components timeparts structure
struct alignas(4) duration_component {
  int32_t day;        //-2,147,483,648 to 2,147,483,647
  int32_t subsecond;  // 000000000 to 999999999
  int8_t hour;        // 00 to 23
  int8_t minute;      // 00 to 59
  int8_t second;      // 00 to 59
  bool is_negative;   // true/false
};

enum class format_char_type : int8_t {
  literal,   // literal char type passed through
  specifier  // duration format specifier
};

/**
 * @brief Represents a format specifier or literal from a duration format string.
 *
 * Created by the format_compiler when parsing a format string.
 */
struct alignas(4) format_item {
  format_char_type item_type;  // specifier or literal indicator
  char value;                  // specifier or literal value
  int8_t length;               // item length in bytes

  static format_item new_specifier(char format_char, int8_t length)
  {
    return format_item{format_char_type::specifier, format_char, length};
  }
  static format_item new_delimiter(char literal)
  {
    return format_item{format_char_type::literal, literal, 1};
  }
};

/**
 * @brief The format_compiler parses a duration format string into a vector of
 * format_items.
 *
 * The vector of format_items are used when parsing a string into duration
 * components and when formatting a string from duration components.
 */
struct format_compiler {
  std::string_view const format;
  rmm::device_uvector<format_item> d_items;
  format_compiler(std::string_view format, cuda::stream_ref stream)
    : format(format), d_items(0, stream)
  {
    static std::map<char, int8_t> const specifier_lengths = {
      {'-', -1},  // '-' if negative
      {'D', -1},  // 1 to 11 (not in std::format)
      {'H', 2},   // HH
      {'I', 2},   // HH
      {'M', 2},   // MM
      {'S', -1},  // 2 to 13 SS[.mmm][uuu][nnn] (uuu,nnn are not in std::format)
      {'p', 2},   // AM/PM
      {'R', 5},   // 5 HH:MM
      {'T', 8},   // 8 HH:MM:SS"
      {'r', 11}   // HH:MM:SS AM/PM
    };
    std::vector<format_item> items;
    auto str    = format.data();
    auto length = format.length();
    bool negative_sign{true};
    while (length > 0) {
      char ch = *str++;
      length--;
      if (ch != '%') {
        items.push_back(format_item::new_delimiter(ch));
        continue;
      }
      CUDF_EXPECTS(length > 0, "Unfinished specifier in duration format");

      ch = *str++;
      length--;
      if (ch == '%')  // escaped % char
      {
        items.push_back(format_item::new_delimiter(ch));
        continue;
      } else if (ch == 'n') {
        items.push_back(format_item::new_delimiter('\n'));
        continue;
      } else if (ch == 't') {
        items.push_back(format_item::new_delimiter('\t'));
        continue;
      }
      if (ch == 'O') {
        CUDF_EXPECTS(*str == 'H' || *str == 'I' || *str == 'M' || *str == 'S',
                     "locale's alternative representation not supported for specifier: " +
                       std::string(1, *str));
        ch = *str++;
        length--;
        items.push_back(format_item::new_specifier(ch, 2));  // without sign
        continue;
      }
      CUDF_EXPECTS(specifier_lengths.find(ch) != specifier_lengths.end(),
                   "invalid format specifier: " + std::string(1, ch));

      // negative sign should be present only once.
      if (negative_sign) {
        if (std::string("DHIMSRT").find_first_of(ch) != std::string::npos) {
          items.push_back(format_item::new_specifier('-', specifier_lengths.at('-')));
          negative_sign = false;
        }
      }

      int8_t spec_length = specifier_lengths.at(ch);
      items.push_back(format_item::new_specifier(ch, spec_length));
    }

    // create program in device memory
    d_items =
      cudf::detail::make_device_uvector(items, stream, cudf::get_current_device_resource_ref());
  }

  format_item const* compiled_format_items() { return d_items.data(); }

  [[nodiscard]] size_type items_count() const { return static_cast<size_type>(d_items.size()); }
};

template <typename T>
__device__ void dissect_duration(T duration, duration_component* timeparts)
{
  timeparts->is_negative = (duration < T{0});
  timeparts->day         = cuda::std::chrono::duration_cast<duration_D>(duration).count();

  if (cuda::std::is_same_v<T, duration_D>) return;

  duration_s seconds = cuda::std::chrono::duration_cast<duration_s>(duration);
  timeparts->hour =
    (cuda::std::chrono::duration_cast<cuda::std::chrono::hours>(seconds) % duration_D(1)).count();
  timeparts->minute = (cuda::std::chrono::duration_cast<cuda::std::chrono::minutes>(seconds) %
                       cuda::std::chrono::hours(1))
                        .count();
  timeparts->second = (seconds % cuda::std::chrono::minutes(1)).count();
  if (not cuda::std::is_same_v<T, duration_s>) {
    timeparts->subsecond = (duration % duration_s(1)).count();
  }
}

namespace {
template <typename T>
struct from_durations_fn {
  column_device_view d_durations;
  format_item const* d_format_items;
  size_type items_count;
  size_type* d_sizes{};
  char* d_chars{};
  cudf::detail::input_offsetalator d_offsets;

  __device__ int8_t format_length(char format_char, duration_component const* const timeparts) const
  {
    switch (format_char) {
      case '-': return timeparts->is_negative; break;
      case 'D': return count_digits(timeparts->day) - (timeparts->day < 0); break;
      case 'S':
        return 2 + (timeparts->subsecond == 0 ? 0 : [] {
                 if (cuda::std::is_same_v<T, duration_ms>) return 3 + 1;  // +1 is for dot
                 if (cuda::std::is_same_v<T, duration_us>) return 6 + 1;  // +1 is for dot
                 if (cuda::std::is_same_v<T, duration_ns>) return 9 + 1;  // +1 is for dot
                 return 0;
               }());
        break;
      default: return 2;
    }
  }

  // utility to create (optionally) 0-padded integers (up to 10 chars) without negative sign.
  // min_digits==-1 indicates no 0-padding.
  __device__ char* int2str(char* str, int min_digits, int32_t value)
  {
    constexpr int MAX_DIGITS = 10;  // largest 32-bit integer is 10 digits
    assert(min_digits <= MAX_DIGITS);
    if (value == 0) {
      do {
        *str++ = '0';
      } while (--min_digits > 0);
      return str;
    }

    char digits[MAX_DIGITS] = {'0', '0', '0', '0', '0', '0', '0', '0', '0', '0'};
    int digits_idx          = 0;
    while (value != 0) {
      assert(digits_idx < MAX_DIGITS);
      digits[digits_idx++] = '0' + cuda::std::abs(value % 10);
      // next digit
      value = value / 10;
    }
    digits_idx = cuda::std::max(digits_idx, min_digits);
    // digits are backwards, reverse the string into the output
    while (digits_idx-- > 0)
      *str++ = digits[digits_idx];
    return str;
  }

  __device__ char* int_to_2digitstr(char* str, int8_t value)
  {
    assert(value >= -99 && value <= 99);
    value  = cuda::std::abs(value);
    str[0] = '0' + value / 10;
    str[1] = '0' + value % 10;
    return str + 2;
  }

  inline __device__ char* day(char* ptr, duration_component const* timeparts)
  {
    return int2str(ptr, -1, timeparts->day);
  }

  inline __device__ char* hour_12(char* ptr, duration_component const* timeparts)
  {
    return int_to_2digitstr(ptr, timeparts->hour % 12);
  }
  inline __device__ char* hour_24(char* ptr, duration_component const* timeparts)
  {
    return int_to_2digitstr(ptr, timeparts->hour);
  }
  inline __device__ char* am_or_pm(char* ptr, duration_component const* timeparts)
  {
    *ptr++ = (timeparts->hour / 12 == 0 ? 'A' : 'P');
    *ptr++ = 'M';
    return ptr;
  }
  inline __device__ char* minute(char* ptr, duration_component const* timeparts)
  {
    return int_to_2digitstr(ptr, timeparts->minute);
  }
  inline __device__ char* second(char* ptr, duration_component const* timeparts)
  {
    return int_to_2digitstr(ptr, timeparts->second);
  }

  inline __device__ char* subsecond(char* ptr, duration_component const* timeparts)
  {
    if (timeparts->subsecond == 0) return ptr;
    int const digits = format_length('S', timeparts) - 3;
    *ptr             = '.';
    auto value       = timeparts->subsecond;
    for (int idx = digits; idx > 0; idx--) {
      *(ptr + idx) = '0' + cuda::std::abs(value % 10);
      value /= 10;
    }
    return ptr + digits + 1;
  }

  __device__ char* format_from_parts(duration_component const* timeparts, char* ptr)
  {
    for (size_t idx = 0; idx < items_count; ++idx) {
      auto item = d_format_items[idx];
      if (item.item_type == format_char_type::literal) {
        *ptr++ = item.value;
        continue;
      }
      // special logic for each specifier
      switch (item.value) {
        case 'D':  // days
          ptr = day(ptr, timeparts);
          break;
        case '-':  // - if value is negative
          if (timeparts->is_negative) *ptr++ = '-';
          break;
        case 'H':  // 24-hour
          ptr = hour_24(ptr, timeparts);
          break;
        case 'I':  // 12-hour
          ptr = hour_12(ptr, timeparts);
          break;
        case 'M':  // minute
          ptr = minute(ptr, timeparts);
          break;
        case 'S':  // second
          ptr = second(ptr, timeparts);
          if (item.length == 2) break;
        case 'f':  // sub-second
          ptr = subsecond(ptr, timeparts);
          break;
        case 'p': ptr = am_or_pm(ptr, timeparts); break;
        case 'R':  // HH:MM 24-hour
          ptr    = hour_24(ptr, timeparts);
          *ptr++ = ':';
          ptr    = minute(ptr, timeparts);
          break;
        case 'T':  // HH:MM:SS 24-hour
          ptr    = hour_24(ptr, timeparts);
          *ptr++ = ':';
          ptr    = minute(ptr, timeparts);
          *ptr++ = ':';
          ptr    = second(ptr, timeparts);
          break;
        case 'r':  // HH:MM:SS AM/PM 12-hour
          ptr    = hour_12(ptr, timeparts);
          *ptr++ = ':';
          ptr    = minute(ptr, timeparts);
          *ptr++ = ':';
          ptr    = second(ptr, timeparts);
          *ptr++ = ' ';
          ptr    = am_or_pm(ptr, timeparts);
          break;
        default:  // ignore everything else
          break;
      }
    }
    return ptr;
  }

  __device__ size_type string_size(T duration)
  {
    duration_component timeparts = {0};  // days, hours, minutes, seconds, subseconds(9)
    dissect_duration(duration, &timeparts);
    return thrust::transform_reduce(
      thrust::seq,
      d_format_items,
      d_format_items + items_count,
      [this, &timeparts] __device__(format_item item) -> size_type {
        if (item.item_type == format_char_type::literal) { return 1; }
        return (item.length != -1) ? item.length : format_length(item.value, &timeparts);
      },
      size_type{0},
      cuda::std::plus<size_type>());
  }

  __device__ void set_chars(size_type idx)
  {
    auto duration                = d_durations.template element<T>(idx);
    duration_component timeparts = {0};  // days, hours, minutes, seconds, subseconds(9)
    dissect_duration(duration, &timeparts);
    // convert to characters
    format_from_parts(&timeparts, d_chars + d_offsets[idx]);
  }

  __device__ void operator()(size_type idx)
  {
    if (d_durations.is_null(idx)) {
      if (d_chars == nullptr) { d_sizes[idx] = 0; }
      return;
    }

    if (d_chars != nullptr) {
      set_chars(idx);
    } else {
      d_sizes[idx] = string_size(d_durations.template element<T>(idx));
    }
  }
};
}  // namespace

/**
 * @brief This dispatch method is for converting durations into strings.
 *
 * The template function declaration ensures only duration types are used.
 */
struct dispatch_from_durations_fn {
  template <typename T>
  std::unique_ptr<column> operator()(column_view const& durations,
                                     std::string_view format,
                                     cuda::stream_ref stream,
                                     rmm::device_async_resource_ref mr) const
    requires(cudf::is_duration<T>())
  {
    CUDF_EXPECTS(!format.empty(), "Format parameter must not be empty.");

    format_compiler compiler(format, stream);
    auto d_format_items = compiler.compiled_format_items();

    size_type strings_count = durations.size();
    auto column             = column_device_view::create(durations, stream);
    auto d_column           = *column;

    // copy null mask
    rmm::device_buffer null_mask = cudf::detail::copy_bitmask(durations, stream, mr);

    auto [offsets, chars] =
      make_strings_children(from_durations_fn<T>{d_column, d_format_items, compiler.items_count()},
                            strings_count,
                            stream,
                            mr);

    return make_strings_column(strings_count,
                               std::move(offsets),
                               chars.release(),
                               durations.null_count(),
                               std::move(null_mask));
  }

  // non-duration types throw an exception
  template <typename T, typename... Args>
  std::unique_ptr<column> operator()(Args&&...) const
    requires(not cudf::is_duration<T>())
  {
    CUDF_FAIL("Values for from_durations function must be a duration type.");
  }
};

// this parses duration string into a duration integer
template <typename T>  // duration type
struct parse_duration {
  column_device_view const d_strings;
  format_item const* d_format_items;
  size_type items_count;

  // function to parse string (maximum 10 digits) to integer.
  __device__ int32_t str2int(char const* str, int8_t max_bytes, int8_t& actual_length)
  {
    char const* ptr = (*str == '-' || *str == '+') ? str + 1 : str;
    int32_t value   = 0;
    for (int8_t idx = 0; idx < max_bytes; ++idx) {
      char chr = *ptr++;
      if (chr < '0' || chr > '9') {
        ptr--;  // roll back
        break;
      }
      value = (value * 10) + static_cast<int32_t>(chr - '0');
    }
    actual_length += (ptr - str);
    return (*str == '-') ? -value : value;
  }

  // function to parse fraction of decimal value with trailing zeros removed.
  __device__ int32_t str2int_fixed(char const* str,
                                   int8_t fixed_width,
                                   size_type string_length,
                                   int8_t& actual_length)
  {
    char const* ptr = (*str == '.') ? str + 1 : str;
    int32_t value   = 0;
    // parse till fixed_width or end of string.
    for (int8_t idx = 0; idx < fixed_width && idx < string_length; ++idx) {
      char chr = *ptr++;
      if (chr < '0' || chr > '9') {
        ptr--;  // roll back
        break;
      }
      value = (value * 10) + static_cast<int32_t>(chr - '0');
    }
    auto parsed_length = ptr - str;
    // compensate for missing trailing zeros
    if (parsed_length < fixed_width)
      value *= cudf::detail::powers_of_ten[fixed_width - parsed_length];
    actual_length += parsed_length;
    return value;
  }

  // parse 2 digit string to integer
  __device__ int8_t parse_2digit_int(char const* str, int8_t& actual_length)
  {
    char const* ptr = (*str == '-' || *str == '+') ? str + 1 : str;
    int8_t value    = 0;
    if (*ptr >= '0' && *ptr <= '9') value = (value * 10) + static_cast<int32_t>(*ptr++ - '0');
    if (*ptr >= '0' && *ptr <= '9') value = (value * 10) + static_cast<int32_t>(*ptr++ - '0');
    actual_length += (ptr - str);
    return (*str == '-') ? -value : value;
  }
  inline __device__ int8_t parse_hour(char const* str, int8_t& actual_length)
  {
    return parse_2digit_int(str, actual_length);
  }
  inline __device__ int8_t parse_minute(char const* str, int8_t& actual_length)
  {
    return parse_2digit_int(str, actual_length);
  }
  inline __device__ int8_t parse_second(char const* str, int8_t& actual_length)
  {
    return parse_2digit_int(str, actual_length);
  }

  // Walk the format_items to read the datetime string.
  // Returns 0 if all ok.
  __device__ int parse_into_parts(string_view const& d_string, duration_component* timeparts)
  {
    auto ptr    = d_string.data();
    auto length = d_string.size_bytes();
    int8_t hour_shift{0};
    for (size_type idx = 0; idx < items_count; ++idx) {
      auto item = d_format_items[idx];
      if (length < item.length) return 1;
      if (item.item_type == format_char_type::literal) {  // static character we'll just skip;
        // consume item.length bytes from string
        ptr += item.length;
        length -= item.length;
        continue;
      }
      timeparts->is_negative |= (*ptr == '-');

      // special logic for each specifier
      int8_t item_length{0};
      switch (item.value) {
        case 'D':  // day
          timeparts->day = str2int(ptr, 11, item_length);
          break;
        case '-':  // skip
          item_length = (*ptr == '-');
          break;
        case 'H':  // 24-hour
          timeparts->hour = parse_hour(ptr, item_length);
          hour_shift      = 0;
          break;
        case 'I':  // 12-hour
          timeparts->hour = parse_hour(ptr, item_length);
          break;
        case 'M':  // minute
          timeparts->minute = parse_minute(ptr, item_length);
          break;
        case 'S':  // [-]SS[.mmm][uuu][nnn]
          timeparts->second = parse_second(ptr, item_length);
          if ((item_length < length) && *(ptr + item_length) == '.') {
            item_length++;
            int64_t nanoseconds = str2int_fixed(
              ptr + item_length, 9, length - item_length, item_length);  // normalize to nanoseconds
            timeparts->subsecond = nanoseconds;
          }
          break;
        case 'p':  // AM/PM
          if (*ptr == 'P' && *(ptr + 1) == 'M')
            hour_shift = 12;
          else
            hour_shift = 0;
          item_length = 2;
          break;
        case 'R':  // [-]HH:SS
          timeparts->hour = parse_hour(ptr, item_length);
          hour_shift      = 0;
          item_length++;  // :
          timeparts->minute = parse_minute(ptr + item_length, item_length);
          break;
        case 'T':  // [-]HH:MM:SS
          timeparts->hour = parse_hour(ptr, item_length);
          hour_shift      = 0;
          item_length++;  // :
          timeparts->minute = parse_minute(ptr + item_length, item_length);
          item_length++;  // :
          timeparts->second = parse_second(ptr + item_length, item_length);
          break;
        case 'r':  // hh:MM:SS AM/PM
          timeparts->hour = parse_hour(ptr, item_length);
          item_length++;  // :
          timeparts->minute = parse_minute(ptr + item_length, item_length);
          item_length++;  // :
          timeparts->second = parse_second(ptr + item_length, item_length);
          item_length++;  // space
          if (*(ptr + item_length) == 'P' && *(ptr + item_length + 1) == 'M')
            hour_shift = 12;
          else
            hour_shift = 0;
          item_length += 2;
          break;
        default: return 3;
      }
      ptr += item_length;
      length -= item_length;
    }
    // negate all if duration has negative sign
    if (timeparts->is_negative) {
      auto negate          = [](auto i) { return (i < 0 ? i : -i); };
      timeparts->day       = negate(timeparts->day);
      timeparts->hour      = negate(timeparts->hour);
      timeparts->minute    = negate(timeparts->minute);
      timeparts->second    = negate(timeparts->second);
      timeparts->subsecond = negate(timeparts->subsecond);
      hour_shift           = -hour_shift;
    }
    timeparts->hour += hour_shift;
    return 0;
  }

  inline __device__ int64_t duration_from_parts(duration_component const* timeparts)
  {
    int32_t days  = timeparts->day;
    auto hour     = timeparts->hour;
    auto minute   = timeparts->minute;
    auto second   = timeparts->second;
    auto duration = duration_D(days) + cuda::std::chrono::hours(hour) +
                    cuda::std::chrono::minutes(minute) + duration_s(second);
    if (cuda::std::is_same_v<T, duration_D>)
      return cuda::std::chrono::duration_cast<duration_D>(duration).count();
    else if (cuda::std::is_same_v<T, duration_s>)
      return cuda::std::chrono::duration_cast<duration_s>(duration).count();

    duration_ns subsecond(timeparts->subsecond);  // ns
    if (cuda::std::is_same_v<T, duration_ms>) {
      return cuda::std::chrono::duration_cast<duration_ms>(duration + subsecond).count();
    } else if (cuda::std::is_same_v<T, duration_us>) {
      return cuda::std::chrono::duration_cast<duration_us>(duration + subsecond).count();
    } else if (cuda::std::is_same_v<T, duration_ns>)
      return cuda::std::chrono::duration_cast<duration_ns>(duration + subsecond).count();
    return cuda::std::chrono::duration_cast<duration_ns>(duration + subsecond).count();
  }

  __device__ T operator()(size_type idx)
  {
    if (d_strings.is_null(idx)) return T{0};
    string_view d_str = d_strings.element<string_view>(idx);
    if (d_str.empty()) return T{0};
    //
    duration_component timeparts = {0};
    if (parse_into_parts(d_str, &timeparts)) return T{0};  // unexpected parse case
    //
    return static_cast<T>(duration_from_parts(&timeparts));
  }
};

/**
 * @brief This dispatch method is for converting strings to durations.
 *
 * The template function declaration ensures only duration types are used.
 */
struct dispatch_to_durations_fn {
  template <typename T>
  void operator()(column_device_view const& d_strings,
                  std::string_view format,
                  mutable_column_view& results_view,
                  cuda::stream_ref stream) const
    requires(cudf::is_duration<T>())
  {
    format_compiler compiler(format, stream);
    auto d_items   = compiler.compiled_format_items();
    auto d_results = results_view.data<T>();
    parse_duration<T> pfn{d_strings, d_items, compiler.items_count()};
    thrust::transform(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                      cuda::counting_iterator<size_type>{0},
                      cuda::counting_iterator<size_type>{results_view.size()},
                      d_results,
                      pfn);
  }
  template <typename T>
  void operator()(column_device_view const&,
                  std::string_view,
                  mutable_column_view&,
                  cuda::stream_ref) const
    requires(not cudf::is_duration<T>())
  {
    CUDF_FAIL("Only durations type are expected for to_durations function");
  }
};

}  // namespace

std::unique_ptr<column> from_durations(column_view const& durations,
                                       std::string_view format,
                                       cuda::stream_ref stream,
                                       rmm::device_async_resource_ref mr)
{
  size_type strings_count = durations.size();
  if (strings_count == 0) return make_empty_column(type_id::STRING);

  return type_dispatcher(
    durations.type(), dispatch_from_durations_fn{}, durations, format, stream, mr);
}

std::unique_ptr<column> to_durations(strings_column_view const& input,
                                     data_type duration_type,
                                     std::string_view format,
                                     cuda::stream_ref stream,
                                     rmm::device_async_resource_ref mr)
{
  size_type strings_count = input.size();
  if (strings_count == 0) {
    return make_duration_column(duration_type, 0, mask_state::UNALLOCATED, stream, mr);
  }

  CUDF_EXPECTS(!format.empty(), "Format parameter must not be empty.");

  auto strings_column = column_device_view::create(input.parent(), stream);
  auto d_column       = *strings_column;

  auto results      = make_duration_column(duration_type,
                                      strings_count,
                                      cudf::detail::copy_bitmask(input.parent(), stream, mr),
                                      input.null_count(),
                                      stream,
                                      mr);
  auto results_view = results->mutable_view();
  cudf::type_dispatcher(
    duration_type, dispatch_to_durations_fn(), d_column, format, results_view, stream);
  results->set_null_count(input.null_count());
  return results;
}

}  // namespace detail

std::unique_ptr<column> from_durations(column_view const& durations,
                                       std::string_view format,
                                       cuda::stream_ref stream,
                                       rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::from_durations(durations, format, stream, mr);
}

std::unique_ptr<column> to_durations(strings_column_view const& input,
                                     data_type duration_type,
                                     std::string_view format,
                                     cuda::stream_ref stream,
                                     rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::to_durations(input, duration_type, format, stream, mr);
}

}  // namespace strings
}  // namespace cudf
