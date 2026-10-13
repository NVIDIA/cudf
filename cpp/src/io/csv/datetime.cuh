/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "io/utilities/parsing_utils.cuh"

#include <cudf/fixed_point/fixed_point.hpp>
#include <cudf/wrappers/durations.hpp>

#include <thrust/equal.h>
#include <thrust/execution_policy.h>
#include <thrust/find.h>

namespace cudf {
namespace io {

/**
 * @brief Parses non-negative integral vales.
 *
 * This helper function is only intended to handle positive integers. The input
 * character string is expected to be well-formed.
 *
 * @param begin Pointer to the first element of the string
 * @param end Pointer to the first element after the string
 * @return The parsed and converted value
 */
template <typename T>
__inline__ __device__ T to_non_negative_integer(char const* begin, char const* end)
{
  T value = 0;

  for (; begin < end; ++begin) {
    if (*begin >= '0' && *begin <= '9') {
      value *= 10;
      value += *begin - '0';
    }
  }

  return value;
}

/**
 * @brief Extracts the Day, Month, and Year from a string.
 *
 * This function takes a string and produces a `year_month_day` representation.
 * Acceptable formats are a combination of `YYYY`, `M`, `MM`, `D` and `DD` with
 * `/` or `-` as separators. Data with only year and month (no day) is also valid.
 *
 * @param begin Pointer to the first element of the string
 * @param end Pointer to the first element after the string
 * @param dayfirst Flag indicating that first field is the day
 * @return Extracted year, month and day in `cuda::std::chrono::year_month_day` format
 */
__inline__ __device__ cuda::std::chrono::year_month_day extract_date(char const* begin,
                                                                     char const* end,
                                                                     bool dayfirst)
{
  using namespace cuda::std::chrono;

  char sep = '/';

  auto sep_pos = thrust::find(thrust::seq, begin, end, sep);

  if (sep_pos == end) {
    sep     = '-';
    sep_pos = thrust::find(thrust::seq, begin, end, sep);
  }

  year y;
  month m;
  day d;

  //--- is year the first filed?
  if ((sep_pos - begin) == 4) {
    y = year{to_non_negative_integer<int32_t>(begin, sep_pos)};  //  year is signed

    // Month
    auto s2 = sep_pos + 1;
    sep_pos = thrust::find(thrust::seq, s2, end, sep);

    if (sep_pos == end) {
      //--- Data is just Year and Month - no day
      m = month{to_non_negative_integer<uint32_t>(s2, end)};  // month and day are unsigned
      d = day{1};

    } else {
      m = month{to_non_negative_integer<uint32_t>(s2, sep_pos)};
      d = day{to_non_negative_integer<uint32_t>((sep_pos + 1), end)};
    }

  } else {
    //--- if the dayfirst flag is set, then restricts the format options
    if (dayfirst) {
      d = day{to_non_negative_integer<uint32_t>(begin, sep_pos)};

      auto s2 = sep_pos + 1;
      sep_pos = thrust::find(thrust::seq, s2, end, sep);

      m = month{to_non_negative_integer<uint32_t>(s2, sep_pos)};
      y = year{to_non_negative_integer<int32_t>((sep_pos + 1), end)};

    } else {
      m = month{to_non_negative_integer<uint32_t>(begin, sep_pos)};

      auto s2 = sep_pos + 1;
      sep_pos = thrust::find(thrust::seq, s2, end, sep);

      if (sep_pos == end) {
        //--- Data is just Year and Month - no day
        y = year{to_non_negative_integer<int32_t>(s2, end)};
        d = day{1};

      } else {
        d = day{to_non_negative_integer<uint32_t>(s2, sep_pos)};
        y = year{to_non_negative_integer<int32_t>((sep_pos + 1), end)};
      }
    }
  }

  return year_month_day{y, m, d};
}

/**
 * @brief Parses a string to extract the hour, minute, second and millisecond time field
 * values of a day.
 *
 * Incoming format is expected to be `HH:MM:SS.MS`, with the latter second and millisecond fields
 * optional. Each time field can be a single, double, or triple (in the case of milliseconds)
 * digits. 12-hr and 24-hr time format is detected via the absence or presence of AM/PM characters
 * at the end.
 *
 * @param begin Pointer to the first element of the string
 * @param end Pointer to the first element after the string
 * @return Extracted hours, minutes, seconds and milliseconds of `chrono::hh_mm_ss` type with a
 * precision of milliseconds
 */
__inline__ __device__ cuda::std::chrono::hh_mm_ss<duration_ms> extract_time_of_day(
  char const* begin, char const* end)
{
  constexpr char sep = ':';

  // Adjust for AM/PM and any whitespace before
  duration_h d_h{0};
  auto last = end - 1;
  if (*last == 'M' || *last == 'm') {
    if (*(last - 1) == 'P' || *(last - 1) == 'p') { d_h = duration_h{12}; }
    last = last - 2;
    while (*last == ' ') {
      --last;
    }
  }
  end = last + 1;

  // Find hour-minute separator
  auto const hm_sep = thrust::find(thrust::seq, begin, end, sep);
  // Extract hours
  d_h += cudf::duration_h{to_non_negative_integer<int>(begin, hm_sep)};

  duration_m d_m{0};
  duration_s d_s{0};
  duration_ms d_ms{0};

  // Find minute-second separator (if present)
  auto const ms_sep = thrust::find(thrust::seq, hm_sep + 1, end, sep);
  if (ms_sep == end) {
    d_m = duration_m{to_non_negative_integer<int32_t>(hm_sep + 1, end)};
  } else {
    d_m = duration_m{to_non_negative_integer<int32_t>(hm_sep + 1, ms_sep)};

    // Find second-millisecond separator (if present)
    auto const sms_sep = thrust::find(thrust::seq, ms_sep + 1, end, '.');
    if (sms_sep == end) {
      d_s = duration_s{to_non_negative_integer<int64_t>(ms_sep + 1, end)};
    } else {
      d_s  = duration_s{to_non_negative_integer<int64_t>(ms_sep + 1, sms_sep)};
      d_ms = duration_ms{to_non_negative_integer<int64_t>(sms_sep + 1, end)};
    }
  }
  return cuda::std::chrono::hh_mm_ss<duration_ms>{d_h + d_m + d_s + d_ms};
}

/**
 * @brief Checks whether `c` is decimal digit
 */
__device__ constexpr bool is_digit(char c) { return c >= '0' and c <= '9'; }

/**
 * @brief Parses a datetime string and computes the corresponding timestamp.
 *
 * Acceptable date formats are a combination of `YYYY`, `M`, `MM`, `D` and `DD` with `/` or `-` as
 * separators. Input with only year and month (no day) is also valid. Character `T` or blank space
 * is expected to be the separator between date and time of day. Optional time of day information
 * like hours, minutes, seconds and milliseconds are expected to be `HH:MM:SS.MS`. Each time field
 * can be a single, double, or triple (in the case of milliseconds) digits. 12-hr and 24-hr time
 * format is detected via the absence or presence of AM/PM characters at the end.
 *
 * @tparam timestamp_type Type of output timestamp
 * @param begin Pointer to the first element of the string
 * @param end Pointer to the first element after the string
 * @param dayfirst Flag to indicate day/month or month/day order
 * @return Timestamp converted to `timestamp_type`
 */
template <typename timestamp_type>
__inline__ __device__ timestamp_type to_timestamp(char const* begin, char const* end, bool dayfirst)
{
  using duration_type = typename timestamp_type::duration;

  auto sep_pos = end;

  // Find end of the date portion
  int count        = 0;
  bool digits_only = true;
  for (auto i = begin; i < end; ++i) {
    digits_only = digits_only and is_digit(*i);
    if (*i == 'T') {
      sep_pos = i;
      break;
    } else if (count == 3 && *i == ' ') {
      sep_pos = i;
      break;
    } else if ((*i == '/' || *i == '-') || (count == 2 && *i != ' ')) {
      count++;
    }
  }

  // Exit if the input string is digit-only
  if (digits_only) {
    return timestamp_type{
      duration_type{to_non_negative_integer<typename timestamp_type::rep>(begin, end)}};
  }

  auto ymd = extract_date(begin, sep_pos, dayfirst);
  timestamp_type answer{cuda::std::chrono::sys_days{ymd}};

  // Extract time only if separator is present
  if (sep_pos != end) {
    auto t = extract_time_of_day(sep_pos + 1, end);
    answer += cuda::std::chrono::duration_cast<duration_type>(t.to_duration());
  }

  return answer;
}

/**
 * @brief Parses the input string into an integral value of the given type.
 *
 * Moves the `begin` iterator past the parsed value.
 *
 * @param[in, out] begin Pointer to the first element of the string
 * @param end Pointer to the first element after the string
 * @return The parsed and converted value
 */
template <typename T>
__inline__ __device__ T parse_integer(char const** begin, char const* end)
{
  bool const is_negative = (**begin == '-');
  T value                = 0;

  auto cur = *begin + is_negative;
  while (cur < end) {
    if (*cur >= '0' && *cur <= '9') {
      value *= 10;
      value += *cur - '0';
    } else
      break;
    ++cur;
  }
  *begin = cur;

  return is_negative ? -value : value;
}

/**
 * @brief Parses the input string into an integral value of the given type if the delimiter is
 * present.
 *
 * Moves the `begin` iterator past the parsed value.
 *
 * @param[in, out] begin Pointer to the first element of the string
 * @param end Pointer to the first element after the string
 * @param delimiter delimiter character
 * @return The parsed and converted value, zero is delimiter is not present
 */
template <typename T>
__inline__ __device__ T parse_optional_integer(char const** begin, char const* end, char delimiter)
{
  if (**begin != delimiter) { return 0; }

  ++(*begin);
  return parse_integer<T>(begin, end);
}

/**
 * @brief Finds the first element after the leading space characters.
 *
 * @param begin Pointer to the first element of the string
 * @param end Pointer to the first element after the string
 * @return Pointer to the first character excluding any leading spaces
 */
__inline__ __device__ auto skip_spaces(char const* begin, char const* end)
{
  return thrust::find_if(thrust::seq, begin, end, [](auto elem) { return elem != ' '; });
}

/**
 * @brief Excludes the prefix from the input range if the string starts with the prefix.
 *
 * @tparam N length of the prefix, plus one
 * @param begin Pointer to the first element of the string
 * @param end Pointer to the first element after the string
 * @param prefix String we're searching for at the start of the input range
 * @return Pointer to the start of the string excluding the prefix
 */
template <int N>
__inline__ __device__ auto skip_if_starts_with(char const* begin,
                                               char const* end,
                                               char const (&prefix)[N])
{
  static constexpr size_t prefix_len = N - 1;
  if (end - begin < prefix_len) return begin;
  return thrust::equal(thrust::seq, begin, begin + prefix_len, prefix) ? begin + prefix_len : begin;
}

/**
 * @brief Parses the input string into a duration of `duration_type`.
 *
 * The expected format can be one of the following: `DD days`, `DD days +HH:MM:SS.NS`, `DD days
 * HH:MM::SS.NS`, `HH:MM::SS.NS` and digits-only string. Note `DD` and optional `NS` field can
 * contain arbitrary number of digits while `HH`, `MM` and `SS` can be single or double digits.
 *
 * @tparam duration_type Type of the parsed duration
 * @param begin Pointer to the first element of the string
 * @param end Pointer to the first element after the string
 * @return The parsed duration in `duration_type`
 */
template <typename duration_type>
__inline__ __device__ duration_type to_duration(char const* begin, char const* end)
{
  using cuda::std::chrono::duration_cast;

  // %d days [+]%H:%M:%S.n => %d days, %d days [+]%H:%M:%S,  %H:%M:%S.n, %H:%M:%S, %value.
  constexpr char sep = ':';

  // single pass to parse days, hour, minute, seconds, nanosecond
  auto cur         = begin;
  auto const value = parse_integer<int32_t>(&cur, end);
  cur              = skip_spaces(cur, end);
  if (std::is_same_v<duration_type, cudf::duration_D> || cur >= end) {
    return duration_type{static_cast<typename duration_type::rep>(value)};
  }

  // " days [+]"
  auto const after_days_sep     = skip_if_starts_with(cur, end, "days");
  auto const has_days_seperator = (after_days_sep != cur);
  cur                           = skip_spaces(after_days_sep, end);
  cur += (*cur == '+');

  duration_D d_d{0};
  duration_h d_h{0};
  if (has_days_seperator) {
    d_d = duration_D{value};
    d_h = duration_h{parse_integer<int32_t>(&cur, end)};
  } else {
    d_h = duration_h{value};
  }

  duration_m d_m{parse_optional_integer<int32_t>(&cur, end, sep)};
  duration_s d_s{parse_optional_integer<int64_t>(&cur, end, sep)};

  // Convert all durations to the given type
  auto output_d = duration_cast<duration_type>(d_d + d_h + d_m + d_s);

  if constexpr (std::is_same_v<duration_type, cudf::duration_s>) { return output_d; }

  auto const d_ns = (*cur != '.') ? duration_ns{0} : [&]() {
    auto const start_subsecond     = ++cur;
    auto const unscaled_subseconds = parse_integer<int64_t>(&cur, end);
    auto const scale               = min(9L, cur - start_subsecond) - 9;
    auto const rescaled = numeric::decimal64{unscaled_subseconds, numeric::scale_type{scale}};
    return duration_ns{rescaled.value()};
  }();

  return output_d + duration_cast<duration_type>(d_ns);
}

}  // namespace io
}  // namespace cudf
