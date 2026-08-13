/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "utilities/time_utils.cuh"

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/strings/convert/convert_datetime.hpp>
#include <cudf/strings/detail/converters.hpp>
#include <cudf/strings/detail/strings_children.cuh>
#include <cudf/strings/detail/utilities.cuh>
#include <cudf/strings/string_view.cuh>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/span.hpp>
#include <cudf/utilities/type_dispatcher.hpp>
#include <cudf/wrappers/timestamps.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>

#include <cuda/iterator>
#include <cuda/std/algorithm>
#include <cuda/std/optional>
#include <cuda/std/utility>
#include <thrust/execution_policy.h>
#include <thrust/logical.h>
#include <thrust/transform.h>

#include <map>
#include <numeric>
#include <vector>

namespace cudf {
namespace strings {
namespace detail {
namespace {

/**
 * @brief Structure of date/time components
 */
struct timestamp_components {
  int16_t year;
  int8_t month;
  int8_t day;
  int16_t day_of_year;
  int8_t hour;
  int8_t minute;
  int8_t second;
  int8_t week;     ///< week of the year
  int8_t weekday;  ///< day of the week
  int32_t subsecond;
  int32_t tz_minutes;
};

enum class format_char_type : int8_t {
  literal,   ///< literal char type passed through
  specifier  ///< timestamp format specifier
};

/**
 * @brief Represents a format specifier or literal from a timestamp format string.
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
  static format_item new_literal(char literal)
  {
    return format_item{format_char_type::literal, literal, 1};
  }
};

/**
 * @brief The format-compiler parses a timestamp format string into a vector of
 * `format_items`.
 *
 * The vector of `format_items` is used when parsing a string into timestamp
 * components and when formatting a string from timestamp components.
 */
using specifier_map = std::map<char, int8_t>;

struct format_compiler {
  std::string_view const format;
  rmm::device_uvector<format_item> d_items;

  // clang-format off
  // The specifiers are documented here (not all are supported):
  // https://en.cppreference.com/cpp/chrono/system_clock/formatter
  specifier_map specifiers = {
    {'Y', 4}, {'y', 2}, {'m', 2}, {'d', 2}, {'H', 2}, {'I', 2}, {'M', 2},
    {'S', 2}, {'f', 6}, {'z', 5}, {'Z', 3}, {'p', 2}, {'j', 3},
    {'W', 2}, {'w', 1}, {'U', 2}, {'u', 1}};
  // clang-format on

  format_compiler(std::string_view fmt,
                  rmm::cuda_stream_view stream,
                  specifier_map extra_specifiers = {})
    : format(fmt), d_items(0, stream)
  {
    specifiers.insert(extra_specifiers.begin(), extra_specifiers.end());
    auto items  = cudf::detail::make_empty_host_vector<format_item>(format.length(), stream);
    auto str    = format.data();
    auto length = format.length();
    while (length > 0) {
      char ch = *str++;
      length--;

      // first check for a literal character
      if (ch != '%') {
        items.push_back(format_item::new_literal(ch));
        continue;
      }
      CUDF_EXPECTS(length > 0, "Unfinished specifier in timestamp format", std::invalid_argument);

      ch = *str++;
      length--;
      if (ch == '%')  // escaped % char
      {
        items.push_back(format_item::new_literal(ch));
        continue;
      }
      if (ch >= '0' && ch <= '9') {
        CUDF_EXPECTS(*str == 'f',
                     "precision not supported for specifier: " + std::string(1, *str),
                     std::invalid_argument);
        specifiers[*str] = static_cast<int8_t>(ch - '0');
        ch               = *str++;
        length--;
      }

      // check if the specifier found is supported
      CUDF_EXPECTS(specifiers.find(ch) != specifiers.end(),
                   "invalid format specifier: " + std::string(1, ch),
                   std::invalid_argument);

      // create the format item for this specifier
      items.push_back(format_item::new_specifier(ch, specifiers[ch]));
    }

    // copy format_items to device memory
    d_items =
      cudf::detail::make_device_uvector(items, stream, cudf::get_current_device_resource_ref());
  }

  device_span<format_item const> format_items() { return device_span<format_item const>(d_items); }

  [[nodiscard]] int8_t subsecond_precision() const { return specifiers.at('f'); }
};

/**
 * @brief Specialized function to return the integer value reading up to the specified
 * bytes or until an invalid character is encountered.
 *
 * @param str Beginning of characters to read.
 * @param bytes Number of bytes in str to read.
 * @return Integer value of valid characters read and how many bytes were not read.
 */
__device__ cuda::std::pair<int32_t, size_type> parse_int(char const* str, size_type bytes)
{
  int32_t value = 0;
  while (bytes-- > 0) {
    char chr = *str++;
    if (chr < '0' || chr > '9') break;
    value = (value * 10) + static_cast<int32_t>(chr - '0');
  }
  return cuda::std::make_pair(value, bytes + 1);
}

/**
 * @brief This parses date/time characters into a timestamp integer
 *
 * @tparam T cudf::timestamp type
 */
template <typename T>
struct parse_datetime {
  column_device_view const d_strings;
  device_span<format_item const> const d_format_items;
  int8_t const subsecond_precision;

  __device__ bool format_contains(char specifier) const
  {
    return thrust::find_if(thrust::seq,
                           d_format_items.begin(),
                           d_format_items.end(),
                           [specifier](auto const item) { return item.value == specifier; }) !=
           d_format_items.end();
  }

  // Walk the format_items to parse the string into date/time components
  [[nodiscard]] __device__ timestamp_components parse_into_parts(string_view const& d_string) const
  {
    timestamp_components timeparts = {1970, 1, 1, 0};  // init to epoch time

    auto ptr    = d_string.data();
    auto length = d_string.size_bytes();
    for (auto item : d_format_items) {
      if (item.value != 'f')
        item.length =
          static_cast<int8_t>(cuda::std::min(static_cast<size_type>(item.length), length));

      if (item.item_type == format_char_type::literal) {
        // static character we'll just skip;
        // consume item.length bytes from the input string
        ptr += item.length;
        length -= item.length;
        continue;
      }

      size_type bytes_read = item.length;  // number of bytes processed
      // special logic for each specifier
      switch (item.value) {
        case 'Y': {
          auto const [year, left] = parse_int(ptr, item.length);
          timeparts.year          = static_cast<int16_t>(year);
          bytes_read -= left;
          break;
        }
        case 'y': {
          auto const [year, left] = parse_int(ptr, item.length);
          timeparts.year          = static_cast<int16_t>(year + (year < 69 ? 2000 : 1900));
          bytes_read -= left;
          break;
        }
        case 'm': {
          auto const [month, left] = parse_int(ptr, item.length);
          timeparts.month          = static_cast<int8_t>(month);
          bytes_read -= left;
          break;
        }
        case 'd': {
          auto const [day, left] = parse_int(ptr, item.length);
          timeparts.day          = static_cast<int8_t>(day);
          bytes_read -= left;
          break;
        }
        case 'j': {
          auto const [day, left] = parse_int(ptr, item.length);
          timeparts.day_of_year  = static_cast<int16_t>(day);
          bytes_read -= left;
          break;
        }
        case 'H':
        case 'I': {
          auto const [hour, left] = parse_int(ptr, item.length);
          timeparts.hour          = static_cast<int8_t>(hour);
          bytes_read -= left;
          break;
        }
        case 'M': {
          auto const [minute, left] = parse_int(ptr, item.length);
          timeparts.minute          = static_cast<int8_t>(minute);
          bytes_read -= left;
          break;
        }
        case 'S': {
          auto const [second, left] = parse_int(ptr, item.length);
          timeparts.second          = static_cast<int8_t>(second);
          bytes_read -= left;
          break;
        }
        case 'f': {
          int32_t const read_size =
            cuda::std::min(static_cast<int32_t>(item.length), static_cast<int32_t>(length));
          auto const [fraction, left] = parse_int(ptr, read_size);
          timeparts.subsecond =
            fraction * cudf::detail::powers_of_ten[item.length - read_size + left];
          bytes_read = read_size - left;
          break;
        }
        case 'p': {
          string_view am_pm(ptr, 2);
          auto hour = timeparts.hour;
          if ((am_pm.compare("AM", 2) == 0) || (am_pm.compare("am", 2) == 0)) {
            if (hour == 12) hour = 0;
          } else if (hour < 12)
            hour += 12;
          timeparts.hour = hour;
          break;
        }
        case 'U': [[fallthrough]];  // week of year: Sunday based
        case 'W': {                 // week of year: Monday based
          auto const [week, left] = parse_int(ptr, item.length);
          timeparts.week          = static_cast<int8_t>(week + 1);
          bytes_read -= left;
          break;
        }
        case 'u': [[fallthrough]];  // day of week: Mon(1)-Sat(6),Sun(7)
        case 'w': {                 // day of week; Sun(0),Mon(1)-Sat(6)
          auto const [weekday, left] = parse_int(ptr, item.length);
          timeparts.weekday          =  // 0 is mapped to 7 for chrono library
            static_cast<int8_t>((item.value == 'w' && weekday == 0) ? 7 : weekday);
          bytes_read -= left;
          break;
        }
        case 'z': {
          // 'z' format is +hh:mm -- single sign char and 2 chars each for hour and minute
          if (item.length == 5) {
            auto const sign     = *ptr == '-' ? 1 : -1;
            auto const [hh, lh] = parse_int(ptr + 1, 2);
            auto const [mm, lm] = parse_int(ptr + 3, 2);
            // revert timezone back to UTC
            timeparts.tz_minutes = sign * ((hh * 60) + mm);
            bytes_read -= lh + lm;
          }
          break;
        }
        case 'Z': break;  // skip
        default: break;
      }
      ptr += bytes_read;
      length -= bytes_read;
    }
    return timeparts;
  }

  [[nodiscard]] __device__ int64_t timestamp_from_parts(timestamp_components const& timeparts) const
  {
    // Reference: https://howardhinnant.github.io/date/date.html#Reference
    auto const days = [timeparts, this] {
      // week and weekday prioritize over month/day
      if ((timeparts.week > 0) && (timeparts.weekday > 0)) {
        auto const y = cuda::std::chrono::year{timeparts.year};
        // clang-format off
        auto const start = format_contains('U')
          ? cuda::std::chrono::sys_days{cuda::std::chrono::Sunday[1]/cuda::std::chrono::January/y}
          : cuda::std::chrono::sys_days{cuda::std::chrono::Monday[1]/cuda::std::chrono::January/y};
        // clang-format on
        auto const days =  // compute days from year, weeks and weekday
          start + cuda::std::chrono::weeks(timeparts.week - 1) - cuda::std::chrono::weeks{1} +
          (cuda::std::chrono::weekday(timeparts.weekday) -
           cuda::std::chrono::weekday{1});  // cuda::std::chrono::Monday causes compile error here
        return days.time_since_epoch().count();
      }
      auto const ymd =  // chrono class handles the leap year calculations for us
        cuda::std::chrono::year_month_day(
          cuda::std::chrono::year{timeparts.year},
          cuda::std::chrono::month{static_cast<uint32_t>(timeparts.month)},
          cuda::std::chrono::day{static_cast<uint32_t>(timeparts.day)});
      return cuda::std::chrono::sys_days(ymd).time_since_epoch().count();
    }();

    if constexpr (std::is_same_v<T, cudf::timestamp_D>) { return days; }

    int64_t timestamp = (days * 24L * 3600L) + (timeparts.hour * 3600L) + (timeparts.minute * 60L) +
                        timeparts.second + (timeparts.tz_minutes * 60L);

    if constexpr (std::is_same_v<T, cudf::timestamp_s>) { return timestamp; }

    int64_t const subsecond =
      static_cast<int64_t>(
        timeparts.subsecond *
        cudf::detail::powers_of_ten[9 - subsecond_precision]) /  // normalize to nanoseconds
      (1000000000L / T::period::type::den);                      // and rescale to T

    timestamp *= T::period::type::den;
    timestamp += subsecond;

    return timestamp;
  }

  __device__ T operator()(size_type idx) const
  {
    T epoch_time{typename T::duration{0}};
    if (d_strings.is_null(idx)) return epoch_time;
    string_view d_str = d_strings.element<string_view>(idx);
    if (d_str.empty()) return epoch_time;

    auto const timeparts = parse_into_parts(d_str);

    return T{T::duration(timestamp_from_parts(timeparts))};
  }
};

/**
 * @brief Type-dispatch operator to convert timestamp strings to native fixed-width-type
 */
struct dispatch_to_timestamps_fn {
  template <typename T>
  void operator()(column_device_view const& d_strings,
                  std::string_view format,
                  mutable_column_view& results_view,
                  rmm::cuda_stream_view stream) const
    requires(cudf::is_timestamp<T>())
  {
    format_compiler compiler(format, stream);
    parse_datetime<T> pfn{d_strings, compiler.format_items(), compiler.subsecond_precision()};
    thrust::transform(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                      cuda::counting_iterator<size_type>{0},
                      cuda::counting_iterator<size_type>{results_view.size()},
                      results_view.data<T>(),
                      pfn);
  }
  template <typename T>
  void operator()(column_device_view const&,
                  std::string_view,
                  mutable_column_view&,
                  rmm::cuda_stream_view) const
    requires(not cudf::is_timestamp<T>())
  {
    CUDF_FAIL("Only timestamps type are expected", std::invalid_argument);
  }
};

}  // namespace

//
std::unique_ptr<cudf::column> to_timestamps(strings_column_view const& input,
                                            data_type timestamp_type,
                                            std::string_view format,
                                            rmm::cuda_stream_view stream,
                                            rmm::device_async_resource_ref mr)
{
  if (input.is_empty()) { return make_empty_column(timestamp_type); }

  CUDF_EXPECTS(!format.empty(), "Format parameter must not be empty.", std::invalid_argument);

  auto d_strings = column_device_view::create(input.parent(), stream);

  auto results = make_timestamp_column(timestamp_type,
                                       input.size(),
                                       cudf::detail::copy_bitmask(input.parent(), stream, mr),
                                       input.null_count(),
                                       stream,
                                       mr);

  auto results_view = results->mutable_view();
  cudf::type_dispatcher(
    timestamp_type, dispatch_to_timestamps_fn(), *d_strings, format, results_view, stream);

  results->set_null_count(input.null_count());
  return results;
}

/**
 * @brief Functor checks the strings against the given format items.
 *
 * This does no data conversion.
 */
struct check_datetime_format {
  column_device_view const d_strings;
  device_span<format_item const> const d_format_items;

  /**
   * @brief Check the specified characters are between ['0','9'].
   *
   * @param str Beginning of characters to check.
   * @param bytes Number of bytes to check.
   * @return true if all digits are 0-9
   */
  __device__ bool check_digits(char const* str, size_type bytes)
  {
    return thrust::all_of(thrust::seq, str, str + bytes, [] __device__(char chr) {
      return (chr >= '0' && chr <= '9');
    });
  }

  /**
   * @brief Check the specified characters are between ['0','9']
   * and the resulting integer is within [`min_value`, `max_value`].
   *
   * @param str Beginning of characters to check.
   * @param bytes Number of bytes to check.
   * @param min_value Inclusive minimum value
   * @param max_value Inclusive maximum value
   * @return If value is valid and number of bytes not successfully processed
   */
  __device__ cuda::std::pair<bool, size_type> check_value(char const* str,
                                                          size_type const bytes,
                                                          int const min_value,
                                                          int const max_value)
  {
    if (*str < '0' || *str > '9') { return cuda::std::make_pair(false, bytes); }
    int32_t value   = 0;
    size_type count = bytes;
    while (count-- > 0) {
      char chr = *str++;
      if (chr < '0' || chr > '9') break;
      value = (value * 10) + static_cast<int32_t>(chr - '0');
    }
    return (value >= min_value && value <= max_value) ? cuda::std::make_pair(true, count + 1)
                                                      : cuda::std::make_pair(false, bytes);
  }

  /**
   * @brief Check the string matches the format.
   *
   * Walk the `format_items` as we read the string characters
   * checking the characters are valid for each format specifier.
   * The checking here is a little more strict than the actual
   * parser used for conversion.
   */
  __device__ cuda::std::optional<timestamp_components> check_string(string_view const& d_string)
  {
    timestamp_components dateparts = {1970, 1, 1, 0};  // init to epoch time

    auto ptr    = d_string.data();
    auto length = d_string.size_bytes();
    for (auto item : d_format_items) {
      // eliminate static character values first
      if (item.item_type == format_char_type::literal) {
        // check static character matches
        if (*ptr != item.value) return cuda::std::nullopt;
        ptr += item.length;
        length -= item.length;
        continue;
      }
      // allow for specifiers to be truncated
      if (item.value != 'f')
        item.length =
          static_cast<int8_t>(cuda::std::min(static_cast<size_type>(item.length), length));

      // special logic for each specifier
      // reference: https://man7.org/linux/man-pages/man3/strptime.3.html
      bool result          = false;
      size_type bytes_read = item.length;
      switch (item.value) {
        case 'Y': {
          auto const [year, left] = parse_int(ptr, item.length);
          result                  = (left < item.length) && (year <= 9999);
          dateparts.year          = static_cast<int16_t>(year);
          bytes_read -= left;
          break;
        }
        case 'y': {
          auto const [year, left] = parse_int(ptr, item.length);
          result                  = (left < item.length);
          dateparts.year          = static_cast<int16_t>(year + (year < 69 ? 2000 : 1900));
          bytes_read -= left;
          break;
        }
        case 'm': {
          auto const [month, left] = parse_int(ptr, item.length);
          result                   = (left < item.length);
          dateparts.month          = static_cast<int8_t>(month);
          bytes_read -= left;
          break;
        }
        case 'd': {
          auto const [day, left] = parse_int(ptr, item.length);
          result                 = (left < item.length);
          dateparts.day          = static_cast<int8_t>(day);  // value.value()
          bytes_read -= left;
          break;
        }
        case 'j': {
          auto const cv = check_value(ptr, item.length, 1, 366);
          result        = cv.first;
          bytes_read -= cv.second;
          break;
        }
        case 'H': {
          auto const cv = check_value(ptr, item.length, 0, 23);
          result        = cv.first;
          bytes_read -= cv.second;
          break;
        }
        case 'I': {
          auto const cv = check_value(ptr, item.length, 1, 12);
          result        = cv.first;
          bytes_read -= cv.second;
          break;
        }
        case 'M': {
          auto const cv = check_value(ptr, item.length, 0, 59);
          result        = cv.first;
          bytes_read -= cv.second;
          break;
        }
        case 'S': {
          auto const cv = check_value(ptr, item.length, 0, 59);  // leap seconds not supported
          result        = cv.first;
          bytes_read -= cv.second;
          break;
        }
        case 'f': {
          int32_t const read_size =
            cuda::std::min(static_cast<int32_t>(item.length), static_cast<int32_t>(length));
          result     = check_digits(ptr, read_size);
          bytes_read = read_size;
          break;
        }
        case 'p': {
          if (item.length == 2) {
            string_view am_pm(ptr, 2);
            result = (am_pm.compare("AM", 2) == 0) || (am_pm.compare("am", 2) == 0) ||
                     (am_pm.compare("PM", 2) == 0) || (am_pm.compare("pm", 2) == 0);
          }
          break;
        }
        case 'U': [[fallthrough]];
        case 'W': {
          auto const cv = check_value(ptr, item.length, 0, 53);
          result        = cv.first;
          bytes_read -= cv.second;
          break;
        }
        case 'u': [[fallthrough]];  // valid values: 1-7
        case 'w': {                 // valid values: 0-6
          auto const first = item.value == 'w' ? 0 : 1;
          auto const cv    = check_value(ptr, item.length, first, first + 6);
          result           = cv.first;
          bytes_read -= cv.second;
          break;
        }
        case 'z': {  // timezone offset
          if (item.length == 5) {
            auto const cvh = check_value(ptr + 1, 2, 0, 23);
            auto const cvm = check_value(ptr + 3, 2, 0, 59);
            result         = (*ptr == '-' || *ptr == '+') && cvh.first && cvm.first;
            bytes_read -= cvh.second + cvm.second;
          } else if (item.length == 1) {
            result = *ptr == 'Z';
          }
          break;
        }
        case 'Z': result = true;  // skip
        default: break;
      }
      if (!result) return cuda::std::nullopt;
      ptr += bytes_read;
      length -= bytes_read;
    }
    return dateparts;
  }

  __device__ bool operator()(size_type idx)
  {
    if (d_strings.is_null(idx)) return false;

    string_view d_str = d_strings.element<string_view>(idx);
    if (d_str.empty()) return false;

    auto const dateparts = check_string(d_str);
    if (!dateparts.has_value()) return false;

    auto const year  = dateparts.value().year;
    auto const month = static_cast<uint32_t>(dateparts.value().month);
    auto const day   = static_cast<uint32_t>(dateparts.value().day);
    return cuda::std::chrono::year_month_day(cuda::std::chrono::year{year},
                                             cuda::std::chrono::month{month},
                                             cuda::std::chrono::day{day})
      .ok();
  }
};

std::unique_ptr<cudf::column> is_timestamp(strings_column_view const& input,
                                           std::string_view const& format,
                                           rmm::cuda_stream_view stream,
                                           rmm::device_async_resource_ref mr)
{
  size_type strings_count = input.size();
  if (strings_count == 0) return make_empty_column(type_id::BOOL8);

  CUDF_EXPECTS(!format.empty(), "Format parameter must not be empty.", std::invalid_argument);

  auto d_strings = column_device_view::create(input.parent(), stream);

  auto results   = make_numeric_column(data_type{type_id::BOOL8},
                                     strings_count,
                                     cudf::detail::copy_bitmask(input.parent(), stream, mr),
                                     input.null_count(),
                                     stream,
                                     mr);
  auto d_results = results->mutable_view().data<bool>();

  format_compiler compiler(format, stream);
  thrust::transform(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                    cuda::counting_iterator<size_type>{0},
                    cuda::counting_iterator<size_type>{strings_count},
                    d_results,
                    check_datetime_format{*d_strings, compiler.format_items()});

  results->set_null_count(input.null_count());
  return results;
}

}  // namespace detail

// external APIs

std::unique_ptr<cudf::column> to_timestamps(strings_column_view const& input,
                                            data_type timestamp_type,
                                            std::string_view format,
                                            rmm::cuda_stream_view stream,
                                            rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::to_timestamps(input, timestamp_type, format, stream, mr);
}

std::unique_ptr<cudf::column> is_timestamp(strings_column_view const& input,
                                           std::string_view format,
                                           rmm::cuda_stream_view stream,
                                           rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::is_timestamp(input, format, stream, mr);
}

namespace detail {
namespace {

constexpr size_type format_names_size = 40;  // 2(am/pm) + 2x7(weekdays) + 2x12(months)
constexpr size_type offset_weekdays   = 2;
constexpr size_type offset_months     = 16;
constexpr size_type days_in_week      = 7;
constexpr size_type months_in_year    = 12;

/**
 * @brief Time components used by the date_time_formatter
 */
struct time_components {
  int8_t hour;
  int8_t minute;
  int8_t second;
  int32_t subsecond;
};

/**
 * @brief Functor for from_timestamps to convert a timestamp to a string
 *
 * This is designed to be used with make_strings_children
 */
template <typename T>
struct datetime_formatter_fn {
  column_device_view const d_timestamps;
  column_device_view const d_format_names;
  device_span<format_item const> const d_format_items;
  size_type* d_sizes{};
  char* d_chars{};
  cudf::detail::input_offsetalator d_offsets;

  /**
   * @brief Specialized modulo expression that handles negative values.
   *
   * @code{.pseudo}
   * Examples:
   *     modulo(1,60)  ->  1
   *     modulo(-1,60) -> 59
   * @endcode
   */
  __device__ int32_t modulo_time(int64_t time, int64_t base) const
  {
    return static_cast<int32_t>(((time % base) + base) % base);
  };

  /**
   * @brief This function handles converting units by dividing and adjusting for negative values.
   *
   * @code{.pseudo}
   * Examples:
   *     scale(-61,60) -> -2
   *     scale(-60,60) -> -1
   *     scale(-59,60) -> -1
   *     scale( 59,60) ->  0
   *     scale( 60,60) ->  1
   *     scale( 61,60) ->  1
   * @endcode
   */
  __device__ int64_t scale_time(int64_t time, int64_t base) const
  {
    return (time - ((time < 0) * (base - 1L))) / base;
  };

  __device__ time_components get_time_components(int64_t tstamp) const
  {
    time_components result = {0};
    if constexpr (std::is_same_v<T, cudf::timestamp_D>) { return result; }

    // Note: Tried using: cuda::std::chrono::hh_mm_ss(T::duration(timestamp));
    // and retrieving the hour, minute, second, and subsecond values from it
    // but it did not scale/modulo the components for negative timestamps
    // correctly -- it simply did an abs(timestamp) as documented here:
    // https://en.cppreference.com/cpp/chrono/hh_mm_ss/hh_mm_ss

    if constexpr (not std::is_same_v<T, cudf::timestamp_s>) {
      int64_t constexpr base = T::period::type::den;  // 1000=ms, 1000000=us, etc
      auto const subsecond   = modulo_time(tstamp, base);
      tstamp                 = tstamp / base - ((tstamp < 0) and (subsecond != 0));
      result.subsecond       = subsecond;
    }

    result.hour   = modulo_time(scale_time(tstamp, 3600), 24);
    result.minute = modulo_time(scale_time(tstamp, 60), 60);
    result.second = modulo_time(tstamp, 60);

    return result;
  }

  __device__ size_type compute_output_size(T const tstamp) const
  {
    // We only dissect the timestamp into components if needed
    // by a specifier. And then we only do it once and reuse it.
    // This can improve performance when not using uncommon specifiers.
    cuda::std::optional<cuda::std::chrono::sys_days> days;

    auto days_from_timestamp = [tstamp]() {
      auto const count = tstamp.time_since_epoch().count();
      return cuda::std::chrono::sys_days(static_cast<cudf::timestamp_D::duration>(
        floor<cuda::std::chrono::days>(T::duration(count))));
    };

    size_type bytes = 0;  // output size
    for (auto item : d_format_items) {
      if (item.item_type == format_char_type::literal) {
        bytes += item.length;
        continue;
      }

      // only specifiers resulting in strings require special logic
      switch (item.value) {
        case 'a':    // weekday abbreviated
        case 'A': {  // weekday full name
          if (!days.has_value()) { days = days_from_timestamp(); }
          auto const day_of_week =
            cuda::std::chrono::year_month_weekday(days.value()).weekday().c_encoding();
          auto const day_idx =
            day_of_week + offset_weekdays + (item.value == 'a' ? days_in_week : 0);
          if (day_idx < d_format_names.size())
            bytes += d_format_names.element<cudf::string_view>(day_idx).size_bytes();
          break;
        }
        case 'b':    // month abbreviated
        case 'B': {  // month full name
          if (!days.has_value()) { days = days_from_timestamp(); }
          auto const month =
            static_cast<uint32_t>(cuda::std::chrono::year_month_day(days.value()).month());
          auto const month_idx =
            month - 1 + offset_months + (item.value == 'b' ? months_in_year : 0);
          if (month_idx < d_format_names.size())
            bytes += d_format_names.element<cudf::string_view>(month_idx).size_bytes();
          break;
        }
        case 'p':  // AM/PM
        {
          auto const times = get_time_components(tstamp.time_since_epoch().count());
          bytes += d_format_names.size() > 1
                     ? d_format_names.element<cudf::string_view>(static_cast<int>(times.hour >= 12))
                         .size_bytes()
                     : 2;
          break;
        }
        default: {
          bytes += item.length;
          break;
        }
      }
    }
    return bytes;
  }

  // utility to create 0-padded integers (up to 9 chars)
  __device__ char* int2str(char* str, int bytes, int val) const
  {
    char tmpl[9] = {'0', '0', '0', '0', '0', '0', '0', '0', '0'};
    char* ptr    = tmpl;
    while (val > 0) {
      int digit = val % 10;
      *ptr++    = '0' + digit;
      val       = val / 10;
    }
    ptr = tmpl + bytes - 1;
    while (bytes-- > 0)
      *str++ = *ptr--;
    return str;
  }

  // from https://howardhinnant.github.io/date/date.html
  __device__ cuda::std::pair<int32_t, int32_t> get_iso_week_year(
    cuda::std::chrono::year_month_day const& ymd) const
  {
    auto const days = cuda::std::chrono::sys_days(ymd);
    auto year       = ymd.year();

    auto iso_week_start = [](cuda::std::chrono::year const y) {
      // clang-format off
      return cuda::std::chrono::sys_days{cuda::std::chrono::Thursday[1]/cuda::std::chrono::January/y} -
             (cuda::std::chrono::Thursday - cuda::std::chrono::Monday);
      // clang-format on
    };

    auto start = iso_week_start(year);
    if (days < start)
      start = iso_week_start(--year);
    else {
      auto const next_start = iso_week_start(year + cuda::std::chrono::years{1});
      if (days >= next_start) {
        ++year;
        start = next_start;
      }
    }
    return cuda::std::make_pair(
      (cuda::std::chrono::duration_cast<cuda::std::chrono::weeks>(days - start) +
       cuda::std::chrono::weeks{1})  // always [1-53]
        .count(),
      static_cast<int32_t>(year));
  }

  __device__ int8_t get_week_of_year(cuda::std::chrono::sys_days const days,
                                     cuda::std::chrono::sys_days const start) const
  {
    return days < start
             ? 0
             : (cuda::std::chrono::duration_cast<cuda::std::chrono::weeks>(days - start) +
                cuda::std::chrono::weeks{1})
                 .count();
  }

  __device__ int32_t get_day_of_year(cuda::std::chrono::year_month_day const& ymd) const
  {
    auto const month               = static_cast<uint32_t>(ymd.month());
    auto const day                 = static_cast<uint32_t>(ymd.day());
    int32_t const monthDayOffset[] = {0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334};
    return static_cast<int32_t>(day + monthDayOffset[month - 1] +
                                (month > 2 and ymd.year().is_leap()));
  }

  __device__ void timestamp_to_string(T const tstamp, char* ptr) const
  {
    auto const days = cuda::std::chrono::sys_days(
      static_cast<cudf::timestamp_D::duration>(cuda::std::chrono::floor<cuda::std::chrono::days>(
        T::duration(tstamp.time_since_epoch().count()))));
    auto const ymd = cuda::std::chrono::year_month_day(days);

    auto timeparts = get_time_components(tstamp.time_since_epoch().count());

    // convert to characters using the format items
    for (auto item : d_format_items) {
      if (item.item_type == format_char_type::literal) {
        *ptr++ = item.value;
        continue;
      }

      // Value to use for int2str call at the end of the switch-statement.
      // This simplifies the case statements and prevents a lot of extra inlining.
      int32_t copy_value = -1;  // default set for non-int2str usage cases

      // special logic for each specifier
      switch (item.value) {
        case 'Y':  // 4-digit year
          copy_value = static_cast<int32_t>(ymd.year());
          break;
        case 'y':  // 2-digit year
        {
          auto year = static_cast<int32_t>(ymd.year());
          // remove hundredths digits and above
          copy_value = year - ((year / 100) * 100);
          break;
        }
        case 'm':  // month
          copy_value = static_cast<int32_t>(static_cast<uint32_t>(ymd.month()));
          break;
        case 'd':  // day of month
          copy_value = static_cast<int32_t>(static_cast<uint32_t>(ymd.day()));
          break;
        case 'j':  // day of year
          copy_value = get_day_of_year(ymd);
          break;
        case 'H':  // 24-hour
          copy_value = timeparts.hour;
          break;
        case 'I':  // 12-hour
        {
          // 0 = 12am; 12 = 12pm; 6 = 06am; 18 = 06pm
          copy_value = [h = timeparts.hour] {
            if (h == 0) return 12;
            return h > 12 ? h - 12 : h;
          }();
          break;
        }
        case 'M':  // minute
          copy_value = timeparts.minute;
          break;
        case 'S':  // second
          copy_value = timeparts.second;
          break;
        case 'f':  // sub-second
        {
          char subsecond_digits[] = "000000000";  // 9 max digits
          int const digits        = [] {
            if constexpr (std::is_same_v<T, cudf::timestamp_ms>) return 3;
            if constexpr (std::is_same_v<T, cudf::timestamp_us>) return 6;
            if constexpr (std::is_same_v<T, cudf::timestamp_ns>) return 9;
            return 0;
          }();
          int2str(subsecond_digits, digits, timeparts.subsecond);
          ptr = copy_and_increment(ptr, subsecond_digits, item.length);
          break;
        }
        case 'p':  // am or pm
        {
          // 0 = 12am, 12 = 12pm
          auto const am_pm = [&] {
            if (d_format_names.size() > 1)
              return d_format_names.element<cudf::string_view>(
                static_cast<int>(timeparts.hour >= 12));
            return string_view(timeparts.hour >= 12 ? "PM" : "AM", 2);
          }();
          ptr = copy_string(ptr, am_pm);
          break;
        }
        case 'z':  // timezone -- always UTC
          ptr = copy_and_increment(ptr, "+0000", 5);
          break;
        case 'Z':  // timezone string -- always UTC
          ptr = copy_and_increment(ptr, "UTC", 3);
          break;
        case 'u':    // day of week ISO
        case 'w': {  // day of week non-ISO
          auto const day_of_week = static_cast<int32_t>(
            cuda::std::chrono::year_month_weekday(days).weekday().c_encoding());
          copy_value = day_of_week == 0 && item.value == 'u' ? 7 : day_of_week;
          break;
        }
          // clang-format off
        case 'U': {  // week of year: first week includes the first Sunday of the year
          copy_value = get_week_of_year(days, cuda::std::chrono::sys_days{
            cuda::std::chrono::Sunday[1]/cuda::std::chrono::January/ymd.year()});
          break;
        }
        case 'W': {  // week of year: first week includes the first Monday of the year
          copy_value = get_week_of_year(days, cuda::std::chrono::sys_days{
            cuda::std::chrono::Monday[1]/cuda::std::chrono::January/ymd.year()});
          break;
        }
        // clang-format on
        case 'V':    // ISO week number
        case 'G': {  // ISO year number
          auto const [week, year] = get_iso_week_year(ymd);
          copy_value              = item.value == 'G' ? year : week;
          break;
        }
        case 'a':    // abbreviated day of the week
        case 'A': {  // day of the week
          auto const day_of_week =
            cuda::std::chrono::year_month_weekday(days).weekday().c_encoding();
          auto const day_idx =
            day_of_week + offset_weekdays + (item.value == 'a' ? days_in_week : 0);
          if (d_format_names.size())
            ptr = copy_string(ptr, d_format_names.element<cudf::string_view>(day_idx));
          break;
        }
        case 'b':    // abbreviated month of the year
        case 'B': {  // month of the year
          auto const month = static_cast<uint32_t>(ymd.month());
          auto const month_idx =
            month - 1 + offset_months + (item.value == 'b' ? months_in_year : 0);
          if (d_format_names.size())
            ptr = copy_string(ptr, d_format_names.element<cudf::string_view>(month_idx));
          break;
        }
        default: break;
      }
      if (copy_value >= 0) ptr = int2str(ptr, item.length, copy_value);
    }
  }

  __device__ void operator()(size_type idx) const
  {
    if (d_timestamps.is_null(idx)) {
      if (!d_chars) { d_sizes[idx] = 0; }
      return;
    }
    auto const tstamp = d_timestamps.element<T>(idx);
    if (d_chars) {
      timestamp_to_string(tstamp, d_chars + d_offsets[idx]);
    } else {
      d_sizes[idx] = compute_output_size(tstamp);
    }
  }
};

//
using strings_children = std::pair<std::unique_ptr<cudf::column>, rmm::device_uvector<char>>;
struct dispatch_from_timestamps_fn {
  template <typename T>
  strings_children operator()(column_device_view const& d_timestamps,
                              column_device_view const& d_format_names,
                              device_span<format_item const> d_format_items,
                              rmm::cuda_stream_view stream,
                              rmm::device_async_resource_ref mr) const
    requires(cudf::is_timestamp<T>())
  {
    return make_strings_children(
      datetime_formatter_fn<T>{d_timestamps, d_format_names, d_format_items},
      d_timestamps.size(),
      stream,
      mr);
  }

  template <typename T, typename... Args>
  strings_children operator()(Args&&...) const
    requires(not cudf::is_timestamp<T>())
  {
    CUDF_FAIL("Only timestamps type are expected", std::invalid_argument);
  }
};

}  // namespace

//
std::unique_ptr<column> from_timestamps(column_view const& timestamps,
                                        std::string_view format,
                                        strings_column_view const& names,
                                        rmm::cuda_stream_view stream,
                                        rmm::device_async_resource_ref mr)
{
  if (timestamps.is_empty()) return make_empty_column(type_id::STRING);

  CUDF_EXPECTS(!format.empty(), "Format parameter must not be empty.", std::invalid_argument);
  CUDF_EXPECTS(names.is_empty() || names.size() == format_names_size,
               "Invalid size for format names.",
               std::invalid_argument);

  auto const d_names = column_device_view::create(names.parent(), stream);

  // This API supports a few more specifiers than to_timestamps.
  // clang-format off
  format_compiler compiler(format, stream,
    specifier_map{{'V', 2}, {'G', 4}, {'a', 3}, {'A', 3}, {'b', 3}, {'B', 3}});
  // clang-format on
  auto const d_format_items = compiler.format_items();
  auto const d_timestamps   = column_device_view::create(timestamps, stream);

  // dispatcher is called to handle the different timestamp types
  auto [offsets_column, chars] = cudf::type_dispatcher(timestamps.type(),
                                                       dispatch_from_timestamps_fn(),
                                                       *d_timestamps,
                                                       *d_names,
                                                       d_format_items,
                                                       stream,
                                                       mr);

  return make_strings_column(timestamps.size(),
                             std::move(offsets_column),
                             chars.release(),
                             timestamps.null_count(),
                             cudf::detail::copy_bitmask(timestamps, stream, mr));
}

}  // namespace detail

// external API

std::unique_ptr<column> from_timestamps(column_view const& timestamps,
                                        std::string_view format,
                                        strings_column_view const& names,
                                        rmm::cuda_stream_view stream,
                                        rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::from_timestamps(timestamps, format, names, stream, mr);
}

}  // namespace strings
}  // namespace cudf
