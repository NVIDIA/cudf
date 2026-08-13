/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/column/column.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <string>
#include <vector>

/**
 * @file
 * @brief APIs for converting strings columns to and from timestamp values
 */

namespace CUDF_EXPORT cudf {
namespace strings {
/**
 * @addtogroup strings_convert
 * @{
 */

/**
 * @brief Returns a new timestamp column converting a strings column into
 * timestamps using the provided format pattern.
 *
 * The format pattern can include the following specifiers:
 *
 * | Specifier | Description |
 * | :-------: | ----------- |
 * | \%d | Day of the month: 01-31 |
 * | \%m | Month of the year: 01-12 |
 * | \%y | Year without century: 00-99. [0,68] maps to [2000,2068] and [69,99] maps to [1969,1999] |
 * | \%Y | Year with century: 0001-9999 |
 * | \%H | 24-hour of the day: 00-23 |
 * | \%I | 12-hour of the day: 01-12 |
 * | \%M | Minute of the hour: 00-59 |
 * | \%S | Second of the minute: 00-59. Leap second is not supported. |
 * | \%f | 6-digit microsecond: 000000-999999 |
 * | \%z | UTC offset with format ±HHMM Example +0500 |
 * | \%j | Day of the year: 001-366 |
 * | \%p | Only 'AM', 'PM' or 'am', 'pm' are recognized |
 * | \%W | Week of the year with Monday as the first day of the week: 00-53 |
 * | \%w | Day of week: 0-6 = Sunday-Saturday |
 * | \%U | Week of the year with Sunday as the first day of the week: 00-53 |
 * | \%u | Day of week: 1-7 = Monday-Sunday |
 *
 * Other specifiers are not currently supported.
 *
 * Invalid formats are not checked. If the string contains unexpected
 * or insufficient characters, that output row entry's timestamp value is undefined.
 *
 * Any null string entry will result in a corresponding null row in the output column.
 *
 * The resulting time units are specified by the `timestamp_type` parameter.
 * The time units are independent of the number of digits parsed by the "%f" specifier.
 * The "%f" supports a precision value to read the numeric digits. Specify the
 * precision with a single integer value (1-9) as follows:
 * use "%3f" for milliseconds, "%6f" for microseconds and "%9f" for nanoseconds.
 *
 * Although leap second is not supported for "%S", no checking is performed on the value.
 * The cudf::strings::is_timestamp can be used to verify the valid range of values.
 *
 * If "%W"/"%w" (or "%U/%u") and "%m"/"%d" are both specified, the "%W"/%U and "%w"/%u values
 * take precedent when computing the date part of the timestamp result.
 *
 * @throw cudf::logic_error if timestamp_type is not a timestamp type.
 *
 * @param input Strings instance for this operation
 * @param timestamp_type The timestamp type used for creating the output column
 * @param format String specifying the timestamp format in strings
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @return New datetime column
 */
std::unique_ptr<column> to_timestamps(
  strings_column_view const& input,
  data_type timestamp_type,
  std::string_view format,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Verifies the given strings column can be parsed to timestamps using the provided format
 * pattern.
 *
 * The format pattern can include the following specifiers:
 *
 * | Specifier | Description |
 * | :-------: | ----------- |
 * | \%d | Day of the month: 01-31 |
 * | \%m | Month of the year: 01-12 |
 * | \%y | Year without century: 00-99. [0,68] maps to [2000,2068] and [69,99] maps to [1969,1999] |
 * | \%Y | Year with century: 0001-9999 |
 * | \%H | 24-hour of the day: 00-23 |
 * | \%I | 12-hour of the day: 01-12 |
 * | \%M | Minute of the hour: 00-59|
 * | \%S | Second of the minute: 00-59. Leap second is not supported. |
 * | \%f | 6-digit microsecond: 000000-999999 |
 * | \%z | UTC offset with format ±HHMM Example +0500 |
 * | \%j | Day of the year: 001-366 |
 * | \%p | Only 'AM', 'PM' or 'am', 'pm' are recognized |
 * | \%W | Week of the year with Monday as the first day of the week: 00-53 |
 * | \%w | Day of week: 0-6 = Sunday-Saturday |
 * | \%U | Week of the year with Sunday as the first day of the week: 00-53 |
 * | \%u | Day of week: 1-7 = Monday-Sunday |
 *
 * Other specifiers are not currently supported.
 * The "%f" supports a precision value to read the numeric digits. Specify the
 * precision with a single integer value (1-9) as follows:
 * use "%3f" for milliseconds, "%6f" for microseconds and "%9f" for nanoseconds.
 *
 * Any null string entry will result in a corresponding null row in the output column.
 *
 * This will return a column of type BOOL8 where a `true` row indicates the corresponding
 * input string can be parsed correctly with the given format.
 *
 * @throw std::invalid_argument if the `format` string is empty
 * @throw std::invalid_argument if a specifier is not supported
 *
 * @param input Strings instance for this operation
 * @param format String specifying the timestamp format in strings
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @return New BOOL8 column
 */
std::unique_ptr<column> is_timestamp(
  strings_column_view const& input,
  std::string_view format,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Returns a new strings column converting a timestamp column into
 * strings using the provided format pattern.
 *
 * The format pattern can include the following specifiers:
 *
 * | Specifier | Description |
 * | :-------: | ----------- |
 * | \%d | Day of the month: 01-31 |
 * | \%m | Month of the year: 01-12 |
 * | \%y | Year without century: 00-99 |
 * | \%Y | Year with century: 0001-9999 |
 * | \%H | 24-hour of the day: 00-23 |
 * | \%I | 12-hour of the day: 01-12 |
 * | \%M | Minute of the hour: 00-59|
 * | \%S | Second of the minute: 00-59 |
 * | \%f | 6-digit microsecond: 000000-999999 |
 * | \%z | Always outputs "+0000" |
 * | \%Z | Always outputs "UTC" |
 * | \%j | Day of the year: 001-366 |
 * | \%u | ISO weekday where Monday is 1 and Sunday is 7 |
 * | \%w | Weekday where Sunday is 0 and Saturday is 6 |
 * | \%U | Week of the year with Sunday as the first day: 00-53 |
 * | \%W | Week of the year with Monday as the first day: 00-53 |
 * | \%V | Week of the year per ISO-8601 format: 01-53 |
 * | \%G | Year based on the ISO-8601 weeks: 0000-9999 |
 * | \%p | AM/PM from `timestamp_names::am_str/pm_str` |
 * | \%a | Weekday abbreviation from the `names` parameter |
 * | \%A | Weekday from the `names` parameter |
 * | \%b | Month name abbreviation from the `names` parameter |
 * | \%B | Month name from the `names` parameter |
 *
 * Additional descriptions can be found here:
 * https://en.cppreference.com/cpp/chrono/system_clock/formatter
 *
 * No checking is done for invalid formats or invalid timestamp values.
 * All timestamps values are formatted to UTC.
 *
 * Any null input entry will result in a corresponding null entry in the output column.
 *
 * The time units of the input column do not influence the number of digits written by
 * the "%f" specifier. The "%f" supports a precision value to write out numeric digits
 * for the subsecond value. Specify the precision with a single integer value (1-9)
 * between the "%" and the "f" as follows: use "%3f" for milliseconds, use "%6f" for
 * microseconds and use "%9f" for nanoseconds. If the precision is higher than the
 * units, then zeroes are padded to the right of the subsecond value. If the precision
 * is lower than the units, the subsecond value may be truncated.
 *
 * If the "%a", "%A", "%b", "%B" specifiers are included in the format, the caller
 * should provide the format names in the `names` strings column using the following
 * as a guide:
 *
 * @code{.pseudo}
 * ["AM", "PM",                             // specify the AM/PM strings
 *  "Sunday", "Monday", ..., "Saturday",    // Weekday full names
 *  "Sun", "Mon", ..., "Sat",               // Weekday abbreviated names
 *  "January", "February", ..., "December", // Month full names
 *  "Jan", "Feb", ..., "Dec"]               // Month abbreviated names
 * @endcode
 *
 * The result is undefined if the format names are not provided for these specifiers.
 *
 * These format names can be retrieved for specific locales using the `nl_langinfo`
 * functions from C++ `clocale` (std) library or the Python `locale` library.
 *
 * The following code is an example of retrieving these strings from the locale
 * using c++ std functions:
 *
 * @code{.cpp}
 * #include <clocale>
 * #include <langinfo.h>
 *
 * // note: install language pack on Ubuntu using 'apt-get install language-pack-de'
 * {
 *   // set to a German language locale for date settings
 *   std::setlocale(LC_TIME, "de_DE.UTF-8");
 *
 *   std::vector<std::string> names({nl_langinfo(AM_STR), nl_langinfo(PM_STR),
 *     nl_langinfo(DAY_1), nl_langinfo(DAY_2), nl_langinfo(DAY_3), nl_langinfo(DAY_4),
 *      nl_langinfo(DAY_5), nl_langinfo(DAY_6), nl_langinfo(DAY_7),
 *     nl_langinfo(ABDAY_1), nl_langinfo(ABDAY_2), nl_langinfo(ABDAY_3), nl_langinfo(ABDAY_4),
 *      nl_langinfo(ABDAY_5), nl_langinfo(ABDAY_6), nl_langinfo(ABDAY_7),
 *     nl_langinfo(MON_1), nl_langinfo(MON_2), nl_langinfo(MON_3), nl_langinfo(MON_4),
 *      nl_langinfo(MON_5), nl_langinfo(MON_6), nl_langinfo(MON_7), nl_langinfo(MON_8),
 *      nl_langinfo(MON_9), nl_langinfo(MON_10), nl_langinfo(MON_11), nl_langinfo(MON_12),
 *     nl_langinfo(ABMON_1), nl_langinfo(ABMON_2), nl_langinfo(ABMON_3), nl_langinfo(ABMON_4),
 *      nl_langinfo(ABMON_5), nl_langinfo(ABMON_6), nl_langinfo(ABMON_7), nl_langinfo(ABMON_8),
 *      nl_langinfo(ABMON_9), nl_langinfo(ABMON_10), nl_langinfo(ABMON_11), nl_langinfo(ABMON_12)});
 *
 *   std::setlocale(LC_TIME,""); // reset to default locale
 * }
 * @endcode
 *
 * @throw std::invalid_argument if `timestamps` column parameter is not a timestamp type.
 * @throw std::invalid_argument if the `format` string is empty
 * @throw std::invalid_argument if `names.size()` is an invalid size. Must be 0 or 40 strings.
 * @throw std::invalid_argument if a specifier is not supported
 *
 * @param timestamps Timestamp values to convert
 * @param format The string specifying output format.
 *        Default format is "%Y-%m-%dT%H:%M:%SZ".
 * @param names The string names to use for weekdays ("%a", "%A") and months ("%b", "%B")
 *        Default is an empty `strings_column_view`.
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @return New strings column with formatted timestamps
 */
std::unique_ptr<column> from_timestamps(
  column_view const& timestamps,
  std::string_view format           = "%Y-%m-%dT%H:%M:%SZ",
  strings_column_view const& names  = strings_column_view(column_view{
    data_type{type_id::STRING}, 0, nullptr, nullptr, 0}),
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/** @} */  // end of doxygen group
}  // namespace strings
}  // namespace CUDF_EXPORT cudf
