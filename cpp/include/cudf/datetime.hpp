/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/export.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <memory>

/**
 * @file datetime.hpp
 * @brief DateTime column APIs.
 */

namespace CUDF_EXPORT cudf {
namespace datetime {
/**
 * @addtogroup datetime_extract
 * @{
 */

/**
 * @brief Types of datetime components that may be extracted.
 */
enum class datetime_component : uint8_t {
  YEAR,
  MONTH,
  DAY,
  WEEKDAY,
  HOUR,
  MINUTE,
  SECOND,
  MILLISECOND,
  MICROSECOND,
  NANOSECOND
};

/**
 * @brief Extracts the specified datetime component from any datetime type and
 * returns an int16_t cudf::column.
 *
 * @param column cudf::column_view of the input datetime values
 * @param component The datetime component to extract
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate device memory of the returned column
 *
 * @returns cudf::column of the extracted int16_t datetime component
 * @throw cudf::logic_error if input column datatype is not TIMESTAMP
 */
std::unique_ptr<cudf::column> extract_datetime_component(
  cudf::column_view const& column,
  datetime_component component,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/** @} */  // end of group
/**
 * @addtogroup datetime_compute
 * @{
 */

/**
 * @brief  Computes the last day of the month in datetime type and returns a TIMESTAMP_DAYS
 * cudf::column.
 *
 * @param column cudf::column_view of the input datetime values
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate device memory of the returned column
 *
 * @returns cudf::column containing last day of the month as TIMESTAMP_DAYS
 * @throw cudf::logic_error if input column datatype is not TIMESTAMP
 */
std::unique_ptr<cudf::column> last_day_of_month(
  cudf::column_view const& column,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief  Computes the day number since the start of the year from the datetime and
 * returns an int16_t cudf::column. The value is between [1, {365-366}]
 *
 * @param column cudf::column_view of the input datetime values
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate device memory of the returned column
 *
 * @returns cudf::column of datatype INT16 containing the day number since the start of the year
 * @throw cudf::logic_error if input column datatype is not a TIMESTAMP
 */
std::unique_ptr<cudf::column> day_of_year(
  cudf::column_view const& column,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief  Adds or subtracts a number of months from the datetime type and returns a
 * timestamp column that is of the same type as the input `timestamps` column.
 *
 * For a given row, if the `timestamps` or the `months` column value is null,
 * the output for that row is null.
 * This method preserves the input time and the day where applicable. The date is rounded
 * down to the last day of the month for that year, if the new day is invalid for that month.
 *
 * @code{.pseudo}
 * Example:
 * timestamps = [5/31/20 08:00:00, 5/31/20 00:00:00, 5/31/20 13:00:00, 5/31/20 23:00:00,
 *               6/30/20 00:00:01, 6/30/20 14:12:13]
 * months     = [1               , -1              , -3              , -15             ,
 *               -1              , 1]
 * r = add_calendrical_months(timestamp_column, months_column)
 * r is [6/30/20 08:00:00, 4/30/20 00:00:00, 2/29/20 13:00:00, 2/28/19 23:00:00,
 *       5/30/20 00:00:01, 7/30/20 14:12:13]
 * @endcode

 * @throw cudf::logic_error if `timestamps` datatype is not a TIMESTAMP or if `months` datatype
 * is not INT16 or INT32.
 * @throw cudf::logic_error if `timestamps` column size is not equal to `months` column size.
 *
 * @param timestamps cudf::column_view of timestamp type
 * @param months cudf::column_view of integer type containing the number of months to add
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate device memory of the returned column
 *
 * @returns cudf::column of timestamp type containing the computed timestamps
 */
std::unique_ptr<cudf::column> add_calendrical_months(
  cudf::column_view const& timestamps,
  cudf::column_view const& months,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief  Adds or subtracts a number of months from the datetime type and returns a
 * timestamp column that is of the same type as the input `timestamps` column.
 *
 * For a given row, if the `timestamps` value is null, the output for that row is null.
 * A null months scalar would result in an all null column.
 * This method preserves the input time and the day where applicable. The date is rounded
 * down to the last day of the month for that year, if the new day is invalid for that month.
 *
 * @code{.pseudo}
 * Example:
 * timestamps = [5/31/20 08:00:00, 6/30/20 00:00:00, 7/31/20 13:00:00]
 * months     = -3
 * output is [2/29/20 08:00:00, 3/30/20 00:00:00, 4/30/20 13:00:00]
 *
 * timestamps = [4/28/20 04:00:00, 5/30/20 01:00:00, 6/30/20 21:00:00]
 * months     = 1
 * output is [5/28/20 04:00:00, 6/30/20 01:00:00, 7/30/20 21:00:00]
 * @endcode
 *
 * @throw cudf::logic_error if `timestamps` datatype is not a TIMESTAMP or if `months` datatype
 * is not INT16 or INT32.
 * @throw cudf::logic_error if `timestamps` column size is not equal to `months` column size.
 *
 * @param timestamps cudf::column_view of timestamp type
 * @param months cudf::scalar of integer type containing the number of months to add
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate device memory of the returned column
 *
 * @return cudf::column of timestamp type containing the computed timestamps
 */
std::unique_ptr<cudf::column> add_calendrical_months(
  cudf::column_view const& timestamps,
  cudf::scalar const& months,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief  Check if the year of the given date is a leap year
 *
 * `output[i] == true` if year of `column[i]` is a leap year, otherwise `output[i] == false`.
 *
 * Any null input rows return corresponding null entries in the output columns.
 *
 * @param column cudf::column_view of the input datetime values
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate device memory of the returned column
 *
 * @returns cudf::column of datatype BOOL8 truth value of the corresponding date
 * @throw cudf::logic_error if input column datatype is not a TIMESTAMP
 */
std::unique_ptr<cudf::column> is_leap_year(
  cudf::column_view const& column,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Extract the number of days in the month
 *
 * `output[i]` contains the number of days in the month of date `column[i]`.
 *
 * Any null input rows return corresponding null entries in the output columns.
 *
 * @throw cudf::logic_error if input column datatype is not a TIMESTAMP
 *
 * @param column cudf::column_view of the input datetime values
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate device memory of the returned column
 * @return cudf::column of datatype INT16 of days in month of the corresponding date
 */
std::unique_ptr<cudf::column> days_in_month(
  cudf::column_view const& column,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief  Returns the quarter of the date
 *
 * `output[i]` will be a value from `{1, 2, 3, 4}` corresponding to the quarter of month given by
 * `column[i]`.
 *
 * Any null input rows return corresponding null entries in the output columns.
 *
 * @throw cudf::logic_error if input column datatype is not a TIMESTAMP
 *
 * @param column The input column containing datetime values
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate device memory of the returned column
 * @return A column of INT16 type indicating which quarter the date is in
 */
std::unique_ptr<cudf::column> extract_quarter(
  cudf::column_view const& column,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Fixed frequencies supported by datetime rounding functions ceil, floor, round.
 *
 */
enum class rounding_frequency : int32_t {
  DAY,
  HOUR,
  MINUTE,
  SECOND,
  MILLISECOND,
  MICROSECOND,
  NANOSECOND
};

/**
 * @brief Round datetimes up to the nearest multiple of the given frequency.
 *
 * @param column cudf::column_view of the input datetime values
 * @param freq rounding_frequency indicating the frequency to round up to
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate device memory of the returned column
 *
 * @throw cudf::logic_error if input column datatype is not TIMESTAMP.
 * @return cudf::column of the same datetime resolution as the input column
 */
std::unique_ptr<cudf::column> ceil_datetimes(
  cudf::column_view const& column,
  rounding_frequency freq,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Round datetimes down to the nearest multiple of the given frequency.
 *
 * @param column cudf::column_view of the input datetime values
 * @param freq rounding_frequency indicating the frequency to round down to
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate device memory of the returned column
 *
 * @throw cudf::logic_error if input column datatype is not TIMESTAMP.
 * @return cudf::column of the same datetime resolution as the input column
 */
std::unique_ptr<cudf::column> floor_datetimes(
  cudf::column_view const& column,
  rounding_frequency freq,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Round datetimes to the nearest multiple of the given frequency.
 *
 * @param column cudf::column_view of the input datetime values
 * @param freq rounding_frequency indicating the frequency to round to
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate device memory of the returned column
 *
 * @throw cudf::logic_error if input column datatype is not TIMESTAMP.
 * @return cudf::column of the same datetime resolution as the input column
 */
std::unique_ptr<cudf::column> round_datetimes(
  cudf::column_view const& column,
  rounding_frequency freq,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/** @} */  // end of group

}  // namespace datetime
}  // namespace CUDF_EXPORT cudf
