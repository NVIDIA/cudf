/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/io/parquet_schema.hpp>
#include <cudf/utilities/export.hpp>
#include <cudf/wrappers/timestamps.hpp>

#include <cuda/std/chrono>
#include <cuda/std/limits>
#include <cuda/std/optional>
#include <cuda/std/utility>

namespace cudf::io::parquet::detail {

/// Size in bytes of an INT96 timestamp: 8-byte nanoseconds of day followed by a 4-byte Julian day
constexpr int32_t int96_size = 12;

/// Julian day number of the Unix epoch (1970-01-01), as used by INT96 timestamps: days from the
/// Julian date epoch (-4713-11-24 12:00:00 UTC), rounded up
constexpr int32_t julian_day_unix_epoch = [] {
  using namespace cuda::std::chrono;
  return static_cast<int32_t>(
    ceil<days>(sys_days{January / 1 / 1970} - (sys_days{November / 24 / -4713} + hours{12}))
      .count());
}();

// The well-known Julian day number of 1970-01-01, as used by Spark and Arrow
static_assert(julian_day_unix_epoch == 2'440'588);

/**
 * @brief Converts the two halves of an INT96 timestamp to a 128-bit count of `Duration` since the
 * epoch, which can't overflow
 *
 * @tparam Duration Output duration type, e.g. `cudf::duration_ms`
 * @param nanos Nanoseconds of the day (first 8 bytes of the INT96 value)
 * @param julian_days Julian day number (last 4 bytes of the INT96 value)
 * @return 128-bit count of `Duration` since the Unix epoch
 */
template <typename Duration>
[[nodiscard]] CUDF_HOST_DEVICE inline __int128_t int96_to_ticks(int64_t nanos, int32_t julian_days)
{
  using cuda::std::chrono::duration_cast;
  auto const ticks_per_day = duration_cast<Duration>(cudf::duration_D{1}).count();
  // TBD: Should be noon instead of midnight, but this matches pyarrow
  return __int128_t{int64_t{julian_days} - julian_day_unix_epoch} * ticks_per_day +
         duration_cast<Duration>(cudf::duration_ns{nanos}).count();
}

/**
 * @brief Converts the two halves of an INT96 timestamp to a count of `Duration` since the epoch.
 *
 * @tparam Duration Output duration type, e.g. `cudf::duration_ms`
 * @param nanos Nanoseconds of the day (first 8 bytes of the INT96 value)
 * @param julian_days Julian day number (last 4 bytes of the INT96 value)
 * @return Count of `Duration` since the Unix epoch
 */
template <typename Duration>
[[nodiscard]] CUDF_HOST_DEVICE inline typename Duration::rep int96_to_duration(int64_t nanos,
                                                                               int32_t julian_days)
{
  return static_cast<typename Duration::rep>(int96_to_ticks<Duration>(nanos, julian_days));
}

/**
 * @brief Checks whether an INT96 timestamp fits in `Duration` without overflow
 *
 * @tparam Duration Output duration type, e.g. `cudf::duration_ns`
 * @param nanos Nanoseconds of the day (first 8 bytes of the INT96 value)
 * @param julian_days Julian day number (last 4 bytes of the INT96 value)
 * @return Whether `int96_to_duration<Duration>` is representable
 */
template <typename Duration>
[[nodiscard]] CUDF_HOST_DEVICE inline bool int96_fits_duration(int64_t nanos, int32_t julian_days)
{
  using rep        = typename Duration::rep;
  auto const ticks = int96_to_ticks<Duration>(nanos, julian_days);
  return ticks >= cuda::std::numeric_limits<rep>::min() and
         ticks <= cuda::std::numeric_limits<rep>::max();
}

/**
 * @brief Splits a timestamp into the two halves of an INT96 timestamp
 *
 * @tparam Duration Input duration type, e.g. `cudf::duration_ms`
 * @param duration Time since the Unix epoch
 * @return Pair of nanoseconds of the day and Julian day number
 */
template <typename Duration>
[[nodiscard]] CUDF_HOST_DEVICE inline cuda::std::pair<int64_t, int32_t> duration_to_int96(
  Duration duration)
{
  using cuda::std::chrono::duration_cast;
  // Floor-divide the count directly, as converting floored days back to `Duration` can overflow
  auto const ticks_per_day = duration_cast<Duration>(cudf::duration_D{1}).count();
  auto days                = duration.count() / ticks_per_day;
  auto time_of_day         = duration.count() % ticks_per_day;
  if (time_of_day < 0) {
    --days;
    time_of_day += ticks_per_day;
  }
  return {duration_cast<cudf::duration_ns>(Duration{time_of_day}).count(),
          static_cast<int32_t>(days) + julian_day_unix_epoch};
}

/**
 * @brief Converts the two halves of an INT96 timestamp to a 64-bit timestamp at `clock_rate`.
 *
 * @param nanos Nanoseconds of the day (first 8 bytes of the INT96 value)
 * @param julian_days Julian day number (last 4 bytes of the INT96 value)
 * @param clock_rate Clock rate of the output timestamp column (e.g. 1000 for ms)
 * @return Timestamp value since the Unix epoch, in units of `clock_rate` (nanoseconds by default)
 */
[[nodiscard]] CUDF_HOST_DEVICE inline int64_t int96_to_timestamp(int64_t nanos,
                                                                 int32_t julian_days,
                                                                 int32_t clock_rate)
{
  switch (clock_rate) {
    case 1: return int96_to_duration<cudf::duration_s>(nanos, julian_days);
    case 1'000: return int96_to_duration<cudf::duration_ms>(nanos, julian_days);
    case 1'000'000: return int96_to_duration<cudf::duration_us>(nanos, julian_days);
    case 1'000'000'000:
    default: return int96_to_duration<cudf::duration_ns>(nanos, julian_days);
  }
}

/**
 * @brief Computes the timestamp scale between a Parquet timestamp logical type's
 * native precision and the output column's clock rate.
 *
 * @param logical_type Logical type of the column
 * @param clock_rate Clock rate of the output timestamp column (e.g. 1000 for ms)
 * @return Scale factor; <0 means divide by `-scale`, >0 means multiply by `scale`, 0 means no-op
 */
[[nodiscard]] CUDF_HOST_DEVICE inline int32_t calc_timestamp_scale(
  cuda::std::optional<LogicalType> logical_type, int32_t clock_rate)
{
  if (not logical_type.has_value()) { return 0; }

  auto units = 0;
  if (logical_type->is_timestamp_millis()) {
    units = cudf::timestamp_ms::period::den;
  } else if (logical_type->is_timestamp_micros()) {
    units = cudf::timestamp_us::period::den;
  } else if (logical_type->is_timestamp_nanos()) {
    units = cudf::timestamp_ns::period::den;
  }

  if (units and units != clock_rate) {
    return (clock_rate < units) ? -(units / clock_rate) : (clock_rate / units);
  }

  return 0;
}

/**
 * @brief Apply timestamp scaling to a raw decoded value.
 *
 * @param value Raw decoded int64 integer value
 * @param ts_scale Timestamp scale
 * @return Scaled 64-bit timestamp value
 */
[[nodiscard]] CUDF_HOST_DEVICE inline int64_t apply_ts_scale(int64_t value, int32_t ts_scale)
{
  if (ts_scale == 0) {
    return value;
  } else if (ts_scale < 0) {
    // Floor division towards negative infinity
    auto const divisor   = -ts_scale;
    auto quotient        = value / divisor;
    auto const remainder = value % divisor;
    return (remainder < 0) ? quotient - 1 : quotient;
  } else {
    return value * ts_scale;
  }
}

}  // namespace cudf::io::parquet::detail
