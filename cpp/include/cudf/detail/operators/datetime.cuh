/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/datetime_types.hpp>
#include <cudf/detail/operators/concepts.cuh>
#include <cudf/detail/utilities/assert.cuh>
#include <cudf/utilities/export.hpp>
#include <cudf/wrappers/timestamps.hpp>

#include <cuda/std/chrono>

namespace CUDF_EXPORT cudf {
namespace detail {
namespace ops {

/** @brief Direction used when rounding a timestamp to a fixed frequency. */
enum class datetime_rounding : int8_t { CEIL = 0, FLOOR = 1, ROUND = 2 };

/**
 * @brief Extracts a calendar or time-of-day component from a timestamp.
 *
 * @tparam Component Component to extract.
 * @tparam Timestamp Input timestamp type.
 * @param ts Input timestamp.
 */
template <cudf::datetime::datetime_component Component, concepts::timestamp Timestamp>
__device__ int16_t extract_datetime_component(Timestamp ts)
{
  using cudf::datetime::datetime_component;
  using namespace cuda::std::chrono;

  auto days_since_epoch = cuda::std::chrono::floor<days>(ts);

  auto time_since_midnight = ts - days_since_epoch;

  if (time_since_midnight.count() < 0) { time_since_midnight += days(1); }

  auto const hrs  = [&] { return cuda::std::chrono::duration_cast<hours>(time_since_midnight); };
  auto const mins = [&] {
    return cuda::std::chrono::duration_cast<minutes>(time_since_midnight) - hrs();
  };
  auto const secs = [&] {
    return cuda::std::chrono::duration_cast<seconds>(time_since_midnight) - hrs() - mins();
  };
  auto const millisecs = [&] {
    return cuda::std::chrono::duration_cast<milliseconds>(time_since_midnight) - hrs() - mins() -
           secs();
  };
  auto const microsecs = [&] {
    return cuda::std::chrono::duration_cast<microseconds>(time_since_midnight) - hrs() - mins() -
           secs() - millisecs();
  };
  auto const nanosecs = [&] {
    return cuda::std::chrono::duration_cast<nanoseconds>(time_since_midnight) - hrs() - mins() -
           secs() - millisecs() - microsecs();
  };

  switch (Component) {
    case datetime_component::YEAR: return static_cast<int>(year_month_day(days_since_epoch).year());
    case datetime_component::MONTH:
      return static_cast<unsigned>(year_month_day(days_since_epoch).month());
    case datetime_component::DAY:
      return static_cast<unsigned>(year_month_day(days_since_epoch).day());
    case datetime_component::WEEKDAY:
      return year_month_weekday(days_since_epoch).weekday().iso_encoding();
    case datetime_component::HOUR: return hrs().count();
    case datetime_component::MINUTE: return mins().count();
    case datetime_component::SECOND: return secs().count();
    case datetime_component::MILLISECOND: return millisecs().count();
    case datetime_component::MICROSECOND: return microsecs().count();
    case datetime_component::NANOSECOND: return nanosecs().count();
    default: return 0;
  }
}

namespace detail {

// Use 64-bit rounding units so large epoch counts do not overflow the 32-bit day type.
static_assert(sizeof(cuda::std::chrono::minutes::rep) == sizeof(int64_t));
static_assert(sizeof(cuda::std::chrono::hours::rep) == sizeof(int64_t));

template <typename Duration, concepts::timestamp Timestamp>
__device__ auto round_datetime_to(datetime_rounding kind, Timestamp ts)
{
  switch (kind) {
    case datetime_rounding::CEIL: return cuda::std::chrono::ceil<Duration>(ts);
    case datetime_rounding::FLOOR: return cuda::std::chrono::floor<Duration>(ts);
    case datetime_rounding::ROUND: return cuda::std::chrono::round<Duration>(ts);
    default: CUDF_UNREACHABLE("Unsupported rounding kind.");
  }
}

static __device__ constexpr int16_t days_until_month[2][13] = {
  {0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 365},
  {0, 31, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335, 366}};

}  // namespace detail

/**
 * @brief Rounds a timestamp while retaining its input resolution.
 *
 * @tparam Timestamp Input timestamp type.
 * @param ts Input timestamp.
 * @param round_kind Rounding direction.
 * @param component Fixed frequency to round to.
 */
template <concepts::timestamp Timestamp>
__device__ Timestamp round_datetime(Timestamp ts,
                                    datetime_rounding round_kind,
                                    cudf::datetime::rounding_frequency component)
{
  using cudf::datetime::rounding_frequency;
  using namespace cuda::std::chrono;
  using days64 = cuda::std::chrono::duration<int64_t, cuda::std::chrono::days::period>;

  switch (component) {
    case rounding_frequency::DAY:
      return cuda::std::chrono::time_point_cast<typename Timestamp::duration>(
        detail::round_datetime_to<days64>(round_kind, ts));
    case rounding_frequency::HOUR:
      return cuda::std::chrono::time_point_cast<typename Timestamp::duration>(
        detail::round_datetime_to<cuda::std::chrono::hours>(round_kind, ts));
    case rounding_frequency::MINUTE:
      return cuda::std::chrono::time_point_cast<typename Timestamp::duration>(
        detail::round_datetime_to<cuda::std::chrono::minutes>(round_kind, ts));
    case rounding_frequency::SECOND:
      return cuda::std::chrono::time_point_cast<typename Timestamp::duration>(
        detail::round_datetime_to<duration_s>(round_kind, ts));
    case rounding_frequency::MILLISECOND:
      return cuda::std::chrono::time_point_cast<typename Timestamp::duration>(
        detail::round_datetime_to<duration_ms>(round_kind, ts));
    case rounding_frequency::MICROSECOND:
      return cuda::std::chrono::time_point_cast<typename Timestamp::duration>(
        detail::round_datetime_to<duration_us>(round_kind, ts));
    case rounding_frequency::NANOSECOND:
      return cuda::std::chrono::time_point_cast<typename Timestamp::duration>(
        detail::round_datetime_to<duration_ns>(round_kind, ts));
    default: CUDF_UNREACHABLE("Unsupported datetime rounding resolution.");
  }
}

/**
 * @brief Returns the last calendar day of the month.
 *
 * @tparam Timestamp Input timestamp type.
 * @param ts Input timestamp.
 */
template <concepts::timestamp Timestamp>
__device__ timestamp_D last_day_of_month(Timestamp ts)
{
  using namespace cuda::std::chrono;
  year_month_day const ymd(cuda::std::chrono::floor<days>(ts));
  auto const ymdl = year_month_day_last{ymd.year() / ymd.month() / last};
  return timestamp_D{sys_days{ymdl}};
}

/**
 * @brief Returns the number of days in the month.
 *
 * @tparam Timestamp Input timestamp type.
 * @param ts Input timestamp.
 */
template <concepts::timestamp Timestamp>
__device__ int16_t days_in_month(Timestamp ts)
{
  using namespace cuda::std::chrono;
  auto const date = year_month_day(cuda::std::chrono::floor<days>(ts));
  auto const ymdl = year_month_day_last(date.year() / date.month() / last);
  return static_cast<int16_t>(unsigned{ymdl.day()});
}

/**
 * @brief Returns the one-based day of the year.
 *
 * @tparam Timestamp Input timestamp type.
 * @param ts Input timestamp.
 */
template <concepts::timestamp Timestamp>
__device__ int16_t day_of_year(Timestamp ts)
{
  using namespace cuda::std::chrono;

  // Only has the days - time component is chopped off, which is what we want
  auto const days_since_epoch = cuda::std::chrono::floor<days>(ts);
  auto const date             = year_month_day(days_since_epoch);

  return detail::days_until_month[date.year().is_leap()][unsigned{date.month()} - 1] +
         unsigned{date.day()};
}

/**
 * @brief Returns the one-based calendar quarter.
 *
 * @tparam Timestamp Input timestamp type.
 * @param ts Input timestamp.
 */
template <concepts::timestamp Timestamp>
__device__ int16_t extract_quarter(Timestamp ts)
{
  using namespace cuda::std::chrono;

  // Only has the days - time component is chopped off, which is what we want
  auto const days_since_epoch = cuda::std::chrono::floor<days>(ts);
  auto const date             = year_month_day(days_since_epoch);
  auto const month            = unsigned{date.month()};

  // (x + y - 1) / y = ceil(x/y), where x and y are unsigned. x = month, y = 3
  return (month + 2) / 3;
}

/**
 * @brief Tests whether the timestamp falls in a leap year.
 *
 * @tparam Timestamp Input timestamp type.
 * @param ts Input timestamp.
 */
template <concepts::timestamp Timestamp>
__device__ bool is_leap_year(Timestamp ts)
{
  using namespace cuda::std::chrono;
  auto const days_since_epoch = cuda::std::chrono::floor<days>(ts);
  auto const date             = year_month_day(days_since_epoch);
  return date.year().is_leap();
}

/**
 * @brief Adds calendar months, clipping an invalid day to the end of the resulting month.
 *
 * @tparam Timestamp Input timestamp type.
 * @param time_val Input timestamp.
 * @param months_val Number of calendar months to add.
 */
template <concepts::timestamp Timestamp>
__device__ Timestamp add_calendrical_months(Timestamp time_val,
                                            cuda::std::chrono::months months_val)
{
  using namespace cuda::std::chrono;
  auto const days_since_epoch = cuda::std::chrono::floor<days>(time_val);

  auto const date = [&]() {
    auto const ymd = year_month_day{days_since_epoch} + months_val;
    return ymd.ok() ? ymd : ymd.year() / ymd.month() / last;
  }();

  auto const time = (time_val - days_since_epoch);

  return sys_days{date} + time;
}

}  // namespace ops
}  // namespace detail
}  // namespace CUDF_EXPORT cudf
