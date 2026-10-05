/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/default_stream.hpp>
#include <cudf_test/testing_main.hpp>

#include <cudf/datetime.hpp>
#include <cudf/scalar/scalar_factories.hpp>

#include <cstdint>

class DatetimeTest : public cudf::test::BaseFixture {
 public:
  cudf::test::fixed_width_column_wrapper<cudf::timestamp_ns, cudf::timestamp_ns::rep> timestamps{
    -23324234,  // 1969-12-31 23:59:59.976675766 GMT
    23432424,   // 1970-01-01 00:00:00.023432424 GMT
    987234623   // 1970-01-01 00:00:00.987234623 GMT
  };
  cudf::test::fixed_width_column_wrapper<int32_t, int32_t> months{{1, -1, 3}};
};

TEST_F(DatetimeTest, ExtractYear)
{
  cudf::datetime::extract_datetime_component(
    timestamps, cudf::datetime::datetime_component::YEAR, cudf::test::get_default_stream());
}

TEST_F(DatetimeTest, ExtractMonth)
{
  cudf::datetime::extract_datetime_component(
    timestamps, cudf::datetime::datetime_component::MONTH, cudf::test::get_default_stream());
}

TEST_F(DatetimeTest, ExtractDay)
{
  cudf::datetime::extract_datetime_component(
    timestamps, cudf::datetime::datetime_component::DAY, cudf::test::get_default_stream());
}

TEST_F(DatetimeTest, ExtractWeekday)
{
  cudf::datetime::extract_datetime_component(
    timestamps, cudf::datetime::datetime_component::WEEKDAY, cudf::test::get_default_stream());
}

TEST_F(DatetimeTest, ExtractHour)
{
  cudf::datetime::extract_datetime_component(
    timestamps, cudf::datetime::datetime_component::HOUR, cudf::test::get_default_stream());
}

TEST_F(DatetimeTest, ExtractMinute)
{
  cudf::datetime::extract_datetime_component(
    timestamps, cudf::datetime::datetime_component::MINUTE, cudf::test::get_default_stream());
}

TEST_F(DatetimeTest, ExtractSecond)
{
  cudf::datetime::extract_datetime_component(
    timestamps, cudf::datetime::datetime_component::SECOND, cudf::test::get_default_stream());
}

TEST_F(DatetimeTest, ExtractMillisecondFraction)
{
  cudf::datetime::extract_datetime_component(
    timestamps, cudf::datetime::datetime_component::MILLISECOND, cudf::test::get_default_stream());
}

TEST_F(DatetimeTest, ExtractMicrosecondFraction)
{
  cudf::datetime::extract_datetime_component(
    timestamps, cudf::datetime::datetime_component::MICROSECOND, cudf::test::get_default_stream());
}

TEST_F(DatetimeTest, ExtractNanosecondFraction)
{
  cudf::datetime::extract_datetime_component(
    timestamps, cudf::datetime::datetime_component::NANOSECOND, cudf::test::get_default_stream());
}

TEST_F(DatetimeTest, LastDayOfMonth)
{ cudf::datetime::last_day_of_month(timestamps, cudf::test::get_default_stream()); }

TEST_F(DatetimeTest, DayOfYear)
{ cudf::datetime::day_of_year(timestamps, cudf::test::get_default_stream()); }

TEST_F(DatetimeTest, AddCalendricalMonths)
{ cudf::datetime::add_calendrical_months(timestamps, months, cudf::test::get_default_stream()); }

TEST_F(DatetimeTest, AddCalendricalMonthsScalar)
{
  auto scalar = cudf::make_fixed_width_scalar(1, cudf::test::get_default_stream());

  cudf::datetime::add_calendrical_months(timestamps, *scalar, cudf::test::get_default_stream());
}

TEST_F(DatetimeTest, IsLeapYear)
{ cudf::datetime::is_leap_year(timestamps, cudf::test::get_default_stream()); }

TEST_F(DatetimeTest, DaysInMonth)
{ cudf::datetime::days_in_month(timestamps, cudf::test::get_default_stream()); }

TEST_F(DatetimeTest, ExtractQuarter)
{ cudf::datetime::extract_quarter(timestamps, cudf::test::get_default_stream()); }

TEST_F(DatetimeTest, CeilDatetimes)
{
  cudf::datetime::ceil_datetimes(
    timestamps, cudf::datetime::rounding_frequency::HOUR, cudf::test::get_default_stream());
}

TEST_F(DatetimeTest, FloorDatetimes)
{
  cudf::datetime::floor_datetimes(
    timestamps, cudf::datetime::rounding_frequency::HOUR, cudf::test::get_default_stream());
}

TEST_F(DatetimeTest, RoundDatetimes)
{
  cudf::datetime::round_datetimes(
    timestamps, cudf::datetime::rounding_frequency::HOUR, cudf::test::get_default_stream());
}

CUDF_TEST_PROGRAM_MAIN()
