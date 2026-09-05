/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/iterator_utilities.hpp>
#include <cudf_test/testing_main.hpp>
#include <cudf_test/timestamp_utilities.cuh>
#include <cudf_test/type_lists.hpp>

#include <cudf/column/column_factories.hpp>
#include <cudf/datetime.hpp>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/types.hpp>
#include <cudf/wrappers/timestamps.hpp>

#define XXX false  // stub for null values

constexpr cudf::test::debug_output_level verbosity{cudf::test::debug_output_level::ALL_ERRORS};

template <typename T>
struct NonTimestampTest : public cudf::test::BaseFixture {
  cudf::data_type type() { return cudf::data_type{cudf::type_to_id<T>()}; }
};

using NonTimestampTypes =
  cudf::test::Concat<cudf::test::NumericTypes, cudf::test::StringTypes, cudf::test::DurationTypes>;

TYPED_TEST_SUITE(NonTimestampTest, NonTimestampTypes);

TYPED_TEST(NonTimestampTest, TestThrowsOnNonTimestamp)
{
  using T = TypeParam;
  using namespace cudf::test;
  using namespace cudf::datetime;
  using namespace cuda::std::chrono;

  cudf::data_type dtype{cudf::type_to_id<T>()};
  cudf::column col{
    dtype, 0, rmm::device_buffer{}, cudf::create_null_mask(0, cudf::mask_state::UNALLOCATED), 0};

  EXPECT_THROW(extract_datetime_component(col, cudf::datetime::datetime_component::YEAR),
               cudf::logic_error);
  EXPECT_THROW(extract_datetime_component(col, cudf::datetime::datetime_component::MONTH),
               cudf::logic_error);
  EXPECT_THROW(extract_datetime_component(col, cudf::datetime::datetime_component::DAY),
               cudf::logic_error);
  EXPECT_THROW(extract_datetime_component(col, cudf::datetime::datetime_component::WEEKDAY),
               cudf::logic_error);
  EXPECT_THROW(extract_datetime_component(col, cudf::datetime::datetime_component::HOUR),
               cudf::logic_error);
  EXPECT_THROW(extract_datetime_component(col, cudf::datetime::datetime_component::MINUTE),
               cudf::logic_error);
  EXPECT_THROW(extract_datetime_component(col, cudf::datetime::datetime_component::SECOND),
               cudf::logic_error);
  EXPECT_THROW(extract_datetime_component(col, cudf::datetime::datetime_component::MILLISECOND),
               cudf::logic_error);
  EXPECT_THROW(extract_datetime_component(col, cudf::datetime::datetime_component::MICROSECOND),
               cudf::logic_error);
  EXPECT_THROW(extract_datetime_component(col, cudf::datetime::datetime_component::NANOSECOND),
               cudf::logic_error);
  EXPECT_THROW(last_day_of_month(col), cudf::logic_error);
  EXPECT_THROW(day_of_year(col), cudf::logic_error);
  EXPECT_THROW(add_calendrical_months(col, *cudf::make_empty_column(cudf::type_id::INT16)),
               cudf::logic_error);
}

struct BasicDatetimeOpsTest : public cudf::test::BaseFixture {};

TEST_F(BasicDatetimeOpsTest, TestExtractingDatetimeComponents)
{
  using namespace cudf::test;
  using namespace cudf::datetime;
  using namespace cuda::std::chrono;

  auto timestamps_D =
    cudf::test::fixed_width_column_wrapper<cudf::timestamp_D, cudf::timestamp_D::rep>{
      -1528,  // 1965-10-26 GMT
      17716,  // 2018-07-04 GMT
      19382   // 2023-01-25 GMT
    };

  auto timestamps_s =
    cudf::test::fixed_width_column_wrapper<cudf::timestamp_s, cudf::timestamp_s::rep>{
      -131968728,  // 1965-10-26 14:01:12 GMT
      1530705600,  // 2018-07-04 12:00:00 GMT
      1674631932   // 2023-01-25 07:32:12 GMT
    };

  auto timestamps_ms =
    cudf::test::fixed_width_column_wrapper<cudf::timestamp_ms, cudf::timestamp_ms::rep>{
      -131968727238,  // 1965-10-26 14:01:12.762 GMT
      1530705600000,  // 2018-07-04 12:00:00.000 GMT
      1674631932929   // 2023-01-25 07:32:12.929 GMT
    };

  auto timestamps_ns =
    cudf::test::fixed_width_column_wrapper<cudf::timestamp_ns, cudf::timestamp_ns::rep>{
      -23324234,  // 1969-12-31 23:59:59.976675766 GMT
      23432424,   // 1970-01-01 00:00:00.023432424 GMT
      987234623   // 1970-01-01 00:00:00.987234623 GMT
    };

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    *extract_datetime_component(timestamps_D, cudf::datetime::datetime_component::YEAR),
    fixed_width_column_wrapper<int16_t>{1965, 2018, 2023});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    *extract_datetime_component(timestamps_s, cudf::datetime::datetime_component::YEAR),
    fixed_width_column_wrapper<int16_t>{1965, 2018, 2023});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    *extract_datetime_component(timestamps_ms, cudf::datetime::datetime_component::YEAR),
    fixed_width_column_wrapper<int16_t>{1965, 2018, 2023});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    *extract_datetime_component(timestamps_ns, cudf::datetime::datetime_component::YEAR),
    fixed_width_column_wrapper<int16_t>{1969, 1970, 1970});

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    *extract_datetime_component(timestamps_D, cudf::datetime::datetime_component::MONTH),
    fixed_width_column_wrapper<int16_t>{10, 7, 1});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    *extract_datetime_component(timestamps_s, cudf::datetime::datetime_component::MONTH),
    fixed_width_column_wrapper<int16_t>{10, 7, 1});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    *extract_datetime_component(timestamps_ms, cudf::datetime::datetime_component::MONTH),
    fixed_width_column_wrapper<int16_t>{10, 7, 1});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    *extract_datetime_component(timestamps_ns, cudf::datetime::datetime_component::MONTH),
    fixed_width_column_wrapper<int16_t>{12, 1, 1});

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    *extract_datetime_component(timestamps_D, cudf::datetime::datetime_component::DAY),
    fixed_width_column_wrapper<int16_t>{26, 4, 25});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    *extract_datetime_component(timestamps_s, cudf::datetime::datetime_component::DAY),
    fixed_width_column_wrapper<int16_t>{26, 4, 25});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    *extract_datetime_component(timestamps_ms, cudf::datetime::datetime_component::DAY),
    fixed_width_column_wrapper<int16_t>{26, 4, 25});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    *extract_datetime_component(timestamps_ns, cudf::datetime::datetime_component::DAY),
    fixed_width_column_wrapper<int16_t>{31, 1, 1});

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    *extract_datetime_component(timestamps_D, cudf::datetime::datetime_component::WEEKDAY),
    fixed_width_column_wrapper<int16_t>{2, 3, 3});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    *extract_datetime_component(timestamps_s, cudf::datetime::datetime_component::WEEKDAY),
    fixed_width_column_wrapper<int16_t>{2, 3, 3});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    *extract_datetime_component(timestamps_ms, cudf::datetime::datetime_component::WEEKDAY),
    fixed_width_column_wrapper<int16_t>{2, 3, 3});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    *extract_datetime_component(timestamps_ms, cudf::datetime::datetime_component::WEEKDAY),
    fixed_width_column_wrapper<int16_t>{2, 3, 3});

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    *extract_datetime_component(timestamps_D, cudf::datetime::datetime_component::HOUR),
    fixed_width_column_wrapper<int16_t>{0, 0, 0});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    *extract_datetime_component(timestamps_s, cudf::datetime::datetime_component::HOUR),
    fixed_width_column_wrapper<int16_t>{14, 12, 7});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    *extract_datetime_component(timestamps_ms, cudf::datetime::datetime_component::HOUR),
    fixed_width_column_wrapper<int16_t>{14, 12, 7});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    *extract_datetime_component(timestamps_ns, cudf::datetime::datetime_component::HOUR),
    fixed_width_column_wrapper<int16_t>{23, 0, 0});

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    *extract_datetime_component(timestamps_D, cudf::datetime::datetime_component::MINUTE),
    fixed_width_column_wrapper<int16_t>{0, 0, 0});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    *extract_datetime_component(timestamps_s, cudf::datetime::datetime_component::MINUTE),
    fixed_width_column_wrapper<int16_t>{1, 0, 32});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    *extract_datetime_component(timestamps_ms, cudf::datetime::datetime_component::MINUTE),
    fixed_width_column_wrapper<int16_t>{1, 0, 32});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    *extract_datetime_component(timestamps_ns, cudf::datetime::datetime_component::MINUTE),
    fixed_width_column_wrapper<int16_t>{59, 0, 0});

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    *extract_datetime_component(timestamps_D, cudf::datetime::datetime_component::SECOND),
    fixed_width_column_wrapper<int16_t>{0, 0, 0});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    *extract_datetime_component(timestamps_s, cudf::datetime::datetime_component::SECOND),
    fixed_width_column_wrapper<int16_t>{12, 0, 12});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    *extract_datetime_component(timestamps_ms, cudf::datetime::datetime_component::SECOND),
    fixed_width_column_wrapper<int16_t>{12, 0, 12});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    *extract_datetime_component(timestamps_ns, cudf::datetime::datetime_component::SECOND),
    fixed_width_column_wrapper<int16_t>{59, 0, 0});

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    *extract_datetime_component(timestamps_D, cudf::datetime::datetime_component::MILLISECOND),
    fixed_width_column_wrapper<int16_t>{0, 0, 0});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    *extract_datetime_component(timestamps_s, cudf::datetime::datetime_component::MILLISECOND),
    fixed_width_column_wrapper<int16_t>{0, 0, 0});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    *extract_datetime_component(timestamps_ms, cudf::datetime::datetime_component::MILLISECOND),
    fixed_width_column_wrapper<int16_t>{762, 0, 929});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    *extract_datetime_component(timestamps_ns, cudf::datetime::datetime_component::MILLISECOND),
    fixed_width_column_wrapper<int16_t>{976, 23, 987});

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    *extract_datetime_component(timestamps_D, cudf::datetime::datetime_component::MICROSECOND),
    fixed_width_column_wrapper<int16_t>{0, 0, 0});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    *extract_datetime_component(timestamps_s, cudf::datetime::datetime_component::MICROSECOND),
    fixed_width_column_wrapper<int16_t>{0, 0, 0});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    *extract_datetime_component(timestamps_ms, cudf::datetime::datetime_component::MICROSECOND),
    fixed_width_column_wrapper<int16_t>{0, 0, 0});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    *extract_datetime_component(timestamps_ns, cudf::datetime::datetime_component::MICROSECOND),
    fixed_width_column_wrapper<int16_t>{675, 432, 234});

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    *extract_datetime_component(timestamps_D, cudf::datetime::datetime_component::NANOSECOND),
    fixed_width_column_wrapper<int16_t>{0, 0, 0});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    *extract_datetime_component(timestamps_s, cudf::datetime::datetime_component::NANOSECOND),
    fixed_width_column_wrapper<int16_t>{0, 0, 0});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    *extract_datetime_component(timestamps_ms, cudf::datetime::datetime_component::NANOSECOND),
    fixed_width_column_wrapper<int16_t>{0, 0, 0});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    *extract_datetime_component(timestamps_ns, cudf::datetime::datetime_component::NANOSECOND),
    fixed_width_column_wrapper<int16_t>{766, 424, 623});
}

template <typename T>
struct TypedDatetimeOpsTest : public cudf::test::BaseFixture {
  cudf::size_type size() { return cudf::size_type(10); }
  cudf::data_type type() { return cudf::data_type{cudf::type_to_id<T>()}; }
};

TYPED_TEST_SUITE(TypedDatetimeOpsTest, cudf::test::TimestampTypes);

TYPED_TEST(TypedDatetimeOpsTest, TestEmptyColumns)
{
  using T = TypeParam;
  using namespace cudf::test;
  using namespace cudf::datetime;
  using namespace cuda::std::chrono;

  auto int16s_dtype     = cudf::data_type{cudf::type_to_id<int16_t>()};
  auto timestamps_dtype = cudf::data_type{cudf::type_to_id<T>()};

  cudf::column int16s{int16s_dtype,
                      0,
                      rmm::device_buffer{},
                      cudf::create_null_mask(0, cudf::mask_state::UNALLOCATED),
                      0};
  cudf::column timestamps{timestamps_dtype,
                          0,
                          rmm::device_buffer{},
                          cudf::create_null_mask(0, cudf::mask_state::UNALLOCATED),
                          0};

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    *extract_datetime_component(timestamps, cudf::datetime::datetime_component::YEAR), int16s);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    *extract_datetime_component(timestamps, cudf::datetime::datetime_component::MONTH), int16s);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    *extract_datetime_component(timestamps, cudf::datetime::datetime_component::DAY), int16s);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    *extract_datetime_component(timestamps, cudf::datetime::datetime_component::WEEKDAY), int16s);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    *extract_datetime_component(timestamps, cudf::datetime::datetime_component::HOUR), int16s);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    *extract_datetime_component(timestamps, cudf::datetime::datetime_component::MINUTE), int16s);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    *extract_datetime_component(timestamps, cudf::datetime::datetime_component::SECOND), int16s);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    *extract_datetime_component(timestamps, cudf::datetime::datetime_component::MILLISECOND),
    int16s);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    *extract_datetime_component(timestamps, cudf::datetime::datetime_component::MICROSECOND),
    int16s);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    *extract_datetime_component(timestamps, cudf::datetime::datetime_component::NANOSECOND),
    int16s);
}

TYPED_TEST(TypedDatetimeOpsTest, TestExtractingGeneratedDatetimeComponents)
{
  using T = TypeParam;
  using namespace cudf::test;
  using namespace cudf::datetime;
  using namespace cuda::std::chrono;

  auto start      = milliseconds(-2500000000000);  // Sat, 11 Oct 1890 19:33:20 GMT
  auto stop       = milliseconds(2500000000000);   // Mon, 22 Mar 2049 04:26:40 GMT
  auto timestamps = generate_timestamps<T>(this->size(), time_point_ms(start), time_point_ms(stop));

  auto expected_years =
    fixed_width_column_wrapper<int16_t>{1890, 1906, 1922, 1938, 1954, 1970, 1985, 2001, 2017, 2033};
  auto expected_months   = fixed_width_column_wrapper<int16_t>{10, 8, 6, 4, 2, 1, 11, 9, 7, 5};
  auto expected_days     = fixed_width_column_wrapper<int16_t>{11, 16, 20, 24, 26, 1, 5, 9, 14, 18};
  auto expected_weekdays = fixed_width_column_wrapper<int16_t>{6, 4, 2, 7, 5, 4, 2, 7, 5, 3};
  auto expected_hours    = fixed_width_column_wrapper<int16_t>{19, 20, 21, 22, 23, 0, 0, 1, 2, 3};
  auto expected_minutes = fixed_width_column_wrapper<int16_t>{33, 26, 20, 13, 6, 0, 53, 46, 40, 33};
  auto expected_seconds = fixed_width_column_wrapper<int16_t>{20, 40, 0, 20, 40, 0, 20, 40, 0, 20};

  // Special cases for timestamp_D: zero out the expected hh/mm/ss cols
  if (std::is_same_v<TypeParam, cudf::timestamp_D>) {
    expected_hours   = fixed_width_column_wrapper<int16_t>{0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    expected_minutes = fixed_width_column_wrapper<int16_t>{0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    expected_seconds = fixed_width_column_wrapper<int16_t>{0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  }

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    *extract_datetime_component(timestamps, cudf::datetime::datetime_component::YEAR),
    expected_years);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    *extract_datetime_component(timestamps, cudf::datetime::datetime_component::MONTH),
    expected_months);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    *extract_datetime_component(timestamps, cudf::datetime::datetime_component::DAY),
    expected_days);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    *extract_datetime_component(timestamps, cudf::datetime::datetime_component::WEEKDAY),
    expected_weekdays);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    *extract_datetime_component(timestamps, cudf::datetime::datetime_component::HOUR),
    expected_hours);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    *extract_datetime_component(timestamps, cudf::datetime::datetime_component::MINUTE),
    expected_minutes);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    *extract_datetime_component(timestamps, cudf::datetime::datetime_component::SECOND),
    expected_seconds);
}

TYPED_TEST(TypedDatetimeOpsTest, TestExtractingGeneratedNullableDatetimeComponents)
{
  using T = TypeParam;
  using namespace cudf::test;
  using namespace cudf::datetime;
  using namespace cuda::std::chrono;

  auto start = milliseconds(-2500000000000);  // Sat, 11 Oct 1890 19:33:20 GMT
  auto stop  = milliseconds(2500000000000);   // Mon, 22 Mar 2049 04:26:40 GMT
  auto timestamps =
    generate_timestamps<T, true>(this->size(), time_point_ms(start), time_point_ms(stop));

  auto expected_years = fixed_width_column_wrapper<int16_t>{
    {1890, 1906, 1922, 1938, 1954, 1970, 1985, 2001, 2017, 2033},
    {true, false, true, false, true, false, true, false, true, false}};
  auto expected_months = fixed_width_column_wrapper<int16_t>{
    {10, 8, 6, 4, 2, 1, 11, 9, 7, 5},
    {true, false, true, false, true, false, true, false, true, false}};
  auto expected_days = fixed_width_column_wrapper<int16_t>{
    {11, 16, 20, 24, 26, 1, 5, 9, 14, 18},
    {true, false, true, false, true, false, true, false, true, false}};
  auto expected_weekdays = fixed_width_column_wrapper<int16_t>{
    {6, 4, 2, 7, 5, 4, 2, 7, 5, 3},
    {true, false, true, false, true, false, true, false, true, false}};
  auto expected_hours = fixed_width_column_wrapper<int16_t>{
    {19, 20, 21, 22, 23, 0, 0, 1, 2, 3},
    {true, false, true, false, true, false, true, false, true, false}};
  auto expected_minutes = fixed_width_column_wrapper<int16_t>{
    {33, 26, 20, 13, 6, 0, 53, 46, 40, 33},
    {true, false, true, false, true, false, true, false, true, false}};
  auto expected_seconds = fixed_width_column_wrapper<int16_t>{
    {20, 40, 0, 20, 40, 0, 20, 40, 0, 20},
    {true, false, true, false, true, false, true, false, true, false}};

  // Special cases for timestamp_D: zero out the expected hh/mm/ss cols
  if (std::is_same_v<TypeParam, cudf::timestamp_D>) {
    expected_hours = fixed_width_column_wrapper<int16_t>{
      {0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
      {true, false, true, false, true, false, true, false, true, false}};
    expected_minutes = fixed_width_column_wrapper<int16_t>{
      {0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
      {true, false, true, false, true, false, true, false, true, false}};
    expected_seconds = fixed_width_column_wrapper<int16_t>{
      {0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
      {true, false, true, false, true, false, true, false, true, false}};
  }

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    *extract_datetime_component(timestamps, cudf::datetime::datetime_component::YEAR),
    expected_years);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    *extract_datetime_component(timestamps, cudf::datetime::datetime_component::MONTH),
    expected_months);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    *extract_datetime_component(timestamps, cudf::datetime::datetime_component::DAY),
    expected_days);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    *extract_datetime_component(timestamps, cudf::datetime::datetime_component::WEEKDAY),
    expected_weekdays);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    *extract_datetime_component(timestamps, cudf::datetime::datetime_component::HOUR),
    expected_hours);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    *extract_datetime_component(timestamps, cudf::datetime::datetime_component::MINUTE),
    expected_minutes);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    *extract_datetime_component(timestamps, cudf::datetime::datetime_component::SECOND),
    expected_seconds);
}

TEST_F(BasicDatetimeOpsTest, TestLastDayOfMonthWithSeconds)
{
  using namespace cudf::test;
  using namespace cudf::datetime;
  using namespace cuda::std::chrono;

  // Time in seconds since epoch
  // Dates converted using epochconverter.com
  auto timestamps_s = fixed_width_column_wrapper<cudf::timestamp_s, cudf::timestamp_s::rep>{
    662688000L,   // 1991-01-01 00:00:00 GMT
    949496401L,   // 2000-02-02 13:00:01 GMT - leap year
    4106854801L,  // 2100-02-21 01:00:01 GMT - not a leap year
    1582391837L,  // 2020-02-22 17:17:17 GMT - leap year
    1363046401L,  // 2013-03-12 00:00:01 GMT
    1302696000L,  // 2011-04-13 12:00:00 GMT
    1495800001L,  // 2017-05-26 12:00:01 GMT
    1056931201L,  // 2003-06-30 00:00:01 GMT - already last day
    1031961599L,  // 2002-09-13 23:59:59 GMT
    0L,           // This is the UNIX epoch - 1970-01-01
    -131968728L,  // 1965-10-26 14:01:12 GMT
  };

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    *last_day_of_month(timestamps_s),
    fixed_width_column_wrapper<cudf::timestamp_D, cudf::timestamp_D::rep>{
      7700,   // 1991-01-31
      11016,  // 2000-02-29
      47540,  // 2100-02-28
      18321,  // 2020-02-29
      15795,  // 2013-03-31
      15094,  // 2011-04-30
      17317,  // 2017-05-31
      12233,  // 2003-06-30
      11960,  // 2002-09-30
      30,     // This is the UNIX epoch - when rounded up becomes 1970-01-31
      -1523   // 1965-10-31
    },
    verbosity);
}

TEST_F(BasicDatetimeOpsTest, TestLastDayOfMonthWithDate)
{
  using namespace cudf::test;
  using namespace cudf::datetime;
  using namespace cuda::std::chrono;

  // Time in days since epoch
  // Dates converted using epochconverter.com
  // Make some nullable fields as well
  auto timestamps_d = fixed_width_column_wrapper<cudf::timestamp_D, cudf::timestamp_D::rep>{
    {
      999,    // Random nullable field
      0,      // This is the UNIX epoch - 1970-01-01
      44376,  // 2091-07-01 00:00:00 GMT
      47695,  // 2100-08-02 00:00:00 GMT
      3,      // Random nullable field
      66068,  // 2150-11-21 00:00:00 GMT
      22270,  // 2030-12-22 00:00:00 GMT
      111,    // Random nullable field
    },
    {false, true, true, true, false, true, true, false},
  };

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    *last_day_of_month(timestamps_d),
    fixed_width_column_wrapper<cudf::timestamp_D, cudf::timestamp_D::rep>{
      {
        999,    // Random nullable field
        30,     // This is the UNIX epoch - when rounded up becomes 1970-01-31
        44406,  // 2091-07-31
        47724,  // 2100-08-31
        3,      // Random nullable field
        66077,  // 2150-11-30
        22279,  // 2030-12-31
        111     // Random nullable field
      },
      {false, true, true, true, false, true, true, false}},
    verbosity);
}

TEST_F(BasicDatetimeOpsTest, TestDayOfYearWithDate)
{
  using namespace cudf::test;
  using namespace cudf::datetime;
  using namespace cuda::std::chrono;

  // Day number in the year
  // Dates converted using epochconverter.com
  // Make some nullable fields as well
  auto timestamps_d =
    cudf::test::fixed_width_column_wrapper<cudf::timestamp_s, cudf::timestamp_s::rep>{
      {
        999L,         // Random nullable field
        0L,           // This is the UNIX epoch - 1970-01-01
        1577865600L,  // 2020-01-01 00:00:00 GMT
        1581667200L,  // 2020-02-14 00:00:00 GMT
        3L,           // Random nullable field
        1609401600L,  // 2020-12-31 00:00:00 GMT
        4133923200L,  // 2100-12-31 00:00:00 GMT
        111L,         // Random nullable field
        -2180188800L  // 1900-11-30 00:00:00 GMT
      },
      {false, true, true, true, false, true, true, false, true}};

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*day_of_year(timestamps_d),
                                 fixed_width_column_wrapper<int16_t>{
                                   {
                                     999,  // Random nullable field
                                     1,    // Number of year days until UNIX epoch time
                                     1,    // Number of year days until 2020-01-01
                                     45,   // Number of year days until 2020-02-14
                                     3,    // Random nullable field
                                     366,  // Number of year days until 2020-12-31
                                     365,  // Number of year days until 2100-12-31
                                     111,  // Random nullable field
                                     334   // Number of year days until 1900-11-30
                                   },
                                   {false, true, true, true, false, true, true, false, true},
                                 },
                                 verbosity);
}

TEST_F(BasicDatetimeOpsTest, TestDayOfYearWithEmptyColumn)
{
  using namespace cudf::test;
  using namespace cudf::datetime;
  using namespace cuda::std::chrono;

  // Create an empty column
  auto timestamps_d = fixed_width_column_wrapper<cudf::timestamp_s>{};
  auto out_col      = day_of_year(timestamps_d);
  EXPECT_EQ(out_col->size(), 0);
}

TEST_F(BasicDatetimeOpsTest, TestAddMonthsWithInvalidColType)
{
  using namespace cudf::test;
  using namespace cudf::datetime;
  using namespace cuda::std::chrono;

  // Time in seconds since epoch
  // Dates converted using epochconverter.com
  auto timestamps_s =
    cudf::test::fixed_width_column_wrapper<cudf::timestamp_s, cudf::timestamp_s::rep>{
      662688000L  // 1991-01-01 00:00:00 GMT
    };

  // Months has to be an INT16 or INT32 type
  EXPECT_NO_THROW(
    add_calendrical_months(timestamps_s, cudf::test::fixed_width_column_wrapper<int32_t>{-2}));
  EXPECT_NO_THROW(
    add_calendrical_months(timestamps_s, cudf::test::fixed_width_column_wrapper<int16_t>{-2}));

  EXPECT_THROW(
    add_calendrical_months(timestamps_s, cudf::test::fixed_width_column_wrapper<int8_t>{-2}),
    cudf::logic_error);
  EXPECT_THROW(
    add_calendrical_months(timestamps_s, cudf::test::fixed_width_column_wrapper<int64_t>{-2}),
    cudf::logic_error);
}

TEST_F(BasicDatetimeOpsTest, TestAddMonthsWithInvalidScalarType)
{
  using namespace cudf::test;
  using namespace cudf::datetime;
  using namespace cuda::std::chrono;

  // Time in seconds since epoch
  // Dates converted using epochconverter.com
  auto timestamps_s = fixed_width_column_wrapper<cudf::timestamp_s, cudf::timestamp_s::rep>{
    662688000L  // 1991-01-01 00:00:00 GMT
  };

  // Months has to be an INT16 or INT32 type
  EXPECT_NO_THROW(add_calendrical_months(timestamps_s, *cudf::make_fixed_width_scalar<int32_t>(5)));
  EXPECT_NO_THROW(
    add_calendrical_months(timestamps_s, *cudf::make_fixed_width_scalar<int16_t>(-3)));

  EXPECT_THROW(add_calendrical_months(timestamps_s, *cudf::make_fixed_width_scalar<int8_t>(-3)),
               cudf::logic_error);
  EXPECT_THROW(add_calendrical_months(timestamps_s, *cudf::make_fixed_width_scalar<int64_t>(-3)),
               cudf::logic_error);
}

TEST_F(BasicDatetimeOpsTest, TestAddMonthsWithIncorrectColSizes)
{
  using namespace cudf::test;
  using namespace cudf::datetime;
  using namespace cuda::std::chrono;

  // Time in seconds since epoch
  // Dates converted using epochconverter.com
  auto timestamps_s =
    cudf::test::fixed_width_column_wrapper<cudf::timestamp_s, cudf::timestamp_s::rep>{
      662688000L  // 1991-01-01 00:00:00 GMT
    };

  // Provide more number of months rows than timestamp rows
  auto months = cudf::test::fixed_width_column_wrapper<int16_t>{-2, 3};

  EXPECT_THROW(add_calendrical_months(timestamps_s, months), cudf::logic_error);
}

using ValidMonthIntegerType = cudf::test::Types<int16_t, int32_t>;

template <typename T>
struct TypedAddMonthsTest : public cudf::test::BaseFixture {};

TYPED_TEST_SUITE(TypedAddMonthsTest, ValidMonthIntegerType);

TYPED_TEST(TypedAddMonthsTest, TestAddMonthsWithSeconds)
{
  using namespace cudf::test;
  using namespace cudf::datetime;
  using namespace cuda::std::chrono;

  // Time in seconds since epoch
  // Dates converted using epochconverter.com
  auto timestamps_s =
    cudf::test::fixed_width_column_wrapper<cudf::timestamp_s, cudf::timestamp_s::rep>{
      662688000L,   // 1991-01-01 00:00:00 GMT
      949496401L,   // 2000-02-02 13:00:01 GMT - leap year
      1056931201L,  // 2003-06-30 00:00:01 GMT - last day of month
      1056964201L,  // 2003-06-30 09:10:01 GMT - last day of month
      1056974401L,  // 2003-06-30 12:00:01 GMT - last day of month
      1056994021L,  // 2003-06-30 17:27:01 GMT - last day of month
      0L,           // This is the UNIX epoch - 1970-01-01
      0L,           // This is the UNIX epoch - 1970-01-01
      -131586588L,  // 1965-10-31 00:10:12 GMT
      -131550590L,  // 1965-10-31 10:10:10 GMT
      -131544000L,  // 1965-10-31 12:00:00 GMT
      -131536728L   // 1965-10-31 14:01:12 GMT
    };

  auto const months =
    cudf::test::fixed_width_column_wrapper<TypeParam>{-2, 6, -1, 1, -4, 8, -2, 10, 4, -20, 1, 3};

  auto const expected =
    cudf::test::fixed_width_column_wrapper<cudf::timestamp_s, cudf::timestamp_s::rep>{
      657417600L,   // 1990-11-01 00:00:00 GMT
      965221201L,   // 2000-08-02 13:00:01 GMT
      1054252801L,  // 2003-05-30 00:00:01 GMT
      1059556201L,  // 2003-07-30 09:10:01 GMT
      1046433601L,  // 2003-02-28 12:00:01 GMT
      1078075621L,  // 2004-02-29 17:27:01 GMT
      -5270400L,    // 1969-11-01
      26265600L,    // 1970-11-01
      -121218588L,  // 1966-02-28 00:10:12 GMT
      -184254590L,  // 1964-02-29 10:10:10 GMT
      -128952000L,  // 1965-11-30 12:00:00 GMT
      -123587928L   // 1966-01-31 14:01:12 GMT
    };

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    *add_calendrical_months(timestamps_s, months), expected, verbosity);
}

TYPED_TEST(TypedAddMonthsTest, TestAddScalarMonthsWithSeconds)
{
  using namespace cudf::test;
  using namespace cudf::datetime;
  using namespace cuda::std::chrono;

  // Time in seconds since epoch
  // Dates converted using epochconverter.com
  auto timestamps_s = fixed_width_column_wrapper<cudf::timestamp_s, cudf::timestamp_s::rep>{
    662688000L,   // 1991-01-01 00:00:00 GMT
    949496401L,   // 2000-02-02 13:00:01 GMT - leap year
    1056964201L,  // 2003-06-30 09:10:01 GMT - last day of month
    0L,           // This is the UNIX epoch - 1970-01-01
    -131536728L   // 1965-10-31 14:01:12 GMT - last day of month
  };

  // add
  auto const months1 = cudf::make_fixed_width_scalar<TypeParam>(11);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    *add_calendrical_months(timestamps_s, *months1),
    fixed_width_column_wrapper<cudf::timestamp_s, cudf::timestamp_s::rep>{
      691545600L,   // 1991-12-01 00:00:00 GMT
      978440401L,   // 2001-01-02 13:00:01 GMT
      1085908201L,  // 2004-05-30 09:10:01 GMT
      28857600L,    // 1970-12-01 00:00:00 GMT
      -102679128L,  // 1966-09-30 14:01:12 GMT
    },
    verbosity);

  // subtract
  auto const months2 = cudf::make_fixed_width_scalar<TypeParam>(-20);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    *add_calendrical_months(timestamps_s, *months2),
    fixed_width_column_wrapper<cudf::timestamp_s, cudf::timestamp_s::rep>{
      609984000L,   // 1989-05-01 00:00:00 GMT
      896792401L,   // 1998-06-02 13:00:01 GMT
      1004433001L,  // 2001-10-30 09:10:01 GMT
      -52704000L,   // 1968-05-01 00:00:00 GMT
      -184240728L,  // 1964-02-29 14:01:12 GMT - lands on a leap year february
    },
    verbosity);
}

TYPED_TEST(TypedAddMonthsTest, TestAddMonthsWithSecondsAndNullValues)
{
  using namespace cudf::test;
  using namespace cudf::datetime;
  using namespace cuda::std::chrono;

  // Time in seconds since epoch
  // Dates converted using epochconverter.com
  auto timestamps_s =
    cudf::test::fixed_width_column_wrapper<cudf::timestamp_s, cudf::timestamp_s::rep>{
      {
        662688000L,   // 1991-01-01 00:00:00 GMT
        949496401L,   // 2000-02-02 13:00:01 GMT - leap year
        1056931201L,  // 2003-06-30 00:00:01 GMT - last day of month
        1056964201L,  // 2003-06-30 09:10:01 GMT - last day of month
        1056974401L,  // 2003-06-30 12:00:01 GMT - last day of month
        1056994021L,  // 2003-06-30 17:27:01 GMT - last day of month
        0L,           // This is the UNIX epoch - 1970-01-01
        0L,           // This is the UNIX epoch - 1970-01-01
        -131586588L,  // 1965-10-31 00:10:12 GMT
        -131550590L,  // 1965-10-31 10:10:10 GMT
        -131544000L,  // 1965-10-31 12:00:00 GMT
        -131536728L   // 1965-10-31 14:01:12 GMT
      },
      {true, false, true, false, true, false, true, false, true, true, true, true}};

  auto const months = cudf::test::fixed_width_column_wrapper<TypeParam>{
    {-2, 6, -1, 1, -4, 8, -2, 10, 4, -20, 1, 3},
    {false, true, true, false, true, true, true, true, true, true, true, true}};

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    *add_calendrical_months(timestamps_s, months),
    cudf::test::fixed_width_column_wrapper<cudf::timestamp_s, cudf::timestamp_s::rep>{
      {
        0L,           // null value
        0L,           // null value
        1054252801L,  // 2003-05-30 00:00:01 GMT
        0L,           // null value
        1046433601L,  // 2003-02-28 12:00:01 GMT
        0L,           // null value
        -5270400L,    // 1969-11-01
        0L,           // null value
        -121218588L,  // 1966-02-28 00:10:12 GMT
        -184254590L,  // 1964-02-29 10:10:10 GMT
        -128952000L,  // 1965-11-30 12:00:00 GMT
        -123587928L   // 1966-01-31 14:01:12 GMT
      },
      {false, false, true, false, true, false, true, false, true, true, true, true}},
    verbosity);
}

TYPED_TEST(TypedAddMonthsTest, TestAddScalarMonthsWithSecondsWithNulls)
{
  using namespace cudf::test;
  using namespace cudf::datetime;
  using namespace cuda::std::chrono;

  // Time in seconds since epoch
  // Dates converted using epochconverter.com
  auto timestamps_s = fixed_width_column_wrapper<cudf::timestamp_s, cudf::timestamp_s::rep>(
    {
      662688000L,   // 1991-01-01 00:00:00 GMT
      0L,           // NULL
      1056964201L,  // 2003-06-30 09:10:01 GMT - last day of month
      0L,           // This is the UNIX epoch - 1970-01-01
      0L            // NULL
    },
    iterators::nulls_at({1, 4}));

  // valid scalar
  auto const months1 = cudf::make_fixed_width_scalar<TypeParam>(11);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    *add_calendrical_months(timestamps_s, *months1),
    fixed_width_column_wrapper<cudf::timestamp_s, cudf::timestamp_s::rep>(
      {
        691545600L,   // 1991-12-01 00:00:00 GMT
        0L,           // NULL
        1085908201L,  // 2004-05-30 09:10:01 GMT
        28857600L,    // 1970-12-01 00:00:00 GMT
        0L,           // NULL
      },
      iterators::nulls_at({1, 4})),
    verbosity);

  // null scalar
  auto const months2 =
    cudf::make_default_constructed_scalar(cudf::data_type{cudf::type_to_id<TypeParam>()});

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    *add_calendrical_months(timestamps_s, *months2),
    fixed_width_column_wrapper<cudf::timestamp_s, cudf::timestamp_s::rep>({0L, 0L, 0L, 0L, 0L},
                                                                          iterators::all_nulls()),
    verbosity);
}

TEST_F(BasicDatetimeOpsTest, TestIsLeapYear)
{
  using namespace cudf::test;
  using namespace cudf::datetime;
  using namespace cuda::std::chrono;

  // Time in seconds since epoch
  // Dates converted using epochconverter.com
  auto timestamps_s =
    cudf::test::fixed_width_column_wrapper<cudf::timestamp_s, cudf::timestamp_s::rep>{
      {
        1594332839L,    // 2020-07-09 10:13:59 GMT - leap year
        0L,             // null
        915148800L,     // 1999-01-01 00:00:00 GMT - non leap year
        -11663029161L,  // 1600-5-31 05:40:39 GMT - leap year
        707904541L,     // 1992-06-07 08:09:01 GMT - leap year
        -2181005247L,   // 1900-11-20 09:12:33 GMT - non leap year
        0L,             // UNIX EPOCH 1970-01-01 00:00:00 GMT - non leap year
        -12212553600L,  // First full year of Gregorian Calendar 1583-01-01 00:00:00 - non-leap-year
        0L,             // null
        13591632822L,   // 2400-09-13 13:33:42 GMT - leap year
        4539564243L,    // 2113-11-08 06:04:03 GMT - non leap year
        0L              // null
      },
      {true, false, true, true, true, true, true, true, false, true, true, false}};

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    *is_leap_year(timestamps_s),
    cudf::test::fixed_width_column_wrapper<bool>{
      {true, XXX, false, true, true, false, false, false, XXX, true, false, XXX},
      {true, false, true, true, true, true, true, true, false, true, true, false}});
}

TEST_F(BasicDatetimeOpsTest, TestDaysInMonths)

{
  using namespace cudf::test;
  using namespace cudf::datetime;
  using namespace cuda::std::chrono;

  auto timestamps_s =
    cudf::test::fixed_width_column_wrapper<cudf::timestamp_s, cudf::timestamp_s::rep>{
      {
        0L,            // NULL
        -1887541682L,  // 1910-03-10 10:51:58
        0L,            // NULL
        -1251006943L,  // 1930-05-11 18:04:17
        -932134638L,   // 1940-06-18 09:42:42
        -614354877L,   // 1950-07-14 09:52:03
        -296070394L,   // 1960-08-14 06:13:26
        22840404L,     // 1970-09-22 08:33:24
        339817190L,    // 1980-10-08 01:39:50
        657928062L,    // 1990-11-06 21:47:42
        976630837L,    // 2000-12-12 14:20:37
        1294699018L,   // 2011-01-10 22:36:58
        1613970182L,   // 2021-02-22 05:03:02 - non leap year February
        1930963331L,   // 2031-03-11 02:42:11
        2249867102L,   // 2041-04-18 03:05:02
        951426858L,    // 2000-02-24 21:14:18 - leap year February
      },
      iterators::nulls_at({0, 2})};

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*days_in_month(timestamps_s),
                                 cudf::test::fixed_width_column_wrapper<int16_t>{
                                   {-1, 31, -1, 31, 30, 31, 31, 30, 31, 30, 31, 31, 28, 31, 30, 29},
                                   iterators::nulls_at({0, 2})});
}

TEST_F(BasicDatetimeOpsTest, TestQuarter)
{
  using namespace cudf::test;
  using namespace cudf::datetime;
  using namespace cuda::std::chrono;
  using namespace cudf::test::iterators;

  // Time in seconds since epoch
  // Dates converted using epochconverter.com
  auto timestamps_s =
    cudf::test::fixed_width_column_wrapper<cudf::timestamp_s, cudf::timestamp_s::rep>{
      {
        1594332839L,    // 2020-07-09 10:13:59 GMT
        0L,             // null
        915148800L,     // 1999-01-01 00:00:00 GMT
        -11663029161L,  // 1600-5-31 05:40:39 GMT
        707904541L,     // 1992-06-07 08:09:01 GMT
        -2181005247L,   // 1900-11-20 09:12:33 GMT
        0L,             // UNIX EPOCH 1970-01-01 00:00:00 GMT
        -12212553600L,  // First full year of Gregorian Calendar 1583-01-01 00:00:00
        0L,             // null
        13591632822L,   // 2400-09-13 13:33:42 GMT
        4539564243L,    // 2113-11-08 06:04:03 GMT
        0L,             // null
        1608581568L,    // 2020-12-21 08:12:48 GMT
        1584821568L,    // 2020-03-21 08:12:48 GMT
      },
      nulls_at({1, 8, 11})};

  auto quarter = cudf::test::fixed_width_column_wrapper<int16_t>{
    {3, 0 /*null*/, 1, 2, 2, 4, 1, 1, 0 /*null*/, 3, 4, 0 /*null*/, 4, 1}, nulls_at({1, 8, 11})};

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*extract_quarter(timestamps_s), quarter);
}

TYPED_TEST(TypedDatetimeOpsTest, TestCeilDatetime)
{
  using T = TypeParam;
  using namespace cudf::test;
  using namespace cudf::datetime;
  using namespace cuda::std::chrono;

  auto start = milliseconds(-2500000000000);  // Sat, 11 Oct 1890 19:33:20 GMT
  auto stop  = milliseconds(2500000000000);   // Mon, 22 Mar 2049 04:26:40 GMT

  auto const input =
    generate_timestamps<T>(this->size(), time_point_ms(start), time_point_ms(stop));
  auto const timestamps = to_host<T>(input).first;

  std::vector<T> ceiled_day(timestamps.size());
  thrust::transform(timestamps.begin(), timestamps.end(), ceiled_day.begin(), [](auto i) {
    return time_point_cast<typename T::duration>(ceil<days>(i));
  });
  auto expected_day =
    fixed_width_column_wrapper<T, typename T::duration::rep>(ceiled_day.begin(), ceiled_day.end());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*ceil_datetimes(input, rounding_frequency::DAY), expected_day);

  std::vector<T> ceiled_hour(timestamps.size());
  thrust::transform(timestamps.begin(), timestamps.end(), ceiled_hour.begin(), [](auto i) {
    return time_point_cast<typename T::duration>(ceil<hours>(i));
  });
  auto expected_hour = fixed_width_column_wrapper<T, typename T::duration::rep>(ceiled_hour.begin(),
                                                                                ceiled_hour.end());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*ceil_datetimes(input, rounding_frequency::HOUR), expected_hour);

  std::vector<T> ceiled_minute(timestamps.size());
  std::transform(timestamps.begin(), timestamps.end(), ceiled_minute.begin(), [](auto i) {
    return time_point_cast<typename T::duration>(ceil<minutes>(i));
  });
  auto expected_minute = fixed_width_column_wrapper<T, typename T::duration::rep>(
    ceiled_minute.begin(), ceiled_minute.end());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*ceil_datetimes(input, rounding_frequency::MINUTE),
                                 expected_minute);

  std::vector<T> ceiled_second(timestamps.size());
  std::transform(timestamps.begin(), timestamps.end(), ceiled_second.begin(), [](auto i) {
    return time_point_cast<typename T::duration>(ceil<seconds>(i));
  });
  auto expected_second = fixed_width_column_wrapper<T, typename T::duration::rep>(
    ceiled_second.begin(), ceiled_second.end());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*ceil_datetimes(input, rounding_frequency::SECOND),
                                 expected_second);

  std::vector<T> ceiled_millisecond(timestamps.size());
  std::transform(timestamps.begin(), timestamps.end(), ceiled_millisecond.begin(), [](auto i) {
    return time_point_cast<typename T::duration>(ceil<milliseconds>(i));
  });
  auto expected_millisecond = fixed_width_column_wrapper<T, typename T::duration::rep>(
    ceiled_millisecond.begin(), ceiled_millisecond.end());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*ceil_datetimes(input, rounding_frequency::MILLISECOND),
                                 expected_millisecond);

  std::vector<T> ceiled_microsecond(timestamps.size());
  std::transform(timestamps.begin(), timestamps.end(), ceiled_microsecond.begin(), [](auto i) {
    return time_point_cast<typename T::duration>(ceil<microseconds>(i));
  });
  auto expected_microsecond = fixed_width_column_wrapper<T, typename T::duration::rep>(
    ceiled_microsecond.begin(), ceiled_microsecond.end());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*ceil_datetimes(input, rounding_frequency::MICROSECOND),
                                 expected_microsecond);

  std::vector<T> ceiled_nanosecond(timestamps.size());
  std::transform(timestamps.begin(), timestamps.end(), ceiled_nanosecond.begin(), [](auto i) {
    return time_point_cast<typename T::duration>(ceil<nanoseconds>(i));
  });
  auto expected_nanosecond = fixed_width_column_wrapper<T, typename T::duration::rep>(
    ceiled_nanosecond.begin(), ceiled_nanosecond.end());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*ceil_datetimes(input, rounding_frequency::NANOSECOND),
                                 expected_nanosecond);
}

TYPED_TEST(TypedDatetimeOpsTest, TestFloorDatetime)
{
  using T = TypeParam;
  using namespace cudf::test;
  using namespace cudf::datetime;
  using namespace cuda::std::chrono;

  auto start = milliseconds(-2500000000000);  // Sat, 11 Oct 1890 19:33:20 GMT
  auto stop  = milliseconds(2500000000000);   // Mon, 22 Mar 2049 04:26:40 GMT

  auto const input =
    generate_timestamps<T>(this->size(), time_point_ms(start), time_point_ms(stop));
  auto const timestamps = to_host<T>(input).first;

  std::vector<T> floored_day(timestamps.size());
  std::transform(timestamps.begin(), timestamps.end(), floored_day.begin(), [](auto i) {
    return time_point_cast<typename T::duration>(floor<days>(i));
  });
  auto expected_day = fixed_width_column_wrapper<T, typename T::duration::rep>(floored_day.begin(),
                                                                               floored_day.end());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*floor_datetimes(input, rounding_frequency::DAY), expected_day);

  std::vector<T> floored_hour(timestamps.size());
  std::transform(timestamps.begin(), timestamps.end(), floored_hour.begin(), [](auto i) {
    return time_point_cast<typename T::duration>(floor<hours>(i));
  });
  auto expected_hour = fixed_width_column_wrapper<T, typename T::duration::rep>(
    floored_hour.begin(), floored_hour.end());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*floor_datetimes(input, rounding_frequency::HOUR), expected_hour);

  std::vector<T> floored_minute(timestamps.size());
  std::transform(timestamps.begin(), timestamps.end(), floored_minute.begin(), [](auto i) {
    return time_point_cast<typename T::duration>(floor<minutes>(i));
  });
  auto expected_minute = fixed_width_column_wrapper<T, typename T::duration::rep>(
    floored_minute.begin(), floored_minute.end());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*floor_datetimes(input, rounding_frequency::MINUTE),
                                 expected_minute);

  std::vector<T> floored_second(timestamps.size());
  std::transform(timestamps.begin(), timestamps.end(), floored_second.begin(), [](auto i) {
    return time_point_cast<typename T::duration>(floor<seconds>(i));
  });
  auto expected_second = fixed_width_column_wrapper<T, typename T::duration::rep>(
    floored_second.begin(), floored_second.end());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*floor_datetimes(input, rounding_frequency::SECOND),
                                 expected_second);

  std::vector<T> floored_millisecond(timestamps.size());
  std::transform(timestamps.begin(), timestamps.end(), floored_millisecond.begin(), [](auto i) {
    return time_point_cast<typename T::duration>(floor<milliseconds>(i));
  });
  auto expected_millisecond = fixed_width_column_wrapper<T, typename T::duration::rep>(
    floored_millisecond.begin(), floored_millisecond.end());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*floor_datetimes(input, rounding_frequency::MILLISECOND),
                                 expected_millisecond);

  std::vector<T> floored_microsecond(timestamps.size());
  std::transform(timestamps.begin(), timestamps.end(), floored_microsecond.begin(), [](auto i) {
    return time_point_cast<typename T::duration>(floor<microseconds>(i));
  });
  auto expected_microsecond = fixed_width_column_wrapper<T, typename T::duration::rep>(
    floored_microsecond.begin(), floored_microsecond.end());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*floor_datetimes(input, rounding_frequency::MICROSECOND),
                                 expected_microsecond);

  std::vector<T> floored_nanosecond(timestamps.size());
  std::transform(timestamps.begin(), timestamps.end(), floored_nanosecond.begin(), [](auto i) {
    return time_point_cast<typename T::duration>(floor<nanoseconds>(i));
  });
  auto expected_nanosecond = fixed_width_column_wrapper<T, typename T::duration::rep>(
    floored_nanosecond.begin(), floored_nanosecond.end());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*floor_datetimes(input, rounding_frequency::NANOSECOND),
                                 expected_nanosecond);
}

TYPED_TEST(TypedDatetimeOpsTest, TestRoundDatetime)
{
  using T = TypeParam;
  using namespace cudf::test;
  using namespace cudf::datetime;
  using namespace cuda::std::chrono;

  auto start = milliseconds(-2500000000000);  // Sat, 11 Oct 1890 19:33:20 GMT
  auto stop  = milliseconds(2500000000000);   // Mon, 22 Mar 2049 04:26:40 GMT

  auto const input =
    generate_timestamps<T>(this->size(), time_point_ms(start), time_point_ms(stop));
  auto const timestamps = to_host<T>(input).first;

  std::vector<T> rounded_day(timestamps.size());
  std::transform(timestamps.begin(), timestamps.end(), rounded_day.begin(), [](auto i) {
    return time_point_cast<typename T::duration>(round<days>(i));
  });
  auto expected_day = fixed_width_column_wrapper<T, typename T::duration::rep>(rounded_day.begin(),
                                                                               rounded_day.end());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*round_datetimes(input, rounding_frequency::DAY), expected_day);

  std::vector<T> rounded_hour(timestamps.size());
  std::transform(timestamps.begin(), timestamps.end(), rounded_hour.begin(), [](auto i) {
    return time_point_cast<typename T::duration>(round<hours>(i));
  });
  auto expected_hour = fixed_width_column_wrapper<T, typename T::duration::rep>(
    rounded_hour.begin(), rounded_hour.end());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*round_datetimes(input, rounding_frequency::HOUR), expected_hour);

  std::vector<T> rounded_minute(timestamps.size());
  std::transform(timestamps.begin(), timestamps.end(), rounded_minute.begin(), [](auto i) {
    return time_point_cast<typename T::duration>(round<minutes>(i));
  });
  auto expected_minute = fixed_width_column_wrapper<T, typename T::duration::rep>(
    rounded_minute.begin(), rounded_minute.end());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*round_datetimes(input, rounding_frequency::MINUTE),
                                 expected_minute);

  std::vector<T> rounded_second(timestamps.size());
  std::transform(timestamps.begin(), timestamps.end(), rounded_second.begin(), [](auto i) {
    return time_point_cast<typename T::duration>(round<seconds>(i));
  });
  auto expected_second = fixed_width_column_wrapper<T, typename T::duration::rep>(
    rounded_second.begin(), rounded_second.end());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*round_datetimes(input, rounding_frequency::SECOND),
                                 expected_second);

  std::vector<T> rounded_millisecond(timestamps.size());
  std::transform(timestamps.begin(), timestamps.end(), rounded_millisecond.begin(), [](auto i) {
    return time_point_cast<typename T::duration>(round<milliseconds>(i));
  });
  auto expected_millisecond = fixed_width_column_wrapper<T, typename T::duration::rep>(
    rounded_millisecond.begin(), rounded_millisecond.end());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*round_datetimes(input, rounding_frequency::MILLISECOND),
                                 expected_millisecond);

  std::vector<T> rounded_microsecond(timestamps.size());
  std::transform(timestamps.begin(), timestamps.end(), rounded_microsecond.begin(), [](auto i) {
    return time_point_cast<typename T::duration>(round<microseconds>(i));
  });
  auto expected_microsecond = fixed_width_column_wrapper<T, typename T::duration::rep>(
    rounded_microsecond.begin(), rounded_microsecond.end());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*round_datetimes(input, rounding_frequency::MICROSECOND),
                                 expected_microsecond);

  std::vector<T> rounded_nanosecond(timestamps.size());
  std::transform(timestamps.begin(), timestamps.end(), rounded_nanosecond.begin(), [](auto i) {
    return time_point_cast<typename T::duration>(round<nanoseconds>(i));
  });
  auto expected_nanosecond = fixed_width_column_wrapper<T, typename T::duration::rep>(
    rounded_nanosecond.begin(), rounded_nanosecond.end());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*round_datetimes(input, rounding_frequency::NANOSECOND),
                                 expected_nanosecond);
}

CUDF_TEST_PROGRAM_MAIN()
