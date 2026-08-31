/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>

#include <cudf/strings/convert/convert_datetime.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/unary.hpp>
#include <cudf/wrappers/durations.hpp>
#include <cudf/wrappers/timestamps.hpp>

#include <cuda/iterator>

#include <vector>

struct StringsDatetimeTest : public cudf::test::BaseFixture {};

TEST_F(StringsDatetimeTest, ToTimestamp)
{
  std::vector<char const*> h_strings{"1974-02-28T01:23:45Z",
                                     "2019-07-17T21:34:37Z",
                                     nullptr,
                                     "",
                                     "2019-03-20T12:34:56Z",
                                     "2020-02-29T00:00:00Z",
                                     "1921-01-07T14:32:07Z",
                                     "1969-12-31T23:59:45Z"};
  cudf::test::strings_column_wrapper strings(
    h_strings.begin(), h_strings.end(), cuda::transform_iterator(h_strings.begin(), [](auto str) {
      return str != nullptr;
    }));
  std::vector<cudf::timestamp_s::rep> h_expected{
    131246625, 1563399277, 0, 0, 1553085296, 1582934400, -1545730073, -15};

  auto strings_view = cudf::strings_column_view(strings);
  auto results      = cudf::strings::to_timestamps(
    strings_view, cudf::data_type{cudf::type_id::TIMESTAMP_SECONDS}, "%Y-%m-%dT%H:%M:%SZ");

  cudf::test::fixed_width_column_wrapper<cudf::timestamp_s, cudf::timestamp_s::rep> expected(
    h_expected.begin(), h_expected.end(), cuda::transform_iterator(h_strings.begin(), [](auto str) {
      return str != nullptr;
    }));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);

  results = cudf::strings::is_timestamp(strings_view, "%Y-%m-%dT%H:%M:%SZ");
  cudf::test::fixed_width_column_wrapper<bool> is_expected(
    {1, 1, 0, 0, 1, 1, 1, 1}, {true, true, false, true, true, true, true, true});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, is_expected);
}

TEST_F(StringsDatetimeTest, ToTimestampAmPm)
{
  cudf::test::strings_column_wrapper strings{"1974-02-28 01:23:45 PM",
                                             "2019-07-17 02:34:56 AM",
                                             "2019-03-20 12:34:56 PM",
                                             "2020-02-29 12:00:00 AM",
                                             "1925-02-07 02:55:08 PM"};
  auto strings_view = cudf::strings_column_view(strings);
  auto results      = cudf::strings::to_timestamps(
    strings_view, cudf::data_type{cudf::type_id::TIMESTAMP_SECONDS}, "%Y-%m-%d %I:%M:%S %p");
  cudf::test::fixed_width_column_wrapper<cudf::timestamp_s, cudf::timestamp_s::rep> expected{
    131289825, 1563330896, 1553085296, 1582934400, -1416819892};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);

  results = cudf::strings::is_timestamp(strings_view, "%Y-%m-%d %I:%M:%S %p");
  cudf::test::fixed_width_column_wrapper<bool> is_expected({1, 1, 1, 1, 1});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, is_expected);
}

TEST_F(StringsDatetimeTest, ToTimestampMicrosecond)
{
  cudf::test::strings_column_wrapper strings{"1974-02-28 01:23:45.987000",
                                             "2019-07-17 02:34:56.001234",
                                             "2019-03-20 12:34:56.100100",
                                             "2020-02-29 00:00:00.555777",
                                             "1969-12-31 00:00:01.000055",
                                             "1944-07-21 11:15:09.333444"};
  auto strings_view = cudf::strings_column_view(strings);
  auto results      = cudf::strings::to_timestamps(
    strings_view, cudf::data_type{cudf::type_id::TIMESTAMP_MILLISECONDS}, "%Y-%m-%d %H:%M:%S.%6f");
  cudf::test::fixed_width_column_wrapper<cudf::timestamp_ms, cudf::timestamp_ms::rep> expected_ms{
    131246625987, 1563330896001, 1553085296100, 1582934400555, -86399000L, -803047490667L};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected_ms);
  results = cudf::strings::to_timestamps(
    strings_view, cudf::data_type{cudf::type_id::TIMESTAMP_NANOSECONDS}, "%Y-%m-%d %H:%M:%S.%6f");
  cudf::test::fixed_width_column_wrapper<cudf::timestamp_ns, cudf::timestamp_ns::rep> expected_ns{
    131246625987000000,
    1563330896001234000,
    1553085296100100000,
    1582934400555777000,
    -86398999945000,
    -803047490666556000};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected_ns);

  results = cudf::strings::is_timestamp(strings_view, "%Y-%m-%d %H:%M:%S.%6f");
  cudf::test::fixed_width_column_wrapper<bool> is_expected({1, 1, 1, 1, 1, 1});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, is_expected);
}

TEST_F(StringsDatetimeTest, ToTimestampMillisecond)
{
  cudf::test::strings_column_wrapper strings{"2018-07-04 12:00:00.123",
                                             "2020-04-06 13:09:00.555",
                                             "1969-12-31 00:00:00.000",
                                             "1956-01-23 17:18:19.000"};
  auto strings_view = cudf::strings_column_view(strings);
  auto results      = cudf::strings::to_timestamps(
    strings_view, cudf::data_type{cudf::type_id::TIMESTAMP_MICROSECONDS}, "%Y-%m-%d %H:%M:%S.%3f");
  cudf::test::fixed_width_column_wrapper<cudf::timestamp_us, cudf::timestamp_us::rep> expected_us{
    1530705600123000, 1586178540555000, -86400000000, -439886501000000};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected_us);
  results = cudf::strings::to_timestamps(
    strings_view, cudf::data_type{cudf::type_id::TIMESTAMP_NANOSECONDS}, "%Y-%m-%d %H:%M:%S.%3f");
  cudf::test::fixed_width_column_wrapper<cudf::timestamp_ns, cudf::timestamp_ns::rep> expected_ns{
    1530705600123000000, 1586178540555000000, -86400000000000, -439886501000000000};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected_ns);

  results = cudf::strings::is_timestamp(strings_view, "%Y-%m-%d %H:%M:%S.%3f");
  cudf::test::fixed_width_column_wrapper<bool> is_expected({1, 1, 1, 1});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, is_expected);
}

TEST_F(StringsDatetimeTest, ToTimestampTimezone)
{
  cudf::test::strings_column_wrapper strings{"1974-02-28 01:23:45+0100",
                                             "2019-07-17 02:34:56-0300",
                                             "2019-03-20 12:34:56+1030",
                                             "2020-02-29 12:00:00-0500",
                                             "2022-04-07 09:15:00Z",
                                             "1938-11-23 10:28:49+0700"};
  auto strings_view = cudf::strings_column_view(strings);
  auto results      = cudf::strings::to_timestamps(
    strings_view, cudf::data_type{cudf::type_id::TIMESTAMP_SECONDS}, "%Y-%m-%d %H:%M:%S%z");
  cudf::test::fixed_width_column_wrapper<cudf::timestamp_s, cudf::timestamp_s::rep> expected{
    131243025, 1563341696, 1553047496, 1582995600, 1649322900, -981664271};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);

  results = cudf::strings::is_timestamp(strings_view, "%Y-%m-%d %H:%M:%S%z");
  cudf::test::fixed_width_column_wrapper<bool> is_expected({1, 1, 1, 1, 1, 1});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, is_expected);
}

TEST_F(StringsDatetimeTest, ToTimestampSingleSpecifier)
{
  cudf::test::strings_column_wrapper strings{"12", "10", "09", "05"};
  auto strings_view = cudf::strings_column_view(strings);
  auto results      = cudf::strings::to_timestamps(
    strings_view, cudf::data_type{cudf::type_id::TIMESTAMP_DAYS}, "%d");
  cudf::test::fixed_width_column_wrapper<cudf::timestamp_D, cudf::timestamp_D::rep> expected_days{
    11, 9, 8, 4};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected_days);

  results = cudf::strings::to_timestamps(
    strings_view, cudf::data_type{cudf::type_id::TIMESTAMP_DAYS}, "%m");
  cudf::test::fixed_width_column_wrapper<cudf::timestamp_D, cudf::timestamp_D::rep> expected_months{
    334, 273, 243, 120};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected_months);

  results = cudf::strings::is_timestamp(strings_view, "%m");
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results,
                                 cudf::test::fixed_width_column_wrapper<bool>{1, 1, 1, 1});
}

TEST_F(StringsDatetimeTest, ToTimestampVariableFractions)
{
  cudf::test::strings_column_wrapper test1{"01:02:03.000001000",
                                           "01:02:03.000001",
                                           "01:02:03.1",
                                           "01:02:03.01",
                                           "01:02:03.0098700",
                                           "01:02:03.0023456"};
  auto strings_view = cudf::strings_column_view(test1);
  auto results      = cudf::strings::to_timestamps(
    strings_view, cudf::data_type{cudf::type_id::TIMESTAMP_NANOSECONDS}, "%H:%M:%S.%9f");
  auto durations =
    cudf::cast(results->view(), cudf::data_type{cudf::type_id::DURATION_NANOSECONDS});

  cudf::test::fixed_width_column_wrapper<cudf::duration_ns> expected1{
    cudf::duration_ns{3723000001000},
    cudf::duration_ns{3723000001000},
    cudf::duration_ns{3723100000000},
    cudf::duration_ns{3723010000000},
    cudf::duration_ns{3723009870000},
    cudf::duration_ns{3723002345600}};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*durations, expected1);

  results = cudf::strings::is_timestamp(strings_view, "%H:%M:%S.%f");
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results,
                                 cudf::test::fixed_width_column_wrapper<bool>{1, 1, 1, 1, 1, 1});

  cudf::test::strings_column_wrapper test2{"01:02:03.100001Z",
                                           "01:02:03.001Z",
                                           "01:02:03.1Z",
                                           "01:02:03.01Z",
                                           "01:02:03.0098Z",
                                           "01:02:03.00234Z"};
  strings_view = cudf::strings_column_view(test2);
  results      = cudf::strings::to_timestamps(
    strings_view, cudf::data_type{cudf::type_id::TIMESTAMP_MICROSECONDS}, "%H:%M:%S.%6f%Z");
  durations = cudf::cast(results->view(), cudf::data_type{cudf::type_id::DURATION_MICROSECONDS});

  cudf::test::fixed_width_column_wrapper<cudf::duration_us> expected2{
    cudf::duration_us{3723100001},
    cudf::duration_us{3723001000},
    cudf::duration_us{3723100000},
    cudf::duration_us{3723010000},
    cudf::duration_us{3723009800},
    cudf::duration_us{3723002340}};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*durations, expected2);
}

TEST_F(StringsDatetimeTest, ToTimestampYear)
{
  cudf::test::strings_column_wrapper strings{
    "28/02/74", "17/07/68", "20/03/19", "29/02/20", "07/02/69"};
  auto strings_view = cudf::strings_column_view(strings);
  auto results      = cudf::strings::to_timestamps(
    strings_view, cudf::data_type{cudf::type_id::TIMESTAMP_DAYS}, "%d/%m/%y");
  cudf::test::fixed_width_column_wrapper<cudf::timestamp_D, cudf::timestamp_D::rep> expected{
    1519, 35992, 17975, 18321, -328};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);

  results = cudf::strings::is_timestamp(strings_view, "%d/%m/%y");
  cudf::test::fixed_width_column_wrapper<bool> is_expected({1, 1, 1, 1, 1});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, is_expected);
}

TEST_F(StringsDatetimeTest, ToTimestampWeeks)
{
  cudf::test::strings_column_wrapper input{
    "2012-01/3", "2012-04/4", "2018-01/3", "2012-52/5", "2020-44/2", "1960-20/0", "1986-04/6"};

  auto format  = std::string("%Y-%W/%w");
  auto results = cudf::strings::to_timestamps(
    cudf::strings_column_view(input), cudf::data_type{cudf::type_id::TIMESTAMP_DAYS}, format);
  auto expected = cudf::test::fixed_width_column_wrapper<cudf::timestamp_D, cudf::timestamp_D::rep>{
    15343, 15365, 17534, 15702, 18569, -3511, 5875};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);

  results          = cudf::strings::is_timestamp(cudf::strings_column_view(input), format);
  auto is_expected = cudf::test::fixed_width_column_wrapper<bool>({1, 1, 1, 1, 1, 1, 1});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, is_expected);

  cudf::test::strings_column_wrapper input_iso{
    "2012-01/3", "2012-04/4", "2018-01/3", "2012-52/5", "2020-44/2", "1960-20/7", "1986-04/6"};

  format  = std::string("%Y-%U/%u");
  results = cudf::strings::to_timestamps(
    cudf::strings_column_view(input_iso), cudf::data_type{cudf::type_id::TIMESTAMP_DAYS}, format);
  expected = cudf::test::fixed_width_column_wrapper<cudf::timestamp_D, cudf::timestamp_D::rep>{
    15343, 15365, 17541, 15702, 18569, -3518, 5875};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);

  results = cudf::strings::is_timestamp(cudf::strings_column_view(input_iso), format);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, is_expected);

  is_expected = cudf::test::fixed_width_column_wrapper<bool>({1, 1, 1, 1, 1, 0, 1});
  results     = cudf::strings::is_timestamp(cudf::strings_column_view(input), format);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, is_expected);
}

TEST_F(StringsDatetimeTest, ToTimestampSingleDigits)
{
  cudf::test::strings_column_wrapper strings{"1974-2-28 01:23:45.987000123",
                                             "2019-7-17 2:34:56.001234567",
                                             "2019-3-20 12:34:56.100100100",
                                             "2020-02-2 00:00:00.555777999",
                                             "1969-12-1 00:00:01.000055000",
                                             "1944-07-21 11:15:09.333444000",
                                             "2021-9-8 12:07:30.000000000"};
  auto strings_view = cudf::strings_column_view(strings);

  auto results = cudf::strings::to_timestamps(
    strings_view, cudf::data_type{cudf::type_id::TIMESTAMP_NANOSECONDS}, "%Y-%m-%d %H:%M:%S.%9f");
  cudf::test::fixed_width_column_wrapper<cudf::timestamp_ns, cudf::timestamp_ns::rep> expected_ns{
    131246625987000123,
    1563330896001234567,
    1553085296100100100,
    1580601600555777999,
    -2678398999945000,
    -803047490666556000,
    1631102850000000000};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected_ns);

  results = cudf::strings::is_timestamp(strings_view, "%Y-%m-%d %H:%M:%S.%6f");
  cudf::test::fixed_width_column_wrapper<bool> is_expected({1, 1, 1, 1, 1, 1, 1});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, is_expected);
}

TEST_F(StringsDatetimeTest, IsTimestamp)
{
  auto const input   = cudf::test::strings_column_wrapper{"2020-10-07 13:02:03 1PM +0130",
                                                        "2020:10:07 01-02-03 1AM +0130",
                                                        "2020-10-7 11:02:03 11AM -1025",
                                                        "2020-13-07 01:02:03 1AM +0000",
                                                        "2020-10-32 01:32:03 1AM +0000",
                                                        "2020-10-07 25:02:03 1AM +0000",
                                                        "2020-10-07 01:62:03 1AM +0000",
                                                        "2020-10-07 01:02:63 1AM +0000",
                                                        "2020-02-29 01:32:03 1AM +0000",
                                                        "2020-02-30 01:32:03 01AM +0000",
                                                        "2020-00-31 01:32:03 1AM +0000",
                                                        "2020-02-00 02:32:03 2AM +0000",
                                                        "2022-08-24 02:32:60 2AM +0000",
                                                        "2020-2-9 9:12:13 9AM +1111",
                                                        "9999-08-24 02:32:06 2AM +0000",
                                                        "20220-08-24 02:32:06 2AM +0000"};
  auto const sv      = cudf::strings_column_view(input);
  auto const results = cudf::strings::is_timestamp(sv, "%Y-%m-%d %H:%M:%S %I%p %z");
  auto const expected =
    cudf::test::fixed_width_column_wrapper<bool>{1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
}

TEST_F(StringsDatetimeTest, FromTimestamp)
{
  std::vector<cudf::timestamp_s::rep> h_timestamps{
    131246625, 1563399277, 0, 1553085296, 1582934400, -1545730073, -86399};
  std::vector<char const*> h_expected{"1974-02-28T01:23:45Z",
                                      "2019-07-17T21:34:37Z",
                                      nullptr,
                                      "2019-03-20T12:34:56Z",
                                      "2020-02-29T00:00:00Z",
                                      "1921-01-07T14:32:07Z",
                                      "1969-12-31T00:00:01Z"};

  cudf::test::fixed_width_column_wrapper<cudf::timestamp_s, cudf::timestamp_s::rep> timestamps(
    h_timestamps.begin(),
    h_timestamps.end(),
    cuda::transform_iterator(h_expected.begin(), [](auto str) { return str != nullptr; }));

  auto results = cudf::strings::from_timestamps(timestamps);

  cudf::test::strings_column_wrapper expected(
    h_expected.begin(),
    h_expected.end(),
    cuda::transform_iterator(h_expected.begin(), [](auto str) { return str != nullptr; }));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
}

TEST_F(StringsDatetimeTest, FromTimestampAmPm)
{
  cudf::test::fixed_width_column_wrapper<cudf::timestamp_s, cudf::timestamp_s::rep> timestamps{
    1530705600L, 1582934461L, 1451430122L, 1318302183L, -6105994200L};
  auto results = cudf::strings::from_timestamps(timestamps, "%Y-%m-%d %I:%M:%S %p");
  cudf::test::strings_column_wrapper expected{"2018-07-04 12:00:00 PM",
                                              "2020-02-29 12:01:01 AM",
                                              "2015-12-29 11:02:02 PM",
                                              "2011-10-11 03:03:03 AM",
                                              "1776-07-04 06:30:00 PM"};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
}

TEST_F(StringsDatetimeTest, FromTimestampMillisecond)
{
  cudf::test::fixed_width_column_wrapper<cudf::timestamp_ms, cudf::timestamp_ms::rep> timestamps_ms{
    1530705600123, 1582934461007, 1451430122421, 1318302183999, -6106017600047, 128849018880000};
  auto results = cudf::strings::from_timestamps(timestamps_ms, "%Y-%m-%d %H:%M:%S.%3f");
  cudf::test::strings_column_wrapper expected_ms{"2018-07-04 12:00:00.123",
                                                 "2020-02-29 00:01:01.007",
                                                 "2015-12-29 23:02:02.421",
                                                 "2011-10-11 03:03:03.999",
                                                 "1776-07-04 11:59:59.953",
                                                 "6053-01-23 02:08:00.000"};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected_ms);

  results = cudf::strings::from_timestamps(timestamps_ms, "%Y-%m-%d %H:%M:%S.%f");
  cudf::test::strings_column_wrapper expected_ms_6f{"2018-07-04 12:00:00.123000",
                                                    "2020-02-29 00:01:01.007000",
                                                    "2015-12-29 23:02:02.421000",
                                                    "2011-10-11 03:03:03.999000",
                                                    "1776-07-04 11:59:59.953000",
                                                    "6053-01-23 02:08:00.000000"};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected_ms_6f);

  cudf::test::fixed_width_column_wrapper<cudf::timestamp_ns, cudf::timestamp_ns::rep> timestamps_ns{
    1530705600123456789,
    1582934461007008009,
    1451430122421310209,
    1318302183999777555,
    -6106017600047047047};
  results = cudf::strings::from_timestamps(timestamps_ns, "%Y-%m-%d %H:%M:%S.%9f");
  cudf::test::strings_column_wrapper expected_ns{"2018-07-04 12:00:00.123456789",
                                                 "2020-02-29 00:01:01.007008009",
                                                 "2015-12-29 23:02:02.421310209",
                                                 "2011-10-11 03:03:03.999777555",
                                                 "1776-07-04 11:59:59.952952953"};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected_ns);

  results = cudf::strings::from_timestamps(timestamps_ns, "%Y-%m-%d %H:%M:%S.%f");
  cudf::test::strings_column_wrapper expected_ns_6f{"2018-07-04 12:00:00.123456",
                                                    "2020-02-29 00:01:01.007008",
                                                    "2015-12-29 23:02:02.421310",
                                                    "2011-10-11 03:03:03.999777",
                                                    "1776-07-04 11:59:59.952952"};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected_ns_6f);
}

TEST_F(StringsDatetimeTest, FromTimestampTimezone)
{
  cudf::test::fixed_width_column_wrapper<cudf::timestamp_s, cudf::timestamp_s::rep> timestamps{
    1530705600L, 1582934461L, 1451430122L, 1318302183L, -2658802500L};
  auto results = cudf::strings::from_timestamps(timestamps, "%m/%d/%y %H%M%S%z");
  cudf::test::strings_column_wrapper expected{"07/04/18 120000+0000",
                                              "02/29/20 000101+0000",
                                              "12/29/15 230202+0000",
                                              "10/11/11 030303+0000",
                                              "09/29/85 194500+0000"};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
}

TEST_F(StringsDatetimeTest, FromTimestampDayOfYear)
{
  cudf::test::fixed_width_column_wrapper<cudf::timestamp_s, cudf::timestamp_s::rep> timestamps{
    118800L,       // 1970-01-02 09:00:00
    1293901860L,   // 2011-01-01 17:11:00
    318402000L,    // 1980-02-03 05:00:00
    604996200L,    // 1989-03-04 06:30:00
    1270413572L,   // 2010-04-04 20:39:32
    1588734621L,   // 2020-05-06 03:10:21
    2550814152L,   // 2050-10-31 07:29:12
    4102518778L,   // 2100-01-01 20:32:58
    702696234L,    // 1992-04-08 01:23:54
    6516816203L,   // 2176-07-05 02:43:23
    26472091292L,  // 2808-11-12 22:41:32
    4133857172L,   // 2100-12-30 01:39:32
    1560948892L,   // 2019-06-19 12:54:52
    4115217600L,   // 2100-05-28 20:00:00
    -265880250L,   // 1961-07-29 16:22:30
  };
  auto results = cudf::strings::from_timestamps(timestamps, "%d/%m/%Y %j");
  cudf::test::strings_column_wrapper expected{"02/01/1970 002",
                                              "01/01/2011 001",
                                              "03/02/1980 034",
                                              "04/03/1989 063",
                                              "04/04/2010 094",
                                              "06/05/2020 127",
                                              "31/10/2050 304",
                                              "01/01/2100 001",
                                              "08/04/1992 099",
                                              "05/07/2176 187",
                                              "12/11/2808 317",
                                              "30/12/2100 364",
                                              "19/06/2019 170",
                                              "28/05/2100 148",
                                              "29/07/1961 210"};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
}

// Format names used for some specifiers in from_timestamps
// clang-format off
cudf::test::strings_column_wrapper format_names() {
  return cudf::test::strings_column_wrapper({"AM", "PM",
    "Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday",
    "Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat",
    "January", "February", "March", "April", "May", "June", "July",
    "August", "September", "October", "November", "December",
    "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"});
}
// clang-format on

TEST_F(StringsDatetimeTest, FromTimestampDayOfWeekOfYear)
{
  cudf::test::fixed_width_column_wrapper<cudf::timestamp_s, cudf::timestamp_s::rep> timestamps{
    1645059720L,  // 2022-02-17
    1647167880L,  // 2022-03-13
    1649276040L,  // 2022-04-06
    1588734621L,  // 2020-05-06
    1560948892L,  // 2019-06-19
    -265880250L,  // 1961-07-29
    1628194442L,  // 2021-08-05
    1632410760L,  // 2021-09-23
    1633464842L,  // 2021-10-05
    1636100042L,  // 2021-11-05
    // These are a sequence of dates which are particular to the ISO week and
    // year numbers which shift through Monday and Thursday and nicely includes
    // a leap year (1980). https://en.wikipedia.org/wiki/ISO_week_date
    220924800L,  // 1977-01-01
    221011200L,  // 1977-01-02
    252374400L,  // 1977-12-31
    252460800L,  // 1978-01-01
    252547200L,  // 1978-01-02
    283910400L,  // 1978-12-31
    283996800L,  // 1979-01-01
    315360000L,  // 1979-12-30
    315446400L,  // 1979-12-31
    315532800L,  // 1980-01-01
    346809600L,  // 1980-12-28
    346896000L,  // 1980-12-29
    346982400L,  // 1980-12-30
    347068800L,  // 1980-12-31
    347155200L,  // 1981-01-01
    378604800L,  // 1981-12-31
    378691200L,  // 1982-01-01
    378777600L,  // 1982-01-02
    378864000L   // 1982-01-03
  };

  cudf::test::strings_column_wrapper expected(
    {"[Thu 17, Feb 2022  4  07  4  07  2022  07]", "[Sun 13, Mar 2022  0  10  7  11  2022  10]",
     "[Wed 06, Apr 2022  3  14  3  14  2022  14]", "[Wed 06, May 2020  3  18  3  18  2020  19]",
     "[Wed 19, Jun 2019  3  24  3  24  2019  25]", "[Sat 29, Jul 1961  6  30  6  30  1961  30]",
     "[Thu 05, Aug 2021  4  31  4  31  2021  31]", "[Thu 23, Sep 2021  4  38  4  38  2021  38]",
     "[Tue 05, Oct 2021  2  40  2  40  2021  40]", "[Fri 05, Nov 2021  5  44  5  44  2021  44]",
     "[Sat 01, Jan 1977  6  00  6  00  1976  53]", "[Sun 02, Jan 1977  0  00  7  01  1976  53]",
     "[Sat 31, Dec 1977  6  52  6  52  1977  52]", "[Sun 01, Jan 1978  0  00  7  01  1977  52]",
     "[Mon 02, Jan 1978  1  01  1  01  1978  01]", "[Sun 31, Dec 1978  0  52  7  53  1978  52]",
     "[Mon 01, Jan 1979  1  01  1  00  1979  01]", "[Sun 30, Dec 1979  0  52  7  52  1979  52]",
     "[Mon 31, Dec 1979  1  53  1  52  1980  01]", "[Tue 01, Jan 1980  2  00  2  00  1980  01]",
     "[Sun 28, Dec 1980  0  51  7  52  1980  52]", "[Mon 29, Dec 1980  1  52  1  52  1981  01]",
     "[Tue 30, Dec 1980  2  52  2  52  1981  01]", "[Wed 31, Dec 1980  3  52  3  52  1981  01]",
     "[Thu 01, Jan 1981  4  00  4  00  1981  01]", "[Thu 31, Dec 1981  4  52  4  52  1981  53]",
     "[Fri 01, Jan 1982  5  00  5  00  1981  53]", "[Sat 02, Jan 1982  6  00  6  00  1981  53]",
     "[Sun 03, Jan 1982  0  00  7  01  1981  53]"});

  auto results = cudf::strings::from_timestamps(timestamps,
                                                "[%a %d, %b %Y  %w  %W  %u  %U  %G  %V]",
                                                cudf::strings_column_view(format_names()));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
}

TEST_F(StringsDatetimeTest, FromTimestampWeekdayMonthYear)
{
  cudf::test::fixed_width_column_wrapper<cudf::timestamp_s, cudf::timestamp_s::rep> timestamps{
    1642951560L,  // 2022-01-23 15:26:00 Sunday
    1645059720L,  // 2022-02-17 01:02:00 Thursday
    1647167880L,  // 2022-03-13 10:38:00 Sunday
    1649276040L,  // 2022-04-06 20:14:00 Wednesday
    1588734621L,  // 2020-05-06 03:10:21 Wednesday
    1560948892L,  // 2019-06-19 12:54:52 Wednesday
    -265880250L,  // 1961-07-29 16:22:30 Saturday
    1628194442L,  // 2021-08-05 20:14:02 Thursday
    1632410760L,  // 2021-09-23 15:26:00 Thursday
    1633464842L,  // 2021-10-05 20:14:02 Tuesday
    1636100042L,  // 2021-11-05 08:14:02 Friday
    1638757202L   // 2021-12-06 02:20:00 Monday
  };

  cudf::test::strings_column_wrapper expected({"[Sunday January 23, 2022: 03 PM]",
                                               "[Thursday February 17, 2022: 01 AM]",
                                               "[Sunday March 13, 2022: 10 AM]",
                                               "[Wednesday April 06, 2022: 08 PM]",
                                               "[Wednesday May 06, 2020: 03 AM]",
                                               "[Wednesday June 19, 2019: 12 PM]",
                                               "[Saturday July 29, 1961: 04 PM]",
                                               "[Thursday August 05, 2021: 08 PM]",
                                               "[Thursday September 23, 2021: 03 PM]",
                                               "[Tuesday October 05, 2021: 08 PM]",
                                               "[Friday November 05, 2021: 08 AM]",
                                               "[Monday December 06, 2021: 02 AM]"});

  auto results = cudf::strings::from_timestamps(
    timestamps, "[%A %B %d, %Y: %I %p]", cudf::strings_column_view(format_names()));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
}

TEST_F(StringsDatetimeTest, FromTimestampAllSpecifiers)
{
  cudf::test::fixed_width_column_wrapper<cudf::timestamp_ns, cudf::timestamp_ns::rep> input{
    1645059720000000001L,
    1647167880000001000L,
    1649276040001000000L,
    1588734621123456789L,
    1560948892987654321L,
    -265880250010203040L,
    1628194442090807060L,
    1632410760500400300L,
    1633464842000000000L,
    1636100042999999999L};

  auto results = cudf::strings::from_timestamps(
    input,
    "[%d/%m/%y/%Y %H:%I:%M:%S.%f %z:%Z %j %u %U %W %V %G %p %a %A %b %B]",
    cudf::strings_column_view(format_names()));

  // clang-format off
  cudf::test::strings_column_wrapper expected({
  "[17/02/22/2022 01:01:02:00.000000 +0000:UTC 048 4 07 07 07 2022 AM Thu Thursday Feb February]",
  "[13/03/22/2022 10:10:38:00.000001 +0000:UTC 072 7 11 10 10 2022 AM Sun Sunday Mar March]",
  "[06/04/22/2022 20:08:14:00.001000 +0000:UTC 096 3 14 14 14 2022 PM Wed Wednesday Apr April]",
  "[06/05/20/2020 03:03:10:21.123456 +0000:UTC 127 3 18 18 19 2020 AM Wed Wednesday May May]",
  "[19/06/19/2019 12:12:54:52.987654 +0000:UTC 170 3 24 24 25 2019 PM Wed Wednesday Jun June]",
  "[29/07/61/1961 16:04:22:29.989796 +0000:UTC 210 6 30 30 30 1961 PM Sat Saturday Jul July]",
  "[05/08/21/2021 20:08:14:02.090807 +0000:UTC 217 4 31 31 31 2021 PM Thu Thursday Aug August]",
  "[23/09/21/2021 15:03:26:00.500400 +0000:UTC 266 4 38 38 38 2021 PM Thu Thursday Sep September]",
  "[05/10/21/2021 20:08:14:02.000000 +0000:UTC 278 2 40 40 40 2021 PM Tue Tuesday Oct October]",
  "[05/11/21/2021 08:08:14:02.999999 +0000:UTC 309 5 44 44 44 2021 AM Fri Friday Nov November]"});
  // clang-format on
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
}

TEST_F(StringsDatetimeTest, ZeroSizeStringsColumn)
{
  auto const zero_size_column = cudf::make_empty_column(cudf::type_id::TIMESTAMP_SECONDS)->view();
  auto results                = cudf::strings::from_timestamps(zero_size_column);
  cudf::test::expect_column_empty(results->view());

  auto const zero_size_strings_column = cudf::make_empty_column(cudf::type_id::STRING)->view();
  results = cudf::strings::to_timestamps(cudf::strings_column_view(zero_size_strings_column),
                                         cudf::data_type{cudf::type_id::TIMESTAMP_SECONDS},
                                         "%Y");
  EXPECT_EQ(0, results->size());
}

TEST_F(StringsDatetimeTest, Errors)
{
  cudf::test::strings_column_wrapper strings{"this column intentionally left blank"};
  cudf::strings_column_view view(strings);
  EXPECT_THROW(cudf::strings::to_timestamps(view, cudf::data_type{cudf::type_id::INT64}, "%Y"),
               cudf::logic_error);
  EXPECT_THROW(
    cudf::strings::to_timestamps(view, cudf::data_type{cudf::type_id::TIMESTAMP_SECONDS}, ""),
    std::invalid_argument);
  EXPECT_THROW(
    cudf::strings::to_timestamps(view, cudf::data_type{cudf::type_id::TIMESTAMP_SECONDS}, "%2Y"),
    std::invalid_argument);
  EXPECT_THROW(
    cudf::strings::to_timestamps(view, cudf::data_type{cudf::type_id::TIMESTAMP_SECONDS}, "%g"),
    std::invalid_argument);

  cudf::test::fixed_width_column_wrapper<int64_t> invalid_timestamps{1530705600};
  EXPECT_THROW(cudf::strings::from_timestamps(invalid_timestamps), std::invalid_argument);
  cudf::test::fixed_width_column_wrapper<cudf::timestamp_s, cudf::timestamp_s::rep> timestamps{
    1530705600};
  EXPECT_THROW(cudf::strings::from_timestamps(timestamps, ""), std::invalid_argument);
  EXPECT_THROW(cudf::strings::from_timestamps(timestamps, "%B", view), std::invalid_argument);

  EXPECT_THROW(cudf::strings::is_timestamp(view, "%D"), std::invalid_argument);
  EXPECT_THROW(cudf::strings::is_timestamp(view, "%p %"), std::invalid_argument);
  EXPECT_THROW(cudf::strings::from_timestamps(timestamps, "%Y:%H", view), std::invalid_argument);
}

TEST_F(StringsDatetimeTest, IsTimestampLiteralOverrun)
{
  auto input = cudf::test::strings_column_wrapper{"2021-01-01 02:00:00"};
  cudf::strings_column_view sv(input);

  auto format   = "%Y-%m-%d %H:%M:%S_";  // extra literal at the end
  auto result   = cudf::strings::is_timestamp(sv, format);
  auto expected = cudf::test::fixed_width_column_wrapper<bool>({0});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result->view(), expected);
}
