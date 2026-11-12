/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>

#include <cudf/strings/convert/convert_durations.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/wrappers/durations.hpp>

#include <cuda/iterator>

#include <array>
#include <vector>

struct StringsDurationsTest : public cudf::test::BaseFixture {};

TEST_F(StringsDurationsTest, FromToDurations)
{
  using T = cudf::duration_s;
  std::vector<cudf::duration_s> h_durations{
    T{131246625}, T{1563399277}, T{0}, T{1553085296}, T{1582934400}, T{-1545730073}, T{-86399}};
  std::vector<char const*> h_expected{"1519 days 01:23:45",
                                      "18094 days 21:34:37",
                                      nullptr,
                                      "17975 days 12:34:56",
                                      "18321 days 00:00:00",
                                      "-17890 days 09:27:53",
                                      "-0 days 23:59:59"};

  cudf::test::fixed_width_column_wrapper<cudf::duration_s> durations(
    h_durations.begin(),
    h_durations.end(),
    cuda::transform_iterator(h_expected.begin(), [](auto str) { return str != nullptr; }));

  auto results = cudf::strings::from_durations(durations);

  cudf::test::strings_column_wrapper expected(
    h_expected.begin(),
    h_expected.end(),
    cuda::transform_iterator(h_expected.begin(), [](auto str) { return str != nullptr; }));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);

  //
  auto new_durations = cudf::strings::to_durations(cudf::strings_column_view(expected),
                                                   cudf::data_type(cudf::type_to_id<T>()),
                                                   "%D days %H:%M:%S");
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*new_durations, durations);
}

// Note: ISO format does not have leading zeros. This test does.
TEST_F(StringsDurationsTest, ISOFormat)
{
  using T = cudf::duration_s;
  cudf::test::fixed_width_column_wrapper<T> durations{
    T{1530705600L}, T{1582934461L}, T{1451430122L}, T{1318302183L}, T{-6105994200L}};
  auto results = cudf::strings::from_durations(durations, "P%DDT%HH%MM%SS");
  cudf::test::strings_column_wrapper expected{"P17716DT12H00M00S",
                                              "P18321DT00H01M01S",
                                              "P16798DT23H02M02S",
                                              "P15258DT03H03M03S",
                                              "P-70671DT05H30M00S"};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);

  //
  cudf::test::strings_column_wrapper string_iso{
    "P17716DT12H0M0S", "P18321DT0H1M1S", "P16798DT23H2M2S", "P15258DT3H3M3S", "P-70671DT5H30M0S"};
  auto new_durations = cudf::strings::to_durations(cudf::strings_column_view(string_iso),
                                                   cudf::data_type(cudf::type_to_id<T>()),
                                                   "P%DDT%HH%MM%SS");
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*new_durations, durations);
  new_durations = cudf::strings::to_durations(
    cudf::strings_column_view(expected), cudf::data_type(cudf::type_to_id<T>()), "P%DDT%HH%MM%SS");
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*new_durations, durations);
}

TEST_F(StringsDurationsTest, ISOFormatDaysOnly)
{
  using T = cudf::duration_D;
  cudf::test::fixed_width_column_wrapper<T> durations{
    T{17716L}, T{18321L}, T{16798L}, T{15258L}, T{-70672L}};
  auto results1 = cudf::strings::from_durations(durations, "P%DDT%HH%MM%SS");
  cudf::test::strings_column_wrapper expected1{"P17716DT00H00M00S",
                                               "P18321DT00H00M00S",
                                               "P16798DT00H00M00S",
                                               "P15258DT00H00M00S",
                                               "P-70672DT00H00M00S"};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results1, expected1);

  auto results2 = cudf::strings::from_durations(durations, "P%DD");
  cudf::test::strings_column_wrapper expected2{
    "P17716D", "P18321D", "P16798D", "P15258D", "P-70672D"};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results2, expected2);

  //
  cudf::test::strings_column_wrapper string_iso{
    "P17716DT0H0M0S", "P18321DT0H0M0S", "P16798DT0H0M0S", "P15258DT0H0M0S", "P-70672DT0H0M0S"};
  auto new_durations1 = cudf::strings::to_durations(cudf::strings_column_view(string_iso),
                                                    cudf::data_type(cudf::type_to_id<T>()),
                                                    "P%DDT%HH%MM%SS");
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*new_durations1, durations);
  new_durations1 = cudf::strings::to_durations(
    cudf::strings_column_view(expected1), cudf::data_type(cudf::type_to_id<T>()), "P%DDT%HH%MM%SS");
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*new_durations1, durations);
  auto new_durations2 = cudf::strings::to_durations(
    cudf::strings_column_view(expected2), cudf::data_type(cudf::type_to_id<T>()), "P%DD");
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*new_durations2, durations);
}

TEST_F(StringsDurationsTest, ISOFormatSubseconds)
{
  using T = cudf::duration_ns;
  cudf::test::fixed_width_column_wrapper<T> durations{T{0L},
                                                      T{7000000000L},
                                                      T{11L},
                                                      T{10L},
                                                      T{17716L},
                                                      T{18321L},
                                                      T{16798L},
                                                      T{15258L},
                                                      T{15258000L},
                                                      T{-70672L}};
  auto results = cudf::strings::from_durations(durations, "P%DDT%HH%MM%SS");
  cudf::test::strings_column_wrapper expected{"P0DT00H00M00S",
                                              "P0DT00H00M07S",
                                              "P0DT00H00M00.000000011S",
                                              "P0DT00H00M00.000000010S",
                                              "P0DT00H00M00.000017716S",
                                              "P0DT00H00M00.000018321S",
                                              "P0DT00H00M00.000016798S",
                                              "P0DT00H00M00.000015258S",
                                              "P0DT00H00M00.015258000S",
                                              "P-0DT00H00M00.000070672S"};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);

  // fully isoformat compliant.
  cudf::test::strings_column_wrapper string_iso{"P0DT0H0M0S",
                                                "P0DT0H0M7S",
                                                "P0DT0H0M0.000000011S",
                                                "P0DT0H0M0.00000001S",
                                                "P0DT0H0M0.000017716S",
                                                "P0DT0H0M0.000018321S",
                                                "P0DT0H0M0.000016798S",
                                                "P0DT0H0M0.000015258S",
                                                "P0DT0H0M0.015258S",
                                                "P-0DT0H0M0.000070672S"};
  auto new_durations = cudf::strings::to_durations(cudf::strings_column_view(string_iso),
                                                   cudf::data_type(cudf::type_to_id<T>()),
                                                   "P%DDT%HH%MM%SS");
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*new_durations, durations);
  new_durations = cudf::strings::to_durations(
    cudf::strings_column_view(expected), cudf::data_type(cudf::type_to_id<T>()), "P%DDT%HH%MM%SS");
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*new_durations, durations);
}

TEST_F(StringsDurationsTest, DurationSeconds)
{
  using T = cudf::duration_s;
  cudf::test::fixed_width_column_wrapper<T> durations{
    T{0L},          // 0 days 00:00:00
    T{1L},          // 0 days 00:00:01
    T{118800L},     // 1 days 09:00:00
    T{31568404L},   // 365 days 09:00:04
    T{-118800L},    // -1 days 09:00:00
    T{-31568404L},  // -366 days +14:59:56
  };
  auto results = cudf::strings::from_durations(durations, "%D days %H:%M:%S");
  cudf::test::strings_column_wrapper expected{"0 days 00:00:00",
                                              "0 days 00:00:01",
                                              "1 days 09:00:00",
                                              "365 days 09:00:04",
                                              "-1 days 09:00:00",
                                              "-365 days 09:00:04"};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);

  auto new_durations = cudf::strings::to_durations(cudf::strings_column_view(expected),
                                                   cudf::data_type(cudf::type_to_id<T>()),
                                                   "%D days %H:%M:%S");
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*new_durations, durations);
}

TEST_F(StringsDurationsTest, DurationDays)
{
  using T = cudf::duration_D;
  cudf::test::fixed_width_column_wrapper<T> durations{
    T{0L},            // 0 days 00:00:00
    T{1L},            // 1 days 00:00:00
    T{-1L},           // -1 days 00:00:00
    T{800L},          // 800 days 00:00:00
    T{-800L},         // -800 days 00:00:00
    T{2147483647L},   //  2147483647 days 00:00:00
    T{-2147483648L},  // -2147483648 days 00:00:00
  };
  auto results = cudf::strings::from_durations(durations, "%D days %H:%M:%S");
  cudf::test::strings_column_wrapper expected{"0 days 00:00:00",
                                              "1 days 00:00:00",
                                              "-1 days 00:00:00",
                                              "800 days 00:00:00",
                                              "-800 days 00:00:00",
                                              "2147483647 days 00:00:00",
                                              "-2147483648 days 00:00:00"};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  auto new_durations = cudf::strings::to_durations(cudf::strings_column_view(expected),
                                                   cudf::data_type(cudf::type_to_id<T>()),
                                                   "%D days %H:%M:%S");
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*new_durations, durations);
}

TEST_F(StringsDurationsTest, DurationMilliseconds)
{
  using ms = cudf::duration_ms;
  cudf::test::fixed_width_column_wrapper<cudf::duration_ms> durations_ms{ms{-60000},
                                                                         ms{1530705600123},
                                                                         ms{1582934461007},
                                                                         ms{1451430122420},
                                                                         ms{1451430122400},
                                                                         ms{1451430122000},
                                                                         ms{1318302183999},
                                                                         ms{-6106017600047}};
  cudf::test::strings_column_wrapper expected_ms_3f{"-0 days 00:01:00.000",
                                                    "17716 days 12:00:00.123",
                                                    "18321 days 00:01:01.007",
                                                    "16798 days 23:02:02.420",
                                                    "16798 days 23:02:02.400",
                                                    "16798 days 23:02:02.000",
                                                    "15258 days 03:03:03.999",
                                                    "-70671 days 12:00:00.047"};

  cudf::test::strings_column_wrapper expected_ms_6f{"-0 days 00:01:00.000000",
                                                    "17716 days 12:00:00.123000",
                                                    "18321 days 00:01:01.007000",
                                                    "16798 days 23:02:02.420000",
                                                    "16798 days 23:02:02.400000",
                                                    "16798 days 23:02:02.000000",
                                                    "15258 days 03:03:03.999000",
                                                    "-70671 days 12:00:00.047000"};

  cudf::test::strings_column_wrapper expected_ms{"-0 days 00:01:00",
                                                 "17716 days 12:00:00.123",
                                                 "18321 days 00:01:01.007",
                                                 "16798 days 23:02:02.420",
                                                 "16798 days 23:02:02.400",
                                                 "16798 days 23:02:02",
                                                 "15258 days 03:03:03.999",
                                                 "-70671 days 12:00:00.047"};
  auto results = cudf::strings::from_durations(durations_ms, "%D days %H:%M:%S");
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected_ms);

  //
  results = cudf::strings::to_durations(cudf::strings_column_view(expected_ms_3f),
                                        cudf::data_type(cudf::type_to_id<ms>()),
                                        "%D days %H:%M:%S");
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, durations_ms);
  results = cudf::strings::to_durations(cudf::strings_column_view(expected_ms_6f),
                                        cudf::data_type(cudf::type_to_id<ms>()),
                                        "%D days %H:%M:%S");
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, durations_ms);
  results = cudf::strings::to_durations(cudf::strings_column_view(expected_ms),
                                        cudf::data_type(cudf::type_to_id<ms>()),
                                        "%D days %H:%M:%S");
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, durations_ms);
}

TEST_F(StringsDurationsTest, DurationMicroseconds)
{
  using us = cudf::duration_us;
  cudf::test::fixed_width_column_wrapper<cudf::duration_us> durations_us{us{-60000},
                                                                         us{1530705600123},
                                                                         us{1582934461007},
                                                                         us{1451430122420},
                                                                         us{1451430122400},
                                                                         us{1451430122000},
                                                                         us{1318302183999},
                                                                         us{-6106017600047}};
  cudf::test::strings_column_wrapper expected_us_3f{"-0 days 00:00:00.060",
                                                    "17 days 17:11:45.600",
                                                    "18 days 07:42:14.461",
                                                    "16 days 19:10:30.122",
                                                    "16 days 19:10:30.122",
                                                    "16 days 19:10:30.122",
                                                    "15 days 06:11:42.183",
                                                    "-70 days 16:06:57.600"};

  cudf::test::strings_column_wrapper expected_us_6f{"-0 days 00:00:00.060000",
                                                    "17 days 17:11:45.600123",
                                                    "18 days 07:42:14.461007",
                                                    "16 days 19:10:30.122420",
                                                    "16 days 19:10:30.122400",
                                                    "16 days 19:10:30.122000",
                                                    "15 days 06:11:42.183999",
                                                    "-70 days 16:06:57.600047"};

  cudf::test::strings_column_wrapper expected_us{"-0 days 00:00:00.060000",
                                                 "17 days 17:11:45.600123",
                                                 "18 days 07:42:14.461007",
                                                 "16 days 19:10:30.122420",
                                                 "16 days 19:10:30.122400",
                                                 "16 days 19:10:30.122000",
                                                 "15 days 06:11:42.183999",
                                                 "-70 days 16:06:57.600047"};
  auto results = cudf::strings::from_durations(durations_us, "%D days %H:%M:%S");
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected_us);

  //
  cudf::test::fixed_width_column_wrapper<cudf::duration_us> durations_us_3f{us{-60000},
                                                                            us{1530705600000},
                                                                            us{1582934461000},
                                                                            us{1451430122000},
                                                                            us{1451430122000},
                                                                            us{1451430122000},
                                                                            us{1318302183000},
                                                                            us{-6106017600000}};
  results = cudf::strings::to_durations(cudf::strings_column_view(expected_us_3f),
                                        cudf::data_type(cudf::type_to_id<us>()),
                                        "%D days %H:%M:%S");
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, durations_us_3f);
  results = cudf::strings::to_durations(cudf::strings_column_view(expected_us_6f),
                                        cudf::data_type(cudf::type_to_id<us>()),
                                        "%D days %H:%M:%S");
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, durations_us);
  results = cudf::strings::to_durations(cudf::strings_column_view(expected_us),
                                        cudf::data_type(cudf::type_to_id<us>()),
                                        "%D days %H:%M:%S");
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, durations_us);
}

TEST_F(StringsDurationsTest, DurationNanoseconds)
{
  using ns = cudf::duration_ns;
  cudf::test::fixed_width_column_wrapper<cudf::duration_ns> durations_ns{ns{1530705600123456789},
                                                                         ns{1582934461007008009},
                                                                         ns{1451430122421310209},
                                                                         ns{1318302183999777550},
                                                                         ns{-6106017600047047047}};
  auto results = cudf::strings::from_durations(durations_ns, "%D days %H:%M:%S");
  cudf::test::strings_column_wrapper expected_ns_9f{"17716 days 12:00:00.123456789",
                                                    "18321 days 00:01:01.007008009",
                                                    "16798 days 23:02:02.421310209",
                                                    "15258 days 03:03:03.999777550",
                                                    "-70671 days 12:00:00.047047047"};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected_ns_9f);

  cudf::test::strings_column_wrapper expected_ns_6f{"17716 days 12:00:00.123456",
                                                    "18321 days 00:01:01.007008",
                                                    "16798 days 23:02:02.421310",
                                                    "15258 days 03:03:03.999777",
                                                    "-70671 days 12:00:00.047047"};

  cudf::test::strings_column_wrapper expected_ns{"17716 days 12:00:00.123456789",
                                                 "18321 days 00:01:01.007008009",
                                                 "16798 days 23:02:02.421310209",
                                                 "15258 days 03:03:03.99977755",
                                                 "-70671 days 12:00:00.047047047"};

  //
  results = cudf::strings::to_durations(cudf::strings_column_view(expected_ns_9f),
                                        cudf::data_type(cudf::type_to_id<ns>()),
                                        "%D days %H:%M:%S");
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, durations_ns);
  cudf::test::fixed_width_column_wrapper<cudf::duration_ns> durations_ns_6f{
    ns{1530705600123456000},
    ns{1582934461007008000},
    ns{1451430122421310000},
    ns{1318302183999777000},
    ns{-6106017600047047000}};
  results = cudf::strings::to_durations(cudf::strings_column_view(expected_ns_6f),
                                        cudf::data_type(cudf::type_to_id<ns>()),
                                        "%D days %H:%M:%S");
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, durations_ns_6f);
  results = cudf::strings::to_durations(cudf::strings_column_view(expected_ns),
                                        cudf::data_type(cudf::type_to_id<ns>()),
                                        "%D days %H:%M:%S");
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, durations_ns);
}

// Hour, Minute, Seconds 0,+,-
TEST_F(StringsDurationsTest, ParseSingle)
{
  cudf::test::strings_column_wrapper string_src{"00",
                                                "-00",
                                                "01",
                                                "-01",
                                                "23",
                                                "-23",
                                                "59",
                                                "-59",
                                                "999",
                                                "-999",
                                                "",  // error
                                                "01",
                                                ""};  // error
  auto size = cudf::column_view(string_src).size();
  std::array expected_v{0, 0, 1, -1, 23, -23, 59, -59, 99, -99, 0, 1, 0};
  auto it1 =
    cuda::transform_iterator(expected_v.data(), [](auto i) { return cudf::duration_s{i * 3600}; });
  cudf::test::fixed_width_column_wrapper<cudf::duration_s> expected_s1(it1, it1 + size);
  auto results = cudf::strings::to_durations(cudf::strings_column_view(string_src),
                                             cudf::data_type(cudf::type_to_id<cudf::duration_s>()),
                                             "%H");
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected_s1);

  auto it2 =
    cuda::transform_iterator(expected_v.data(), [](auto i) { return cudf::duration_s{i * 60}; });
  cudf::test::fixed_width_column_wrapper<cudf::duration_s> expected_s2(it2, it2 + size);
  results = cudf::strings::to_durations(cudf::strings_column_view(string_src),
                                        cudf::data_type(cudf::type_to_id<cudf::duration_s>()),
                                        "%M");
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected_s2);

  auto it3 =
    cuda::transform_iterator(expected_v.data(), [](auto i) { return cudf::duration_s{i}; });
  cudf::test::fixed_width_column_wrapper<cudf::duration_s> expected_s3(it3, it3 + size);
  results = cudf::strings::to_durations(cudf::strings_column_view(string_src),
                                        cudf::data_type(cudf::type_to_id<cudf::duration_s>()),
                                        "%S");
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected_s3);

  auto it4 = cuda::transform_iterator(expected_v.data(),
                                      [](auto i) { return cudf::duration_ms{i * 60000}; });
  cudf::test::fixed_width_column_wrapper<cudf::duration_ms> expected_ms(it4, it4 + size);
  results = cudf::strings::to_durations(cudf::strings_column_view(string_src),
                                        cudf::data_type(cudf::type_to_id<cudf::duration_ms>()),
                                        "%M");
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected_ms);
}

// Hour, Minute, Seconds
TEST_F(StringsDurationsTest, ParseMultiple)
{
  cudf::test::strings_column_wrapper string_src{"00:00:00",
                                                "-00:00:00",
                                                "-00:00:01",
                                                "-01:01:01",
                                                "23:00:01",
                                                "-23:00:01",
                                                "59:00:00",
                                                "-59:00:00",
                                                "999:00:00",
                                                "-999:00:00",
                                                "",  // error
                                                "01:01:01",
                                                ""};  // error
  auto size = cudf::column_view(string_src).size();
  std::array expected_v{0,
                        0,
                        -1,
                        -(3600 + 60 + 1),
                        23 * 3600 + 1,
                        -(23 * 3600 + 1),
                        59 * 3600,
                        -59 * 3600,
                        99 * 3600,
                        -99 * 3600,
                        0,
                        3661,
                        0};
  auto it1 =
    cuda::transform_iterator(expected_v.data(), [](auto i) { return cudf::duration_s{i}; });
  cudf::test::fixed_width_column_wrapper<cudf::duration_s> expected_s1(it1, it1 + size);
  auto results = cudf::strings::to_durations(cudf::strings_column_view(string_src),
                                             cudf::data_type(cudf::type_to_id<cudf::duration_s>()),
                                             "%H:%M:%S");
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected_s1);

  auto it2 = cuda::transform_iterator(expected_v.data(),
                                      [](auto i) { return cudf::duration_D{i / (24 * 3600)}; });
  cudf::test::fixed_width_column_wrapper<cudf::duration_D> expected_D2(it2, it2 + size);
  results = cudf::strings::to_durations(cudf::strings_column_view(string_src),
                                        cudf::data_type(cudf::type_to_id<cudf::duration_D>()),
                                        "%H:%M:%S");
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected_D2);

  cudf::test::fixed_width_column_wrapper<cudf::duration_us> expected_us3(it1, it1 + size);
  results = cudf::strings::to_durations(cudf::strings_column_view(string_src),
                                        cudf::data_type(cudf::type_to_id<cudf::duration_us>()),
                                        "%H:%M:%S");
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected_us3);
}

// 0,+,- on DHMSs
// subsecond=0,1,2,3,4,5,6,8,9,digits, also leading zeros. subsecond with/without zero HMS.
TEST_F(StringsDurationsTest, ParseSubsecond)
{
  cudf::test::strings_column_wrapper string_src{"00:00:00.000000000",
                                                "-00:00:00.123456789",
                                                "-00:00:01.000666999",  // leading zeros
                                                "-01:01:01.100000000",
                                                "23:00:01.00000008",    // trailing zero missing
                                                "-23:00:01.123000000",  // trailing zeros
                                                "59:00:00",
                                                "-59:00:00",
                                                "999:00:00",
                                                "-999:00:00",
                                                "",  // error
                                                "01:01:01",
                                                ""};  // error
  auto size = cudf::column_view(string_src).size();
  std::array<int64_t, 13> expected_v{0,
                                     -123456789L,
                                     -1000666999L,
                                     -((3600 + 60 + 1) * 1000000000L + 100000000L),
                                     (23 * 3600 + 1) * 1000000000L + 80L,
                                     -((23 * 3600 + 1) * 1000000000L + 123000000L),
                                     (59 * 3600) * 1000000000L,
                                     -(59 * 3600) * 1000000000L,
                                     (99 * 3600) * 1000000000L,
                                     -(99 * 3600) * 1000000000L,
                                     0,
                                     (3661) * 1000000000L,
                                     0};
  auto it1 =
    cuda::transform_iterator(expected_v.data(), [](auto i) { return cudf::duration_ns{i}; });
  cudf::test::fixed_width_column_wrapper<cudf::duration_ns> expected_ns1(it1, it1 + size);
  auto results = cudf::strings::to_durations(cudf::strings_column_view(string_src),
                                             cudf::data_type(cudf::type_to_id<cudf::duration_ns>()),
                                             "%H:%M:%S");
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected_ns1);

  auto it2 = cuda::transform_iterator(expected_v.data(),
                                      [](auto i) { return cudf::duration_ms{i / 1000000}; });
  cudf::test::fixed_width_column_wrapper<cudf::duration_ms> expected_ms2(it2, it2 + size);
  results = cudf::strings::to_durations(cudf::strings_column_view(string_src),
                                        cudf::data_type(cudf::type_to_id<cudf::duration_ms>()),
                                        "%H:%M:%S");
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected_ms2);
}

// AM/PM  0,+,- on DHMSs
TEST_F(StringsDurationsTest, ParseAMPM)
{
  cudf::test::strings_column_wrapper string_src{"00:00:00 AM",
                                                "00:00:00 PM",
                                                "-00:00:00 AM",
                                                "-00:00:00 PM",
                                                "-00:00:01 AM",
                                                "-00:00:01 PM",
                                                "-01:01:01 AM",
                                                "-01:01:01 PM",
                                                "11:59:59 AM",
                                                "11:59:59 PM",
                                                "-11:59:59 AM",
                                                "-11:59:59 PM",
                                                "09:00:00",   // error
                                                "-09:00:00",  // error
                                                "",           // error
                                                "01:01:01",   // error
                                                ""};          // error
  auto size = cudf::column_view(string_src).size();
  std::array expected_v{0,
                        0 + 12 * 3600,
                        0,
                        0 - 12 * 3600,
                        -1,
                        -1 - 12 * 3600,
                        -(3600 + 60 + 1),
                        -(3600 + 60 + 1) - 12 * 3600,
                        11 * 3600 + 59 * 60 + 59,
                        11 * 3600 + 59 * 60 + 59 + 12 * 3600,
                        -(11 * 3600 + 59 * 60 + 59),
                        -(11 * 3600 + 59 * 60 + 59 + 12 * 3600),
                        0,
                        0,
                        0,
                        0,
                        0};
  auto it1 =
    cuda::transform_iterator(expected_v.data(), [](auto i) { return cudf::duration_s{i}; });
  cudf::test::fixed_width_column_wrapper<cudf::duration_s> expected_s1(it1, it1 + size);
  auto results = cudf::strings::to_durations(cudf::strings_column_view(string_src),
                                             cudf::data_type(cudf::type_to_id<cudf::duration_s>()),
                                             "%H:%M:%S %p");
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected_s1);

  auto it2 = cuda::transform_iterator(expected_v.data(),
                                      [](auto i) { return cudf::duration_D{i / (24 * 3600)}; });
  cudf::test::fixed_width_column_wrapper<cudf::duration_D> expected_D2(it2, it2 + size);
  results = cudf::strings::to_durations(cudf::strings_column_view(string_src),
                                        cudf::data_type(cudf::type_to_id<cudf::duration_D>()),
                                        "%H:%M:%S %p");
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected_D2);

  cudf::test::fixed_width_column_wrapper<cudf::duration_us> expected_us3(it1, it1 + size);
  results = cudf::strings::to_durations(cudf::strings_column_view(string_src),
                                        cudf::data_type(cudf::type_to_id<cudf::duration_us>()),
                                        "%H:%M:%S %p");
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected_us3);
}

// R, T, r
TEST_F(StringsDurationsTest, ParseCompoundSpecifier)
{
  // %r
  cudf::test::strings_column_wrapper string_src{"00:00:00 AM",
                                                "00:00:00 PM",
                                                "00:00:01 AM",
                                                "00:00:01 PM",
                                                "01:01:01 AM",
                                                "01:01:01 PM",
                                                "11:59:59 AM",
                                                "11:59:59 PM",
                                                "09:00:00",  // error
                                                "",          // error
                                                "01:01:01",  // error
                                                ""};         // error
  auto size = cudf::column_view(string_src).size();
  std::array expected_v{0,
                        0 + 12 * 3600,
                        1,
                        1 + 12 * 3600,
                        (3600 + 60 + 1),
                        (3600 + 60 + 1) + 12 * 3600,
                        11 * 3600 + 59 * 60 + 59,
                        11 * 3600 + 59 * 60 + 59 + 12 * 3600,
                        0,
                        0,
                        0,
                        0};
  auto it1 =
    cuda::transform_iterator(expected_v.data(), [](auto i) { return cudf::duration_s{i}; });
  cudf::test::fixed_width_column_wrapper<cudf::duration_s> expected_s1(it1, it1 + size);
  auto results = cudf::strings::to_durations(cudf::strings_column_view(string_src),
                                             cudf::data_type(cudf::type_to_id<cudf::duration_s>()),
                                             "%r");
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected_s1);

  results = cudf::strings::to_durations(cudf::strings_column_view(string_src),
                                        cudf::data_type(cudf::type_to_id<cudf::duration_s>()),
                                        "%OI:%OM:%OS %p");
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected_s1);

  auto it2 =
    cuda::transform_iterator(expected_v.data(), [](auto i) { return cudf::duration_ms{i * 1000}; });
  cudf::test::fixed_width_column_wrapper<cudf::duration_ms> expected_s2(it2, it2 + size);
  results = cudf::strings::to_durations(cudf::strings_column_view(string_src),
                                        cudf::data_type(cudf::type_to_id<cudf::duration_ms>()),
                                        "%r");
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected_s2);

  // %T, %R
  cudf::test::strings_column_wrapper string_src2{"00:00:00",
                                                 "12:00:00",
                                                 "20:44:01",
                                                 "-20:44:01",
                                                 "08:01:01",
                                                 "-08:01:01",
                                                 "11:59:59",
                                                 "-11:59:59 AM",
                                                 "09:00 AM",  // error
                                                 "",          // error
                                                 "01:01:01",
                                                 ""};  // error

  cudf::test::fixed_width_column_wrapper<cudf::duration_s, int64_t> expected_s3(
    {0,
     12 * 3600,
     (20 * 3600 + 44 * 60 + 1),
     -(20 * 3600 + 44 * 60 + 1),
     (8 * 3600 + 60 + 1),
     -(8 * 3600 + 60 + 1),
     (11 * 3600 + 59 * 60 + 59),
     -(11 * 3600 + 59 * 60 + 59),
     9 * 3600,
     0,
     3661,
     0});
  results = cudf::strings::to_durations(cudf::strings_column_view(string_src2),
                                        cudf::data_type(cudf::type_to_id<cudf::duration_s>()),
                                        "%T");
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected_s3);

  cudf::test::fixed_width_column_wrapper<cudf::duration_s, int64_t> expected_s4(
    {0,
     12 * 3600,
     (20 * 3600 + 44 * 60),
     -(20 * 3600 + 44 * 60),
     (8 * 3600 + 60),
     -(8 * 3600 + 60),
     (11 * 3600 + 59 * 60),
     -(11 * 3600 + 59 * 60),
     9 * 3600,
     0,
     3660,
     0});
  results = cudf::strings::to_durations(cudf::strings_column_view(string_src2),
                                        cudf::data_type(cudf::type_to_id<cudf::duration_s>()),
                                        "%R");
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected_s4);
}

// Escape characters %% %n %t
// Mixed (for checking only one negative sign)
TEST_F(StringsDurationsTest, ParseEscapeCharacters)
{
  cudf::test::strings_column_wrapper string_src{
    "00:00%00", "01:01%01", "11:59%59", "11:-59%59", "09:00%00"};
  cudf::test::fixed_width_column_wrapper<cudf::duration_s, int64_t> expected_s1(
    {0, 3661, (11 * 3600 + 59 * 60 + 59), -(11 * 3600 + 59 * 60 + 59), 9 * 3600});
  auto results = cudf::strings::to_durations(cudf::strings_column_view(string_src),
                                             cudf::data_type(cudf::type_to_id<cudf::duration_s>()),
                                             "%OH:%M%%%S");
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected_s1);

  results = cudf::strings::from_durations(expected_s1, "%OH:%M%%%S");
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, string_src);

  cudf::test::strings_column_wrapper string_src2{
    "00\t00\n00", "01\t01\n01", "11\t59\n59", "11\t-59\n59", "09\t00\n00"};
  results = cudf::strings::to_durations(cudf::strings_column_view(string_src2),
                                        cudf::data_type(cudf::type_to_id<cudf::duration_s>()),
                                        "%OH%t%M%n%S");
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected_s1);

  results = cudf::strings::from_durations(expected_s1, "%OH%t%M%n%S");
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, string_src2);
}

TEST_F(StringsDurationsTest, ZeroSizeStringsColumn)
{
  auto const zero_size_column = cudf::make_empty_column(cudf::type_id::DURATION_SECONDS)->view();
  auto results                = cudf::strings::from_durations(zero_size_column);
  cudf::test::expect_column_empty(results->view());

  auto const zero_size_strings_column = cudf::make_empty_column(cudf::type_id::STRING)->view();
  results = cudf::strings::to_durations(cudf::strings_column_view(zero_size_strings_column),
                                        cudf::data_type{cudf::type_id::DURATION_SECONDS},
                                        "%S");
  EXPECT_EQ(0, results->size());
}

TEST_F(StringsDurationsTest, Errors)
{
  cudf::test::strings_column_wrapper strings{"this column intentionally left blank"};
  cudf::strings_column_view view(strings);
  EXPECT_THROW(cudf::strings::to_durations(view, cudf::data_type{cudf::type_id::INT64}, "%D"),
               cudf::logic_error);
  EXPECT_THROW(
    cudf::strings::to_durations(view, cudf::data_type{cudf::type_id::DURATION_SECONDS}, ""),
    cudf::logic_error);
  EXPECT_THROW(
    cudf::strings::to_durations(view, cudf::data_type{cudf::type_id::DURATION_SECONDS}, "%2H"),
    cudf::logic_error);
  EXPECT_THROW(
    cudf::strings::to_durations(view, cudf::data_type{cudf::type_id::DURATION_SECONDS}, "%g"),
    cudf::logic_error);
  EXPECT_THROW(
    cudf::strings::to_durations(view, cudf::data_type{cudf::type_id::DURATION_SECONDS}, "%Op"),
    cudf::logic_error);

  cudf::test::fixed_width_column_wrapper<int64_t> invalid_durations{1530705600};
  EXPECT_THROW(cudf::strings::from_durations(invalid_durations), cudf::logic_error);
  cudf::test::fixed_width_column_wrapper<cudf::duration_s> durations{cudf::duration_s{1530705600}};
  EXPECT_THROW(cudf::strings::from_durations(durations, ""), cudf::logic_error);
}
