/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/debug_utilities.hpp>
#include <cudf_test/type_lists.hpp>

#include <cuda/iterator>

#include <type_traits>

template <typename T>
struct ColumnDebugTestIntegral : public cudf::test::BaseFixture {};
template <typename T>
struct ColumnDebugTestFloatingPoint : public cudf::test::BaseFixture {};

TYPED_TEST_SUITE(ColumnDebugTestIntegral, cudf::test::IntegralTypes);
TYPED_TEST_SUITE(ColumnDebugTestFloatingPoint, cudf::test::FloatingPointTypes);

TYPED_TEST(ColumnDebugTestIntegral, PrintColumnNumeric)
{
  char const* delimiter = ",";

  cudf::test::fixed_width_column_wrapper<TypeParam> cudf_col({1, 2, 3, 4, 5});
  auto std_col = cudf::test::make_type_param_vector<TypeParam>({1, 2, 3, 4, 5});

  std::stringstream tmp;
  auto string_iter =
    cuda::transform_iterator(std::begin(std_col), [](auto e) { return std::to_string(e); });

  std::copy(string_iter,
            string_iter + std_col.size() - 1,
            std::ostream_iterator<std::string>(tmp, delimiter));

  tmp << std::to_string(std_col.back());

  EXPECT_EQ(cudf::test::to_string(cudf_col, delimiter), tmp.str());
}

TYPED_TEST(ColumnDebugTestIntegral, PrintColumnWithInvalids)
{
  char const* delimiter = ",";

  cudf::test::fixed_width_column_wrapper<TypeParam> cudf_col{{1, 2, 3, 4, 5}, {1, 0, 1, 0, 1}};
  auto std_col = cudf::test::make_type_param_vector<TypeParam>({1, 2, 3, 4, 5});

  std::ostringstream tmp;
  tmp << std::to_string(std_col[0]) << delimiter << "NULL" << delimiter
      << std::to_string(std_col[2]) << delimiter << "NULL" << delimiter
      << std::to_string(std_col[4]);

  EXPECT_EQ(cudf::test::to_string(cudf_col, delimiter), tmp.str());
}

TYPED_TEST(ColumnDebugTestFloatingPoint, PrintColumnNumeric)
{
  char const* delimiter = ",";

  cudf::test::fixed_width_column_wrapper<TypeParam> cudf_col(
    {10001523.25, 2.0, 3.75, 0.000000034, 5.3});

  auto expected = std::is_same_v<TypeParam, double>
                    ? "10001523.25,2,3.75,3.4e-08,5.2999999999999998"
                    : "10001523,2,3.75,3.39999993e-08,5.30000019";

  EXPECT_EQ(cudf::test::to_string(cudf_col, delimiter), expected);
}

TYPED_TEST(ColumnDebugTestFloatingPoint, PrintColumnWithInvalids)
{
  char const* delimiter = ",";

  cudf::test::fixed_width_column_wrapper<TypeParam> cudf_col(
    {10001523.25, 2.0, 3.75, 0.000000034, 5.3}, {1, 0, 1, 0, 1});

  auto expected = std::is_same_v<TypeParam, double>
                    ? "10001523.25,NULL,3.75,NULL,5.2999999999999998"
                    : "10001523,NULL,3.75,NULL,5.30000019";

  EXPECT_EQ(cudf::test::to_string(cudf_col, delimiter), expected);
}

struct ColumnDebugStringsTest : public cudf::test::BaseFixture {};

TEST_F(ColumnDebugStringsTest, PrintColumnDuration)
{
  char const* delimiter = ",";

  cudf::test::fixed_width_column_wrapper<cudf::duration_s, int32_t> cudf_col({100, 0, 7, 140000});

  auto expected = "100 seconds,0 seconds,7 seconds,140000 seconds";

  EXPECT_EQ(cudf::test::to_string(cudf_col, delimiter), expected);
}

TEST_F(ColumnDebugStringsTest, StringsToString)
{
  char const* delimiter = ",";

  std::vector<char const*> h_strings{"eee", "bb", nullptr, "", "aa", "bbb", "ééé"};
  cudf::test::strings_column_wrapper strings(
    h_strings.begin(), h_strings.end(), cuda::transform_iterator(h_strings.begin(), [](auto str) {
      return str != nullptr;
    }));

  std::ostringstream tmp;
  tmp << h_strings[0] << delimiter << h_strings[1] << delimiter << "NULL" << delimiter
      << h_strings[3] << delimiter << h_strings[4] << delimiter << h_strings[5] << delimiter
      << h_strings[6];

  EXPECT_EQ(cudf::test::to_string(strings, delimiter), tmp.str());
}

TEST_F(ColumnDebugStringsTest, PrintEscapeStrings)
{
  char const* delimiter = ",";
  cudf::test::strings_column_wrapper input({"e\te\ne", "é\bé\ré", "e\vé\fé\abell"});
  std::string expected{"e\\te\\ne,é\\bé\\ré,e\\vé\\fé\\abell"};
  EXPECT_EQ(cudf::test::to_string(input, delimiter), expected);
}
