/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>

#include <cudf/strings/convert/convert_booleans.hpp>
#include <cudf/strings/strings_column_view.hpp>

#include <cuda/iterator>

#include <vector>

struct StringsConvertTest : public cudf::test::BaseFixture {};

TEST_F(StringsConvertTest, ToBooleans)
{
  std::vector<char const*> h_strings{"false", nullptr, "", "true", "True", "False"};
  cudf::test::strings_column_wrapper strings(
    h_strings.begin(), h_strings.end(), cuda::transform_iterator(h_strings.begin(), [](auto str) {
      return str != nullptr;
    }));

  auto strings_view = cudf::strings_column_view(strings);
  auto true_scalar  = cudf::string_scalar("true");
  auto results      = cudf::strings::to_booleans(strings_view, true_scalar);

  std::vector<bool> h_expected{false, false, false, true, false, false};
  cudf::test::fixed_width_column_wrapper<bool> expected(
    h_expected.begin(), h_expected.end(), cuda::transform_iterator(h_strings.begin(), [](auto str) {
      return str != nullptr;
    }));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
}

TEST_F(StringsConvertTest, FromBooleans)
{
  std::vector<char const*> h_strings{"true", nullptr, "false", "true", "true", "false"};
  cudf::test::strings_column_wrapper strings(
    h_strings.begin(), h_strings.end(), cuda::transform_iterator(h_strings.begin(), [](auto str) {
      return str != nullptr;
    }));

  std::vector<bool> h_column{true, false, false, true, true, false};
  cudf::test::fixed_width_column_wrapper<bool> column(
    h_column.begin(), h_column.end(), cuda::transform_iterator(h_strings.begin(), [](auto str) {
      return str != nullptr;
    }));

  auto true_scalar  = cudf::string_scalar("true");
  auto false_scalar = cudf::string_scalar("false");
  auto results      = cudf::strings::from_booleans(column, true_scalar, false_scalar);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, strings);
}

TEST_F(StringsConvertTest, ZeroSizeStringsColumnBoolean)
{
  auto const zero_size_column = cudf::make_empty_column(cudf::type_id::BOOL8)->view();
  auto true_scalar            = cudf::string_scalar("true");
  auto false_scalar           = cudf::string_scalar("false");
  auto results = cudf::strings::from_booleans(zero_size_column, true_scalar, false_scalar);
  cudf::test::expect_column_empty(results->view());
}

TEST_F(StringsConvertTest, ZeroSizeBooleansColumn)
{
  auto const zero_size_strings_column = cudf::make_empty_column(cudf::type_id::STRING)->view();
  auto true_scalar                    = cudf::string_scalar("true");
  auto results = cudf::strings::to_booleans(zero_size_strings_column, true_scalar);
  EXPECT_EQ(0, results->size());
}

TEST_F(StringsConvertTest, BooleanError)
{
  auto int_column   = cudf::test::fixed_width_column_wrapper<int32_t>({1, 2, 3});
  auto true_scalar  = cudf::string_scalar("true");
  auto false_scalar = cudf::string_scalar("false");
  EXPECT_THROW(cudf::strings::from_booleans(int_column, true_scalar, false_scalar),
               cudf::logic_error);

  auto bool_column = cudf::test::fixed_width_column_wrapper<bool>({1, 0, 1});
  auto null_scalar = cudf::string_scalar("", false);
  EXPECT_THROW(cudf::strings::from_booleans(bool_column, null_scalar, false_scalar),
               cudf::logic_error);
  EXPECT_THROW(cudf::strings::from_booleans(bool_column, true_scalar, null_scalar),
               cudf::logic_error);
  auto empty_scalar = cudf::string_scalar("", true);
  EXPECT_THROW(cudf::strings::from_booleans(int_column, empty_scalar, false_scalar),
               cudf::logic_error);
  EXPECT_THROW(cudf::strings::from_booleans(int_column, true_scalar, empty_scalar),
               cudf::logic_error);
}
