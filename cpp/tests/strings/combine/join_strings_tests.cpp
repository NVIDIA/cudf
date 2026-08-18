/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>

#include <cudf/column/column_factories.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/strings/combine.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/types.hpp>

#include <cuda/iterator>

struct JoinStringsTest : public cudf::test::BaseFixture {};

TEST_F(JoinStringsTest, Join)
{
  std::vector<char const*> h_strings{"eee", "bb", nullptr, "zzzz", "", "aaa", "ééé"};
  cudf::test::strings_column_wrapper strings(
    h_strings.begin(), h_strings.end(), cuda::transform_iterator(h_strings.begin(), [](auto str) {
      return str != nullptr;
    }));
  auto view1 = cudf::strings_column_view(strings);

  {
    auto results = cudf::strings::join_strings(view1);

    cudf::test::strings_column_wrapper expected{"eeebbzzzzaaaééé"};
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
  {
    auto results = cudf::strings::join_strings(view1, cudf::string_scalar("+"));

    cudf::test::strings_column_wrapper expected{"eee+bb+zzzz++aaa+ééé"};
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
  {
    auto results =
      cudf::strings::join_strings(view1, cudf::string_scalar("+"), cudf::string_scalar("___"));

    cudf::test::strings_column_wrapper expected{"eee+bb+___+zzzz++aaa+ééé"};
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
}

TEST_F(JoinStringsTest, JoinLongStrings)
{
  std::string data(200, '0');
  cudf::test::strings_column_wrapper input({data, data, data, data});

  auto results =
    cudf::strings::join_strings(cudf::strings_column_view(input), cudf::string_scalar("+"));

  auto expected_data = data + "+" + data + "+" + data + "+" + data;
  cudf::test::strings_column_wrapper expected({expected_data});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
}

TEST_F(JoinStringsTest, JoinZeroSizeStringsColumn)
{
  auto const zero_size_strings_column = cudf::make_empty_column(cudf::type_id::STRING)->view();

  auto strings_view = cudf::strings_column_view(zero_size_strings_column);
  auto results      = cudf::strings::join_strings(strings_view);
  cudf::test::expect_column_empty(results->view());
}

TEST_F(JoinStringsTest, JoinAllNullStringsColumn)
{
  cudf::test::strings_column_wrapper strings({"", "", ""}, {false, false, false});

  auto results = cudf::strings::join_strings(cudf::strings_column_view(strings));
  cudf::test::strings_column_wrapper expected1({""}, {false});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected1);

  results = cudf::strings::join_strings(
    cudf::strings_column_view(strings), cudf::string_scalar(""), cudf::string_scalar("3"));
  cudf::test::strings_column_wrapper expected2({"333"});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected2);

  results = cudf::strings::join_strings(
    cudf::strings_column_view(strings), cudf::string_scalar("-"), cudf::string_scalar("*"));
  cudf::test::strings_column_wrapper expected3({"*-*-*"});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected3);
}
