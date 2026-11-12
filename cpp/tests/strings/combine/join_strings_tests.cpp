/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>

#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
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

TEST_F(JoinStringsTest, JoinWithNulls)
{
  auto const sep   = cudf::string_scalar("|");
  auto const narep = cudf::string_scalar("-");
  {
    auto input = cudf::test::strings_column_wrapper({"x", ""}, {true, false});
    auto view  = cudf::strings_column_view(input);

    auto results = cudf::strings::join_strings(view, sep);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, cudf::test::strings_column_wrapper({"x"}));
    results = cudf::strings::join_strings(view, cudf::string_scalar("<>"));
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, cudf::test::strings_column_wrapper({"x"}));
    results = cudf::strings::join_strings(view, sep, narep);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, cudf::test::strings_column_wrapper({"x|-"}));
  }
  {
    auto input = cudf::test::strings_column_wrapper({"x", "y", ""}, {true, true, false});
    auto view  = cudf::strings_column_view(input);

    auto results = cudf::strings::join_strings(view, sep);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, cudf::test::strings_column_wrapper({"x|y"}));
    results = cudf::strings::join_strings(view, sep, narep);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, cudf::test::strings_column_wrapper({"x|y|-"}));
  }
  {
    auto input = cudf::test::strings_column_wrapper({"x", "", ""}, {true, false, false});
    auto view  = cudf::strings_column_view(input);

    auto results = cudf::strings::join_strings(view, sep);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, cudf::test::strings_column_wrapper({"x"}));
    results = cudf::strings::join_strings(view, sep, narep);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, cudf::test::strings_column_wrapper({"x|-|-"}));
  }
  {
    auto input = cudf::test::strings_column_wrapper({"", "", "z"}, {false, false, true});
    auto view  = cudf::strings_column_view(input);

    auto results = cudf::strings::join_strings(view, sep);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, cudf::test::strings_column_wrapper({"z"}));
    results = cudf::strings::join_strings(view, sep, narep);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, cudf::test::strings_column_wrapper({"-|-|z"}));
  }
  {
    auto input =
      cudf::test::strings_column_wrapper({"", "x", "", "z", ""}, {false, true, false, true, false});
    auto view = cudf::strings_column_view(input);

    auto results = cudf::strings::join_strings(view, sep);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, cudf::test::strings_column_wrapper({"x|z"}));
    results = cudf::strings::join_strings(view, sep, narep);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, cudf::test::strings_column_wrapper({"-|x|-|z|-"}));
  }
  {
    auto input =
      cudf::test::strings_column_wrapper({"w", "x", "", "y", "z"}, {true, true, false, true, true});
    auto sliced = cudf::slice(input, {1, 3}).front();
    auto view   = cudf::strings_column_view(sliced);

    auto results = cudf::strings::join_strings(view, sep);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, cudf::test::strings_column_wrapper({"x"}));
    results = cudf::strings::join_strings(view, sep, narep);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, cudf::test::strings_column_wrapper({"x|-"}));
  }
}

TEST_F(JoinStringsTest, JoinLongStringsWithNulls)
{
  // long strings exercise the string-gather code path
  std::string data(200, '0');
  auto input =
    cudf::test::strings_column_wrapper({data, data, data, data}, {true, false, true, false});

  auto results =
    cudf::strings::join_strings(cudf::strings_column_view(input), cudf::string_scalar("+"));

  auto expected_data = data + "+" + data;
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, cudf::test::strings_column_wrapper({expected_data}));
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
