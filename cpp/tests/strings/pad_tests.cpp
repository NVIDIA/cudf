/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/strings/padding.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/strings/wrap.hpp>
#include <cudf/utilities/error.hpp>

#include <thrust/iterator/transform_iterator.h>

#include <vector>

struct StringsPadTest : public cudf::test::BaseFixture {};

TEST_F(StringsPadTest, Padding)
{
  std::vector<char const*> h_strings{"eee ddd", "bb cc", nullptr, "", "aa", "bbb", "ééé", "o"};
  cudf::test::strings_column_wrapper strings(
    h_strings.begin(),
    h_strings.end(),
    thrust::make_transform_iterator(h_strings.begin(), [](auto str) { return str != nullptr; }));
  cudf::size_type width = 6;
  std::string phil      = "+";
  auto strings_view     = cudf::strings_column_view(strings);

  {
    auto results = cudf::strings::pad(strings_view, width, cudf::strings::side_type::RIGHT, phil);

    std::vector<char const*> h_expected{
      "eee ddd", "bb cc+", nullptr, "++++++", "aa++++", "bbb+++", "ééé+++", "o+++++"};
    cudf::test::strings_column_wrapper expected(
      h_expected.begin(),
      h_expected.end(),
      thrust::make_transform_iterator(h_expected.begin(), [](auto str) { return str != nullptr; }));
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
  {
    auto results = cudf::strings::pad(strings_view, width, cudf::strings::side_type::LEFT, phil);

    std::vector<char const*> h_expected{
      "eee ddd", "+bb cc", nullptr, "++++++", "++++aa", "+++bbb", "+++ééé", "+++++o"};
    cudf::test::strings_column_wrapper expected(
      h_expected.begin(),
      h_expected.end(),
      thrust::make_transform_iterator(h_expected.begin(), [](auto str) { return str != nullptr; }));
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
  {
    auto results = cudf::strings::pad(strings_view, width, cudf::strings::side_type::BOTH, phil);

    std::vector<char const*> h_expected{
      "eee ddd", "bb cc+", nullptr, "++++++", "++aa++", "+bbb++", "+ééé++", "++o+++"};
    cudf::test::strings_column_wrapper expected(
      h_expected.begin(),
      h_expected.end(),
      thrust::make_transform_iterator(h_expected.begin(), [](auto str) { return str != nullptr; }));
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
}

TEST_F(StringsPadTest, PaddingBoth)
{
  cudf::test::strings_column_wrapper strings({"koala", "foxx", "fox", "chameleon"});
  std::string phil  = "+";
  auto strings_view = cudf::strings_column_view(strings);

  {  // even width left justify
    auto results = cudf::strings::pad(strings_view, 6, cudf::strings::side_type::BOTH, phil);
    cudf::test::strings_column_wrapper expected({"koala+", "+foxx+", "+fox++", "chameleon"});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
  {  // odd width right justify
    auto results = cudf::strings::pad(strings_view, 7, cudf::strings::side_type::BOTH, phil);
    cudf::test::strings_column_wrapper expected({"+koala+", "++foxx+", "++fox++", "chameleon"});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
}

TEST_F(StringsPadTest, ZeroSizeStringsColumn)
{
  auto const zero_size_strings_column = cudf::make_empty_column(cudf::type_id::STRING)->view();

  auto strings_view = cudf::strings_column_view(zero_size_strings_column);
  auto results      = cudf::strings::pad(strings_view, 5);
  cudf::test::expect_column_empty(results->view());
}

class PadParameters : public StringsPadTest, public testing::WithParamInterface<cudf::size_type> {};

TEST_P(PadParameters, Padding)
{
  std::vector<std::string> h_strings{"eee ddd", "bb cc", "aa", "bbb", "fff", "", "o"};
  cudf::test::strings_column_wrapper strings(h_strings.begin(), h_strings.end());
  cudf::size_type width = GetParam();
  auto strings_view     = cudf::strings_column_view(strings);
  auto results          = cudf::strings::pad(strings_view, width, cudf::strings::side_type::RIGHT);

  std::vector<std::string> h_expected;
  for (auto str : h_strings) {
    cudf::size_type size = str.size();
    if (size < width) str.insert(size, width - size, ' ');
    h_expected.push_back(str);
  }
  cudf::test::strings_column_wrapper expected(h_expected.begin(), h_expected.end());

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
}

INSTANTIATE_TEST_CASE_P(StringsPadTest,
                        PadParameters,
                        testing::ValuesIn(std::array<cudf::size_type, 3>{5, 6, 7}));

TEST_F(StringsPadTest, ZFill)
{
  auto input = cudf::test::strings_column_wrapper(
    {"654321", "-12345", "", "", "-5", "0987", "4", "+8.5", "éé", "+abé", "é+a", "100-"},
    {1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1});
  auto strings_view = cudf::strings_column_view(input);

  auto results = cudf::strings::zfill(strings_view, 6);

  auto expected = cudf::test::strings_column_wrapper({"654321",
                                                      "-12345",
                                                      "",
                                                      "000000",
                                                      "-00005",
                                                      "000987",
                                                      "000004",
                                                      "+008.5",
                                                      "0000éé",
                                                      "+00abé",
                                                      "000é+a",
                                                      "00100-"},
                                                     {1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
}

TEST_F(StringsPadTest, ZFillByWidths)
{
  auto input = cudf::test::strings_column_wrapper(
    {"654321", "-12345", "", "", "-5", "0987", "4", "+8.5", "éé", "+abé", "é+a", "100-", "123"},
    {1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1});
  auto sv     = cudf::strings_column_view(input);
  auto widths = cudf::test::fixed_width_column_wrapper<cudf::size_type>(
    {6, 5, 4, 3, 4, 5, 6, 5, 4, 6, 5, 7, 0}, {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0});

  auto results = cudf::strings::zfill_by_widths(sv, widths);

  auto expected = cudf::test::strings_column_wrapper({"654321",
                                                      "-12345",
                                                      "",
                                                      "000",
                                                      "-005",
                                                      "00987",
                                                      "000004",
                                                      "+08.5",
                                                      "00éé",
                                                      "+00abé",
                                                      "00é+a",
                                                      "000100-",
                                                      ""},
                                                     {1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
}

TEST_F(StringsPadTest, ZFillByWidthsSliced)
{
  auto input = cudf::test::strings_column_wrapper(
    {"654321", "-12345", "", "", "-5", "0987", "4", "+8.5", "éé", "+abé", "é+a", "100-", "123"},
    {1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1});
  auto widths = cudf::test::fixed_width_column_wrapper<cudf::size_type>(
    {6, 5, 4, 3, 4, 5, 6, 5, 4, 6, 5, 7, 0}, {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0});

  // slice such that both the input and the widths retain a null row
  auto input_sliced  = cudf::slice(input, {2, 13}).front();
  auto widths_sliced = cudf::slice(widths, {2, 13}).front();
  auto sv            = cudf::strings_column_view(input_sliced);

  auto results = cudf::strings::zfill_by_widths(sv, widths_sliced);

  auto expected = cudf::test::strings_column_wrapper(
    {"", "000", "-005", "00987", "000004", "+08.5", "00éé", "+00abé", "00é+a", "000100-", ""},
    {0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
}

TEST_F(StringsPadTest, ZFillError)
{
  auto input  = cudf::test::strings_column_wrapper({"654321", "-12345", "", ""}, {1, 1, 0, 1});
  auto sv     = cudf::strings_column_view(input);
  auto widths = cudf::test::fixed_width_column_wrapper<cudf::size_type>({6, 5, 4, 3, 2});
  EXPECT_THROW(cudf::strings::zfill_by_widths(sv, widths), std::invalid_argument);
}

TEST_F(StringsPadTest, Wrap1)
{
  std::vector<char const*> h_strings{"12345", "thesé", nullptr, "ARE THE", "tést strings", ""};
  cudf::test::strings_column_wrapper strings(
    h_strings.begin(),
    h_strings.end(),
    thrust::make_transform_iterator(h_strings.begin(), [](auto str) { return str != nullptr; }));
  cudf::size_type width = 3;

  auto strings_view = cudf::strings_column_view(strings);

  auto results = cudf::strings::wrap(strings_view, width);

  std::vector<char const*> h_expected{"12345", "thesé", nullptr, "ARE\nTHE", "tést\nstrings", ""};
  cudf::test::strings_column_wrapper expected(
    h_expected.begin(),
    h_expected.end(),
    thrust::make_transform_iterator(h_expected.begin(), [](auto str) { return str != nullptr; }));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
}

TEST_F(StringsPadTest, Wrap2)
{
  std::vector<char const*> h_strings{"the quick brown fox jumped over the lazy brown dog",
                                     "hello, world"};
  cudf::test::strings_column_wrapper strings(
    h_strings.begin(),
    h_strings.end(),
    thrust::make_transform_iterator(h_strings.begin(), [](auto str) { return str != nullptr; }));
  cudf::size_type width = 12;

  auto strings_view = cudf::strings_column_view(strings);

  auto results = cudf::strings::wrap(strings_view, width);

  std::vector<char const*> h_expected{"the quick\nbrown fox\njumped over\nthe lazy\nbrown dog",
                                      "hello, world"};
  cudf::test::strings_column_wrapper expected(
    h_expected.begin(),
    h_expected.end(),
    thrust::make_transform_iterator(h_expected.begin(), [](auto str) { return str != nullptr; }));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
}

TEST_F(StringsPadTest, WrapExpectFailure)
{
  std::vector<char const*> h_strings{"12345", "thesé", nullptr, "ARE THE", "tést strings", ""};
  cudf::test::strings_column_wrapper strings(
    h_strings.begin(),
    h_strings.end(),
    thrust::make_transform_iterator(h_strings.begin(), [](auto str) { return str != nullptr; }));

  cudf::size_type width = 0;  // this should trigger failure

  auto strings_view = cudf::strings_column_view(strings);

  EXPECT_THROW(cudf::strings::wrap(strings_view, width), cudf::logic_error);
}
