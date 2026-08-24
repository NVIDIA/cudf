/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "special_chars.h"

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/iterator_utilities.hpp>

#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/strings/contains.hpp>
#include <cudf/strings/regex/regex_program.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <cuda/iterator>
#include <thrust/host_vector.h>

#include <algorithm>
#include <array>
#include <ranges>
#include <span>
#include <type_traits>
#include <vector>

struct StringsContainsTests : public cudf::test::BaseFixture {};

TEST_F(StringsContainsTests, ContainsTest)
{
  std::vector<char const*> h_strings{"5",
                                     "hej",
                                     "\t \n",
                                     "12345",
                                     "\\",
                                     "d",
                                     "c:\\Tools",
                                     "+27",
                                     "1c2",
                                     "1C2",
                                     "0:00:0",
                                     "0:0:00",
                                     "00:0:0",
                                     "00:00:0",
                                     "00:0:00",
                                     "0:00:00",
                                     "00:00:00",
                                     "Hello world !",
                                     "Hello world!   ",
                                     "Hello worldcup  !",
                                     "0123456789",
                                     "1C2",
                                     "Xaa",
                                     "abcdefghxxx",
                                     "ABCDEFGH",
                                     "abcdefgh",
                                     "abc def",
                                     "abc\ndef",
                                     "aa\r\nbb\r\ncc\r\n\r\n",
                                     "abcabc",
                                     nullptr,
                                     ""};

  cudf::test::strings_column_wrapper strings(
    h_strings.begin(), h_strings.end(), cuda::transform_iterator(h_strings.begin(), [](auto str) {
      return str != nullptr;
    }));
  auto strings_view = cudf::strings_column_view(strings);

  std::vector<std::string> patterns{"\\d",
                                    "\\w+",
                                    "\\s",
                                    "\\S",
                                    "^.*\\\\.*$",
                                    "[1-5]+",
                                    "[a-h]+",
                                    "[A-H]+",
                                    "[a-h]*",
                                    "\n",
                                    "b.\\s*\n",
                                    ".*c",
                                    R"(\d\d:\d\d:\d\d)",
                                    R"(\d\d?:\d\d?:\d\d?)",
                                    "[Hh]ello [Ww]orld",
                                    "\\bworld\\b",
                                    ".*"};

  std::vector<bool> h_expecteds_std{
    // strings.size x patterns.size
    true,  false, false, true,  false, false, false, true,  true,  true,  true,  true,  true,
    true,  true,  true,  true,  false, false, false, true,  true,  false, false, false, false,
    false, false, false, false, false, false, true,  true,  false, true,  false, true,  true,
    true,  true,  true,  true,  true,  true,  true,  true,  true,  true,  true,  true,  true,
    true,  true,  true,  true,  true,  true,  true,  true,  true,  true,  false, false, false,
    false, true,  false, false, false, false, false, false, false, false, false, false, false,
    false, false, false, true,  true,  true,  false, false, false, false, false, false, true,
    true,  true,  false, false, false, true,  true,  false, true,  true,  true,  true,  true,
    true,  true,  true,  true,  true,  true,  true,  true,  true,  true,  true,  true,  true,
    true,  true,  true,  true,  true,  true,  true,  true,  true,  false, false, false, false,
    false, false, true,  false, true,  false, false, false, false, false, false, false, false,
    false, false, false, false, false, false, false, false, false, false, false, false, false,
    false, false, false, false, true,  false, false, true,  false, false, false, true,  true,
    true,  false, false, false, false, false, false, false, false, false, false, true,  true,
    false, false, false, false, false, false, false, false, false, false, false, true,  false,
    false, false, true,  true,  false, true,  false, false, false, false, false, false, false,
    false, true,  true,  true,  false, false, true,  true,  false, true,  true,  true,  true,
    true,  false, false, false, false, false, false, false, false, false, false, false, true,
    false, false, false, false, false, false, false, true,  true,  true,  false, true,  false,
    false, true,  false, false, false, false, false, false, false, true,  true,  true,  true,
    true,  true,  true,  true,  true,  true,  true,  true,  true,  true,  true,  true,  true,
    true,  true,  true,  true,  true,  true,  true,  true,  true,  true,  true,  true,  true,
    false, true,  false, false, true,  false, false, false, false, false, false, false, false,
    false, false, false, false, false, false, false, false, false, false, false, false, false,
    false, false, false, true,  true,  false, false, false, false, false, false, false, false,
    false, false, false, false, false, false, false, false, false, false, false, false, false,
    false, false, false, false, false, false, false, false, false, true,  true,  false, false,
    false, false, false, false, false, false, false, true,  false, true,  false, false, false,
    false, false, false, false, false, false, false, true,  false, false, false, true,  false,
    true,  true,  true,  true,  true,  false, false, false, false, false, false, false, false,
    false, false, false, false, false, false, false, false, false, false, true,  false, false,
    false, false, false, false, false, false, false, false, false, false, false, false, false,
    false, false, false, false, false, false, false, false, false, false, true,  true,  true,
    true,  true,  true,  true,  false, false, false, false, false, false, false, false, false,
    false, false, false, false, false, false, false, false, false, false, false, false, false,
    false, false, false, false, false, false, false, false, false, false, true,  true,  true,
    false, false, false, false, false, false, false, false, false, false, false, false, false,
    false, false, false, false, false, false, false, false, false, false, false, false, false,
    false, false, false, true,  true,  false, false, false, false, false, false, false, false,
    false, false, false, false, false, true,  true,  true,  true,  true,  true,  true,  true,
    true,  true,  true,  true,  true,  true,  true,  true,  true,  true,  true,  true,  true,
    true,  true,  true,  true,  true,  true,  true,  true,  true,  false, true};

  thrust::host_vector<bool> h_expecteds(h_expecteds_std);

  for (int idx = 0; idx < static_cast<int>(patterns.size()); ++idx) {
    std::string ptn  = patterns[idx];
    bool* h_expected = h_expecteds.data() + (idx * h_strings.size());
    cudf::test::fixed_width_column_wrapper<bool> expected(
      h_expected,
      h_expected + h_strings.size(),
      cuda::transform_iterator(h_strings.begin(), [](auto str) { return str != nullptr; }));
    auto prog    = cudf::strings::regex_program::create(ptn);
    auto results = cudf::strings::contains_re(strings_view, *prog);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
}

TEST_F(StringsContainsTests, MatchesTest)
{
  std::vector<char const*> h_strings{
    "The quick brown @fox jumps", "ovér the", "lazy @dog", "1234", "00:0:00", nullptr, ""};
  cudf::test::strings_column_wrapper strings(
    h_strings.begin(), h_strings.end(), cuda::transform_iterator(h_strings.begin(), [](auto str) {
      return str != nullptr;
    }));

  auto strings_view = cudf::strings_column_view(strings);
  {
    auto const pattern = std::string("lazy");
    cudf::test::fixed_width_column_wrapper<bool> expected(
      {false, false, true, false, false, false, false},
      cuda::transform_iterator(h_strings.begin(), [](auto str) { return str != nullptr; }));
    auto prog    = cudf::strings::regex_program::create(pattern);
    auto results = cudf::strings::matches_re(strings_view, *prog);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
  {
    auto const pattern = std::string("\\d+");
    cudf::test::fixed_width_column_wrapper<bool> expected(
      {false, false, false, true, true, false, false},
      cuda::transform_iterator(h_strings.begin(), [](auto str) { return str != nullptr; }));
    auto prog    = cudf::strings::regex_program::create(pattern);
    auto results = cudf::strings::matches_re(strings_view, *prog);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
  {
    auto const pattern = std::string("@\\w+");
    cudf::test::fixed_width_column_wrapper<bool> expected(
      {false, false, false, false, false, false, false},
      cuda::transform_iterator(h_strings.begin(), [](auto str) { return str != nullptr; }));
    auto prog    = cudf::strings::regex_program::create(pattern);
    auto results = cudf::strings::matches_re(strings_view, *prog);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
  {
    auto const pattern = std::string(".*");
    cudf::test::fixed_width_column_wrapper<bool> expected(
      {true, true, true, true, true, false, true},
      cuda::transform_iterator(h_strings.begin(), [](auto str) { return str != nullptr; }));
    auto prog    = cudf::strings::regex_program::create(pattern);
    auto results = cudf::strings::matches_re(strings_view, *prog);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
}

TEST_F(StringsContainsTests, MatchesIPV4Test)
{
  cudf::test::strings_column_wrapper strings({"5.79.97.178",
                                              "1.2.3.4",
                                              "5",
                                              "5.79",
                                              "5.79.97",
                                              "5.79.97.178.100",
                                              "224.0.0.0",
                                              "239.255.255.255",
                                              "5.79.97.178",
                                              "127.0.0.1"});
  auto strings_view = cudf::strings_column_view(strings);
  {  // is_ip: 58 instructions
    std::string pattern =
      "^(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)"
      "$";
    cudf::test::fixed_width_column_wrapper<bool> expected(
      {true, true, false, false, false, false, true, true, true, true});
    auto prog    = cudf::strings::regex_program::create(pattern);
    auto results = cudf::strings::matches_re(strings_view, *prog);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(results->view(), expected);
  }
  {  // is_loopback: 72 instructions
    std::string pattern =
      "^127\\.([0-9]|[1-9][0-9]|1([0-9][0-9])|2([0-4][0-9]|5[0-5]))"
      "\\.([0-9]|[1-9][0-9]|1([0-9][0-9])|2([0-4][0-9]|5[0-5]))"
      "\\.([0-9]|[1-9][0-9]|1([0-9][0-9])|2([0-4][0-9]|5[0-5]))$";
    cudf::test::fixed_width_column_wrapper<bool> expected(
      {false, false, false, false, false, false, false, false, false, true});
    auto prog    = cudf::strings::regex_program::create(pattern);
    auto results = cudf::strings::matches_re(strings_view, *prog);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(results->view(), expected);
  }
  {  // is_multicast: 79 instructions
    std::string pattern =
      "^(2(2[4-9]|3[0-9]))\\.([0-9]|[1-9][0-9]|1([0-9][0-9])|2([0-4][0-9]|5[0-5]))"
      "\\.([0-9]|[1-9][0-9]|1([0-9][0-9])|2([0-4][0-9]|5[0-5]))"
      "\\.([0-9]|[1-9][0-9]|1([0-9][0-9])|2([0-4][0-9]|5[0-5]))$";
    cudf::test::fixed_width_column_wrapper<bool> expected(
      {false, false, false, false, false, false, true, true, false, false});
    auto prog    = cudf::strings::regex_program::create(pattern);
    auto results = cudf::strings::matches_re(strings_view, *prog);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(results->view(), expected);
  }
}

TEST_F(StringsContainsTests, OctalTest)
{
  cudf::test::strings_column_wrapper strings({"A3", "B", "CDA3EY", "", "99", "\a\t\r"});
  auto strings_view = cudf::strings_column_view(strings);
  auto expected     = cudf::test::fixed_width_column_wrapper<bool>({1, 0, 1, 0, 0, 0});

  auto pattern = std::string("\\101");
  auto prog    = cudf::strings::regex_program::create(pattern);
  auto results = cudf::strings::contains_re(strings_view, *prog);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);

  pattern = std::string("\\1013");
  prog    = cudf::strings::regex_program::create(pattern);
  results = cudf::strings::contains_re(strings_view, *prog);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);

  pattern = std::string("D*\\101\\063");
  prog    = cudf::strings::regex_program::create(pattern);
  results = cudf::strings::contains_re(strings_view, *prog);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);

  pattern  = std::string("\\719");
  expected = cudf::test::fixed_width_column_wrapper<bool>({0, 0, 0, 0, 1, 0});
  prog     = cudf::strings::regex_program::create(pattern);
  results  = cudf::strings::contains_re(strings_view, *prog);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);

  pattern  = std::string(R"([\7][\11][\15])");
  expected = cudf::test::fixed_width_column_wrapper<bool>({0, 0, 0, 0, 0, 1});
  prog     = cudf::strings::regex_program::create(pattern);
  results  = cudf::strings::contains_re(strings_view, *prog);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
}

TEST_F(StringsContainsTests, HexTest)
{
  std::vector<char> ascii_chars(  // all possible matchable chars
    {cuda::counting_iterator<char>{0}, cuda::counting_iterator<char>{127}});
  auto const count = static_cast<cudf::size_type>(ascii_chars.size());
  std::vector<cudf::size_type> offsets({cuda::counting_iterator<cudf::size_type>{0},
                                        cuda::counting_iterator<cudf::size_type>{0} + count + 1});
  auto d_chars = cudf::detail::make_device_uvector(
    ascii_chars, cudf::get_default_stream(), cudf::get_current_device_resource_ref());
  auto d_offsets = std::make_unique<cudf::column>(
    cudf::detail::make_device_uvector(
      offsets, cudf::get_default_stream(), cudf::get_current_device_resource_ref()),
    rmm::device_buffer{},
    0);
  auto input = cudf::make_strings_column(count, std::move(d_offsets), d_chars.release(), 0, {});

  auto strings_view = cudf::strings_column_view(input->view());
  for (auto ch : ascii_chars) {
    std::stringstream str;
    str << "\\x" << std::setfill('0') << std::setw(2) << std::hex << static_cast<int32_t>(ch);
    std::string pattern = str.str();

    // only one element in the input should match ch
    auto true_dat = cudf::detail::make_counting_transform_iterator(
      0, [ch](auto idx) { return ch == static_cast<char>(idx); });
    cudf::test::fixed_width_column_wrapper<bool> expected(true_dat, true_dat + count);
    auto prog    = cudf::strings::regex_program::create(pattern);
    auto results = cudf::strings::contains_re(strings_view, *prog);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);

    // also test hex character appearing in character class brackets
    pattern = "[" + pattern + "]";
    prog    = cudf::strings::regex_program::create(pattern);
    results = cudf::strings::contains_re(strings_view, *prog);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
}

TEST_F(StringsContainsTests, EmbeddedNullCharacter)
{
  std::vector<std::string> data(10);
  std::generate(data.begin(), data.end(), [n = 0]() mutable {
    char first          = static_cast<char>('A' + n++);
    std::array raw_data = {first, '\0', 'B'};
    return std::string{raw_data.data(), 3};
  });
  cudf::test::strings_column_wrapper input(data.begin(), data.end());
  auto strings_view = cudf::strings_column_view(input);

  auto pattern  = std::string("A");
  auto expected = cudf::test::fixed_width_column_wrapper<bool>({1, 0, 0, 0, 0, 0, 0, 0, 0, 0});
  auto prog     = cudf::strings::regex_program::create(pattern);
  auto results  = cudf::strings::contains_re(strings_view, *prog);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);

  pattern  = std::string("B");
  expected = cudf::test::fixed_width_column_wrapper<bool>({1, 1, 1, 1, 1, 1, 1, 1, 1, 1});
  prog     = cudf::strings::regex_program::create(pattern);
  results  = cudf::strings::contains_re(strings_view, *prog);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);

  pattern  = std::string("J\\0B");
  expected = cudf::test::fixed_width_column_wrapper<bool>({0, 0, 0, 0, 0, 0, 0, 0, 0, 1});
  prog     = cudf::strings::regex_program::create(pattern);
  results  = cudf::strings::contains_re(strings_view, *prog);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);

  pattern  = std::string("[G-J][\\0]B");
  expected = cudf::test::fixed_width_column_wrapper<bool>({0, 0, 0, 0, 0, 0, 1, 1, 1, 1});
  prog     = cudf::strings::regex_program::create(pattern);
  results  = cudf::strings::contains_re(strings_view, *prog);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);

  pattern  = std::string("[A-D][\\x00]B");
  expected = cudf::test::fixed_width_column_wrapper<bool>({1, 1, 1, 1, 0, 0, 0, 0, 0, 0});
  prog     = cudf::strings::regex_program::create(pattern);
  results  = cudf::strings::contains_re(strings_view, *prog);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
}

TEST_F(StringsContainsTests, Errors)
{
  EXPECT_THROW(cudf::strings::regex_program::create("(3?)+"), cudf::logic_error);
  EXPECT_THROW(cudf::strings::regex_program::create("(?:3?)+"), cudf::logic_error);
  EXPECT_THROW(cudf::strings::regex_program::create("3?+"), cudf::logic_error);
  EXPECT_THROW(cudf::strings::regex_program::create("{3}a"), cudf::logic_error);

  EXPECT_THROW(cudf::strings::regex_program::create("aaaa{1234,5678}"), cudf::logic_error);
  EXPECT_THROW(cudf::strings::regex_program::create("aaaa{123,5678}"), cudf::logic_error);

  EXPECT_THROW(cudf::strings::regex_program::create("[a-C]"), cudf::logic_error);
}

TEST_F(StringsContainsTests, CountTest)
{
  std::vector<char const*> h_strings{
    "The quick brown @fox jumps ovér the", "lazy @dog lazy", "1:2:3:4", "00:0:00", nullptr, ""};
  cudf::test::strings_column_wrapper strings(
    h_strings.begin(), h_strings.end(), cudf::test::iterators::nulls_from_nullptrs(h_strings));

  auto strings_view = cudf::strings_column_view(strings);
  {
    auto pattern = std::string("[tT]he");
    cudf::test::fixed_width_column_wrapper<int32_t> expected(
      {2, 0, 0, 0, 0, 0}, cudf::test::iterators::nulls_from_nullptrs(h_strings));
    auto prog    = cudf::strings::regex_program::create(pattern);
    auto results = cudf::strings::count_re(strings_view, *prog);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
  {
    auto pattern = std::string("@\\w+");
    cudf::test::fixed_width_column_wrapper<int32_t> expected(
      {1, 1, 0, 0, 0, 0}, cudf::test::iterators::nulls_from_nullptrs(h_strings));
    auto prog    = cudf::strings::regex_program::create(pattern);
    auto results = cudf::strings::count_re(strings_view, *prog);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
  {
    auto pattern = std::string("\\d+:\\d+");
    cudf::test::fixed_width_column_wrapper<int32_t> expected(
      {0, 0, 2, 1, 0, 0}, cudf::test::iterators::nulls_from_nullptrs(h_strings));
    auto prog    = cudf::strings::regex_program::create(pattern);
    auto results = cudf::strings::count_re(strings_view, *prog);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
  {
    auto pattern = std::string("o");
    cudf::test::fixed_width_column_wrapper<int32_t> expected(
      {3, 1, 0, 0, 0, 0}, cudf::test::iterators::nulls_from_nullptrs(h_strings));
    auto prog    = cudf::strings::regex_program::create(pattern);
    auto results = cudf::strings::count_re(strings_view, *prog);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
  {
    auto pattern = std::string("\\blazy\\b");
    cudf::test::fixed_width_column_wrapper<int32_t> expected(
      {0, 2, 0, 0, 0, 0}, cudf::test::iterators::nulls_from_nullptrs(h_strings));
    auto prog    = cudf::strings::regex_program::create(pattern);
    auto results = cudf::strings::count_re(strings_view, *prog);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
}

TEST_F(StringsContainsTests, CountEmptyMatching)
{
  auto input    = cudf::test::strings_column_wrapper({"hello", "world", "", "abc"});
  auto sv       = cudf::strings_column_view(input);
  auto patterns = std::vector<std::string>{"a*", "X?", "b{0,}", "()", "(?:)", "[A-Z]*"};
  auto expected = cudf::test::fixed_width_column_wrapper<int32_t>({6, 6, 1, 4});
  for (auto pattern : patterns) {
    auto prog    = cudf::strings::regex_program::create(pattern);
    auto results = cudf::strings::count_re(sv, *prog);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
  // "\\b", "\\B",
  expected     = cudf::test::fixed_width_column_wrapper<int32_t>({1, 1, 1, 1});
  auto prog    = cudf::strings::regex_program::create("^");
  auto results = cudf::strings::count_re(sv, *prog);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  prog    = cudf::strings::regex_program::create("$");
  results = cudf::strings::count_re(sv, *prog);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);

  expected = cudf::test::fixed_width_column_wrapper<int32_t>({0, 0, 1, 0});
  prog     = cudf::strings::regex_program::create("^$");
  results  = cudf::strings::count_re(sv, *prog);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);

  expected = cudf::test::fixed_width_column_wrapper<int32_t>({2, 2, 0, 2});
  prog     = cudf::strings::regex_program::create("\\b");
  results  = cudf::strings::count_re(sv, *prog);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);

  expected = cudf::test::fixed_width_column_wrapper<int32_t>({4, 4, 1, 2});
  prog     = cudf::strings::regex_program::create("\\B");
  results  = cudf::strings::count_re(sv, *prog);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
}

TEST_F(StringsContainsTests, FixedQuantifier)
{
  auto input = cudf::test::strings_column_wrapper({"a", "aa", "aaa", "aaaa", "aaaaa", "aaaaaa"});
  auto sv    = cudf::strings_column_view(input);

  {
    // exact match
    auto pattern = std::string("a{3}");
    cudf::test::fixed_width_column_wrapper<int32_t> expected({0, 0, 1, 1, 1, 2});
    auto prog    = cudf::strings::regex_program::create(pattern);
    auto results = cudf::strings::count_re(sv, *prog);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
  {
    // range match (greedy quantifier)
    auto pattern = std::string("a{3,5}");
    cudf::test::fixed_width_column_wrapper<int32_t> expected({0, 0, 1, 1, 1, 1});
    auto prog    = cudf::strings::regex_program::create(pattern);
    auto results = cudf::strings::count_re(sv, *prog);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
  {
    // minimum match (greedy quantifier)
    auto pattern = std::string("a{2,}");
    cudf::test::fixed_width_column_wrapper<int32_t> expected({0, 1, 1, 1, 1, 1});
    auto prog    = cudf::strings::regex_program::create(pattern);
    auto results = cudf::strings::count_re(sv, *prog);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
  {
    // range match (lazy quantifier)
    auto pattern = std::string("a{2,4}?");
    cudf::test::fixed_width_column_wrapper<int32_t> expected({0, 1, 1, 2, 2, 3});
    auto prog    = cudf::strings::regex_program::create(pattern);
    auto results = cudf::strings::count_re(sv, *prog);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
  {
    // minimum match (lazy quantifier)
    auto pattern = std::string("a{1,}?");
    cudf::test::fixed_width_column_wrapper<int32_t> expected({1, 2, 3, 4, 5, 6});
    auto prog    = cudf::strings::regex_program::create(pattern);
    auto results = cudf::strings::count_re(sv, *prog);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
  {
    // zero match
    auto pattern = std::string("aaaa{0}");
    cudf::test::fixed_width_column_wrapper<int32_t> expected({0, 0, 1, 1, 1, 2});
    auto prog    = cudf::strings::regex_program::create(pattern);
    auto results = cudf::strings::count_re(sv, *prog);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
  {
    // poorly formed
    auto pattern = std::string("aaaa{n,m}");
    cudf::test::fixed_width_column_wrapper<int32_t> expected({0, 0, 0, 0, 0, 0});
    auto prog    = cudf::strings::regex_program::create(pattern);
    auto results = cudf::strings::count_re(sv, *prog);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
}

TEST_F(StringsContainsTests, ZeroRangeQuantifier)
{
  EXPECT_NO_THROW(cudf::strings::regex_program::create("a{0}"));
  EXPECT_NO_THROW(cudf::strings::regex_program::create("a{0,1}"));
  EXPECT_NO_THROW(cudf::strings::regex_program::create("a{0,}"));
  EXPECT_NO_THROW(cudf::strings::regex_program::create("(ab){0}"));
  EXPECT_NO_THROW(cudf::strings::regex_program::create("(ab){0,1}"));
  EXPECT_NO_THROW(cudf::strings::regex_program::create("(ab){0,}"));

  auto input = cudf::test::strings_column_wrapper({"a", "", "abc", "XYAZ", "ABC", "ZYXA"});
  auto sv    = cudf::strings_column_view(input);

  auto pattern = std::string("A{0,}");  // should match everyting
  auto prog    = cudf::strings::regex_program::create(pattern);

  {
    auto expected = cudf::test::fixed_width_column_wrapper<bool>({1, 1, 1, 1, 1, 1});
    auto results  = cudf::strings::contains_re(sv, *prog);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
  {
    auto expected = cudf::test::fixed_width_column_wrapper<cudf::size_type>({2, 1, 4, 5, 4, 5});
    auto results  = cudf::strings::count_re(sv, *prog);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }

  pattern = std::string("(?:ab){0,3}");
  prog    = cudf::strings::regex_program::create(pattern);

  {
    auto expected = cudf::test::fixed_width_column_wrapper<bool>({1, 1, 1, 1, 1, 1});
    auto results  = cudf::strings::contains_re(sv, *prog);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
  {
    auto expected = cudf::test::fixed_width_column_wrapper<cudf::size_type>({2, 1, 3, 5, 4, 5});
    auto results  = cudf::strings::count_re(sv, *prog);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
}

TEST_F(StringsContainsTests, NestedQuantifier)
{
  auto input   = cudf::test::strings_column_wrapper({"TEST12 1111 2222 3333 4444 5555",
                                                     "0000 AAAA 9999 BBBB 8888",
                                                     "7777 6666 4444 3333",
                                                     "12345 3333 4444 1111 ABCD"});
  auto sv      = cudf::strings_column_view(input);
  auto pattern = std::string(R"((\d{4}\s){4})");
  cudf::test::fixed_width_column_wrapper<bool> expected({true, false, false, true});
  auto prog    = cudf::strings::regex_program::create(pattern);
  auto results = cudf::strings::contains_re(sv, *prog);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
}

TEST_F(StringsContainsTests, DeeplyNestedNoStackOverflow)
{
  // A pattern with very deep group nesting but only a single character-consuming
  // position.  It passes the Glushkov state cap (1 position) yet forces the
  // compiler's ε-closure traversal to walk a very long ε-chain.
  int32_t const depth = 100000;
  std::string pattern;
  pattern.reserve(static_cast<size_t>(depth) * 3 + 1 + static_cast<size_t>(depth));
  for (int32_t i = 0; i < depth; ++i) {
    pattern += "(?:";
  }
  pattern += "a";
  for (int32_t i = 0; i < depth; ++i) {
    pattern += ")";
  }

  auto input = cudf::test::strings_column_wrapper({"a", "b", "aaa", ""});
  auto sv    = cudf::strings_column_view(input);

  // The pattern matches any row containing 'a'.
  cudf::test::fixed_width_column_wrapper<bool> expected({true, false, true, false});
  auto prog    = cudf::strings::regex_program::create(pattern);
  auto results = cudf::strings::contains_re(sv, *prog);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
}

TEST_F(StringsContainsTests, QuantifierErrors)
{
  EXPECT_THROW(cudf::strings::regex_program::create("^+"), cudf::logic_error);
  EXPECT_THROW(cudf::strings::regex_program::create("$+"), cudf::logic_error);
  EXPECT_THROW(cudf::strings::regex_program::create("(^)+"), cudf::logic_error);
  EXPECT_THROW(cudf::strings::regex_program::create("($)+"), cudf::logic_error);
  EXPECT_THROW(cudf::strings::regex_program::create("\\A+"), cudf::logic_error);
  EXPECT_THROW(cudf::strings::regex_program::create("\\Z+"), cudf::logic_error);
  EXPECT_THROW(cudf::strings::regex_program::create("(\\A)+"), cudf::logic_error);
  EXPECT_THROW(cudf::strings::regex_program::create("(\\Z)+"), cudf::logic_error);

  EXPECT_THROW(cudf::strings::regex_program::create("(^($))+"), cudf::logic_error);
  EXPECT_NO_THROW(cudf::strings::regex_program::create("(^a($))+"));
  EXPECT_NO_THROW(cudf::strings::regex_program::create("(^(a$))+"));
}

TEST_F(StringsContainsTests, OverlappedClasses)
{
  auto input = cudf::test::strings_column_wrapper({"abcdefg", "defghí", "", "éééééé", "ghijkl"});
  auto sv    = cudf::strings_column_view(input);

  {
    auto pattern = std::string("[e-gb-da-c]");
    cudf::test::fixed_width_column_wrapper<int32_t> expected({7, 4, 0, 0, 1});
    auto prog    = cudf::strings::regex_program::create(pattern);
    auto results = cudf::strings::count_re(sv, *prog);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
  {
    auto pattern = std::string("[á-éê-ú]");
    cudf::test::fixed_width_column_wrapper<int32_t> expected({0, 1, 0, 6, 0});
    auto prog    = cudf::strings::regex_program::create(pattern);
    auto results = cudf::strings::count_re(sv, *prog);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
}

TEST_F(StringsContainsTests, NegatedClasses)
{
  auto input = cudf::test::strings_column_wrapper({"abcdefg", "def\tghí", "", "éeé\néeé", "ABC"});
  auto sv    = cudf::strings_column_view(input);

  {
    auto pattern = std::string("[^a-f]");
    cudf::test::fixed_width_column_wrapper<int32_t> expected({1, 4, 0, 5, 3});
    auto prog    = cudf::strings::regex_program::create(pattern);
    auto results = cudf::strings::count_re(sv, *prog);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
  {
    auto pattern = std::string("[^a-eá-é]");
    cudf::test::fixed_width_column_wrapper<int32_t> expected({2, 5, 0, 1, 3});
    auto prog    = cudf::strings::regex_program::create(pattern);
    auto results = cudf::strings::count_re(sv, *prog);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
}

TEST_F(StringsContainsTests, IncompleteClassesRange)
{
  auto input = cudf::test::strings_column_wrapper({"abc-def", "---", "", "ghijkl", "-wxyz-"});
  auto sv    = cudf::strings_column_view(input);

  {
    auto expected = cudf::test::fixed_width_column_wrapper<bool>({1, 0, 0, 1, 1});
    auto prog     = cudf::strings::regex_program::create("[a-z]");
    auto results  = cudf::strings::contains_re(sv, *prog);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);

    expected = cudf::test::fixed_width_column_wrapper<bool>({1, 1, 0, 1, 1});
    prog     = cudf::strings::regex_program::create("[a-m-z]");  // same as [a-mz-]
    results  = cudf::strings::contains_re(sv, *prog);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);

    expected = cudf::test::fixed_width_column_wrapper<bool>({1, 1, 0, 0, 1});
    prog     = cudf::strings::regex_program::create("[a-f-q-z]");  // same as [a-fq-z-]
    results  = cudf::strings::contains_re(sv, *prog);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);

    expected = cudf::test::fixed_width_column_wrapper<bool>({1, 1, 0, 1, 1});
    prog     = cudf::strings::regex_program::create("[g-g-z]");  // same as [gz-]
    results  = cudf::strings::contains_re(sv, *prog);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
  {
    cudf::test::fixed_width_column_wrapper<bool> expected({1, 1, 0, 1, 1});
    auto prog    = cudf::strings::regex_program::create("[g-]");
    auto results = cudf::strings::contains_re(sv, *prog);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);

    prog    = cudf::strings::regex_program::create("[-k]");
    results = cudf::strings::contains_re(sv, *prog);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
  {
    cudf::test::fixed_width_column_wrapper<bool> expected({1, 1, 0, 0, 1});
    auto prog    = cudf::strings::regex_program::create("[-]");
    auto results = cudf::strings::contains_re(sv, *prog);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
    prog    = cudf::strings::regex_program::create("[+--]");
    results = cudf::strings::contains_re(sv, *prog);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);

    prog    = cudf::strings::regex_program::create("[a-c-]");
    results = cudf::strings::contains_re(sv, *prog);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
    prog    = cudf::strings::regex_program::create("[-d-f]");
    results = cudf::strings::contains_re(sv, *prog);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
}

TEST_F(StringsContainsTests, MultiLine)
{
  auto input = cudf::test::strings_column_wrapper(
    {"abé\nfff\nabé", "fff\nabé\nlll", "abé", "", "abé\n", "abe\nabé\n"});
  auto view = cudf::strings_column_view(input);

  auto pattern = std::string("^abé$");
  auto prog    = cudf::strings::regex_program::create(pattern);
  auto prog_ml =
    cudf::strings::regex_program::create(pattern, cudf::strings::regex_flags::MULTILINE);

  auto expected_contains = cudf::test::fixed_width_column_wrapper<bool>({1, 1, 1, 0, 1, 1});
  auto results           = cudf::strings::contains_re(view, *prog_ml);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results, expected_contains);
  expected_contains = cudf::test::fixed_width_column_wrapper<bool>({0, 0, 1, 0, 1, 0});
  results           = cudf::strings::contains_re(view, *prog);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results, expected_contains);

  auto expected_matches = cudf::test::fixed_width_column_wrapper<bool>({1, 0, 1, 0, 1, 0});
  results               = cudf::strings::matches_re(view, *prog_ml);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results, expected_matches);
  expected_matches = cudf::test::fixed_width_column_wrapper<bool>({0, 0, 1, 0, 1, 0});
  results          = cudf::strings::matches_re(view, *prog);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results, expected_matches);

  auto expected_count = cudf::test::fixed_width_column_wrapper<int32_t>({2, 1, 1, 0, 1, 1});
  results             = cudf::strings::count_re(view, *prog_ml);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results, expected_count);
  expected_count = cudf::test::fixed_width_column_wrapper<int32_t>({0, 0, 1, 0, 1, 0});
  results        = cudf::strings::count_re(view, *prog);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results, expected_count);
}

TEST_F(StringsContainsTests, SpecialNewLines)
{
  auto input = cudf::test::strings_column_wrapper({"zzé" LINE_SEPARATOR "qqq" NEXT_LINE "zzé",
                                                   "qqq\rzzé" LINE_SEPARATOR "lll",
                                                   "zzé",
                                                   "",
                                                   "zzé" PARAGRAPH_SEPARATOR,
                                                   "abc\nzzé" NEXT_LINE});
  auto view  = cudf::strings_column_view(input);

  auto pattern = std::string("^zzé$");
  auto prog =
    cudf::strings::regex_program::create(pattern, cudf::strings::regex_flags::EXT_NEWLINE);
  auto ml_flags = static_cast<cudf::strings::regex_flags>(cudf::strings::regex_flags::EXT_NEWLINE |
                                                          cudf::strings::regex_flags::MULTILINE);
  auto prog_ml  = cudf::strings::regex_program::create(pattern, ml_flags);

  auto expected = cudf::test::fixed_width_column_wrapper<bool>({0, 0, 1, 0, 1, 0});
  auto results  = cudf::strings::contains_re(view, *prog);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  expected = cudf::test::fixed_width_column_wrapper<bool>({1, 1, 1, 0, 1, 1});
  results  = cudf::strings::contains_re(view, *prog_ml);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);

  expected = cudf::test::fixed_width_column_wrapper<bool>({0, 0, 1, 0, 1, 0});
  results  = cudf::strings::matches_re(view, *prog);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  expected = cudf::test::fixed_width_column_wrapper<bool>({1, 0, 1, 0, 1, 0});
  results  = cudf::strings::matches_re(view, *prog_ml);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);

  auto counts = cudf::test::fixed_width_column_wrapper<int32_t>({0, 0, 1, 0, 1, 0});
  results     = cudf::strings::count_re(view, *prog);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, counts);
  counts  = cudf::test::fixed_width_column_wrapper<int32_t>({2, 1, 1, 0, 1, 1});
  results = cudf::strings::count_re(view, *prog_ml);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, counts);

  pattern  = std::string("q.*l");
  prog     = cudf::strings::regex_program::create(pattern);
  expected = cudf::test::fixed_width_column_wrapper<bool>({0, 1, 0, 0, 0, 0});
  results  = cudf::strings::contains_re(view, *prog);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  // inst ANY will stop matching on first 'newline' and so should not match anything here
  prog     = cudf::strings::regex_program::create(pattern, cudf::strings::regex_flags::EXT_NEWLINE);
  expected = cudf::test::fixed_width_column_wrapper<bool>({0, 0, 0, 0, 0, 0});
  results  = cudf::strings::contains_re(view, *prog);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  // including the DOTALL flag accepts the newline characters
  auto dot_flags = static_cast<cudf::strings::regex_flags>(cudf::strings::regex_flags::EXT_NEWLINE |
                                                           cudf::strings::regex_flags::DOTALL);
  prog           = cudf::strings::regex_program::create(pattern, dot_flags);
  expected       = cudf::test::fixed_width_column_wrapper<bool>({0, 1, 0, 0, 0, 0});
  results        = cudf::strings::contains_re(view, *prog);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
}

TEST_F(StringsContainsTests, EndOfString)
{
  auto input = cudf::test::strings_column_wrapper(
    {"abé\nfff\nabé", "fff\nabé\nlll", "abé", "", "abé\n", "abe\nabé\n"});
  auto view = cudf::strings_column_view(input);

  auto pattern = std::string("\\Aabé\\Z");
  auto prog    = cudf::strings::regex_program::create(pattern);
  auto prog_ml =
    cudf::strings::regex_program::create(pattern, cudf::strings::regex_flags::MULTILINE);

  auto results  = cudf::strings::contains_re(view, *prog);
  auto expected = cudf::test::fixed_width_column_wrapper<bool>({0, 0, 1, 0, 0, 0});
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results, expected);
  results = cudf::strings::contains_re(view, *prog_ml);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results, expected);

  results = cudf::strings::matches_re(view, *prog);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results, expected);
  results = cudf::strings::matches_re(view, *prog_ml);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results, expected);

  results             = cudf::strings::count_re(view, *prog);
  auto expected_count = cudf::test::fixed_width_column_wrapper<int32_t>({0, 0, 1, 0, 0, 0});
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results, expected_count);
  results = cudf::strings::count_re(view, *prog_ml);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results, expected_count);

  pattern  = std::string("abé$");
  prog     = cudf::strings::regex_program::create(pattern);
  expected = cudf::test::fixed_width_column_wrapper<bool>({1, 0, 1, 0, 1, 1});
  results  = cudf::strings::contains_re(view, *prog);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results, expected);
  prog_ml  = cudf::strings::regex_program::create(pattern, cudf::strings::regex_flags::MULTILINE);
  expected = cudf::test::fixed_width_column_wrapper<bool>({1, 1, 1, 0, 1, 1});
  results  = cudf::strings::contains_re(view, *prog_ml);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results, expected);
}

TEST_F(StringsContainsTests, DotAll)
{
  auto input = cudf::test::strings_column_wrapper({"abc\nfa\nef", "fff\nabbc\nfff", "abcdef", ""});
  auto view  = cudf::strings_column_view(input);

  auto pattern = std::string("a.*f");
  auto prog    = cudf::strings::regex_program::create(pattern);
  auto prog_dotall =
    cudf::strings::regex_program::create(pattern, cudf::strings::regex_flags::DOTALL);

  auto expected_contains = cudf::test::fixed_width_column_wrapper<bool>({1, 1, 1, 0});
  auto results           = cudf::strings::contains_re(view, *prog_dotall);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results, expected_contains);
  expected_contains = cudf::test::fixed_width_column_wrapper<bool>({0, 0, 1, 0});
  results           = cudf::strings::contains_re(view, *prog);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results, expected_contains);

  auto expected_matches = cudf::test::fixed_width_column_wrapper<bool>({1, 0, 1, 0});
  results               = cudf::strings::matches_re(view, *prog_dotall);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results, expected_matches);
  expected_matches = cudf::test::fixed_width_column_wrapper<bool>({0, 0, 1, 0});
  results          = cudf::strings::matches_re(view, *prog);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results, expected_matches);

  pattern     = std::string("a.*?f");
  prog        = cudf::strings::regex_program::create(pattern);
  prog_dotall = cudf::strings::regex_program::create(pattern, cudf::strings::regex_flags::DOTALL);

  auto expected_count = cudf::test::fixed_width_column_wrapper<int32_t>({2, 1, 1, 0});
  results             = cudf::strings::count_re(view, *prog_dotall);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results, expected_count);
  expected_count = cudf::test::fixed_width_column_wrapper<int32_t>({0, 0, 1, 0});
  results        = cudf::strings::count_re(view, *prog);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results, expected_count);

  auto both_flags = static_cast<cudf::strings::regex_flags>(cudf::strings::regex_flags::DOTALL |
                                                            cudf::strings::regex_flags::MULTILINE);
  expected_count  = cudf::test::fixed_width_column_wrapper<int32_t>({2, 1, 1, 0});
  auto prog_both  = cudf::strings::regex_program::create(pattern, both_flags);
  results         = cudf::strings::count_re(view, *prog_both);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results, expected_count);
}

TEST_F(StringsContainsTests, ASCII)
{
  auto input = cudf::test::strings_column_wrapper({"abc \t\f\r 12", "áé 　❽❽", "aZ ❽4", "XYZ　8"});
  auto view  = cudf::strings_column_view(input);

  std::array patterns = {R"(\w+[\s]+\d+)",
                         R"([^\W]+\s+[^\D]+)",
                         R"([\w]+[^\S]+[\d]+)",
                         R"([\w]+\s+[\d]+)",
                         R"(\w+\s+\d+)"};

  for (auto ptn : patterns) {
    auto expected_contains = cudf::test::fixed_width_column_wrapper<bool>({1, 0, 0, 0});
    auto prog    = cudf::strings::regex_program::create(ptn, cudf::strings::regex_flags::ASCII);
    auto results = cudf::strings::contains_re(view, *prog);
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results, expected_contains);

    expected_contains = cudf::test::fixed_width_column_wrapper<bool>({1, 1, 1, 1});
    prog              = cudf::strings::regex_program::create(ptn);
    results           = cudf::strings::contains_re(view, *prog);
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results, expected_contains);
  }
}

TEST_F(StringsContainsTests, IgnoreCase)
{
  auto input = cudf::test::strings_column_wrapper({"abc", "ABC", "aBc", "123áéſ", "ÁÉS123"});
  auto view  = cudf::strings_column_view(input);

  auto expected = cudf::test::fixed_width_column_wrapper<bool>({1, 1, 1, 0, 0});
  auto prog = cudf::strings::regex_program::create("abc", cudf::strings::regex_flags::IGNORECASE);
  auto results = cudf::strings::contains_re(view, *prog);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results, expected);

  expected = cudf::test::fixed_width_column_wrapper<bool>({1, 1, 1, 0, 0});
  prog     = cudf::strings::regex_program::create("[a-c]", cudf::strings::regex_flags::IGNORECASE);
  results  = cudf::strings::contains_re(view, *prog);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results, expected);

  expected = cudf::test::fixed_width_column_wrapper<bool>({0, 0, 0, 1, 1});
  prog     = cudf::strings::regex_program::create("áéſ", cudf::strings::regex_flags::IGNORECASE);
  results  = cudf::strings::contains_re(view, *prog);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results, expected);

  expected = cudf::test::fixed_width_column_wrapper<bool>({0, 0, 0, 1, 1});
  prog     = cudf::strings::regex_program::create("[á-é]", cudf::strings::regex_flags::IGNORECASE);
  results  = cudf::strings::contains_re(view, *prog);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results, expected);
}

TEST_F(StringsContainsTests, MediumRegex)
{
  // This results in 95 regex instructions and falls in the 'medium' range.
  std::string medium_regex =
    "hello @abc @def world The quick brown @fox jumps over the lazy @dog hello "
    "http://www.world.com";
  auto prog = cudf::strings::regex_program::create(medium_regex);

  std::vector<char const*> h_strings{
    "hello @abc @def world The quick brown @fox jumps over the lazy @dog hello "
    "http://www.world.com thats all",
    "1234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234"
    "5678901234567890",
    "abcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyzabcdefghijklmnop"
    "qrstuvwxyzabcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyz"};
  cudf::test::strings_column_wrapper strings(h_strings.begin(), h_strings.end());

  auto strings_view = cudf::strings_column_view(strings);
  {
    auto results = cudf::strings::contains_re(strings_view, *prog);
    cudf::test::fixed_width_column_wrapper<bool> expected({true, false, false});
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results, expected);
  }
  {
    auto results = cudf::strings::matches_re(strings_view, *prog);
    cudf::test::fixed_width_column_wrapper<bool> expected({true, false, false});
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results, expected);
  }
  {
    auto results = cudf::strings::count_re(strings_view, *prog);
    cudf::test::fixed_width_column_wrapper<int32_t> expected({1, 0, 0});
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results, expected);
  }
}

TEST_F(StringsContainsTests, LargeRegex)
{
  // This results in 115 regex instructions and falls in the 'large' range.
  std::string large_regex =
    "hello @abc @def world The quick brown @fox jumps over the lazy @dog hello "
    "http://www.world.com I'm here @home zzzz";
  auto prog = cudf::strings::regex_program::create(large_regex);

  std::vector<char const*> h_strings{
    "hello @abc @def world The quick brown @fox jumps over the lazy @dog hello "
    "http://www.world.com I'm here @home zzzz",
    "1234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234"
    "5678901234567890",
    "abcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyzabcdefghijklmnop"
    "qrstuvwxyzabcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyz"};
  cudf::test::strings_column_wrapper strings(h_strings.begin(), h_strings.end());

  auto strings_view = cudf::strings_column_view(strings);
  {
    auto results = cudf::strings::contains_re(strings_view, *prog);
    cudf::test::fixed_width_column_wrapper<bool> expected({true, false, false});
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results, expected);
  }
  {
    auto results = cudf::strings::matches_re(strings_view, *prog);
    cudf::test::fixed_width_column_wrapper<bool> expected({true, false, false});
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results, expected);
  }
  {
    auto results = cudf::strings::count_re(strings_view, *prog);
    cudf::test::fixed_width_column_wrapper<int32_t> expected({1, 0, 0});
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results, expected);
  }
}

TEST_F(StringsContainsTests, ExtraLargeRegex)
{
  // This results in 321 regex instructions which is above the 'large' range.
  std::string data(320, '0');
  cudf::test::strings_column_wrapper strings({data, data, data, data, data, "00"});
  auto prog = cudf::strings::regex_program::create(data);

  auto strings_view = cudf::strings_column_view(strings);
  {
    auto results = cudf::strings::contains_re(strings_view, *prog);
    cudf::test::fixed_width_column_wrapper<bool> expected({true, true, true, true, true, false});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
  {
    auto results = cudf::strings::matches_re(strings_view, *prog);
    cudf::test::fixed_width_column_wrapper<bool> expected({true, true, true, true, true, false});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
  {
    auto results = cudf::strings::count_re(strings_view, *prog);
    cudf::test::fixed_width_column_wrapper<int32_t> expected({1, 1, 1, 1, 1, 0});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
}

TEST_F(StringsContainsTests, CrlfLineAnchorExtNewline)
{
  // JDK/Spark treat \r\n as a SINGLE line terminator: '$' matches before the \r and
  // never between the \r and \n. Expected values verified against OpenJDK 17
  // java.util.regex (default == EXT non-multiline; Pattern.MULTILINE == EXT|MULTILINE).
  auto input = cudf::test::strings_column_wrapper(
    {"abc\r\n", "abc\n", "abc\r", "abc", "a\r\nb", "abc\r\n\r\n", "", "abc" NEXT_LINE});
  auto view = cudf::strings_column_view(input);

  auto prog =
    cudf::strings::regex_program::create("^abc$", cudf::strings::regex_flags::EXT_NEWLINE);
  auto both    = static_cast<cudf::strings::regex_flags>(cudf::strings::regex_flags::EXT_NEWLINE |
                                                      cudf::strings::regex_flags::MULTILINE);
  auto prog_ml = cudf::strings::regex_program::create("^abc$", both);

  // Java: ^abc$ EXT(non-ml) = {1,1,1,1,0,0,0,1}
  auto expected = cudf::test::fixed_width_column_wrapper<bool>({1, 1, 1, 1, 0, 0, 0, 1});
  auto results  = cudf::strings::contains_re(view, *prog);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);

  // Java: ^abc$ EXT|MULTILINE = {1,1,1,1,0,1,0,1}
  expected = cudf::test::fixed_width_column_wrapper<bool>({1, 1, 1, 1, 0, 1, 0, 1});
  results  = cudf::strings::contains_re(view, *prog_ml);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
}

TEST_F(StringsContainsTests, CrlfBolAnchorExtNewline)
{
  // Symmetric ^ side: with \r\n as one terminator, multiline ^ matches AFTER the \n,
  // never between the \r and \n. So "^\n" must never match (a \n is never a line start).
  // Verified against OpenJDK 17: contains "(?m)^\n" = false for all of these.
  auto input = cudf::test::strings_column_wrapper({"abc\r\nDEF", "a\r\nb", "ab\rc", "x\ny"});
  auto view  = cudf::strings_column_view(input);
  auto both  = static_cast<cudf::strings::regex_flags>(cudf::strings::regex_flags::EXT_NEWLINE |
                                                      cudf::strings::regex_flags::MULTILINE);
  auto prog  = cudf::strings::regex_program::create("^\n", both);

  auto expected = cudf::test::fixed_width_column_wrapper<bool>({0, 0, 0, 0});
  auto results  = cudf::strings::contains_re(view, *prog);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
}

TEST_F(StringsContainsTests, CrlfEdgeCasesExtNewline)
{
  // Comprehensive \r\n / mixed-terminator coverage across contains/matches/count,
  // both EXT_NEWLINE and EXT_NEWLINE|MULTILINE. All expecteds verified vs OpenJDK 17.
  // Each row bundles every expected outcome for one input, so adding an input or a
  // new regex touches a single row or column instead of 8 parallel arrays.
  struct edge_case {
    char const* s;
    bool abc_anchored_en;     // contains "^abc$"   EXT
    bool abc_anchored_ml;     // contains "^abc$"   EXT|MULTILINE
    bool abc_dollar_matches;  // matches_re "abc$"  EXT  (anchored at start of string)
    int az_dollar_count_en;   // count_re "[a-z]+$" EXT
    int az_dollar_count_ml;   // count_re "[a-z]+$" EXT|MULTILINE
    int az_start_count_ml;    // count_re "^[a-z]+" EXT|MULTILINE
    bool cr_dollar_ml;        // contains "\r$"     EXT|MULTILINE  (never inside \r\n)
    bool start_nl_ml;         // contains "^\n"     EXT|MULTILINE  (^ never lands between \r\n)
    bool alt_a_or_b_en;       // contains "(a$|b)"  EXT            (#14856 alternation)
  };
  // clang-format off
  constexpr static edge_case cases[] = {
    // Column: CRLF, lone CR, lone LF, no-term, mid-CRLF, double-CRLF, empty, NEL,
    //         mixed, CRLF-only, leading-CRLF, reversed \n\r, \r\r, \n\n.
    //                  abc^$ abc^$ abc$    [a-z]+$ [a-z]+$ ^[a-z]+ \r$  ^\n  (a$|b)
    //                  EXT   MLX   matches EXT     MLX     MLX     MLX  MLX  EXT
    {"abc\r\n",         1,    1,    1,      1,      1,      1,      0,   0,   1},
    {"abc\n",           1,    1,    1,      1,      1,      1,      0,   0,   1},
    {"abc\r",           1,    1,    1,      1,      1,      1,      1,   0,   1},
    {"abc",             1,    1,    1,      1,      1,      1,      0,   0,   1},
    {"a\r\nb",          0,    0,    0,      1,      2,      2,      0,   0,   1},
    {"abc\r\n\r\n",     0,    1,    0,      0,      1,      1,      0,   0,   1},
    {"",                0,    0,    0,      0,      0,      0,      0,   0,   0},
    {"abc" NEXT_LINE,   1,    1,    1,      1,      1,      1,      0,   0,   1},
    {"a\nb\r\nc",       0,    0,    0,      1,      3,      3,      0,   0,   1},
    {"\r\n",            0,    0,    0,      0,      0,      0,      0,   0,   0},
    {"\r\nabc",         0,    1,    0,      1,      1,      1,      0,   0,   1},
    {"x\n\r",           0,    0,    0,      0,      1,      1,      1,   0,   0},
    {"a\r\rb",          0,    0,    0,      1,      2,      2,      1,   0,   1},
    {"a\n\nb",          0,    0,    0,      1,      2,      2,      0,   1,   1},
  };
  // clang-format on

  auto strings_view = std::span(cases) | std::views::transform(&edge_case::s);
  auto input        = cudf::test::strings_column_wrapper(strings_view.begin(), strings_view.end());
  auto view         = cudf::strings_column_view(input);
  auto const EXT    = cudf::strings::regex_flags::EXT_NEWLINE;
  auto const MLX = static_cast<cudf::strings::regex_flags>(cudf::strings::regex_flags::EXT_NEWLINE |
                                                           cudf::strings::regex_flags::MULTILINE);

  auto to_col = [](auto edge_case::* m) {
    auto v  = std::span(cases) | std::views::transform(m);
    using T = std::remove_cvref_t<decltype(*v.begin())>;
    return cudf::test::fixed_width_column_wrapper<T>(v.begin(), v.end());
  };

  {  // contains ^abc$  (non-multiline / multiline)
    auto p  = cudf::strings::regex_program::create("^abc$", EXT);
    auto pm = cudf::strings::regex_program::create("^abc$", MLX);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*cudf::strings::contains_re(view, *p),
                                   to_col(&edge_case::abc_anchored_en));
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*cudf::strings::contains_re(view, *pm),
                                   to_col(&edge_case::abc_anchored_ml));
  }
  {  // matches_re abc$  (match at start of string)
    auto p = cudf::strings::regex_program::create("abc$", EXT);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*cudf::strings::matches_re(view, *p),
                                   to_col(&edge_case::abc_dollar_matches));
  }
  {  // count_re [a-z]+$  (non-multiline / multiline)
    auto p  = cudf::strings::regex_program::create("[a-z]+$", EXT);
    auto pm = cudf::strings::regex_program::create("[a-z]+$", MLX);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*cudf::strings::count_re(view, *p),
                                   to_col(&edge_case::az_dollar_count_en));
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*cudf::strings::count_re(view, *pm),
                                   to_col(&edge_case::az_dollar_count_ml));
  }
  {  // count_re ^[a-z]+  (multiline line-starts)
    auto pm = cudf::strings::regex_program::create("^[a-z]+", MLX);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*cudf::strings::count_re(view, *pm),
                                   to_col(&edge_case::az_start_count_ml));
  }
  {  // CRLF-coupling discriminators: \r$ never matches inside \r\n; ^\n likewise
    auto pe = cudf::strings::regex_program::create("\\r$", MLX);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*cudf::strings::contains_re(view, *pe),
                                   to_col(&edge_case::cr_dollar_ml));
    auto pb = cudf::strings::regex_program::create("^\n", MLX);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*cudf::strings::contains_re(view, *pb),
                                   to_col(&edge_case::start_nl_ml));
  }
  {  // alternation containing $ (the #14856 construct) works natively, no transpiler
    auto p = cudf::strings::regex_program::create("(a$|b)", EXT);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*cudf::strings::contains_re(view, *p),
                                   to_col(&edge_case::alt_a_or_b_en));
  }
}

TEST_F(StringsContainsTests, CrlfDefaultLfOnlyNoExtNewline)
{
  // Regression guard for the default (non-EXT_NEWLINE) path: WITHOUT EXT_NEWLINE only \n is a
  // line terminator -- \r, \r\n, and NEL are ordinary characters. The CRLF-coupling change must
  // NOT touch this path. Oracle here is cuDF's own "only \n" rule, NOT the JDK (which treats all
  // terminators); so e.g. ^abc$ matches "abc\n"/"abc" but not "abc\r\n"/"abc\r"/"abc"+NEL.
  auto input = cudf::test::strings_column_wrapper({"abc\r\n",
                                                   "abc\n",
                                                   "abc\r",
                                                   "abc",
                                                   "a\r\nb",
                                                   "abc\r\n\r\n",
                                                   "",
                                                   "abc" NEXT_LINE,
                                                   "a\nb\r\nc",
                                                   "\r\n",
                                                   "\r\nabc",
                                                   "x\n\r",
                                                   "a\r\rb",
                                                   "a\n\nb"});
  auto view  = cudf::strings_column_view(input);

  auto prog = cudf::strings::regex_program::create("^abc$");  // default flags: no EXT_NEWLINE
  auto expected =
    cudf::test::fixed_width_column_wrapper<bool>({0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*cudf::strings::contains_re(view, *prog), expected);
}

TEST_F(StringsContainsTests, AlternationNullableBranch)
{
  // "a(bc|de|fg|)h" has an explicit empty branch that makes 'h' directly reachable after 'a'.
  auto input = cudf::test::strings_column_wrapper(
    {"ah", "abch", "adeh", "afghh", "abcde", "a", "h", "", "abcdefgh", "xabchx"});
  auto sv = cudf::strings_column_view(input);

  auto prog = cudf::strings::regex_program::create("a(bc|de|fg|)h");
  {
    auto results = cudf::strings::contains_re(sv, *prog);
    cudf::test::fixed_width_column_wrapper<bool> expected({1, 1, 1, 1, 0, 0, 0, 0, 1, 1});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
  {
    auto results = cudf::strings::count_re(sv, *prog);
    cudf::test::fixed_width_column_wrapper<int32_t> expected({1, 1, 1, 1, 0, 0, 0, 0, 1, 1});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
}

TEST_F(StringsContainsTests, BoundedRepetitionGap)
{
  // "ab{0,4}cv" — 'b' may repeat 0–4 times; five or more b's yield no match.
  auto input = cudf::test::strings_column_wrapper(
    {"acv", "abcv", "abbcv", "abbbcv", "abbbbcv", "abbbbbcv", "av", "acvx", "xacvx", ""});
  auto sv = cudf::strings_column_view(input);

  auto prog    = cudf::strings::regex_program::create("ab{0,4}cv");
  auto results = cudf::strings::contains_re(sv, *prog);
  cudf::test::fixed_width_column_wrapper<bool> expected({1, 1, 1, 1, 1, 0, 0, 1, 1, 0});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
}

TEST_F(StringsContainsTests, ExtNewlineDotAny)
{
  // DEFAULT mode excludes only \n from '.'.
  // EXT_NEWLINE additionally excludes \r, U+0085 (NEL), U+2028 (LS), and U+2029 (PS).
  auto input = cudf::test::strings_column_wrapper({"axb",
                                                   "a\nb",
                                                   "a\rb",
                                                   "a\xc2\x85"
                                                   "b",  // U+0085 NEL between 'a' and 'b'
                                                   "a\xe2\x80\xa8"
                                                   "b",  // U+2028 LINE SEPARATOR
                                                   "a\xe2\x80\xa9"
                                                   "b",  // U+2029 PARAGRAPH SEPARATOR
                                                   "abc",
                                                   ""});
  auto sv    = cudf::strings_column_view(input);

  // DEFAULT: only \n excluded — \r and extended newlines are matched by '.'
  {
    auto prog    = cudf::strings::regex_program::create("a.b");
    auto results = cudf::strings::contains_re(sv, *prog);
    cudf::test::fixed_width_column_wrapper<bool> expected({1, 0, 1, 1, 1, 1, 0, 0});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
  // EXT_NEWLINE: \r and all extended newlines also excluded
  {
    auto prog =
      cudf::strings::regex_program::create("a.b", cudf::strings::regex_flags::EXT_NEWLINE);
    auto results = cudf::strings::contains_re(sv, *prog);
    cudf::test::fixed_width_column_wrapper<bool> expected({1, 0, 0, 0, 0, 0, 0, 0});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
  // A string composed entirely of extended newlines yields no '.+' match under EXT_NEWLINE
  {
    auto input2 = cudf::test::strings_column_wrapper(
      {"hello",
       "\xc2\x85\xe2\x80\xa8\xe2\x80\xa9",  // only extended newlines
       "a\xc2\x85"
       "b",
       ""});
    auto sv2  = cudf::strings_column_view(input2);
    auto prog = cudf::strings::regex_program::create(".+", cudf::strings::regex_flags::EXT_NEWLINE);
    auto results = cudf::strings::contains_re(sv2, *prog);
    cudf::test::fixed_width_column_wrapper<bool> expected2({1, 0, 1, 0});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected2);
  }
}

TEST_F(StringsContainsTests, AlternationPriorityCount)
{
  // Leftmost-first (first-alternative-wins): the shorter first branch is consumed, leaving
  // subsequent characters available for the next match.
  {
    // "a|aa": "a" wins, so "aaaa" counts as 4 individual matches, not 2 "aa" matches.
    auto input   = cudf::test::strings_column_wrapper({"aaaa", "aaaaaa", "aaab", "a", "b", ""});
    auto sv      = cudf::strings_column_view(input);
    auto prog    = cudf::strings::regex_program::create("a|aa");
    auto results = cudf::strings::count_re(sv, *prog);
    cudf::test::fixed_width_column_wrapper<int32_t> expected({4, 6, 3, 1, 0, 0});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
  {
    // "foo|foobar": "foo" wins when both alternatives start at the same position.
    auto input   = cudf::test::strings_column_wrapper({"foo", "foobar", "foofoo", "bar", ""});
    auto sv      = cudf::strings_column_view(input);
    auto prog    = cudf::strings::regex_program::create("foo|foobar");
    auto results = cudf::strings::count_re(sv, *prog);
    cudf::test::fixed_width_column_wrapper<int32_t> expected({1, 1, 2, 0, 0});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
}

TEST_F(StringsContainsTests, LazyQuantifiers)
{
  // Lazy star/plus in non-DOTALL mode: prefer the shortest match.
  auto input = cudf::test::strings_column_wrapper(
    {"ab", "abc", "xdefx", "xghix", "jkl", "abc xdefx xghix jkl"});
  auto sv = cudf::strings_column_view(input);

  {
    auto prog    = cudf::strings::regex_program::create("x.*?x");
    auto results = cudf::strings::contains_re(sv, *prog);
    cudf::test::fixed_width_column_wrapper<bool> expected({0, 0, 1, 1, 0, 1});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
  {
    auto prog    = cudf::strings::regex_program::create("x.+?x");
    auto results = cudf::strings::contains_re(sv, *prog);
    cudf::test::fixed_width_column_wrapper<bool> expected({0, 0, 1, 1, 0, 1});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
}

TEST_F(StringsContainsTests, WhitespaceControlCharacters)
{
  // \x1c-\x1f (file/group/record/unit separators) are classified as whitespace
  // by cudf's codepoint flags table and should match \s.
  auto input =
    cudf::test::strings_column_wrapper({"a\x1c"
                                        "b",
                                        "a\x1d"
                                        "b",
                                        "a\x1e"
                                        "b",
                                        "a\x1f"
                                        "b",
                                        "a b",
                                        "ab"});
  auto sv      = cudf::strings_column_view(input);
  auto prog    = cudf::strings::regex_program::create("a\\sb");
  auto results = cudf::strings::contains_re(sv, *prog);
  cudf::test::fixed_width_column_wrapper<bool> expected({1, 1, 1, 1, 1, 0});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
}
