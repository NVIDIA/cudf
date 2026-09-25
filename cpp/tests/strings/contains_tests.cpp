/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "regex_test_utilities.hpp"
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

template <typename RegexBackend>
struct StringsContainsTests : public cudf::test::BaseFixture {};

TYPED_TEST_SUITE(StringsContainsTests, cudf::test::regex_backends);

struct RegexJitProgramTest : public cudf::test::BaseFixture {};

TEST_F(RegexJitProgramTest, MetadataAndOperationValidation)
{
  auto program =
    cudf::experimental::regex_jit_program::create("(a)",
                                                  cudf::experimental::regex_operation::COUNT,
                                                  {},
                                                  cudf::strings::regex_flags::MULTILINE,
                                                  cudf::strings::capture_groups::NON_CAPTURE);

  EXPECT_EQ(program->pattern(), "(a)");
  EXPECT_EQ(program->flags(), cudf::strings::regex_flags::MULTILINE);
  EXPECT_EQ(program->capture(), cudf::strings::capture_groups::NON_CAPTURE);
  EXPECT_EQ(program->operation(), cudf::experimental::regex_operation::COUNT);
  EXPECT_EQ(program->groups_count(), 1);

  auto input = cudf::test::strings_column_wrapper({"a"});
  EXPECT_THROW(cudf::experimental::contains_re(cudf::strings_column_view{input}, *program),
               cudf::logic_error);
}

TYPED_TEST(StringsContainsTests, ContainsTest)
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
    auto prog    = TypeParam::create(cudf::experimental::regex_operation::CONTAINS, ptn);
    auto results = TypeParam::contains_re(strings_view, *prog);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
}

TYPED_TEST(StringsContainsTests, MatchesTest)
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
    auto prog    = TypeParam::create(cudf::experimental::regex_operation::MATCHES, pattern);
    auto results = TypeParam::matches_re(strings_view, *prog);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
  {
    auto const pattern = std::string("\\d+");
    cudf::test::fixed_width_column_wrapper<bool> expected(
      {false, false, false, true, true, false, false},
      cuda::transform_iterator(h_strings.begin(), [](auto str) { return str != nullptr; }));
    auto prog    = TypeParam::create(cudf::experimental::regex_operation::MATCHES, pattern);
    auto results = TypeParam::matches_re(strings_view, *prog);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
  {
    auto const pattern = std::string("@\\w+");
    cudf::test::fixed_width_column_wrapper<bool> expected(
      {false, false, false, false, false, false, false},
      cuda::transform_iterator(h_strings.begin(), [](auto str) { return str != nullptr; }));
    auto prog    = TypeParam::create(cudf::experimental::regex_operation::MATCHES, pattern);
    auto results = TypeParam::matches_re(strings_view, *prog);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
  {
    auto const pattern = std::string(".*");
    cudf::test::fixed_width_column_wrapper<bool> expected(
      {true, true, true, true, true, false, true},
      cuda::transform_iterator(h_strings.begin(), [](auto str) { return str != nullptr; }));
    auto prog    = TypeParam::create(cudf::experimental::regex_operation::MATCHES, pattern);
    auto results = TypeParam::matches_re(strings_view, *prog);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
}

TYPED_TEST(StringsContainsTests, MatchesIPV4Test)
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
    auto prog    = TypeParam::create(cudf::experimental::regex_operation::MATCHES, pattern);
    auto results = TypeParam::matches_re(strings_view, *prog);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(results->view(), expected);
  }
  {  // is_loopback: 72 instructions
    std::string pattern =
      "^127\\.([0-9]|[1-9][0-9]|1([0-9][0-9])|2([0-4][0-9]|5[0-5]))"
      "\\.([0-9]|[1-9][0-9]|1([0-9][0-9])|2([0-4][0-9]|5[0-5]))"
      "\\.([0-9]|[1-9][0-9]|1([0-9][0-9])|2([0-4][0-9]|5[0-5]))$";
    cudf::test::fixed_width_column_wrapper<bool> expected(
      {false, false, false, false, false, false, false, false, false, true});
    auto prog    = TypeParam::create(cudf::experimental::regex_operation::MATCHES, pattern);
    auto results = TypeParam::matches_re(strings_view, *prog);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(results->view(), expected);
  }
  {  // is_multicast: 79 instructions
    std::string pattern =
      "^(2(2[4-9]|3[0-9]))\\.([0-9]|[1-9][0-9]|1([0-9][0-9])|2([0-4][0-9]|5[0-5]))"
      "\\.([0-9]|[1-9][0-9]|1([0-9][0-9])|2([0-4][0-9]|5[0-5]))"
      "\\.([0-9]|[1-9][0-9]|1([0-9][0-9])|2([0-4][0-9]|5[0-5]))$";
    cudf::test::fixed_width_column_wrapper<bool> expected(
      {false, false, false, false, false, false, true, true, false, false});
    auto prog    = TypeParam::create(cudf::experimental::regex_operation::MATCHES, pattern);
    auto results = TypeParam::matches_re(strings_view, *prog);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(results->view(), expected);
  }
}

TYPED_TEST(StringsContainsTests, OctalTest)
{
  cudf::test::strings_column_wrapper strings({"A3", "B", "CDA3EY", "", "99", "\a\t\r"});
  auto strings_view = cudf::strings_column_view(strings);
  auto expected     = cudf::test::fixed_width_column_wrapper<bool>({1, 0, 1, 0, 0, 0});

  auto pattern = std::string("\\101");
  auto prog    = TypeParam::create(cudf::experimental::regex_operation::CONTAINS, pattern);
  auto results = TypeParam::contains_re(strings_view, *prog);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);

  pattern = std::string("\\1013");
  prog    = TypeParam::create(cudf::experimental::regex_operation::CONTAINS, pattern);
  results = TypeParam::contains_re(strings_view, *prog);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);

  pattern = std::string("D*\\101\\063");
  prog    = TypeParam::create(cudf::experimental::regex_operation::CONTAINS, pattern);
  results = TypeParam::contains_re(strings_view, *prog);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);

  pattern  = std::string("\\719");
  expected = cudf::test::fixed_width_column_wrapper<bool>({0, 0, 0, 0, 1, 0});
  prog     = TypeParam::create(cudf::experimental::regex_operation::CONTAINS, pattern);
  results  = TypeParam::contains_re(strings_view, *prog);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);

  pattern  = std::string(R"([\7][\11][\15])");
  expected = cudf::test::fixed_width_column_wrapper<bool>({0, 0, 0, 0, 0, 1});
  prog     = TypeParam::create(cudf::experimental::regex_operation::CONTAINS, pattern);
  results  = TypeParam::contains_re(strings_view, *prog);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
}

TYPED_TEST(StringsContainsTests, HexTest)
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
    auto prog    = TypeParam::create(cudf::experimental::regex_operation::CONTAINS, pattern);
    auto results = TypeParam::contains_re(strings_view, *prog);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);

    // also test hex character appearing in character class brackets
    pattern = "[" + pattern + "]";
    prog    = TypeParam::create(cudf::experimental::regex_operation::CONTAINS, pattern);
    results = TypeParam::contains_re(strings_view, *prog);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
}

TYPED_TEST(StringsContainsTests, EmbeddedNullCharacter)
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
  auto prog     = TypeParam::create(cudf::experimental::regex_operation::CONTAINS, pattern);
  auto results  = TypeParam::contains_re(strings_view, *prog);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);

  pattern  = std::string("B");
  expected = cudf::test::fixed_width_column_wrapper<bool>({1, 1, 1, 1, 1, 1, 1, 1, 1, 1});
  prog     = TypeParam::create(cudf::experimental::regex_operation::CONTAINS, pattern);
  results  = TypeParam::contains_re(strings_view, *prog);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);

  pattern  = std::string("J\\0B");
  expected = cudf::test::fixed_width_column_wrapper<bool>({0, 0, 0, 0, 0, 0, 0, 0, 0, 1});
  prog     = TypeParam::create(cudf::experimental::regex_operation::CONTAINS, pattern);
  results  = TypeParam::contains_re(strings_view, *prog);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);

  pattern  = std::string("[G-J][\\0]B");
  expected = cudf::test::fixed_width_column_wrapper<bool>({0, 0, 0, 0, 0, 0, 1, 1, 1, 1});
  prog     = TypeParam::create(cudf::experimental::regex_operation::CONTAINS, pattern);
  results  = TypeParam::contains_re(strings_view, *prog);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);

  pattern  = std::string("[A-D][\\x00]B");
  expected = cudf::test::fixed_width_column_wrapper<bool>({1, 1, 1, 1, 0, 0, 0, 0, 0, 0});
  prog     = TypeParam::create(cudf::experimental::regex_operation::CONTAINS, pattern);
  results  = TypeParam::contains_re(strings_view, *prog);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
}

TYPED_TEST(StringsContainsTests, Errors)
{
  EXPECT_THROW(TypeParam::create(cudf::experimental::regex_operation::CONTAINS, "(3?)+"),
               cudf::logic_error);
  EXPECT_THROW(TypeParam::create(cudf::experimental::regex_operation::CONTAINS, "(?:3?)+"),
               cudf::logic_error);
  EXPECT_THROW(TypeParam::create(cudf::experimental::regex_operation::CONTAINS, "3?+"),
               cudf::logic_error);
  EXPECT_THROW(TypeParam::create(cudf::experimental::regex_operation::CONTAINS, "{3}a"),
               cudf::logic_error);

  EXPECT_THROW(TypeParam::create(cudf::experimental::regex_operation::CONTAINS, "aaaa{1234,5678}"),
               cudf::logic_error);
  EXPECT_THROW(TypeParam::create(cudf::experimental::regex_operation::CONTAINS, "aaaa{123,5678}"),
               cudf::logic_error);

  EXPECT_THROW(TypeParam::create(cudf::experimental::regex_operation::CONTAINS, "[a-C]"),
               cudf::logic_error);
}

TYPED_TEST(StringsContainsTests, CountTest)
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
    auto prog    = TypeParam::create(cudf::experimental::regex_operation::COUNT, pattern);
    auto results = TypeParam::count_re(strings_view, *prog);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
  {
    auto pattern = std::string("@\\w+");
    cudf::test::fixed_width_column_wrapper<int32_t> expected(
      {1, 1, 0, 0, 0, 0}, cudf::test::iterators::nulls_from_nullptrs(h_strings));
    auto prog    = TypeParam::create(cudf::experimental::regex_operation::COUNT, pattern);
    auto results = TypeParam::count_re(strings_view, *prog);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
  {
    auto pattern = std::string("\\d+:\\d+");
    cudf::test::fixed_width_column_wrapper<int32_t> expected(
      {0, 0, 2, 1, 0, 0}, cudf::test::iterators::nulls_from_nullptrs(h_strings));
    auto prog    = TypeParam::create(cudf::experimental::regex_operation::COUNT, pattern);
    auto results = TypeParam::count_re(strings_view, *prog);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
  {
    auto pattern = std::string("o");
    cudf::test::fixed_width_column_wrapper<int32_t> expected(
      {3, 1, 0, 0, 0, 0}, cudf::test::iterators::nulls_from_nullptrs(h_strings));
    auto prog    = TypeParam::create(cudf::experimental::regex_operation::COUNT, pattern);
    auto results = TypeParam::count_re(strings_view, *prog);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
  {
    auto pattern = std::string("\\blazy\\b");
    cudf::test::fixed_width_column_wrapper<int32_t> expected(
      {0, 2, 0, 0, 0, 0}, cudf::test::iterators::nulls_from_nullptrs(h_strings));
    auto prog    = TypeParam::create(cudf::experimental::regex_operation::COUNT, pattern);
    auto results = TypeParam::count_re(strings_view, *prog);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
}

TYPED_TEST(StringsContainsTests, CountEmptyMatching)
{
  auto input    = cudf::test::strings_column_wrapper({"hello", "world", "", "abc"});
  auto sv       = cudf::strings_column_view(input);
  auto patterns = std::vector<std::string>{"a*", "X?", "b{0,}", "()", "(?:)", "[A-Z]*"};
  auto expected = cudf::test::fixed_width_column_wrapper<int32_t>({6, 6, 1, 4});
  for (auto pattern : patterns) {
    auto prog    = TypeParam::create(cudf::experimental::regex_operation::COUNT, pattern);
    auto results = TypeParam::count_re(sv, *prog);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
  // "\\b", "\\B",
  expected     = cudf::test::fixed_width_column_wrapper<int32_t>({1, 1, 1, 1});
  auto prog    = TypeParam::create(cudf::experimental::regex_operation::COUNT, "^");
  auto results = TypeParam::count_re(sv, *prog);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  prog    = TypeParam::create(cudf::experimental::regex_operation::COUNT, "$");
  results = TypeParam::count_re(sv, *prog);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);

  expected = cudf::test::fixed_width_column_wrapper<int32_t>({0, 0, 1, 0});
  prog     = TypeParam::create(cudf::experimental::regex_operation::COUNT, "^$");
  results  = TypeParam::count_re(sv, *prog);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);

  expected = cudf::test::fixed_width_column_wrapper<int32_t>({2, 2, 0, 2});
  prog     = TypeParam::create(cudf::experimental::regex_operation::COUNT, "\\b");
  results  = TypeParam::count_re(sv, *prog);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);

  expected = cudf::test::fixed_width_column_wrapper<int32_t>({4, 4, 1, 2});
  prog     = TypeParam::create(cudf::experimental::regex_operation::COUNT, "\\B");
  results  = TypeParam::count_re(sv, *prog);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
}

TYPED_TEST(StringsContainsTests, FixedQuantifier)
{
  auto input = cudf::test::strings_column_wrapper({"a", "aa", "aaa", "aaaa", "aaaaa", "aaaaaa"});
  auto sv    = cudf::strings_column_view(input);

  {
    // exact match
    auto pattern = std::string("a{3}");
    cudf::test::fixed_width_column_wrapper<int32_t> expected({0, 0, 1, 1, 1, 2});
    auto prog    = TypeParam::create(cudf::experimental::regex_operation::COUNT, pattern);
    auto results = TypeParam::count_re(sv, *prog);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
  {
    // range match (greedy quantifier)
    auto pattern = std::string("a{3,5}");
    cudf::test::fixed_width_column_wrapper<int32_t> expected({0, 0, 1, 1, 1, 1});
    auto prog    = TypeParam::create(cudf::experimental::regex_operation::COUNT, pattern);
    auto results = TypeParam::count_re(sv, *prog);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
  {
    // minimum match (greedy quantifier)
    auto pattern = std::string("a{2,}");
    cudf::test::fixed_width_column_wrapper<int32_t> expected({0, 1, 1, 1, 1, 1});
    auto prog    = TypeParam::create(cudf::experimental::regex_operation::COUNT, pattern);
    auto results = TypeParam::count_re(sv, *prog);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
  {
    // range match (lazy quantifier)
    auto pattern = std::string("a{2,4}?");
    cudf::test::fixed_width_column_wrapper<int32_t> expected({0, 1, 1, 2, 2, 3});
    auto prog    = TypeParam::create(cudf::experimental::regex_operation::COUNT, pattern);
    auto results = TypeParam::count_re(sv, *prog);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
  {
    // minimum match (lazy quantifier)
    auto pattern = std::string("a{1,}?");
    cudf::test::fixed_width_column_wrapper<int32_t> expected({1, 2, 3, 4, 5, 6});
    auto prog    = TypeParam::create(cudf::experimental::regex_operation::COUNT, pattern);
    auto results = TypeParam::count_re(sv, *prog);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
  {
    // zero match
    auto pattern = std::string("aaaa{0}");
    cudf::test::fixed_width_column_wrapper<int32_t> expected({0, 0, 1, 1, 1, 2});
    auto prog    = TypeParam::create(cudf::experimental::regex_operation::COUNT, pattern);
    auto results = TypeParam::count_re(sv, *prog);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
  {
    // poorly formed
    auto pattern = std::string("aaaa{n,m}");
    cudf::test::fixed_width_column_wrapper<int32_t> expected({0, 0, 0, 0, 0, 0});
    auto prog    = TypeParam::create(cudf::experimental::regex_operation::COUNT, pattern);
    auto results = TypeParam::count_re(sv, *prog);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
}

TYPED_TEST(StringsContainsTests, ZeroRangeQuantifier)
{
  EXPECT_NO_THROW(TypeParam::create(cudf::experimental::regex_operation::CONTAINS, "a{0}"));
  EXPECT_NO_THROW(TypeParam::create(cudf::experimental::regex_operation::CONTAINS, "a{0,1}"));
  EXPECT_NO_THROW(TypeParam::create(cudf::experimental::regex_operation::CONTAINS, "a{0,}"));
  EXPECT_NO_THROW(TypeParam::create(cudf::experimental::regex_operation::CONTAINS, "(ab){0}"));
  EXPECT_NO_THROW(TypeParam::create(cudf::experimental::regex_operation::CONTAINS, "(ab){0,1}"));
  EXPECT_NO_THROW(TypeParam::create(cudf::experimental::regex_operation::CONTAINS, "(ab){0,}"));

  auto input = cudf::test::strings_column_wrapper({"a", "", "abc", "XYAZ", "ABC", "ZYXA"});
  auto sv    = cudf::strings_column_view(input);

  auto pattern = std::string("A{0,}");  // should match everyting
  auto prog    = TypeParam::create(cudf::experimental::regex_operation::CONTAINS, pattern);

  {
    auto expected = cudf::test::fixed_width_column_wrapper<bool>({1, 1, 1, 1, 1, 1});
    auto results  = TypeParam::contains_re(sv, *prog);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
  {
    auto expected = cudf::test::fixed_width_column_wrapper<cudf::size_type>({2, 1, 4, 5, 4, 5});
    prog          = TypeParam::create(cudf::experimental::regex_operation::COUNT, pattern);
    auto results  = TypeParam::count_re(sv, *prog);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }

  pattern = std::string("(?:ab){0,3}");
  prog    = TypeParam::create(cudf::experimental::regex_operation::CONTAINS, pattern);

  {
    auto expected = cudf::test::fixed_width_column_wrapper<bool>({1, 1, 1, 1, 1, 1});
    auto results  = TypeParam::contains_re(sv, *prog);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
  {
    auto expected = cudf::test::fixed_width_column_wrapper<cudf::size_type>({2, 1, 3, 5, 4, 5});
    prog          = TypeParam::create(cudf::experimental::regex_operation::COUNT, pattern);
    auto results  = TypeParam::count_re(sv, *prog);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
}

TYPED_TEST(StringsContainsTests, NestedQuantifier)
{
  auto input   = cudf::test::strings_column_wrapper({"TEST12 1111 2222 3333 4444 5555",
                                                     "0000 AAAA 9999 BBBB 8888",
                                                     "7777 6666 4444 3333",
                                                     "12345 3333 4444 1111 ABCD"});
  auto sv      = cudf::strings_column_view(input);
  auto pattern = std::string(R"((\d{4}\s){4})");
  cudf::test::fixed_width_column_wrapper<bool> expected({true, false, false, true});
  auto prog    = TypeParam::create(cudf::experimental::regex_operation::CONTAINS, pattern);
  auto results = TypeParam::contains_re(sv, *prog);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
}

TYPED_TEST(StringsContainsTests, DeeplyNestedNoStackOverflow)
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
  auto prog    = TypeParam::create(cudf::experimental::regex_operation::CONTAINS, pattern);
  auto results = TypeParam::contains_re(sv, *prog);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
}

TYPED_TEST(StringsContainsTests, QuantifierErrors)
{
  EXPECT_THROW(TypeParam::create(cudf::experimental::regex_operation::CONTAINS, "^+"),
               cudf::logic_error);
  EXPECT_THROW(TypeParam::create(cudf::experimental::regex_operation::CONTAINS, "$+"),
               cudf::logic_error);
  EXPECT_THROW(TypeParam::create(cudf::experimental::regex_operation::CONTAINS, "(^)+"),
               cudf::logic_error);
  EXPECT_THROW(TypeParam::create(cudf::experimental::regex_operation::CONTAINS, "($)+"),
               cudf::logic_error);
  EXPECT_THROW(TypeParam::create(cudf::experimental::regex_operation::CONTAINS, "\\A+"),
               cudf::logic_error);
  EXPECT_THROW(TypeParam::create(cudf::experimental::regex_operation::CONTAINS, "\\Z+"),
               cudf::logic_error);
  EXPECT_THROW(TypeParam::create(cudf::experimental::regex_operation::CONTAINS, "(\\A)+"),
               cudf::logic_error);
  EXPECT_THROW(TypeParam::create(cudf::experimental::regex_operation::CONTAINS, "(\\Z)+"),
               cudf::logic_error);

  EXPECT_THROW(TypeParam::create(cudf::experimental::regex_operation::CONTAINS, "(^($))+"),
               cudf::logic_error);
  EXPECT_NO_THROW(TypeParam::create(cudf::experimental::regex_operation::CONTAINS, "(^a($))+"));
  EXPECT_NO_THROW(TypeParam::create(cudf::experimental::regex_operation::CONTAINS, "(^(a$))+"));
}

TYPED_TEST(StringsContainsTests, OverlappedClasses)
{
  auto input = cudf::test::strings_column_wrapper({"abcdefg", "defghí", "", "éééééé", "ghijkl"});
  auto sv    = cudf::strings_column_view(input);

  {
    auto pattern = std::string("[e-gb-da-c]");
    cudf::test::fixed_width_column_wrapper<int32_t> expected({7, 4, 0, 0, 1});
    auto prog    = TypeParam::create(cudf::experimental::regex_operation::COUNT, pattern);
    auto results = TypeParam::count_re(sv, *prog);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
  {
    auto pattern = std::string("[á-éê-ú]");
    cudf::test::fixed_width_column_wrapper<int32_t> expected({0, 1, 0, 6, 0});
    auto prog    = TypeParam::create(cudf::experimental::regex_operation::COUNT, pattern);
    auto results = TypeParam::count_re(sv, *prog);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
}

TYPED_TEST(StringsContainsTests, NegatedClasses)
{
  auto input = cudf::test::strings_column_wrapper({"abcdefg", "def\tghí", "", "éeé\néeé", "ABC"});
  auto sv    = cudf::strings_column_view(input);

  {
    auto pattern = std::string("[^a-f]");
    cudf::test::fixed_width_column_wrapper<int32_t> expected({1, 4, 0, 5, 3});
    auto prog    = TypeParam::create(cudf::experimental::regex_operation::COUNT, pattern);
    auto results = TypeParam::count_re(sv, *prog);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
  {
    auto pattern = std::string("[^a-eá-é]");
    cudf::test::fixed_width_column_wrapper<int32_t> expected({2, 5, 0, 1, 3});
    auto prog    = TypeParam::create(cudf::experimental::regex_operation::COUNT, pattern);
    auto results = TypeParam::count_re(sv, *prog);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
}

TYPED_TEST(StringsContainsTests, IncompleteClassesRange)
{
  auto input = cudf::test::strings_column_wrapper({"abc-def", "---", "", "ghijkl", "-wxyz-"});
  auto sv    = cudf::strings_column_view(input);

  {
    auto expected = cudf::test::fixed_width_column_wrapper<bool>({1, 0, 0, 1, 1});
    auto prog     = TypeParam::create(cudf::experimental::regex_operation::CONTAINS, "[a-z]");
    auto results  = TypeParam::contains_re(sv, *prog);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);

    expected = cudf::test::fixed_width_column_wrapper<bool>({1, 1, 0, 1, 1});
    prog     = TypeParam::create(cudf::experimental::regex_operation::CONTAINS,
                             "[a-m-z]");  // same as [a-mz-]
    results  = TypeParam::contains_re(sv, *prog);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);

    expected = cudf::test::fixed_width_column_wrapper<bool>({1, 1, 0, 0, 1});
    prog     = TypeParam::create(cudf::experimental::regex_operation::CONTAINS,
                             "[a-f-q-z]");  // same as [a-fq-z-]
    results  = TypeParam::contains_re(sv, *prog);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);

    expected = cudf::test::fixed_width_column_wrapper<bool>({1, 1, 0, 1, 1});
    prog     = TypeParam::create(cudf::experimental::regex_operation::CONTAINS,
                             "[g-g-z]");  // same as [gz-]
    results  = TypeParam::contains_re(sv, *prog);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
  {
    cudf::test::fixed_width_column_wrapper<bool> expected({1, 1, 0, 1, 1});
    auto prog    = TypeParam::create(cudf::experimental::regex_operation::CONTAINS, "[g-]");
    auto results = TypeParam::contains_re(sv, *prog);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);

    prog    = TypeParam::create(cudf::experimental::regex_operation::CONTAINS, "[-k]");
    results = TypeParam::contains_re(sv, *prog);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
  {
    cudf::test::fixed_width_column_wrapper<bool> expected({1, 1, 0, 0, 1});
    auto prog    = TypeParam::create(cudf::experimental::regex_operation::CONTAINS, "[-]");
    auto results = TypeParam::contains_re(sv, *prog);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
    prog    = TypeParam::create(cudf::experimental::regex_operation::CONTAINS, "[+--]");
    results = TypeParam::contains_re(sv, *prog);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);

    prog    = TypeParam::create(cudf::experimental::regex_operation::CONTAINS, "[a-c-]");
    results = TypeParam::contains_re(sv, *prog);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
    prog    = TypeParam::create(cudf::experimental::regex_operation::CONTAINS, "[-d-f]");
    results = TypeParam::contains_re(sv, *prog);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
}

TYPED_TEST(StringsContainsTests, MultiLine)
{
  auto input = cudf::test::strings_column_wrapper(
    {"abé\nfff\nabé", "fff\nabé\nlll", "abé", "", "abé\n", "abe\nabé\n"});
  auto view = cudf::strings_column_view(input);

  auto pattern = std::string("^abé$");
  auto prog    = TypeParam::create(cudf::experimental::regex_operation::CONTAINS, pattern);
  auto prog_ml = TypeParam::create(
    cudf::experimental::regex_operation::CONTAINS, pattern, cudf::strings::regex_flags::MULTILINE);

  auto expected_contains = cudf::test::fixed_width_column_wrapper<bool>({1, 1, 1, 0, 1, 1});
  auto results           = TypeParam::contains_re(view, *prog_ml);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results, expected_contains);
  expected_contains = cudf::test::fixed_width_column_wrapper<bool>({0, 0, 1, 0, 1, 0});
  results           = TypeParam::contains_re(view, *prog);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results, expected_contains);

  prog    = TypeParam::create(cudf::experimental::regex_operation::MATCHES, pattern);
  prog_ml = TypeParam::create(
    cudf::experimental::regex_operation::MATCHES, pattern, cudf::strings::regex_flags::MULTILINE);

  auto expected_matches = cudf::test::fixed_width_column_wrapper<bool>({1, 0, 1, 0, 1, 0});
  results               = TypeParam::matches_re(view, *prog_ml);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results, expected_matches);
  expected_matches = cudf::test::fixed_width_column_wrapper<bool>({0, 0, 1, 0, 1, 0});
  results          = TypeParam::matches_re(view, *prog);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results, expected_matches);

  prog    = TypeParam::create(cudf::experimental::regex_operation::COUNT, pattern);
  prog_ml = TypeParam::create(
    cudf::experimental::regex_operation::COUNT, pattern, cudf::strings::regex_flags::MULTILINE);

  auto expected_count = cudf::test::fixed_width_column_wrapper<int32_t>({2, 1, 1, 0, 1, 1});
  results             = TypeParam::count_re(view, *prog_ml);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results, expected_count);
  expected_count = cudf::test::fixed_width_column_wrapper<int32_t>({0, 0, 1, 0, 1, 0});
  results        = TypeParam::count_re(view, *prog);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results, expected_count);
}

TYPED_TEST(StringsContainsTests, SpecialNewLines)
{
  auto input = cudf::test::strings_column_wrapper({"zzé" LINE_SEPARATOR "qqq" NEXT_LINE "zzé",
                                                   "qqq\rzzé" LINE_SEPARATOR "lll",
                                                   "zzé",
                                                   "",
                                                   "zzé" PARAGRAPH_SEPARATOR,
                                                   "abc\nzzé" NEXT_LINE});
  auto view  = cudf::strings_column_view(input);

  auto pattern  = std::string("^zzé$");
  auto prog     = TypeParam::create(cudf::experimental::regex_operation::CONTAINS,
                                pattern,
                                cudf::strings::regex_flags::EXT_NEWLINE);
  auto ml_flags = static_cast<cudf::strings::regex_flags>(cudf::strings::regex_flags::EXT_NEWLINE |
                                                          cudf::strings::regex_flags::MULTILINE);
  auto prog_ml =
    TypeParam::create(cudf::experimental::regex_operation::CONTAINS, pattern, ml_flags);

  auto expected = cudf::test::fixed_width_column_wrapper<bool>({0, 0, 1, 0, 1, 0});
  auto results  = TypeParam::contains_re(view, *prog);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  expected = cudf::test::fixed_width_column_wrapper<bool>({1, 1, 1, 0, 1, 1});
  results  = TypeParam::contains_re(view, *prog_ml);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);

  prog = TypeParam::create(
    cudf::experimental::regex_operation::MATCHES, pattern, cudf::strings::regex_flags::EXT_NEWLINE);
  prog_ml = TypeParam::create(cudf::experimental::regex_operation::MATCHES, pattern, ml_flags);

  expected = cudf::test::fixed_width_column_wrapper<bool>({0, 0, 1, 0, 1, 0});
  results  = TypeParam::matches_re(view, *prog);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  expected = cudf::test::fixed_width_column_wrapper<bool>({1, 0, 1, 0, 1, 0});
  results  = TypeParam::matches_re(view, *prog_ml);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);

  prog = TypeParam::create(
    cudf::experimental::regex_operation::COUNT, pattern, cudf::strings::regex_flags::EXT_NEWLINE);
  prog_ml = TypeParam::create(cudf::experimental::regex_operation::COUNT, pattern, ml_flags);

  auto counts = cudf::test::fixed_width_column_wrapper<int32_t>({0, 0, 1, 0, 1, 0});
  results     = TypeParam::count_re(view, *prog);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, counts);
  counts  = cudf::test::fixed_width_column_wrapper<int32_t>({2, 1, 1, 0, 1, 1});
  results = TypeParam::count_re(view, *prog_ml);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, counts);

  pattern  = std::string("q.*l");
  prog     = TypeParam::create(cudf::experimental::regex_operation::CONTAINS, pattern);
  expected = cudf::test::fixed_width_column_wrapper<bool>({0, 1, 0, 0, 0, 0});
  results  = TypeParam::contains_re(view, *prog);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  // inst ANY will stop matching on first 'newline' and so should not match anything here
  prog     = TypeParam::create(cudf::experimental::regex_operation::CONTAINS,
                           pattern,
                           cudf::strings::regex_flags::EXT_NEWLINE);
  expected = cudf::test::fixed_width_column_wrapper<bool>({0, 0, 0, 0, 0, 0});
  results  = TypeParam::contains_re(view, *prog);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  // including the DOTALL flag accepts the newline characters
  auto dot_flags = static_cast<cudf::strings::regex_flags>(cudf::strings::regex_flags::EXT_NEWLINE |
                                                           cudf::strings::regex_flags::DOTALL);
  prog     = TypeParam::create(cudf::experimental::regex_operation::CONTAINS, pattern, dot_flags);
  expected = cudf::test::fixed_width_column_wrapper<bool>({0, 1, 0, 0, 0, 0});
  results  = TypeParam::contains_re(view, *prog);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
}

TYPED_TEST(StringsContainsTests, EndOfString)
{
  auto input = cudf::test::strings_column_wrapper(
    {"abé\nfff\nabé", "fff\nabé\nlll", "abé", "", "abé\n", "abe\nabé\n"});
  auto view = cudf::strings_column_view(input);

  auto pattern = std::string("\\Aabé\\Z");
  auto prog    = TypeParam::create(cudf::experimental::regex_operation::CONTAINS, pattern);
  auto prog_ml = TypeParam::create(
    cudf::experimental::regex_operation::CONTAINS, pattern, cudf::strings::regex_flags::MULTILINE);

  auto results  = TypeParam::contains_re(view, *prog);
  auto expected = cudf::test::fixed_width_column_wrapper<bool>({0, 0, 1, 0, 0, 0});
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results, expected);
  results = TypeParam::contains_re(view, *prog_ml);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results, expected);

  prog    = TypeParam::create(cudf::experimental::regex_operation::MATCHES, pattern);
  prog_ml = TypeParam::create(
    cudf::experimental::regex_operation::MATCHES, pattern, cudf::strings::regex_flags::MULTILINE);
  results = TypeParam::matches_re(view, *prog);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results, expected);
  results = TypeParam::matches_re(view, *prog_ml);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results, expected);

  prog    = TypeParam::create(cudf::experimental::regex_operation::COUNT, pattern);
  prog_ml = TypeParam::create(
    cudf::experimental::regex_operation::COUNT, pattern, cudf::strings::regex_flags::MULTILINE);
  results             = TypeParam::count_re(view, *prog);
  auto expected_count = cudf::test::fixed_width_column_wrapper<int32_t>({0, 0, 1, 0, 0, 0});
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results, expected_count);
  results = TypeParam::count_re(view, *prog_ml);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results, expected_count);

  pattern  = std::string("abé$");
  prog     = TypeParam::create(cudf::experimental::regex_operation::CONTAINS, pattern);
  expected = cudf::test::fixed_width_column_wrapper<bool>({1, 0, 1, 0, 1, 1});
  results  = TypeParam::contains_re(view, *prog);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results, expected);
  prog_ml = TypeParam::create(
    cudf::experimental::regex_operation::CONTAINS, pattern, cudf::strings::regex_flags::MULTILINE);
  expected = cudf::test::fixed_width_column_wrapper<bool>({1, 1, 1, 0, 1, 1});
  results  = TypeParam::contains_re(view, *prog_ml);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results, expected);
}

TYPED_TEST(StringsContainsTests, DotAll)
{
  auto input = cudf::test::strings_column_wrapper({"abc\nfa\nef", "fff\nabbc\nfff", "abcdef", ""});
  auto view  = cudf::strings_column_view(input);

  auto pattern     = std::string("a.*f");
  auto prog        = TypeParam::create(cudf::experimental::regex_operation::CONTAINS, pattern);
  auto prog_dotall = TypeParam::create(
    cudf::experimental::regex_operation::CONTAINS, pattern, cudf::strings::regex_flags::DOTALL);

  auto expected_contains = cudf::test::fixed_width_column_wrapper<bool>({1, 1, 1, 0});
  auto results           = TypeParam::contains_re(view, *prog_dotall);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results, expected_contains);
  expected_contains = cudf::test::fixed_width_column_wrapper<bool>({0, 0, 1, 0});
  results           = TypeParam::contains_re(view, *prog);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results, expected_contains);

  prog        = TypeParam::create(cudf::experimental::regex_operation::MATCHES, pattern);
  prog_dotall = TypeParam::create(
    cudf::experimental::regex_operation::MATCHES, pattern, cudf::strings::regex_flags::DOTALL);

  auto expected_matches = cudf::test::fixed_width_column_wrapper<bool>({1, 0, 1, 0});
  results               = TypeParam::matches_re(view, *prog_dotall);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results, expected_matches);
  expected_matches = cudf::test::fixed_width_column_wrapper<bool>({0, 0, 1, 0});
  results          = TypeParam::matches_re(view, *prog);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results, expected_matches);

  pattern     = std::string("a.*?f");
  prog        = TypeParam::create(cudf::experimental::regex_operation::COUNT, pattern);
  prog_dotall = TypeParam::create(
    cudf::experimental::regex_operation::COUNT, pattern, cudf::strings::regex_flags::DOTALL);

  auto expected_count = cudf::test::fixed_width_column_wrapper<int32_t>({2, 1, 1, 0});
  results             = TypeParam::count_re(view, *prog_dotall);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results, expected_count);
  expected_count = cudf::test::fixed_width_column_wrapper<int32_t>({0, 0, 1, 0});
  results        = TypeParam::count_re(view, *prog);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results, expected_count);

  auto both_flags = static_cast<cudf::strings::regex_flags>(cudf::strings::regex_flags::DOTALL |
                                                            cudf::strings::regex_flags::MULTILINE);
  expected_count  = cudf::test::fixed_width_column_wrapper<int32_t>({2, 1, 1, 0});
  auto prog_both =
    TypeParam::create(cudf::experimental::regex_operation::COUNT, pattern, both_flags);
  results = TypeParam::count_re(view, *prog_both);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results, expected_count);
}

TYPED_TEST(StringsContainsTests, ASCII)
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
    auto prog              = TypeParam::create(
      cudf::experimental::regex_operation::CONTAINS, ptn, cudf::strings::regex_flags::ASCII);
    auto results = TypeParam::contains_re(view, *prog);
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results, expected_contains);

    expected_contains = cudf::test::fixed_width_column_wrapper<bool>({1, 1, 1, 1});
    prog              = TypeParam::create(cudf::experimental::regex_operation::CONTAINS, ptn);
    results           = TypeParam::contains_re(view, *prog);
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results, expected_contains);
  }
}

TYPED_TEST(StringsContainsTests, IgnoreCase)
{
  auto input = cudf::test::strings_column_wrapper({"abc", "ABC", "aBc", "123áéſ", "ÁÉS123"});
  auto view  = cudf::strings_column_view(input);

  auto expected = cudf::test::fixed_width_column_wrapper<bool>({1, 1, 1, 0, 0});
  auto prog     = TypeParam::create(
    cudf::experimental::regex_operation::CONTAINS, "abc", cudf::strings::regex_flags::IGNORECASE);
  auto results = TypeParam::contains_re(view, *prog);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results, expected);

  expected = cudf::test::fixed_width_column_wrapper<bool>({1, 1, 1, 0, 0});
  prog     = TypeParam::create(
    cudf::experimental::regex_operation::CONTAINS, "[a-c]", cudf::strings::regex_flags::IGNORECASE);
  results = TypeParam::contains_re(view, *prog);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results, expected);

  expected = cudf::test::fixed_width_column_wrapper<bool>({0, 0, 0, 1, 1});
  prog     = TypeParam::create(
    cudf::experimental::regex_operation::CONTAINS, "áéſ", cudf::strings::regex_flags::IGNORECASE);
  results = TypeParam::contains_re(view, *prog);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results, expected);

  expected = cudf::test::fixed_width_column_wrapper<bool>({0, 0, 0, 1, 1});
  prog     = TypeParam::create(
    cudf::experimental::regex_operation::CONTAINS, "[á-é]", cudf::strings::regex_flags::IGNORECASE);
  results = TypeParam::contains_re(view, *prog);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results, expected);
}

TYPED_TEST(StringsContainsTests, MediumRegex)
{
  // This results in 95 regex instructions and falls in the 'medium' range.
  std::string medium_regex =
    "hello @abc @def world The quick brown @fox jumps over the lazy @dog hello "
    "http://www.world.com";
  auto prog = TypeParam::create(cudf::experimental::regex_operation::CONTAINS, medium_regex);

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
    auto results = TypeParam::contains_re(strings_view, *prog);
    cudf::test::fixed_width_column_wrapper<bool> expected({true, false, false});
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results, expected);
  }
  prog = TypeParam::create(cudf::experimental::regex_operation::MATCHES, medium_regex);
  {
    auto results = TypeParam::matches_re(strings_view, *prog);
    cudf::test::fixed_width_column_wrapper<bool> expected({true, false, false});
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results, expected);
  }
  prog = TypeParam::create(cudf::experimental::regex_operation::COUNT, medium_regex);
  {
    auto results = TypeParam::count_re(strings_view, *prog);
    cudf::test::fixed_width_column_wrapper<int32_t> expected({1, 0, 0});
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results, expected);
  }
}

TYPED_TEST(StringsContainsTests, LargeRegex)
{
  // This results in 115 regex instructions and falls in the 'large' range.
  std::string large_regex =
    "hello @abc @def world The quick brown @fox jumps over the lazy @dog hello "
    "http://www.world.com I'm here @home zzzz";
  auto prog = TypeParam::create(cudf::experimental::regex_operation::CONTAINS, large_regex);

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
    auto results = TypeParam::contains_re(strings_view, *prog);
    cudf::test::fixed_width_column_wrapper<bool> expected({true, false, false});
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results, expected);
  }
  prog = TypeParam::create(cudf::experimental::regex_operation::MATCHES, large_regex);
  {
    auto results = TypeParam::matches_re(strings_view, *prog);
    cudf::test::fixed_width_column_wrapper<bool> expected({true, false, false});
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results, expected);
  }
  prog = TypeParam::create(cudf::experimental::regex_operation::COUNT, large_regex);
  {
    auto results = TypeParam::count_re(strings_view, *prog);
    cudf::test::fixed_width_column_wrapper<int32_t> expected({1, 0, 0});
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results, expected);
  }
}

TYPED_TEST(StringsContainsTests, ExtraLargeRegex)
{
  // This results in 321 regex instructions which is above the 'large' range.
  std::string data(320, '0');
  cudf::test::strings_column_wrapper strings({data, data, data, data, data, "00"});
  auto prog = TypeParam::create(cudf::experimental::regex_operation::CONTAINS, data);

  auto strings_view = cudf::strings_column_view(strings);
  {
    auto results = TypeParam::contains_re(strings_view, *prog);
    cudf::test::fixed_width_column_wrapper<bool> expected({true, true, true, true, true, false});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
  prog = TypeParam::create(cudf::experimental::regex_operation::MATCHES, data);
  {
    auto results = TypeParam::matches_re(strings_view, *prog);
    cudf::test::fixed_width_column_wrapper<bool> expected({true, true, true, true, true, false});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
  prog = TypeParam::create(cudf::experimental::regex_operation::COUNT, data);
  {
    auto results = TypeParam::count_re(strings_view, *prog);
    cudf::test::fixed_width_column_wrapper<int32_t> expected({1, 1, 1, 1, 1, 0});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
}

TYPED_TEST(StringsContainsTests, CrlfLineAnchorExtNewline)
{
  // JDK/Spark treat \r\n as a SINGLE line terminator: '$' matches before the \r and
  // never between the \r and \n. Expected values verified against OpenJDK 17
  // java.util.regex (default == EXT non-multiline; Pattern.MULTILINE == EXT|MULTILINE).
  auto input = cudf::test::strings_column_wrapper(
    {"abc\r\n", "abc\n", "abc\r", "abc", "a\r\nb", "abc\r\n\r\n", "", "abc" NEXT_LINE});
  auto view = cudf::strings_column_view(input);

  auto prog    = TypeParam::create(cudf::experimental::regex_operation::CONTAINS,
                                "^abc$",
                                cudf::strings::regex_flags::EXT_NEWLINE);
  auto both    = static_cast<cudf::strings::regex_flags>(cudf::strings::regex_flags::EXT_NEWLINE |
                                                      cudf::strings::regex_flags::MULTILINE);
  auto prog_ml = TypeParam::create(cudf::experimental::regex_operation::CONTAINS, "^abc$", both);

  // Java: ^abc$ EXT(non-ml) = {1,1,1,1,0,0,0,1}
  auto expected = cudf::test::fixed_width_column_wrapper<bool>({1, 1, 1, 1, 0, 0, 0, 1});
  auto results  = TypeParam::contains_re(view, *prog);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);

  // Java: ^abc$ EXT|MULTILINE = {1,1,1,1,0,1,0,1}
  expected = cudf::test::fixed_width_column_wrapper<bool>({1, 1, 1, 1, 0, 1, 0, 1});
  results  = TypeParam::contains_re(view, *prog_ml);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
}

TYPED_TEST(StringsContainsTests, CrlfBolAnchorExtNewline)
{
  // Symmetric ^ side: with \r\n as one terminator, multiline ^ matches AFTER the \n,
  // never between the \r and \n. So "^\n" must never match (a \n is never a line start).
  // Verified against OpenJDK 17: contains "(?m)^\n" = false for all of these.
  auto input = cudf::test::strings_column_wrapper({"abc\r\nDEF", "a\r\nb", "ab\rc", "x\ny"});
  auto view  = cudf::strings_column_view(input);
  auto both  = static_cast<cudf::strings::regex_flags>(cudf::strings::regex_flags::EXT_NEWLINE |
                                                      cudf::strings::regex_flags::MULTILINE);
  auto prog  = TypeParam::create(cudf::experimental::regex_operation::CONTAINS, "^\n", both);

  auto expected = cudf::test::fixed_width_column_wrapper<bool>({0, 0, 0, 0});
  auto results  = TypeParam::contains_re(view, *prog);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
}

TYPED_TEST(StringsContainsTests, CrlfEdgeCasesExtNewline)
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
    auto p  = TypeParam::create(cudf::experimental::regex_operation::CONTAINS, "^abc$", EXT);
    auto pm = TypeParam::create(cudf::experimental::regex_operation::CONTAINS, "^abc$", MLX);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*TypeParam::contains_re(view, *p),
                                   to_col(&edge_case::abc_anchored_en));
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*TypeParam::contains_re(view, *pm),
                                   to_col(&edge_case::abc_anchored_ml));
  }
  {  // matches_re abc$  (match at start of string)
    auto p = TypeParam::create(cudf::experimental::regex_operation::MATCHES, "abc$", EXT);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*TypeParam::matches_re(view, *p),
                                   to_col(&edge_case::abc_dollar_matches));
  }
  {  // count_re [a-z]+$  (non-multiline / multiline)
    auto p  = TypeParam::create(cudf::experimental::regex_operation::COUNT, "[a-z]+$", EXT);
    auto pm = TypeParam::create(cudf::experimental::regex_operation::COUNT, "[a-z]+$", MLX);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*TypeParam::count_re(view, *p),
                                   to_col(&edge_case::az_dollar_count_en));
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*TypeParam::count_re(view, *pm),
                                   to_col(&edge_case::az_dollar_count_ml));
  }
  {  // count_re ^[a-z]+  (multiline line-starts)
    auto pm = TypeParam::create(cudf::experimental::regex_operation::COUNT, "^[a-z]+", MLX);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*TypeParam::count_re(view, *pm),
                                   to_col(&edge_case::az_start_count_ml));
  }
  {  // CRLF-coupling discriminators: \r$ never matches inside \r\n; ^\n likewise
    auto pe = TypeParam::create(cudf::experimental::regex_operation::CONTAINS, "\\r$", MLX);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*TypeParam::contains_re(view, *pe),
                                   to_col(&edge_case::cr_dollar_ml));
    auto pb = TypeParam::create(cudf::experimental::regex_operation::CONTAINS, "^\n", MLX);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*TypeParam::contains_re(view, *pb),
                                   to_col(&edge_case::start_nl_ml));
  }
  {  // alternation containing $ (the #14856 construct) works natively, no transpiler
    auto p = TypeParam::create(cudf::experimental::regex_operation::CONTAINS, "(a$|b)", EXT);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*TypeParam::contains_re(view, *p),
                                   to_col(&edge_case::alt_a_or_b_en));
  }
}

TYPED_TEST(StringsContainsTests, CrlfDefaultLfOnlyNoExtNewline)
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

  auto prog = TypeParam::create(cudf::experimental::regex_operation::CONTAINS,
                                "^abc$");  // default flags: no EXT_NEWLINE
  auto expected =
    cudf::test::fixed_width_column_wrapper<bool>({0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*TypeParam::contains_re(view, *prog), expected);
}

TYPED_TEST(StringsContainsTests, AlternationNullableBranch)
{
  // "a(bc|de|fg|)h" has an explicit empty branch that makes 'h' directly reachable after 'a'.
  auto input = cudf::test::strings_column_wrapper(
    {"ah", "abch", "adeh", "afghh", "abcde", "a", "h", "", "abcdefgh", "xabchx"});
  auto sv = cudf::strings_column_view(input);

  auto prog = TypeParam::create(cudf::experimental::regex_operation::CONTAINS, "a(bc|de|fg|)h");
  // The interpreter currently accepts "abcdefgh" by continuing through successive alternatives.
  // Regex IR retains standard contiguous-alternation semantics for that row.
  auto constexpr concatenated_alternatives_match =
    std::is_same_v<TypeParam, cudf::test::interpreter_regex_backend> ? 1 : 0;
  {
    auto results = TypeParam::contains_re(sv, *prog);
    cudf::test::fixed_width_column_wrapper<bool> expected(
      {1, 1, 1, 1, 0, 0, 0, 0, concatenated_alternatives_match, 1});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
  prog = TypeParam::create(cudf::experimental::regex_operation::COUNT, "a(bc|de|fg|)h");
  {
    auto results = TypeParam::count_re(sv, *prog);
    cudf::test::fixed_width_column_wrapper<int32_t> expected(
      {1, 1, 1, 1, 0, 0, 0, 0, concatenated_alternatives_match, 1});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
}

TYPED_TEST(StringsContainsTests, BoundedRepetitionGap)
{
  // "ab{0,4}cv" — 'b' may repeat 0–4 times; five or more b's yield no match.
  auto input = cudf::test::strings_column_wrapper(
    {"acv", "abcv", "abbcv", "abbbcv", "abbbbcv", "abbbbbcv", "av", "acvx", "xacvx", ""});
  auto sv = cudf::strings_column_view(input);

  auto prog    = TypeParam::create(cudf::experimental::regex_operation::CONTAINS, "ab{0,4}cv");
  auto results = TypeParam::contains_re(sv, *prog);
  cudf::test::fixed_width_column_wrapper<bool> expected({1, 1, 1, 1, 1, 0, 0, 1, 1, 0});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
}

TYPED_TEST(StringsContainsTests, ExtNewlineDotAny)
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
    auto prog    = TypeParam::create(cudf::experimental::regex_operation::CONTAINS, "a.b");
    auto results = TypeParam::contains_re(sv, *prog);
    cudf::test::fixed_width_column_wrapper<bool> expected({1, 0, 1, 1, 1, 1, 0, 0});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
  // EXT_NEWLINE: \r and all extended newlines also excluded
  {
    auto prog    = TypeParam::create(cudf::experimental::regex_operation::CONTAINS,
                                  "a.b",
                                  cudf::strings::regex_flags::EXT_NEWLINE);
    auto results = TypeParam::contains_re(sv, *prog);
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
    auto prog = TypeParam::create(
      cudf::experimental::regex_operation::CONTAINS, ".+", cudf::strings::regex_flags::EXT_NEWLINE);
    auto results = TypeParam::contains_re(sv2, *prog);
    cudf::test::fixed_width_column_wrapper<bool> expected2({1, 0, 1, 0});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected2);
  }
}

TYPED_TEST(StringsContainsTests, AlternationPriorityCount)
{
  // Leftmost-first (first-alternative-wins): the shorter first branch is consumed, leaving
  // subsequent characters available for the next match.
  {
    // "a|aa": "a" wins, so "aaaa" counts as 4 individual matches, not 2 "aa" matches.
    auto input   = cudf::test::strings_column_wrapper({"aaaa", "aaaaaa", "aaab", "a", "b", ""});
    auto sv      = cudf::strings_column_view(input);
    auto prog    = TypeParam::create(cudf::experimental::regex_operation::COUNT, "a|aa");
    auto results = TypeParam::count_re(sv, *prog);
    cudf::test::fixed_width_column_wrapper<int32_t> expected({4, 6, 3, 1, 0, 0});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
  {
    // "foo|foobar": "foo" wins when both alternatives start at the same position.
    auto input   = cudf::test::strings_column_wrapper({"foo", "foobar", "foofoo", "bar", ""});
    auto sv      = cudf::strings_column_view(input);
    auto prog    = TypeParam::create(cudf::experimental::regex_operation::COUNT, "foo|foobar");
    auto results = TypeParam::count_re(sv, *prog);
    cudf::test::fixed_width_column_wrapper<int32_t> expected({1, 1, 2, 0, 0});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
}

TYPED_TEST(StringsContainsTests, LazyQuantifiers)
{
  // Lazy star/plus in non-DOTALL mode: prefer the shortest match.
  auto input = cudf::test::strings_column_wrapper(
    {"ab", "abc", "xdefx", "xghix", "jkl", "abc xdefx xghix jkl"});
  auto sv = cudf::strings_column_view(input);

  {
    auto prog    = TypeParam::create(cudf::experimental::regex_operation::CONTAINS, "x.*?x");
    auto results = TypeParam::contains_re(sv, *prog);
    cudf::test::fixed_width_column_wrapper<bool> expected({0, 0, 1, 1, 0, 1});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
  {
    auto prog    = TypeParam::create(cudf::experimental::regex_operation::CONTAINS, "x.+?x");
    auto results = TypeParam::contains_re(sv, *prog);
    cudf::test::fixed_width_column_wrapper<bool> expected({0, 0, 1, 1, 0, 1});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
}

TYPED_TEST(StringsContainsTests, WhitespaceControlCharacters)
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
  auto prog    = TypeParam::create(cudf::experimental::regex_operation::CONTAINS, "a\\sb");
  auto results = TypeParam::contains_re(sv, *prog);
  cudf::test::fixed_width_column_wrapper<bool> expected({1, 1, 1, 1, 1, 0});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
}
