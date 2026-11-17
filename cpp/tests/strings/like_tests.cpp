/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>

#include <cudf/scalar/scalar.hpp>
#include <cudf/strings/contains.hpp>
#include <cudf/strings/strings_column_view.hpp>

struct StringsLikeTests : public cudf::test::BaseFixture {};

TEST_F(StringsLikeTests, Basic)
{
  auto input = cudf::test::strings_column_wrapper(
    {"abc", "a bc", "ABC", "abcd", " abc", "", "", "áéêú", "abc "},
    {true, true, true, true, true, true, false, true, true});
  auto sv       = cudf::strings_column_view(input);
  auto pattern  = std::string_view("abc");
  auto results  = cudf::strings::like(sv, pattern);
  auto expected = cudf::test::fixed_width_column_wrapper<bool>(
    {true, false, false, false, false, false, false, false, false},
    {true, true, true, true, true, true, false, true, true});
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(results->view(), expected);

  input = cudf::test::strings_column_wrapper(
    {"abcdefghijklmnopqrstuvwxyz01234567890ABCDEFGHIJKLMNOPQRSTUVWXYZáéêú",
     "bcdefghijklmnopqrstuvwxyz01234567890ABCDEFGHIJKLMNOPQRSTUVWXYZáéêú",
     "01234567890123456789012345678901234567890123456789012345678901234567890123456789"
     "01234567890123456789012345678901234567890123456789012345678901234567890123456789"});
  sv      = cudf::strings_column_view(input);
  pattern = std::string_view("abcdefghijklmnopqrstuvwxyz01234567890ABCDEFGHIJKLMNOPQRSTUVWXYZáéêú");
  results = cudf::strings::like(sv, pattern);
  expected = cudf::test::fixed_width_column_wrapper<bool>({true, false, false});
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(results->view(), expected);
}

TEST_F(StringsLikeTests, Leading)
{
  auto input = cudf::test::strings_column_wrapper({"a", "aa", "aaa", "b", "bb", "bba", "", "áéêú"});
  auto const sv = cudf::strings_column_view(input);

  auto results  = cudf::strings::like(sv, std::string_view("a%"));
  auto expected = cudf::test::fixed_width_column_wrapper<bool>(
    {true, true, true, false, false, false, false, false});
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(results->view(), expected);

  results  = cudf::strings::like(sv, std::string_view("__a%"));
  expected = cudf::test::fixed_width_column_wrapper<bool>(
    {false, false, true, false, false, true, false, false});
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(results->view(), expected);

  results  = cudf::strings::like(sv, std::string_view("á%"));
  expected = cudf::test::fixed_width_column_wrapper<bool>(
    {false, false, false, false, false, false, false, true});
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(results->view(), expected);

  auto big = cudf::test::strings_column_wrapper(
    {"abcdéfghijklmnopqrstuvwxyz0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZáéêú",
     "ábcdefghijklmnopqrstuvwxyz0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZáéêú",
     "01234567890123456789012345678901234567890123456789012345678901234567890123456789"
     "01234567890123456789012345678901234567890123456789012345678901234567890123456789"});
  auto const sv_big = cudf::strings_column_view(big);

  results  = cudf::strings::like(sv_big, std::string_view("a%"));
  expected = cudf::test::fixed_width_column_wrapper<bool>({true, false, false});
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(results->view(), expected);

  results  = cudf::strings::like(sv_big, std::string_view("__c%"));
  expected = cudf::test::fixed_width_column_wrapper<bool>({true, true, false});
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(results->view(), expected);

  results  = cudf::strings::like(sv_big, std::string_view("áb%"));
  expected = cudf::test::fixed_width_column_wrapper<bool>({false, true, false});
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(results->view(), expected);
}

TEST_F(StringsLikeTests, Trailing)
{
  auto input = cudf::test::strings_column_wrapper({"a", "aa", "aaa", "b", "bb", "bba", "", "áéêú"});
  auto const sv = cudf::strings_column_view(input);

  auto results  = cudf::strings::like(sv, std::string_view("%a"));
  auto expected = cudf::test::fixed_width_column_wrapper<bool>(
    {true, true, true, false, false, true, false, false});
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(results->view(), expected);
  results = cudf::strings::like(sv, std::string_view("%a%"));
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(results->view(), expected);

  results  = cudf::strings::like(sv, std::string_view("%_a"));
  expected = cudf::test::fixed_width_column_wrapper<bool>(
    {false, true, true, false, false, true, false, false});
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(results->view(), expected);

  results  = cudf::strings::like(sv, std::string_view("%_êú"));
  expected = cudf::test::fixed_width_column_wrapper<bool>(
    {false, false, false, false, false, false, false, true});
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(results->view(), expected);

  auto big = cudf::test::strings_column_wrapper(
    {"bcdéfghijklmnopqrstuvwxyz0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZáéêúa",
     "ábcdefghijklmnopqrstuvwxyz0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZáéêú",
     "01234567890123456789012345678901234567890123456789012345678901234567890123456789"
     "01234567890123456789012345678901234567890123456789012345678901234567890123456789"});
  auto const sv_big = cudf::strings_column_view(big);

  results  = cudf::strings::like(sv_big, std::string_view("%a"));
  expected = cudf::test::fixed_width_column_wrapper<bool>({true, false, false});
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(results->view(), expected);
  results = cudf::strings::like(sv_big, std::string_view("%a%"));
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(results->view(), expected);
  results = cudf::strings::like(sv_big, std::string_view("%déf%"));
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(results->view(), expected);

  results  = cudf::strings::like(sv_big, std::string_view("%_a"));
  expected = cudf::test::fixed_width_column_wrapper<bool>({true, false, false});
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(results->view(), expected);

  results  = cudf::strings::like(sv_big, std::string_view("%_êú"));
  expected = cudf::test::fixed_width_column_wrapper<bool>({false, true, false});
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(results->view(), expected);
}

TEST_F(StringsLikeTests, Middle)
{
  auto input =
    cudf::test::strings_column_wrapper({"a", "aa", "aaa", "b", "bb", "bba", "", "ábéêú"});
  auto const sv = cudf::strings_column_view(input);

  auto results  = cudf::strings::like(sv, std::string_view("a%a"));
  auto expected = cudf::test::fixed_width_column_wrapper<bool>(
    {false, true, true, false, false, false, false, false});
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(results->view(), expected);

  results  = cudf::strings::like(sv, std::string_view("á%ê%ú"));
  expected = cudf::test::fixed_width_column_wrapper<bool>(
    {false, false, false, false, false, false, false, true});
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(results->view(), expected);

  auto big = cudf::test::strings_column_wrapper(
    {"bcdéfghijklmnopqrstuvwxyz0123456789aBCDEFGHIJKLMNOPQRSTUVWXYZáéêúa",
     "ábcdêfghijklmnopqrstúvwxyz0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZáéêú",
     "0123456789aaabbb67890aaa45678901234567890123456789012345678901234567890123456789"
     "01234567890123456789012345678901234567890123456789012345678901234567890123456fff"});
  auto const sv_big = cudf::strings_column_view(big);

  results  = cudf::strings::like(sv_big, std::string_view("b%a"));
  expected = cudf::test::fixed_width_column_wrapper<bool>({true, false, false});
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(results->view(), expected);

  results  = cudf::strings::like(sv_big, std::string_view("á%ê%ú"));
  expected = cudf::test::fixed_width_column_wrapper<bool>({false, true, false});
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(results->view(), expected);

  results  = cudf::strings::like(sv_big, std::string_view("%aaa%bbb%fff"));
  expected = cudf::test::fixed_width_column_wrapper<bool>({false, false, true});
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(results->view(), expected);
}

TEST_F(StringsLikeTests, MiddleOnly)
{
  auto input =
    cudf::test::strings_column_wrapper({"a", "aa", "aaa", "b", "bb", "bba", "", "ábéêú"});
  auto const sv = cudf::strings_column_view(input);

  auto results  = cudf::strings::like(sv, std::string_view("%a%"));
  auto expected = cudf::test::fixed_width_column_wrapper<bool>(
    {true, true, true, false, false, true, false, false});
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(results->view(), expected);

  results  = cudf::strings::like(sv, std::string_view("%á%ê%ú%"));
  expected = cudf::test::fixed_width_column_wrapper<bool>(
    {false, false, false, false, false, false, false, true});
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(results->view(), expected);

  auto big = cudf::test::strings_column_wrapper(
    {"bcdéfghijklmnopqrstuvwxyz0123456789aBCDEFGHIJKLMNOPQRSTUVWXYZáéêúa",
     "ábcdêfghijklmnopqrstúvwxyz0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZáéêú",
     "0123456789aaabbb67890aaa45678901234567890123456789012345678901234567890123456789"
     "01234567890123456789012345678901234567890123456789012345678901234567890123456fff"});
  auto const sv_big = cudf::strings_column_view(big);

  results  = cudf::strings::like(sv_big, std::string_view("%TUVWXYZá%"));
  expected = cudf::test::fixed_width_column_wrapper<bool>({true, true, false});
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(results->view(), expected);

  results  = cudf::strings::like(sv_big, std::string_view("%aaa%bbb%fff%"));
  expected = cudf::test::fixed_width_column_wrapper<bool>({false, false, true});
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(results->view(), expected);

  results = cudf::strings::like(
    sv_big,
    std::string_view("%0123456789012345678901234567890123456789012345678901234567890123456f%"));
  expected = cudf::test::fixed_width_column_wrapper<bool>({false, false, true});
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(results->view(), expected);
}

TEST_F(StringsLikeTests, Place)
{
  auto input =
    cudf::test::strings_column_wrapper({"a", "aa", "aaa", "bab", "ab", "aba", "", "éaé"});
  auto const sv = cudf::strings_column_view(input);

  auto results  = cudf::strings::like(sv, std::string_view("a_"));
  auto expected = cudf::test::fixed_width_column_wrapper<bool>(
    {false, true, false, false, true, false, false, false});
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(results->view(), expected);

  results  = cudf::strings::like(sv, std::string_view("_a_"));
  expected = cudf::test::fixed_width_column_wrapper<bool>(
    {false, false, true, true, false, false, false, true});
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(results->view(), expected);

  results  = cudf::strings::like(sv, std::string_view("__a"));
  expected = cudf::test::fixed_width_column_wrapper<bool>(
    {false, false, true, false, false, true, false, false});
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(results->view(), expected);

  results  = cudf::strings::like(sv, std::string_view("é_é"));
  expected = cudf::test::fixed_width_column_wrapper<bool>(
    {false, false, false, false, false, false, false, true});
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(results->view(), expected);

  auto big = cudf::test::strings_column_wrapper(
    {"abcdéfghijklmnopqrstuvwxyz0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZáéêú",
     "ábcdefghijklmnopqrstuvwxyz0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZáéêú",
     "01234567890123456789012345678901234567890123456789012345678901234567890123456789"
     "01234567890123456789012345678901234567890123456789012345678901234567890123456789"});
  auto const sv_big = cudf::strings_column_view(big);

  results = cudf::strings::like(
    sv_big, std::string_view("a_________________________________________________________________"));
  expected = cudf::test::fixed_width_column_wrapper<bool>({true, false, false});
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(results->view(), expected);

  results = cudf::strings::like(
    sv_big, std::string_view("__________________________0123456789______________________________"));
  expected = cudf::test::fixed_width_column_wrapper<bool>({true, true, false});
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(results->view(), expected);

  results = cudf::strings::like(
    sv_big, std::string_view("_________________________________________________________________ú"));
  expected = cudf::test::fixed_width_column_wrapper<bool>({true, true, false});
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(results->view(), expected);

  results = cudf::strings::like(
    sv_big, std::string_view("á________________________________________________________________ú"));
  expected = cudf::test::fixed_width_column_wrapper<bool>({false, true, false});
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(results->view(), expected);
}

TEST_F(StringsLikeTests, Escape)
{
  auto input = cudf::test::strings_column_wrapper(
    {"10%-20%", "10-20", "10%%-20%", "a_b", "b_a", "___", "", "aéb"});
  auto const sv = cudf::strings_column_view(input);

  auto big = cudf::test::strings_column_wrapper(
    {"abcdéfghijklmnopqrstuvwxyz0_2_4_6_8_ABCDEFGHIJKLMNOPQRSTUVWXYZáéêú",
     "abcdefghijklmnopqrstuvwxyz%1%3%5%7%9ABCDEFGHIJKLMNOPQRSTUVWXYZáéêú",
     "01234567890123456789012345678901234567890123456789012345678901234567890123456789"
     "01234567890123456789012345678901234567890123456789012345678901234567890123456789"});
  auto const sv_big = cudf::strings_column_view(big);

  auto escape   = std::string_view("\\");
  auto results  = cudf::strings::like(sv, std::string_view("10\\%-20\\%"), escape);
  auto expected = cudf::test::fixed_width_column_wrapper<bool>(
    {true, false, false, false, false, false, false, false});
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(results->view(), expected);
  results  = cudf::strings::like(sv_big, std::string_view("a%\\%_\\%_\\%_\\%_\\%_A%ú"), escape);
  expected = cudf::test::fixed_width_column_wrapper<bool>({false, true, false});
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(results->view(), expected);

  results  = cudf::strings::like(sv, std::string_view("\\__\\_"), escape);
  expected = cudf::test::fixed_width_column_wrapper<bool>(
    {false, false, false, false, false, true, false, false});
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(results->view(), expected);
  results  = cudf::strings::like(sv_big, std::string_view("a%0\\_2\\_4\\_6\\_8\\_A%ú"), escape);
  expected = cudf::test::fixed_width_column_wrapper<bool>({true, false, false});
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(results->view(), expected);

  escape   = std::string_view("%");
  results  = cudf::strings::like(sv, std::string_view("10%%%%-20%%"), escape);
  expected = cudf::test::fixed_width_column_wrapper<bool>(
    {false, false, true, false, false, false, false, false});
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(results->view(), expected);
  results = cudf::strings::like(
    sv_big,
    std::string_view("abcd_fghijklmnopqrstuvwxyz%%1%%3%%5%%7%%9ABCDEFGHIJKLMNOPQRSTUVWXYZáéêú"),
    escape);
  expected = cudf::test::fixed_width_column_wrapper<bool>({false, true, false});
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(results->view(), expected);

  escape   = std::string_view("%");
  results  = cudf::strings::like(sv, std::string_view("_%__"), escape);
  expected = cudf::test::fixed_width_column_wrapper<bool>(
    {false, false, false, true, true, true, false, false});
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(results->view(), expected);
  results = cudf::strings::like(
    sv_big,
    std::string_view("abcd_fghijklmnopqrstuvwxyz0%_2%_4%_6%_8%_ABCDEFGHIJKLMNOPQRSTUVWXYZáéêú"),
    escape);
  expected = cudf::test::fixed_width_column_wrapper<bool>({true, false, false});
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(results->view(), expected);

  escape   = std::string_view("_");
  results  = cudf::strings::like(sv, std::string_view("a__b"), escape);
  expected = cudf::test::fixed_width_column_wrapper<bool>(
    {false, false, false, true, false, false, false, false});
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(results->view(), expected);
  results = cudf::strings::like(
    sv_big,
    std::string_view("abcd%fghijklmnopqrstuvwxyz_%1_%3_%5_%7_%9ABCDEFGHIJKLMNOPQRSTUVWXYZáéêú"),
    escape);
  expected = cudf::test::fixed_width_column_wrapper<bool>({false, true, false});
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(results->view(), expected);

  escape   = std::string_view("^");
  results  = cudf::strings::like(sv_big, std::string_view("%0^_2^_4_6_8%"), escape);
  expected = cudf::test::fixed_width_column_wrapper<bool>({true, false, false});
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(results->view(), expected);

  results  = cudf::strings::like(sv, std::string_view("a^_b%"), escape);
  expected = cudf::test::fixed_width_column_wrapper<bool>(
    {false, false, false, true, false, false, false, false});
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(results->view(), expected);
  results = cudf::strings::like(sv, std::string_view("%a^_b"), escape);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(results->view(), expected);
}

TEST_F(StringsLikeTests, MultiplePatterns)
{
  auto input =
    cudf::test::strings_column_wrapper({"abc", "a1a2b3b4c", "aaabbb", "bbbc", "", "áéêú"});
  auto patterns = cudf::test::strings_column_wrapper({"a%b%c", "a%c", "a__b", "b__c", "", "áéêú"});

  auto sv_input    = cudf::strings_column_view(input);
  auto sv_patterns = cudf::strings_column_view(patterns);
  auto results     = cudf::strings::like(sv_input, sv_patterns);
  auto expected =
    cudf::test::fixed_width_column_wrapper<bool>({true, true, false, true, true, true});
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(results->view(), expected);

  input = cudf::test::strings_column_wrapper(
    {"abcdéfghijklmnopqrstuvwxyz0_2_4_6_8_ABCDEFGHIJKLMNOPQRSTUVWXYZáéêú",
     "1abcdefghijklmnopqrstuvwxyz%1%3%5%7%9ABCDEFGHIJKLMNOPQRSTUVWXYZáéêú",
     "abcdefghijklmnopqrstuvwxyz%1%3%5%7%9ABCDEFGHIJKLMNOPQRSTUVWXYZáéêú2",
     "01234567890123456789012345678901234567890123456789012345678901234567890123456789",
     "01234567890123456789012345678901234567890123456789012345678901234567890123456789"
     "01234567890123456789012345678901234567890123456789012345678901234567890123456789"});
  sv_input = cudf::strings_column_view(input);
  patterns = cudf::test::strings_column_wrapper(
    {"a%b%ú",
     "1ab%",
     "%áéêú2",
     "01234567890123456789012345678901234567890123456789012345678901234567890123456789",
     "abcdef"});
  sv_patterns = cudf::strings_column_view(patterns);
  results     = cudf::strings::like(sv_input, sv_patterns);
  expected    = cudf::test::fixed_width_column_wrapper<bool>({true, true, true, true, false});
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(results->view(), expected);
}

TEST_F(StringsLikeTests, Empty)
{
  cudf::test::strings_column_wrapper input({"ooo", "20%", ""});
  auto sv       = cudf::strings_column_view(input);
  auto results  = cudf::strings::like(sv, std::string_view(""));
  auto expected = cudf::test::fixed_width_column_wrapper<bool>({false, false, true});
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(results->view(), expected);

  auto empty          = cudf::make_empty_column(cudf::type_id::STRING);
  sv                  = cudf::strings_column_view(empty->view());
  results             = cudf::strings::like(sv, std::string_view("20%"));
  auto expected_empty = cudf::make_empty_column(cudf::type_id::BOOL8);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(results->view(), expected_empty->view());

  results = cudf::strings::like(sv, sv);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(results->view(), expected_empty->view());
}

TEST_F(StringsLikeTests, AllNulls)
{
  auto input    = cudf::test::strings_column_wrapper({"", "", ""}, {0, 0, 0});
  auto sv       = cudf::strings_column_view(input);
  auto results  = cudf::strings::like(sv, std::string_view(""));
  auto expected = cudf::test::fixed_width_column_wrapper<bool>({0, 0, 0}, {0, 0, 0});
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(results->view(), expected);

  auto patterns = cudf::test::strings_column_wrapper({"", "", ""});
  results       = cudf::strings::like(sv, cudf::strings_column_view(patterns));
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(results->view(), expected);
}

TEST_F(StringsLikeTests, Errors)
{
  auto const input = cudf::test::strings_column_wrapper({"3", "33"});
  auto const sv    = cudf::strings_column_view(input);

  EXPECT_THROW(cudf::strings::like(sv, std::string_view("3"), std::string_view("ee")),
               std::invalid_argument);

  auto patterns          = cudf::test::strings_column_wrapper({"3", ""}, {true, false});
  auto const sv_patterns = cudf::strings_column_view(patterns);
  EXPECT_THROW(cudf::strings::like(sv, sv_patterns), std::invalid_argument);
  EXPECT_THROW(cudf::strings::like(sv, sv_patterns, cudf::string_scalar("ee")),
               std::invalid_argument);
}
