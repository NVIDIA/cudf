/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>

#include <cudf/column/column.hpp>
#include <cudf/strings/capitalize.hpp>
#include <cudf/strings/case.hpp>
#include <cudf/strings/strings_column_view.hpp>

#include <cuda/iterator>

#include <vector>

struct StringsCaseTest : public cudf::test::BaseFixture {};

TEST_F(StringsCaseTest, ToLower)
{
  std::vector<char const*> h_strings{
    "Éxamples aBc", "123 456", nullptr, "ARE THE", "tést strings", ""};
  std::vector<char const*> h_expected{
    "éxamples abc", "123 456", nullptr, "are the", "tést strings", ""};

  cudf::test::strings_column_wrapper strings(
    h_strings.begin(), h_strings.end(), cuda::transform_iterator(h_strings.begin(), [](auto str) {
      return str != nullptr;
    }));
  auto strings_view = cudf::strings_column_view(strings);

  auto results = cudf::strings::to_lower(strings_view);

  cudf::test::strings_column_wrapper expected(
    h_expected.begin(),
    h_expected.end(),
    cuda::transform_iterator(h_expected.begin(), [](auto str) { return str != nullptr; }));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
}

TEST_F(StringsCaseTest, ToUpper)
{
  std::vector<char const*> h_strings{
    "Éxamples aBc", "123 456", nullptr, "ARE THE", "tést strings", ""};
  std::vector<char const*> h_expected{
    "ÉXAMPLES ABC", "123 456", nullptr, "ARE THE", "TÉST STRINGS", ""};

  cudf::test::strings_column_wrapper strings(
    h_strings.begin(), h_strings.end(), cuda::transform_iterator(h_strings.begin(), [](auto str) {
      return str != nullptr;
    }));
  auto strings_view = cudf::strings_column_view(strings);

  auto results = cudf::strings::to_upper(strings_view);

  cudf::test::strings_column_wrapper expected(
    h_expected.begin(),
    h_expected.end(),
    cuda::transform_iterator(h_expected.begin(), [](auto str) { return str != nullptr; }));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
}

TEST_F(StringsCaseTest, Swapcase)
{
  std::vector<char const*> h_strings{
    "Éxamples aBc", "123 456", nullptr, "ARE THE", "tést strings", ""};
  std::vector<char const*> h_expected{
    "éXAMPLES AbC", "123 456", nullptr, "are the", "TÉST STRINGS", ""};

  cudf::test::strings_column_wrapper strings(
    h_strings.begin(), h_strings.end(), cuda::transform_iterator(h_strings.begin(), [](auto str) {
      return str != nullptr;
    }));
  auto strings_view = cudf::strings_column_view(strings);

  auto results = cudf::strings::swapcase(strings_view);

  cudf::test::strings_column_wrapper expected(
    h_expected.begin(),
    h_expected.end(),
    cuda::transform_iterator(h_expected.begin(), [](auto str) { return str != nullptr; }));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
}

TEST_F(StringsCaseTest, Capitalize)
{
  cudf::test::strings_column_wrapper strings(
    {"SȺȺnich xyZ", "Examples aBc", "thesé", "", "ARE\tTHE", "tést\tstrings", ""},
    {true, true, true, false, true, true, true});
  auto strings_view = cudf::strings_column_view(strings);

  {
    auto results = cudf::strings::capitalize(strings_view);
    cudf::test::strings_column_wrapper expected(
      {"Sⱥⱥnich xyz", "Examples abc", "Thesé", "", "Are\tthe", "Tést\tstrings", ""},
      {true, true, true, false, true, true, true});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
  {
    auto results = cudf::strings::capitalize(strings_view, std::string_view(" "));
    cudf::test::strings_column_wrapper expected(
      {"Sⱥⱥnich Xyz", "Examples Abc", "Thesé", "", "Are\tthe", "Tést\tstrings", ""},
      {true, true, true, false, true, true, true});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
  {
    auto results = cudf::strings::capitalize(strings_view, std::string_view(" \t"));
    cudf::test::strings_column_wrapper expected(
      {"Sⱥⱥnich Xyz", "Examples Abc", "Thesé", "", "Are\tThe", "Tést\tStrings", ""},
      {true, true, true, false, true, true, true});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
}

TEST_F(StringsCaseTest, Title)
{
  cudf::test::strings_column_wrapper input(
    {"SȺȺnich", "Examples aBc", "thesé", "", "ARE THE", "tést strings", "", "n2viDIA corp"},
    {true, true, true, false, true, true, true, true});
  auto strings_view = cudf::strings_column_view(input);

  auto results = cudf::strings::title(strings_view);

  cudf::test::strings_column_wrapper expected(
    {"Sⱥⱥnich", "Examples Abc", "Thesé", "", "Are The", "Tést Strings", "", "N2Vidia Corp"},
    {true, true, true, false, true, true, true, true});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);

  results = cudf::strings::title(strings_view, cudf::strings::string_character_types::ALPHANUM);

  cudf::test::strings_column_wrapper expected2(
    {"Sⱥⱥnich", "Examples Abc", "Thesé", "", "Are The", "Tést Strings", "", "N2vidia Corp"},
    {true, true, true, false, true, true, true, true});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected2);
}

TEST_F(StringsCaseTest, IsTitle)
{
  cudf::test::strings_column_wrapper input(
    {"Sⱥⱥnich",
     "Examples Abc",
     "Thesé Strings",
     "",
     "Are The",
     "Tést strings",
     "",
     "N2Vidia Corp",
     "SNAKE",
     "!Abc",
     " Eagle",
     "A Test",
     "12345",
     "Alpha Not Upper Or Lower: ƻC",
     "one More"},
    {true, true, true, false, true, true, true, true, true, true, true, true, true, true, true});

  auto results = cudf::strings::is_title(cudf::strings_column_view(input));

  cudf::test::fixed_width_column_wrapper<bool> expected(
    {1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0},
    {true, true, true, false, true, true, true, true, true, true, true, true, true, true, true});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
}

TEST_F(StringsCaseTest, MultiCharUpper)
{
  cudf::test::strings_column_wrapper strings{"\u1f52 \u1f83", "\u1e98 \ufb05", "\u0149"};
  cudf::test::strings_column_wrapper expected{
    "\u03a5\u0313\u0300 \u1f0b\u0399", "\u0057\u030a \u0053\u0054", "\u02bc\u004e"};
  auto strings_view = cudf::strings_column_view(strings);

  auto results = cudf::strings::to_upper(strings_view);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
}

TEST_F(StringsCaseTest, MultiCharLower)
{
  // there's only one of these
  cudf::test::strings_column_wrapper strings{"\u0130"};
  cudf::test::strings_column_wrapper expected{"\u0069\u0307"};
  auto strings_view = cudf::strings_column_view(strings);

  auto results = cudf::strings::to_lower(strings_view);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
}

TEST_F(StringsCaseTest, MultiCharTitle)
{
  auto input = cudf::test::strings_column_wrapper{
    "\u00df \u01c4 \u01c6 \u01c7 \u01c9 \u01ca \u01cc \u01f1 \u01f3 \u0587 \u10d0 "
    "\u10d1 \u10d2 \u10d3 \u10d4 \u10d5 \u10d6 \u10d7 \u10d8 \u10d9 \u10da \u10db "
    "\u10dc \u10dd \u10de \u10df \u10e0 \u10e1 \u10e2 \u10e3 \u10e4 \u10e5 \u10e6 "
    "\u10e7 \u10e8 \u10e9 \u10ea \u10eb \u10ec \u10ed \u10ee \u10ef \u10f0 \u10f1 "
    "\u10f2 \u10f3 \u10f4 \u10f5 \u10f6 \u10f7 \u10f8 \u10f9 \u10fa \u1f80 \u1f81 "
    "\u1f82 \u1f83 \u1f84 \u1f85 \u1f86 \u1f87 \u1f90 \u1f91 \u1f92 \u1f93 \u1f94 "
    "\u1f95 \u1f96 \u1f97 \u1fa0 \u1fa1 \u1fa2 \u1fa3 \u1fa4 \u1fa5 \u1fa6 \u1fa7 "
    "\u1fb2 \u1fb3 \u1fb4 \u1fb7 \u1fc2 \u1fc3 \u1fc4 \u1fc7 \u1ff2 \u1ff3 \u1ff4 "
    "\u1ff7 \ufb00 \ufb01 \ufb02 \ufb03 \ufb04 \ufb05 \ufb06 \ufb13 \ufb14 \ufb15 "
    "\ufb16 \ufb17 "};
  auto expected = cudf::test::strings_column_wrapper{
    "Ss ǅ ǅ ǈ ǈ ǋ ǋ ǲ ǲ Եւ ა ბ გ დ ე ვ ზ თ ი კ ლ მ ნ ო პ ჟ რ ს ტ უ ფ ქ ღ ყ შ ჩ ც ძ წ ჭ ხ ჯ ჰ ჱ ჲ ჳ "
    "ჴ ჵ ჶ ჷ ჸ ჹ ჺ ᾈ ᾉ ᾊ ᾋ ᾌ ᾍ ᾎ ᾏ ᾘ ᾙ ᾚ ᾛ ᾜ ᾝ ᾞ ᾟ ᾨ ᾩ ᾪ ᾫ ᾬ ᾭ ᾮ ᾯ Ὰͅ ᾼ Άͅ ᾼ͂ Ὴͅ ῌ Ήͅ ῌ͂ Ὼͅ ῼ Ώͅ ῼ͂ Ff Fi "
    "Fl Ffi Ffl St St Մն Մե Մի Վն Մխ "};
  auto sv = cudf::strings_column_view(input);

  auto results = cudf::strings::capitalize(sv, std::string_view(" "));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);

  results = cudf::strings::title(sv);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
}

TEST_F(StringsCaseTest, Ascii)
{
  // triggering the ascii code path requires some long-ish strings
  cudf::test::strings_column_wrapper input{
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz1234567890!@#$%^&*()_+=- ",
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz1234567890!@#$%^&*()_+=- ",
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz1234567890!@#$%^&*()_+=- ",
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz1234567890!@#$%^&*()_+=-"};
  auto view     = cudf::strings_column_view(input);
  auto expected = cudf::test::strings_column_wrapper{
    "abcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyz1234567890!@#$%^&*()_+=- ",
    "abcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyz1234567890!@#$%^&*()_+=- ",
    "abcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyz1234567890!@#$%^&*()_+=- ",
    "abcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyz1234567890!@#$%^&*()_+=-"};
  auto results = cudf::strings::to_lower(view);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);

  expected = cudf::test::strings_column_wrapper{
    "ABCDEFGHIJKLMNOPQRSTUVWXYZABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890!@#$%^&*()_+=- ",
    "ABCDEFGHIJKLMNOPQRSTUVWXYZABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890!@#$%^&*()_+=- ",
    "ABCDEFGHIJKLMNOPQRSTUVWXYZABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890!@#$%^&*()_+=- ",
    "ABCDEFGHIJKLMNOPQRSTUVWXYZABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890!@#$%^&*()_+=-"};
  results = cudf::strings::to_upper(view);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);

  results = cudf::strings::to_upper(cudf::strings_column_view(cudf::slice(input, {1, 3}).front()));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, cudf::slice(expected, {1, 3}).front());
}

TEST_F(StringsCaseTest, LongStrings)
{
  // average string length >= AVG_CHAR_BYTES_THRESHOLD as defined in case.cu
  auto input = cudf::test::strings_column_wrapper(
    {"abcdéfghijklmnopqrstuvwxyzABCDÉFGHIJKLMNOPQRSTUVWXYZ1234567890!@#$%^&*()_+=- ",
     "ABCDÉFGHIJKLMNOPQRSTUVWXYZabcdéfghijklmnopqrstuvwxyz1234567890!@#$%^&*()_+=- ",
     "ABCDÉFGHIJKLMNOPQRSTUVWXYZabcdéfghijklmnopqrstuvwxyz1234567890!@#$%^&*()_+=- ",
     "",
     "ABCDÉFGHIJKLMNOPQRSTUVWXYZabcdéfghijklmnopqrstuvwxyz1234567890!@#$%^&*()_+=-"},
    {1, 1, 1, 0, 1});
  auto view     = cudf::strings_column_view(input);
  auto expected = cudf::test::strings_column_wrapper(
    {"abcdéfghijklmnopqrstuvwxyzabcdéfghijklmnopqrstuvwxyz1234567890!@#$%^&*()_+=- ",
     "abcdéfghijklmnopqrstuvwxyzabcdéfghijklmnopqrstuvwxyz1234567890!@#$%^&*()_+=- ",
     "abcdéfghijklmnopqrstuvwxyzabcdéfghijklmnopqrstuvwxyz1234567890!@#$%^&*()_+=- ",
     "",
     "abcdéfghijklmnopqrstuvwxyzabcdéfghijklmnopqrstuvwxyz1234567890!@#$%^&*()_+=-"},
    {1, 1, 1, 0, 1});
  auto results = cudf::strings::to_lower(view);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);

  expected = cudf::test::strings_column_wrapper(
    {"ABCDÉFGHIJKLMNOPQRSTUVWXYZABCDÉFGHIJKLMNOPQRSTUVWXYZ1234567890!@#$%^&*()_+=- ",
     "ABCDÉFGHIJKLMNOPQRSTUVWXYZABCDÉFGHIJKLMNOPQRSTUVWXYZ1234567890!@#$%^&*()_+=- ",
     "ABCDÉFGHIJKLMNOPQRSTUVWXYZABCDÉFGHIJKLMNOPQRSTUVWXYZ1234567890!@#$%^&*()_+=- ",
     "",
     "ABCDÉFGHIJKLMNOPQRSTUVWXYZABCDÉFGHIJKLMNOPQRSTUVWXYZ1234567890!@#$%^&*()_+=-"},
    {1, 1, 1, 0, 1});
  results = cudf::strings::to_upper(view);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);

  view    = cudf::strings_column_view(cudf::slice(input, {1, 3}).front());
  results = cudf::strings::to_upper(view);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, cudf::slice(expected, {1, 3}).front());
}

TEST_F(StringsCaseTest, LongStringsSpecial)
{
  auto input = cudf::test::strings_column_wrapper(
    {"abcdéfghijklmnopqrstuvwxyzȺßCDÉFGHİJKLMNOPQRSTUVWXYZ1234567890!@#$%^&*()_+=- ",
     "ȺßCDÉFGHİJKLMNOPQRSTUVWXYZ1234567890!@#$%^&*()_+=-abcdéfghijklmnopqrstuvwxyz ",
     "bcdéfghijklmnopqrstuvwxyzȺßCDÉFGHİJKLMNOPQRSTUVWXYZ1234567890!@#$%^&*()_+=-a ",
     "cdéfghijklmnopqrstuvwxyzȺßCDÉFGHİJKLMNOPQRSTUVWXYZ1234567890!@#$%^&*()_+=-ab ",
     "défghijklmnopqrstuvwxyzȺßCDÉFGHİJKLMNOPQRSTUVWXYZ1234567890!@#$%^&*()_+=-abc ",
     "éfghijklmnopqrstuvwxyzȺßCDÉFGHİJKLMNOPQRSTUVWXYZ1234567890!@#$%^&*()_+=-abcd ",
     "",
     "",
     "",
     "",
     "",
     "ȺßCDÉFGHİJKLMNOPQRSTUVWXYZ1234567890!@#$%^&*()_+=-abcdéfghijklmnopqrstuvwxyz ",
     "bcdéfghijklmnopqrstuvwxyzȺßCDÉFGHİJKLMNOPQRSTUVWXYZ1234567890!@#$%^&*()_+=-a ",
     "cdéfghijklmnopqrstuvwxyzȺßCDÉFGHİJKLMNOPQRSTUVWXYZ1234567890!@#$%^&*()_+=-ab ",
     "défghijklmnopqrstuvwxyzȺßCDÉFGHİJKLMNOPQRSTUVWXYZ1234567890!@#$%^&*()_+=-abc ",
     "éfghijklmnopqrstuvwxyzȺßCDÉFGHİJKLMNOPQRSTUVWXYZ1234567890!@#$%^&*()_+=-abcd ",
     "ȺßCDÉFGHİJKLMNOPQRSTUVWXYZabcdéfghijklmnopqrstuvwxyz1234567890!@#$%^&*()_+=-"},
    {1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1});
  auto view     = cudf::strings_column_view(input);
  auto results  = cudf::strings::to_lower(view);
  auto expected = cudf::test::strings_column_wrapper(
    {"abcdéfghijklmnopqrstuvwxyzⱥßcdéfghi̇jklmnopqrstuvwxyz1234567890!@#$%^&*()_+=- ",
     "ⱥßcdéfghi̇jklmnopqrstuvwxyz1234567890!@#$%^&*()_+=-abcdéfghijklmnopqrstuvwxyz ",
     "bcdéfghijklmnopqrstuvwxyzⱥßcdéfghi̇jklmnopqrstuvwxyz1234567890!@#$%^&*()_+=-a ",
     "cdéfghijklmnopqrstuvwxyzⱥßcdéfghi̇jklmnopqrstuvwxyz1234567890!@#$%^&*()_+=-ab ",
     "défghijklmnopqrstuvwxyzⱥßcdéfghi̇jklmnopqrstuvwxyz1234567890!@#$%^&*()_+=-abc ",
     "éfghijklmnopqrstuvwxyzⱥßcdéfghi̇jklmnopqrstuvwxyz1234567890!@#$%^&*()_+=-abcd ",
     "",
     "",
     "",
     "",
     "",
     "ⱥßcdéfghi̇jklmnopqrstuvwxyz1234567890!@#$%^&*()_+=-abcdéfghijklmnopqrstuvwxyz ",
     "bcdéfghijklmnopqrstuvwxyzⱥßcdéfghi̇jklmnopqrstuvwxyz1234567890!@#$%^&*()_+=-a ",
     "cdéfghijklmnopqrstuvwxyzⱥßcdéfghi̇jklmnopqrstuvwxyz1234567890!@#$%^&*()_+=-ab ",
     "défghijklmnopqrstuvwxyzⱥßcdéfghi̇jklmnopqrstuvwxyz1234567890!@#$%^&*()_+=-abc ",
     "éfghijklmnopqrstuvwxyzⱥßcdéfghi̇jklmnopqrstuvwxyz1234567890!@#$%^&*()_+=-abcd ",
     "ⱥßcdéfghi̇jklmnopqrstuvwxyzabcdéfghijklmnopqrstuvwxyz1234567890!@#$%^&*()_+=-"},
    {1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
}

TEST_F(StringsCaseTest, EmptyStringsColumn)
{
  auto const zero_size_strings_column = cudf::make_empty_column(cudf::type_id::STRING)->view();

  auto strings_view = cudf::strings_column_view(zero_size_strings_column);

  auto results = cudf::strings::to_lower(strings_view);
  cudf::test::expect_column_empty(results->view());

  results = cudf::strings::to_upper(strings_view);
  cudf::test::expect_column_empty(results->view());

  results = cudf::strings::swapcase(strings_view);
  cudf::test::expect_column_empty(results->view());

  results = cudf::strings::capitalize(strings_view);
  cudf::test::expect_column_empty(results->view());

  results = cudf::strings::title(strings_view);
  cudf::test::expect_column_empty(results->view());
}

TEST_F(StringsCaseTest, ErrorTest)
{
  cudf::test::strings_column_wrapper input{"the column intentionally left blank"};
  auto view = cudf::strings_column_view(input);

  EXPECT_THROW(cudf::strings::capitalize(view, cudf::string_scalar("", false)), cudf::logic_error);
}
