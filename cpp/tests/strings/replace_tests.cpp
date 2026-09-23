/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/iterator_utilities.hpp>

#include <cudf/column/column.hpp>
#include <cudf/strings/combine.hpp>
#include <cudf/strings/replace.hpp>
#include <cudf/strings/strings_column_view.hpp>

#include <cuda/iterator>

#include <stdexcept>
#include <vector>

struct StringsReplaceTest : public cudf::test::BaseFixture {
  cudf::test::strings_column_wrapper build_corpus()
  {
    std::vector<char const*> h_strings{"the quick brown fox jumps over the lazy dog",
                                       "the fat cat lays next to the other accénted cat",
                                       "a slow moving turtlé cannot catch the bird",
                                       "which can be composéd together to form a more complete",
                                       "The result does not include the value in the sum in",
                                       "",
                                       nullptr};

    return {h_strings.begin(),
            h_strings.end(),
            cuda::transform_iterator(h_strings.begin(), [](auto str) { return str != nullptr; })};
  }

  std::unique_ptr<cudf::column> build_large(cudf::column_view const& first,
                                            cudf::column_view const& remaining)
  {
    return cudf::strings::concatenate(cudf::table_view(
      {first, remaining, remaining, remaining, remaining, remaining, remaining, remaining}));
  }
};

TEST_F(StringsReplaceTest, Replace)
{
  auto input        = build_corpus();
  auto strings_view = cudf::strings_column_view(input);
  // replace all occurrences of 'the ' with '++++ '
  std::vector<char const*> h_expected{"++++ quick brown fox jumps over ++++ lazy dog",
                                      "++++ fat cat lays next to ++++ other accénted cat",
                                      "a slow moving turtlé cannot catch ++++ bird",
                                      "which can be composéd together to form a more complete",
                                      "The result does not include ++++ value in ++++ sum in",
                                      "",
                                      nullptr};
  cudf::test::strings_column_wrapper expected(
    h_expected.begin(), h_expected.end(), cudf::test::iterators::nulls_from_nullptrs(h_expected));

  auto target      = cudf::string_scalar("the ");
  auto replacement = cudf::string_scalar("++++ ");

  auto results = cudf::strings::replace(strings_view, target, replacement);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);

  auto input_large    = build_large(input, input);
  strings_view        = cudf::strings_column_view(input_large->view());
  auto expected_large = build_large(expected, expected);
  results             = cudf::strings::replace(strings_view, target, replacement);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, *expected_large);
}

TEST_F(StringsReplaceTest, ReplaceReplLimit)
{
  auto input        = build_corpus();
  auto strings_view = cudf::strings_column_view(input);

  // only remove the first occurrence of 'the '
  std::vector<char const*> h_expected{"quick brown fox jumps over the lazy dog",
                                      "fat cat lays next to the other accénted cat",
                                      "a slow moving turtlé cannot catch bird",
                                      "which can be composéd together to form a more complete",
                                      "The result does not include value in the sum in",
                                      "",
                                      nullptr};
  cudf::test::strings_column_wrapper expected(
    h_expected.begin(), h_expected.end(), cudf::test::iterators::nulls_from_nullptrs(h_expected));
  auto target      = cudf::string_scalar("the ");
  auto replacement = cudf::string_scalar("");
  auto results     = cudf::strings::replace(strings_view, target, replacement, 1);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);

  auto input_large    = build_large(input, input);
  strings_view        = cudf::strings_column_view(input_large->view());
  auto expected_large = build_large(expected, input);
  results             = cudf::strings::replace(strings_view, target, replacement, 1);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, *expected_large);
}

TEST_F(StringsReplaceTest, ReplaceReplLimitInputSliced)
{
  auto input = build_corpus();
  // replace first two occurrences of ' ' with '--'
  std::vector<char const*> h_expected{"the--quick--brown fox jumps over the lazy dog",
                                      "the--fat--cat lays next to the other accénted cat",
                                      "a--slow--moving turtlé cannot catch the bird",
                                      "which--can--be composéd together to form a more complete",
                                      "The--result--does not include the value in the sum in",
                                      "",
                                      nullptr};
  cudf::test::strings_column_wrapper expected(
    h_expected.begin(), h_expected.end(), cudf::test::iterators::nulls_from_nullptrs(h_expected));
  std::vector<cudf::size_type> slice_indices{0, 2, 2, 3, 3, 7};
  auto sliced_strings  = cudf::slice(input, slice_indices);
  auto sliced_expected = cudf::slice(expected, slice_indices);

  auto input_large    = build_large(input, input);
  auto expected_large = build_large(expected, input);

  auto sliced_large          = cudf::slice(input_large->view(), slice_indices);
  auto sliced_expected_large = cudf::slice(expected_large->view(), slice_indices);

  auto target      = cudf::string_scalar(" ");
  auto replacement = cudf::string_scalar("--");

  for (size_t i = 0; i < sliced_strings.size(); ++i) {
    auto strings_view = cudf::strings_column_view(sliced_strings[i]);
    auto results      = cudf::strings::replace(strings_view, target, replacement, 2);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, sliced_expected[i]);

    strings_view = cudf::strings_column_view(sliced_large[i]);
    results =
      cudf::strings::replace(strings_view, cudf::string_scalar(" "), cudf::string_scalar("--"), 2);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, sliced_expected_large[i]);
  }
}

TEST_F(StringsReplaceTest, ReplaceTargetOverlap)
{
  auto corpus      = build_corpus();
  auto corpus_view = cudf::strings_column_view(corpus);
  // replace all occurrences of 'the ' with '+++++++ '
  auto input = cudf::strings::replace(
    corpus_view, cudf::string_scalar("the "), cudf::string_scalar("++++++++ "));
  auto strings_view = cudf::strings_column_view(*input);
  // replace all occurrences of '+++' with 'plus '
  std::vector<char const*> h_expected{
    "plus plus ++ quick brown fox jumps over plus plus ++ lazy dog",
    "plus plus ++ fat cat lays next to plus plus ++ other accénted cat",
    "a slow moving turtlé cannot catch plus plus ++ bird",
    "which can be composéd together to form a more complete",
    "The result does not include plus plus ++ value in plus plus ++ sum in",
    "",
    nullptr};
  cudf::test::strings_column_wrapper expected(
    h_expected.begin(), h_expected.end(), cudf::test::iterators::nulls_from_nullptrs(h_expected));

  auto target      = cudf::string_scalar("+++");
  auto replacement = cudf::string_scalar("plus ");

  auto results = cudf::strings::replace(strings_view, target, replacement);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);

  auto input_large    = build_large(input->view(), input->view());
  strings_view        = cudf::strings_column_view(input_large->view());
  auto expected_large = build_large(expected, expected);

  results = cudf::strings::replace(strings_view, target, replacement);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, *expected_large);
}

TEST_F(StringsReplaceTest, ReplaceTargetOverlapsStrings)
{
  auto input        = build_corpus();
  auto strings_view = cudf::strings_column_view(input);

  // replace all occurrences of 'dogthe' with '+'
  auto target      = cudf::string_scalar("dogthe");
  auto replacement = cudf::string_scalar("+");

  // should not replace anything unless it incorrectly matches across a string boundary
  auto results = cudf::strings::replace(strings_view, target, replacement);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, input);

  auto input_large = cudf::strings::concatenate(
    cudf::table_view({input, input, input, input, input, input, input, input}),
    cudf::string_scalar(" "));
  strings_view = cudf::strings_column_view(input_large->view());
  results      = cudf::strings::replace(strings_view, target, replacement);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, *input_large);
}

TEST_F(StringsReplaceTest, ReplaceAllNullInput)
{
  std::vector<char const*> h_null_strings(128);
  auto input = cudf::test::strings_column_wrapper(
    h_null_strings.begin(), h_null_strings.end(), cuda::make_constant_iterator(false));
  auto strings_view = cudf::strings_column_view(input);
  auto results =
    cudf::strings::replace(strings_view, cudf::string_scalar("+"), cudf::string_scalar(""));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, input);
}

TEST_F(StringsReplaceTest, ReplaceEndOfString)
{
  auto input        = build_corpus();
  auto strings_view = cudf::strings_column_view(input);

  // replace all occurrences of 'in' with  ' '
  std::vector<char const*> h_expected{"the quick brown fox jumps over the lazy dog",
                                      "the fat cat lays next to the other accénted cat",
                                      "a slow mov g turtlé cannot catch the bird",
                                      "which can be composéd together to form a more complete",
                                      "The result does not  clude the value   the sum  ",
                                      "",
                                      nullptr};

  cudf::test::strings_column_wrapper expected(
    h_expected.begin(), h_expected.end(), cudf::test::iterators::nulls_from_nullptrs(h_expected));

  auto target      = cudf::string_scalar("in");
  auto replacement = cudf::string_scalar(" ");

  auto results = cudf::strings::replace(strings_view, target, replacement);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);

  auto input_large    = build_large(input, input);
  strings_view        = cudf::strings_column_view(input_large->view());
  auto expected_large = build_large(expected, expected);
  results             = cudf::strings::replace(strings_view, target, replacement);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, *expected_large);
}

TEST_F(StringsReplaceTest, ReplaceAdjacentMultiByteTarget)
{
  auto input        = cudf::test::strings_column_wrapper({"ééééééééééééééééééééé",
                                                          "eéeéeéeeéeéeéeeéeéeée",
                                                          "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee"});
  auto strings_view = cudf::strings_column_view(input);
  // replace all occurrences of 'é' with 'e'
  cudf::test::strings_column_wrapper expected({"eeeeeeeeeeeeeeeeeeeee",
                                               "eeeeeeeeeeeeeeeeeeeee",
                                               "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee"});

  auto target      = cudf::string_scalar("é");
  auto replacement = cudf::string_scalar("e");

  auto results = cudf::strings::replace(strings_view, target, replacement);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);

  auto input_large    = build_large(input, input);
  strings_view        = cudf::strings_column_view(input_large->view());
  auto expected_large = build_large(expected, expected);
  results             = cudf::strings::replace(strings_view, target, replacement);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, *expected_large);
}

TEST_F(StringsReplaceTest, ReplaceErrors)
{
  auto input = cudf::test::strings_column_wrapper({"this column intentionally left blank"});

  auto target      = cudf::string_scalar(" ");
  auto replacement = cudf::string_scalar("_");
  auto null_input  = cudf::string_scalar("", false);
  auto empty_input = cudf::string_scalar("");
  auto sv          = cudf::strings_column_view(input);

  EXPECT_THROW(cudf::strings::replace(sv, target, null_input), cudf::logic_error);
  EXPECT_THROW(cudf::strings::replace(sv, null_input, replacement), cudf::logic_error);
  EXPECT_THROW(cudf::strings::replace(sv, empty_input, replacement), cudf::logic_error);

  auto const empty       = cudf::test::strings_column_wrapper();
  auto const ev          = cudf::strings_column_view(empty);
  auto const targets     = cudf::test::strings_column_wrapper({"x"});
  auto const tv          = cudf::strings_column_view(targets);
  auto const target_null = cudf::test::strings_column_wrapper({""}, {0});
  auto const tv_null     = cudf::strings_column_view(target_null);
  auto const repls       = cudf::test::strings_column_wrapper({"y", "z"});
  auto const rv          = cudf::strings_column_view(repls);
  auto const repl_null   = cudf::test::strings_column_wrapper({""}, {0});
  auto const rv_null     = cudf::strings_column_view(repl_null);

  EXPECT_THROW(cudf::strings::replace_multiple(sv, ev, rv), cudf::logic_error);
  EXPECT_THROW(cudf::strings::replace_multiple(sv, tv_null, rv), cudf::logic_error);
  EXPECT_THROW(cudf::strings::replace_multiple(sv, tv, ev), cudf::logic_error);
  EXPECT_THROW(cudf::strings::replace_multiple(sv, tv, rv_null), cudf::logic_error);
  EXPECT_THROW(cudf::strings::replace_multiple(sv, tv, rv), cudf::logic_error);
}

TEST_F(StringsReplaceTest, ReplaceSlice)
{
  std::vector<char const*> h_strings{"Héllo", "thesé", nullptr, "ARE THE", "tést strings", ""};

  cudf::test::strings_column_wrapper strings(
    h_strings.begin(), h_strings.end(), cuda::transform_iterator(h_strings.begin(), [](auto str) {
      return str != nullptr;
    }));
  auto strings_view = cudf::strings_column_view(strings);

  {
    auto results = cudf::strings::replace_slice(strings_view, cudf::string_scalar("___"), 2, 3);
    std::vector<char const*> h_expected{
      "Hé___lo", "th___sé", nullptr, "AR___ THE", "té___t strings", "___"};
    cudf::test::strings_column_wrapper expected(
      h_expected.begin(),
      h_expected.end(),
      cuda::transform_iterator(h_expected.begin(), [](auto str) { return str != nullptr; }));
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
  {
    auto results = cudf::strings::replace_slice(strings_view, cudf::string_scalar("||"), 3, 3);
    std::vector<char const*> h_expected{
      "Hél||lo", "the||sé", nullptr, "ARE|| THE", "tés||t strings", "||"};
    cudf::test::strings_column_wrapper expected(
      h_expected.begin(),
      h_expected.end(),
      cuda::transform_iterator(h_expected.begin(), [](auto str) { return str != nullptr; }));
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
  {
    auto results = cudf::strings::replace_slice(strings_view, cudf::string_scalar("x"), -1, -1);
    std::vector<char const*> h_expected{
      "Héllox", "theséx", nullptr, "ARE THEx", "tést stringsx", "x"};
    cudf::test::strings_column_wrapper expected(
      h_expected.begin(),
      h_expected.end(),
      cuda::transform_iterator(h_expected.begin(), [](auto str) { return str != nullptr; }));
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
}

TEST_F(StringsReplaceTest, ReplaceSliceError)
{
  cudf::test::strings_column_wrapper input({"Héllo", "thesé", "are not", "important", ""});
  EXPECT_THROW(
    cudf::strings::replace_slice(cudf::strings_column_view(input), cudf::string_scalar(""), 4, 1),
    cudf::logic_error);
}

TEST_F(StringsReplaceTest, ReplaceMulti)
{
  auto input        = build_corpus();
  auto strings_view = cudf::strings_column_view(input);

  cudf::test::strings_column_wrapper targets({"the ", "a ", "to "});
  auto targets_view = cudf::strings_column_view(targets);

  {
    cudf::test::strings_column_wrapper repls({"_ ", "A ", "2 "});
    auto repls_view = cudf::strings_column_view(repls);

    auto results = cudf::strings::replace_multiple(strings_view, targets_view, repls_view);

    std::vector<char const*> h_expected{"_ quick brown fox jumps over _ lazy dog",
                                        "_ fat cat lays next 2 _ other accénted cat",
                                        "A slow moving turtlé cannot catch _ bird",
                                        "which can be composéd together 2 form A more complete",
                                        "The result does not include _ value in _ sum in",
                                        "",
                                        nullptr};
    cudf::test::strings_column_wrapper expected(
      h_expected.begin(),
      h_expected.end(),
      cuda::transform_iterator(h_expected.begin(), [](auto str) { return str != nullptr; }));
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }

  {
    cudf::test::strings_column_wrapper repls({"* "});
    auto repls_view = cudf::strings_column_view(repls);

    auto results = cudf::strings::replace_multiple(strings_view, targets_view, repls_view);

    std::vector<char const*> h_expected{"* quick brown fox jumps over * lazy dog",
                                        "* fat cat lays next * * other accénted cat",
                                        "* slow moving turtlé cannot catch * bird",
                                        "which can be composéd together * form * more complete",
                                        "The result does not include * value in * sum in",
                                        "",
                                        nullptr};
    cudf::test::strings_column_wrapper expected(
      h_expected.begin(),
      h_expected.end(),
      cuda::transform_iterator(h_expected.begin(), [](auto str) { return str != nullptr; }));
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
}

TEST_F(StringsReplaceTest, ReplaceMultiLong)
{
  // The length of the strings are to trigger the code path governed by the
  // AVG_CHAR_BYTES_THRESHOLD setting in the multi.cu.
  auto input = cudf::test::strings_column_wrapper(
    {"This string needs to be very long to trigger the long-replace internal functions. "
     "This string needs to be very long to trigger the long-replace internal functions. "
     "This string needs to be very long to trigger the long-replace internal functions. "
     "This string needs to be very long to trigger the long-replace internal functions.",
     "0123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890"
     "12"
     "3456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123"
     "45"
     "6789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456"
     "78"
     "9012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789"
     "01"
     "2345678901234567890123456789",
     "0123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890"
     "12"
     "3456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123"
     "45"
     "6789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456"
     "78"
     "9012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789"
     "01"
     "2345678901234567890123456789",
     "Test string for overlap check: bananaápple bananá ápplebananá banápple ápple bananá "
     "Test string for overlap check: bananaápple bananá ápplebananá banápple ápple bananá "
     "Test string for overlap check: bananaápple bananá ápplebananá banápple ápple bananá "
     "Test string for overlap check: bananaápple bananá ápplebananá banápple ápple bananá "
     "Test string for overlap check: bananaápple bananá ápplebananá banápple ápple bananá",
     "",
     ""},
    {true, true, true, true, false, true});
  auto strings_view = cudf::strings_column_view(input);

  auto targets      = cudf::test::strings_column_wrapper({"78901", "bananá", "ápple", "78"});
  auto targets_view = cudf::strings_column_view(targets);

  {
    cudf::test::strings_column_wrapper repls({"x", "PEAR", "avocado", "$$"});
    auto repls_view = cudf::strings_column_view(repls);

    auto results = cudf::strings::replace_multiple(strings_view, targets_view, repls_view);

    cudf::test::strings_column_wrapper expected(
      {"This string needs to be very long to trigger the long-replace internal functions. "
       "This string needs to be very long to trigger the long-replace internal functions. "
       "This string needs to be very long to trigger the long-replace internal functions. "
       "This string needs to be very long to trigger the long-replace internal functions.",
       "0123456x23456x23456x23456x23456x23456x23456x23456x23456x23456x23456x23456x23456x23456x234"
       "56"
       "x23456x23456x23456x23456x23456x23456x23456x23456x23456x23456x23456x23456x23456x23456x2345"
       "6x"
       "23456x23456x23456x23456x23456x23456x23456x23456x23456x23456$$9",
       "0123456x23456x23456x23456x23456x23456x23456x23456x23456x23456x23456x23456x23456x23456x234"
       "56"
       "x23456x23456x23456x23456x23456x23456x23456x23456x23456x23456x23456x23456x23456x23456x2345"
       "6x"
       "23456x23456x23456x23456x23456x23456x23456x23456x23456x23456$$9",
       "Test string for overlap check: bananaavocado PEAR avocadoPEAR banavocado avocado PEAR "
       "Test string for overlap check: bananaavocado PEAR avocadoPEAR banavocado avocado PEAR "
       "Test string for overlap check: bananaavocado PEAR avocadoPEAR banavocado avocado PEAR "
       "Test string for overlap check: bananaavocado PEAR avocadoPEAR banavocado avocado PEAR "
       "Test string for overlap check: bananaavocado PEAR avocadoPEAR banavocado avocado PEAR",
       "",
       ""},
      {true, true, true, true, false, true});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }

  {
    cudf::test::strings_column_wrapper repls({"*"});
    auto repls_view = cudf::strings_column_view(repls);

    auto results = cudf::strings::replace_multiple(strings_view, targets_view, repls_view);

    cudf::test::strings_column_wrapper expected(
      {"This string needs to be very long to trigger the long-replace internal functions. "
       "This string needs to be very long to trigger the long-replace internal functions. "
       "This string needs to be very long to trigger the long-replace internal functions. "
       "This string needs to be very long to trigger the long-replace internal functions.",
       "0123456*23456*23456*23456*23456*23456*23456*23456*23456*23456*23456*23456*23456*23456*"
       "23456*23456*23456*23456*23456*23456*23456*23456*23456*23456*23456*23456*23456*23456*"
       "23456*23456*23456*23456*23456*23456*23456*23456*23456*23456*23456*23456*9",
       "0123456*23456*23456*23456*23456*23456*23456*23456*23456*23456*23456*23456*23456*23456*"
       "23456*23456*23456*23456*23456*23456*23456*23456*23456*23456*23456*23456*23456*23456*"
       "23456*23456*23456*23456*23456*23456*23456*23456*23456*23456*23456*23456*9",
       "Test string for overlap check: banana* * ** ban* * * Test string for overlap check: "
       "banana* * ** ban* * * Test string for overlap check: banana* * ** ban* * * Test string "
       "for "
       "overlap check: banana* * ** ban* * * Test string for overlap check: banana* * ** ban* * "
       "*",
       "",
       ""},
      {true, true, true, true, false, true});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }

  {
    targets =
      cudf::test::strings_column_wrapper({"01234567890123456789012345678901234567890123456789012345"
                                          "6789012345678901234567890123456789012"
                                          "34567890123456789012345678901234567890123456789012345678"
                                          "9012345678901234567890123456789012345"
                                          "67890123456789012345678901234567890123456789012345678901"
                                          "2345678901234567890123456789012345678"
                                          "90123456789012345678901234567890123456789012345678901234"
                                          "5678901234567890123456789012345678901"
                                          "2345678901234567890123456789",
                                          "78"});
    targets_view    = cudf::strings_column_view(targets);
    auto repls      = cudf::test::strings_column_wrapper({""});
    auto repls_view = cudf::strings_column_view(repls);

    auto results = cudf::strings::replace_multiple(strings_view, targets_view, repls_view);

    cudf::test::strings_column_wrapper expected(
      {"This string needs to be very long to trigger the long-replace internal functions. "
       "This string needs to be very long to trigger the long-replace internal functions. "
       "This string needs to be very long to trigger the long-replace internal functions. "
       "This string needs to be very long to trigger the long-replace internal functions.",
       "",
       "",
       "Test string for overlap check: bananaápple bananá ápplebananá banápple ápple bananá "
       "Test string for overlap check: bananaápple bananá ápplebananá banápple ápple bananá "
       "Test string for overlap check: bananaápple bananá ápplebananá banápple ápple bananá "
       "Test string for overlap check: bananaápple bananá ápplebananá banápple ápple bananá "
       "Test string for overlap check: bananaápple bananá ápplebananá banápple ápple bananá",
       "",
       ""},
      {true, true, true, true, false, true});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
}

TEST_F(StringsReplaceTest, EmptyTarget)
{
  auto const input = cudf::test::strings_column_wrapper({"hello", "world", "", "accénted"});
  auto const sv    = cudf::strings_column_view(input);

  auto const targets = cudf::test::strings_column_wrapper({"e", "", "d"});
  auto const tv      = cudf::strings_column_view(targets);

  auto const repls = cudf::test::strings_column_wrapper({"E", "_", "D"});
  auto const rv    = cudf::strings_column_view(repls);

  // empty target should be ignored
  auto results  = cudf::strings::replace_multiple(sv, tv, rv);
  auto expected = cudf::test::strings_column_wrapper({"hEllo", "worlD", "", "accéntED"});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(results->view(), expected);
}

TEST_F(StringsReplaceTest, ReplaceColumnBasic)
{
  // Basic per-row replacement: each row uses its own target/repl pair
  auto const input   = cudf::test::strings_column_wrapper({"hello world", "foo bar", "aaa", ""});
  auto const targets = cudf::test::strings_column_wrapper({"o", "bar", "a", "x"});
  auto const repls   = cudf::test::strings_column_wrapper({"0", "BAR", "X", "y"});
  auto const iv      = cudf::strings_column_view(input);
  auto const tv      = cudf::strings_column_view(targets);
  auto const rv      = cudf::strings_column_view(repls);

  auto const result   = cudf::strings::replace(iv, tv, rv);
  auto const expected = cudf::test::strings_column_wrapper({"hell0 w0rld", "foo BAR", "XXX", ""});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result, expected);
}

TEST_F(StringsReplaceTest, ReplaceColumnNullInput)
{
  // Null in input[i] → null output row; non-null rows are processed normally
  auto const input   = cudf::test::strings_column_wrapper({"hello", "", "foo"}, {1, 0, 1});
  auto const targets = cudf::test::strings_column_wrapper({"l", "o", "o"});
  auto const repls   = cudf::test::strings_column_wrapper({"L", "0", "0"});
  auto const iv      = cudf::strings_column_view(input);
  auto const tv      = cudf::strings_column_view(targets);
  auto const rv      = cudf::strings_column_view(repls);

  auto const result   = cudf::strings::replace(iv, tv, rv);
  auto const expected = cudf::test::strings_column_wrapper({"heLLo", "", "f00"}, {1, 0, 1});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result, expected);
}

TEST_F(StringsReplaceTest, ReplaceColumnNullTarget)
{
  // Null in targets[i] → null output row; other rows unaffected
  auto const input   = cudf::test::strings_column_wrapper({"hello", "world", "foo"});
  auto const targets = cudf::test::strings_column_wrapper({"l", "", "o"}, {1, 0, 1});
  auto const repls   = cudf::test::strings_column_wrapper({"L", "0", "0"});
  auto const iv      = cudf::strings_column_view(input);
  auto const tv      = cudf::strings_column_view(targets);
  auto const rv      = cudf::strings_column_view(repls);

  auto const result   = cudf::strings::replace(iv, tv, rv);
  auto const expected = cudf::test::strings_column_wrapper({"heLLo", "", "f00"}, {1, 0, 1});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result, expected);
}

TEST_F(StringsReplaceTest, ReplaceColumnNullRepl)
{
  // Null in repls[i] → null output row; other rows unaffected
  auto const input   = cudf::test::strings_column_wrapper({"hello", "world", "foo"});
  auto const targets = cudf::test::strings_column_wrapper({"l", "o", "o"});
  auto const repls   = cudf::test::strings_column_wrapper({"L", "", "0"}, {1, 0, 1});
  auto const iv      = cudf::strings_column_view(input);
  auto const tv      = cudf::strings_column_view(targets);
  auto const rv      = cudf::strings_column_view(repls);

  auto const result   = cudf::strings::replace(iv, tv, rv);
  auto const expected = cudf::test::strings_column_wrapper({"heLLo", "", "f00"}, {1, 0, 1});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result, expected);
}

TEST_F(StringsReplaceTest, ReplaceColumnEmptyTarget)
{
  // Empty targets[i] → input[i] copied unchanged (no replacement)
  auto const input   = cudf::test::strings_column_wrapper({"hello", "world", "foo"});
  auto const targets = cudf::test::strings_column_wrapper({"l", "", "o"});
  auto const repls   = cudf::test::strings_column_wrapper({"L", "X", "0"});
  auto const iv      = cudf::strings_column_view(input);
  auto const tv      = cudf::strings_column_view(targets);
  auto const rv      = cudf::strings_column_view(repls);

  auto const result   = cudf::strings::replace(iv, tv, rv);
  auto const expected = cudf::test::strings_column_wrapper({"heLLo", "world", "f00"});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result, expected);
}

TEST_F(StringsReplaceTest, ReplaceColumnMismatchedSizes)
{
  // targets.size() != input.size() → throws
  auto const input   = cudf::test::strings_column_wrapper({"hello", "world"});
  auto const targets = cudf::test::strings_column_wrapper({"l"});
  auto const repls   = cudf::test::strings_column_wrapper({"L"});
  auto const iv      = cudf::strings_column_view(input);
  auto const tv      = cudf::strings_column_view(targets);
  auto const rv      = cudf::strings_column_view(repls);

  EXPECT_THROW(cudf::strings::replace(iv, tv, rv), std::invalid_argument);

  // repls.size() != input.size() → throws
  auto const targets2 = cudf::test::strings_column_wrapper({"l", "o"});
  auto const repls2   = cudf::test::strings_column_wrapper({"L"});
  auto const tv2      = cudf::strings_column_view(targets2);
  auto const rv2      = cudf::strings_column_view(repls2);

  EXPECT_THROW(cudf::strings::replace(iv, tv2, rv2), std::invalid_argument);
}

TEST_F(StringsReplaceTest, ReplaceColumnAllNullInput)
{
  // All-null input → all-null output
  auto const input   = cudf::test::strings_column_wrapper({"", "", ""}, {0, 0, 0});
  auto const targets = cudf::test::strings_column_wrapper({"a", "b", "c"});
  auto const repls   = cudf::test::strings_column_wrapper({"x", "y", "z"});
  auto const iv      = cudf::strings_column_view(input);
  auto const tv      = cudf::strings_column_view(targets);
  auto const rv      = cudf::strings_column_view(repls);

  auto const result = cudf::strings::replace(iv, tv, rv);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result, input);
}

TEST_F(StringsReplaceTest, ReplaceColumnEmpty)
{
  // Empty input column → empty output column
  auto const input   = cudf::test::strings_column_wrapper();
  auto const targets = cudf::test::strings_column_wrapper();
  auto const repls   = cudf::test::strings_column_wrapper();
  auto const iv      = cudf::strings_column_view(input);
  auto const tv      = cudf::strings_column_view(targets);
  auto const rv      = cudf::strings_column_view(repls);

  auto const result = cudf::strings::replace(iv, tv, rv);
  cudf::test::expect_column_empty(result->view());
}

TEST_F(StringsReplaceTest, ReplaceColumnEmptyInputMismatchedSizes)
{
  // Empty input with mismatched targets/repls sizes must throw, not silently succeed.
  // The size contract (targets.size() == input.size()) holds even when input is empty.
  auto const empty   = cudf::test::strings_column_wrapper();
  auto const nonzero = cudf::test::strings_column_wrapper({"x"});
  auto const ev      = cudf::strings_column_view(empty);
  auto const nv      = cudf::strings_column_view(nonzero);

  EXPECT_THROW(cudf::strings::replace(ev, nv, ev), std::invalid_argument);
  EXPECT_THROW(cudf::strings::replace(ev, ev, nv), std::invalid_argument);
}

TEST_F(StringsReplaceTest, ReplaceColumnCombinedNulls)
{
  // Nulls at different positions across input, targets, and repls.
  // Output validity is the bitwise AND of all three masks.
  // Row 0: input=valid, targets=valid, repls=valid  → valid, "heLLo"
  // Row 1: input=null,  targets=valid, repls=valid  → null  (input null)
  // Row 2: input=valid, targets=null,  repls=valid  → null  (target null; exercises Issue #1)
  // Row 3: input=valid, targets=valid, repls=null   → null  (repl null)
  auto const input = cudf::test::strings_column_wrapper({"hello", "", "foo", "bar"}, {1, 0, 1, 1});
  auto const targets = cudf::test::strings_column_wrapper({"l", "o", "", "a"}, {1, 1, 0, 1});
  auto const repls   = cudf::test::strings_column_wrapper({"L", "0", "0", ""}, {1, 1, 1, 0});
  auto const iv      = cudf::strings_column_view(input);
  auto const tv      = cudf::strings_column_view(targets);
  auto const rv      = cudf::strings_column_view(repls);

  auto const result   = cudf::strings::replace(iv, tv, rv);
  auto const expected = cudf::test::strings_column_wrapper({"heLLo", "", "", ""}, {1, 0, 0, 0});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result, expected);
}

TEST_F(StringsReplaceTest, ReplaceColumnUtf8)
{
  // Multi-byte UTF-8 characters in targets and replacements.
  // Verifies that byte-level replacement works correctly at UTF-8 character boundaries.
  auto const input   = cudf::test::strings_column_wrapper({"café", "naïve", "résumé"});
  auto const targets = cudf::test::strings_column_wrapper({"é", "ï", "é"});
  auto const repls   = cudf::test::strings_column_wrapper({"e", "i", "e"});
  auto const iv      = cudf::strings_column_view(input);
  auto const tv      = cudf::strings_column_view(targets);
  auto const rv      = cudf::strings_column_view(repls);

  auto const result   = cudf::strings::replace(iv, tv, rv);
  auto const expected = cudf::test::strings_column_wrapper({"cafe", "naive", "resume"});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result, expected);
}

TEST_F(StringsReplaceTest, ReplaceColumnTargetLongerThanInput)
{
  // targets[i] longer than input[i]: the fit check must prevent any match,
  // leaving the input row unchanged.
  auto const input   = cudf::test::strings_column_wrapper({"hi", "ab", ""});
  auto const targets = cudf::test::strings_column_wrapper({"hello", "abcd", "x"});
  auto const repls   = cudf::test::strings_column_wrapper({"x", "y", "z"});
  auto const iv      = cudf::strings_column_view(input);
  auto const tv      = cudf::strings_column_view(targets);
  auto const rv      = cudf::strings_column_view(repls);

  auto const result   = cudf::strings::replace(iv, tv, rv);
  auto const expected = cudf::test::strings_column_wrapper({"hi", "ab", ""});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result, expected);
}

TEST_F(StringsReplaceTest, EmptyStringsColumn)
{
  auto const zero_size_strings_column = cudf::make_empty_column(cudf::type_id::STRING)->view();

  auto strings_view = cudf::strings_column_view(zero_size_strings_column);
  auto results      = cudf::strings::replace(
    strings_view, cudf::string_scalar("not"), cudf::string_scalar("pertinent"));
  cudf::test::expect_column_empty(results->view());

  auto const target      = cudf::test::strings_column_wrapper({"x"});
  auto const target_view = cudf::strings_column_view(target);
  results                = cudf::strings::replace_multiple(strings_view, target_view, target_view);
  cudf::test::expect_column_empty(results->view());
}
