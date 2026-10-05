/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/iterator_utilities.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/scalar/scalar.hpp>
#include <cudf/strings/repeat_strings.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/strings/utilities.hpp>

using namespace cudf::test::iterators;

namespace {
using strs_col   = cudf::test::strings_column_wrapper;
using int32s_col = cudf::test::fixed_width_column_wrapper<int32_t>;

constexpr int32_t null{0};  // mark for null elements in a column of int32_t values
constexpr cudf::test::debug_output_level verbosity{cudf::test::debug_output_level::FIRST_ERROR};
}  // namespace

struct RepeatStringsTest : public cudf::test::BaseFixture {};

template <typename T>
struct RepeatStringsTypedTest : public cudf::test::BaseFixture {};

// Test for signed types only, as we will need to use non-positive values.
using TypesForTest = cudf::test::Types<int8_t, int16_t, int32_t, int64_t>;
TYPED_TEST_SUITE(RepeatStringsTypedTest, TypesForTest);

TYPED_TEST(RepeatStringsTypedTest, InvalidStringScalar)
{
  auto const str    = cudf::string_scalar("", false);
  auto const result = cudf::strings::repeat_string(str, 3);
  EXPECT_EQ(result->is_valid(), false);
}

TYPED_TEST(RepeatStringsTypedTest, ZeroSizeStringScalar)
{
  auto const str    = cudf::string_scalar("");
  auto const result = cudf::strings::repeat_string(str, 3);
  EXPECT_EQ(result->is_valid(), true);
  EXPECT_EQ(result->size(), 0);
}

TYPED_TEST(RepeatStringsTypedTest, ValidStringScalar)
{
  auto const str = cudf::string_scalar("abc123xyz-");

  {
    auto const result        = cudf::strings::repeat_string(str, 3);
    auto const expected_strs = cudf::string_scalar("abc123xyz-abc123xyz-abc123xyz-");
    CUDF_TEST_EXPECT_EQUAL_BUFFERS(expected_strs.data(), result->data(), expected_strs.size());
  }

  // Repeat once.
  {
    auto const result = cudf::strings::repeat_string(str, 1);
    CUDF_TEST_EXPECT_EQUAL_BUFFERS(str.data(), result->data(), str.size());
  }

  // Zero repeat times.
  {
    auto const result = cudf::strings::repeat_string(str, 0);
    EXPECT_EQ(result->is_valid(), true);
    EXPECT_EQ(result->size(), 0);
  }

  // Negative repeat times.
  {
    auto const result = cudf::strings::repeat_string(str, -10);
    EXPECT_EQ(result->is_valid(), true);
    EXPECT_EQ(result->size(), 0);
  }

  // Repeat too many times.
  {
    EXPECT_THROW(cudf::strings::repeat_string(str, std::numeric_limits<int32_t>::max() / 2),
                 std::overflow_error);
  }
}

TYPED_TEST(RepeatStringsTypedTest, ZeroSizeStringsColumnWithScalarRepeatTimes)
{
  auto const strs    = strs_col{};
  auto const results = cudf::strings::repeat_strings(cudf::strings_column_view(strs), 10);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(strs, *results, verbosity);
}

TYPED_TEST(RepeatStringsTypedTest, ZeroSizeStringsColumnWithColumnRepeatTimes)
{
  using ints_col = cudf::test::fixed_width_column_wrapper<TypeParam>;

  auto const strs         = strs_col{};
  auto const repeat_times = ints_col{};
  auto const results = cudf::strings::repeat_strings(cudf::strings_column_view(strs), repeat_times);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(strs, *results, verbosity);
}

TYPED_TEST(RepeatStringsTypedTest, AllEmptyStringsColumnWithScalarRepeatTimes)
{
  auto const strs    = strs_col{"", "", "", "", ""};
  auto const results = cudf::strings::repeat_strings(cudf::strings_column_view(strs), 10);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(strs, *results, verbosity);
}

TYPED_TEST(RepeatStringsTypedTest, AllEmptyStringsColumnWithColumnRepeatTimes)
{
  using ints_col = cudf::test::fixed_width_column_wrapper<TypeParam>;

  auto const strs         = strs_col{"", "", "", "", ""};
  auto const repeat_times = ints_col{-2, -1, 0, 1, 2};
  auto const results = cudf::strings::repeat_strings(cudf::strings_column_view(strs), repeat_times);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(strs, *results, verbosity);
}

TYPED_TEST(RepeatStringsTypedTest, AllNullStringsColumnWithScalarRepeatTimes)
{
  auto const strs    = strs_col{{"" /*NULL*/, "" /*NULL*/, "" /*NULL*/}, all_nulls()};
  auto const results = cudf::strings::repeat_strings(cudf::strings_column_view(strs), 10);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(strs, *results, verbosity);
}

TYPED_TEST(RepeatStringsTypedTest, AllNullStringsColumnWithColumnRepeatTimes)
{
  using ints_col = cudf::test::fixed_width_column_wrapper<TypeParam>;

  auto const strs    = strs_col{{"" /*NULL*/, "" /*NULL*/, "" /*NULL*/}, all_nulls()};
  auto const strs_cv = cudf::strings_column_view(strs);

  // The repeat_times column contains all valid numbers.
  {
    auto const repeat_times = ints_col{-1, 0, 1};
    auto const results      = cudf::strings::repeat_strings(strs_cv, repeat_times);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(strs, *results, verbosity);
  }

  // The repeat_times column also contains some nulls and some valid numbers.
  {
    auto const repeat_times = ints_col{{null, 1, null}, nulls_at({0, 2})};
    auto const results      = cudf::strings::repeat_strings(strs_cv, repeat_times);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(strs, *results, verbosity);
  }

  // The repeat_times column also contains all nulls.
  {
    auto const repeat_times = ints_col{{null, null, null}, all_nulls()};
    auto const results      = cudf::strings::repeat_strings(strs_cv, repeat_times);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(strs, *results, verbosity);
  }
}

TYPED_TEST(RepeatStringsTypedTest, StringsColumnWithAllNullColumnRepeatTimes)
{
  using ints_col = cudf::test::fixed_width_column_wrapper<TypeParam>;

  auto const strs         = strs_col{"ABC", "abc", "xyz"};
  auto const repeat_times = ints_col{{null, null, null}, all_nulls()};
  auto const results = cudf::strings::repeat_strings(cudf::strings_column_view(strs), repeat_times);
  auto const expected_strs = strs_col{{"" /*NULL*/, "" /*NULL*/, "" /*NULL*/}, all_nulls()};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_strs, *results, verbosity);
}

TYPED_TEST(RepeatStringsTypedTest, ZeroSizeAndNullStringsColumnWithScalarRepeatTimes)
{
  auto const strs =
    strs_col{{"" /*NULL*/, "", "" /*NULL*/, "", "", "" /*NULL*/}, nulls_at({0, 2, 5})};
  auto const results = cudf::strings::repeat_strings(cudf::strings_column_view(strs), 10);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(strs, *results, verbosity);
}

TYPED_TEST(RepeatStringsTypedTest, ZeroSizeAndNullStringsColumnWithColumnRepeatTimes)
{
  using ints_col = cudf::test::fixed_width_column_wrapper<TypeParam>;

  auto const strs =
    strs_col{{"" /*NULL*/, "", "" /*NULL*/, "", "", "" /*NULL*/}, nulls_at({0, 2, 5})};
  auto const repeat_times = ints_col{1, 2, 3, 4, 5, 6};
  auto const results      = cudf::strings::repeat_strings(cudf::strings_column_view(strs), 10);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(strs, *results, verbosity);
}

TEST_F(RepeatStringsTest, StringsColumnWithColumnRepeatTimesInvalidInput)
{
  auto const strs    = strs_col{"abc", "xyz"};
  auto const strs_cv = cudf::strings_column_view(strs);

  // Sizes mismatched between strings column and repeat_times column.
  {
    auto const repeat_times = int32s_col{1, 2, 3, 4, 5, 6};
    EXPECT_THROW(cudf::strings::repeat_strings(strs_cv, repeat_times), cudf::logic_error);
  }

  // Invalid data type for repeat_times column.
  {
    auto const repeat_times = cudf::test::fixed_width_column_wrapper<float>{1, 2, 3, 4, 5, 6};
    EXPECT_THROW(cudf::strings::repeat_strings(strs_cv, repeat_times), cudf::logic_error);
  }

  // Invalid data type for repeat_times column.
  {
    auto const repeat_times = strs_col{"xxx", "xxx"};
    EXPECT_THROW(cudf::strings::repeat_strings(strs_cv, repeat_times), cudf::logic_error);
  }
}

TEST_F(RepeatStringsTest, StringsColumnWithColumnRepeatTimesOverflowOutput)
{
  if (cudf::strings::is_large_strings_enabled()) { return; }

  auto const strs    = strs_col{"1", "12", "123", "1234", "12345", "123456", "1234567"};
  auto const strs_cv = cudf::strings_column_view(strs);

  auto const half_max = std::numeric_limits<int32_t>::max() / 2;
  auto const repeat_times =
    int32s_col{half_max, half_max, half_max, half_max, half_max, half_max, half_max};

  EXPECT_THROW(cudf::strings::repeat_strings(strs_cv, repeat_times), std::overflow_error);
}

TYPED_TEST(RepeatStringsTypedTest, StringsColumnNoNullWithScalarRepeatTimes)
{
  auto const strs    = strs_col{"0a0b0c", "abcxyz", "xyzééé", "ááá", "íí"};
  auto const strs_cv = cudf::strings_column_view(strs);

  {
    auto const results = cudf::strings::repeat_strings(strs_cv, 2);
    auto const expected_strs =
      strs_col{"0a0b0c0a0b0c", "abcxyzabcxyz", "xyzéééxyzééé", "áááááá", "íííí"};
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_strs, *results, verbosity);
  }

  // Repeat once.
  {
    auto const results = cudf::strings::repeat_strings(strs_cv, 1);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(strs, *results, verbosity);
  }

  // Non-positive repeat times.
  {
    auto const expected_strs = strs_col{"", "", "", "", ""};

    auto results = cudf::strings::repeat_strings(strs_cv, 0);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_strs, *results, verbosity);

    results = cudf::strings::repeat_strings(strs_cv, -100);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_strs, *results, verbosity);
  }
}

TYPED_TEST(RepeatStringsTypedTest, StringsColumnNoNullWithColumnRepeatTimes)
{
  using ints_col = cudf::test::fixed_width_column_wrapper<TypeParam>;

  auto const strs    = strs_col{"0a0b0c", "abcxyz", "xyzééé", "ááá", "íí"};
  auto const strs_cv = cudf::strings_column_view(strs);

  // Repeat once.
  {
    auto const repeat_times = ints_col{1, 1, 1, 1, 1};
    auto const results      = cudf::strings::repeat_strings(strs_cv, repeat_times);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(strs, *results, verbosity);
  }

  // repeat_times column has negative values.
  {
    auto const repeat_times  = ints_col{1, 2, 3, -1, -2};
    auto const expected_strs = strs_col{"0a0b0c", "abcxyzabcxyz", "xyzéééxyzéééxyzééé", "", ""};

    auto results = cudf::strings::repeat_strings(strs_cv, repeat_times);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_strs, *results, verbosity);
  }

  // repeat_times column has nulls.
  {
    auto const repeat_times  = ints_col{{1, null, 3, 2, null}, nulls_at({1, 4})};
    auto const expected_strs = strs_col{
      {"0a0b0c", "" /*NULL*/, "xyzéééxyzéééxyzééé", "áááááá", "" /*NULL*/}, nulls_at({1, 4})};

    auto results = cudf::strings::repeat_strings(strs_cv, repeat_times);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_strs, *results, verbosity);
  }
}

TYPED_TEST(RepeatStringsTypedTest, SlicedStringsColumnNoNullWithScalarRepeatTimes)
{
  auto const strs = strs_col{"0a0b0c", "abcxyz", "xyzééé", "ááá", "íí"};

  // Sliced the first half of the column.
  {
    auto const sliced_strs = cudf::slice(strs, {0, 3})[0];
    auto const results = cudf::strings::repeat_strings(cudf::strings_column_view(sliced_strs), 2);
    auto const expected_strs = strs_col{"0a0b0c0a0b0c", "abcxyzabcxyz", "xyzéééxyzééé"};
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_strs, *results, verbosity);
  }

  // Sliced the middle of the column.
  {
    auto const sliced_strs = cudf::slice(strs, {1, 3})[0];
    auto const results = cudf::strings::repeat_strings(cudf::strings_column_view(sliced_strs), 2);
    auto const expected_strs = strs_col{"abcxyzabcxyz", "xyzéééxyzééé"};
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_strs, *results, verbosity);
  }

  // Sliced the second half of the column.
  {
    auto const sliced_strs = cudf::slice(strs, {2, 5})[0];
    auto const results = cudf::strings::repeat_strings(cudf::strings_column_view(sliced_strs), 2);
    auto const expected_strs = strs_col{"xyzéééxyzééé", "áááááá", "íííí"};
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_strs, *results, verbosity);
  }
}

TYPED_TEST(RepeatStringsTypedTest, SlicedStringsColumnNoNullWithColumnRepeatTimes)
{
  using ints_col = cudf::test::fixed_width_column_wrapper<TypeParam>;

  auto const strs         = strs_col{"0a0b0c", "abcxyz", "xyzééé", "ááá", "íí"};
  auto const repeat_times = ints_col{1, 2, 3, 2, 3, 4, 5, 6, 7, 8, 9, 10};

  // Sliced the first half of the column.
  {
    auto const sliced_strs    = cudf::slice(strs, {0, 3})[0];
    auto const sliced_rtimes  = cudf::slice(repeat_times, {0, 3})[0];
    auto const sliced_strs_cv = cudf::strings_column_view(sliced_strs);
    auto const expected_strs  = strs_col{"0a0b0c", "abcxyzabcxyz", "xyzéééxyzéééxyzééé"};

    auto results = cudf::strings::repeat_strings(sliced_strs_cv, sliced_rtimes);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_strs, *results, verbosity);
  }

  // Sliced the middle of the column.
  {
    auto const sliced_strs    = cudf::slice(strs, {1, 3})[0];
    auto const sliced_rtimes  = cudf::slice(repeat_times, {1, 3})[0];
    auto const sliced_strs_cv = cudf::strings_column_view(sliced_strs);
    auto const expected_strs  = strs_col{"abcxyzabcxyz", "xyzéééxyzéééxyzééé"};

    auto results = cudf::strings::repeat_strings(sliced_strs_cv, sliced_rtimes);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_strs, *results, verbosity);
  }

  // Sliced the second half of the column.
  {
    auto const sliced_strs    = cudf::slice(strs, {2, 5})[0];
    auto const sliced_rtimes  = cudf::slice(repeat_times, {2, 5})[0];
    auto const sliced_strs_cv = cudf::strings_column_view(sliced_strs);
    auto const expected_strs  = strs_col{"xyzéééxyzéééxyzééé", "áááááá", "íííííí"};

    auto results = cudf::strings::repeat_strings(sliced_strs_cv, sliced_rtimes);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_strs, *results, verbosity);
  }
}

TYPED_TEST(RepeatStringsTypedTest, StringsColumnWithNullsWithScalarRepeatTimes)
{
  auto const strs    = strs_col{{"0a0b0c",
                                 "" /*NULL*/,
                                 "abcxyz",
                                 "" /*NULL*/,
                                 "xyzééé",
                                 "" /*NULL*/,
                                 "ááá",
                                 "íí",
                                 "",
                                 "Hello World"},
                                nulls_at({1, 3, 5})};
  auto const strs_cv = cudf::strings_column_view(strs);

  {
    auto const results       = cudf::strings::repeat_strings(strs_cv, 2);
    auto const expected_strs = strs_col{{"0a0b0c0a0b0c",
                                         "" /*NULL*/,
                                         "abcxyzabcxyz",
                                         "" /*NULL*/,
                                         "xyzéééxyzééé",
                                         "" /*NULL*/,
                                         "áááááá",
                                         "íííí",
                                         "",
                                         "Hello WorldHello World"},
                                        nulls_at({1, 3, 5})};
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_strs, *results, verbosity);
  }

  // Repeat once.
  {
    auto const results = cudf::strings::repeat_strings(strs_cv, 1);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(strs, *results, verbosity);
  }

  // Non-positive repeat times.
  {
    auto const expected_strs = strs_col{
      {"", "" /*NULL*/, "", "" /*NULL*/, "", "" /*NULL*/, "", "", "", ""}, nulls_at({1, 3, 5})};

    auto results = cudf::strings::repeat_strings(strs_cv, 0);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_strs, *results, verbosity);

    results = cudf::strings::repeat_strings(strs_cv, -100);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_strs, *results, verbosity);
  }
}

TYPED_TEST(RepeatStringsTypedTest, StringsColumnWithNullsWithColumnRepeatTimes)
{
  using ints_col = cudf::test::fixed_width_column_wrapper<TypeParam>;

  auto const strs    = strs_col{{"0a0b0c",
                                 "" /*NULL*/,
                                 "abcxyz",
                                 "" /*NULL*/,
                                 "xyzééé",
                                 "" /*NULL*/,
                                 "ááá",
                                 "íí",
                                 "",
                                 "Hello World"},
                                nulls_at({1, 3, 5})};
  auto const strs_cv = cudf::strings_column_view(strs);

  // Repeat once.
  {
    auto const repeat_times = ints_col{1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    auto const results      = cudf::strings::repeat_strings(strs_cv, repeat_times);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(strs, *results, verbosity);
  }

  // repeat_times column has negative values.
  {
    auto const repeat_times  = ints_col{1, 2, 3, -1, -2, 1, 2, 3, -5, 0};
    auto const expected_strs = strs_col{{"0a0b0c",
                                         "" /*NULL*/,
                                         "abcxyzabcxyzabcxyz",
                                         "" /*NULL*/,
                                         "",
                                         "" /*NULL*/,
                                         "áááááá",
                                         "íííííí",
                                         "",
                                         ""},
                                        nulls_at({1, 3, 5})};

    auto results = cudf::strings::repeat_strings(strs_cv, repeat_times);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_strs, *results, verbosity);
  }

  // repeat_times column has nulls.
  {
    auto const repeat_times =
      ints_col{{1, 2, null, -1, null, 1, 2, null, -5, 0}, nulls_at({2, 4, 7})};
    auto const expected_strs = strs_col{{"0a0b0c",
                                         "" /*NULL*/,
                                         "" /*NULL*/,
                                         "" /*NULL*/,
                                         "" /*NULL*/,
                                         "" /*NULL*/,
                                         "áááááá",
                                         "" /*NULL*/,
                                         "",
                                         ""},
                                        nulls_at({1, 2, 3, 4, 5, 7})};

    auto results = cudf::strings::repeat_strings(strs_cv, repeat_times);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_strs, *results, verbosity);
  }
}

TYPED_TEST(RepeatStringsTypedTest, SlicedStringsColumnWithNullsWithScalarRepeatTimes)
{
  auto const strs = strs_col{{"0a0b0c",
                              "" /*NULL*/,
                              "abcxyz",
                              "" /*NULL*/,
                              "xyzééé",
                              "" /*NULL*/,
                              "ááá",
                              "íí",
                              "",
                              "Hello World"},
                             nulls_at({1, 3, 5})};

  // Sliced the first half of the column.
  {
    auto const sliced_strs = cudf::slice(strs, {0, 3})[0];
    auto const results = cudf::strings::repeat_strings(cudf::strings_column_view(sliced_strs), 2);
    auto const expected_strs = strs_col{{"0a0b0c0a0b0c", "" /*NULL*/, "abcxyzabcxyz"}, null_at(1)};
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_strs, *results, verbosity);
  }

  // Sliced the middle of the column.
  {
    auto const sliced_strs = cudf::slice(strs, {2, 7})[0];
    auto const results = cudf::strings::repeat_strings(cudf::strings_column_view(sliced_strs), 2);
    auto const expected_strs = strs_col{
      {"abcxyzabcxyz", "" /*NULL*/, "xyzéééxyzééé", "" /*NULL*/, "áááááá"}, nulls_at({1, 3})};
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_strs, *results, verbosity);
  }

  // Sliced the second half of the column.
  {
    auto const sliced_strs = cudf::slice(strs, {6, 10})[0];
    auto const results = cudf::strings::repeat_strings(cudf::strings_column_view(sliced_strs), 2);
    auto const expected_strs = strs_col{"áááááá", "íííí", "", "Hello WorldHello World"};

    // The results strings column may have a bitmask with all valid values.
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected_strs, *results, verbosity);
  }
}

TYPED_TEST(RepeatStringsTypedTest, SlicedStringsColumnWithNullsWithColumnRepeatTimes)
{
  using ints_col = cudf::test::fixed_width_column_wrapper<TypeParam>;

  auto const strs = strs_col{{"0a0b0c",
                              "" /*NULL*/,
                              "abcxyz",
                              "" /*NULL*/,
                              "xyzééé",
                              "" /*NULL*/,
                              "ááá",
                              "íí",
                              "",
                              "Hello World"},
                             nulls_at({1, 3, 5})};

  auto const repeat_times =
    ints_col{{1, 2, null, -1, null, 1, 2, null, -5, 0, 6, 7, 8, 9, 10}, nulls_at({2, 4, 7})};

  // Sliced the first half of the column.
  {
    auto const sliced_strs    = cudf::slice(strs, {0, 3})[0];
    auto const sliced_rtimes  = cudf::slice(repeat_times, {0, 3})[0];
    auto const sliced_strs_cv = cudf::strings_column_view(sliced_strs);
    auto const expected_strs  = strs_col{{"0a0b0c", "" /*NULL*/, "" /*NULL*/}, nulls_at({1, 2})};

    auto results = cudf::strings::repeat_strings(sliced_strs_cv, sliced_rtimes);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_strs, *results, verbosity);
  }

  // Sliced the middle of the column.
  {
    auto const sliced_strs    = cudf::slice(strs, {2, 7})[0];
    auto const sliced_rtimes  = cudf::slice(repeat_times, {2, 7})[0];
    auto const sliced_strs_cv = cudf::strings_column_view(sliced_strs);
    auto const expected_strs  = strs_col{
      {"" /*NULL*/, "" /*NULL*/, "" /*NULL*/, "" /*NULL*/, "áááááá"}, nulls_at({0, 1, 2, 3})};

    auto results = cudf::strings::repeat_strings(sliced_strs_cv, sliced_rtimes);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_strs, *results, verbosity);
  }

  // Sliced the second half of the column, output has nulls.
  {
    auto const sliced_strs    = cudf::slice(strs, {6, 10})[0];
    auto const sliced_rtimes  = cudf::slice(repeat_times, {6, 10})[0];
    auto const sliced_strs_cv = cudf::strings_column_view(sliced_strs);
    auto const expected_strs  = strs_col{{"áááááá", "" /*NULL*/, "", ""}, null_at(1)};

    auto results = cudf::strings::repeat_strings(sliced_strs_cv, sliced_rtimes);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_strs, *results, verbosity);
  }

  // Sliced the second half of the column, output does not have null.
  // Since the input has nulls, the output column is nullable (but doesn't have nulls).
  {
    auto const sliced_strs    = cudf::slice(strs, {8, 10})[0];
    auto const sliced_rtimes  = cudf::slice(repeat_times, {8, 10})[0];
    auto const sliced_strs_cv = cudf::strings_column_view(sliced_strs);
    auto const expected_strs  = strs_col{"", ""};

    auto results = cudf::strings::repeat_strings(sliced_strs_cv, sliced_rtimes);
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected_strs, *results, verbosity);
  }
}
