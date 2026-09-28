/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/iterator_utilities.hpp>
#include <cudf_test/table_utilities.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/table/table_view.hpp>
#include <cudf/transform.hpp>
#include <cudf/utilities/error.hpp>

#include <limits>

using lists_col   = cudf::test::lists_column_wrapper<int32_t>;
using structs_col = cudf::test::structs_column_wrapper;
using bool_col    = cudf::test::fixed_width_column_wrapper<bool>;

using cudf::test::iterators::null_at;
using cudf::test::iterators::nulls_at;

template <typename T>
struct OneHotEncodingTestTyped : public cudf::test::BaseFixture {};

struct OneHotEncodingTest : public cudf::test::BaseFixture {};

TYPED_TEST_SUITE(OneHotEncodingTestTyped, cudf::test::NumericTypes);

TYPED_TEST(OneHotEncodingTestTyped, Basic)
{
  auto input    = cudf::test::fixed_width_column_wrapper<int32_t>{8, 8, 8, 9, 9};
  auto category = cudf::test::fixed_width_column_wrapper<int32_t>{8, 9};

  auto col0 = cudf::test::fixed_width_column_wrapper<bool>{1, 1, 1, 0, 0};
  auto col1 = cudf::test::fixed_width_column_wrapper<bool>{0, 0, 0, 1, 1};

  auto expected = cudf::table_view{{col0, col1}};

  [[maybe_unused]] auto [res_ptr, got] = cudf::one_hot_encode(input, category);

  CUDF_TEST_EXPECT_TABLES_EQUAL(expected, got);
}

TYPED_TEST(OneHotEncodingTestTyped, Nulls)
{
  auto input    = cudf::test::fixed_width_column_wrapper<int32_t>{{8, 8, 8, 9, 9},
                                                                  {true, true, false, true, true}};
  auto category = cudf::test::fixed_width_column_wrapper<int32_t>({8, 9, -1}, {true, true, false});

  auto col0 = cudf::test::fixed_width_column_wrapper<bool>{1, 1, 0, 0, 0};
  auto col1 = cudf::test::fixed_width_column_wrapper<bool>{0, 0, 0, 1, 1};
  auto col2 = cudf::test::fixed_width_column_wrapper<bool>{0, 0, 1, 0, 0};

  auto expected = cudf::table_view{{col0, col1, col2}};

  [[maybe_unused]] auto [res_ptr, got] = cudf::one_hot_encode(input, category);

  CUDF_TEST_EXPECT_TABLES_EQUAL(expected, got);
}

TEST_F(OneHotEncodingTest, Diagonal)
{
  auto input    = cudf::test::fixed_width_column_wrapper<int32_t>{1, 2, 3, 4, 5};
  auto category = cudf::test::fixed_width_column_wrapper<int32_t>{1, 2, 3, 4, 5};

  auto col0 = cudf::test::fixed_width_column_wrapper<bool>{1, 0, 0, 0, 0};
  auto col1 = cudf::test::fixed_width_column_wrapper<bool>{0, 1, 0, 0, 0};
  auto col2 = cudf::test::fixed_width_column_wrapper<bool>{0, 0, 1, 0, 0};
  auto col3 = cudf::test::fixed_width_column_wrapper<bool>{0, 0, 0, 1, 0};
  auto col4 = cudf::test::fixed_width_column_wrapper<bool>{0, 0, 0, 0, 1};

  auto expected = cudf::table_view{{col0, col1, col2, col3, col4}};

  [[maybe_unused]] auto [res_ptr, got] = cudf::one_hot_encode(input, category);

  CUDF_TEST_EXPECT_TABLES_EQUAL(expected, got);
}

TEST_F(OneHotEncodingTest, ZeroInput)
{
  auto input    = cudf::test::strings_column_wrapper{};
  auto category = cudf::test::strings_column_wrapper{"rapids", "cudf"};

  auto col0 = cudf::test::fixed_width_column_wrapper<bool>{};
  auto col1 = cudf::test::fixed_width_column_wrapper<bool>{};

  auto expected = cudf::table_view{{col0, col1}};

  [[maybe_unused]] auto [res_ptr, got] = cudf::one_hot_encode(input, category);

  CUDF_TEST_EXPECT_TABLES_EQUAL(expected, got);
}

TEST_F(OneHotEncodingTest, ZeroCat)
{
  auto input    = cudf::test::strings_column_wrapper{"ji", "ji", "ji"};
  auto category = cudf::test::strings_column_wrapper{};

  auto expected = cudf::table_view{};

  [[maybe_unused]] auto [res_ptr, got] = cudf::one_hot_encode(input, category);

  CUDF_TEST_EXPECT_TABLES_EQUAL(expected, got);
}

TEST_F(OneHotEncodingTest, ZeroInputCat)
{
  auto input    = cudf::test::strings_column_wrapper{};
  auto category = cudf::test::strings_column_wrapper{};

  auto expected = cudf::table_view{};

  [[maybe_unused]] auto [res_ptr, got] = cudf::one_hot_encode(input, category);

  CUDF_TEST_EXPECT_TABLES_EQUAL(expected, got);
}

TEST_F(OneHotEncodingTest, OneCat)
{
  auto input    = cudf::test::strings_column_wrapper{"ji", "ji", "ji"};
  auto category = cudf::test::strings_column_wrapper{"ji"};

  auto col0 = cudf::test::fixed_width_column_wrapper<bool>{1, 1, 1};

  auto expected = cudf::table_view{{col0}};

  [[maybe_unused]] auto [res_ptr, got] = cudf::one_hot_encode(input, category);

  CUDF_TEST_EXPECT_TABLES_EQUAL(expected, got);
}

TEST_F(OneHotEncodingTest, NaNs)
{
  auto const nan = std::numeric_limits<float>::signaling_NaN();

  auto input    = cudf::test::fixed_width_column_wrapper<float>{8.f, 8.f, 8.f, 9.f, nan};
  auto category = cudf::test::fixed_width_column_wrapper<float>{8.f, 9.f, nan};

  auto col0 = cudf::test::fixed_width_column_wrapper<bool>{1, 1, 1, 0, 0};
  auto col1 = cudf::test::fixed_width_column_wrapper<bool>{0, 0, 0, 1, 0};
  auto col2 = cudf::test::fixed_width_column_wrapper<bool>{0, 0, 0, 0, 1};

  auto expected = cudf::table_view{{col0, col1, col2}};

  [[maybe_unused]] auto [res_ptr, got] = cudf::one_hot_encode(input, category);

  CUDF_TEST_EXPECT_TABLES_EQUAL(expected, got);
}

TEST_F(OneHotEncodingTest, Strings)
{
  auto input = cudf::test::strings_column_wrapper{
    {"hello", "rapidsai", "cudf", "hello", "cuspatial", "hello", "world", "!"},
    {true, true, true, true, false, true, true, false}};
  auto category = cudf::test::strings_column_wrapper{{"hello", "world", ""}, {true, true, false}};

  auto col0 = cudf::test::fixed_width_column_wrapper<bool>{1, 0, 0, 1, 0, 1, 0, 0};
  auto col1 = cudf::test::fixed_width_column_wrapper<bool>{0, 0, 0, 0, 0, 0, 1, 0};
  auto col2 = cudf::test::fixed_width_column_wrapper<bool>{0, 0, 0, 0, 1, 0, 0, 1};

  auto expected = cudf::table_view{{col0, col1, col2}};

  [[maybe_unused]] auto [res_ptr, got] = cudf::one_hot_encode(input, category);

  CUDF_TEST_EXPECT_TABLES_EQUAL(expected, got);
}

TEST_F(OneHotEncodingTest, Dictionary)
{
  auto input =
    cudf::test::dictionary_column_wrapper<std::string>{"aa", "xx", "aa", "aa", "yy", "ef"};
  auto category = cudf::test::dictionary_column_wrapper<std::string>{"aa", "ef"};

  auto col0 = cudf::test::fixed_width_column_wrapper<bool>{1, 0, 1, 1, 0, 0};
  auto col1 = cudf::test::fixed_width_column_wrapper<bool>{0, 0, 0, 0, 0, 1};

  auto expected = cudf::table_view{{col0, col1}};

  [[maybe_unused]] auto [res_ptr, got] = cudf::one_hot_encode(input, category);

  CUDF_TEST_EXPECT_TABLES_EQUAL(expected, got);
}

TEST_F(OneHotEncodingTest, MismatchTypes)
{
  auto input    = cudf::test::strings_column_wrapper{"xx", "yy", "xx"};
  auto category = cudf::test::fixed_width_column_wrapper<int64_t>{1};

  EXPECT_THROW(cudf::one_hot_encode(input, category), cudf::data_type_error);
}

TEST_F(OneHotEncodingTest, List)
{
  auto const input =
    lists_col{{{}, {1}, {2, 2}, {2, 2}, {}, {2} /*NULL*/, {2}, {5} /*NULL*/}, nulls_at({5, 7})};
  auto const categories = lists_col{{{}, {1}, {2, 2}, {2}, {-1}}, null_at(4)};

  auto const col0 = bool_col{1, 0, 0, 0, 1, 0, 0, 0};
  auto const col1 = bool_col{0, 1, 0, 0, 0, 0, 0, 0};
  auto const col2 = bool_col{0, 0, 1, 1, 0, 0, 0, 0};
  auto const col3 = bool_col{0, 0, 0, 0, 0, 0, 1, 0};
  auto const col4 = bool_col{0, 0, 0, 0, 0, 1, 0, 1};

  auto const expected = cudf::table_view{{col0, col1, col2, col3, col4}};

  [[maybe_unused]] auto const [res_ptr, got] = cudf::one_hot_encode(input, categories);

  CUDF_TEST_EXPECT_TABLES_EQUAL(expected, got);
}

TEST_F(OneHotEncodingTest, StructsOfStructs)
{
  //  +-----------------+
  //  |  s1{s2{a,b}, c} |
  //  +-----------------+
  // 0 |  Null          |
  // 1 |  { {1, 2}, 4}  |
  // 2 |  { Null,   4}  |
  // 3 |  Null          |
  // 4 |  { {2, 1}, 5}  |
  // 5 |  { Null,   4}  |
  // 6 |  { {2, 1}, 5}  |

  auto const input = [&] {
    auto a  = cudf::test::fixed_width_column_wrapper<int32_t>{-1, 1, -1, -1, 2, -1, 2};
    auto b  = cudf::test::fixed_width_column_wrapper<int32_t>{-1, 2, -1, -1, 1, -1, 1};
    auto s2 = structs_col{{a, b}, nulls_at({2, 5})};

    auto c = cudf::test::fixed_width_column_wrapper<int32_t>{-1, 4, 4, -1, 5, 4, 5};
    std::vector<std::unique_ptr<cudf::column>> s1_children;
    s1_children.emplace_back(s2.release());
    s1_children.emplace_back(c.release());
    auto const null_it = nulls_at({0, 3});
    return structs_col(std::move(s1_children), std::vector<bool>{null_it, null_it + 7});
  }();

  auto const categories = [&] {
    auto a  = cudf::test::fixed_width_column_wrapper<int32_t>{-1, 1, -1, 2};
    auto b  = cudf::test::fixed_width_column_wrapper<int32_t>{-1, 2, -1, 1};
    auto s2 = structs_col{{a, b}, null_at(2)};

    auto c = cudf::test::fixed_width_column_wrapper<int32_t>{-1, 4, 4, 5};
    std::vector<std::unique_ptr<cudf::column>> s1_children;
    s1_children.emplace_back(s2.release());
    s1_children.emplace_back(c.release());
    auto const null_it = null_at(0);
    return structs_col(std::move(s1_children), std::vector<bool>{null_it, null_it + 4});
  }();

  auto const col0 = bool_col{1, 0, 0, 1, 0, 0, 0};
  auto const col1 = bool_col{0, 1, 0, 0, 0, 0, 0};
  auto const col2 = bool_col{0, 0, 1, 0, 0, 1, 0};
  auto const col3 = bool_col{0, 0, 0, 0, 1, 0, 1};

  auto const expected = cudf::table_view{{col0, col1, col2, col3}};

  [[maybe_unused]] auto const [res_ptr, got] = cudf::one_hot_encode(input, categories);

  CUDF_TEST_EXPECT_TABLES_EQUAL(expected, got);
}
