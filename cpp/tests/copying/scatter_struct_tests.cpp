/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/iterator_utilities.hpp>
#include <cudf_test/lists_column_wrapper.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/copying.hpp>
#include <cudf/table/table_view.hpp>

using namespace cudf::test::iterators;

using bools_col   = cudf::test::fixed_width_column_wrapper<bool>;
using int32s_col  = cudf::test::fixed_width_column_wrapper<int32_t>;
using structs_col = cudf::test::structs_column_wrapper;
using strings_col = cudf::test::strings_column_wrapper;

constexpr cudf::test::debug_output_level verbosity{cudf::test::debug_output_level::FIRST_ERROR};
constexpr int32_t null{0};  // Mark for null child elements
constexpr int32_t XXX{0};   // Mark for null struct elements

template <typename T>
struct TypedStructScatterTest : public cudf::test::BaseFixture {};

using TestTypes = cudf::test::Concat<cudf::test::IntegralTypes,
                                     cudf::test::FloatingPointTypes,
                                     cudf::test::DurationTypes,
                                     cudf::test::TimestampTypes>;

TYPED_TEST_SUITE(TypedStructScatterTest, TestTypes);

namespace {
auto scatter_structs(std::unique_ptr<cudf::column> const& structs_src,
                     std::unique_ptr<cudf::column> const& structs_tgt,
                     std::unique_ptr<cudf::column> const& scatter_map)
{
  auto const source = cudf::table_view{std::vector<cudf::column_view>{structs_src->view()}};
  auto const target = cudf::table_view{std::vector<cudf::column_view>{structs_tgt->view()}};
  auto const result = cudf::scatter(source, scatter_map->view(), target);
  return result->get_column(0);
}
}  // namespace

// Test case when all input columns are empty
TYPED_TEST(TypedStructScatterTest, EmptyInputTest)
{
  using col_wrapper = cudf::test::fixed_width_column_wrapper<TypeParam, int32_t>;

  auto child_col_src     = col_wrapper{};
  auto const structs_src = structs_col{{child_col_src}}.release();

  auto child_col_tgt     = col_wrapper{};
  auto const structs_tgt = structs_col{{child_col_tgt}}.release();

  auto const scatter_map = int32s_col{}.release();
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    *structs_src, scatter_structs(structs_src, structs_tgt, scatter_map), verbosity);
}

// Test case when only the scatter map is empty
TYPED_TEST(TypedStructScatterTest, EmptyScatterMapTest)
{
  using col_wrapper = cudf::test::fixed_width_column_wrapper<TypeParam, int32_t>;

  auto child_col_src     = col_wrapper{{0, 1, 2, 3, null, XXX}, null_at(4)};
  auto const structs_src = structs_col{{child_col_src}, null_at(5)}.release();

  auto child_col_tgt     = col_wrapper{{50, null, 70, XXX, 90, 100}, null_at(1)};
  auto const structs_tgt = structs_col{{child_col_tgt}, null_at(3)}.release();

  auto const scatter_map = int32s_col{}.release();
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    *structs_tgt, scatter_structs(structs_src, structs_tgt, scatter_map), verbosity);
}

TYPED_TEST(TypedStructScatterTest, ScatterAsCopyTest)
{
  using col_wrapper = cudf::test::fixed_width_column_wrapper<TypeParam, int32_t>;

  auto child_col_src     = col_wrapper{{0, 1, 2, 3, null, XXX}, null_at(4)};
  auto const structs_src = structs_col{{child_col_src}, null_at(5)}.release();

  auto child_col_tgt     = col_wrapper{{50, null, 70, XXX, 90, 100}, null_at(1)};
  auto const structs_tgt = structs_col{{child_col_tgt}, null_at(3)}.release();

  // Scatter as copy: the target should be the same as source
  auto const scatter_map = int32s_col{0, 1, 2, 3, 4, 5}.release();
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    *structs_src, scatter_structs(structs_src, structs_tgt, scatter_map), verbosity);
}

TYPED_TEST(TypedStructScatterTest, ScatterAsLeftShiftTest)
{
  using col_wrapper = cudf::test::fixed_width_column_wrapper<TypeParam, int32_t>;

  auto child_col_src     = col_wrapper{{0, 1, 2, 3, null, XXX}, null_at(4)};
  auto const structs_src = structs_col{{child_col_src}, null_at(5)}.release();

  auto child_col_tgt     = col_wrapper{{50, null, 70, XXX, 90, 100}, null_at(1)};
  auto const structs_tgt = structs_col{{child_col_tgt}, null_at(3)}.release();

  auto child_col_expected = col_wrapper{{2, 3, null, XXX, 0, 1}, null_at(2)};
  auto structs_expected   = structs_col{{child_col_expected}, null_at(3)}.release();

  auto const scatter_map = int32s_col{-2, -1, 0, 1, 2, 3}.release();
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    *structs_expected, scatter_structs(structs_src, structs_tgt, scatter_map), verbosity);
}

TYPED_TEST(TypedStructScatterTest, SimpleScatterTests)
{
  using col_wrapper = cudf::test::fixed_width_column_wrapper<TypeParam, int32_t>;

  // Source data
  auto child_col_src     = col_wrapper{{0, 1, 2, 3, null, XXX}, null_at(4)};
  auto const structs_src = structs_col{{child_col_src}, null_at(5)}.release();

  // Target data
  auto child_col_tgt     = col_wrapper{{50, null, 70, XXX, 90, 100}, null_at(1)};
  auto const structs_tgt = structs_col{{child_col_tgt}, null_at(3)}.release();

  // Expected data
  auto child_col_expected1     = col_wrapper{{1, null, 70, XXX, 0, 2}, null_at(1)};
  auto const structs_expected1 = structs_col{{child_col_expected1}, null_at(3)}.release();
  auto const scatter_map1      = int32s_col{-2, 0, 5}.release();
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    *structs_expected1, scatter_structs(structs_src, structs_tgt, scatter_map1), verbosity);

  // Expected data
  auto child_col_expected2     = col_wrapper{{1, null, 70, 3, 0, 2}, null_at(1)};
  auto const structs_expected2 = structs_col{{child_col_expected2}, no_nulls()}.release();
  auto const scatter_map2      = int32s_col{-2, 0, 5, 3}.release();
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    *structs_expected2, scatter_structs(structs_src, structs_tgt, scatter_map2), verbosity);
}

TYPED_TEST(TypedStructScatterTest, ComplexDataScatterTest)
{
  // Testing scatter() on struct<string, numeric, bool>.
  using col_wrapper = cudf::test::fixed_width_column_wrapper<TypeParam, int32_t>;

  // Source data
  auto names_column_src =
    strings_col{{"Newton", "Washington", "Cherry", "Kiwi", "Lemon", "Tomato" /*XXX*/}, no_nulls()};
  auto ages_column_src = col_wrapper{{5, 10, 15, 20, null, XXX}, null_at(4)};
  auto is_human_col_src =
    bools_col{{true, true, false, false /*null*/, false, false /*XXX*/}, null_at(3)};
  auto const structs_src =
    structs_col{{names_column_src, ages_column_src, is_human_col_src}, null_at(5)}.release();

  // Target data
  auto names_column_tgt = strings_col{
    {"String 0" /*null*/, "String 1", "String 2" /*XXX*/, "String 3", "String 4", "String 5"},
    null_at(0)};
  auto ages_column_tgt  = col_wrapper{{50, null, XXX, 80, 90, 100}, null_at(1)};
  auto is_human_col_tgt = bools_col{{true, true, true /*XXX*/, true, true, true}, no_nulls()};
  auto const structs_tgt =
    structs_col{{names_column_tgt, ages_column_tgt, is_human_col_tgt}, null_at(2)}.release();

  // Expected data
  auto names_column_expected = strings_col{
    {"String 0" /*null*/, "Lemon", "Kiwi", "Cherry", "Washington", "Newton"}, null_at(0)};
  auto ages_column_expected = col_wrapper{{50, null, 20, 15, 10, 5}, null_at(1)};
  auto is_human_col_expected =
    bools_col{{true, false, false /*null*/, false, true, true}, null_at(2)};
  auto const structs_expected =
    structs_col{{names_column_expected, ages_column_expected, is_human_col_expected}, no_nulls()}
      .release();

  // The first element of the target is not overwritten
  auto const scatter_map = int32s_col{-1, 4, 3, 2, 1}.release();
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    *structs_expected, scatter_structs(structs_src, structs_tgt, scatter_map), verbosity);
}

TYPED_TEST(TypedStructScatterTest, ScatterStructOfListsTest)
{
  // Testing scatter() on struct<list<numeric>>
  using lists_col = cudf::test::lists_column_wrapper<TypeParam, int32_t>;

  // Source data
  auto lists_col_src =
    lists_col{{{5}, {10, 15}, {20, 25, 30}, {35, 40, 45, 50}, {55, 60, 65}, {70, 75}, {80}, {}, {}},
              // Valid for elements 0, 3, 6,...
              valids_at_multiples_of(3)};
  auto const structs_src = structs_col{{lists_col_src}}.release();

  // Target data
  auto lists_col_tgt =
    lists_col{{{1}, {2, 3}, {4, 5, 6}, {7, 8}, {9}, {10, 11, 12, 13}, {}, {14}, {15, 16}},
              // Valid for elements 1, 3, 5, 7,...
              nulls_at_multiples_of(2)};
  auto const structs_tgt = structs_col{{lists_col_tgt}}.release();

  // Expected data
  auto const validity_expected =
    std::vector<bool>{false, true, true, false, false, true, true, false, false};
  auto lists_col_expected = lists_col{
    {{1}, {2, 3}, {80}, {70, 75}, {55, 60, 65}, {35, 40, 45, 50}, {5}, {10, 15}, {20, 25, 30}},
    validity_expected.begin()};
  auto const structs_expected = structs_col{{lists_col_expected}}.release();

  // The first 2 elements of the target is not overwritten
  auto const scatter_map = int32s_col{-3, -2, -1, 5, 4, 3, 2}.release();
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    *structs_expected, scatter_structs(structs_src, structs_tgt, scatter_map), verbosity);
}

TYPED_TEST(TypedStructScatterTest, SourceSmallerThanTargetScatterTest)
{
  using col_wrapper = cudf::test::fixed_width_column_wrapper<TypeParam, int32_t>;

  auto child_col_src     = col_wrapper{22, 55};
  auto const structs_src = structs_col{{child_col_src}}.release();

  auto child_col_tgt     = col_wrapper{0, 1, 2, 3, 4, 5, 6};
  auto const structs_tgt = structs_col{{child_col_tgt}}.release();

  auto child_col_expected     = col_wrapper{0, 1, 22, 3, 4, 55, 6};
  auto const structs_expected = structs_col{{child_col_expected}}.release();

  auto const scatter_map = int32s_col{2, 5}.release();
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    *structs_expected, scatter_structs(structs_src, structs_tgt, scatter_map), verbosity);
}

TYPED_TEST(TypedStructScatterTest, IntStructNullMaskRegression)
{
  using col_wrapper = cudf::test::fixed_width_column_wrapper<TypeParam, int32_t>;

  auto child_tgt      = col_wrapper({0, null, 2}, null_at(1));
  auto struct_col_tgt = structs_col({child_tgt}).release();

  auto child_src      = col_wrapper{20};
  auto struct_col_src = structs_col({child_src}).release();

  auto scatter_map = int32s_col{2}.release();

  auto expected_child  = col_wrapper({0, null, 20}, null_at(1));
  auto expected_struct = structs_col({expected_child}).release();

  auto const result = scatter_structs(struct_col_src, struct_col_tgt, scatter_map);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*expected_struct, result, verbosity);
}
