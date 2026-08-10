/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/iterator_utilities.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/copying.hpp>
#include <cudf/scalar/scalar_factories.hpp>

using namespace cudf::test::iterators;

struct CopyIfElseNestedTest : cudf::test::BaseFixture {};

template <typename T>
struct TypedCopyIfElseNestedTest : CopyIfElseNestedTest {};

TYPED_TEST_SUITE(TypedCopyIfElseNestedTest, cudf::test::FixedWidthTypes);

TYPED_TEST(TypedCopyIfElseNestedTest, Structs)
{
  using T = TypeParam;

  using ints    = cudf::test::fixed_width_column_wrapper<T, int32_t>;
  using strings = cudf::test::strings_column_wrapper;
  using structs = cudf::test::structs_column_wrapper;
  using bools   = cudf::test::fixed_width_column_wrapper<bool, int32_t>;

  auto lhs_ints_child     = ints{0, 1, 2, 3, 4, 5, 6};
  auto lhs_strings_child  = strings{"0", "1", "2", "3", "4", "5", "6"};
  auto lhs_structs_column = structs{{lhs_ints_child, lhs_strings_child}}.release();

  auto rhs_ints_child     = ints{0, 11, 22, 33, 44, 55, 66};
  auto rhs_strings_child  = strings{"00", "11", "22", "33", "44", "55", "66"};
  auto rhs_structs_column = structs{{rhs_ints_child, rhs_strings_child}}.release();

  auto selector_column = bools{1, 1, 0, 1, 1, 0, 1}.release();

  auto result_column = cudf::copy_if_else(
    lhs_structs_column->view(), rhs_structs_column->view(), selector_column->view());

  auto expected_ints    = ints{0, 1, 22, 3, 4, 55, 6};
  auto expected_strings = strings{"0", "1", "22", "3", "4", "55", "6"};
  auto expected_result  = structs{{expected_ints, expected_strings}}.release();

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(result_column->view(), expected_result->view());
}

TYPED_TEST(TypedCopyIfElseNestedTest, StructsWithNulls)
{
  using T = TypeParam;

  using ints    = cudf::test::fixed_width_column_wrapper<T, int32_t>;
  using strings = cudf::test::strings_column_wrapper;
  using structs = cudf::test::structs_column_wrapper;
  using bools   = cudf::test::fixed_width_column_wrapper<bool, int32_t>;

  auto null_at_0 = null_at(0);
  auto null_at_3 = null_at(3);
  auto null_at_5 = null_at(5);

  auto lhs_ints_child     = ints{{0, 1, 2, 3, 4, 5, 6}, null_at_0};
  auto lhs_strings_child  = strings{"0", "1", "2", "3", "4", "5", "6"};
  auto lhs_structs_column = structs{{lhs_ints_child, lhs_strings_child}, null_at_3}.release();

  auto rhs_ints_child     = ints{0, 11, 22, 33, 44, 55, 66};
  auto rhs_strings_child  = strings{{"00", "11", "22", "33", "44", "55", "66"}, null_at_5};
  auto rhs_structs_column = structs{{rhs_ints_child, rhs_strings_child}}.release();

  auto selector_column = bools{1, 1, 0, 1, 1, 0, 1}.release();

  auto result_column = cudf::copy_if_else(
    lhs_structs_column->view(), rhs_structs_column->view(), selector_column->view());

  auto null_at_0_3 = nulls_at(std::vector<cudf::size_type>{0, 3});
  auto null_at_3_5 = nulls_at(std::vector<cudf::size_type>{3, 5});

  auto expected_ints    = ints{{-1, 1, 22, 3, 4, 55, 6}, null_at_0_3};
  auto expected_strings = strings{{"0", "1", "22", "", "4", "", "6"}, null_at_3_5};
  auto expected_result  = structs{{expected_ints, expected_strings}, null_at_3}.release();

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(result_column->view(), expected_result->view());
}

TYPED_TEST(TypedCopyIfElseNestedTest, LongerStructsWithNulls)
{
  using T = TypeParam;

  using ints    = cudf::test::fixed_width_column_wrapper<T, int32_t>;
  using structs = cudf::test::structs_column_wrapper;
  using bools   = cudf::test::fixed_width_column_wrapper<bool, int32_t>;

  auto selector_column = bools{1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0,
                               0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0,
                               0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0}
                           .release();
  auto lhs_child_1 =
    ints{{27, -80, -24, 76,  -56, 42,  5,   13,  -69, -77, 61,   -77,  72,  0,   31,  118, -30,
          86, 125, 0,   0,   0,   75,  -49, 125, 60,  116, 118,  64,   20,  -70, -18, 0,   -25,
          22, -46, -89, -9,  27,  -56, -77, 123, 0,   -90, 87,   -113, -37, 22,  -22, -53, 73,
          99, 113, -2,  -24, 113, 75,  6,   82,  -58, 122, -123, -127, 19,  -62, -24},
         nulls_at(std::vector<cudf::size_type>{13, 19, 20, 21, 32, 42})};

  auto lhs_structs_column = structs{{lhs_child_1}}.release();
  auto result_column      = cudf::copy_if_else(
    lhs_structs_column->view(), lhs_structs_column->view(), selector_column->view());

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(result_column->view(), lhs_structs_column->view());
}

TYPED_TEST(TypedCopyIfElseNestedTest, ScalarStructBothInvalid)
{
  using T = TypeParam;

  using ints    = cudf::test::fixed_width_column_wrapper<T, int32_t>;
  using strings = cudf::test::strings_column_wrapper;
  using structs = cudf::test::structs_column_wrapper;
  using bools   = cudf::test::fixed_width_column_wrapper<bool, int32_t>;

  auto lhs_child_ints    = ints{11};
  auto lhs_child_strings = strings{"11"};
  auto lhs_children      = std::vector<cudf::column_view>{{lhs_child_ints, lhs_child_strings}};
  auto lhs_scalar        = cudf::struct_scalar{lhs_children, false};

  auto rhs_child_ints    = ints{{22}, null_at(0)};
  auto rhs_child_strings = strings{"22"};
  auto rhs_children      = std::vector<cudf::column_view>{{rhs_child_ints, rhs_child_strings}};
  auto rhs_scalar        = cudf::struct_scalar{rhs_children, false};

  auto selector_column = bools{1, 1, 0, 1, 1, 0, 1}.release();

  auto expected_ints    = ints{-11, -11, -22, -11, -11, -22, -11};
  auto expected_strings = strings{"-11", "-11", "-22", "-11", "-22", "-11", "-11"};
  auto expected_result  = structs{{expected_ints, expected_strings}, all_nulls()}.release();

  auto result_column = cudf::copy_if_else(lhs_scalar, rhs_scalar, selector_column->view());

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected_result->view(), result_column->view());
}

TYPED_TEST(TypedCopyIfElseNestedTest, ScalarStructBothValid)
{
  using T = TypeParam;

  using ints    = cudf::test::fixed_width_column_wrapper<T, int32_t>;
  using strings = cudf::test::strings_column_wrapper;
  using structs = cudf::test::structs_column_wrapper;
  using bools   = cudf::test::fixed_width_column_wrapper<bool, int32_t>;

  auto lhs_child_ints    = ints{11};
  auto lhs_child_strings = strings{{"11"}, null_at(0)};
  auto lhs_children      = std::vector<cudf::column_view>{{lhs_child_ints, lhs_child_strings}};
  auto lhs_scalar        = cudf::make_struct_scalar(lhs_children);

  auto rhs_child_ints    = ints{{22}, null_at(0)};
  auto rhs_child_strings = strings{"22"};
  auto rhs_children      = std::vector<cudf::column_view>{{rhs_child_ints, rhs_child_strings}};
  auto rhs_scalar        = cudf::make_struct_scalar(rhs_children);

  auto selector_column = bools{1, 1, 0, 1, 1, 0, 1}.release();

  auto expected_ints =
    ints{{11, 11, -22, 11, 11, -22, 11}, nulls_at(std::vector<cudf::size_type>{2, 5})};
  auto expected_strings = strings{{"NA", "NA", "22", "NA", "NA", "22", "NA"},
                                  nulls_at(std::vector<cudf::size_type>{0, 1, 3, 4, 6})};
  auto expected_result  = structs{{expected_ints, expected_strings}}.release();

  auto result_column = cudf::copy_if_else(*lhs_scalar, *rhs_scalar, selector_column->view());

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected_result->view(), result_column->view());
}

TYPED_TEST(TypedCopyIfElseNestedTest, ScalarStructLeft)
{
  using T = TypeParam;

  using ints    = cudf::test::fixed_width_column_wrapper<T, int32_t>;
  using strings = cudf::test::strings_column_wrapper;
  using structs = cudf::test::structs_column_wrapper;
  using bools   = cudf::test::fixed_width_column_wrapper<bool, int32_t>;

  auto lhs_child_ints    = ints{11};
  auto lhs_child_strings = strings{{"11"}, null_at(0)};
  auto lhs_children      = std::vector<cudf::column_view>{{lhs_child_ints, lhs_child_strings}};
  auto lhs_scalar        = cudf::make_struct_scalar(lhs_children);

  auto rhs_child_ints    = ints{{22, 22, 22, 22, 22, 22, 22}, null_at(2)};
  auto rhs_child_strings = strings{"22", "22", "22", "22", "22", "22", "22"};
  auto rhs_column        = structs{{rhs_child_ints, rhs_child_strings}, null_at(5)}.release();

  auto selector_column = bools{1, 1, 0, 1, 1, 0, 1}.release();

  auto lhs_column = cudf::make_column_from_scalar(*lhs_scalar, selector_column->size());

  auto expected_ints    = ints{{11, 11, -22, 11, 11, 22, 11}, null_at(2)};
  auto expected_strings = strings{{"NA", "NA", "22", "NA", "NA", "22", "NA"},
                                  nulls_at(std::vector<cudf::size_type>{0, 1, 3, 4, 6})};
  auto expected_result  = structs{{expected_ints, expected_strings}, null_at(5)}.release();

  auto result_column = cudf::copy_if_else(*lhs_scalar, rhs_column->view(), selector_column->view());

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected_result->view(), result_column->view());
}

TYPED_TEST(TypedCopyIfElseNestedTest, ScalarStructRight)
{
  using T = TypeParam;

  using ints    = cudf::test::fixed_width_column_wrapper<T, int32_t>;
  using strings = cudf::test::strings_column_wrapper;
  using structs = cudf::test::structs_column_wrapper;
  using bools   = cudf::test::fixed_width_column_wrapper<bool, int32_t>;

  auto lhs_child_ints =
    ints{{11, 11, 11, 11, 11, 11, 11}, nulls_at(std::vector<cudf::size_type>{1, 4})};
  auto lhs_child_strings = strings{"11", "11", "11", "11", "11", "11", "11"};
  auto lhs_column        = structs{{lhs_child_ints, lhs_child_strings}, null_at(6)}.release();

  auto rhs_child_ints    = ints{{22}, null_at(0)};
  auto rhs_child_strings = strings{"22"};
  auto rhs_children      = std::vector<cudf::column_view>{{rhs_child_ints, rhs_child_strings}};
  auto rhs_scalar        = cudf::make_struct_scalar(rhs_children);

  auto selector_column = bools{1, 1, 0, 1, 1, 0, 1}.release();

  auto expected_ints =
    ints{{11, 11, -22, 11, 11, -22, 11}, nulls_at(std::vector<cudf::size_type>{1, 2, 4, 5})};
  auto expected_strings = strings{"11", "11", "22", "11", "11", "22", "11"};
  auto expected_result  = structs{{expected_ints, expected_strings}, null_at(6)}.release();

  auto result_column = cudf::copy_if_else(lhs_column->view(), *rhs_scalar, selector_column->view());
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected_result->view(), result_column->view());
}

TYPED_TEST(TypedCopyIfElseNestedTest, Lists)
{
  using T = TypeParam;

  using lcw = cudf::test::lists_column_wrapper<T, int32_t>;

  auto lhs =
    lcw{{0, 0}, {1, 1}, {2, 2}, {3, 3, 3}, {4, 4, 4, 4}, {5, 5, 5, 5, 5}, {6, 6, 6, 6, 6, 6}}
      .release();

  auto rhs = lcw{{0, 0},
                 {11, 11},
                 {22, 22},
                 {33, 33, 33},
                 {44, 44, 44, 44},
                 {55, 55, 55, 55, 55},
                 {66, 66, 66, 66, 66, 66}}
               .release();

  auto selector_column =
    cudf::test::fixed_width_column_wrapper<bool, int32_t>{1, 1, 0, 1, 1, 0, 1}.release();

  auto result_column = cudf::copy_if_else(lhs->view(), rhs->view(), selector_column->view());

  auto expected_output =
    lcw{{0, 0}, {1, 1}, {22, 22}, {3, 3, 3}, {4, 4, 4, 4}, {55, 55, 55, 55, 55}, {6, 6, 6, 6, 6, 6}}
      .release();

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(result_column->view(), expected_output->view());
}

TYPED_TEST(TypedCopyIfElseNestedTest, ListsWithNulls)
{
  using T = TypeParam;

  using lcw = cudf::test::lists_column_wrapper<T, int32_t>;

  auto null_at_0 = null_at(0);
  auto null_at_4 = null_at(4);
  auto null_at_5 = null_at(5);

  auto lhs = lcw{{{0, 0},
                  {1, 1},
                  lcw{{2, 2}, null_at_0},
                  lcw{{3, 3, 3}, null_at_0},
                  {4, 4, 4, 4},
                  {5, 5, 5, 5, 5},
                  {6, 6, 6, 6, 6, 6}},
                 null_at_4}
               .release();

  auto rhs = lcw{{{0, 0},
                  {11, 11},
                  {22, 22},
                  {33, 33, 33},
                  {44, 44, 44, 44},
                  {55, 55, 55, 55, 55},
                  {66, 66, 66, 66, 66, 66}},
                 null_at_5}
               .release();

  auto selector_column =
    cudf::test::fixed_width_column_wrapper<bool, int32_t>{1, 1, 0, 1, 1, 0, 1}.release();

  auto result_column = cudf::copy_if_else(lhs->view(), rhs->view(), selector_column->view());

  auto null_at_4_5 = nulls_at(std::vector<cudf::size_type>{4, 5});

  auto expected_output =
    lcw{{{0, 0}, {1, 1}, {22, 22}, lcw{{3, 3, 3}, null_at_0}, {}, {}, {6, 6, 6, 6, 6, 6}},
        null_at_4_5}
      .release();

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(result_column->view(), expected_output->view());
}

TYPED_TEST(TypedCopyIfElseNestedTest, ListsWithStructs)
{
  using T = TypeParam;

  using ints    = cudf::test::fixed_width_column_wrapper<T, int32_t>;
  using strings = cudf::test::strings_column_wrapper;
  using structs = cudf::test::structs_column_wrapper;
  using bools   = cudf::test::fixed_width_column_wrapper<bool, int32_t>;
  using offsets = cudf::test::fixed_width_column_wrapper<cudf::size_type, int32_t>;

  auto const null_at_0 = null_at(0);
  auto const null_at_3 = null_at(3);
  auto const null_at_4 = null_at(4);
  auto const null_at_6 = null_at(6);
  auto const null_at_7 = null_at(7);
  auto const null_at_8 = null_at(8);

  auto lhs_ints    = ints{{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}, null_at_3};
  auto lhs_strings = strings{{"0", "1", "2", "3", "4", "5", "6", "7", "8", "9"}, null_at_4};
  auto lhs_structs = structs{{lhs_ints, lhs_strings}}.release();
  auto lhs_offsets = offsets{0, 2, 4, 6, 10, 10}.release();

  auto [null_mask, null_count] = cudf::test::detail::make_null_mask(null_at_4, null_at_4 + 5);
  auto const lhs               = cudf::make_lists_column(
    5, std::move(lhs_offsets), std::move(lhs_structs), null_count, std::move(null_mask));

  auto rhs_ints = ints{{0, 11, 22, 33, 44, 55, 66, 77, 88, 99}, null_at_6};
  auto rhs_strings =
    strings{{"00", "11", "22", "33", "44", "55", "66", "77", "88", "99"}, null_at_7};
  auto rhs_structs = structs{{rhs_ints, rhs_strings}, null_at_8};
  auto rhs_offsets = offsets{0, 0, 4, 6, 8, 10};

  std::tie(null_mask, null_count) = cudf::test::detail::make_null_mask(null_at_0, null_at_0 + 5);
  auto const rhs                  = cudf::make_lists_column(
    5, rhs_offsets.release(), rhs_structs.release(), null_count, std::move(null_mask));

  auto selector_column = bools{1, 0, 1, 0, 1}.release();

  auto result_column = cudf::copy_if_else(lhs->view(), rhs->view(), selector_column->view());

  auto const null_at_6_9 = nulls_at(std::vector<cudf::size_type>{6, 9});
  auto expected_ints     = ints{{0, 1, 0, 11, 22, 33, 4, 5, -1, 77}, null_at_8};
  auto expected_strings =
    strings{{"0", "1", "00", "11", "22", "33", "", "5", "66", ""}, null_at_6_9};
  auto expected_structs = structs{{expected_ints, expected_strings}};
  auto expected_offsets = offsets{0, 2, 6, 8, 10, 10};

  std::tie(null_mask, null_count) = cudf::test::detail::make_null_mask(null_at_4, null_at_4 + 5);
  auto const expected             = cudf::make_lists_column(
    5, expected_offsets.release(), expected_structs.release(), null_count, std::move(null_mask));

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(result_column->view(), expected->view());
}

TYPED_TEST(TypedCopyIfElseNestedTest, ScalarListBothInvalid)
{
  using T = TypeParam;

  using ints  = cudf::test::fixed_width_column_wrapper<T, int32_t>;
  using bools = cudf::test::fixed_width_column_wrapper<bool, int32_t>;
  using lcw   = cudf::test::lists_column_wrapper<T, int32_t>;

  auto lhs_scalar = cudf::list_scalar{ints{}, false};
  auto rhs_scalar = cudf::list_scalar{ints{}, false};

  auto selector_column = bools{1, 1, 0, 1, 1, 0, 1}.release();

  auto expected = lcw{{lcw{}, lcw{}, lcw{}, lcw{}, lcw{}, lcw{}, lcw{}}, all_nulls()}.release();

  auto result = cudf::copy_if_else(lhs_scalar, rhs_scalar, selector_column->view());

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected->view(), result->view());
}

TYPED_TEST(TypedCopyIfElseNestedTest, ScalarListBothValid)
{
  using T = TypeParam;

  using ints  = cudf::test::fixed_width_column_wrapper<T, int32_t>;
  using bools = cudf::test::fixed_width_column_wrapper<bool, int32_t>;
  using lcw   = cudf::test::lists_column_wrapper<T, int32_t>;

  auto lhs_scalar = cudf::make_list_scalar(ints{{33, 33, 33}, null_at(1)});
  auto rhs_scalar = cudf::make_list_scalar(ints{{22, 22}, null_at(0)});

  auto selector_column = bools{1, 1, 0, 1, 1, 0, 1}.release();

  auto expected =
    lcw{
      lcw{{33, 33, 33}, null_at(1)},
      lcw{{33, 33, 33}, null_at(1)},
      lcw{{22, 22}, null_at(0)},
      lcw{{33, 33, 33}, null_at(1)},
      lcw{{33, 33, 33}, null_at(1)},
      lcw{{22, 22}, null_at(0)},
      lcw{{33, 33, 33}, null_at(1)},
    }
      .release();

  auto result = cudf::copy_if_else(*lhs_scalar, *rhs_scalar, selector_column->view());
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected->view(), result->view());
}

TYPED_TEST(TypedCopyIfElseNestedTest, ScalarListLeft)
{
  using T = TypeParam;

  using ints  = cudf::test::fixed_width_column_wrapper<T, int32_t>;
  using bools = cudf::test::fixed_width_column_wrapper<bool, int32_t>;
  using lcw   = cudf::test::lists_column_wrapper<T, int32_t>;

  auto lhs_scalar = cudf::make_list_scalar(ints{{33, 33, 33}, null_at(1)});
  auto rhs_column = lcw{{{-2, -1},
                         {-2, -1, 0},
                         {21, 22},
                         {-20, -10, 0},
                         {-200, -100, 0, 100},
                         lcw{{23, 24, 25, 26, 27, 28}, null_at(1)},
                         {-400}},
                        null_at(2)}
                      .release();

  auto selector_column = bools{1, 1, 0, 1, 1, 0, 1}.release();

  auto expected = lcw{{lcw{{33, 33, 33}, null_at(1)},
                       lcw{{33, -33, 33}, null_at(1)},
                       {-21, -22},
                       lcw{{33, -33, 33}, null_at(1)},
                       lcw{{33, -33, 33}, null_at(1)},
                       lcw{{23, -24, 25, 26, 27, 28}, null_at(1)},
                       lcw{{33, -33, 33}, null_at(1)}},
                      null_at(2)}
                    .release();

  auto result = cudf::copy_if_else(*lhs_scalar, rhs_column->view(), selector_column->view());

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected->view(), result->view());
}

TYPED_TEST(TypedCopyIfElseNestedTest, ScalarListRight)
{
  using T = TypeParam;

  using ints  = cudf::test::fixed_width_column_wrapper<T, int32_t>;
  using bools = cudf::test::fixed_width_column_wrapper<bool, int32_t>;
  using lcw   = cudf::test::lists_column_wrapper<T, int32_t>;

  auto lhs_column = lcw{{{-2, -1},
                         {-2, -1, 0},
                         {21, 22},
                         {-20, -10, 0},
                         {-200, -100, 0, 100},
                         lcw{{23, 24, 25, 26, 27, 28}, null_at(1)},
                         {-400}},
                        null_at(2)}
                      .release();

  auto rhs_scalar = cudf::make_list_scalar(ints{{33, 33, 33}, null_at(1)});

  auto selector_column = bools{0, 0, 1, 0, 0, 1, 0}.release();

  auto expected = lcw{{
                        lcw{{33, -33, 33}, null_at(1)},
                        lcw{{33, -33, 33}, null_at(1)},
                        {-21, -22},
                        lcw{{33, -33, 33}, null_at(1)},
                        lcw{{33, -33, 33}, null_at(1)},
                        lcw{{23, -24, 25, 26, 27, 28}, null_at(1)},
                        lcw{{33, -33, 33}, null_at(1)},
                      },
                      null_at(2)}
                    .release();

  auto result = cudf::copy_if_else(lhs_column->view(), *rhs_scalar, selector_column->view());

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected->view(), result->view());
}
