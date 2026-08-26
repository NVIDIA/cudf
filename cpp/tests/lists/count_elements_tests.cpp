/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/lists_column_wrapper.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/copying.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/lists/count_elements.hpp>

struct ListsElementsTest : public cudf::test::BaseFixture {};

using NumericTypesNotBool =
  cudf::test::Concat<cudf::test::IntegralTypesNotBool, cudf::test::FloatingPointTypes>;

template <typename T>
class ListsElementsNumericsTest : public ListsElementsTest {};

TYPED_TEST_SUITE(ListsElementsNumericsTest, NumericTypesNotBool);

TYPED_TEST(ListsElementsNumericsTest, CountElements)
{
  auto validity = cudf::detail::make_counting_transform_iterator(cudf::size_type{0},
                                                                 [](auto i) { return i != 1; });
  using LCW     = cudf::test::lists_column_wrapper<TypeParam>;
  LCW input({LCW{3, 2, 1}, LCW{}, LCW{30, 20, 10, 50}, LCW{100, 120}, LCW{0}}, validity);

  auto result = cudf::lists::count_elements(cudf::lists_column_view(input));
  cudf::test::fixed_width_column_wrapper<int32_t> expected({3, 0, 4, 2, 1},
                                                           {true, false, true, true, true});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *result);
}

TEST_F(ListsElementsTest, CountElementsStrings)
{
  auto validity = cudf::detail::make_counting_transform_iterator(cudf::size_type{0},
                                                                 [](auto i) { return i != 1; });
  using LCW     = cudf::test::lists_column_wrapper<cudf::string_view>;
  LCW input(
    {LCW{"", "Héllo", "thesé"}, LCW{}, LCW{"are", "some", "", "z"}, LCW{"tést", "String"}, LCW{""}},
    validity);

  auto result = cudf::lists::count_elements(cudf::lists_column_view(input));
  cudf::test::fixed_width_column_wrapper<int32_t> expected({3, 0, 4, 2, 1},
                                                           {true, false, true, true, true});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *result);
}

TEST_F(ListsElementsTest, CountElementsSliced)
{
  auto validity = cudf::detail::make_counting_transform_iterator(cudf::size_type{0},
                                                                 [](auto i) { return i != 1; });
  using LCW     = cudf::test::lists_column_wrapper<cudf::string_view>;
  LCW input(
    {LCW{"", "Héllo", "thesé"}, LCW{}, LCW{"are", "some", "", "z"}, LCW{"tést", "String"}, LCW{""}},
    validity);

  auto sliced = cudf::slice(input, {1, 4}).front();
  auto result = cudf::lists::count_elements(cudf::lists_column_view(sliced));
  cudf::test::fixed_width_column_wrapper<int32_t> expected({0, 4, 2}, {false, true, true});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *result);
}

TYPED_TEST(ListsElementsNumericsTest, CountElementsNestedLists)
{
  std::vector<int32_t> validity{1, 0, 1, 1};
  using LCW = cudf::test::lists_column_wrapper<TypeParam>;
  LCW list({LCW{LCW{2, 3}, LCW{4, 5}},
            LCW{LCW{}},
            LCW{LCW{6, 7, 8}, LCW{9, 10, 11}, LCW({12, 13, 14}, validity.begin())},
            LCW{LCW{15, 16}, LCW{17, 18}, LCW{19, 20}, LCW{21, 22}, LCW{23, 24}}},
           validity.begin());

  auto result = cudf::lists::count_elements(cudf::lists_column_view(list));
  cudf::test::fixed_width_column_wrapper<int32_t> expected({2, 1, 3, 5}, {true, false, true, true});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *result);
}

TEST_F(ListsElementsTest, CountElementsEmpty)
{
  using LCW = cudf::test::lists_column_wrapper<cudf::string_view>;

  LCW empty{};
  auto result = cudf::lists::count_elements(cudf::lists_column_view(empty));
  EXPECT_EQ(0, result->size());
}
