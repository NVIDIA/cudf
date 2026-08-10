/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/iterator_utilities.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/column/column_view.hpp>
#include <cudf/copying.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>

template <typename T>
class TypedScatterListsTest : public cudf::test::BaseFixture {};

TYPED_TEST_SUITE(TypedScatterListsTest, cudf::test::FixedWidthTypes);

class ScatterListsTest : public cudf::test::BaseFixture {};

TYPED_TEST(TypedScatterListsTest, ListsOfFixedWidth)
{
  using T = TypeParam;

  auto src_list_column = cudf::test::lists_column_wrapper<T, int32_t>{{9, 9, 9, 9}, {8, 8, 8}};

  auto target_list_column = cudf::test::lists_column_wrapper<T, int32_t>{
    {0, 0}, {1, 1}, {2, 2}, {3, 3}, {4, 4}, {5, 5}, {6, 6}};

  auto scatter_map = cudf::test::fixed_width_column_wrapper<cudf::size_type>{2, 0};

  auto ret = cudf::scatter(
    cudf::table_view({src_list_column}), scatter_map, cudf::table_view({target_list_column}));

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(
    ret->get_column(0),
    cudf::test::lists_column_wrapper<T, int32_t>{
      {8, 8, 8}, {1, 1}, {9, 9, 9, 9}, {3, 3}, {4, 4}, {5, 5}, {6, 6}});
}

TYPED_TEST(TypedScatterListsTest, SlicedInputLists)
{
  using T = TypeParam;

  auto src_list_column =
    cudf::test::lists_column_wrapper<T, int32_t>{{0, 0, 0, 0}, {9, 9, 9, 9}, {8, 8, 8}, {7, 7, 7}}
      .release();
  auto src_sliced = cudf::slice(src_list_column->view(), {1, 3}).front();

  auto target_list_column =
    cudf::test::lists_column_wrapper<T, int32_t>{
      {0, 0}, {1, 1}, {2, 2}, {3, 3}, {4, 4}, {5, 5}, {6, 6}}
      .release();

  auto scatter_map = cudf::test::fixed_width_column_wrapper<cudf::size_type>{2, 0};

  auto ret_1 = cudf::scatter(
    cudf::table_view({src_sliced}), scatter_map, cudf::table_view({target_list_column->view()}));

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(
    ret_1->get_column(0),
    cudf::test::lists_column_wrapper<T, int32_t>{
      {8, 8, 8}, {1, 1}, {9, 9, 9, 9}, {3, 3}, {4, 4}, {5, 5}, {6, 6}});

  auto target_sliced = cudf::slice(target_list_column->view(), {1, 6});

  auto ret_2 =
    cudf::scatter(cudf::table_view({src_sliced}), scatter_map, cudf::table_view({target_sliced}));

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(
    ret_2->get_column(0),
    cudf::test::lists_column_wrapper<T, int32_t>{{8, 8, 8}, {2, 2}, {9, 9, 9, 9}, {4, 4}, {5, 5}});
}

TYPED_TEST(TypedScatterListsTest, EmptyListsOfFixedWidth)
{
  using T = TypeParam;

  auto src_child = cudf::test::fixed_width_column_wrapper<T, int32_t>{
    {9, 9, 9, 9, 8, 8, 8},
  };

  // One null list row, and one row with nulls.
  auto src_list_column = cudf::make_lists_column(
    3,
    cudf::test::fixed_width_column_wrapper<cudf::size_type>{0, 4, 7, 7}.release(),
    src_child.release(),
    0,
    {});

  auto target_list_column = cudf::test::lists_column_wrapper<T, int32_t>{
    {0, 0}, {1, 1}, {2, 2}, {3, 3}, {4, 4}, {5, 5}, {6, 6}};

  auto scatter_map = cudf::test::fixed_width_column_wrapper<cudf::size_type>{2, 0, 5};

  auto ret = cudf::scatter(cudf::table_view({src_list_column->view()}),
                           scatter_map,
                           cudf::table_view({target_list_column}));

  auto expected_child_ints = cudf::test::fixed_width_column_wrapper<T, int32_t>{
    {8, 8, 8, 1, 1, 9, 9, 9, 9, 3, 3, 4, 4, 6, 6}};
  auto expected_lists_column = cudf::make_lists_column(
    7,
    cudf::test::fixed_width_column_wrapper<cudf::size_type>{0, 3, 5, 9, 11, 13, 13, 15}.release(),
    expected_child_ints.release(),
    0,
    {});

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected_lists_column->view(), ret->get_column(0));
}

TYPED_TEST(TypedScatterListsTest, EmptyListsOfNullableFixedWidth)
{
  using T = TypeParam;

  auto src_child = cudf::test::fixed_width_column_wrapper<T, int32_t>{{9, 9, 9, 9, 8, 8, 8},
                                                                      {1, 1, 1, 0, 1, 1, 1}};

  // One null list row, and one row with nulls.
  auto src_list_column = cudf::make_lists_column(
    3,
    cudf::test::fixed_width_column_wrapper<cudf::size_type>{0, 4, 7, 7}.release(),
    src_child.release(),
    0,
    {});

  auto target_list_column = cudf::test::lists_column_wrapper<T, int32_t>{
    {0, 0}, {1, 1}, {2, 2}, {3, 3}, {4, 4}, {5, 5}, {6, 6}};

  auto scatter_map = cudf::test::fixed_width_column_wrapper<cudf::size_type>{2, 0, 5};

  auto ret = cudf::scatter(cudf::table_view({src_list_column->view()}),
                           scatter_map,
                           cudf::table_view({target_list_column}));

  auto expected_child_ints = cudf::test::fixed_width_column_wrapper<T, int32_t>{
    {8, 8, 8, 1, 1, 9, 9, 9, 9, 3, 3, 4, 4, 6, 6}, {1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1}};
  auto expected_lists_column = cudf::make_lists_column(
    7,
    cudf::test::fixed_width_column_wrapper<cudf::size_type>{0, 3, 5, 9, 11, 13, 13, 15}.release(),
    expected_child_ints.release(),
    0,
    {});

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected_lists_column->view(), ret->get_column(0));
}

TYPED_TEST(TypedScatterListsTest, NullableListsOfNullableFixedWidth)
{
  using T = TypeParam;

  auto src_child = cudf::test::fixed_width_column_wrapper<T, int32_t>{{9, 9, 9, 9, 8, 8, 8},
                                                                      {1, 1, 1, 0, 1, 1, 1}};

  auto src_list_validity = cudf::test::iterators::null_at(2);
  auto [null_mask, null_count] =
    cudf::test::detail::make_null_mask(src_list_validity, src_list_validity + 3);
  // One null list row, and one row with nulls.
  auto src_list_column = cudf::make_lists_column(
    3,
    cudf::test::fixed_width_column_wrapper<cudf::size_type>{0, 4, 7, 7}.release(),
    src_child.release(),
    null_count,
    std::move(null_mask));

  auto target_list_column = cudf::test::lists_column_wrapper<T, int32_t>{
    {0, 0}, {1, 1}, {2, 2}, {3, 3}, {4, 4}, {5, 5}, {6, 6}};

  auto scatter_map = cudf::test::fixed_width_column_wrapper<cudf::size_type>{2, 0, 5};

  auto ret = cudf::scatter(cudf::table_view({src_list_column->view()}),
                           scatter_map,
                           cudf::table_view({target_list_column}));

  auto expected_child_ints = cudf::test::fixed_width_column_wrapper<T, int32_t>{
    {8, 8, 8, 1, 1, 9, 9, 9, 9, 3, 3, 4, 4, 6, 6}, {1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1}};

  auto expected_validity = cudf::test::iterators::null_at(5);
  std::tie(null_mask, null_count) =
    cudf::test::detail::make_null_mask(expected_validity, expected_validity + 7);
  auto expected_lists_column = cudf::make_lists_column(
    7,
    cudf::test::fixed_width_column_wrapper<cudf::size_type>{0, 3, 5, 9, 11, 13, 13, 15}.release(),
    expected_child_ints.release(),
    null_count,
    std::move(null_mask));

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected_lists_column->view(), ret->get_column(0));
}

TEST_F(ScatterListsTest, ListsOfStrings)
{
  auto src_list_column = cudf::test::lists_column_wrapper<cudf::string_view>{
    {"all", "the", "leaves", "are", "brown"}, {"california", "dreaming"}};

  auto target_list_column =
    cudf::test::lists_column_wrapper<cudf::string_view>{{"zero"},
                                                        {"one", "one"},
                                                        {"two", "two"},
                                                        {"three", "three", "three"},
                                                        {"four", "four", "four", "four"}};

  auto scatter_map = cudf::test::fixed_width_column_wrapper<int32_t>{2, 0};

  auto ret = cudf::scatter(
    cudf::table_view({src_list_column}), scatter_map, cudf::table_view({target_list_column}));

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(
    cudf::test::lists_column_wrapper<cudf::string_view>{{"california", "dreaming"},
                                                        {"one", "one"},
                                                        {"all", "the", "leaves", "are", "brown"},
                                                        {"three", "three", "three"},
                                                        {"four", "four", "four", "four"}},
    ret->get_column(0));
}

TEST_F(ScatterListsTest, ListsOfNullableStrings)
{
  auto src_strings_column =
    cudf::test::strings_column_wrapper{{"all", "the", "leaves", "", "brown", "", "dreaming"},
                                       {true, true, true, false, true, false, true}};

  auto src_list_column = cudf::make_lists_column(
    2,
    cudf::test::fixed_width_column_wrapper<cudf::size_type>{0, 5, 7}.release(),
    src_strings_column.release(),
    0,
    {});

  auto target_list_column = cudf::test::lists_column_wrapper<cudf::string_view>{{"zero"},
                                                                                {"one", "one"},
                                                                                {"two", "two"},
                                                                                {"three", "three"},
                                                                                {"four", "four"},
                                                                                {"five", "five"}};

  auto scatter_map = cudf::test::fixed_width_column_wrapper<int32_t>{2, 0};

  auto ret = cudf::scatter(cudf::table_view({src_list_column->view()}),
                           scatter_map,
                           cudf::table_view({target_list_column}));

  auto expected_strings =
    cudf::test::strings_column_wrapper{{"",
                                        "dreaming",
                                        "one",
                                        "one",
                                        "all",
                                        "the",
                                        "leaves",
                                        "",
                                        "brown",
                                        "three",
                                        "three",
                                        "four",
                                        "four",
                                        "five",
                                        "five"},
                                       cudf::test::iterators::nulls_at({0, 7})};

  auto expected_lists = cudf::make_lists_column(
    6,
    cudf::test::fixed_width_column_wrapper<cudf::size_type>{0, 2, 4, 9, 11, 13, 15}.release(),
    expected_strings.release(),
    0,
    {});

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected_lists->view(), ret->get_column(0));
}

TEST_F(ScatterListsTest, EmptyListsOfNullableStrings)
{
  auto src_strings_column =
    cudf::test::strings_column_wrapper{{"all", "the", "leaves", "", "brown", "", "dreaming"},
                                       {true, true, true, false, true, false, true}};

  auto src_list_column = cudf::make_lists_column(
    3,
    cudf::test::fixed_width_column_wrapper<cudf::size_type>{0, 5, 5, 7}.release(),
    src_strings_column.release(),
    0,
    {});

  auto target_list_column = cudf::test::lists_column_wrapper<cudf::string_view>{{"zero"},
                                                                                {"one", "one"},
                                                                                {"two", "two"},
                                                                                {"three", "three"},
                                                                                {"four", "four"},
                                                                                {"five", "five"}};

  auto scatter_map = cudf::test::fixed_width_column_wrapper<int32_t>{2, 4, 0};

  auto ret = cudf::scatter(cudf::table_view({src_list_column->view()}),
                           scatter_map,
                           cudf::table_view({target_list_column}));

  auto expected_strings =
    cudf::test::strings_column_wrapper{{"",
                                        "dreaming",
                                        "one",
                                        "one",
                                        "all",
                                        "the",
                                        "leaves",
                                        "",
                                        "brown",
                                        "three",
                                        "three",
                                        "five",
                                        "five"},
                                       cudf::test::iterators::nulls_at({0, 7})};

  auto expected_lists = cudf::make_lists_column(
    6,
    cudf::test::fixed_width_column_wrapper<cudf::size_type>{0, 2, 4, 9, 11, 11, 13}.release(),
    expected_strings.release(),
    0,
    {});

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected_lists->view(), ret->get_column(0));
}

TEST_F(ScatterListsTest, NullableListsOfNullableStrings)
{
  auto src_strings_column =
    cudf::test::strings_column_wrapper{{"all", "the", "leaves", "", "brown", "", "dreaming"},
                                       {true, true, true, false, true, false, true}};

  auto src_validity            = cudf::test::iterators::null_at(1);
  auto [null_mask, null_count] = cudf::test::detail::make_null_mask(src_validity, src_validity + 3);
  auto src_list_column         = cudf::make_lists_column(
    3,
    cudf::test::fixed_width_column_wrapper<cudf::size_type>{0, 5, 5, 7}.release(),
    src_strings_column.release(),
    null_count,
    std::move(null_mask));

  auto target_list_column = cudf::test::lists_column_wrapper<cudf::string_view>{{"zero"},
                                                                                {"one", "one"},
                                                                                {"two", "two"},
                                                                                {"three", "three"},
                                                                                {"four", "four"},
                                                                                {"five", "five"}};

  auto scatter_map = cudf::test::fixed_width_column_wrapper<int32_t>{2, 4, 0};

  auto ret = cudf::scatter(cudf::table_view({src_list_column->view()}),
                           scatter_map,
                           cudf::table_view({target_list_column}));

  auto expected_strings =
    cudf::test::strings_column_wrapper{{"",
                                        "dreaming",
                                        "one",
                                        "one",
                                        "all",
                                        "the",
                                        "leaves",
                                        "",
                                        "brown",
                                        "three",
                                        "three",
                                        "five",
                                        "five"},
                                       cudf::test::iterators::nulls_at({0, 7})};

  auto expected_validity = cudf::test::iterators::null_at(4);
  std::tie(null_mask, null_count) =
    cudf::test::detail::make_null_mask(expected_validity, expected_validity + 6);
  auto expected_lists = cudf::make_lists_column(
    6,
    cudf::test::fixed_width_column_wrapper<cudf::size_type>{0, 2, 4, 9, 11, 11, 13}.release(),
    expected_strings.release(),
    null_count,
    std::move(null_mask));

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected_lists->view(), ret->get_column(0));
}

TYPED_TEST(TypedScatterListsTest, ListsOfLists)
{
  using T = TypeParam;

  auto src_list_column = cudf::test::lists_column_wrapper<T, int32_t>{{{1, 1, 1, 1}, {2, 2, 2, 2}},
                                                                      {{3, 3, 3, 3}, {4, 4, 4, 4}}};

  auto target_list_column =
    cudf::test::lists_column_wrapper<T, int32_t>{{{9, 9, 9}, {8, 8, 8}, {7, 7, 7}},
                                                 {{6, 6, 6}, {5, 5, 5}, {4, 4, 4}},
                                                 {{3, 3, 3}, {2, 2, 2}, {1, 1, 1}},
                                                 {{9, 9}, {8, 8}, {7, 7}},
                                                 {{6, 6}, {5, 5}, {4, 4}},
                                                 {{3, 3}, {2, 2}, {1, 1}}};

  auto scatter_map = cudf::test::fixed_width_column_wrapper<cudf::size_type>{2, 0};

  auto ret = cudf::scatter(
    cudf::table_view({src_list_column}), scatter_map, cudf::table_view({target_list_column}));

  auto expected = cudf::test::lists_column_wrapper<T, int32_t>{{{3, 3, 3, 3}, {4, 4, 4, 4}},
                                                               {{6, 6, 6}, {5, 5, 5}, {4, 4, 4}},
                                                               {{1, 1, 1, 1}, {2, 2, 2, 2}},
                                                               {{9, 9}, {8, 8}, {7, 7}},
                                                               {{6, 6}, {5, 5}, {4, 4}},
                                                               {{3, 3}, {2, 2}, {1, 1}}};

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected, ret->get_column(0));
}

TYPED_TEST(TypedScatterListsTest, EmptyListsOfLists)
{
  using T = TypeParam;

  auto src_list_column = cudf::test::lists_column_wrapper<T, int32_t>{
    {{1, 1, 1, 1}, {2, 2, 2, 2}}, {{3, 3, 3, 3}, {}}, {}};

  auto target_list_column =
    cudf::test::lists_column_wrapper<T, int32_t>{{{9, 9, 9}, {8, 8, 8}, {7, 7, 7}},
                                                 {{6, 6, 6}, {5, 5, 5}, {4, 4, 4}},
                                                 {{3, 3, 3}, {2, 2, 2}, {1, 1, 1}},
                                                 {{9, 9}, {8, 8}, {7, 7}},
                                                 {{6, 6}, {5, 5}, {4, 4}},
                                                 {{3, 3}, {2, 2}, {1, 1}}};

  auto scatter_map = cudf::test::fixed_width_column_wrapper<cudf::size_type>{2, 0, 4};

  auto ret = cudf::scatter(
    cudf::table_view({src_list_column}), scatter_map, cudf::table_view({target_list_column}));

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(
    cudf::test::lists_column_wrapper<T, int32_t>{{{3, 3, 3, 3}, {}},
                                                 {{6, 6, 6}, {5, 5, 5}, {4, 4, 4}},
                                                 {{1, 1, 1, 1}, {2, 2, 2, 2}},
                                                 {{9, 9}, {8, 8}, {7, 7}},
                                                 {},
                                                 {{3, 3}, {2, 2}, {1, 1}}},
    ret->get_column(0));
}

TYPED_TEST(TypedScatterListsTest, NullListsOfLists)
{
  using T = TypeParam;

  auto src_list_column = cudf::test::lists_column_wrapper<T, int32_t>{
    {{{1, 1, 1, 1}, {2, 2, 2, 2}}, {{3, 3, 3, 3}, {}}, {}}, cudf::test::iterators::null_at(2)};

  auto target_list_column =
    cudf::test::lists_column_wrapper<T, int32_t>{{{9, 9, 9}, {8, 8, 8}, {7, 7, 7}},
                                                 {{6, 6, 6}, {5, 5, 5}, {4, 4, 4}},
                                                 {{3, 3, 3}, {2, 2, 2}, {1, 1, 1}},
                                                 {{9, 9}, {8, 8}, {7, 7}},
                                                 {{6, 6}, {5, 5}, {4, 4}},
                                                 {{3, 3}, {2, 2}, {1, 1}}};

  auto scatter_map = cudf::test::fixed_width_column_wrapper<cudf::size_type>{2, 0, 4};

  auto ret = cudf::scatter(
    cudf::table_view({src_list_column}), scatter_map, cudf::table_view({target_list_column}));

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(
    cudf::test::lists_column_wrapper<T, int32_t>{{{{3, 3, 3, 3}, {}},
                                                  {{6, 6, 6}, {5, 5, 5}, {4, 4, 4}},
                                                  {{1, 1, 1, 1}, {2, 2, 2, 2}},
                                                  {{9, 9}, {8, 8}, {7, 7}},
                                                  {},
                                                  {{3, 3}, {2, 2}, {1, 1}}},
                                                 cudf::test::iterators::null_at(4)},
    ret->get_column(0));
}

TYPED_TEST(TypedScatterListsTest, ListsOfStructs)
{
  using T               = TypeParam;
  using offsets_column  = cudf::test::fixed_width_column_wrapper<cudf::size_type>;
  using numerics_column = cudf::test::fixed_width_column_wrapper<T>;

  // clang-format off
  auto source_numerics = numerics_column{
    9, 9, 9, 9,
    8, 8, 8
  };

  auto source_strings = cudf::test::strings_column_wrapper{
    "nine", "nine", "nine", "nine",
    "eight", "eight", "eight"
  };
  // clang-format on

  auto source_structs = cudf::test::structs_column_wrapper{{source_numerics, source_strings}};

  auto source_lists =
    cudf::make_lists_column(2, offsets_column{0, 4, 7}.release(), source_structs.release(), 0, {});

  // clang-format off
  auto target_ints    = numerics_column{
    0, 0,
    1, 1,
    2, 2,
    3, 3,
    4, 4,
    5, 5
  };

  auto target_strings = cudf::test::strings_column_wrapper{
    "zero",  "zero",
    "one",   "one",
    "two",   "two",
    "three", "three",
    "four",  "four",
    "five",  "five"
  };
  // clang-format on

  auto target_structs = cudf::test::structs_column_wrapper{{target_ints, target_strings}};

  auto target_lists = cudf::make_lists_column(
    6, offsets_column{0, 2, 4, 6, 8, 10, 12}.release(), target_structs.release(), 0, {});

  auto scatter_map = offsets_column{2, 0};

  auto scatter_result = cudf::scatter(cudf::table_view({source_lists->view()}),
                                      scatter_map,
                                      cudf::table_view({target_lists->view()}));

  // clang-format off
  auto expected_numerics = numerics_column{
    8, 8, 8,
    1, 1,
    9, 9, 9, 9,
    3, 3, 4, 4, 5, 5
  };

  auto expected_strings = cudf::test::strings_column_wrapper{
    "eight", "eight", "eight",
    "one", "one",
    "nine", "nine", "nine", "nine",
    "three", "three",
    "four", "four",
    "five", "five"
  };
  // clang-format on

  auto expected_structs = cudf::test::structs_column_wrapper{{expected_numerics, expected_strings}};

  auto expected_lists = cudf::make_lists_column(
    6, offsets_column{0, 3, 5, 9, 11, 13, 15}.release(), expected_structs.release(), 0, {});

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected_lists->view(), scatter_result->get_column(0));
}

TYPED_TEST(TypedScatterListsTest, ListsOfStructsWithNullMembers)
{
  using T               = TypeParam;
  using offsets_column  = cudf::test::fixed_width_column_wrapper<cudf::size_type>;
  using numerics_column = cudf::test::fixed_width_column_wrapper<T>;

  // clang-format off
  auto source_numerics = numerics_column{
    {
      9, 9, 9, 9,
      8, 8, 8
    },
    cudf::test::iterators::null_at(3)
  };

  auto source_strings = cudf::test::strings_column_wrapper{
    {
      "nine",  "nine",  "nine", "nine",
      "eight", "", "eight"
    },
    cudf::test::iterators::null_at(5)
  };
  // clang-format on

  auto source_structs = cudf::test::structs_column_wrapper{{source_numerics, source_strings}};

  auto source_lists =
    cudf::make_lists_column(2, offsets_column{0, 4, 7}.release(), source_structs.release(), 0, {});

  // clang-format off
  auto target_ints    = numerics_column{
    0, 0,
    1, 1,
    2, 2,
    3, 3,
    4, 4,
    5, 5
  };

  auto target_strings = cudf::test::strings_column_wrapper{
    "zero", "zero",
    "one",  "one",
    "two",  "two",
    "three","three",
    "four", "four",
    "five", "five"
  };
  // clang-format on

  auto target_structs = cudf::test::structs_column_wrapper{{target_ints, target_strings}};

  auto target_lists = cudf::make_lists_column(
    6, offsets_column{0, 2, 4, 6, 8, 10, 12}.release(), target_structs.release(), 0, {});
  // clang-format on

  auto scatter_map = offsets_column{2, 0};

  auto scatter_result = cudf::scatter(cudf::table_view({source_lists->view()}),
                                      scatter_map,
                                      cudf::table_view({target_lists->view()}));

  // clang-format off
  auto expected_numerics = numerics_column{
    {
      8, 8, 8,
      1, 1,
      9, 9, 9, 9,
      3, 3,
      4, 4,
      5, 5
    },
    cudf::test::iterators::null_at(8)
  };

  auto expected_strings = cudf::test::strings_column_wrapper{
    {
      "eight", "", "eight",
      "one",   "one",
      "nine",  "nine",  "nine", "nine",
      "three", "three",
      "four",  "four",
      "five",  "five"
    },
    cudf::test::iterators::null_at(1)
  };
  // clang-format on

  auto expected_structs = cudf::test::structs_column_wrapper{{expected_numerics, expected_strings}};

  auto expected_lists = cudf::make_lists_column(
    6, offsets_column{0, 3, 5, 9, 11, 13, 15}.release(), expected_structs.release(), 0, {});

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected_lists->view(), scatter_result->get_column(0));
}

TYPED_TEST(TypedScatterListsTest, ListsOfNullStructs)
{
  using T               = TypeParam;
  using offsets_column  = cudf::test::fixed_width_column_wrapper<cudf::size_type>;
  using numerics_column = cudf::test::fixed_width_column_wrapper<T>;

  // clang-format off
  auto source_numerics = numerics_column{
    {
      9, 9, 9, 9,
      8, 8, 8
    },
    cudf::test::iterators::null_at(3)
  };

  auto source_strings = cudf::test::strings_column_wrapper{
    {
      "nine",  "nine",  "nine", "nine",
      "eight", "", "eight"
    },
    cudf::test::iterators::null_at(5)
  };
  // clang-format on

  auto source_structs = cudf::test::structs_column_wrapper{{source_numerics, source_strings},
                                                           cudf::test::iterators::null_at(1)};

  auto source_lists =
    cudf::make_lists_column(2, offsets_column{0, 4, 7}.release(), source_structs.release(), 0, {});

  // clang-format off
  auto target_ints    = numerics_column{
    0, 0,
    1, 1,
    2, 2,
    3, 3,
    4, 4,
    5, 5
  };

  auto target_strings = cudf::test::strings_column_wrapper{
    "zero",  "zero",
    "one",   "one",
    "two",   "two",
    "three", "three",
    "four",  "four",
    "five",  "five"
  };
  // clang-format on

  auto target_structs = cudf::test::structs_column_wrapper{{target_ints, target_strings}};

  auto target_lists = cudf::make_lists_column(
    6, offsets_column{0, 2, 4, 6, 8, 10, 12}.release(), target_structs.release(), 0, {});

  auto scatter_map = offsets_column{2, 0};

  auto scatter_result = cudf::scatter(cudf::table_view({source_lists->view()}),
                                      scatter_map,
                                      cudf::table_view({target_lists->view()}));

  // clang-format off
  auto expected_numerics = numerics_column{
    {
      8, 8, 8,
      1, 1,
      9, 9, 9, 9,
      3, 3,
      4, 4,
      5, 5
    },
    cudf::test::iterators::nulls_at({6, 8})
  };

  auto expected_strings = cudf::test::strings_column_wrapper{
    {
      "eight", "", "eight",
      "one",   "one",
      "nine",  "",  "nine", "nine",
      "three", "three",
      "four",  "four",
      "five",  "five"
    },
    cudf::test::iterators::nulls_at({1, 6})
  };
  // clang-format on

  auto expected_structs = cudf::test::structs_column_wrapper{{expected_numerics, expected_strings},
                                                             cudf::test::iterators::null_at(6)};

  auto expected_lists = cudf::make_lists_column(
    6, offsets_column{0, 3, 5, 9, 11, 13, 15}.release(), expected_structs.release(), 0, {});

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected_lists->view(), scatter_result->get_column(0));
}

TYPED_TEST(TypedScatterListsTest, EmptyListsOfStructs)
{
  using T               = TypeParam;
  using offsets_column  = cudf::test::fixed_width_column_wrapper<cudf::size_type>;
  using numerics_column = cudf::test::fixed_width_column_wrapper<T>;

  // clang-format off
  auto source_numerics = numerics_column{
    {
      9, 9, 9, 9,
      8, 8, 8
    },
    cudf::test::iterators::null_at(3)
  };

  auto source_strings = cudf::test::strings_column_wrapper{
    {
      "nine",  "nine",  "nine", "nine",
      "eight", "", "eight"
    },
    cudf::test::iterators::null_at(5)
  };
  // clang-format on

  auto source_structs = cudf::test::structs_column_wrapper{{source_numerics, source_strings},
                                                           cudf::test::iterators::null_at(1)};

  auto source_lists = cudf::make_lists_column(
    3, offsets_column{0, 4, 7, 7}.release(), source_structs.release(), 0, {});

  // clang-format off
  auto target_ints    = numerics_column{
    0, 0,
    1, 1,
    2, 2,
    3, 3,
    4, 4,
    5, 5
  };

  auto target_strings = cudf::test::strings_column_wrapper{
    "zero",  "zero",
    "one",   "one",
    "two",   "two",
    "three", "three",
    "four",  "four",
    "five",  "five"
  };
  // clang-format on

  auto target_structs = cudf::test::structs_column_wrapper{{target_ints, target_strings}};

  auto target_lists = cudf::make_lists_column(
    6, offsets_column{0, 2, 4, 6, 8, 10, 12}.release(), target_structs.release(), 0, {});

  auto scatter_map = offsets_column{2, 0, 4};

  auto scatter_result = cudf::scatter(cudf::table_view({source_lists->view()}),
                                      scatter_map,
                                      cudf::table_view({target_lists->view()}));

  // clang-format off
  auto expected_numerics = numerics_column{
    {
      8, 8, 8,
      1, 1,
      9, 9, 9, 9,
      3, 3,
      5, 5
    },
    cudf::test::iterators::nulls_at({6, 8})
  };

  auto expected_strings = cudf::test::strings_column_wrapper{
    {
      "eight", "", "eight",
      "one",   "one",
      "nine",  "",  "nine", "nine",
      "three", "three",
      "five",  "five"
    },
    cudf::test::iterators::nulls_at({1, 6})
  };
  // clang-format on

  auto expected_structs = cudf::test::structs_column_wrapper{{expected_numerics, expected_strings},
                                                             cudf::test::iterators::null_at(6)};

  auto expected_lists = cudf::make_lists_column(
    6, offsets_column{0, 3, 5, 9, 11, 11, 13}.release(), expected_structs.release(), 0, {});

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected_lists->view(), scatter_result->get_column(0));
}

TYPED_TEST(TypedScatterListsTest, NullListsOfStructs)
{
  using T               = TypeParam;
  using offsets_column  = cudf::test::fixed_width_column_wrapper<cudf::size_type>;
  using numerics_column = cudf::test::fixed_width_column_wrapper<T>;

  // clang-format off
  auto source_numerics = numerics_column{
    {
      9, 9, 9, 9,
      8, 8, 8
    },
    cudf::test::iterators::null_at(3)
  };

  auto source_strings = cudf::test::strings_column_wrapper{
    {
      "nine",  "nine",  "nine", "nine",
      "eight", "", "eight"
    },
    cudf::test::iterators::null_at(5)
  };
  // clang-format on

  auto source_structs = cudf::test::structs_column_wrapper{{source_numerics, source_strings},
                                                           cudf::test::iterators::null_at(1)};

  auto source_list_null_mask_begin = cudf::test::iterators::null_at(2);

  auto [null_mask, null_count] = cudf::test::detail::make_null_mask(
    source_list_null_mask_begin, source_list_null_mask_begin + 3);
  auto source_lists = cudf::make_lists_column(3,
                                              offsets_column{0, 4, 7, 7}.release(),
                                              source_structs.release(),
                                              null_count,
                                              std::move(null_mask));

  // clang-format off
  auto target_ints    = numerics_column{
    0, 0,
    1, 1,
    2, 2,
    3, 3,
    4, 4,
    5, 5
  };
  auto target_strings = cudf::test::strings_column_wrapper{
    "zero",  "zero",
    "one",   "one",
    "two",   "two",
    "three", "three",
    "four",  "four",
    "five",  "five"
  };
  // clang-format on

  auto target_structs = cudf::test::structs_column_wrapper{{target_ints, target_strings}};

  auto target_lists = cudf::make_lists_column(
    6, offsets_column{0, 2, 4, 6, 8, 10, 12}.release(), target_structs.release(), 0, {});

  auto scatter_map = offsets_column{2, 0, 4};

  auto scatter_result = cudf::scatter(cudf::table_view({source_lists->view()}),
                                      scatter_map,
                                      cudf::table_view({target_lists->view()}));

  // clang-format off
  auto expected_numerics = numerics_column{
    {
      8, 8, 8,
      1, 1,
      9, 9, 9, 9,
      3, 3,
      5, 5
    },
    cudf::test::iterators::nulls_at({6, 8})
  };

  auto expected_strings = cudf::test::strings_column_wrapper{
    {
      "eight", "", "eight",
      "one",   "one",
      "nine",  "",  "nine", "nine",
      "three", "three",
      "five",  "five"
    },
    cudf::test::iterators::nulls_at({1, 6})
  };
  // clang-format on

  auto expected_structs = cudf::test::structs_column_wrapper{{expected_numerics, expected_strings},
                                                             cudf::test::iterators::null_at(6)};

  auto expected_lists_null_mask_begin = cudf::test::iterators::null_at(4);

  std::tie(null_mask, null_count) = cudf::test::detail::make_null_mask(
    expected_lists_null_mask_begin, expected_lists_null_mask_begin + 6);
  auto expected_lists = cudf::make_lists_column(6,
                                                offsets_column{0, 3, 5, 9, 11, 11, 13}.release(),
                                                expected_structs.release(),
                                                null_count,
                                                std::move(null_mask));

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected_lists->view(), scatter_result->get_column(0));
}
