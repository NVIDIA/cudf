/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/iterator_utilities.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/lists/reverse.hpp>
#include <cudf/null_mask.hpp>

using namespace cudf::test::iterators;

auto constexpr null{0};  // null at current level
auto constexpr XXX{0};   // null pushed down from parent level

using ints_lists  = cudf::test::lists_column_wrapper<int32_t>;
using ints_col    = cudf::test::fixed_width_column_wrapper<int32_t>;
using strings_col = cudf::test::strings_column_wrapper;
using structs_col = cudf::test::structs_column_wrapper;

struct ListsReverseTest : public cudf::test::BaseFixture {};

template <typename T>
struct ListsReverseTypedTest : public cudf::test::BaseFixture {};

using TestTypes =
  cudf::test::Concat<cudf::test::IntegralTypesNotBool, cudf::test::FloatingPointTypes>;

TYPED_TEST_SUITE(ListsReverseTypedTest, TestTypes);

TEST_F(ListsReverseTest, EmptyInput)
{
  // Empty column.
  {
    auto const input   = ints_lists{};
    auto const results = cudf::lists::reverse(cudf::lists_column_view(input));
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(input, *results);
  }

  // Empty lists.
  {
    auto const input   = ints_lists{ints_lists{}, ints_lists{}, ints_lists{}};
    auto const results = cudf::lists::reverse(cudf::lists_column_view(input));
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(input, *results);
  }

  // Empty nested lists.
  {
    auto const input   = ints_lists{ints_lists{ints_lists{}}, ints_lists{}, ints_lists{}};
    auto const results = cudf::lists::reverse(cudf::lists_column_view(input));
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(input, *results);
  }
}

TYPED_TEST(ListsReverseTypedTest, SimpleInputNoNulls)
{
  using lists_col           = cudf::test::lists_column_wrapper<TypeParam>;
  auto const input_original = lists_col{{}, {1, 2, 3}, {}, {4, 5}, {6, 7, 8}, {9}};

  {
    auto const expected = lists_col{{}, {3, 2, 1}, {}, {5, 4}, {8, 7, 6}, {9}};
    auto const results  = cudf::lists::reverse(cudf::lists_column_view(input_original));
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *results);
  }

  {
    auto const expected = lists_col{{3, 2, 1}, {}, {5, 4}};
    auto const input    = cudf::slice(input_original, {1, 4})[0];
    auto const results  = cudf::lists::reverse(cudf::lists_column_view(input));
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *results);
  }

  {
    auto const expected = lists_col{lists_col{}};
    auto const input    = cudf::slice(input_original, {2, 3})[0];
    auto const results  = cudf::lists::reverse(cudf::lists_column_view(input));
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *results);
  }

  {
    auto const expected = lists_col{{}, {5, 4}};
    auto const input    = cudf::slice(input_original, {2, 4})[0];
    auto const results  = cudf::lists::reverse(cudf::lists_column_view(input));
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *results);
  }
}

TYPED_TEST(ListsReverseTypedTest, SimpleInputWithNulls)
{
  using lists_col           = cudf::test::lists_column_wrapper<TypeParam>;
  auto const input_original = lists_col{{lists_col{},
                                         lists_col{1, 2, 3},
                                         lists_col{} /*null*/,
                                         lists_col{{4, 5, null}, null_at(2)},
                                         lists_col{6, 7, 8},
                                         lists_col{9}},
                                        null_at(2)};

  {
    auto const expected = lists_col{{lists_col{},
                                     lists_col{3, 2, 1},
                                     lists_col{} /*null*/,
                                     lists_col{{null, 5, 4}, null_at(0)},
                                     lists_col{8, 7, 6},
                                     lists_col{9}},
                                    null_at(2)};
    auto const results  = cudf::lists::reverse(cudf::lists_column_view(input_original));
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *results);
  }

  {
    auto const expected = lists_col{
      {lists_col{3, 2, 1}, lists_col{} /*null*/, lists_col{{null, 5, 4}, null_at(0)}}, null_at(1)};
    auto const input   = cudf::slice(input_original, {1, 4})[0];
    auto const results = cudf::lists::reverse(cudf::lists_column_view(input));
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *results);
  }

  {
    auto const expected = lists_col{{lists_col{} /*null*/}, null_at(0)};
    auto const input    = cudf::slice(input_original, {2, 3})[0];
    auto const results  = cudf::lists::reverse(cudf::lists_column_view(input));
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *results);
  }

  {
    auto const expected =
      lists_col{{lists_col{} /*null*/, lists_col{{null, 5, 4}, null_at(0)}}, null_at(0)};
    auto const input   = cudf::slice(input_original, {2, 4})[0];
    auto const results = cudf::lists::reverse(cudf::lists_column_view(input));
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *results);
  }

  {
    auto const input    = cudf::slice(input_original, {4, 6})[0];
    auto const expected = lists_col{{8, 7, 6}, {9}};
    auto const results  = cudf::lists::reverse(cudf::lists_column_view(input));
    // The result doesn't have nulls, but it is nullable.
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected, *results);
  }
}

TYPED_TEST(ListsReverseTypedTest, InputListsOfListsNoNulls)
{
  using lists_col           = cudf::test::lists_column_wrapper<TypeParam>;
  auto const input_original = [] {
    auto child =
      lists_col{{1, 2, 3}, {4, 5, 6}, {7}, {4, 5}, {}, {4, 5, 6}, {}, {6, 7, 8}, {}, {9}}.release();
    auto offsets = cudf::test::fixed_width_column_wrapper<cudf::size_type>{0, 0, 3, 3, 6, 9, 10, 10, 10}.release();
    return cudf::make_lists_column(8, std::move(offsets), std::move(child), 0, {});
  }();

  {
    auto const expected = [] {
      auto child =
        lists_col{{7}, {4, 5, 6}, {1, 2, 3}, {4, 5, 6}, {}, {4, 5}, {}, {6, 7, 8}, {}, {9}}
          .release();
      auto offsets = cudf::test::fixed_width_column_wrapper<cudf::size_type>{0, 0, 3, 3, 6, 9, 10, 10, 10}.release();
      return cudf::make_lists_column(8, std::move(offsets), std::move(child), 0, {});
    }();
    auto const results = cudf::lists::reverse(cudf::lists_column_view(*input_original));
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*expected, *results);
  }

  {
    auto const expected = [] {
      auto child   = lists_col{{7}, {4, 5, 6}, {1, 2, 3}, {4, 5, 6}, {}, {4, 5}}.release();
      auto offsets = cudf::test::fixed_width_column_wrapper<cudf::size_type>{0, 3, 3, 6}.release();
      return cudf::make_lists_column(3, std::move(offsets), std::move(child), 0, {});
    }();
    auto const input   = cudf::slice(*input_original, {1, 4})[0];
    auto const results = cudf::lists::reverse(cudf::lists_column_view(input));
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*expected, *results);
  }

  {
    auto const input   = cudf::slice(*input_original, {2, 3})[0];
    auto const results = cudf::lists::reverse(cudf::lists_column_view(input));
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(input, *results);
  }

  {
    auto const expected = [] {
      auto child   = lists_col{{4, 5, 6}, {}, {4, 5}}.release();
      auto offsets = cudf::test::fixed_width_column_wrapper<cudf::size_type>{0, 0, 3}.release();
      return cudf::make_lists_column(2, std::move(offsets), std::move(child), 0, {});
    }();
    auto const input   = cudf::slice(*input_original, {2, 4})[0];
    auto const results = cudf::lists::reverse(cudf::lists_column_view(input));
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*expected, *results);
  }
}

TYPED_TEST(ListsReverseTypedTest, InputListsOfListsWithNulls)
{
  using lists_col           = cudf::test::lists_column_wrapper<TypeParam>;
  auto const input_original = [] {
    auto child = lists_col{{{1, 2, 3},
                            {4, 5, 6},
                            {7},
                            {4, 5},
                            {} /*null*/,
                            {4, 5, 6},
                            {},
                            {6, 7, 8},
                            {} /*null*/,
                            {9}},
                           nulls_at({4, 8})}
                   .release();
    auto offsets   = cudf::test::fixed_width_column_wrapper<cudf::size_type>{0, 0, 3, 3, 6, 9, 10, 10, 10}.release();
    auto null_mask = cudf::create_null_mask(8, cudf::mask_state::ALL_VALID);
    cudf::set_null_mask(static_cast<cudf::bitmask_type*>(null_mask.data()), 2, 3, false);

    return cudf::make_lists_column(
      8, std::move(offsets), std::move(child), 1, std::move(null_mask));
  }();

  {
    auto const expected = [] {
      auto child = lists_col{{{7},
                              {4, 5, 6},
                              {1, 2, 3},
                              {4, 5, 6},
                              {} /*null*/,
                              {4, 5},
                              {} /*null*/,
                              {6, 7, 8},
                              {},
                              {9}},
                             nulls_at({4, 6})}
                     .release();
      auto offsets   = cudf::test::fixed_width_column_wrapper<cudf::size_type>{0, 0, 3, 3, 6, 9, 10, 10, 10}.release();
      auto null_mask = cudf::create_null_mask(8, cudf::mask_state::ALL_VALID);
      cudf::set_null_mask(static_cast<cudf::bitmask_type*>(null_mask.data()), 2, 3, false);

      return cudf::make_lists_column(
        8, std::move(offsets), std::move(child), 1, std::move(null_mask));
    }();
    auto const results = cudf::lists::reverse(cudf::lists_column_view(*input_original));
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*expected, *results);
  }

  {
    auto const expected = [] {
      auto child   = lists_col{{7}, {4, 5, 6}, {1, 2, 3}}.release();
      auto offsets = cudf::test::fixed_width_column_wrapper<cudf::size_type>{0, 3}.release();
      return cudf::make_lists_column(1, std::move(offsets), std::move(child), 0, {});
    }();
    auto const input   = cudf::slice(*input_original, {0, 1})[0];
    auto const results = cudf::lists::reverse(cudf::lists_column_view(input));
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(input, *results);
  }

  {
    auto const expected = [] {
      auto child =
        lists_col{{{7}, {4, 5, 6}, {1, 2, 3}, {4, 5, 6}, {} /*null*/, {4, 5}}, null_at(4)}
          .release();
      auto offsets   = cudf::test::fixed_width_column_wrapper<cudf::size_type>{0, 3, 3, 6}.release();
      auto null_mask = cudf::create_null_mask(3, cudf::mask_state::ALL_VALID);
      cudf::set_null_mask(static_cast<cudf::bitmask_type*>(null_mask.data()), 1, 2, false);

      return cudf::make_lists_column(
        3, std::move(offsets), std::move(child), 1, std::move(null_mask));
    }();
    auto const input   = cudf::slice(*input_original, {1, 4})[0];
    auto const results = cudf::lists::reverse(cudf::lists_column_view(input));
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*expected, *results);
  }

  {
    auto const expected = [] {
      auto child     = lists_col{{{4, 5, 6}, {} /*null*/, {4, 5}}, null_at(1)}.release();
      auto offsets   = cudf::test::fixed_width_column_wrapper<cudf::size_type>{0, 0, 3}.release();
      auto null_mask = cudf::create_null_mask(2, cudf::mask_state::ALL_VALID);
      cudf::set_null_mask(static_cast<cudf::bitmask_type*>(null_mask.data()), 0, 1, false);

      return cudf::make_lists_column(
        2, std::move(offsets), std::move(child), 1, std::move(null_mask));
    }();
    auto const input   = cudf::slice(*input_original, {2, 4})[0];
    auto const results = cudf::lists::reverse(cudf::lists_column_view(input));
    // The result doesn't have nulls, but it is nullable.
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*expected, *results);
  }
}

TYPED_TEST(ListsReverseTypedTest, InputListsOfStructsWithNulls)
{
  using data_col = cudf::test::fixed_width_column_wrapper<TypeParam>;

  auto const input_original = [] {
    auto child = [] {
      auto grandchild1 = data_col{{
                                    1,    XXX,  null, XXX, XXX, 2,  3,    4,   // list1
                                    5,    6,    7,    8,   9,   10, null, 11,  // list2
                                    null, null, 12,   13,  14,  15, 16,   17   // list3
                                  },
                                  nulls_at({2, 14, 16, 17})};
      auto grandchild2 = strings_col{{
                                       // begin list1
                                       "Banana",
                                       "YYY", /*NULL*/
                                       "Apple",
                                       "XXX", /*NULL*/
                                       "YYY", /*NULL*/
                                       "Banana",
                                       "Cherry",
                                       "Kiwi",  // end list1
                                                // begin list2
                                       "Bear",
                                       "Duck",
                                       "Cat",
                                       "Dog",
                                       "Panda",
                                       "Bear",
                                       "" /*NULL*/,
                                       "Panda",  // end list2
                                                 // begin list3
                                       "ÁÁÁ",
                                       "ÉÉÉÉÉ",
                                       "ÍÍÍÍÍ",
                                       "ÁBC",
                                       "" /*NULL*/,
                                       "ÁÁÁ",
                                       "ÁBC",
                                       "XYZ"  // end list3
                                     },
                                     nulls_at({14, 20})};
      return structs_col{{grandchild1, grandchild2}, nulls_at({1, 3, 4})}.release();
    }();
    auto offsets   = cudf::test::fixed_width_column_wrapper<cudf::size_type>{0, 0, 8, 16, 16, 16, 24}.release();
    auto null_mask = cudf::create_null_mask(6, cudf::mask_state::ALL_VALID);
    cudf::set_null_mask(static_cast<cudf::bitmask_type*>(null_mask.data()), 0, 1, false);
    cudf::set_null_mask(static_cast<cudf::bitmask_type*>(null_mask.data()), 4, 5, false);

    return cudf::make_lists_column(
      6, std::move(offsets), std::move(child), 2, std::move(null_mask));
  }();

  {
    auto const expected = [] {
      auto child = [] {
        auto grandchild1 = data_col{{
                                      4,  3,    2,  null, null, null, null, 1,     // list1
                                      11, null, 10, 9,    8,    7,    6,    5,     // list2
                                      17, 16,   15, 14,   13,   12,   null, null,  // list3
                                    },
                                    nulls_at({3, 4, 5, 6, 9, 22, 23})};
        auto grandchild2 = strings_col{{
                                         // begin list1
                                         "Kiwi",
                                         "Cherry",
                                         "Banana",
                                         "", /*NULL*/
                                         "", /*NULL*/
                                         "Apple",
                                         "",        /*NULL*/
                                         "Banana",  // end list1
                                                    // begin list2
                                         "Panda",
                                         "" /*NULL*/,
                                         "Bear",
                                         "Panda",
                                         "Dog",
                                         "Cat",
                                         "Duck",
                                         "Bear",  // end list2
                                                  // begin list3
                                         "XYZ",
                                         "ÁBC",
                                         "ÁÁÁ",
                                         "" /*NULL*/,
                                         "ÁBC",
                                         "ÍÍÍÍÍ",
                                         "ÉÉÉÉÉ",
                                         "ÁÁÁ"  // end list3
                                       },
                                       nulls_at({3, 4, 6, 9, 19})};
        return structs_col{{grandchild1, grandchild2}, nulls_at({3, 4, 6})}.release();
      }();
      auto offsets   = cudf::test::fixed_width_column_wrapper<cudf::size_type>{0, 0, 8, 16, 16, 16, 24}.release();
      auto null_mask = cudf::create_null_mask(6, cudf::mask_state::ALL_VALID);
      cudf::set_null_mask(static_cast<cudf::bitmask_type*>(null_mask.data()), 0, 1, false);
      cudf::set_null_mask(static_cast<cudf::bitmask_type*>(null_mask.data()), 4, 5, false);

      return cudf::make_lists_column(
        6, std::move(offsets), std::move(child), 2, std::move(null_mask));
    }();
    auto const results = cudf::lists::reverse(cudf::lists_column_view(*input_original));
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*expected, *results);
  }

  {
    auto const expected = [] {
      auto child = [] {
        auto grandchild1 = data_col{{
                                      4,
                                      3,
                                      2,
                                      null,
                                      null,
                                      null,
                                      null,
                                      1,  // end list1
                                      11,
                                      null,
                                      10,
                                      9,
                                      8,
                                      7,
                                      6,
                                      5  // end list2
                                    },
                                    nulls_at({3, 4, 5, 6, 9})};
        auto grandchild2 = strings_col{{
                                         // begin list1
                                         "Kiwi",
                                         "Cherry",
                                         "Banana",
                                         "", /*NULL*/
                                         "", /*NULL*/
                                         "Apple",
                                         "",        /*NULL*/
                                         "Banana",  // end list1
                                                    // begin list2
                                         "Panda",
                                         "" /*NULL*/,
                                         "Bear",
                                         "Panda",
                                         "Dog",
                                         "Cat",
                                         "Duck",
                                         "Bear"  // end list2
                                       },
                                       nulls_at({3, 4, 6, 9})};
        return structs_col{{grandchild1, grandchild2}, nulls_at({3, 4, 6})}.release();
      }();

      auto offsets = cudf::test::fixed_width_column_wrapper<cudf::size_type>{0, 8, 16, 16}.release();
      return cudf::make_lists_column(3, std::move(offsets), std::move(child), 0, {});
    }();
    auto const input   = cudf::slice(*input_original, {1, 4})[0];
    auto const results = cudf::lists::reverse(cudf::lists_column_view(input));
    // The result doesn't have nulls, but it is nullable.
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*expected, *results);
  }

  {
    auto const input   = cudf::slice(*input_original, {4, 5})[0];
    auto const results = cudf::lists::reverse(cudf::lists_column_view(input));
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(input, *results);
  }
}
