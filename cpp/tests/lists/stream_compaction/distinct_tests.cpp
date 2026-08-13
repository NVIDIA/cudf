/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/iterator_utilities.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/column/column_factories.hpp>
#include <cudf/lists/sorting.hpp>
#include <cudf/lists/stream_compaction.hpp>

using float_type = double;
using namespace cudf::test::iterators;

auto constexpr null{0};  // null at current level
auto constexpr XXX{0};   // null pushed down from parent level
auto constexpr neg_NaN      = -std::numeric_limits<float_type>::quiet_NaN();
auto constexpr neg_Inf      = -std::numeric_limits<float_type>::infinity();
auto constexpr NaN          = std::numeric_limits<float_type>::quiet_NaN();
auto constexpr Inf          = std::numeric_limits<float_type>::infinity();
auto constexpr NULL_EQUAL   = cudf::null_equality::EQUAL;
auto constexpr NULL_UNEQUAL = cudf::null_equality::UNEQUAL;
auto constexpr NAN_EQUAL    = cudf::nan_equality::ALL_EQUAL;
auto constexpr NAN_UNEQUAL  = cudf::nan_equality::UNEQUAL;

using int32s_col    = cudf::test::fixed_width_column_wrapper<int32_t>;
using floats_lists  = cudf::test::lists_column_wrapper<float_type>;
using strings_lists = cudf::test::lists_column_wrapper<cudf::string_view>;
using strings_col   = cudf::test::strings_column_wrapper;
using structs_col   = cudf::test::structs_column_wrapper;
using lists_cv      = cudf::lists_column_view;

namespace {

auto distinct_sorted(cudf::column_view const& input,
                     cudf::null_equality nulls_equal = NULL_EQUAL,
                     cudf::nan_equality nans_equal   = NAN_EQUAL)
{
  auto const results = cudf::lists::distinct(lists_cv{input}, nulls_equal, nans_equal);

  // The sorted result will have nulls first and NaNs last.
  // In addition, row equality comparisons in tests just ignore NaN sign thus the expected values
  // can be just NaN while the input can be mixed of NaN and neg_NaN.
  return cudf::lists::sort_lists(
    lists_cv{*results}, cudf::order::ASCENDING, cudf::null_order::BEFORE);
}

}  // namespace

struct ListDistinctTest : public cudf::test::BaseFixture {};

template <typename T>
struct ListDistinctTypedTest : public cudf::test::BaseFixture {};

using TestTypes =
  cudf::test::Concat<cudf::test::IntegralTypesNotBool, cudf::test::FloatingPointTypes>;

TYPED_TEST_SUITE(ListDistinctTypedTest, TestTypes);

TEST_F(ListDistinctTest, TrivialTest)
{
  auto const input =
    floats_lists{{floats_lists{{NaN, 5.0, 0.0, 0.0, 0.0, 0.0, null, 0.0}, null_at(6)},
                  floats_lists{{NaN, 5.0, 0.0, 0.0, 0.0, 0.0, null, 1.0}, null_at(6)},
                  {} /*NULL*/,
                  floats_lists{{NaN, 5.0, 0.0, 0.0, 0.0, 0.0, null, 1.0}, null_at(6)}},
                 null_at(2)};

  auto const expected = floats_lists{{floats_lists{{null, 0.0, 5.0, NaN}, null_at(0)},
                                      floats_lists{{null, 0.0, 1.0, 5.0, NaN}, null_at(0)},
                                      floats_lists{} /*NULL*/,
                                      floats_lists{{null, 0.0, 1.0, 5.0, NaN}, null_at(0)}},
                                     null_at(2)};

  auto const results_sorted = distinct_sorted(input);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *results_sorted);
}

TEST_F(ListDistinctTest, FloatingPointTestsWithSignedZero)
{
  // -0.0 and 0.0 should be considered equal.
  auto const input    = floats_lists{0.0, 1, 2, -0.0, 1, 2, 0.0, 1, 2, -0.0, -0.0, 0.0, 0.0, 3};
  auto const expected = floats_lists{0, 1, 2, 3};

  auto const results_sorted = distinct_sorted(input);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *results_sorted);
}

TEST_F(ListDistinctTest, FloatingPointTestsWithInf)
{
  auto const input    = floats_lists{Inf, 0, neg_Inf, 0, Inf, 0, neg_Inf, 0, Inf, 0, neg_Inf};
  auto const expected = floats_lists{neg_Inf, 0, Inf};

  auto const results_sorted = distinct_sorted(input);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *results_sorted);
}

TEST_F(ListDistinctTest, FloatingPointTestsWithNaNs)
{
  auto const input =
    floats_lists{0, -1, 1, NaN, 2, 0, neg_NaN, 1, -2, 2, 0, 1, 2, neg_NaN, NaN, NaN, NaN, neg_NaN};

  // NaNs are equal.
  {
    auto const expected = floats_lists{-2, -1, 0, 1, 2, NaN};

    auto const results_sorted = distinct_sorted(input);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *results_sorted);
  }

  // NaNs are unequal.
  {
    auto const expected = floats_lists{-2, -1, 0, 1, 2, NaN, NaN, NaN, NaN, NaN, NaN, NaN};

    auto const results_sorted = distinct_sorted(input, NULL_EQUAL, NAN_UNEQUAL);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *results_sorted);
  }
}

TEST_F(ListDistinctTest, StringTestsNonNull)
{
  // Trivial cases - empty input.
  {
    auto const input    = strings_lists{{}};
    auto const expected = strings_lists{{}};

    auto const results_sorted = distinct_sorted(input);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *results_sorted);
  }

  // No duplicate.
  {
    auto const input    = strings_lists{"this", "is", "a", "string"};
    auto const expected = strings_lists{"a", "is", "string", "this"};

    auto const results_sorted = distinct_sorted(input);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *results_sorted);
  }

  // One list column.
  {
    auto const input    = strings_lists{"this", "is", "is", "is", "a", "string", "string"};
    auto const expected = strings_lists{"a", "is", "string", "this"};

    auto const results_sorted = distinct_sorted(input);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *results_sorted);
  }

  // Multiple lists column.
  {
    auto const input = strings_lists{
      strings_lists{"this", "is", "a", "no duplicate", "string"},
      strings_lists{"this", "is", "is", "a", "one duplicate", "string"},
      strings_lists{"this", "is", "is", "is", "a", "two duplicates", "string"},
      strings_lists{"this", "is", "is", "is", "is", "a", "three duplicates", "string"}};
    auto const expected =
      strings_lists{strings_lists{"a", "is", "no duplicate", "string", "this"},
                    strings_lists{"a", "is", "one duplicate", "string", "this"},
                    strings_lists{"a", "is", "string", "this", "two duplicates"},
                    strings_lists{"a", "is", "string", "this", "three duplicates"}};

    auto const results_sorted = distinct_sorted(input);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *results_sorted);
  }
}

TEST_F(ListDistinctTest, StringTestsWithNullsEqual)
{
  auto const null = std::string("");

  // One list column with null entries.
  {
    auto const input = strings_lists{
      {"this", null, "is", "is", "is", "a", null, "string", null, "string"}, nulls_at({1, 6, 8})};
    auto const expected = strings_lists{{null, "a", "is", "string", "this"}, null_at(0)};

    auto const results_sorted = distinct_sorted(input);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *results_sorted);
  }

  // Multiple lists column with null lists and null entries.
  {
    auto const input = strings_lists{
      {strings_lists{{"this", null, "is", null, "a", null, "no duplicate", null, "string"},
                     nulls_at({1, 3, 5, 7})},
       strings_lists{}, /* NULL */
       strings_lists{"this", "is", "is", "a", "one duplicate", "string"}},
      null_at(1)};
    auto const expected =
      strings_lists{{strings_lists{{null, "a", "is", "no duplicate", "string", "this"}, null_at(0)},
                     strings_lists{}, /* NULL */
                     strings_lists{"a", "is", "one duplicate", "string", "this"}},
                    null_at(1)};

    auto const results_sorted = distinct_sorted(input);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *results_sorted);
  }
}

TEST_F(ListDistinctTest, StringTestsWithNullsUnequal)
{
  auto const null = std::string("");

  // One list column with null entries.
  {
    auto const input = strings_lists{
      {"this", null, "is", "is", "is", "a", null, "string", null, "string"}, nulls_at({1, 6, 8})};
    auto const expected =
      strings_lists{{null, null, null, "a", "is", "string", "this"}, nulls_at({0, 1, 2})};

    auto const results_sorted = distinct_sorted(input, NULL_UNEQUAL);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *results_sorted);
  }

  // Multiple lists column with null lists and null entries.
  {
    auto const input = strings_lists{
      {strings_lists{{"this", null, "is", null, "a", null, "no duplicate", null, "string"},
                     nulls_at({1, 3, 5, 7})},
       strings_lists{}, /* NULL */
       strings_lists{"this", "is", "is", "a", "one duplicate", "string"}},
      null_at(1)};
    auto const expected = strings_lists{
      {strings_lists{{null, null, null, null, "a", "is", "no duplicate", "string", "this"},
                     nulls_at({0, 1, 2, 3})},
       strings_lists{}, /* NULL */
       strings_lists{"a", "is", "one duplicate", "string", "this"}},
      null_at(1)};

    auto const results_sorted = distinct_sorted(input, NULL_UNEQUAL);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *results_sorted);
  }
}

TYPED_TEST(ListDistinctTypedTest, TrivialInputTests)
{
  using lists_col = cudf::test::lists_column_wrapper<TypeParam>;

  // Empty input.
  {
    auto const input    = lists_col{};
    auto const expected = lists_col{};

    auto const results_sorted = distinct_sorted(input);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *results_sorted);
  }

  // All input lists are empty.
  {
    auto const input    = lists_col{lists_col{}, lists_col{}, lists_col{}};
    auto const expected = lists_col{lists_col{}, lists_col{}, lists_col{}};

    auto const results_sorted = distinct_sorted(input);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *results_sorted);
  }

  // Trivial cases.
  {
    auto const input    = lists_col{0, 1, 2, 3, 4, 5};
    auto const expected = lists_col{0, 1, 2, 3, 4, 5};

    auto const results_sorted = distinct_sorted(input);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *results_sorted);
  }

  // Multiple empty lists.
  {
    auto const input    = lists_col{{}, {}, {5, 4, 3, 2, 1, 0}, {}, {6}, {}};
    auto const expected = lists_col{{}, {}, {0, 1, 2, 3, 4, 5}, {}, {6}, {}};

    auto const results_sorted = distinct_sorted(input);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *results_sorted);
  }
}

TYPED_TEST(ListDistinctTypedTest, SlicedNonNullInputTests)
{
  using lists_col = cudf::test::lists_column_wrapper<TypeParam>;

  auto const input_original =
    lists_col{{1, 2, 3, 2, 3, 2, 3, 2, 3}, {3, 2, 1, 4, 1}, {5}, {10, 8, 9}, {6, 7}};

  {
    auto const expected = lists_col{{1, 2, 3}, {1, 2, 3, 4}, {5}, {8, 9, 10}, {6, 7}};

    auto const results_sorted = distinct_sorted(input_original);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *results_sorted);
  }

  {
    auto const input    = cudf::slice(input_original, {0, 5})[0];
    auto const expected = lists_col{{1, 2, 3}, {1, 2, 3, 4}, {5}, {8, 9, 10}, {6, 7}};

    auto const results_sorted = distinct_sorted(input);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *results_sorted);
  }

  {
    auto const input    = cudf::slice(input_original, {1, 5})[0];
    auto const expected = lists_col{{1, 2, 3, 4}, {5}, {8, 9, 10}, {6, 7}};

    auto const results_sorted = distinct_sorted(input);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *results_sorted);
  }

  {
    auto const input    = cudf::slice(input_original, {1, 3})[0];
    auto const expected = lists_col{{1, 2, 3, 4}, {5}};

    auto const results_sorted = distinct_sorted(input);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *results_sorted);
  }

  {
    auto const input    = cudf::slice(input_original, {0, 3})[0];
    auto const expected = lists_col{{1, 2, 3}, {1, 2, 3, 4}, {5}};

    auto const results_sorted = distinct_sorted(input);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *results_sorted);
  }
}

TYPED_TEST(ListDistinctTypedTest, InputHaveNullsTests)
{
  using lists_col     = cudf::test::lists_column_wrapper<TypeParam>;
  auto constexpr null = TypeParam{0};

  // Nullable lists.
  {
    auto const input = lists_col{
      {{3, 2, 1, 4, 1}, {5}, {} /*NULL*/, {} /*NULL*/, {10, 8, 9}, {6, 7}}, nulls_at({2, 3})};
    auto const expected = lists_col{
      {{1, 2, 3, 4}, {5}, {} /*NULL*/, {} /*NULL*/, {8, 9, 10}, {6, 7}}, nulls_at({2, 3})};

    auto const results_sorted = distinct_sorted(input);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *results_sorted);
  }

  // Nullable child and nulls are equal.
  {
    auto const input =
      lists_col{{null, 1, null, 3, null, 5, null, 7, null, 9}, nulls_at({0, 2, 4, 6, 8})};
    auto const expected = lists_col{{null, 1, 3, 5, 7, 9}, null_at(0)};

    auto const results_sorted = distinct_sorted(input, NULL_EQUAL);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *results_sorted);
  }

  // Nullable child and nulls are unequal.
  {
    auto const input = lists_col{{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}, nulls_at({0, 2, 4, 6, 8})};
    auto const expected =
      lists_col{{null, null, null, null, null, 1, 3, 5, 7, 9}, nulls_at({0, 1, 2, 3, 4})};

    auto const results_sorted = distinct_sorted(input, NULL_UNEQUAL);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *results_sorted);
  }
}

TEST_F(ListDistinctTest, InputListsOfStructsNoNull)
{
  auto const get_structs = [] {
    auto child1 = int32s_col{
      1, 1, 1, 1, 1, 1, 1, 1,  // list1
      1, 1, 1, 1, 2, 1, 2, 2,  // list2
      2, 2, 2, 2, 3, 2, 3, 3   // list3
    };
    auto child2 = strings_col{
      // begin list1
      "Banana",
      "Mango",
      "Apple",
      "Cherry",
      "Kiwi",
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
      "Cat",
      "Panda",  // end list2
      // begin list3
      "ÁÁÁ",
      "ÉÉÉÉÉ",
      "ÍÍÍÍÍ",
      "ÁBC",
      "XYZ",
      "ÁÁÁ",
      "ÁBC",
      "XYZ"  // end list3
    };
    return structs_col{{child1, child2}};
  };

  auto const get_expected = [] {
    auto child1 = int32s_col{1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3};
    auto child2 = strings_col{
      // begin list1
      "Apple",
      "Banana",
      "Cherry",
      "Kiwi",
      "Mango",  // end list1
      // begin list2
      "Bear",
      "Cat",
      "Dog",
      "Duck",
      "Cat",
      "Panda",  // end list2
      // begin list3
      "ÁBC",
      "ÁÁÁ",
      "ÉÉÉÉÉ",
      "ÍÍÍÍÍ",
      "XYZ",
      "ÁBC"  // end list3
    };
    return structs_col{{child1, child2}};
  };

  // Test full columns.
  {
    auto const input = cudf::make_lists_column(
      3, int32s_col{0, 8, 16, 24}.release(), get_structs().release(), 0, {});
    auto const expected = cudf::make_lists_column(
      3, int32s_col{0, 5, 11, 17}.release(), get_expected().release(), 0, {});

    auto const results_sorted = distinct_sorted(*input);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*expected, *results_sorted);
  }

  // Test sliced columns.
  {
    auto const input_original = cudf::make_lists_column(
      3, int32s_col{0, 8, 16, 24}.release(), get_structs().release(), 0, {});
    auto const expected_original = cudf::make_lists_column(
      3, int32s_col{0, 5, 11, 17}.release(), get_expected().release(), 0, {});
    auto const input    = cudf::slice(*input_original, {1, 3})[0];
    auto const expected = cudf::slice(*expected_original, {1, 3})[0];

    auto const results_sorted = distinct_sorted(input);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *results_sorted);
  }
}

TEST_F(ListDistinctTest, InputListsOfStructsHaveNull)
{
  auto const get_structs = [] {
    auto child1 = int32s_col{{
                               1,    1,    null, XXX, XXX, 1, 1,    1,  // list1
                               1,    1,    1,    1,   2,   1, null, 2,  // list2
                               null, null, 2,    2,   3,   2, 3,    3   // list3
                             },
                             nulls_at({2, 14, 16, 17})};
    auto child2 = strings_col{{
                                // begin list1
                                "Banana",
                                "Mango",
                                "Apple",
                                "XXX", /*NULL*/
                                "XXX", /*NULL*/
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
    return structs_col{{child1, child2}, nulls_at({3, 4})};
  };

  auto const get_expected = [] {
    auto child1 = int32s_col{{      // begin list1
                              XXX,  // end list1
                              null,
                              1,
                              1,
                              1,
                              1,
                              // begin list2
                              null,  // end list2
                              1,
                              1,
                              1,
                              1,
                              2,
                              // begin list3
                              null,
                              null,
                              2,
                              2,
                              2,
                              3,
                              3,
                              3},  // end list3
                             nulls_at({1, 6, 12, 13})};
    auto child2 = strings_col{{       // begin list1
                               "XXX", /*NULL*/
                               "Apple",
                               "Banana",
                               "Cherry",
                               "Kiwi",
                               "Mango",  // end list1
                                         // begin list2
                               "",       /*NULL*/
                               "Bear",
                               "Cat",
                               "Dog",
                               "Duck",
                               "Panda",  // end list2
                                         // begin list3
                               "ÁÁÁ",
                               "ÉÉÉÉÉ",
                               "ÁBC",
                               "ÁÁÁ",
                               "ÍÍÍÍÍ",
                               "", /*NULL*/
                               "XYZ",
                               "ÁBC"},  // end list3
                              nulls_at({6, 17})};
    return structs_col{{child1, child2}, null_at(0)};
  };

  // Test full columns.
  {
    auto const input = cudf::make_lists_column(
      3, int32s_col{0, 8, 16, 24}.release(), get_structs().release(), 0, {});
    auto const expected = cudf::make_lists_column(
      3, int32s_col{0, 6, 12, 20}.release(), get_expected().release(), 0, {});

    auto const results_sorted = distinct_sorted(*input);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*expected, *results_sorted);
  }

  // Test sliced columns.
  {
    auto const input_original = cudf::make_lists_column(
      3, int32s_col{0, 8, 16, 24}.release(), get_structs().release(), 0, {});
    auto const input = cudf::slice(*input_original, {1, 3})[0];

    // The sliced range holds no struct-level nulls, so the result's struct child is non-nullable;
    // build the expected column directly (slicing `get_expected()` would keep its allocated mask).
    auto const expected = [] {
      auto child1 =
        int32s_col{{null, 1, 1, 1, 1, 2, null, null, 2, 2, 2, 3, 3, 3}, nulls_at({0, 6, 7})};
      auto child2  = strings_col{{"", /*NULL*/
                                  "Bear",
                                  "Cat",
                                  "Dog",
                                  "Duck",
                                  "Panda",
                                  "ÁÁÁ",
                                  "ÉÉÉÉÉ",
                                  "ÁBC",
                                  "ÁÁÁ",
                                  "ÍÍÍÍÍ",
                                  "", /*NULL*/
                                  "XYZ",
                                  "ÁBC"},
                                nulls_at({0, 11})};
      auto structs = structs_col{{child1, child2}};
      return cudf::make_lists_column(2, int32s_col{0, 6, 14}.release(), structs.release(), 0, {});
    }();

    auto const results_sorted = distinct_sorted(input);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*expected, *results_sorted);
  }
}

TEST_F(ListDistinctTest, InputListsOfNestedStructsHaveNull)
{
  auto const get_structs = [] {
    auto grandchild1 = int32s_col{{
                                    1,    XXX,  null, XXX, XXX, 1, 1,    1,  // list1
                                    1,    1,    1,    1,   2,   1, null, 2,  // list2
                                    null, null, 2,    2,   3,   2, 3,    3   // list3
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
    auto child1      = structs_col{{grandchild1, grandchild2}, nulls_at({1, 3, 4})};
    return structs_col{{child1}};
  };

  auto const get_expected = [] {
    auto grandchild1 = int32s_col{{// begin list1
                                   XXX,
                                   null,
                                   1,
                                   1,
                                   1,  // end list1
                                       // begin list2
                                   null,
                                   1,
                                   1,
                                   1,
                                   1,
                                   2,  // end list2
                                       // begin list3
                                   null,
                                   null,
                                   2,
                                   2,
                                   2,
                                   3,
                                   3,
                                   3},
                                  nulls_at({1, 5, 11, 12})};
    auto grandchild2 = strings_col{{
                                     // begin list1
                                     "XXX" /*NULL*/,
                                     "Apple",
                                     "Banana",
                                     "Cherry",
                                     "Kiwi",  // end list1
                                              // begin list2
                                     "" /*NULL*/,
                                     "Bear",
                                     "Cat",
                                     "Dog",
                                     "Duck",
                                     "Panda",  // end list2
                                               // begin list3
                                     "ÁÁÁ",
                                     "ÉÉÉÉÉ",
                                     "ÁBC",
                                     "ÁÁÁ",
                                     "ÍÍÍÍÍ",
                                     "", /*NULL*/
                                     "XYZ",
                                     "ÁBC"  // end list3
                                   },
                                   nulls_at({5, 16})};
    auto child1      = structs_col{{grandchild1, grandchild2}, nulls_at({0})};
    return structs_col{{child1}};
  };

  // Test full columns.
  {
    auto const input = cudf::make_lists_column(
      3, int32s_col{0, 8, 16, 24}.release(), get_structs().release(), 0, {});
    auto const expected = cudf::make_lists_column(
      3, int32s_col{0, 5, 11, 19}.release(), get_expected().release(), 0, {});

    auto const results_sorted = distinct_sorted(*input);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*expected, *results_sorted);
  }

  // Test sliced columns.
  {
    auto const input_original = cudf::make_lists_column(
      3, int32s_col{0, 8, 16, 24}.release(), get_structs().release(), 0, {});
    auto const expected_original = cudf::make_lists_column(
      3, int32s_col{0, 5, 11, 19}.release(), get_expected().release(), 0, {});
    auto const input    = cudf::slice(*input_original, {1, 3})[0];
    auto const expected = cudf::slice(*expected_original, {1, 3})[0];

    auto const results_sorted = distinct_sorted(input);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *results_sorted);
  }
}

TEST_F(ListDistinctTest, InputListsOfStructsOfLists)
{
  auto const input = [] {
    auto const get_structs = [] {
      auto child1 = int32s_col{// begin list1
                               0,
                               0,
                               0,  // end list1
                                   // begin list2
                               1,  // end list2
                                   // begin list3
                               2,
                               2,  // end list3
                                   // begin list4
                               3,
                               3,
                               3};
      auto child2 = floats_lists{// begin list1
                                 floats_lists{0, 1},
                                 floats_lists{0, 1},
                                 floats_lists{0, 1},     // end list1
                                                         // begin list2
                                 floats_lists{3, 4, 5},  // end list2
                                                         // begin list3
                                 floats_lists{},
                                 floats_lists{},  // end list3
                                                  // begin list4
                                 floats_lists{6, 7},
                                 floats_lists{6, 7},
                                 floats_lists{6, 7}};
      return structs_col{{child1, child2}};
    };

    return cudf::make_lists_column(
      4, int32s_col{0, 3, 4, 6, 9}.release(), get_structs().release(), 0, {});
  }();

  auto const expected = [] {
    auto const get_structs = [] {
      auto child1 = int32s_col{0, 1, 2, 3};
      auto child2 =
        floats_lists{floats_lists{0, 1}, floats_lists{3, 4, 5}, floats_lists{}, floats_lists{6, 7}};
      return structs_col{{child1, child2}};
    };

    return cudf::make_lists_column(
      4, int32s_col{0, 1, 2, 3, 4}.release(), get_structs().release(), 0, {});
  }();

  auto const results = cudf::lists::distinct(lists_cv{*input});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*expected, *results);
}
