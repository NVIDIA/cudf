/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/iterator_utilities.hpp>
#include <cudf_test/lists_column_wrapper.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/concatenate.hpp>
#include <cudf/copying.hpp>
#include <cudf/groupby.hpp>
#include <cudf/table/table_view.hpp>

using namespace cudf::test::iterators;

namespace {
constexpr cudf::test::debug_output_level verbosity{cudf::test::debug_output_level::FIRST_ERROR};
constexpr int32_t null{0};  // Mark for null elements

using vcol_views = std::vector<cudf::column_view>;

auto merge_lists(vcol_views const& keys_cols, vcol_views const& values_cols)
{
  // Append all the keys and lists together.
  auto const keys   = cudf::concatenate(keys_cols);
  auto const values = cudf::concatenate(values_cols);

  std::vector<cudf::groupby::aggregation_request> requests;
  requests.emplace_back();
  requests[0].values = *values;
  requests[0].aggregations.emplace_back(
    cudf::make_merge_lists_aggregation<cudf::groupby_aggregation>());

  auto gb_obj = cudf::groupby::groupby(cudf::table_view({*keys}));
  auto result = gb_obj.aggregate(requests);
  return std::pair(std::move(result.first->release()[0]), std::move(result.second[0].results[0]));
}

}  // namespace

template <typename V>
struct GroupbyMergeListsTypedTest : public cudf::test::BaseFixture {};

using FixedWidthTypesNotBool = cudf::test::Concat<cudf::test::IntegralTypesNotBool,
                                                  cudf::test::FloatingPointTypes,
                                                  cudf::test::TimestampTypes>;
TYPED_TEST_SUITE(GroupbyMergeListsTypedTest, FixedWidthTypesNotBool);

TYPED_TEST(GroupbyMergeListsTypedTest, InvalidInput)
{
  using keys_col  = cudf::test::fixed_width_column_wrapper<TypeParam, int32_t>;
  using lists_col = cudf::test::lists_column_wrapper<TypeParam, int32_t>;

  auto const keys = keys_col{1, 2, 3};

  // The input lists column must NOT be nullable.
  auto const lists = lists_col{{lists_col{1}, lists_col{} /*NULL*/, lists_col{2}}, null_at(1)};
  EXPECT_THROW(merge_lists({keys}, {lists}), cudf::logic_error);

  // The input column must be a lists column.
  auto const non_lists = keys_col{1, 2, 3, 4, 5};
  EXPECT_THROW(merge_lists({keys}, {non_lists}), cudf::logic_error);
}

TYPED_TEST(GroupbyMergeListsTypedTest, EmptyInput)
{
  using keys_col  = cudf::test::fixed_width_column_wrapper<TypeParam, int32_t>;
  using lists_col = cudf::test::lists_column_wrapper<TypeParam, int32_t>;

  // Keys and lists columns are all empty.
  auto const keys   = keys_col{};
  auto const lists0 = lists_col{{1, 2, 3}, {4, 5, 6}};
  auto const lists  = cudf::empty_like(lists0);

  auto const [out_keys, out_lists] = merge_lists(vcol_views{keys}, vcol_views{*lists});
  auto const expected_keys         = keys_col{};
  auto const expected_lists        = cudf::empty_like(lists0);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_keys, *out_keys, verbosity);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*expected_lists, *out_lists, verbosity);
}

TYPED_TEST(GroupbyMergeListsTypedTest, InputWithoutNull)
{
  using keys_col  = cudf::test::fixed_width_column_wrapper<TypeParam, int32_t>;
  using lists_col = cudf::test::lists_column_wrapper<TypeParam, int32_t>;

  auto const keys1 = keys_col{1, 2};
  auto const keys2 = keys_col{1, 3};
  auto const keys3 = keys_col{2, 3, 4};

  auto const lists1 = lists_col{
    {1, 2, 3},  // key = 1
    {4, 5, 6}   // key = 2
  };
  auto const lists2 = lists_col{
    {10, 11},  // key = 1
    {11, 12}   // key = 3
  };
  auto const lists3 = lists_col{
    {20, 21, 22},  // key = 2
    {23, 24, 25},  // key = 3
    {24, 25, 26}   // key = 4
  };

  auto const [out_keys, out_lists] =
    merge_lists(vcol_views{keys1, keys2, keys3}, vcol_views{lists1, lists2, lists3});
  auto const expected_keys  = keys_col{1, 2, 3, 4};
  auto const expected_lists = lists_col{
    {1, 2, 3, 10, 11},      // key = 1
    {4, 5, 6, 20, 21, 22},  // key = 2
    {11, 12, 23, 24, 25},   // key = 3
    {24, 25, 26}            // key = 4
  };

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_keys, *out_keys, verbosity);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_lists, *out_lists, verbosity);
}

TYPED_TEST(GroupbyMergeListsTypedTest, InputHasNulls)
{
  using keys_col  = cudf::test::fixed_width_column_wrapper<TypeParam, int32_t>;
  using lists_col = cudf::test::lists_column_wrapper<TypeParam, int32_t>;

  auto const keys1 = keys_col{1, 2};
  auto const keys2 = keys_col{1, 3};
  auto const keys3 = keys_col{2, 3, 4};

  // Note that the null elements here are not sorted, while the results from current collect_list
  // are sorted.
  auto const lists1 = lists_col{
    lists_col{{1, null, 3}, null_at(1)},  // key = 1
    lists_col{4, 5, 6}                    // key = 2
  };
  auto const lists2 = lists_col{
    lists_col{10, 11},                          // key = 1
    lists_col{{null, null, null}, all_nulls()}  // key = 3
  };
  auto const lists3 = lists_col{
    lists_col{20, 21, 22},                          // key = 2
    lists_col{{null, 24, null}, nulls_at({0, 2})},  // key = 3
    lists_col{{24, 25, 26}, no_nulls()}             // key = 4
  };

  auto const [out_keys, out_lists] =
    merge_lists(vcol_views{keys1, keys2, keys3}, vcol_views{lists1, lists2, lists3});
  auto const expected_keys  = keys_col{1, 2, 3, 4};
  auto const expected_lists = lists_col{
    lists_col{{1, null, 3, 10, 11}, null_at(1)},                               // key = 1
    lists_col{4, 5, 6, 20, 21, 22},                                            // key = 2
    lists_col{{null, null, null, null, 24, null}, nulls_at({0, 1, 2, 3, 5})},  // key = 3
    lists_col{24, 25, 26}                                                      // key = 4
  };

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_keys, *out_keys, verbosity);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_lists, *out_lists, verbosity);
}

TYPED_TEST(GroupbyMergeListsTypedTest, InputHasEmptyLists)
{
  using keys_col  = cudf::test::fixed_width_column_wrapper<TypeParam, int32_t>;
  using lists_col = cudf::test::lists_column_wrapper<TypeParam, int32_t>;

  auto const keys1 = keys_col{1, 2};
  auto const keys2 = keys_col{1, 3};
  auto const keys3 = keys_col{2, 3, 4};

  auto const lists1 = lists_col{
    {1, 2, 3},  // key = 1
    {}          // key = 2
  };
  auto const lists2 = lists_col{
    {},       // key = 1
    {11, 12}  // key = 3
  };
  auto const lists3 = lists_col{
    {},           // key = 2
    {},           // key = 3
    {24, 25, 26}  // key = 4
  };

  auto const [out_keys, out_lists] =
    merge_lists(vcol_views{keys1, keys2, keys3}, vcol_views{lists1, lists2, lists3});
  auto const expected_keys  = keys_col{1, 2, 3, 4};
  auto const expected_lists = lists_col{
    {1, 2, 3},    // key = 1
    {},           // key = 2
    {11, 12},     // key = 3
    {24, 25, 26}  // key = 4
  };

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_keys, *out_keys, verbosity);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_lists, *out_lists, verbosity);
}

TYPED_TEST(GroupbyMergeListsTypedTest, InputHasNullsAndEmptyLists)
{
  using keys_col  = cudf::test::fixed_width_column_wrapper<TypeParam, int32_t>;
  using lists_col = cudf::test::lists_column_wrapper<TypeParam, int32_t>;

  auto const keys1 = keys_col{1, 2, 3};
  auto const keys2 = keys_col{1, 3, 4};
  auto const keys3 = keys_col{2, 3, 4};

  // Note that the null elements here are not sorted, while the results from current collect_list
  // are sorted.
  auto const lists1 = lists_col{
    lists_col{{1, null, 3}, null_at(1)},  // key = 1
    lists_col{},                          // key = 2
    lists_col{4, 5}                       // key = 3
  };
  auto const lists2 = lists_col{
    lists_col{10, 11},                           // key = 1
    lists_col{{null, null, null}, all_nulls()},  // key = 3
    lists_col{}                                  // key = 4
  };
  auto const lists3 = lists_col{
    lists_col{20, 21, 22},                          // key = 2
    lists_col{{null, 24, null}, nulls_at({0, 2})},  // key = 3
    lists_col{{24, 25, 26}, no_nulls()}             // key = 4
  };

  auto const [out_keys, out_lists] =
    merge_lists(vcol_views{keys1, keys2, keys3}, vcol_views{lists1, lists2, lists3});
  auto const expected_keys  = keys_col{1, 2, 3, 4};
  auto const expected_lists = lists_col{
    lists_col{{1, null, 3, 10, 11}, null_at(1)},                                     // key = 1
    lists_col{20, 21, 22},                                                           // key = 2
    lists_col{{4, 5, null, null, null, null, 24, null}, nulls_at({2, 3, 4, 5, 7})},  // key = 3
    lists_col{24, 25, 26}                                                            // key = 4
  };

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_keys, *out_keys, verbosity);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_lists, *out_lists, verbosity);
}

TYPED_TEST(GroupbyMergeListsTypedTest, InputHasListsOfLists)
{
  using keys_col  = cudf::test::fixed_width_column_wrapper<TypeParam, int32_t>;
  using lists_col = cudf::test::lists_column_wrapper<TypeParam, int32_t>;

  auto const keys1 = keys_col{1, 2};
  auto const keys2 = keys_col{1, 3};
  auto const keys3 = keys_col{2, 3, 4};

  auto const lists1 = lists_col{
    lists_col{lists_col{1, 2, 3}, lists_col{4}, lists_col{5, 6}},  // key = 1
    lists_col{lists_col{}, lists_col{7}}                           // key = 2
  };
  auto const lists2 = lists_col{
    lists_col{lists_col{}, lists_col{8, 9}},     // key = 1
    lists_col{lists_col{11}, lists_col{12, 13}}  // key = 3
  };
  auto const lists3 = lists_col{
    lists_col{lists_col{14}, lists_col{15, 16, 17, 18}},             // key = 2
    lists_col{lists_col{}},                                          // key = 3
    lists_col{lists_col{17, 18, 19, 20, 21}, lists_col{18, 19, 20}}  // key = 4
  };

  auto const [out_keys, out_lists] =
    merge_lists(vcol_views{keys1, keys2, keys3}, vcol_views{lists1, lists2, lists3});
  auto const expected_keys  = keys_col{1, 2, 3, 4};
  auto const expected_lists = lists_col{
    lists_col{
      lists_col{1, 2, 3}, lists_col{4}, lists_col{5, 6}, lists_col{}, lists_col{8, 9}},  // key = 1
    lists_col{lists_col{}, lists_col{7}, lists_col{14}, lists_col{15, 16, 17, 18}},      // key = 2
    lists_col{lists_col{11}, lists_col{12, 13}, lists_col{}},                            // key = 3
    lists_col{lists_col{17, 18, 19, 20, 21}, lists_col{18, 19, 20}}                      // key = 4
  };

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_keys, *out_keys, verbosity);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_lists, *out_lists, verbosity);
}

TYPED_TEST(GroupbyMergeListsTypedTest, SlicedColumnsInput)
{
  using keys_col  = cudf::test::fixed_width_column_wrapper<TypeParam, int32_t>;
  using lists_col = cudf::test::lists_column_wrapper<TypeParam, int32_t>;

  auto const keys1_original = keys_col{1, 2, 4, 5, 6, 7, 8, 9, 10};
  auto const keys2_original = keys_col{0, 0, 1, 1, 1, 3, 4, 5, 6};
  auto const keys3_original = keys_col{0, 1, 2, 3, 4, 5, 6, 7, 8};

  auto const keys1 = cudf::slice(keys1_original, {0, 2})[0];  // { 1, 2 }
  auto const keys2 = cudf::slice(keys2_original, {4, 6})[0];  // { 1, 3 }
  auto const keys3 = cudf::slice(keys3_original, {2, 5})[0];  // { 2, 3, 4 }

  auto const lists1_original = lists_col{
    {10, 11, 12},
    {12, 13, 14},
    {1, 2, 3},  // key = 1
    {4, 5, 6}   // key = 2
  };
  auto const lists2_original = lists_col{{1, 2},
                                         {10, 11},  // key = 1
                                         {11, 12},  // key = 3
                                         {13},
                                         {14},
                                         {15, 16}};
  auto const lists3_original = lists_col{{20, 21, 22},  // key = 2
                                         {23, 24, 25},  // key = 3
                                         {24, 25, 26},  // key = 4
                                         {1, 2, 3, 4, 5}};

  auto const lists1 = cudf::slice(lists1_original, {2, 4})[0];
  auto const lists2 = cudf::slice(lists2_original, {1, 3})[0];
  auto const lists3 = cudf::slice(lists3_original, {0, 3})[0];

  auto const [out_keys, out_lists] =
    merge_lists(vcol_views{keys1, keys2, keys3}, vcol_views{lists1, lists2, lists3});
  auto const expected_keys  = keys_col{1, 2, 3, 4};
  auto const expected_lists = lists_col{
    {1, 2, 3, 10, 11},      // key = 1
    {4, 5, 6, 20, 21, 22},  // key = 2
    {11, 12, 23, 24, 25},   // key = 3
    {24, 25, 26}            // key = 4
  };

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_keys, *out_keys, verbosity);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_lists, *out_lists, verbosity);
}

struct GroupbyMergeListsTest : public cudf::test::BaseFixture {};

TEST_F(GroupbyMergeListsTest, StringsColumnInput)
{
  using strings_col = cudf::test::strings_column_wrapper;
  using lists_col   = cudf::test::lists_column_wrapper<cudf::string_view>;

  auto const keys1 = strings_col{"dog", "unknown"};
  auto const keys2 = strings_col{"banana", "unknown", "dog"};
  auto const keys3 = strings_col{"apple", "dog", "water melon"};

  auto const lists1 = lists_col{
    lists_col{"Poodle", "Golden Retriever", "Corgi"},            // key = "dog"
    lists_col{{"Whale", "" /*NULL*/, "Polar Bear"}, null_at(1)}  // key = "unknown"
  };
  auto const lists2 = lists_col{
    lists_col{"Green", "Yellow"},                       // key = "banana"
    lists_col{},                                        // key = "unknown"
    lists_col{{"" /*NULL*/, "" /*NULL*/}, all_nulls()}  // key = "dog"
  };
  auto const lists3 = lists_col{
    lists_col{"Fuji", "Red Delicious"},                                          // key = "apple"
    lists_col{{"" /*NULL*/, "German Shepherd", "" /*NULL*/}, nulls_at({0, 2})},  // key = "dog"
    lists_col{{"Seeedless", "Mini"}, no_nulls()}  // key = "water melon"
  };

  auto const [out_keys, out_lists] =
    merge_lists(vcol_views{keys1, keys2, keys3}, vcol_views{lists1, lists2, lists3});
  auto const expected_keys  = strings_col{"apple", "banana", "dog", "unknown", "water melon"};
  auto const expected_lists = lists_col{
    lists_col{"Fuji", "Red Delicious"},  // key = "apple"
    lists_col{"Green", "Yellow"},        // key = "banana"
    lists_col{{
                "Poodle",
                "Golden Retriever",
                "Corgi",
                "" /*NULL*/,
                "" /*NULL*/,
                "" /*NULL*/,
                "German Shepherd",
                "" /*NULL*/
              },
              nulls_at({3, 4, 5, 7})},                            // key = "dog"
    lists_col{{"Whale", "" /*NULL*/, "Polar Bear"}, null_at(1)},  // key = "unknown"
    lists_col{{"Seeedless", "Mini"}, no_nulls()}                  // key = "water melon"
  };

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_keys, *out_keys, verbosity);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_lists, *out_lists, verbosity);
}
