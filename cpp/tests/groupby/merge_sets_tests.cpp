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
#include <cudf/lists/sorting.hpp>
#include <cudf/sorting.hpp>
#include <cudf/table/table_view.hpp>

using namespace cudf::test::iterators;

namespace {
constexpr cudf::test::debug_output_level verbosity{cudf::test::debug_output_level::FIRST_ERROR};
constexpr int32_t null{0};  // Mark for null elements

using vcol_views = std::vector<cudf::column_view>;

auto merge_sets(vcol_views const& keys_cols, vcol_views const& values_cols)
{
  // Append all the keys and lists together.
  auto const keys   = cudf::concatenate(keys_cols);
  auto const values = cudf::concatenate(values_cols);

  std::vector<cudf::groupby::aggregation_request> requests;
  requests.emplace_back();
  requests[0].values = *values;
  requests[0].aggregations.emplace_back(
    cudf::make_merge_sets_aggregation<cudf::groupby_aggregation>());

  auto const result      = cudf::groupby::groupby(cudf::table_view({*keys})).aggregate(requests);
  auto const result_keys = result.first->view();                 // <== table_view of 1 column
  auto const result_vals = result.second[0].results[0]->view();  // <== column_view

  // Sort the output columns based on the output keys.
  // This is to facilitate comparison of the output with the expected columns.
  auto keys_vals_sorted = cudf::sort_by_key(cudf::table_view{{result_keys.column(0), result_vals}},
                                            result_keys,
                                            {},
                                            {cudf::null_order::AFTER})
                            ->release();

  // After the columns were reordered, individual rows of the output values column (which are lists)
  // also need to be sorted.
  auto out_values =
    cudf::lists::sort_lists(cudf::lists_column_view{keys_vals_sorted.back()->view()},
                            cudf::order::ASCENDING,
                            cudf::null_order::AFTER);

  return std::pair(std::move(keys_vals_sorted.front()), std::move(out_values));
}

}  // namespace

template <typename V>
struct GroupbyMergeSetsTypedTest : public cudf::test::BaseFixture {};

using FixedWidthTypesNotBool = cudf::test::Concat<cudf::test::IntegralTypesNotBool,
                                                  cudf::test::FloatingPointTypes,
                                                  cudf::test::TimestampTypes>;
TYPED_TEST_SUITE(GroupbyMergeSetsTypedTest, FixedWidthTypesNotBool);

TYPED_TEST(GroupbyMergeSetsTypedTest, InvalidInput)
{
  using keys_col  = cudf::test::fixed_width_column_wrapper<TypeParam, int32_t>;
  using lists_col = cudf::test::lists_column_wrapper<TypeParam, int32_t>;

  auto const keys = keys_col{1, 2, 3};

  // The input lists column must NOT be nullable.
  auto const lists = lists_col{{lists_col{1}, lists_col{} /*NULL*/, lists_col{2}}, null_at(1)};
  EXPECT_THROW(merge_sets({keys}, {lists}), cudf::logic_error);

  // The input column must be a lists column.
  auto const non_lists = keys_col{1, 2, 3, 4, 5};
  EXPECT_THROW(merge_sets({keys}, {non_lists}), cudf::logic_error);
}

TYPED_TEST(GroupbyMergeSetsTypedTest, EmptyInput)
{
  using keys_col  = cudf::test::fixed_width_column_wrapper<TypeParam, int32_t>;
  using lists_col = cudf::test::lists_column_wrapper<TypeParam, int32_t>;

  // Keys and lists columns are all empty.
  auto const keys   = keys_col{};
  auto const lists0 = lists_col{{1, 2, 3}, {4, 5, 6}};
  auto const lists  = cudf::empty_like(lists0);

  auto const [out_keys, out_lists] = merge_sets(vcol_views{keys}, vcol_views{*lists});
  auto const expected_keys         = keys_col{};
  auto const expected_lists        = cudf::empty_like(lists0);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_keys, *out_keys, verbosity);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*expected_lists, *out_lists, verbosity);
}

TYPED_TEST(GroupbyMergeSetsTypedTest, InputWithoutNull)
{
  using keys_col  = cudf::test::fixed_width_column_wrapper<TypeParam, int32_t>;
  using lists_col = cudf::test::lists_column_wrapper<TypeParam, int32_t>;

  auto const keys1 = keys_col{1, 2};
  auto const keys2 = keys_col{1, 3};
  auto const keys3 = keys_col{2, 3, 4};

  auto const lists1 = lists_col{
    {1, 2, 3, 4, 5, 6},       // key = 1
    {10, 11, 12, 13, 14, 15}  // key = 2
  };
  auto const lists2 = lists_col{
    {4, 5, 6, 7, 8, 9},       // key = 1
    {20, 21, 22, 23, 24, 25}  // key = 3
  };
  auto const lists3 = lists_col{
    {11, 12},                  // key = 2
    {23, 24, 25, 26, 27, 28},  // key = 3
    {30, 31, 32}               // key = 4
  };

  auto const [out_keys, out_lists] =
    merge_sets(vcol_views{keys1, keys2, keys3}, vcol_views{lists1, lists2, lists3});
  auto const expected_keys  = keys_col{1, 2, 3, 4};
  auto const expected_lists = lists_col{
    {1, 2, 3, 4, 5, 6, 7, 8, 9},           // key = 1
    {10, 11, 12, 13, 14, 15},              // key = 2
    {20, 21, 22, 23, 24, 25, 26, 27, 28},  // key = 3
    {30, 31, 32}                           // key = 4
  };

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_keys, *out_keys, verbosity);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_lists, *out_lists, verbosity);
}

TYPED_TEST(GroupbyMergeSetsTypedTest, InputHasNulls)
{
  using keys_col  = cudf::test::fixed_width_column_wrapper<TypeParam, int32_t>;
  using lists_col = cudf::test::lists_column_wrapper<TypeParam, int32_t>;

  auto const keys1 = keys_col{1, 2};
  auto const keys2 = keys_col{1, 3};
  auto const keys3 = keys_col{2, 3, 4};

  auto const lists1 = lists_col{
    lists_col{{1, null, null, null, 5, 6}, nulls_at({1, 2, 3})},  // key = 1
    lists_col{10, 11, 12, 13, 14, 15}                             // key = 2
  };
  auto const lists2 = lists_col{
    lists_col{{null, null, 6, 7, 8, 9}, nulls_at({0, 1})},  // key = 1
    lists_col{{null, 21, 22, 23, 24, 25}, null_at(0)}       // key = 3
  };
  auto const lists3 = lists_col{
    lists_col{11, 12},                     // key = 2
    lists_col{23, 24, 25, 26, 27, 28},     // key = 3
    lists_col{{30, null, 32}, null_at(1)}  // key = 4
  };

  auto const [out_keys, out_lists] =
    merge_sets(vcol_views{keys1, keys2, keys3}, vcol_views{lists1, lists2, lists3});
  auto const expected_keys  = keys_col{1, 2, 3, 4};
  auto const expected_lists = lists_col{
    lists_col{{1, 5, 6, 7, 8, 9, null}, null_at(6)},                // key = 1
    lists_col{10, 11, 12, 13, 14, 15},                              // key = 2
    lists_col{{21, 22, 23, 24, 25, 26, 27, 28, null}, null_at(8)},  // key = 3
    lists_col{{30, 32, null}, null_at(2)}                           // key = 4
  };

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_keys, *out_keys, verbosity);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_lists, *out_lists, verbosity);
}

TYPED_TEST(GroupbyMergeSetsTypedTest, InputHasEmptyLists)
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
    {0, 1, 2, 3, 4, 5},       // key = 1
    {11, 12, 12, 12, 12, 12}  // key = 3
  };
  auto const lists3 = lists_col{
    {},           // key = 2
    {},           // key = 3
    {24, 25, 26}  // key = 4
  };

  auto const [out_keys, out_lists] =
    merge_sets(vcol_views{keys1, keys2, keys3}, vcol_views{lists1, lists2, lists3});
  auto const expected_keys  = keys_col{1, 2, 3, 4};
  auto const expected_lists = lists_col{
    {0, 1, 2, 3, 4, 5},  // key = 1
    {},                  // key = 2
    {11, 12},            // key = 3
    {24, 25, 26}         // key = 4
  };

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_keys, *out_keys, verbosity);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_lists, *out_lists, verbosity);
}

TYPED_TEST(GroupbyMergeSetsTypedTest, InputHasNullsAndEmptyLists)
{
  using keys_col  = cudf::test::fixed_width_column_wrapper<TypeParam, int32_t>;
  using lists_col = cudf::test::lists_column_wrapper<TypeParam, int32_t>;

  auto const keys1 = keys_col{1, 2, 3};
  auto const keys2 = keys_col{1, 3, 4};
  auto const keys3 = keys_col{2, 3, 4};

  auto const lists1 = lists_col{
    lists_col{{null, 1, 2, 3}, null_at(0)},  // key = 1
    lists_col{},                             // key = 2
    lists_col{}                              // key = 3
  };
  auto const lists2 = lists_col{
    lists_col{0, 1, 2, 3, 4, 5},                                        // key = 1
    lists_col{{null, 11, null, 12, 12, 12, 12, 12}, nulls_at({0, 2})},  // key = 3
    lists_col{20}                                                       // key = 4
  };
  auto const lists3 = lists_col{
    lists_col{},                                                    // key = 2
    lists_col{},                                                    // key = 3
    lists_col{{24, 25, null, null, null, 26}, nulls_at({2, 3, 4})}  // key = 4
  };

  auto const [out_keys, out_lists] =
    merge_sets(vcol_views{keys1, keys2, keys3}, vcol_views{lists1, lists2, lists3});
  auto const expected_keys  = keys_col{1, 2, 3, 4};
  auto const expected_lists = lists_col{
    lists_col{{0, 1, 2, 3, 4, 5, null}, null_at(6)},  // key = 1
    lists_col{},                                      // key = 2
    lists_col{{11, 12, null}, null_at(2)},            // key = 3
    lists_col{{20, 24, 25, 26, null}, null_at(4)}     // key = 4
  };

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_keys, *out_keys, verbosity);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_lists, *out_lists, verbosity);
}

TYPED_TEST(GroupbyMergeSetsTypedTest, SlicedColumnsInput)
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
    {10, 11, 12, 10, 11, 12, 10, 11, 12},
    {12, 13, 12, 13, 12, 13, 12, 13, 14},
    {1, 2, 3, 1, 2, 3, 1, 2, 3},  // key = 1
    {4, 5, 6, 4, 5, 6, 4, 5, 6}   // key = 2
  };
  auto const lists2_original = lists_col{{1, 1, 1, 1, 1, 1, 1, 2},
                                         {10, 11, 11, 11, 11, 11, 12},                  // key = 1
                                         {11, 12, 13, 12, 13, 12, 13, 12, 13, 14, 15},  // key = 3
                                         {13, 14, 15},
                                         {14, 15, 16},
                                         {15, 16}};
  auto const lists3_original = lists_col{{20, 21, 20, 21, 20, 21, 20, 21, 22},  // key = 2
                                         {23, 24, 25, 23, 24, 25},              // key = 3
                                         {24, 25, 26},                          // key = 4
                                         {1, 2, 3, 4, 5}};

  auto const lists1 = cudf::slice(lists1_original, {2, 4})[0];
  auto const lists2 = cudf::slice(lists2_original, {1, 3})[0];
  auto const lists3 = cudf::slice(lists3_original, {0, 3})[0];

  auto const [out_keys, out_lists] =
    merge_sets(vcol_views{keys1, keys2, keys3}, vcol_views{lists1, lists2, lists3});
  auto const expected_keys  = keys_col{1, 2, 3, 4};
  auto const expected_lists = lists_col{
    {1, 2, 3, 10, 11, 12},             // key = 1
    {4, 5, 6, 20, 21, 22},             // key = 2
    {11, 12, 13, 14, 15, 23, 24, 25},  // key = 3
    {24, 25, 26}                       // key = 4
  };

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_keys, *out_keys, verbosity);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_lists, *out_lists, verbosity);
}

struct GroupbyMergeSetsTest : public cudf::test::BaseFixture {};

TEST_F(GroupbyMergeSetsTest, StringsColumnInput)
{
  using strings_col = cudf::test::strings_column_wrapper;
  using lists_col   = cudf::test::lists_column_wrapper<cudf::string_view>;

  auto const keys1 = strings_col{"apple", "dog", "unknown"};
  auto const keys2 = strings_col{"banana", "unknown", "dog"};
  auto const keys3 = strings_col{"apple", "dog", "water melon"};

  auto const lists1 = lists_col{
    lists_col{"Fuji", "Honey Bee"},                              // key = "apple"
    lists_col{"Poodle", "Golden Retriever", "Corgi"},            // key = "dog"
    lists_col{{"Whale", "" /*NULL*/, "Polar Bear"}, null_at(1)}  // key = "unknown"
  };
  auto const lists2 = lists_col{
    lists_col{"Green", "Yellow"},                                    // key = "banana"
    lists_col{},                                                     // key = "unknown"
    lists_col{{"" /*NULL*/, "" /*NULL*/, "" /*NULL*/}, all_nulls()}  // key = "dog"
  };
  auto const lists3 = lists_col{
    lists_col{"Fuji", "Red Delicious"},  // key = "apple"
    lists_col{{"" /*NULL*/, "Corgi", "German Shepherd", "" /*NULL*/, "Golden Retriever"},
              nulls_at({0, 3})},                  // key = "dog"
    lists_col{{"Seeedless", "Mini"}, no_nulls()}  // key = "water melon"
  };

  auto const [out_keys, out_lists] =
    merge_sets(vcol_views{keys1, keys2, keys3}, vcol_views{lists1, lists2, lists3});
  auto const expected_keys  = strings_col{"apple", "banana", "dog", "unknown", "water melon"};
  auto const expected_lists = lists_col{
    lists_col{"Fuji", "Honey Bee", "Red Delicious"},  // key = "apple"
    lists_col{"Green", "Yellow"},                     // key = "banana"
    lists_col{{
                "Corgi", "German Shepherd", "Golden Retriever", "Poodle", "" /*NULL*/
              },
              null_at(4)},                                        // key = "dog"
    lists_col{{"Polar Bear", "Whale", "" /*NULL*/}, null_at(2)},  // key = "unknown"
    lists_col{{"Mini", "Seeedless"}, no_nulls()}                  // key = "water melon"
  };

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_keys, *out_keys, verbosity);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_lists, *out_lists, verbosity);
}
