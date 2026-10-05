/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/iterator_utilities.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/aggregation.hpp>
#include <cudf/reduction.hpp>
#include <cudf/sorting.hpp>

namespace {

auto collect_set(cudf::column_view const& input,
                 std::unique_ptr<cudf::reduce_aggregation> const& agg)
{
  auto const result_scalar = cudf::reduce(input, *agg, cudf::data_type{cudf::type_id::LIST});

  // The results of `collect_set` are unordered thus we need to sort them for comparison.
  auto const result_sorted_table =
    cudf::sort(cudf::table_view{{dynamic_cast<cudf::list_scalar*>(result_scalar.get())->view()}},
               {},
               {cudf::null_order::AFTER});

  return std::make_unique<cudf::list_scalar>(std::move(result_sorted_table->get_column(0)));
}

}  // namespace

template <typename T>
struct CollectTestFixedWidth : public cudf::test::BaseFixture {};

using CollectFixedWidthTypes = cudf::test::Concat<cudf::test::IntegralTypesNotBool,
                                                  cudf::test::FloatingPointTypes,
                                                  cudf::test::ChronoTypes,
                                                  cudf::test::FixedPointTypes>;
TYPED_TEST_SUITE(CollectTestFixedWidth, CollectFixedWidthTypes);

// ------------------------------------------------------------------------
TYPED_TEST(CollectTestFixedWidth, CollectList)
{
  using fw_wrapper = cudf::test::fixed_width_column_wrapper<TypeParam, int32_t>;

  std::vector<int> values({5, 0, -120, -111, 0, 64, 63, 99, 123, -16});
  std::vector<bool> null_mask({true, true, false, true, true, true, false, true, false, true});

  // null_include without nulls
  fw_wrapper col(values.begin(), values.end());
  auto const ret = cudf::reduce(col,
                                *cudf::make_collect_list_aggregation<cudf::reduce_aggregation>(),
                                cudf::data_type{cudf::type_id::LIST});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(col, dynamic_cast<cudf::list_scalar*>(ret.get())->view());

  // null_include with nulls
  fw_wrapper col_with_null(values.begin(), values.end(), null_mask.begin());
  auto const ret1 = cudf::reduce(col_with_null,
                                 *cudf::make_collect_list_aggregation<cudf::reduce_aggregation>(),
                                 cudf::data_type{cudf::type_id::LIST});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(col_with_null,
                                 dynamic_cast<cudf::list_scalar*>(ret1.get())->view());

  // null_exclude with nulls
  fw_wrapper col_null_filtered{{5, 0, -111, 0, 64, 99, -16}};
  auto const ret2 = cudf::reduce(
    col_with_null,
    *cudf::make_collect_list_aggregation<cudf::reduce_aggregation>(cudf::null_policy::EXCLUDE),
    cudf::data_type{cudf::type_id::LIST});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(col_null_filtered,
                                 dynamic_cast<cudf::list_scalar*>(ret2.get())->view());
}

TYPED_TEST(CollectTestFixedWidth, CollectSet)
{
  using fw_wrapper = cudf::test::fixed_width_column_wrapper<TypeParam, int32_t>;

  std::vector<int> values({5, 0, 120, 0, 0, 64, 64, 99, 120, 99});
  std::vector<bool> null_mask({true, true, false, true, true, true, false, true, false, true});

  fw_wrapper col(values.begin(), values.end());
  fw_wrapper col_with_null(values.begin(), values.end(), null_mask.begin());

  auto null_exclude = cudf::make_collect_set_aggregation<cudf::reduce_aggregation>(
    cudf::null_policy::EXCLUDE, cudf::null_equality::UNEQUAL, cudf::nan_equality::ALL_EQUAL);
  auto null_eq = cudf::make_collect_set_aggregation<cudf::reduce_aggregation>(
    cudf::null_policy::INCLUDE, cudf::null_equality::EQUAL, cudf::nan_equality::ALL_EQUAL);
  auto null_unequal = cudf::make_collect_set_aggregation<cudf::reduce_aggregation>(
    cudf::null_policy::INCLUDE, cudf::null_equality::UNEQUAL, cudf::nan_equality::ALL_EQUAL);

  // test without nulls
  auto const ret = collect_set(col, null_eq);
  fw_wrapper expected{{0, 5, 64, 99, 120}};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, dynamic_cast<cudf::list_scalar*>(ret.get())->view());

  // null exclude
  auto const ret1 = collect_set(col_with_null, null_exclude);
  fw_wrapper expected1{{0, 5, 64, 99}};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected1, dynamic_cast<cudf::list_scalar*>(ret1.get())->view());

  // null equal
  auto const ret2 = collect_set(col_with_null, null_eq);
  fw_wrapper expected2{{0, 5, 64, 99, -1}, {1, 1, 1, 1, 0}};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected2, dynamic_cast<cudf::list_scalar*>(ret2.get())->view());

  // null unequal
  auto const ret3 = collect_set(col_with_null, null_unequal);
  fw_wrapper expected3{{0, 5, 64, 99, -1, -1, -1}, {1, 1, 1, 1, 0, 0, 0}};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected3, dynamic_cast<cudf::list_scalar*>(ret3.get())->view());
}

TYPED_TEST(CollectTestFixedWidth, MergeLists)
{
  using fw_wrapper = cudf::test::fixed_width_column_wrapper<TypeParam, int32_t>;
  using lists_col  = cudf::test::lists_column_wrapper<TypeParam, int32_t>;

  // test without nulls
  auto const lists1    = lists_col{{1, 2, 3}, {}, {}, {4}, {5, 6, 7}, {8, 9}, {}};
  auto const expected1 = fw_wrapper{{1, 2, 3, 4, 5, 6, 7, 8, 9}};
  auto const ret1 = cudf::reduce(lists1,
                                 *cudf::make_merge_lists_aggregation<cudf::reduce_aggregation>(),
                                 cudf::data_type{cudf::type_id::LIST});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected1, dynamic_cast<cudf::list_scalar*>(ret1.get())->view());

  // test with nulls
  auto const lists2 = lists_col{{
                                  lists_col{1, 2, 3},
                                  lists_col{},
                                  lists_col{{0, 4, 0, 5}, cudf::test::iterators::nulls_at({0, 2})},
                                  lists_col{{0, 0, 0}, cudf::test::iterators::all_nulls()},
                                  lists_col{6},
                                  lists_col{-1, -1},  // null_list
                                  lists_col{7, 8, 9},
                                },
                                cudf::test::iterators::null_at(5)};
  auto const expected2 = fw_wrapper{{1, 2, 3, 0, 4, 0, 5, 0, 0, 0, 6, 7, 8, 9},
                                    {1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1}};
  auto const ret2 = cudf::reduce(lists2,
                                 *cudf::make_merge_lists_aggregation<cudf::reduce_aggregation>(),
                                 cudf::data_type{cudf::type_id::LIST});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected2, dynamic_cast<cudf::list_scalar*>(ret2.get())->view());
}

TYPED_TEST(CollectTestFixedWidth, MergeSets)
{
  using fw_wrapper = cudf::test::fixed_width_column_wrapper<TypeParam, int32_t>;
  using lists_col  = cudf::test::lists_column_wrapper<TypeParam, int32_t>;

  // test without nulls
  auto const lists1    = lists_col{{1, 2, 3}, {}, {}, {4}, {1, 3, 4}, {0, 3, 10}, {}};
  auto const expected1 = fw_wrapper{{0, 1, 2, 3, 4, 10}};
  auto const ret1 =
    collect_set(lists1, cudf::make_merge_sets_aggregation<cudf::reduce_aggregation>());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected1, dynamic_cast<cudf::list_scalar*>(ret1.get())->view());

  // test with null_equal
  auto const lists2 = lists_col{{
                                  lists_col{1, 2, 3},
                                  lists_col{},
                                  lists_col{{0, 4, 0, 5}, cudf::test::iterators::nulls_at({0, 2})},
                                  lists_col{{0, 0, 0}, cudf::test::iterators::all_nulls()},
                                  lists_col{5},
                                  lists_col{-1, -1},  // null_list
                                  lists_col{1, 3, 5},
                                },
                                cudf::test::iterators::null_at(5)};
  auto const expected2 = fw_wrapper{{1, 2, 3, 4, 5, 0}, {1, 1, 1, 1, 1, 0}};
  auto const ret2 =
    collect_set(lists2, cudf::make_merge_sets_aggregation<cudf::reduce_aggregation>());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected2, dynamic_cast<cudf::list_scalar*>(ret2.get())->view());

  // test with null_unequal
  auto const& lists3   = lists2;
  auto const expected3 = fw_wrapper{{1, 2, 3, 4, 5, 0, 0, 0, 0, 0}, {1, 1, 1, 1, 1, 0, 0, 0, 0, 0}};
  auto const ret3      = collect_set(
    lists3,
    cudf::make_merge_sets_aggregation<cudf::reduce_aggregation>(cudf::null_equality::UNEQUAL));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected3, dynamic_cast<cudf::list_scalar*>(ret3.get())->view());
}

struct CollectTest : public cudf::test::BaseFixture {};

TEST_F(CollectTest, CollectSetWithNaN)
{
  using fp_wrapper = cudf::test::fixed_width_column_wrapper<float>;

  fp_wrapper col{{1.0f, 1.0f, -2.3e-5f, -2.3e-5f, 2.3e5f, 2.3e5f, -NAN, -NAN, NAN, NAN, 0.0f, 0.0f},
                 {true, true, true, true, true, true, true, true, true, true, false, false}};

  // nan unequal with null equal
  fp_wrapper expected1{{-2.3e-5f, 1.0f, 2.3e5f, -NAN, -NAN, NAN, NAN, 0.0f},
                       {true, true, true, true, true, true, true, false}};
  auto const ret1 = collect_set(
    col,
    cudf::make_collect_set_aggregation<cudf::reduce_aggregation>(
      cudf::null_policy::INCLUDE, cudf::null_equality::EQUAL, cudf::nan_equality::UNEQUAL));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected1, dynamic_cast<cudf::list_scalar*>(ret1.get())->view());

  // nan unequal with null unequal
  fp_wrapper expected2{{-2.3e-5f, 1.0f, 2.3e5f, -NAN, -NAN, NAN, NAN, 0.0f, 0.0f},
                       {true, true, true, true, true, true, true, false, false}};
  auto const ret2 = collect_set(
    col,
    cudf::make_collect_set_aggregation<cudf::reduce_aggregation>(
      cudf::null_policy::INCLUDE, cudf::null_equality::UNEQUAL, cudf::nan_equality::UNEQUAL));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected2, dynamic_cast<cudf::list_scalar*>(ret2.get())->view());

  // nan equal with null equal
  fp_wrapper expected3{{-2.3e-5f, 1.0f, 2.3e5f, NAN, 0.0f}, {true, true, true, true, false}};
  auto const ret3 = collect_set(
    col,
    cudf::make_collect_set_aggregation<cudf::reduce_aggregation>(
      cudf::null_policy::INCLUDE, cudf::null_equality::EQUAL, cudf::nan_equality::ALL_EQUAL));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected3, dynamic_cast<cudf::list_scalar*>(ret3.get())->view());

  // nan equal with null unequal
  fp_wrapper expected4{{-2.3e-5f, 1.0f, 2.3e5f, -NAN, 0.0f, 0.0f},
                       {true, true, true, true, false, false}};
  auto const ret4 = collect_set(
    col,
    cudf::make_collect_set_aggregation<cudf::reduce_aggregation>(
      cudf::null_policy::INCLUDE, cudf::null_equality::UNEQUAL, cudf::nan_equality::ALL_EQUAL));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected4, dynamic_cast<cudf::list_scalar*>(ret4.get())->view());
}

TEST_F(CollectTest, MergeSetsWithNaN)
{
  using fp_wrapper = cudf::test::fixed_width_column_wrapper<float>;
  using lists_col  = cudf::test::lists_column_wrapper<float>;

  auto const col = lists_col{
    lists_col{1.0f, -2.3e-5f, NAN},
    lists_col{},
    lists_col{{-2.3e-5f, 2.3e5f, NAN, 0.0f}, cudf::test::iterators::nulls_at({3})},
    lists_col{{0.0f, 0.0f}, cudf::test::iterators::all_nulls()},
    lists_col{-NAN},
  };

  // nan unequal with null equal
  fp_wrapper expected1{{-2.3e-5f, 1.0f, 2.3e5f, -NAN, NAN, NAN, 0.0f},
                       {true, true, true, true, true, true, false}};
  auto const ret1 = collect_set(col,
                                cudf::make_merge_sets_aggregation<cudf::reduce_aggregation>(
                                  cudf::null_equality::EQUAL, cudf::nan_equality::UNEQUAL));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected1, dynamic_cast<cudf::list_scalar*>(ret1.get())->view());

  // nan unequal with null unequal
  fp_wrapper expected2{{-2.3e-5f, 1.0f, 2.3e5f, -NAN, NAN, NAN, 0.0f, 0.0f, 0.0f},
                       {true, true, true, true, true, true, false, false, false}};
  auto const ret2 = collect_set(col,
                                cudf::make_merge_sets_aggregation<cudf::reduce_aggregation>(
                                  cudf::null_equality::UNEQUAL, cudf::nan_equality::UNEQUAL));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected2, dynamic_cast<cudf::list_scalar*>(ret2.get())->view());

  // nan equal with null equal
  fp_wrapper expected3{{-2.3e-5f, 1.0f, 2.3e5f, -NAN, 0.0f}, {true, true, true, true, false}};
  auto const ret3 = collect_set(col,
                                cudf::make_merge_sets_aggregation<cudf::reduce_aggregation>(
                                  cudf::null_equality::EQUAL, cudf::nan_equality::ALL_EQUAL));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected3, dynamic_cast<cudf::list_scalar*>(ret3.get())->view());

  // nan equal with null unequal
  fp_wrapper expected4{{-2.3e-5f, 1.0f, 2.3e5f, -NAN, 0.0f, 0.0f, 0.0f},
                       {true, true, true, true, false, false, false}};
  auto const ret4 = collect_set(col,
                                cudf::make_merge_sets_aggregation<cudf::reduce_aggregation>(
                                  cudf::null_equality::UNEQUAL, cudf::nan_equality::ALL_EQUAL));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected4, dynamic_cast<cudf::list_scalar*>(ret4.get())->view());
}

TEST_F(CollectTest, CollectStrings)
{
  using str_col   = cudf::test::strings_column_wrapper;
  using lists_col = cudf::test::lists_column_wrapper<cudf::string_view>;

  auto const s_col = str_col{{"a", "a", "b", "b", "b", "c", "c", "d", "e", "e"},
                             {true, true, true, false, true, true, false, true, true, true}};

  // collect_list including nulls
  auto const ret1 = cudf::reduce(s_col,
                                 *cudf::make_collect_list_aggregation<cudf::reduce_aggregation>(),
                                 cudf::data_type{cudf::type_id::LIST});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(s_col, dynamic_cast<cudf::list_scalar*>(ret1.get())->view());

  // collect_list excluding nulls
  auto const expected2 = str_col{"a", "a", "b", "b", "c", "d", "e", "e"};
  auto const ret2      = cudf::reduce(
    s_col,
    *cudf::make_collect_list_aggregation<cudf::reduce_aggregation>(cudf::null_policy::EXCLUDE),
    cudf::data_type{cudf::type_id::LIST});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected2, dynamic_cast<cudf::list_scalar*>(ret2.get())->view());

  // collect_set with null_equal
  auto const expected3 = str_col{{"a", "b", "c", "d", "e", ""}, cudf::test::iterators::null_at(5)};
  auto const ret3 =
    collect_set(s_col, cudf::make_collect_set_aggregation<cudf::reduce_aggregation>());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected3, dynamic_cast<cudf::list_scalar*>(ret3.get())->view());

  // collect_set with null_unequal
  auto const expected4 =
    str_col{{"a", "b", "c", "d", "e", "", ""}, {true, true, true, true, true, false, false}};
  auto const ret4 = collect_set(s_col,
                                cudf::make_collect_set_aggregation<cudf::reduce_aggregation>(
                                  cudf::null_policy::INCLUDE, cudf::null_equality::UNEQUAL));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected4, dynamic_cast<cudf::list_scalar*>(ret4.get())->view());

  lists_col strings{{"a"},
                    {},
                    {"a", "b"},
                    lists_col{{"b", "null", "c"}, cudf::test::iterators::null_at(1)},
                    lists_col{{"null", "d"}, cudf::test::iterators::null_at(0)},
                    lists_col{{"null"}, cudf::test::iterators::null_at(0)},
                    {"e"}};

  // merge_lists
  auto const expected5 = str_col{{"a", "a", "b", "b", "null", "c", "null", "d", "null", "e"},
                                 {true, true, true, true, false, true, false, true, false, true}};
  auto const ret5 = cudf::reduce(strings,
                                 *cudf::make_merge_lists_aggregation<cudf::reduce_aggregation>(),
                                 cudf::data_type{cudf::type_id::LIST});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected5, dynamic_cast<cudf::list_scalar*>(ret5.get())->view());

  // merge_sets with null_equal
  auto const expected6 =
    str_col{{"a", "b", "c", "d", "e", "null"}, {true, true, true, true, true, false}};
  auto const ret6 =
    collect_set(strings, cudf::make_merge_sets_aggregation<cudf::reduce_aggregation>());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected6, dynamic_cast<cudf::list_scalar*>(ret6.get())->view());

  // merge_sets with null_unequal
  auto const expected7 = str_col{{"a", "b", "c", "d", "e", "null", "null", "null"},
                                 {true, true, true, true, true, false, false, false}};
  auto const ret7      = collect_set(
    strings,
    cudf::make_merge_sets_aggregation<cudf::reduce_aggregation>(cudf::null_equality::UNEQUAL));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected7, dynamic_cast<cudf::list_scalar*>(ret7.get())->view());
}

TEST_F(CollectTest, CollectEmptys)
{
  using int_col = cudf::test::fixed_width_column_wrapper<int32_t>;

  // test collect empty columns
  auto empty = int_col{};
  auto ret   = cudf::reduce(empty,
                            *cudf::make_collect_list_aggregation<cudf::reduce_aggregation>(),
                            cudf::data_type{cudf::type_id::LIST});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(int_col{}, dynamic_cast<cudf::list_scalar*>(ret.get())->view());

  ret = collect_set(empty, cudf::make_collect_set_aggregation<cudf::reduce_aggregation>());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(int_col{}, dynamic_cast<cudf::list_scalar*>(ret.get())->view());

  // test collect all null columns
  auto all_nulls = int_col{{1, 2, 3, 4, 5}, {false, false, false, false, false}};
  ret            = cudf::reduce(all_nulls,
                                *cudf::make_collect_list_aggregation<cudf::reduce_aggregation>(),
                                cudf::data_type{cudf::type_id::LIST});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(int_col{}, dynamic_cast<cudf::list_scalar*>(ret.get())->view());

  ret = collect_set(all_nulls, cudf::make_collect_set_aggregation<cudf::reduce_aggregation>());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(int_col{}, dynamic_cast<cudf::list_scalar*>(ret.get())->view());
}

TEST_F(CollectTest, CollectAllNulls)
{
  using int_col = cudf::test::fixed_width_column_wrapper<int32_t>;
  using namespace cudf::test::iterators;

  auto const input    = int_col{{0, 0, 0, 0, 0, 0}, all_nulls()};
  auto const expected = int_col{};

  {
    auto const agg =
      cudf::make_collect_list_aggregation<cudf::reduce_aggregation>(cudf::null_policy::EXCLUDE);
    auto const result = cudf::reduce(input, *agg, cudf::data_type{cudf::type_id::LIST});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected,
                                   dynamic_cast<cudf::list_scalar*>(result.get())->view());
  }
  {
    auto const agg = cudf::make_collect_set_aggregation<cudf::reduce_aggregation>(
      cudf::null_policy::EXCLUDE, cudf::null_equality::UNEQUAL, cudf::nan_equality::ALL_EQUAL);
    auto const result = cudf::reduce(input, *agg, cudf::data_type{cudf::type_id::LIST});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected,
                                   dynamic_cast<cudf::list_scalar*>(result.get())->view());
  }
}

TEST_F(CollectTest, CollectAllNullsWithLists)
{
  using LCW = cudf::test::lists_column_wrapper<int32_t>;
  using namespace cudf::test::iterators;

  // list<list<int>>
  auto const input    = LCW{{LCW{LCW{1, 2, 3}, LCW{4, 5, 6}}, LCW{{1, 2, 3}}}, all_nulls()};
  auto const expected = cudf::empty_like(input);

  {
    auto const agg =
      cudf::make_collect_list_aggregation<cudf::reduce_aggregation>(cudf::null_policy::EXCLUDE);
    auto const result = cudf::reduce(input, *agg, cudf::data_type{cudf::type_id::LIST});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected->view(),
                                   dynamic_cast<cudf::list_scalar*>(result.get())->view());
  }
  {
    auto const agg = cudf::make_collect_set_aggregation<cudf::reduce_aggregation>(
      cudf::null_policy::EXCLUDE, cudf::null_equality::UNEQUAL, cudf::nan_equality::ALL_EQUAL);
    auto const result = cudf::reduce(input, *agg, cudf::data_type{cudf::type_id::LIST});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected->view(),
                                   dynamic_cast<cudf::list_scalar*>(result.get())->view());
  }
}
