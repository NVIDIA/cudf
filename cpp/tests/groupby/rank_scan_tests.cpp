/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <tests/groupby/groupby_test_util.hpp>

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/iterator_utilities.hpp>
#include <cudf_test/table_utilities.hpp>
#include <cudf_test/type_lists.hpp>

using namespace cudf::test::iterators;

template <typename T>
using input              = cudf::test::fixed_width_column_wrapper<T>;
using rank_result_col    = cudf::test::fixed_width_column_wrapper<cudf::size_type>;
using percent_result_col = cudf::test::fixed_width_column_wrapper<double>;
using null_iter_t        = decltype(nulls_at({}));

auto constexpr X     = int32_t{0};  // Placeholder for NULL rows.
auto const all_valid = nulls_at({});

void test_rank_scans(cudf::column_view const& keys,
                     cudf::column_view const& order,
                     cudf::column_view const& expected_dense,
                     cudf::column_view const& expected_rank,
                     cudf::column_view const& expected_percent_rank)
{
  test_single_scan(keys,
                   order,
                   keys,
                   expected_dense,
                   cudf::make_rank_aggregation<cudf::groupby_scan_aggregation>(
                     cudf::rank_method::DENSE, {}, cudf::null_policy::INCLUDE),
                   cudf::null_policy::INCLUDE,
                   cudf::sorted::YES);
  test_single_scan(keys,
                   order,
                   keys,
                   expected_rank,
                   cudf::make_rank_aggregation<cudf::groupby_scan_aggregation>(
                     cudf::rank_method::MIN, {}, cudf::null_policy::INCLUDE),
                   cudf::null_policy::INCLUDE,
                   cudf::sorted::YES);
  test_single_scan(keys,
                   order,
                   keys,
                   expected_percent_rank,
                   cudf::make_rank_aggregation<cudf::groupby_scan_aggregation>(
                     cudf::rank_method::MIN,
                     {},
                     cudf::null_policy::INCLUDE,
                     {},
                     cudf::rank_percentage::ONE_NORMALIZED),
                   cudf::null_policy::INCLUDE,
                   cudf::sorted::YES);
}

struct groupby_rank_scan_test : public cudf::test::BaseFixture {};

struct groupby_rank_scan_test_failures : public cudf::test::BaseFixture {};

template <typename T>
struct typed_groupby_rank_scan_test : public cudf::test::BaseFixture {};

using testing_type_set = cudf::test::Concat<cudf::test::IntegralTypesNotBool,
                                            cudf::test::FloatingPointTypes,
                                            cudf::test::FixedPointTypes,
                                            cudf::test::ChronoTypes>;

TYPED_TEST_SUITE(typed_groupby_rank_scan_test, testing_type_set);

TYPED_TEST(typed_groupby_rank_scan_test, empty_cols)
{
  using T = TypeParam;

  auto const keys            = input<T>{};
  auto const order_by        = input<T>{};
  auto const order_by_struct = cudf::test::structs_column_wrapper{};

  auto const expected_dense   = rank_result_col{};
  auto const expected_rank    = rank_result_col{};
  auto const expected_percent = percent_result_col{};

  test_rank_scans(keys, order_by, expected_dense, expected_rank, expected_percent);
  test_rank_scans(keys, order_by_struct, expected_dense, expected_rank, expected_percent);
}

TYPED_TEST(typed_groupby_rank_scan_test, zero_valid_keys)
{
  using T = TypeParam;

  auto const keys            = input<T>{{X, X, X}, all_nulls()};
  auto const order_by        = input<T>{{3, 3, 1}};
  auto const order_by_struct = [] {
    auto member_1 = input<T>{{3, 3, 1}};
    auto member_2 = input<T>{{3, 3, 1}};
    return cudf::test::structs_column_wrapper{member_1, member_2};
  }();

  auto const dense_rank_results  = rank_result_col{1, 1, 2};
  auto const rank_results        = rank_result_col{1, 1, 3};
  auto const percent_rank_result = percent_result_col{0, 0, 1};

  test_rank_scans(keys, order_by, dense_rank_results, rank_results, percent_rank_result);
  test_rank_scans(keys, order_by_struct, dense_rank_results, rank_results, percent_rank_result);
}

TYPED_TEST(typed_groupby_rank_scan_test, zero_valid_orders)
{
  using T           = TypeParam;
  using null_iter_t = decltype(all_nulls());

  auto const keys                 = input<T>{{1, 1, 3, 3}};
  auto const make_order_by        = [&] { return input<T>{{X, X, X, X}, all_nulls()}; };
  auto const make_struct_order_by = [&](null_iter_t const& null_iter = no_nulls()) {
    auto member1 = make_order_by();
    auto member2 = make_order_by();
    return cudf::test::structs_column_wrapper{{member1, member2}, null_iter};
  };
  auto const order_by                  = make_order_by();
  auto const order_by_struct           = make_struct_order_by();
  auto const order_by_struct_all_nulls = make_struct_order_by(all_nulls());

  auto const expected_dense   = rank_result_col{1, 1, 1, 1};
  auto const expected_rank    = rank_result_col{1, 1, 1, 1};
  auto const expected_percent = percent_result_col{0, 0, 0, 0};

  test_rank_scans(keys, order_by, expected_dense, expected_rank, expected_percent);
  test_rank_scans(keys, order_by_struct, expected_dense, expected_rank, expected_percent);
  test_rank_scans(keys, order_by_struct_all_nulls, expected_dense, expected_rank, expected_percent);
}

TYPED_TEST(typed_groupby_rank_scan_test, basic)
{
  using T = TypeParam;

  auto const keys            = /*        */ input<T>{0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1};
  auto const make_order_by   = [&] { return input<T>{5, 5, 5, 4, 4, 4, 3, 3, 2, 2, 1, 1}; };
  auto const order_by        = make_order_by();
  auto const order_by_struct = [&] {
    auto order2 = make_order_by();
    auto order3 = make_order_by();
    return cudf::test::structs_column_wrapper{order2, order3};
  }();

  auto const expected_dense   = rank_result_col{1, 1, 1, 2, 2, 2, 3, 1, 2, 2, 3, 3};
  auto const expected_rank    = rank_result_col{1, 1, 1, 4, 4, 4, 7, 1, 2, 2, 4, 4};
  auto const expected_percent = percent_result_col{
    0.0, 0.0, 0.0, 3.0 / 6, 3.0 / 6, 3.0 / 6, 6.0 / 6, 0.0, 1.0 / 4, 1.0 / 4, 3.0 / 4, 3.0 / 4};

  test_rank_scans(keys, order_by, expected_dense, expected_rank, expected_percent);
  test_rank_scans(keys, order_by_struct, expected_dense, expected_rank, expected_percent);
}

TYPED_TEST(typed_groupby_rank_scan_test, null_orders)
{
  using T = TypeParam;

  auto const null_mask     = nulls_at({2, 8});
  auto const keys          = input<T>{{0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1}};
  auto const make_order_by = [&] {
    return input<T>{{-1, -2, X, -2, -3, -3, -4, -4, X, -5, -5, -5}, null_mask};
  };
  auto const make_struct_order_by = [&](null_iter_t const& null_iter = all_valid) {
    auto member1 = make_order_by();
    auto member2 = make_order_by();
    return cudf::test::structs_column_wrapper{{member1, member2}, null_iter};
  };
  auto const order_by                   = make_order_by();
  auto const order_by_struct            = make_struct_order_by();
  auto const order_by_struct_with_nulls = make_struct_order_by(null_mask);

  auto const expected_dense   = rank_result_col{1, 2, 3, 4, 5, 5, 1, 1, 2, 3, 3, 3};
  auto const expected_rank    = rank_result_col{1, 2, 3, 4, 5, 5, 1, 1, 3, 4, 4, 4};
  auto const expected_percent = percent_result_col{
    0.0, 1.0 / 5, 2.0 / 5, 3.0 / 5, 4.0 / 5, 4.0 / 5, 0.0, 0.0, 2.0 / 5, 3.0 / 5, 3.0 / 5, 3.0 / 5};

  test_rank_scans(keys, order_by, expected_dense, expected_rank, expected_percent);
  test_rank_scans(keys, order_by_struct, expected_dense, expected_rank, expected_percent);
  test_rank_scans(
    keys, order_by_struct_with_nulls, expected_dense, expected_rank, expected_percent);
}

TYPED_TEST(typed_groupby_rank_scan_test, null_orders_and_keys)
{
  using T = TypeParam;

  auto const null_mask     = nulls_at({2, 8});
  auto const keys          = input<T>{{0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1}, nulls_at({9, 10, 11})};
  auto const make_order_by = [&] {
    return input<T>{{-1, -2, -2, -2, -3, -3, -4, -4, -4, -5, -5, -6}, null_mask};
  };
  auto const make_struct_order_by = [&](null_iter_t const& null_iter = all_valid) {
    auto member1 = make_order_by();
    auto member2 = make_order_by();
    return cudf::test::structs_column_wrapper{{member1, member2}, null_iter};
  };
  auto const order_by                   = make_order_by();
  auto const order_by_struct            = make_struct_order_by();
  auto const order_by_struct_with_nulls = make_struct_order_by(null_mask);

  auto const expected_dense   = rank_result_col{{1, 2, 3, 4, 5, 5, 1, 1, 2, 1, 1, 2}};
  auto const expected_rank    = rank_result_col{{1, 2, 3, 4, 5, 5, 1, 1, 3, 1, 1, 3}};
  auto const expected_percent = percent_result_col{
    {0.0, 1.0 / 5, 2.0 / 5, 3.0 / 5, 4.0 / 5, 4.0 / 5, 0.0, 0.0, 2.0 / 2, 0.0, 0.0, 2.0 / 2}};

  test_rank_scans(keys, order_by, expected_dense, expected_rank, expected_percent);
  test_rank_scans(keys, order_by_struct, expected_dense, expected_rank, expected_percent);
  test_rank_scans(
    keys, order_by_struct_with_nulls, expected_dense, expected_rank, expected_percent);
}

TYPED_TEST(typed_groupby_rank_scan_test, mixedStructs)
{
  auto const struct_col = [] {
    auto nums    = input<TypeParam>{{0, 0, 7, 7, 7, X, 4, 4, 4, 9, 9, 9}, null_at(5)};
    auto strings = cudf::test::strings_column_wrapper{
      {"0a", "0a", "2a", "2a", "3b", "5", "6c", "6c", "XX", "9", "9", "10d"}, null_at(8)};
    return cudf::test::structs_column_wrapper{{nums, strings}, null_at(11)}.release();
  }();

  auto const keys = cudf::test::strings_column_wrapper{
    {"0", "0", "0", "0", "0", "0", "1", "1", "1", "X", "X", "X"}, nulls_at({9, 10, 11})};

  auto const expected_dense   = rank_result_col{1, 1, 2, 2, 3, 4, 1, 1, 2, 1, 1, 2};
  auto const expected_rank    = rank_result_col{1, 1, 3, 3, 5, 6, 1, 1, 3, 1, 1, 3};
  auto const expected_percent = percent_result_col{
    0.0, 0.0, 2.0 / 5, 2.0 / 5, 4.0 / 5, 5.0 / 5, 0.0, 0.0, 2.0 / 2, 0.0, 0.0, 2.0 / 2};

  std::vector<cudf::groupby::scan_request> requests;
  requests.emplace_back();
  requests[0].values = *struct_col;
  requests[0].aggregations.push_back(cudf::make_rank_aggregation<cudf::groupby_scan_aggregation>(
    cudf::rank_method::DENSE, {}, cudf::null_policy::INCLUDE));
  requests[0].aggregations.push_back(cudf::make_rank_aggregation<cudf::groupby_scan_aggregation>(
    cudf::rank_method::MIN, {}, cudf::null_policy::INCLUDE));
  requests[0].aggregations.push_back(cudf::make_rank_aggregation<cudf::groupby_scan_aggregation>(
    cudf::rank_method::MIN,
    {},
    cudf::null_policy::INCLUDE,
    {},
    cudf::rank_percentage::ONE_NORMALIZED));

  cudf::groupby::groupby gb_obj(
    cudf::table_view({keys}), cudf::null_policy::INCLUDE, cudf::sorted::YES);
  auto [result_keys, agg_results] = gb_obj.scan(requests);

  CUDF_TEST_EXPECT_TABLES_EQUAL(cudf::table_view({keys}), result_keys->view());
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*agg_results[0].results[0], expected_dense);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*agg_results[0].results[1], expected_rank);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*agg_results[0].results[2], expected_percent);
}

TYPED_TEST(typed_groupby_rank_scan_test, nestedStructs)
{
  using T = TypeParam;

  auto nested_structs = [] {
    auto structs_member = [] {
      auto nums_member    = input<T>{{0, 0, 7, 7, 7, 5, 4, 4, 4, 9, 9, 9}, null_at(5)};
      auto strings_member = cudf::test::strings_column_wrapper{
        {"0a", "0a", "2a", "2a", "3b", "5", "6c", "6c", "6c", "9", "9", "10d"}, null_at(8)};
      return cudf::test::structs_column_wrapper{nums_member, strings_member};
    }();
    auto nums_member = input<T>{{0, 0, 7, 7, 7, 5, 4, 4, 4, 9, 9, 9}, null_at(5)};
    return cudf::test::structs_column_wrapper{structs_member, nums_member}.release();
  }();

  auto flat_struct = [] {
    auto nums_member    = input<T>{{0, 0, 7, 7, 7, 5, 4, 4, 4, 9, 9, 9}, null_at(5)};
    auto strings_member = cudf::test::strings_column_wrapper{
      {"0a", "0a", "2a", "2a", "3b", "5", "6c", "6c", "6c", "9", "9", "10d"}, null_at(8)};
    auto nuther_nums =
      cudf::test::fixed_width_column_wrapper<T>{{0, 0, 7, 7, 7, 5, 4, 4, 4, 9, 9, 9}, null_at(5)};
    return cudf::test::structs_column_wrapper{nums_member, strings_member, nuther_nums}.release();
  }();

  auto const keys = cudf::test::strings_column_wrapper{
    {"0", "0", "0", "0", "0", "0", "1", "1", "1", "1", "0", "1"}, nulls_at({9, 10, 11})};

  std::vector<cudf::groupby::scan_request> requests;
  requests.emplace_back();
  requests.emplace_back();
  requests[0].values = *nested_structs;
  requests[0].aggregations.push_back(
    cudf::make_rank_aggregation<cudf::groupby_scan_aggregation>(cudf::rank_method::DENSE));
  requests[0].aggregations.push_back(
    cudf::make_rank_aggregation<cudf::groupby_scan_aggregation>(cudf::rank_method::MIN));
  requests[0].aggregations.push_back(cudf::make_rank_aggregation<cudf::groupby_scan_aggregation>(
    cudf::rank_method::MIN,
    {},
    cudf::null_policy::INCLUDE,
    {},
    cudf::rank_percentage::ONE_NORMALIZED));
  requests[1].values = *flat_struct;
  requests[1].aggregations.push_back(
    cudf::make_rank_aggregation<cudf::groupby_scan_aggregation>(cudf::rank_method::DENSE));
  requests[1].aggregations.push_back(
    cudf::make_rank_aggregation<cudf::groupby_scan_aggregation>(cudf::rank_method::MIN));
  requests[1].aggregations.push_back(cudf::make_rank_aggregation<cudf::groupby_scan_aggregation>(
    cudf::rank_method::MIN,
    {},
    cudf::null_policy::INCLUDE,
    {},
    cudf::rank_percentage::ONE_NORMALIZED));

  cudf::groupby::groupby gb_obj(
    cudf::table_view({keys}), cudf::null_policy::INCLUDE, cudf::sorted::YES);
  auto [result_keys, agg_results] = gb_obj.scan(requests);

  CUDF_TEST_EXPECT_TABLES_EQUAL(cudf::table_view({keys}), result_keys->view());
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*agg_results[0].results[0], *agg_results[1].results[0]);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*agg_results[0].results[1], *agg_results[1].results[1]);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*agg_results[0].results[2], *agg_results[1].results[2]);
}

TYPED_TEST(typed_groupby_rank_scan_test, structsWithNullPushdown)
{
  using T = TypeParam;

  auto constexpr num_rows = 12;

  auto get_struct_column = [] {
    auto nums_member =
      cudf::test::fixed_width_column_wrapper<T>{{0, 0, 7, 7, 7, 5, 4, 4, 4, 9, 9, 9}, null_at(5)};
    auto strings_member = cudf::test::strings_column_wrapper{
      {"0a", "0a", "2a", "2a", "3b", "5", "6c", "6c", "6c", "9", "9", "10d"}, null_at(8)};
    auto struct_column = cudf::test::structs_column_wrapper{nums_member, strings_member}.release();
    // Reset null-mask, a posteriori. Nulls will not be pushed down to children.
    auto const null_iter = nulls_at({1, 2, 11});
    auto [null_mask, null_count] =
      cudf::test::detail::make_null_mask(null_iter, null_iter + num_rows);
    struct_column->set_null_mask(std::move(null_mask), null_count);
    return struct_column;
  };

  auto const possibly_null_structs = get_struct_column();

  auto const definitely_null_structs = [&] {
    auto struct_column = get_struct_column();
    struct_column->set_null_mask(cudf::create_null_mask(num_rows, cudf::mask_state::ALL_NULL),
                                 num_rows);
    return struct_column;
  }();

  cudf::test::strings_column_wrapper keys = {
    {"0", "0", "0", "0", "0", "0", "1", "1", "1", "X", "X", "X"}, nulls_at({9, 10, 11})};

  std::vector<cudf::groupby::scan_request> requests;
  requests.emplace_back();
  requests.emplace_back();
  requests[0].values = *possibly_null_structs;
  requests[0].aggregations.push_back(cudf::make_rank_aggregation<cudf::groupby_scan_aggregation>(
    cudf::rank_method::DENSE, {}, cudf::null_policy::INCLUDE));
  requests[0].aggregations.push_back(cudf::make_rank_aggregation<cudf::groupby_scan_aggregation>(
    cudf::rank_method::MIN, {}, cudf::null_policy::INCLUDE));
  requests[0].aggregations.push_back(cudf::make_rank_aggregation<cudf::groupby_scan_aggregation>(
    cudf::rank_method::MIN,
    {},
    cudf::null_policy::INCLUDE,
    {},
    cudf::rank_percentage::ONE_NORMALIZED));
  requests[1].values = *definitely_null_structs;
  requests[1].aggregations.push_back(cudf::make_rank_aggregation<cudf::groupby_scan_aggregation>(
    cudf::rank_method::DENSE, {}, cudf::null_policy::INCLUDE));
  requests[1].aggregations.push_back(cudf::make_rank_aggregation<cudf::groupby_scan_aggregation>(
    cudf::rank_method::MIN, {}, cudf::null_policy::INCLUDE));
  requests[1].aggregations.push_back(cudf::make_rank_aggregation<cudf::groupby_scan_aggregation>(
    cudf::rank_method::MIN,
    {},
    cudf::null_policy::INCLUDE,
    {},
    cudf::rank_percentage::ONE_NORMALIZED));

  cudf::groupby::groupby gb_obj(
    cudf::table_view({keys}), cudf::null_policy::INCLUDE, cudf::sorted::YES);
  auto [result_keys, agg_results] = gb_obj.scan(requests);

  auto expected_dense   = rank_result_col{1, 2, 2, 3, 4, 5, 1, 1, 2, 1, 1, 2};
  auto expected_rank    = rank_result_col{1, 2, 2, 4, 5, 6, 1, 1, 3, 1, 1, 3};
  auto expected_percent = percent_result_col{
    0.0, 1.0 / 5, 1.0 / 5, 3.0 / 5, 4.0 / 5, 5.0 / 5, 0.0, 0.0, 2.0 / 2, 0.0, 0.0, 2.0 / 2};
  auto expected_rank_for_null = rank_result_col{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
  auto expected_percent_for_null =
    percent_result_col{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*agg_results[0].results[0], expected_dense);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*agg_results[0].results[1], expected_rank);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*agg_results[0].results[2], expected_percent);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*agg_results[1].results[0], expected_rank_for_null);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*agg_results[1].results[1], expected_rank_for_null);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*agg_results[1].results[2], expected_percent_for_null);
}

/* List support dependent on https://github.com/NVIDIA/cudf/issues/8683
template <typename T>
struct list_groupby_rank_scan_test : public cudf::test::BaseFixture {
};

using list_test_type_set = Concat<IntegralTypesNotBool,
                                              FloatingPointTypes,
                                              FixedPointTypes>;

TYPED_TEST_SUITE(list_groupby_rank_scan_test, list_test_type_set);

TYPED_TEST(list_groupby_rank_scan_test, lists)
{
  using T = TypeParam;

  auto list_col = lists_column_wrapper<T>{
    {0, 0},
    {0, 0},
    {7, 2},
    {7, 2},
    {7, 3},
    {5, 5},
    {4, 6},
    {4, 6},
    {4, 6},
    {9, 9},
    {9, 9},
    {9, 10}}.release();
  cudf::test::fixed_width_column_wrapper<T> element1{0, 0, 7, 7, 7, 5, 4, 4, 4, 9, 9, 9};
  cudf::test::fixed_width_column_wrapper<T> element2{0, 0, 2, 2, 3, 5, 6, 6, 6, 9, 9, 10};
  auto struct_col =  cudf::test::structs_column_wrapper{element1, element2}.release();

  cudf::test::fixed_width_column_wrapper<T> keys = {{0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1},
                                        {1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0}};

  std::vector<cudf::groupby::scan_request> requests;
  requests.emplace_back(groupby::aggregation_request());
  requests.emplace_back(groupby::aggregation_request());
  requests[0].values = list_col;
  requests[0].aggregations.push_back(make_rank_aggregation<cudf::groupby_scan_aggregation>(rank_method::DENSE));
  requests[0].aggregations.push_back(make_rank_aggregation<cudf::groupby_scan_aggregation>(rank_method::MIN));
  requests[1].values = struct_col;
  requests[1].aggregations.push_back(make_rank_aggregation<cudf::groupby_scan_aggregation>(rank_method::DENSE));
  requests[1].aggregations.push_back(make_rank_aggregation<cudf::groupby_scan_aggregation>(rank_method::MIN));

  cudf::groupby::groupby gb_obj(table_view({keys}), null_policy::INCLUDE, cudf::sorted::YES);
  auto result = gb_obj.scan(requests);

  CUDF_TEST_EXPECT_TABLES_EQUAL(table_view({keys}), result.first->view());
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(
    *result.second[0].results[0], *result.second[1].results[0]);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(
    *result.second[0].results[2], *result.second[1].results[2]);
}
*/

TEST(groupby_rank_scan_test, bools)
{
  using bools       = cudf::test::fixed_width_column_wrapper<bool>;
  using null_iter_t = decltype(nulls_at({}));

  auto const keys          = bools{{0, 0, 0, 0, 0, 0, 1, 1, 1, X, X, X}, nulls_at({9, 10, 11})};
  auto const nulls_6_8     = nulls_at({6, 8});
  auto const make_order_by = [&] { return bools{{0, 0, 1, 1, 1, 1, X, 1, X, 0, 0, 1}, nulls_6_8}; };
  auto const make_structs  = [&](null_iter_t const& null_iter = all_valid) {
    auto member_1 = make_order_by();
    auto member_2 = make_order_by();
    return cudf::test::structs_column_wrapper{{member_1, member_2}, null_iter};
  };

  auto const order_by                    = make_order_by();
  auto const order_by_structs            = make_structs();
  auto const order_by_structs_with_nulls = make_structs(nulls_6_8);

  auto const expected_dense   = rank_result_col{{1, 1, 2, 2, 2, 2, 1, 2, 3, 1, 1, 2}};
  auto const expected_rank    = rank_result_col{{1, 1, 3, 3, 3, 3, 1, 2, 3, 1, 1, 3}};
  auto const expected_percent = percent_result_col{
    {0.0, 0.0, 2.0 / 5, 2.0 / 5, 2.0 / 5, 2.0 / 5, 0.0, 1.0 / 2, 2.0 / 2, 0.0, 0.0, 2.0 / 2}};

  test_rank_scans(keys, order_by, expected_dense, expected_rank, expected_percent);
  test_rank_scans(keys, order_by_structs, expected_dense, expected_rank, expected_percent);
  test_rank_scans(
    keys, order_by_structs_with_nulls, expected_dense, expected_rank, expected_percent);
}

TEST(groupby_rank_scan_test, strings)
{
  using strings     = cudf::test::strings_column_wrapper;
  using null_iter_t = decltype(nulls_at({}));

  auto const keys =
    strings{{"0", "0", "0", "0", "0", "0", "1", "1", "1", "X", "X", "X"}, nulls_at({9, 10, 11})};
  auto const nulls_2_8     = nulls_at({2, 8});
  auto const make_order_by = [&] {
    return strings{{"-1", "-2", "X", "-2", "-3", "-3", "-4", "-4", "X", "-5", "-5", "-6"},
                   nulls_2_8};
  };
  auto const make_structs = [&](null_iter_t const& null_iter = all_valid) {
    auto member_1 = make_order_by();
    auto member_2 = make_order_by();
    return cudf::test::structs_column_wrapper{{member_1, member_2}, null_iter};
  };

  auto const order_by                    = make_order_by();
  auto const order_by_structs            = make_structs();
  auto const order_by_structs_with_nulls = make_structs(nulls_at({4, 5, 11}));

  auto const expected_dense   = rank_result_col{{1, 2, 3, 4, 5, 5, 1, 1, 2, 1, 1, 2}};
  auto const expected_rank    = rank_result_col{{1, 2, 3, 4, 5, 5, 1, 1, 3, 1, 1, 3}};
  auto const expected_percent = percent_result_col{
    {0.0, 1.0 / 5, 2.0 / 5, 3.0 / 5, 4.0 / 5, 4.0 / 5, 0.0, 0.0, 2.0 / 2, 0.0, 0.0, 2.0 / 2}};

  test_rank_scans(keys, order_by, expected_dense, expected_rank, expected_percent);
  test_rank_scans(keys, order_by_structs, expected_dense, expected_rank, expected_percent);
  test_rank_scans(
    keys, order_by_structs_with_nulls, expected_dense, expected_rank, expected_percent);
}

TEST_F(groupby_rank_scan_test_failures, DISABLED_test_exception_triggers)
{
  using T = uint32_t;

  auto const keys = input<T>{{1, 2, 3}, null_at(2)};
  auto const col  = input<T>{3, 3, 1};

  // All of these aggregations raise exceptions unless provided presorted keys
  EXPECT_THROW(test_single_scan(keys,
                                col,
                                keys,
                                col,
                                cudf::make_rank_aggregation<cudf::groupby_scan_aggregation>(
                                  cudf::rank_method::DENSE),
                                cudf::null_policy::INCLUDE,
                                cudf::sorted::NO),
               cudf::logic_error);

  EXPECT_THROW(test_single_scan(keys,
                                col,
                                keys,
                                col,
                                cudf::make_rank_aggregation<cudf::groupby_scan_aggregation>(
                                  cudf::rank_method::MIN),
                                cudf::null_policy::INCLUDE,
                                cudf::sorted::NO),
               cudf::logic_error);

  EXPECT_THROW(test_single_scan(keys,
                                col,
                                keys,
                                col,
                                cudf::make_rank_aggregation<cudf::groupby_scan_aggregation>(
                                  cudf::rank_method::DENSE),
                                cudf::null_policy::EXCLUDE,
                                cudf::sorted::YES),
               cudf::logic_error);

  EXPECT_THROW(test_single_scan(keys,
                                col,
                                keys,
                                col,
                                cudf::make_rank_aggregation<cudf::groupby_scan_aggregation>(
                                  cudf::rank_method::MIN),
                                cudf::null_policy::EXCLUDE,
                                cudf::sorted::YES),
               cudf::logic_error);

  EXPECT_THROW(test_single_scan(keys,
                                col,
                                keys,
                                col,
                                cudf::make_rank_aggregation<cudf::groupby_scan_aggregation>(
                                  cudf::rank_method::DENSE),
                                cudf::null_policy::EXCLUDE,
                                cudf::sorted::NO),
               cudf::logic_error);

  EXPECT_THROW(test_single_scan(keys,
                                col,
                                keys,
                                col,
                                cudf::make_rank_aggregation<cudf::groupby_scan_aggregation>(
                                  cudf::rank_method::MIN),
                                cudf::null_policy::EXCLUDE,
                                cudf::sorted::NO),
               cudf::logic_error);
}
