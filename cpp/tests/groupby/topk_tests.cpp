/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <tests/groupby/groupby_test_util.hpp>

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/iterator_utilities.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/aggregation.hpp>
#include <cudf/copying.hpp>
#include <cudf/groupby.hpp>
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/lists/sorting.hpp>
#include <cudf/sorting.hpp>
#include <cudf/table/table_view.hpp>

#include <cmath>
#include <limits>
#include <memory>
#include <vector>

template <typename V>
struct groupby_topk_test_types : public cudf::test::BaseFixture {};

using FixedWidthTypesNotBool = cudf::test::Concat<cudf::test::IntegralTypesNotBool,
                                                  cudf::test::FloatingPointTypes,
                                                  cudf::test::TimestampTypes,
                                                  cudf::test::FixedPointTypes>;
TYPED_TEST_SUITE(groupby_topk_test_types, FixedWidthTypesNotBool);

TYPED_TEST(groupby_topk_test_types, FixedWidth)
{
  using K = int32_t;
  using V = TypeParam;

  auto keys   = cudf::test::fixed_width_column_wrapper<K, int32_t>{1, 2, 1, 2, 1, 2};
  auto values = cudf::test::fixed_width_column_wrapper<V, int32_t>{1, 4, 2, 5, 3, 6};

  auto expect_keys = cudf::test::fixed_width_column_wrapper<K, int32_t>{1, 2};

  auto expect_vals = cudf::test::lists_column_wrapper<V, int32_t>{{3, 2, 1}, {6, 5, 4}};
  auto agg         = cudf::make_top_k_aggregation<cudf::groupby_aggregation>(3);
  test_single_agg(keys, values, expect_keys, expect_vals, std::move(agg));

  expect_vals = cudf::test::lists_column_wrapper<V, int32_t>{{1, 2, 3}, {4, 5, 6}};
  agg         = cudf::make_top_k_aggregation<cudf::groupby_aggregation>(4, cudf::order::ASCENDING);
  test_single_agg(keys, values, expect_keys, expect_vals, std::move(agg));
}

struct groupby_topk_test : public cudf::test::BaseFixture {};

TEST_F(groupby_topk_test, WithNulls)
{
  // clang-format off
  auto keys   = cudf::test::fixed_width_column_wrapper<int32_t>({1, 2, 1, 2, 1, 2, 3, 3, 2, 1, 3, 3});
  auto values = cudf::test::fixed_width_column_wrapper<int32_t>({1, 4, 2, 5, 3, 6, 7, 8, 0, 9, 0, 10},
    cudf::test::iterators::nulls_at({8,10}));
  // clang-format on

  auto expect_keys = cudf::test::fixed_width_column_wrapper<int32_t>({1, 2, 3});

  auto expect_vals = cudf::test::lists_column_wrapper<int32_t>({{9, 3, 2}, {6, 5, 4}, {10, 8, 7}});
  auto agg         = cudf::make_top_k_aggregation<cudf::groupby_aggregation>(3);
  test_single_agg(keys, values, expect_keys, expect_vals, std::move(agg));

  agg         = cudf::make_top_k_aggregation<cudf::groupby_aggregation>(2, cudf::order::ASCENDING);
  expect_vals = cudf::test::lists_column_wrapper<int32_t>({{1, 2}, {4, 5}, {7, 8}});
  test_single_agg(keys, values, expect_keys, expect_vals, std::move(agg));
}

TEST_F(groupby_topk_test, KGreaterThanGroupSize)
{
  auto keys   = cudf::test::fixed_width_column_wrapper<int32_t>({1, 2, 1, 2, 1, 2, 1, 2, 1});
  auto values = cudf::test::fixed_width_column_wrapper<int32_t>({1, 4, 2, 5, 3, 6, 7, 8, 9});

  auto expect_keys = cudf::test::fixed_width_column_wrapper<int32_t>({1, 2});

  auto expect_vals = cudf::test::lists_column_wrapper<int32_t>({{9, 7, 3, 2, 1}, {8, 6, 5, 4}});
  auto agg         = cudf::make_top_k_aggregation<cudf::groupby_aggregation>(6);
  test_single_agg(keys, values, expect_keys, expect_vals, std::move(agg));
  agg = cudf::make_top_k_aggregation<cudf::groupby_aggregation>(5);
  test_single_agg(keys, values, expect_keys, expect_vals, std::move(agg));

  agg         = cudf::make_top_k_aggregation<cudf::groupby_aggregation>(5, cudf::order::ASCENDING);
  expect_vals = cudf::test::lists_column_wrapper<int32_t>({{1, 2, 3, 7, 9}, {4, 5, 6, 8}});
  test_single_agg(keys, values, expect_keys, expect_vals, std::move(agg));
}

TEST_F(groupby_topk_test, Strings)
{
  auto keys   = cudf::test::fixed_width_column_wrapper<int32_t>({1, 2, 1, 2, 1, 2, 1, 2, 1});
  auto values = cudf::test::strings_column_wrapper({"a", "d", "b", "e", "c", "f", "g", "h", "i"});

  auto expect_keys = cudf::test::fixed_width_column_wrapper<int32_t>({1, 2});

  auto expect_vals = cudf::test::lists_column_wrapper<cudf::string_view>({{"i", "g"}, {"h", "f"}});
  auto agg         = cudf::make_top_k_aggregation<cudf::groupby_aggregation>(2);
  test_single_agg(keys, values, expect_keys, expect_vals, std::move(agg));

  agg         = cudf::make_top_k_aggregation<cudf::groupby_aggregation>(5, cudf::order::ASCENDING);
  expect_vals = cudf::test::lists_column_wrapper<cudf::string_view>(
    {{"a", "b", "c", "g", "i"}, {"d", "e", "f", "h"}});
  test_single_agg(keys, values, expect_keys, expect_vals, std::move(agg));
}

TEST_F(groupby_topk_test, EmptyInput)
{
  cudf::test::fixed_width_column_wrapper<int32_t> keys{};
  cudf::test::fixed_width_column_wrapper<int32_t> values{};

  cudf::test::fixed_width_column_wrapper<int32_t> expect_keys{};
  cudf::test::lists_column_wrapper<int32_t> expect_vals{};

  auto agg = cudf::make_top_k_aggregation<cudf::groupby_aggregation>(3);
  test_single_agg(keys, values, expect_keys, expect_vals, std::move(agg));
}

namespace {
// Returns the TOP_K result of each group ordered by key, with each list sorted, since TOP_K
// returns a group's values in no particular order
std::unique_ptr<cudf::column> sorted_top_k(cudf::column_view const& keys,
                                           cudf::column_view const& values,
                                           cudf::size_type k,
                                           cudf::order topk_order)
{
  auto gb            = cudf::groupby::groupby(cudf::table_view({keys}));
  auto requests      = std::vector<cudf::groupby::aggregation_request>(1);
  requests[0].values = values;
  requests[0].aggregations.push_back(
    cudf::make_top_k_aggregation<cudf::groupby_aggregation>(k, topk_order));
  auto const [result_keys, results] = gb.aggregate(requests);
  auto const by_key                 = cudf::sorted_order(result_keys->view());
  auto const gathered = cudf::gather(cudf::table_view({results[0].results[0]->view()}), *by_key);
  return cudf::lists::sort_lists(cudf::lists_column_view(gathered->get_column(0).view()),
                                 cudf::order::ASCENDING,
                                 cudf::null_order::AFTER);
}
}  // namespace

template <typename V>
struct groupby_topk_floating_point_test : public cudf::test::BaseFixture {};

TYPED_TEST_SUITE(groupby_topk_floating_point_test, cudf::test::FloatingPointTypes);

TYPED_TEST(groupby_topk_floating_point_test, NaNs)
{
  using T = TypeParam;

  auto const nan     = std::numeric_limits<T>::quiet_NaN();
  auto const neg_nan = std::copysign(nan, T{-1});
  auto const inf     = std::numeric_limits<T>::infinity();

  // NaNs compare equal to each other and greater than every other value, whatever their sign, and
  // nulls are ordered last in either direction. Row 6 is null.
  // clang-format off
  auto keys   = cudf::test::fixed_width_column_wrapper<int32_t>({1, 2, 1, 2, 1, 2, 3, 3, 3, 1, 3});
  auto values = cudf::test::fixed_width_column_wrapper<T>(
    {T{2}, nan, nan, -inf, T{1}, inf, T{0}, nan, T{4}, neg_nan, neg_nan},
    {1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1});
  // clang-format on

  // The lists are sorted ascending, which places NaN last
  auto expected =
    cudf::test::lists_column_wrapper<T>{{T{2}, nan, nan}, {-inf, inf, nan}, {T{4}, nan, nan}};
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(
    expected, sorted_top_k(keys, values, 3, cudf::order::DESCENDING)->view());

  expected =
    cudf::test::lists_column_wrapper<T>{{T{1}, T{2}, nan}, {-inf, inf, nan}, {T{4}, nan, nan}};
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(
    expected, sorted_top_k(keys, values, 3, cudf::order::ASCENDING)->view());
}
