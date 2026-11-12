/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <tests/groupby/groupby_test_util.hpp>

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/aggregation.hpp>
#include <cudf/copying.hpp>
#include <cudf/groupby.hpp>
#include <cudf/sorting.hpp>
#include <cudf/table/table_view.hpp>

#include <cmath>
#include <limits>
#include <source_location>
#include <string>
#include <vector>

// min/max follow the IEEE 754-2019 minimum/maximum operations for floating-point values:
// NaN propagates and -0 is less than +0. argmin/argmax select the first NaN.

/**
 * @brief Checks the MIN, MAX, ARGMIN, and ARGMAX groupby results for the same input
 */
void test_minmax_aggs(cudf::column_view const& keys,
                      cudf::column_view const& vals,
                      cudf::column_view const& expect_keys,
                      cudf::column_view const& expect_min,
                      cudf::column_view const& expect_max,
                      cudf::column_view const& expect_argmin,
                      cudf::column_view const& expect_argmax,
                      std::source_location const& location = std::source_location::current())
{
  SCOPED_TRACE("Original failure location: " + std::string{location.file_name()} + ":" +
               std::to_string(location.line()));

  auto min_agg    = cudf::make_min_aggregation<cudf::groupby_aggregation>();
  auto max_agg    = cudf::make_max_aggregation<cudf::groupby_aggregation>();
  auto argmin_agg = cudf::make_argmin_aggregation<cudf::groupby_aggregation>();
  auto argmax_agg = cudf::make_argmax_aggregation<cudf::groupby_aggregation>();
  test_single_agg(keys, vals, expect_keys, expect_min, std::move(min_agg));
  test_single_agg(keys, vals, expect_keys, expect_max, std::move(max_agg));

  // streaming groupby does not support ARGMIN/ARGMAX
  auto constexpr null_keys = cudf::null_policy::EXCLUDE;
  auto constexpr unsorted  = cudf::sorted::NO;
  test_single_agg(keys,
                  vals,
                  expect_keys,
                  expect_argmin,
                  std::move(argmin_agg),
                  null_keys,
                  unsorted,
                  {},
                  {},
                  unsorted,
                  test_streaming::NO);
  test_single_agg(keys,
                  vals,
                  expect_keys,
                  expect_argmax,
                  std::move(argmax_agg),
                  null_keys,
                  unsorted,
                  {},
                  {},
                  unsorted,
                  test_streaming::NO);
}

template <typename T>
struct GroupbyMinMaxNaNTest : public cudf::test::BaseFixture {};

TYPED_TEST_SUITE(GroupbyMinMaxNaNTest, cudf::test::FloatingPointTypes);

TYPED_TEST(GroupbyMinMaxNaNTest, MinMaxArgMinArgMax)
{
  using T            = TypeParam;
  auto constexpr NaN = std::numeric_limits<T>::quiet_NaN();
  using keys_col     = cudf::test::fixed_width_column_wrapper<int32_t>;
  using vals_col     = cudf::test::fixed_width_column_wrapper<T>;
  using idx_col      = cudf::test::fixed_width_column_wrapper<cudf::size_type>;

  // groups: [1, NaN, 3], [NaN, 1, 3], [NaN, NaN], [1, 3, 2], [null(NaN), 5]
  auto const keys = keys_col{0, 0, 0, 1, 1, 1, 2, 2, 3, 3, 3, 4, 4};
  auto const vals =
    vals_col{{T{1}, NaN, T{3}, NaN, T{1}, T{3}, NaN, NaN, T{1}, T{3}, T{2}, NaN, T{5}},
             {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1}};
  auto const expect_keys = keys_col{0, 1, 2, 3, 4};

  test_minmax_aggs(keys,
                   vals,
                   expect_keys,
                   vals_col{NaN, NaN, NaN, T{1}, T{5}},
                   vals_col{NaN, NaN, NaN, T{3}, T{5}},
                   idx_col{1, 3, 6, 8, 12},
                   idx_col{1, 3, 6, 9, 12});
}

TYPED_TEST(GroupbyMinMaxNaNTest, Large)
{
  using T            = TypeParam;
  auto constexpr NaN = std::numeric_limits<T>::quiet_NaN();

  // Rows are interleaved across 4 groups so each group is updated from many threads
  cudf::size_type constexpr size = 100'000;
  std::vector<int32_t> keys(size);
  std::vector<T> values(size);
  for (cudf::size_type i = 0; i < size; ++i) {
    keys[i]   = i % 4;
    values[i] = static_cast<T>(i);
  }
  // groups 1 and 2 contain NaNs; groups 0 and 3 do not
  for (auto row : {50'001, 70'002, 90'002}) {
    values[row] = NaN;
  }
  auto const keys_col = cudf::test::fixed_width_column_wrapper<int32_t>(keys.begin(), keys.end());
  auto const vals_col = cudf::test::fixed_width_column_wrapper<T>(values.begin(), values.end());
  auto const expect_keys = cudf::test::fixed_width_column_wrapper<int32_t>{0, 1, 2, 3};
  using vals_wrapper     = cudf::test::fixed_width_column_wrapper<T>;
  using idx_wrapper      = cudf::test::fixed_width_column_wrapper<cudf::size_type>;

  test_minmax_aggs(keys_col,
                   vals_col,
                   expect_keys,
                   vals_wrapper{T{0}, NaN, NaN, T{3}},
                   vals_wrapper{T{99'996}, NaN, NaN, T{99'999}},
                   idx_wrapper{0, 50'001, 70'002, 3},
                   idx_wrapper{99'996, 50'001, 70'002, 99'999});
}

TYPED_TEST(GroupbyMinMaxNaNTest, Sliced)
{
  using T            = TypeParam;
  auto constexpr NaN = std::numeric_limits<T>::quiet_NaN();
  using keys_col     = cudf::test::fixed_width_column_wrapper<int32_t>;
  using vals_col     = cudf::test::fixed_width_column_wrapper<T>;
  using idx_col      = cudf::test::fixed_width_column_wrapper<cudf::size_type>;

  // The NaNs outside the slice belong to group 0 and must not affect it
  auto const keys_all    = keys_col{0, 0, 0, 1, 1, 0};
  auto const vals_all    = vals_col{NaN, T{1}, T{2}, NaN, T{3}, NaN};
  auto const keys        = cudf::slice(keys_all, {1, 5}).front();
  auto const vals        = cudf::slice(vals_all, {1, 5}).front();  // [1, 2, NaN, 3]
  auto const expect_keys = keys_col{0, 1};

  test_minmax_aggs(keys,
                   vals,
                   expect_keys,
                   vals_col{T{1}, NaN},
                   vals_col{T{2}, NaN},
                   idx_col{0, 2},
                   idx_col{1, 2});
}

TYPED_TEST(GroupbyMinMaxNaNTest, SignedZeros)
{
  using T = TypeParam;

  auto const keys = cudf::test::fixed_width_column_wrapper<int32_t>{0, 0, 0, 1, 1, 1};
  auto const vals =
    cudf::test::fixed_width_column_wrapper<T>{T{0}, -T{0}, T{0}, -T{0}, T{0}, -T{0}};

  std::vector<cudf::groupby::aggregation_request> requests(1);
  requests[0].values = vals;
  requests[0].aggregations.push_back(cudf::make_min_aggregation<cudf::groupby_aggregation>());
  requests[0].aggregations.push_back(cudf::make_max_aggregation<cudf::groupby_aggregation>());
  auto gb_obj                       = cudf::groupby::groupby(cudf::table_view({keys}));
  auto const [result_keys, results] = gb_obj.aggregate(requests);

  auto const order  = cudf::sorted_order(result_keys->view());
  auto const sorted = cudf::gather(
    cudf::table_view({results[0].results[0]->view(), results[0].results[1]->view()}), *order);
  auto const mins = cudf::test::to_host<T>(sorted->get_column(0).view()).first;
  auto const maxs = cudf::test::to_host<T>(sorted->get_column(1).view()).first;
  for (std::size_t i = 0; i < mins.size(); ++i) {
    EXPECT_EQ(mins[i], T{0});
    EXPECT_TRUE(std::signbit(mins[i]));
    EXPECT_EQ(maxs[i], T{0});
    EXPECT_FALSE(std::signbit(maxs[i]));
  }
}

TYPED_TEST(GroupbyMinMaxNaNTest, Scan)
{
  using T            = TypeParam;
  auto constexpr NaN = std::numeric_limits<T>::quiet_NaN();
  using keys_col     = cudf::test::fixed_width_column_wrapper<int32_t>;
  using vals_col     = cudf::test::fixed_width_column_wrapper<T>;

  auto const keys        = keys_col{0, 0, 0, 0, 1, 1, 1};
  auto const vals        = vals_col{T{1}, NaN, T{3}, T{0}, T{2}, T{1}, T{3}};
  auto const expect_keys = keys_col{0, 0, 0, 0, 1, 1, 1};

  auto min_agg = cudf::make_min_aggregation<cudf::groupby_scan_aggregation>();
  auto max_agg = cudf::make_max_aggregation<cudf::groupby_scan_aggregation>();
  test_single_scan(
    keys, vals, expect_keys, vals_col{T{1}, NaN, NaN, NaN, T{2}, T{1}, T{1}}, std::move(min_agg));
  test_single_scan(
    keys, vals, expect_keys, vals_col{T{1}, NaN, NaN, NaN, T{2}, T{2}, T{3}}, std::move(max_agg));
}

// With nan_semantics::TOTALLY_ORDERED, NaN is greater than every other value: MIN/ARGMIN ignore NaN
// unless every valid value in the group is NaN, while MAX/ARGMAX match UNORDERED (NaN is the
// maximum).

namespace {

auto constexpr ordered = cudf::nan_semantics::TOTALLY_ORDERED;

/**
 * @brief Checks the TOTALLY_ORDERED MIN, MAX, ARGMIN, and ARGMAX groupby results for the same input
 */
void test_totally_ordered_minmax_aggs(
  cudf::column_view const& keys,
  cudf::column_view const& vals,
  cudf::column_view const& expect_keys,
  cudf::column_view const& expect_min,
  cudf::column_view const& expect_max,
  cudf::column_view const& expect_argmin,
  cudf::column_view const& expect_argmax,
  std::source_location const& location = std::source_location::current())
{
  SCOPED_TRACE("Original failure location: " + std::string{location.file_name()} + ":" +
               std::to_string(location.line()));

  // streaming groupby does not support ARGMIN/ARGMAX
  auto constexpr null_keys = cudf::null_policy::EXCLUDE;
  auto constexpr unsorted  = cudf::sorted::NO;
  auto const test = [&](cudf::column_view const& expect, auto&& agg, test_streaming streaming) {
    test_single_agg(keys,
                    vals,
                    expect_keys,
                    expect,
                    std::move(agg),
                    null_keys,
                    unsorted,
                    {},
                    {},
                    unsorted,
                    streaming);
  };
  auto constexpr streaming    = test_streaming::YES;
  auto constexpr no_streaming = test_streaming::NO;
  test(expect_min, cudf::make_min_aggregation<cudf::groupby_aggregation>(ordered), streaming);
  test(expect_max, cudf::make_max_aggregation<cudf::groupby_aggregation>(ordered), streaming);
  test(
    expect_argmin, cudf::make_argmin_aggregation<cudf::groupby_aggregation>(ordered), no_streaming);
  test(
    expect_argmax, cudf::make_argmax_aggregation<cudf::groupby_aggregation>(ordered), no_streaming);
}

/**
 * @brief Runs a groupby and returns every result column ordered by the (unique) keys
 */
std::vector<std::unique_ptr<cudf::column>> aggregate_by_sorted_keys(
  cudf::column_view const& keys, std::vector<cudf::groupby::aggregation_request> const& requests)
{
  auto gb_obj                       = cudf::groupby::groupby(cudf::table_view({keys}));
  auto const [result_keys, results] = gb_obj.aggregate(requests);
  auto const order                  = cudf::sorted_order(result_keys->view());
  std::vector<std::unique_ptr<cudf::column>> sorted;
  for (auto const& result : results) {
    for (auto const& column : result.results) {
      sorted.push_back(
        std::move(cudf::gather(cudf::table_view({column->view()}), *order)->release().front()));
    }
  }
  return sorted;
}

}  // namespace

TYPED_TEST(GroupbyMinMaxNaNTest, TotallyOrderedMinMaxArgMinArgMax)
{
  using T            = TypeParam;
  auto constexpr NaN = std::numeric_limits<T>::quiet_NaN();
  using keys_col     = cudf::test::fixed_width_column_wrapper<int32_t>;
  using vals_col     = cudf::test::fixed_width_column_wrapper<T>;
  using idx_col      = cudf::test::fixed_width_column_wrapper<cudf::size_type>;

  // groups: [1, NaN, 3], [NaN, 1, 3], [NaN, NaN], [1, 3, 2], [null(NaN), 5], [null(1), NaN],
  //         [inf, NaN]
  auto constexpr inf = std::numeric_limits<T>::infinity();
  auto const keys    = keys_col{0, 0, 0, 1, 1, 1, 2, 2, 3, 3, 3, 4, 4, 5, 5, 6, 6};
  auto const vals    = vals_col{
       {T{1}, NaN, T{3}, NaN, T{1}, T{3}, NaN, NaN, T{1}, T{3}, T{2}, NaN, T{5}, T{1}, NaN, inf, NaN},
       {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1}};
  auto const expect_keys = keys_col{0, 1, 2, 3, 4, 5, 6};

  test_totally_ordered_minmax_aggs(keys,
                                   vals,
                                   expect_keys,
                                   vals_col{T{1}, T{1}, NaN, T{1}, T{5}, NaN, inf},
                                   vals_col{NaN, NaN, NaN, T{3}, T{5}, NaN, NaN},
                                   idx_col{0, 4, 6, 8, 12, 14, 15},
                                   idx_col{1, 3, 6, 9, 12, 14, 16});
}

TYPED_TEST(GroupbyMinMaxNaNTest, TotallyOrderedLarge)
{
  using T            = TypeParam;
  auto constexpr NaN = std::numeric_limits<T>::quiet_NaN();

  // Rows are interleaved across 4 groups so each group is updated from many threads. Group 1 has
  // NaNs before and after its minimum, group 2 is mostly NaN, and group 3 is all NaN.
  cudf::size_type constexpr size = 100'000;
  std::vector<int32_t> keys(size);
  std::vector<T> values(size);
  for (cudf::size_type i = 0; i < size; ++i) {
    keys[i]   = i % 4;
    values[i] = (i % 4 == 3 || (i % 4 == 2 && i != 50'002)) ? NaN : static_cast<T>(i);
  }
  for (auto row : {1, 50'001, 99'997}) {
    values[row] = NaN;
  }
  auto const keys_col = cudf::test::fixed_width_column_wrapper<int32_t>(keys.begin(), keys.end());
  auto const vals_col = cudf::test::fixed_width_column_wrapper<T>(values.begin(), values.end());
  auto const expect_keys = cudf::test::fixed_width_column_wrapper<int32_t>{0, 1, 2, 3};
  using vals_wrapper     = cudf::test::fixed_width_column_wrapper<T>;
  using idx_wrapper      = cudf::test::fixed_width_column_wrapper<cudf::size_type>;

  test_totally_ordered_minmax_aggs(keys_col,
                                   vals_col,
                                   expect_keys,
                                   vals_wrapper{T{0}, T{5}, T{50'002}, NaN},
                                   vals_wrapper{T{99'996}, NaN, NaN, NaN},
                                   idx_wrapper{0, 5, 50'002, 3},
                                   idx_wrapper{99'996, 1, 2, 3});
}

TYPED_TEST(GroupbyMinMaxNaNTest, TotallyOrderedSliced)
{
  using T            = TypeParam;
  auto constexpr NaN = std::numeric_limits<T>::quiet_NaN();
  using keys_col     = cudf::test::fixed_width_column_wrapper<int32_t>;
  using vals_col     = cudf::test::fixed_width_column_wrapper<T>;
  using idx_col      = cudf::test::fixed_width_column_wrapper<cudf::size_type>;

  // The smaller values outside the slice belong to each group and must not affect it
  auto const keys_all    = keys_col{0, 0, 0, 1, 1, 0, 1};
  auto const vals_all    = vals_col{T{-1}, NaN, T{2}, NaN, NaN, T{-2}, T{-3}};
  auto const keys        = cudf::slice(keys_all, {1, 5}).front();
  auto const vals        = cudf::slice(vals_all, {1, 5}).front();  // [NaN, 2, NaN, NaN]
  auto const expect_keys = keys_col{0, 1};

  test_totally_ordered_minmax_aggs(
    keys, vals, expect_keys, vals_col{T{2}, NaN}, vals_col{NaN, NaN}, idx_col{1, 2}, idx_col{0, 2});
}

TYPED_TEST(GroupbyMinMaxNaNTest, TotallyOrderedSignedZeros)
{
  using T            = TypeParam;
  auto constexpr NaN = std::numeric_limits<T>::quiet_NaN();

  auto const keys = cudf::test::fixed_width_column_wrapper<int32_t>{0, 0, 0, 1, 1, 1};
  auto const vals = cudf::test::fixed_width_column_wrapper<T>{T{0}, NaN, -T{0}, -T{0}, T{0}, NaN};

  std::vector<cudf::groupby::aggregation_request> requests(1);
  requests[0].values = vals;
  requests[0].aggregations.push_back(
    cudf::make_min_aggregation<cudf::groupby_aggregation>(ordered));
  requests[0].aggregations.push_back(
    cudf::make_argmin_aggregation<cudf::groupby_aggregation>(ordered));
  auto const results = aggregate_by_sorted_keys(keys, requests);

  auto const mins = cudf::test::to_host<T>(results[0]->view()).first;
  for (auto min : mins) {
    EXPECT_EQ(min, T{0});
    EXPECT_TRUE(std::signbit(min));
  }
  // -0 and +0 are equal values for argmin, so the first zero is selected
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(cudf::test::fixed_width_column_wrapper<cudf::size_type>{0, 3},
                                 results[1]->view());
}

TYPED_TEST(GroupbyMinMaxNaNTest, TotallyOrderedAndUnorderedInOneRequest)
{
  using T            = TypeParam;
  auto constexpr NaN = std::numeric_limits<T>::quiet_NaN();
  using vals_col     = cudf::test::fixed_width_column_wrapper<T>;
  using idx_col      = cudf::test::fixed_width_column_wrapper<cudf::size_type>;

  // Aggregations that differ only in nan_semantics must not share a result
  auto const keys = cudf::test::fixed_width_column_wrapper<int32_t>{0, 0, 0, 1, 1};
  auto const vals = vals_col{T{2}, NaN, T{1}, NaN, T{4}};

  std::vector<cudf::groupby::aggregation_request> requests(1);
  requests[0].values = vals;
  auto& aggs         = requests[0].aggregations;
  aggs.push_back(cudf::make_min_aggregation<cudf::groupby_aggregation>());
  aggs.push_back(cudf::make_min_aggregation<cudf::groupby_aggregation>(ordered));
  aggs.push_back(cudf::make_argmin_aggregation<cudf::groupby_aggregation>(ordered));
  aggs.push_back(cudf::make_argmin_aggregation<cudf::groupby_aggregation>());
  aggs.push_back(
    cudf::make_min_aggregation<cudf::groupby_aggregation>(cudf::nan_semantics::UNORDERED));
  auto const results = aggregate_by_sorted_keys(keys, requests);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(vals_col{NaN, NaN}, results[0]->view());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(vals_col{T{1}, T{4}}, results[1]->view());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(idx_col{2, 4}, results[2]->view());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(idx_col{1, 3}, results[3]->view());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(vals_col{NaN, NaN}, results[4]->view());
}

TYPED_TEST(GroupbyMinMaxNaNTest, TotallyOrderedWithFusedAndBatchedAggregations)
{
  using T            = TypeParam;
  auto constexpr NaN = std::numeric_limits<T>::quiet_NaN();
  using vals_col     = cudf::test::fixed_width_column_wrapper<T>;

  // A SUM, MAX, and MIN on one column are normally fused into one pass, and MINs on columns of
  // the same type are batched. Neither may mix TOTALLY_ORDERED and UNORDERED MIN.
  auto const keys = cudf::test::fixed_width_column_wrapper<int32_t>{0, 0, 0, 1, 1, 1};
  auto const a    = vals_col{T{2}, NaN, T{1}, T{4}, T{3}, T{5}};
  auto const b    = vals_col{T{6}, T{5}, NaN, NaN, NaN, T{7}};
  auto const c    = vals_col{NaN, T{8}, T{9}, T{1}, NaN, T{0}};

  std::vector<cudf::groupby::aggregation_request> requests(3);
  requests[0].values = a;
  requests[0].aggregations.push_back(cudf::make_sum_aggregation<cudf::groupby_aggregation>());
  requests[0].aggregations.push_back(cudf::make_max_aggregation<cudf::groupby_aggregation>());
  requests[0].aggregations.push_back(
    cudf::make_min_aggregation<cudf::groupby_aggregation>(ordered));
  requests[1].values = b;
  requests[1].aggregations.push_back(cudf::make_min_aggregation<cudf::groupby_aggregation>());
  requests[2].values = c;
  requests[2].aggregations.push_back(
    cudf::make_min_aggregation<cudf::groupby_aggregation>(ordered));
  auto const results = aggregate_by_sorted_keys(keys, requests);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(vals_col{NaN, T{12}}, results[0]->view());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(vals_col{NaN, T{5}}, results[1]->view());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(vals_col{T{1}, T{3}}, results[2]->view());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(vals_col{NaN, NaN}, results[3]->view());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(vals_col{T{8}, T{0}}, results[4]->view());
}

TYPED_TEST(GroupbyMinMaxNaNTest, TotallyOrderedDictionary)
{
  using T            = TypeParam;
  auto constexpr NaN = std::numeric_limits<T>::quiet_NaN();

  auto const keys = cudf::test::fixed_width_column_wrapper<int32_t>{0, 0, 0, 1, 1};
  auto const vals =
    cudf::test::dictionary_column_wrapper<T>({NaN, T{3}, T{2}, NaN, T{1}}, {1, 1, 1, 1, 0});

  std::vector<cudf::groupby::aggregation_request> requests(1);
  requests[0].values = vals;
  requests[0].aggregations.push_back(
    cudf::make_min_aggregation<cudf::groupby_aggregation>(ordered));
  requests[0].aggregations.push_back(
    cudf::make_argmin_aggregation<cudf::groupby_aggregation>(ordered));
  auto const results = aggregate_by_sorted_keys(keys, requests);

  // The results carry the null mask of the values even though every group is valid

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(cudf::test::fixed_width_column_wrapper<T>{T{2}, NaN},
                                      results[0]->view());
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(cudf::test::fixed_width_column_wrapper<cudf::size_type>{2, 3},
                                      results[1]->view());
}

TYPED_TEST(GroupbyMinMaxNaNTest, TotallyOrderedScan)
{
  using T            = TypeParam;
  auto constexpr NaN = std::numeric_limits<T>::quiet_NaN();
  using keys_col     = cudf::test::fixed_width_column_wrapper<int32_t>;
  using vals_col     = cudf::test::fixed_width_column_wrapper<T>;

  auto const keys        = keys_col{0, 0, 0, 0, 1, 1, 1};
  auto const vals        = vals_col{{NaN, T{3}, NaN, T{1}, NaN, T{0}, T{2}}, {1, 1, 1, 1, 1, 0, 1}};
  auto const expect_keys = keys_col{0, 0, 0, 0, 1, 1, 1};

  test_single_scan(keys,
                   vals,
                   expect_keys,
                   vals_col{{NaN, T{3}, T{3}, T{1}, NaN, T{0}, T{2}}, {1, 1, 1, 1, 1, 0, 1}},
                   cudf::make_min_aggregation<cudf::groupby_scan_aggregation>(ordered));
  test_single_scan(keys,
                   vals,
                   expect_keys,
                   vals_col{{NaN, NaN, NaN, NaN, NaN, T{0}, NaN}, {1, 1, 1, 1, 1, 0, 1}},
                   cudf::make_max_aggregation<cudf::groupby_scan_aggregation>(ordered));
}
