/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <cudf/sorting.hpp>
#include <cudf/table/table_view.hpp>

using namespace cudf::test::iterators;

namespace {
// Run SUM aggregation through hash, sort, AND streaming groupby paths.
void test_sum_all_paths(cudf::column_view const& keys,
                        cudf::column_view const& values,
                        cudf::column_view const& expect_keys,
                        cudf::column_view const& expect_vals,
                        std::source_location const& loc = std::source_location::current())
{
  test_single_agg(keys,
                  values,
                  expect_keys,
                  expect_vals,
                  cudf::make_sum_aggregation<cudf::groupby_aggregation>(),
                  force_use_sort_impl::NO,
                  cudf::null_policy::EXCLUDE,
                  cudf::sorted::NO,
                  {},
                  {},
                  cudf::sorted::NO,
                  test_streaming::YES,
                  loc);
  test_single_agg(keys,
                  values,
                  expect_keys,
                  expect_vals,
                  cudf::make_sum_aggregation<cudf::groupby_aggregation>(),
                  force_use_sort_impl::YES,
                  cudf::null_policy::EXCLUDE,
                  cudf::sorted::NO,
                  {},
                  {},
                  cudf::sorted::NO,
                  test_streaming::NO,
                  loc);
}
}  // namespace

template <typename V>
struct groupby_sum_test : public cudf::test::BaseFixture {};

using K = int32_t;
using supported_types =
  cudf::test::Concat<cudf::test::Types<int8_t, int16_t, int32_t, int64_t, float, double>,
                     cudf::test::DurationTypes>;

TYPED_TEST_SUITE(groupby_sum_test, supported_types);

TYPED_TEST(groupby_sum_test, basic)
{
  using V = TypeParam;
  using R = std::conditional_t<std::is_integral_v<V>, int64_t, V>;

  cudf::test::fixed_width_column_wrapper<K> keys{1, 2, 3, 1, 2, 2, 1, 3, 3, 2};
  cudf::test::fixed_width_column_wrapper<V> vals{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};

  cudf::test::fixed_width_column_wrapper<K> expect_keys{1, 2, 3};
  cudf::test::fixed_width_column_wrapper<R> expect_vals{9, 19, 17};

  test_sum_all_paths(keys, vals, expect_keys, expect_vals);
}

TYPED_TEST(groupby_sum_test, empty_cols)
{
  using V = TypeParam;
  using R = std::conditional_t<std::is_integral_v<V>, int64_t, V>;

  cudf::test::fixed_width_column_wrapper<K> keys{};
  cudf::test::fixed_width_column_wrapper<V> vals{};

  cudf::test::fixed_width_column_wrapper<K> expect_keys{};
  cudf::test::fixed_width_column_wrapper<R> expect_vals{};

  test_sum_all_paths(keys, vals, expect_keys, expect_vals);
}

TYPED_TEST(groupby_sum_test, zero_valid_keys)
{
  using V = TypeParam;
  using R = std::conditional_t<std::is_integral_v<V>, int64_t, V>;

  cudf::test::fixed_width_column_wrapper<K> keys({1, 2, 3}, cudf::test::iterators::all_nulls());
  cudf::test::fixed_width_column_wrapper<V> vals{3, 4, 5};

  cudf::test::fixed_width_column_wrapper<K> expect_keys{};
  cudf::test::fixed_width_column_wrapper<R> expect_vals{};

  test_sum_all_paths(keys, vals, expect_keys, expect_vals);
}

TYPED_TEST(groupby_sum_test, zero_valid_values)
{
  using V = TypeParam;
  using R = std::conditional_t<std::is_integral_v<V>, int64_t, V>;

  cudf::test::fixed_width_column_wrapper<K> keys{1, 1, 1};
  cudf::test::fixed_width_column_wrapper<V> vals({3, 4, 5}, cudf::test::iterators::all_nulls());

  cudf::test::fixed_width_column_wrapper<K> expect_keys{1};
  cudf::test::fixed_width_column_wrapper<R> expect_vals({0}, cudf::test::iterators::all_nulls());

  test_sum_all_paths(keys, vals, expect_keys, expect_vals);
}

TYPED_TEST(groupby_sum_test, null_keys_and_values)
{
  using V = TypeParam;
  using R = std::conditional_t<std::is_integral_v<V>, int64_t, V>;

  cudf::test::fixed_width_column_wrapper<K> keys(
    {1, 2, 3, 1, 2, 2, 1, 3, 3, 2, 4},
    {true, true, true, true, true, true, true, false, true, true, true});
  cudf::test::fixed_width_column_wrapper<V> vals({0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 4},
                                                 {0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0});

  //  { 1, 1,     2, 2, 2,   3, 3,    4}
  cudf::test::fixed_width_column_wrapper<K> expect_keys({1, 2, 3, 4},
                                                        cudf::test::iterators::no_nulls());
  //  { 3, 6,     1, 4, 9,   2, 8,    -}
  cudf::test::fixed_width_column_wrapper<R> expect_vals({9, 14, 10, 0}, {1, 1, 1, 0});

  test_sum_all_paths(keys, vals, expect_keys, expect_vals);
}

// streaming_groupby does not accept dictionary-typed value columns, so this case
// runs only the stateless hash and sort paths via test_single_agg.
TYPED_TEST(groupby_sum_test, dictionary)
{
  using V = TypeParam;
  using R = std::conditional_t<std::is_integral_v<V>, int64_t, V>;

  cudf::test::fixed_width_column_wrapper<K> keys{1, 2, 3, 1, 2, 2, 1, 3, 3, 2};
  cudf::test::dictionary_column_wrapper<V> vals{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  cudf::test::fixed_width_column_wrapper<K> expect_keys{1, 2, 3};
  cudf::test::fixed_width_column_wrapper<R> expect_vals{9, 19, 17};

  test_single_agg(
    keys, vals, expect_keys, expect_vals, cudf::make_sum_aggregation<cudf::groupby_aggregation>());
  test_single_agg(keys,
                  vals,
                  expect_keys,
                  expect_vals,
                  cudf::make_sum_aggregation<cudf::groupby_aggregation>(),
                  force_use_sort_impl::YES);
}

struct overflow_test : public cudf::test::BaseFixture {};
TEST_F(overflow_test, overflow_integer)
{
  using int32_col = cudf::test::fixed_width_column_wrapper<int32_t>;
  using int64_col = cudf::test::fixed_width_column_wrapper<int64_t>;

  auto const keys        = int32_col{0, 0};
  auto const vals        = int32_col{-2147483648, -2147483648};
  auto const expect_keys = int32_col{0};
  auto const expect_vals = int64_col{-4294967296L};

  auto test_sum = [&](auto const use_sort) {
    auto agg = cudf::make_sum_aggregation<cudf::groupby_aggregation>();
    test_single_agg(keys, vals, expect_keys, expect_vals, std::move(agg), use_sort);
  };

  test_sum(force_use_sort_impl::NO);
  test_sum(force_use_sort_impl::YES);
}

template <typename T>
struct GroupBySumFixedPointTest : public cudf::test::BaseFixture {};

TYPED_TEST_SUITE(GroupBySumFixedPointTest, cudf::test::FixedPointTypes);

TYPED_TEST(GroupBySumFixedPointTest, GroupBySortSumDecimalAsValue)
{
  using namespace numeric;
  using decimalXX  = TypeParam;
  using RepType    = cudf::device_storage_type_t<decimalXX>;
  using fp_wrapper = cudf::test::fixed_point_column_wrapper<RepType>;
  using K          = int32_t;

  for (auto const i : {2, 1, 0, -1, -2}) {
    auto const scale = scale_type{i};
    // clang-format off
    auto const keys  = cudf::test::fixed_width_column_wrapper<K>{1, 2, 3, 1, 2, 2, 1, 3, 3, 2};
    auto const vals  = fp_wrapper{                              {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}, scale};
    // clang-format on

    auto const expect_keys     = cudf::test::fixed_width_column_wrapper<K>{1, 2, 3};
    auto const expect_vals_sum = fp_wrapper{{9, 19, 17}, scale};

    auto agg1 = cudf::make_sum_aggregation<cudf::groupby_aggregation>();
    test_single_agg(
      keys, vals, expect_keys, expect_vals_sum, std::move(agg1), force_use_sort_impl::YES);

    auto agg4 = cudf::make_product_aggregation<cudf::groupby_aggregation>();
    EXPECT_THROW(
      test_single_agg(keys, vals, expect_keys, {}, std::move(agg4), force_use_sort_impl::YES),
      cudf::logic_error);
  }
}

TYPED_TEST(GroupBySumFixedPointTest, GroupByHashSumDecimalAsValue)
{
  using namespace numeric;
  using decimalXX  = TypeParam;
  using RepType    = cudf::device_storage_type_t<decimalXX>;
  using fp_wrapper = cudf::test::fixed_point_column_wrapper<RepType>;
  using K          = int32_t;

  for (auto const i : {2, 1, 0, -1, -2}) {
    auto const scale = scale_type{i};
    // clang-format off
    auto const keys  = cudf::test::fixed_width_column_wrapper<K>{1, 2, 3, 1, 2, 2, 1, 3, 3, 2};
    auto const vals  = fp_wrapper{                              {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}, scale};
    // clang-format on

    auto const expect_keys     = cudf::test::fixed_width_column_wrapper<K>{1, 2, 3};
    auto const expect_vals_sum = fp_wrapper{{9, 19, 17}, scale};

    auto agg5 = cudf::make_sum_aggregation<cudf::groupby_aggregation>();
    test_single_agg(keys, vals, expect_keys, expect_vals_sum, std::move(agg5));

    auto agg6 = cudf::make_sum_aggregation<cudf::groupby_aggregation>();
    test_single_agg(
      keys, vals, expect_keys, expect_vals_sum, std::move(agg6), force_use_sort_impl::NO);

    auto agg8 = cudf::make_product_aggregation<cudf::groupby_aggregation>();
    EXPECT_THROW(test_single_agg(keys, vals, expect_keys, {}, std::move(agg8)), cudf::logic_error);
  }
}

struct GroupByDecimal128ShmemAlignmentTest : public cudf::test::BaseFixture {};

TEST_F(GroupByDecimal128ShmemAlignmentTest, Decimal128SumAfterInt32Sum)
{
  using namespace numeric;
  using fp128 = cudf::test::fixed_point_column_wrapper<__int128_t>;

  auto const scale = scale_type{0};

  // 11 unique keys → num_agg_locations = 22, valid_col_size = round_up_safe(22, 8) = 24.
  // INT32 SUM output is INT64 (8 bytes): data = 176, total = 176 + 24 = 200.
  // 200 % 16 = 8 → DECIMAL128 slot starts misaligned for 16-byte shared-memory access.
  // clang-format off
  auto const keys = cudf::test::fixed_width_column_wrapper<int32_t>{
    0,1,2,3,4,5,6,7,8,9,10, 0,1,2,3,4,5,6,7,8,9,10};
  auto const i32  = cudf::test::fixed_width_column_wrapper<int32_t>{
    1,1,1,1,1,1,1,1,1,1,1,  1,1,1,1,1,1,1,1,1,1,1};
  auto const d128 = fp128{
    {1,1,1,1,1,1,1,1,1,1,1, 1,1,1,1,1,1,1,1,1,1,1}, scale};
  // clang-format on

  std::vector<cudf::groupby::aggregation_request> requests(2);
  requests[0].values = i32;
  requests[0].aggregations.push_back(cudf::make_sum_aggregation<cudf::groupby_aggregation>());
  requests[1].values = d128;
  requests[1].aggregations.push_back(cudf::make_sum_aggregation<cudf::groupby_aggregation>());

  cudf::groupby::groupby gb(cudf::table_view({keys}));
  auto [result_keys, results] = gb.aggregate(requests);

  auto const sort_order = cudf::sorted_order(result_keys->view());
  auto const sorted_result =
    cudf::gather(cudf::table_view({results[1].results[0]->view()}), *sort_order);
  auto const expected = fp128{{2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2}, scale};
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected, sorted_result->get_column(0));
}

// Regression test for https://github.com/rapidsai/cudf/issues/23150.
// Blackwell returned incorrect sums when aggregating three DECIMAL128 columns.
TEST_F(GroupByDecimal128ShmemAlignmentTest, MultiColumnDecimal128Sum)
{
  using namespace numeric;
  using fp128 = cudf::test::fixed_point_column_wrapper<__int128_t>;

  constexpr int num_cols             = 3;
  constexpr cudf::size_type num_rows = 1'000'000;
  constexpr int num_groups           = 4;
  auto const scale                   = scale_type{-2};

  // A large base makes each group's running sum cross the 2^64 low-word boundary, exercising
  // carry propagation in the fallback, and every fifth row is negated to exercise borrow.
  constexpr __int128_t base = static_cast<__int128_t>(1) << 50;

  std::vector<int32_t> keys_data(num_rows);
  std::vector<std::vector<__int128_t>> vals_data(num_cols, std::vector<__int128_t>(num_rows));
  std::vector<std::vector<__int128_t>> sums(num_cols, std::vector<__int128_t>(num_groups, 0));
  for (cudf::size_type i = 0; i < num_rows; ++i) {
    auto const k = i % num_groups;
    keys_data[i] = k;
    for (int c = 0; c < num_cols; ++c) {
      auto v = base + static_cast<__int128_t>((100 + i % 7) * 100 + (13 * c + i) % 100);
      if (i % 5 == 0) { v = -v; }
      vals_data[c][i] = v;
      sums[c][k] += v;
    }
  }

  auto const keys =
    cudf::test::fixed_width_column_wrapper<int32_t>(keys_data.begin(), keys_data.end());
  std::vector<fp128> vals;
  std::vector<cudf::groupby::aggregation_request> requests(num_cols);
  for (int c = 0; c < num_cols; ++c) {
    vals.emplace_back(vals_data[c].begin(), vals_data[c].end(), scale);
    requests[c].values = vals[c];
    requests[c].aggregations.push_back(cudf::make_sum_aggregation<cudf::groupby_aggregation>());
  }

  cudf::groupby::groupby gb(cudf::table_view({keys}));
  auto [result_keys, results] = gb.aggregate(requests);

  auto const sort_order = cudf::sorted_order(result_keys->view());
  for (int c = 0; c < num_cols; ++c) {
    auto const sorted =
      cudf::gather(cudf::table_view({results[c].results[0]->view()}), *sort_order);
    auto const expected = fp128(sums[c].begin(), sums[c].end(), scale);
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected, sorted->get_column(0));
  }
}
