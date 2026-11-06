/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <tests/groupby/groupby_test_util.hpp>

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/iterator_utilities.hpp>
#include <cudf_test/type_list_utilities.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/aggregation.hpp>
#include <cudf/copying.hpp>
#include <cudf/groupby.hpp>
#include <cudf/reduction.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/sorting.hpp>
#include <cudf/table/table_view.hpp>

#include <cstdint>
#include <initializer_list>
#include <limits>
#include <vector>

using namespace cudf::test::iterators;

template <typename V>
struct groupby_mean_test : public cudf::test::BaseFixture {};

template <typename Target, typename Source>
std::vector<Target> convert(std::initializer_list<Source> in)
{
  std::vector<Target> out(std::cbegin(in), std::cend(in));
  return out;
}

using supported_types =
  cudf::test::Concat<cudf::test::Types<int8_t, int16_t, int32_t, int64_t, float, double>,
                     cudf::test::DurationTypes>;
TYPED_TEST_SUITE(groupby_mean_test, supported_types);
using K = int32_t;

TYPED_TEST(groupby_mean_test, basic)
{
  using V  = TypeParam;
  using R  = std::conditional_t<cudf::is_duration<V>(), V, double>;
  using RT = std::conditional_t<cudf::is_duration<R>(), int, double>;

  cudf::test::fixed_width_column_wrapper<K> keys{1, 2, 3, 1, 2, 2, 1, 3, 3, 2};
  cudf::test::fixed_width_column_wrapper<V> vals{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};

  // clang-format off
  cudf::test::fixed_width_column_wrapper<K> expect_keys{1,       2,          3};
  //                                       {0, 3, 6, 1, 4, 5, 9, 2, 7, 8}
  std::vector<RT> expect_v = convert<RT>(  {3.,      19. / 4,    17. / 3});
  cudf::test::fixed_width_column_wrapper<R, RT> expect_vals(expect_v.cbegin(), expect_v.cend());
  // clang-format on

  auto agg = cudf::make_mean_aggregation<cudf::groupby_aggregation>();
  test_single_agg(keys, vals, expect_keys, expect_vals, std::move(agg));
}

TYPED_TEST(groupby_mean_test, empty_cols)
{
  using V = TypeParam;
  using R = std::conditional_t<cudf::is_duration<V>(), V, double>;

  cudf::test::fixed_width_column_wrapper<K> keys{};
  cudf::test::fixed_width_column_wrapper<V> vals{};

  cudf::test::fixed_width_column_wrapper<K> expect_keys{};
  cudf::test::fixed_width_column_wrapper<R> expect_vals{};

  auto agg = cudf::make_mean_aggregation<cudf::groupby_aggregation>();
  test_single_agg(keys, vals, expect_keys, expect_vals, std::move(agg));
}

TYPED_TEST(groupby_mean_test, zero_valid_keys)
{
  using V = TypeParam;
  using R = std::conditional_t<cudf::is_duration<V>(), V, double>;

  cudf::test::fixed_width_column_wrapper<K> keys({1, 2, 3}, all_nulls());
  cudf::test::fixed_width_column_wrapper<V> vals{3, 4, 5};

  cudf::test::fixed_width_column_wrapper<K> expect_keys{};
  cudf::test::fixed_width_column_wrapper<R> expect_vals{};

  auto agg = cudf::make_mean_aggregation<cudf::groupby_aggregation>();
  test_single_agg(keys, vals, expect_keys, expect_vals, std::move(agg));
}

TYPED_TEST(groupby_mean_test, zero_valid_values)
{
  using V = TypeParam;
  using R = std::conditional_t<cudf::is_duration<V>(), V, double>;

  cudf::test::fixed_width_column_wrapper<K> keys{1, 1, 1};
  cudf::test::fixed_width_column_wrapper<V> vals({3, 4, 5}, all_nulls());

  cudf::test::fixed_width_column_wrapper<K> expect_keys{1};
  cudf::test::fixed_width_column_wrapper<R> expect_vals({0}, all_nulls());

  auto agg = cudf::make_mean_aggregation<cudf::groupby_aggregation>();
  test_single_agg(keys, vals, expect_keys, expect_vals, std::move(agg));
}

TYPED_TEST(groupby_mean_test, null_keys_and_values)
{
  using V  = TypeParam;
  using R  = std::conditional_t<cudf::is_duration<V>(), V, double>;
  using RT = std::conditional_t<cudf::is_duration<R>(), int, double>;

  cudf::test::fixed_width_column_wrapper<K> keys(
    {1, 2, 3, 1, 2, 2, 1, 3, 3, 2, 4},
    {true, true, true, true, true, true, true, false, true, true, true});
  cudf::test::fixed_width_column_wrapper<V> vals({0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 4},
                                                 {0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0});

  // clang-format off
  //                                                    {1, 1,     2, 2, 2,   3, 3,    4}
  cudf::test::fixed_width_column_wrapper<K> expect_keys({1,        2,         3,       4}, no_nulls());
  //                                                    {3, 6,     1, 4, 9,   2, 8,    -}
  std::vector<RT> expect_v = convert<RT>(               {4.5,      14. / 3,   5.,      0.});
  // clang-format on
  cudf::test::fixed_width_column_wrapper<R, RT> expect_vals(
    expect_v.cbegin(), expect_v.cend(), {1, 1, 1, 0});

  auto agg = cudf::make_mean_aggregation<cudf::groupby_aggregation>();
  test_single_agg(keys, vals, expect_keys, expect_vals, std::move(agg));
}
// clang-format on

struct groupby_dictionary_mean_test : public cudf::test::BaseFixture {};

TEST_F(groupby_dictionary_mean_test, basic)
{
  using V = int16_t;
  using R = double;

  // clang-format off
  cudf::test::fixed_width_column_wrapper<K> keys{1, 2, 3, 1, 2, 2, 1, 3, 3, 2};
  cudf::test::dictionary_column_wrapper<V>  vals{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};

  cudf::test::fixed_width_column_wrapper<K> expect_keys(        {1,      2,       3});
  cudf::test::fixed_width_column_wrapper<R, double> expect_vals({9. / 3, 19. / 4, 17. / 3});
  // clang-format on

  test_single_agg(
    keys, vals, expect_keys, expect_vals, cudf::make_mean_aggregation<cudf::groupby_aggregation>());
}

template <typename T>
struct FixedPointTestBothReps : public cudf::test::BaseFixture {};

TYPED_TEST_SUITE(FixedPointTestBothReps, cudf::test::FixedPointTypes);

TYPED_TEST(FixedPointTestBothReps, GroupByHashMeanDecimalAsValue)
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
    auto const expect_vals_min = fp_wrapper{{3, 4, 5}, scale};

    auto agg = cudf::make_mean_aggregation<cudf::groupby_aggregation>();
    test_single_agg(keys, vals, expect_keys, expect_vals_min, std::move(agg));
  }
}

// MEAN of integers accumulates in its double output type, like reduction MEAN, so a group whose
// int64 SUM overflows still has the correct mean. SUM itself keeps its int64 accumulator.
struct groupby_mean_overflow_test : public cudf::test::BaseFixture {};

namespace {

/// Sorts the groupby results by key so that they can be compared with the expected columns.
std::unique_ptr<cudf::column> by_key(cudf::table_view const& keys, cudf::column_view const& col)
{
  auto const order = cudf::sorted_order(keys);
  return std::move(cudf::gather(cudf::table_view{{col}}, *order)->release().front());
}

}  // namespace

TEST_F(groupby_mean_overflow_test, int64)
{
  auto constexpr max = std::numeric_limits<int64_t>::max();
  auto constexpr min = std::numeric_limits<int64_t>::min();

  cudf::test::fixed_width_column_wrapper<K> keys{0, 0, 1, 1, 1};
  cudf::test::fixed_width_column_wrapper<int64_t> vals{max, max, min, min, min};

  cudf::test::fixed_width_column_wrapper<K> expect_keys{0, 1};
  cudf::test::fixed_width_column_wrapper<double> expect_vals{static_cast<double>(max),
                                                             static_cast<double>(min)};

  test_single_agg(
    keys, vals, expect_keys, expect_vals, cudf::make_mean_aggregation<cudf::groupby_aggregation>());
}

TEST_F(groupby_mean_overflow_test, uint64)
{
  auto constexpr max = std::numeric_limits<uint64_t>::max();

  cudf::test::fixed_width_column_wrapper<K> keys{0, 0, 1};
  cudf::test::fixed_width_column_wrapper<uint64_t> vals{max, max, uint64_t{7}};

  cudf::test::fixed_width_column_wrapper<K> expect_keys{0, 1};
  cudf::test::fixed_width_column_wrapper<double> expect_vals{static_cast<double>(max), 7.};

  test_single_agg(
    keys, vals, expect_keys, expect_vals, cudf::make_mean_aggregation<cudf::groupby_aggregation>());
}

TEST_F(groupby_mean_overflow_test, int64_with_nulls)
{
  auto constexpr max = std::numeric_limits<int64_t>::max();

  cudf::test::fixed_width_column_wrapper<K> keys{0, 0, 0, 1, 1};
  cudf::test::fixed_width_column_wrapper<int64_t> vals{
    {max, max, int64_t{5}, int64_t{7}, int64_t{7}}, {1, 1, 0, 0, 0}};

  cudf::test::fixed_width_column_wrapper<K> expect_keys{0, 1};
  cudf::test::fixed_width_column_wrapper<double> expect_vals{{static_cast<double>(max), 0.},
                                                             {1, 0}};

  test_single_agg(
    keys, vals, expect_keys, expect_vals, cudf::make_mean_aggregation<cudf::groupby_aggregation>());
}

// Groupby MEAN of a single group matches reduction MEAN, which also accumulates in double.
TEST_F(groupby_mean_overflow_test, matches_reduction)
{
  auto constexpr max = std::numeric_limits<int64_t>::max();

  cudf::test::fixed_width_column_wrapper<K> keys{0, 0, 0, 0};
  cudf::test::fixed_width_column_wrapper<int64_t> vals{max, max, max, int64_t{-3}};

  auto const reduced = cudf::reduce(vals,
                                    *cudf::make_mean_aggregation<cudf::reduce_aggregation>(),
                                    cudf::data_type{cudf::type_id::FLOAT64});
  auto const expect  = static_cast<cudf::numeric_scalar<double> const&>(*reduced).value();

  cudf::test::fixed_width_column_wrapper<K> expect_keys{0};
  cudf::test::fixed_width_column_wrapper<double> expect_vals{expect};
  test_single_agg(
    keys, vals, expect_keys, expect_vals, cudf::make_mean_aggregation<cudf::groupby_aggregation>());
}

// SUM and MEAN of one column use separate accumulators; results keep the request order.
TEST_F(groupby_mean_overflow_test, sum_and_mean_of_same_column)
{
  auto constexpr max = std::numeric_limits<int64_t>::max();

  cudf::test::fixed_width_column_wrapper<K> keys{1, 0, 0, 1};
  cudf::test::fixed_width_column_wrapper<int64_t> vals{int64_t{2}, max, max, int64_t{4}};
  cudf::test::fixed_width_column_wrapper<int32_t> other{10, 20, 30, 40};

  auto const wrapped_sum = static_cast<int64_t>(2 * static_cast<uint64_t>(max));
  cudf::test::fixed_width_column_wrapper<K> expect_keys{0, 1};
  cudf::test::fixed_width_column_wrapper<double> expect_mean{static_cast<double>(max), 3.};
  cudf::test::fixed_width_column_wrapper<int64_t> expect_sum{wrapped_sum, int64_t{6}};
  cudf::test::fixed_width_column_wrapper<cudf::size_type> expect_count{2, 2};
  cudf::test::fixed_width_column_wrapper<double> expect_other_mean{25., 25.};

  std::vector<cudf::groupby::aggregation_request> requests(2);
  requests[0].values = vals;
  requests[0].aggregations.push_back(cudf::make_mean_aggregation<cudf::groupby_aggregation>());
  requests[0].aggregations.push_back(cudf::make_sum_aggregation<cudf::groupby_aggregation>());
  requests[0].aggregations.push_back(cudf::make_mean_aggregation<cudf::groupby_aggregation>());
  requests[0].aggregations.push_back(cudf::make_count_aggregation<cudf::groupby_aggregation>());
  requests[1].values = other;
  requests[1].aggregations.push_back(cudf::make_mean_aggregation<cudf::groupby_aggregation>());

  cudf::groupby::groupby gb{cudf::table_view{{keys}}};
  auto [result_keys, results] = gb.aggregate(requests);

  auto const result_keys_view = result_keys->view();
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expect_keys,
                                 *by_key(result_keys_view, result_keys_view.column(0)));
  ASSERT_EQ(results.size(), 2);
  ASSERT_EQ(results[0].results.size(), 4);
  ASSERT_EQ(results[1].results.size(), 1);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expect_mean, *by_key(result_keys_view, *results[0].results[0]));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expect_sum, *by_key(result_keys_view, *results[0].results[1]));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expect_mean, *by_key(result_keys_view, *results[0].results[2]));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expect_count, *by_key(result_keys_view, *results[0].results[3]));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expect_other_mean,
                                 *by_key(result_keys_view, *results[1].results[0]));
}

// Decimal MEAN keeps the decimal type and shares its accumulator with a requested SUM.
TEST_F(groupby_mean_overflow_test, decimal_sum_and_mean)
{
  using decimal64  = numeric::decimal64;
  auto const scale = numeric::scale_type{-1};

  cudf::test::fixed_width_column_wrapper<K> keys{0, 0, 1, 1};
  cudf::test::fixed_point_column_wrapper<int64_t> vals{{10, 30, 50, 70}, scale};

  cudf::test::fixed_width_column_wrapper<K> expect_keys{0, 1};
  cudf::test::fixed_point_column_wrapper<int64_t> expect_sum{{40, 120}, scale};
  cudf::test::fixed_point_column_wrapper<int64_t> expect_mean{{20, 60}, scale};

  std::vector<cudf::groupby::aggregation_request> requests(1);
  requests[0].values = vals;
  requests[0].aggregations.push_back(cudf::make_sum_aggregation<cudf::groupby_aggregation>());
  requests[0].aggregations.push_back(cudf::make_mean_aggregation<cudf::groupby_aggregation>());

  cudf::groupby::groupby gb{cudf::table_view{{keys}}};
  auto [result_keys, results] = gb.aggregate(requests);

  auto const result_keys_view = result_keys->view();
  EXPECT_EQ(results[0].results[1]->type(), cudf::data_type(cudf::type_to_id<decimal64>(), scale));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expect_keys,
                                 *by_key(result_keys_view, result_keys_view.column(0)));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expect_sum, *by_key(result_keys_view, *results[0].results[0]));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expect_mean, *by_key(result_keys_view, *results[0].results[1]));
}
