/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/iterator_utilities.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/aggregation.hpp>
#include <cudf/dictionary/encode.hpp>
#include <cudf/groupby.hpp>
#include <cudf/sorting.hpp>
#include <cudf/utilities/bit.hpp>

#include <algorithm>
#include <cmath>
#include <vector>

using namespace cudf::test::iterators;

namespace {
constexpr cudf::test::debug_output_level verbosity{cudf::test::debug_output_level::FIRST_ERROR};
constexpr int32_t null{0};                                       // Mark for null elements
constexpr double NaN{std::numeric_limits<double>::quiet_NaN()};  // Mark for NaN double elements

template <class T>
using keys_col = cudf::test::fixed_width_column_wrapper<T, int32_t>;

template <class T>
using vals_col = cudf::test::fixed_width_column_wrapper<T>;

template <class T>
using M2s_col = cudf::test::fixed_width_column_wrapper<T>;

auto compute_M2(cudf::column_view const& keys, cudf::column_view const& values)
{
  auto gb_obj = cudf::groupby::groupby(cudf::table_view({keys}));

  auto [hash_gb_keys, hash_gb_vals] = [&] {
    std::vector<cudf::groupby::aggregation_request> requests;
    requests.emplace_back();
    requests[0].values = values;
    requests[0].aggregations.emplace_back(cudf::make_m2_aggregation<cudf::groupby_aggregation>());
    auto const result      = gb_obj.aggregate(requests);
    auto const sort_order  = cudf::sorted_order(result.first->view(), {}, {});
    auto const sorted_keys = cudf::gather(result.first->view(), *sort_order);
    auto const sorted_vals =
      cudf::gather(cudf::table_view({result.second[0].results[0]->view()}), *sort_order);
    return std::pair(std::move(sorted_keys->release()[0]), std::move(sorted_vals->release()[0]));
  }();

  auto const [sort_gb_keys, sort_gb_vals] = [&] {
    // Create a fresh aggregation request for sort-based aggregation instead of reusing.
    // This is to avoid wrong output when the previous groupby aggregation has not been executed
    // while the requests vector is modified.
    std::vector<cudf::groupby::aggregation_request> requests;
    requests.emplace_back();
    requests[0].values = values;
    requests[0].aggregations.emplace_back(cudf::make_m2_aggregation<cudf::groupby_aggregation>());
    requests[0].aggregations.emplace_back(
      cudf::make_nth_element_aggregation<cudf::groupby_aggregation>(0));
    auto result = gb_obj.aggregate(requests);
    return std::pair(std::move(result.first->release()[0]), std::move(result.second[0].results[0]));
  }();

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*hash_gb_keys, *sort_gb_keys, verbosity);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*hash_gb_vals, *sort_gb_vals, verbosity);

  return std::pair(std::move(hash_gb_keys), std::move(hash_gb_vals));
}
}  // namespace

template <class T>
struct GroupbyM2TypedTest : public cudf::test::BaseFixture {};

using TestTypes = cudf::test::Concat<cudf::test::Types<int8_t, int16_t, int32_t, int64_t>,
                                     cudf::test::FloatingPointTypes>;
TYPED_TEST_SUITE(GroupbyM2TypedTest, TestTypes);

TYPED_TEST(GroupbyM2TypedTest, EmptyInput)
{
  using T = TypeParam;
  using R = double;

  auto const keys = keys_col<T>{};
  auto const vals = vals_col<T>{};

  auto const [out_keys, out_M2s] = compute_M2(keys, vals);
  auto const expected_M2s        = M2s_col<R>{};

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(keys, *out_keys, verbosity);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected_M2s, *out_M2s, verbosity);
}

TYPED_TEST(GroupbyM2TypedTest, AllNullKeysInput)
{
  using T = TypeParam;
  using R = double;

  auto const keys = keys_col<T>{{1, 2, 3}, all_nulls()};
  auto const vals = vals_col<T>{3, 4, 5};

  auto const [out_keys, out_M2s] = compute_M2(keys, vals);
  auto const expected_keys       = keys_col<T>{};
  auto const expected_M2s        = M2s_col<R>{};

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected_keys, *out_keys, verbosity);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected_M2s, *out_M2s, verbosity);
}

TYPED_TEST(GroupbyM2TypedTest, AllNullValuesInput)
{
  using T = TypeParam;
  using R = double;

  auto const keys = keys_col<T>{1, 2, 3};
  auto const vals = vals_col<T>{{3, 4, 5}, all_nulls()};

  auto const [out_keys, out_M2s] = compute_M2(keys, vals);
  auto const expected_M2s        = M2s_col<R>{0, 0, 0};

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(keys, *out_keys, verbosity);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected_M2s, *out_M2s, verbosity);
}

TYPED_TEST(GroupbyM2TypedTest, SimpleInput)
{
  using T = TypeParam;
  using R = double;

  // key = 1: vals = [0, 3, 6]
  // key = 2: vals = [1, 4, 5, 9]
  // key = 3: vals = [2, 7, 8]
  auto const keys = keys_col<T>{1, 2, 3, 1, 2, 2, 1, 3, 3, 2};
  auto const vals = vals_col<T>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};

  auto const [out_keys, out_M2s] = compute_M2(keys, vals);
  auto const expected_keys       = keys_col<T>{1, 2, 3};
  auto const expected_M2s        = M2s_col<R>{18.0, 32.75, 20.0 + 2.0 / 3.0};

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected_keys, *out_keys, verbosity);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected_M2s, *out_M2s, verbosity);
}

TYPED_TEST(GroupbyM2TypedTest, SimpleInputHavingNegativeValues)
{
  using T = TypeParam;
  using R = double;

  // key = 1: vals = [0,  3, -6]
  // key = 2: vals = [1, -4, -5, 9]
  // key = 3: vals = [-2, 7, -8]
  auto const keys = keys_col<T>{1, 2, 3, 1, 2, 2, 1, 3, 3, 2};
  auto const vals = vals_col<T>{0, 1, -2, 3, -4, -5, -6, 7, -8, 9};

  auto const [out_keys, out_M2s] = compute_M2(keys, vals);
  auto const expected_keys       = keys_col<T>{1, 2, 3};
  auto const expected_M2s        = M2s_col<R>{42.0, 122.75, 114.0};

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected_keys, *out_keys, verbosity);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected_M2s, *out_M2s, verbosity);
}

TYPED_TEST(GroupbyM2TypedTest, ValuesHaveNulls)
{
  using T = TypeParam;
  using R = double;

  auto const keys = keys_col<T>{1, 2, 3, 4, 5, 2, 3, 2};
  auto const vals = vals_col<T>{{0, null, 2, 3, null, 5, 6, 7}, nulls_at({1, 4})};

  auto const [out_keys, out_M2s] = compute_M2(keys, vals);
  auto const expected_keys       = keys_col<T>{1, 2, 3, 4, 5};
  auto const expected_M2s        = M2s_col<R>{0.0, 2.0, 8.0, 0.0, 0.0};

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected_keys, *out_keys, verbosity);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected_M2s, *out_M2s, verbosity);
}

TYPED_TEST(GroupbyM2TypedTest, KeysAndValuesHaveNulls)
{
  using T = TypeParam;
  using R = double;

  // key = 1: vals = [null, 3, 6]
  // key = 2: vals = [1, 4, null, 9]
  // key = 3: vals = [2, 8]
  // key = 4: vals = [null]
  auto const keys = keys_col<T>{{1, 2, 3, 1, 2, 2, 1, null, 3, 2, 4}, null_at(7)};
  auto const vals = vals_col<T>{{null, 1, 2, 3, 4, null, 6, 7, 8, 9, null}, nulls_at({0, 5, 10})};

  auto const [out_keys, out_M2s] = compute_M2(keys, vals);
  auto const expected_keys       = keys_col<T>{1, 2, 3, 4};
  auto const expected_M2s        = M2s_col<R>{4.5, 32.0 + 2.0 / 3.0, 18.0, 0.0};

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected_keys, *out_keys, verbosity);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected_M2s, *out_M2s, verbosity);
}

TYPED_TEST(GroupbyM2TypedTest, InputHaveNullsAndNaNs)
{
  using T = TypeParam;
  using R = double;

  // key = 1: vals = [0, 3, 6]
  // key = 2: vals = [1, 4, NaN, 9]
  // key = 3: vals = [null, 2, 8]
  // key = 4: vals = [null, 10, NaN]
  auto const keys = keys_col<T>{{4, 3, 1, 2, 3, 1, 2, 2, 1, null, 3, 2, 4, 4}, null_at(9)};
  auto const vals = vals_col<double>{
    {0.0 /*NULL*/, 0.0 /*NULL*/, 0.0, 1.0, 2.0, 3.0, 4.0, NaN, 6.0, 7.0, 8.0, 9.0, 10.0, NaN},
    nulls_at({0, 1})};

  auto const [out_keys, out_M2s] = compute_M2(keys, vals);
  auto const expected_keys       = keys_col<T>{1, 2, 3, 4};
  auto const expected_M2s        = M2s_col<R>{18.0, NaN, 18.0, NaN};

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected_keys, *out_keys, verbosity);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected_M2s, *out_M2s, verbosity);
}

TYPED_TEST(GroupbyM2TypedTest, SlicedColumnsInput)
{
  using T = TypeParam;
  using R = double;

  // This test should compute M2 aggregation on the same dataset as the InputHaveNullsAndNaNs test.
  // i.e.:
  //
  // key = 1: vals = [0, 3, 6]
  // key = 2: vals = [1, 4, NaN, 9]
  // key = 3: vals = [null, 2, 8]
  // key = 4: vals = [null, 10, NaN]

  auto const keys_original =
    keys_col<T>{{
                  1, 2, 3, 4, 5, 1, 2, 3, 4, 5,                 // will not use, don't care
                  4, 3, 1, 2, 3, 1, 2, 2, 1, null, 3, 2, 4, 4,  // use this
                  1, 2, 3, 4, 5, 1, 2, 3, 4, 5                  // will not use, don't care
                },
                null_at(19)};
  auto const vals_original = vals_col<double>{
    {
      3.0, 2.0,  5.0,  4.0,  6.0, 9.0, 1.0, 0.0,  1.0,  7.0,  // will not use, don't care
      0.0, 0.0,  0.0,  1.0,  2.0, 3.0, 4.0, NaN,  6.0,  7.0, 8.0, 9.0, 10.0, NaN,  // use this
      9.0, 10.0, 11.0, 12.0, 0.0, 5.0, 1.0, 20.0, 19.0, 15.0  // will not use, don't care
    },
    nulls_at({10, 11})};

  auto const keys = cudf::slice(keys_original, {10, 24})[0];
  auto const vals = cudf::slice(vals_original, {10, 24})[0];

  auto const [out_keys, out_M2s] = compute_M2(keys, vals);
  auto const expected_keys       = keys_col<T>{1, 2, 3, 4};
  auto const expected_M2s        = M2s_col<R>{18.0, NaN, 18.0, NaN};

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected_keys, *out_keys, verbosity);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected_M2s, *out_M2s, verbosity);
}

struct GroupbyStableM2Test : public cudf::test::BaseFixture {};

TEST_F(GroupbyStableM2Test, LargeOffsetAcrossReductionBoundaries)
{
  for (int n : {1, 31, 32, 33, 1023, 1024, 1025, 4097}) {
    // Interleaved groups: spread, constant, all-null, one valid value, and a null key.
    // Padding on both ends verifies the original row indices respect sliced column offsets.
    std::vector<int32_t> keys{99};
    std::vector<double> values{0};
    std::vector<bool> key_valid{true}, valid{true};
    std::vector<std::vector<long double>> reference(5);
    for (int i = 0; i < n; ++i) {
      for (int g = 0; g < 5; ++g) {
        auto const x        = 0x1p40 + (g == 1 ? 0 : i % 8);
        auto const is_valid = g != 2 && (g == 3 ? i == 0 : i % 7 != 6);
        keys.push_back(g);
        values.push_back(x);
        key_valid.push_back(g != 4);
        valid.push_back(is_valid);
        if (is_valid) { reference[g].push_back(static_cast<long double>(x)); }
      }
    }
    keys.push_back(99);
    values.push_back(0);
    key_valid.push_back(true);
    valid.push_back(true);
    keys_col<int32_t> input_keys(keys.begin(), keys.end(), key_valid.begin());
    vals_col<double> input_values(values.begin(), values.end(), valid.begin());
    auto dictionary        = cudf::dictionary::encode(input_values);
    auto const sliced_keys = cudf::slice(input_keys, {1, 1 + 5 * n}).front();
    for (bool encoded : {false, true}) {
      auto const sliced_values =
        cudf::slice(encoded ? dictionary->view() : cudf::column_view(input_values), {1, 1 + 5 * n})
          .front();
      for (auto null_keys : {cudf::null_policy::EXCLUDE, cudf::null_policy::INCLUDE}) {
        SCOPED_TRACE(::testing::Message()
                     << "n=" << n << " dictionary=" << encoded
                     << " include_null=" << (null_keys == cudf::null_policy::INCLUDE));
        std::vector<cudf::groupby::aggregation_request> requests(2);
        // COUNT before M2, and M2 explicit after a compound request on the same column.
        requests[0].values = sliced_values;
        requests[0].aggregations.push_back(
          cudf::make_count_aggregation<cudf::groupby_aggregation>());
        requests[0].aggregations.push_back(
          cudf::make_variance_aggregation<cudf::groupby_aggregation>(0));
        requests[0].aggregations.push_back(
          cudf::make_std_aggregation<cudf::groupby_aggregation>(1));
        requests[0].aggregations.push_back(
          cudf::make_variance_aggregation<cudf::groupby_aggregation>(n));
        requests[1].values = sliced_values;
        requests[1].aggregations.push_back(cudf::make_m2_aggregation<cudf::groupby_aggregation>());
        requests[1].aggregations.push_back(cudf::make_m2_aggregation<cudf::groupby_aggregation>());
        cudf::groupby::groupby gb(cudf::table_view{{sliced_keys}}, null_keys);
        auto result         = gb.aggregate(requests);
        auto const out_keys = cudf::test::to_host<int32_t>(result.first->view().column(0));
        auto const counts   = cudf::test::to_host<int32_t>(*result.second[0].results[0]).first;
        auto const m2       = cudf::test::to_host<double>(*result.second[1].results[0]).first;
        auto const variance = cudf::test::to_host<double>(*result.second[0].results[1]);
        auto const stddev   = cudf::test::to_host<double>(*result.second[0].results[2]);
        ASSERT_EQ(m2.size(), null_keys == cudf::null_policy::INCLUDE ? 5 : 4);
        EXPECT_FALSE(result.second[1].results[0]->nullable());
        EXPECT_EQ(result.second[0].results[3]->null_count(), m2.size());
        CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result.second[1].results[0], *result.second[1].results[1]);
        for (std::size_t row = 0; row < m2.size(); ++row) {
          auto const g   = out_keys.second.empty() || cudf::bit_is_set(out_keys.second.data(), row)
                             ? out_keys.first[row]
                             : 4;
          auto const& xs = reference[g];
          long double mean = 0, expected = 0;
          for (auto x : xs) {
            mean += x;
          }
          if (!xs.empty()) { mean /= xs.size(); }
          for (auto x : xs) {
            expected += (x - mean) * (x - mean);
          }
          auto const expected_m2 = static_cast<double>(expected);
          EXPECT_EQ(counts[row], xs.size());
          EXPECT_NEAR(m2[row], expected_m2, std::max(1e-8, expected_m2 * 1e-4));
          auto const var_valid =
            variance.second.empty() || cudf::bit_is_set(variance.second.data(), row);
          auto const std_valid =
            stddev.second.empty() || cudf::bit_is_set(stddev.second.data(), row);
          EXPECT_EQ(var_valid, !xs.empty());
          EXPECT_EQ(std_valid, xs.size() > 1);
          if (var_valid) {
            auto const v = expected_m2 / xs.size();
            EXPECT_NEAR(variance.first[row], v, std::max(1e-8, v * 1e-4));
          }
          if (std_valid) {
            auto const v = std::sqrt(expected_m2 / (xs.size() - 1));
            EXPECT_NEAR(stddev.first[row], v, std::max(1e-8, v * 1e-4));
          }
        }
      }
    }
  }
}

TEST_F(GroupbyStableM2Test, NonFiniteAndExtremeConstants)
{
  auto const inf  = std::numeric_limits<double>::infinity();
  auto const huge = std::numeric_limits<double>::max();
  keys_col<int32_t> keys{0, 1, 2, 3, 3, 4, 4, 5, 5, 6};
  vals_col<double> values{NaN, inf, -inf, inf, inf, -inf, inf, huge, huge, huge};
  std::vector<cudf::groupby::aggregation_request> requests(1);
  requests[0].values = values;
  requests[0].aggregations.push_back(cudf::make_m2_aggregation<cudf::groupby_aggregation>());
  cudf::groupby::groupby gb(cudf::table_view{{keys}});
  auto result = gb.aggregate(requests);
  auto sorted =
    cudf::sort(cudf::table_view{{result.first->view().column(0), *result.second[0].results[0]}});
  M2s_col<double> expected{NaN, NaN, NaN, NaN, NaN, 0.0, 0.0};
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected, sorted->view().column(1));
}

TEST_F(GroupbyStableM2Test, IndependentSumsAndRequestOrder)
{
  keys_col<int32_t> keys{0, 0, 0, 1, 1};
  vals_col<double> values{1, 2, 3, 4, 6};
  for (bool m2_first : {false, true}) {
    std::vector<cudf::groupby::aggregation_request> requests(2);
    requests[0].values = values;
    requests[0].aggregations.push_back(cudf::make_m2_aggregation<cudf::groupby_aggregation>());
    requests[1].values = values;
    requests[1].aggregations.push_back(cudf::make_sum_aggregation<cudf::groupby_aggregation>());
    requests[1].aggregations.push_back(cudf::make_mean_aggregation<cudf::groupby_aggregation>());
    requests[1].aggregations.push_back(
      cudf::make_sum_of_squares_aggregation<cudf::groupby_aggregation>());
    if (!m2_first) { std::swap(requests[0], requests[1]); }
    cudf::groupby::groupby gb(cudf::table_view{{keys}});
    auto result          = gb.aggregate(requests);
    auto const m2_index  = m2_first ? 0 : 1;
    auto const sum_index = 1 - m2_index;
    auto sorted          = cudf::sort(cudf::table_view{{result.first->view().column(0),
                                                        *result.second[m2_index].results[0],
                                                        *result.second[sum_index].results[0],
                                                        *result.second[sum_index].results[1],
                                                        *result.second[sum_index].results[2]}});
    vals_col<double> m2{2, 2}, sums{6, 10}, means{2, 5}, squares{14, 52};
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(m2, sorted->view().column(1));
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(sums, sorted->view().column(2));
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(means, sorted->view().column(3));
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(squares, sorted->view().column(4));
  }
}
