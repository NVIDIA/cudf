/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/iterator_utilities.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/aggregation.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/groupby.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/unary.hpp>

#include <limits>

using namespace cudf::test::iterators;

namespace {
constexpr cudf::test::debug_output_level verbosity{cudf::test::debug_output_level::FIRST_ERROR};
constexpr int32_t null{0};                                       // Mark for null elements
constexpr double NaN{std::numeric_limits<double>::quiet_NaN()};  // Mark for NaN double elements

template <class T>
using keys_col = cudf::test::fixed_width_column_wrapper<T, int32_t>;

template <class T>
using vals_col = cudf::test::fixed_width_column_wrapper<T>;

using counts_col = cudf::test::fixed_width_column_wrapper<int32_t>;

template <class T>
using means_col = cudf::test::fixed_width_column_wrapper<T>;

template <class T>
using M2s_col = cudf::test::fixed_width_column_wrapper<T>;

using structs_col = cudf::test::structs_column_wrapper;
using vcol_views  = std::vector<cudf::column_view>;

/**
 * @brief Compute `COUNT_VALID`, `MEAN`, `M2` aggregations for the given values columns.
 * @return A pair of unique keys column and a structs column containing the computed values of
 *         (`COUNT_VALID`, `MEAN`, `M2`).
 */
auto compute_partial_results(cudf::column_view const& keys, cudf::column_view const& values)
{
  std::vector<cudf::groupby::aggregation_request> requests;
  requests.emplace_back();
  requests[0].values = values;
  requests[0].aggregations.emplace_back(cudf::make_count_aggregation<cudf::groupby_aggregation>());
  requests[0].aggregations.emplace_back(cudf::make_mean_aggregation<cudf::groupby_aggregation>());
  requests[0].aggregations.emplace_back(cudf::make_m2_aggregation<cudf::groupby_aggregation>());

  auto gb_obj                  = cudf::groupby::groupby(cudf::table_view({keys}));
  auto [out_keys, out_results] = gb_obj.aggregate(requests);

  // Cast the `COUNT_VALID` column to `INT64` type.
  out_results[0].results.front() = cudf::cast(out_results[0].results.front()->view(),
                                              cudf::data_type(cudf::type_id::INT64),
                                              cudf::get_default_stream());
  auto const num_output_rows     = out_keys->num_rows();
  return std::pair(
    std::move(out_keys->release()[0]),
    cudf::make_structs_column(num_output_rows,
                              std::move(out_results[0].results),
                              0,
                              cudf::create_null_mask(0, cudf::mask_state::UNALLOCATED)));
}

/**
 * @brief Perform merging for partial results of M2 aggregations.
 *
 * @return A pair of unique keys column and a structs column containing the merged values of
 *         (`COUNT_VALID`, `MEAN`, `M2`).
 */
auto merge_M2(vcol_views const& keys_cols, vcol_views const& values_cols)
{
  // Append all the keys and values together.
  auto const keys   = cudf::concatenate(keys_cols);
  auto const values = cudf::concatenate(values_cols);

  std::vector<cudf::groupby::aggregation_request> requests;
  requests.emplace_back();
  requests[0].values = *values;
  requests[0].aggregations.emplace_back(
    cudf::make_merge_m2_aggregation<cudf::groupby_aggregation>());

  auto gb_obj = cudf::groupby::groupby(cudf::table_view({*keys}));
  auto result = gb_obj.aggregate(requests);
  return std::pair(std::move(result.first->release()[0]), std::move(result.second[0].results[0]));
}

template <typename CountType>
void test_extreme_finite_first_partial()
{
  auto const key  = keys_col<int32_t>{1};
  auto counts     = cudf::test::fixed_width_column_wrapper<CountType>{CountType{1}};
  auto means      = means_col<double>{std::numeric_limits<double>::max()};
  auto m2s        = M2s_col<double>{0.0};
  auto const vals = structs_col{counts, means, m2s};

  auto const [out_key, out_vals] = merge_M2({key}, {vals});

  auto const expected_keys   = keys_col<int32_t>{1};
  auto expected_counts       = cudf::test::fixed_width_column_wrapper<CountType>{CountType{1}};
  auto expected_means        = means_col<double>{std::numeric_limits<double>::max()};
  auto expected_m2s          = M2s_col<double>{0.0};
  auto const expected_values = structs_col{expected_counts, expected_means, expected_m2s};
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected_keys, *out_key, verbosity);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected_values, *out_vals, verbosity);
}

template <typename CountType>
void test_extreme_finite_merged_partials()
{
  auto const keys = keys_col<int32_t>{1, 1};
  auto counts     = cudf::test::fixed_width_column_wrapper<CountType>{CountType{1}, CountType{1}};
  auto means      = means_col<double>{std::numeric_limits<double>::max(), 0.0};
  auto m2s        = M2s_col<double>{0.0, 0.0};
  auto const vals = structs_col{counts, means, m2s};

  auto const [out_keys, out_vals] = merge_M2({keys}, {vals});

  auto const expected_keys   = keys_col<int32_t>{1};
  auto expected_counts       = cudf::test::fixed_width_column_wrapper<CountType>{CountType{2}};
  auto expected_means        = means_col<double>{std::numeric_limits<double>::max() / 2};
  auto expected_m2s          = M2s_col<double>{std::numeric_limits<double>::infinity()};
  auto const expected_values = structs_col{expected_counts, expected_means, expected_m2s};
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected_keys, *out_keys, verbosity);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected_values, *out_vals, verbosity);
}

// A non-finite mean in a partial must propagate, not be coerced. This pins the
// identity-branch behavior: a single non-empty partial with NaN/Inf statistics
// should be preserved as-is, since coercing to NaN would discard upstream
// signal (e.g. an upstream overflow that already produced +Inf).
template <typename CountType>
void test_nan_mean_first_partial()
{
  auto const key  = keys_col<int32_t>{1};
  auto counts     = cudf::test::fixed_width_column_wrapper<CountType>{CountType{1}};
  auto means      = means_col<double>{std::numeric_limits<double>::quiet_NaN()};
  auto m2s        = M2s_col<double>{std::numeric_limits<double>::quiet_NaN()};
  auto const vals = structs_col{counts, means, m2s};

  auto const [out_key, out_vals] = merge_M2({key}, {vals});

  auto const expected_keys   = keys_col<int32_t>{1};
  auto expected_counts       = cudf::test::fixed_width_column_wrapper<CountType>{CountType{1}};
  auto expected_means        = means_col<double>{std::numeric_limits<double>::quiet_NaN()};
  auto expected_m2s          = M2s_col<double>{std::numeric_limits<double>::quiet_NaN()};
  auto const expected_values = structs_col{expected_counts, expected_means, expected_m2s};
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected_keys, *out_key, verbosity);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected_values, *out_vals, verbosity);
}

template <typename CountType>
void test_inf_mean_first_partial()
{
  auto const key  = keys_col<int32_t>{1};
  auto counts     = cudf::test::fixed_width_column_wrapper<CountType>{CountType{1}};
  auto means      = means_col<double>{std::numeric_limits<double>::infinity()};
  auto m2s        = M2s_col<double>{0.0};
  auto const vals = structs_col{counts, means, m2s};

  auto const [out_key, out_vals] = merge_M2({key}, {vals});

  auto const expected_keys   = keys_col<int32_t>{1};
  auto expected_counts       = cudf::test::fixed_width_column_wrapper<CountType>{CountType{1}};
  auto expected_means        = means_col<double>{std::numeric_limits<double>::infinity()};
  auto expected_m2s          = M2s_col<double>{0.0};
  auto const expected_values = structs_col{expected_counts, expected_means, expected_m2s};
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected_keys, *out_key, verbosity);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected_values, *out_vals, verbosity);
}

// Once the accumulator holds a NaN, any subsequent merge step must keep
// propagating NaN (NaN ⊕ anything = NaN), regardless of partial order.
template <typename CountType>
void test_nan_mean_merged_with_finite()
{
  auto const keys = keys_col<int32_t>{1, 1};
  auto counts     = cudf::test::fixed_width_column_wrapper<CountType>{CountType{10}, CountType{10}};
  auto means      = means_col<double>{std::numeric_limits<double>::quiet_NaN(), 5.0};
  auto m2s        = M2s_col<double>{std::numeric_limits<double>::quiet_NaN(), 20.0};
  auto const vals = structs_col{counts, means, m2s};

  auto const [out_keys, out_vals] = merge_M2({keys}, {vals});

  auto const expected_keys   = keys_col<int32_t>{1};
  auto expected_counts       = cudf::test::fixed_width_column_wrapper<CountType>{CountType{20}};
  auto expected_means        = means_col<double>{std::numeric_limits<double>::quiet_NaN()};
  auto expected_m2s          = M2s_col<double>{std::numeric_limits<double>::quiet_NaN()};
  auto const expected_values = structs_col{expected_counts, expected_means, expected_m2s};
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected_keys, *out_keys, verbosity);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected_values, *out_vals, verbosity);
}
}  // namespace

template <class T>
struct GroupbyMergeM2TypedTest : public cudf::test::BaseFixture {};

struct GroupbyMergeM2ExtremeTest : public cudf::test::BaseFixture {};

using TestTypes = cudf::test::Concat<cudf::test::Types<int8_t, int16_t, int32_t, int64_t>,
                                     cudf::test::FloatingPointTypes>;
TYPED_TEST_SUITE(GroupbyMergeM2TypedTest, TestTypes);

TYPED_TEST(GroupbyMergeM2TypedTest, InvalidInput)
{
  using T = TypeParam;

  auto const keys = keys_col<T>{1, 2, 3};

  // The input column must be a structs column.
  {
    auto const values = keys_col<T>{1, 2, 3};
    EXPECT_THROW(merge_M2({keys}, {values}), cudf::logic_error);
  }

  // The input column must be a structs column having 3 children.
  {
    auto vals1      = keys_col<T>{1, 2, 3};
    auto vals2      = vals_col<double>{1.0, 2.0, 3.0};
    auto const vals = structs_col{vals1, vals2};
    EXPECT_THROW(merge_M2({keys}, {vals}), cudf::logic_error);
  }

  // The input column must be a structs column having types (int64_t/double, double, double).
  {
    if constexpr (!std::is_same_v<T, double>) {
      auto vals1      = keys_col<T>{1, 2, 3};
      auto vals2      = keys_col<T>{1, 2, 3};
      auto vals3      = keys_col<T>{1, 2, 3};
      auto const vals = structs_col{vals1, vals2, vals3};
      EXPECT_THROW(merge_M2({keys}, {vals}), cudf::logic_error);
    }
  }
}

TYPED_TEST(GroupbyMergeM2TypedTest, EmptyInput)
{
  using T      = TypeParam;
  using M2_t   = double;
  using mean_t = double;

  auto const keys = keys_col<T>{};
  auto vals_count = counts_col{};
  auto vals_mean  = means_col<mean_t>{};
  auto vals_M2    = M2s_col<M2_t>{};
  auto const vals = structs_col{vals_count, vals_mean, vals_M2};

  auto const [out_keys, out_vals] = merge_M2({keys}, {vals});
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(keys, *out_keys, verbosity);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(vals, *out_vals, verbosity);
}

TEST_F(GroupbyMergeM2ExtremeTest, ExtremeFiniteFirstPartialInt64Count)
{
  test_extreme_finite_first_partial<int64_t>();
}

TEST_F(GroupbyMergeM2ExtremeTest, ExtremeFiniteFirstPartialDoubleCount)
{
  test_extreme_finite_first_partial<double>();
}

TEST_F(GroupbyMergeM2ExtremeTest, ExtremeFiniteMergedPartialsInt64Count)
{
  test_extreme_finite_merged_partials<int64_t>();
}

TEST_F(GroupbyMergeM2ExtremeTest, ExtremeFiniteMergedPartialsDoubleCount)
{
  test_extreme_finite_merged_partials<double>();
}

TEST_F(GroupbyMergeM2ExtremeTest, NanMeanFirstPartialInt64Count)
{
  test_nan_mean_first_partial<int64_t>();
}

TEST_F(GroupbyMergeM2ExtremeTest, NanMeanFirstPartialDoubleCount)
{
  test_nan_mean_first_partial<double>();
}

TEST_F(GroupbyMergeM2ExtremeTest, InfMeanFirstPartialInt64Count)
{
  test_inf_mean_first_partial<int64_t>();
}

TEST_F(GroupbyMergeM2ExtremeTest, InfMeanFirstPartialDoubleCount)
{
  test_inf_mean_first_partial<double>();
}

TEST_F(GroupbyMergeM2ExtremeTest, NanMeanMergedWithFiniteInt64Count)
{
  test_nan_mean_merged_with_finite<int64_t>();
}

TEST_F(GroupbyMergeM2ExtremeTest, NanMeanMergedWithFiniteDoubleCount)
{
  test_nan_mean_merged_with_finite<double>();
}

TYPED_TEST(GroupbyMergeM2TypedTest, SimpleInput)
{
  using T = TypeParam;
  using R = double;

  // Full dataset:
  //
  // keys = [1, 2, 3, 1, 2, 2, 1, 3, 3, 2]
  // vals = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
  //
  // key = 1: vals = [0, 3, 6]
  // key = 2: vals = [1, 4, 5, 9]
  // key = 3: vals = [2, 7, 8]

  // Partitioned datasets:
  auto const keys1 = keys_col<T>{1, 2, 3};
  auto const keys2 = keys_col<T>{1, 2, 2};
  auto const keys3 = keys_col<T>{1, 3, 3, 2};

  auto const vals1 = vals_col<T>{0, 1, 2};
  auto const vals2 = vals_col<T>{3, 4, 5};
  auto const vals3 = vals_col<T>{6, 7, 8, 9};

  // The expected results to validate.
  auto const expected_keys = keys_col<T>{1, 2, 3};
  auto const expected_M2s  = M2s_col<R>{18.0, 32.75, 20.0 + 2.0 / 3.0};

  // Compute partial results (`COUNT_VALID`, `MEAN`, `M2`) of each dataset.
  // The partial results are also assembled into a structs column.
  auto const [out1_keys, out1_vals] = compute_partial_results(keys1, vals1);
  auto const [out2_keys, out2_vals] = compute_partial_results(keys2, vals2);
  auto const [out3_keys, out3_vals] = compute_partial_results(keys3, vals3);

  // Merge the partial results to the final results.
  // Merging can be done in just one merge step, or in multiple steps.

  // Multiple steps merging:
  {
    auto const [out4_keys, out4_vals] =
      merge_M2(vcol_views{*out1_keys, *out2_keys}, vcol_views{*out1_vals, *out2_vals});
    auto const [final_keys, final_vals] =
      merge_M2(vcol_views{*out3_keys, *out4_keys}, vcol_views{*out3_vals, *out4_vals});

    auto const out_M2s = final_vals->child(2);
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected_keys, *final_keys, verbosity);
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected_M2s, out_M2s, verbosity);
  }

  // One step merging:
  {
    auto const [final_keys, final_vals] = merge_M2(vcol_views{*out1_keys, *out2_keys, *out3_keys},
                                                   vcol_views{*out1_vals, *out2_vals, *out3_vals});

    auto const out_M2s = final_vals->child(2);
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected_keys, *final_keys, verbosity);
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected_M2s, out_M2s, verbosity);
  }
}

TYPED_TEST(GroupbyMergeM2TypedTest, SimpleInputHavingNegativeValues)
{
  using T = TypeParam;
  using R = double;

  // Full dataset:
  //
  // keys = [1, 2,  3, 1,  2,  2,  1, 3,  3, 2]
  // vals = [0, 1, -2, 3, -4, -5, -6, 7, -8, 9]
  //
  // key = 1: vals = [0,  3, -6]
  // key = 2: vals = [1, -4, -5, 9]
  // key = 3: vals = [-2, 7, -8]

  // Partitioned datasets:
  auto const keys1 = keys_col<T>{1, 2, 3};
  auto const keys2 = keys_col<T>{1, 2, 2};
  auto const keys3 = keys_col<T>{1, 3, 3, 2};

  auto const vals1 = vals_col<T>{0, 1, -2};
  auto const vals2 = vals_col<T>{3, -4, -5};
  auto const vals3 = vals_col<T>{-6, 7, -8, 9};

  // The expected results to validate.
  auto const expected_keys = keys_col<T>{1, 2, 3};
  auto const expected_M2s  = M2s_col<R>{42.0, 122.75, 114.0};

  // Compute partial results (`COUNT_VALID`, `MEAN`, `M2`) of each dataset.
  // The partial results are also assembled into a structs column.
  auto const [out1_keys, out1_vals] = compute_partial_results(keys1, vals1);
  auto const [out2_keys, out2_vals] = compute_partial_results(keys2, vals2);
  auto const [out3_keys, out3_vals] = compute_partial_results(keys3, vals3);

  // Merge the partial results to the final results.
  // Merging can be done in just one merge step, or in multiple steps.

  // Multiple steps merging:
  {
    auto const [out4_keys, out4_vals] =
      merge_M2(vcol_views{*out1_keys, *out2_keys}, vcol_views{*out1_vals, *out2_vals});
    auto const [final_keys, final_vals] =
      merge_M2(vcol_views{*out3_keys, *out4_keys}, vcol_views{*out3_vals, *out4_vals});

    auto const out_M2s = final_vals->child(2);
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected_keys, *final_keys, verbosity);
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected_M2s, out_M2s, verbosity);
  }

  // One step merging:
  {
    auto const [final_keys, final_vals] = merge_M2(vcol_views{*out1_keys, *out2_keys, *out3_keys},
                                                   vcol_views{*out1_vals, *out2_vals, *out3_vals});

    auto const out_M2s = final_vals->child(2);
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected_keys, *final_keys, verbosity);
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected_M2s, out_M2s, verbosity);
  }
}

TYPED_TEST(GroupbyMergeM2TypedTest, InputHasNulls)
{
  using T = TypeParam;
  using R = double;

  // Full dataset:
  //
  // keys = [1,    2, 3, 1, 2, 2,    1, null, 3, 2, 4]
  // vals = [null, 1, 2, 3, 4, null, 6, 7,    8, 9, null]
  //
  // key = 1: vals = [null, 3, 6]
  // key = 2: vals = [1, 4, null, 9]
  // key = 3: vals = [2, 8]
  // key = 4: vals = [null]

  // Partitioned datasets:
  auto const keys1 = keys_col<T>{1, 2, 3, 1};
  auto const keys2 = keys_col<T>{{2, 2, 1, null}, null_at(3)};
  auto const keys3 = keys_col<T>{3, 2, 4};

  auto const vals1 = vals_col<T>{{null, 1, 2, 3}, null_at(0)};
  auto const vals2 = vals_col<T>{{4, null, 6, 7}, null_at(1)};
  auto const vals3 = vals_col<T>{{8, 9, null}, null_at(2)};

  // The expected results to validate.
  auto const expected_keys = keys_col<T>{1, 2, 3, 4};
  auto const expected_M2s  = M2s_col<R>{4.5, 32.0 + 2.0 / 3.0, 18.0, 0.0};

  // Compute partial results (`COUNT_VALID`, `MEAN`, `M2`) of each dataset.
  // The partial results are also assembled into a structs column.
  auto const [out1_keys, out1_vals] = compute_partial_results(keys1, vals1);
  auto const [out2_keys, out2_vals] = compute_partial_results(keys2, vals2);
  auto const [out3_keys, out3_vals] = compute_partial_results(keys3, vals3);

  // Merge the partial results to the final results.
  // Merging can be done in just one merge step, or in multiple steps.

  // Multiple steps merging:
  {
    auto const [out4_keys, out4_vals] =
      merge_M2(vcol_views{*out1_keys, *out2_keys}, vcol_views{*out1_vals, *out2_vals});
    auto const [final_keys, final_vals] =
      merge_M2(vcol_views{*out3_keys, *out4_keys}, vcol_views{*out3_vals, *out4_vals});

    auto const out_M2s = final_vals->child(2);
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected_keys, *final_keys, verbosity);
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected_M2s, out_M2s, verbosity);
  }

  // One step merging:
  {
    auto const [final_keys, final_vals] = merge_M2(vcol_views{*out1_keys, *out2_keys, *out3_keys},
                                                   vcol_views{*out1_vals, *out2_vals, *out3_vals});

    auto const out_M2s = final_vals->child(2);
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected_keys, *final_keys, verbosity);
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected_M2s, out_M2s, verbosity);
  }
}

TYPED_TEST(GroupbyMergeM2TypedTest, InputHaveNullsAndNaNs)
{
  using T = TypeParam;
  using R = double;

  // Full dataset:
  //
  // keys = [4, 3, 1, 2, 3, 1, 2, 2, 1, null, 3, 2, 4, 4]
  // vals = [null, null, 0.0, 1.0, 2.0, 3.0, 4.0, NaN, 6.0, 7.0, 8.0, 9.0, 10.0, NaN]
  //
  // key = 1: vals = [0, 3, 6]
  // key = 2: vals = [1, 4, NaN, 9]
  // key = 3: vals = [null, 2, 8]
  // key = 4: vals = [null, 10, NaN]

  // Partitioned datasets:
  auto const keys1 = keys_col<T>{4, 3, 1, 2};
  auto const keys2 = keys_col<T>{3, 1, 2};
  auto const keys3 = keys_col<T>{{2, 1, null}, null_at(2)};
  auto const keys4 = keys_col<T>{3, 2, 4, 4};

  auto const vals1 = vals_col<double>{{0.0 /*NULL*/, 0.0 /*NULL*/, 0.0, 1.0}, nulls_at({0, 1})};
  auto const vals2 = vals_col<double>{2.0, 3.0, 4.0};
  auto const vals3 = vals_col<double>{NaN, 6.0, 7.0};
  auto const vals4 = vals_col<double>{8.0, 9.0, 10.0, NaN};

  // The expected results to validate.
  auto const expected_keys = keys_col<T>{1, 2, 3, 4};
  auto const expected_M2s  = M2s_col<R>{18.0, NaN, 18.0, NaN};

  // Compute partial results (`COUNT_VALID`, `MEAN`, `M2`) of each dataset.
  // The partial results are also assembled into a structs column.
  auto const [out1_keys, out1_vals] = compute_partial_results(keys1, vals1);
  auto const [out2_keys, out2_vals] = compute_partial_results(keys2, vals2);
  auto const [out3_keys, out3_vals] = compute_partial_results(keys3, vals3);
  auto const [out4_keys, out4_vals] = compute_partial_results(keys4, vals4);

  // Merge the partial results to the final results.
  // Merging can be done in just one merge step, or in multiple steps.

  // Multiple steps merging:
  {
    auto const [out5_keys, out5_vals] =
      merge_M2(vcol_views{*out1_keys, *out2_keys}, vcol_views{*out1_vals, *out2_vals});
    auto const [out6_keys, out6_vals] =
      merge_M2(vcol_views{*out3_keys, *out4_keys}, vcol_views{*out3_vals, *out4_vals});
    auto const [final_keys, final_vals] =
      merge_M2(vcol_views{*out5_keys, *out6_keys}, vcol_views{*out5_vals, *out6_vals});

    auto const out_M2s = final_vals->child(2);
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected_keys, *final_keys, verbosity);
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected_M2s, out_M2s, verbosity);
  }

  // One step merging:
  {
    auto const [final_keys, final_vals] =
      merge_M2(vcol_views{*out1_keys, *out2_keys, *out3_keys, *out4_keys},
               vcol_views{*out1_vals, *out2_vals, *out3_vals, *out4_vals});

    auto const out_M2s = final_vals->child(2);
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected_keys, *final_keys, verbosity);
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected_M2s, out_M2s, verbosity);
  }
}

TYPED_TEST(GroupbyMergeM2TypedTest, SlicedColumnsInput)
{
  using T = TypeParam;
  using R = double;

  // This test should compute M2 aggregation on the same dataset as the InputHaveNullsAndNaNs test.
  // i.e.:
  //
  // keys = [4, 3, 1, 2, 3, 1, 2, 2, 1, null, 3, 2, 4, 4]
  // vals = [null, null, 0.0, 1.0, 2.0, 3.0, 4.0, NaN, 6.0, 7.0, 8.0, 9.0, 10.0, NaN]
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

  // Partitioned datasets, taken from the original dataset in the range [10, 24).
  auto const keys1 = cudf::slice(keys_original, {10, 14})[0];  // {4, 3, 1, 2}
  auto const keys2 = cudf::slice(keys_original, {14, 17})[0];  // {3, 1, 2}
  auto const keys3 = cudf::slice(keys_original, {17, 20})[0];  // {2, 1, null}
  auto const keys4 = cudf::slice(keys_original, {20, 24})[0];  // {3, 2, 4, 4}

  auto const vals1 = cudf::slice(vals_original, {10, 14})[0];  // {null, null, 0.0, 1.0}
  auto const vals2 = cudf::slice(vals_original, {14, 17})[0];  // {2.0, 3.0, 4.0}
  auto const vals3 = cudf::slice(vals_original, {17, 20})[0];  // {NaN, 6.0, 7.0}
  auto const vals4 = cudf::slice(vals_original, {20, 24})[0];  // {8.0, 9.0, 10.0, NaN}

  // The expected results to validate.
  auto const expected_keys = keys_col<T>{1, 2, 3, 4};
  auto const expected_M2s  = M2s_col<R>{18.0, NaN, 18.0, NaN};

  // Compute partial results (`COUNT_VALID`, `MEAN`, `M2`) of each dataset.
  // The partial results are also assembled into a structs column.
  auto const [out1_keys, out1_vals] = compute_partial_results(keys1, vals1);
  auto const [out2_keys, out2_vals] = compute_partial_results(keys2, vals2);
  auto const [out3_keys, out3_vals] = compute_partial_results(keys3, vals3);
  auto const [out4_keys, out4_vals] = compute_partial_results(keys4, vals4);

  // Merge the partial results to the final results.
  // Merging can be done in just one merge step, or in multiple steps.

  // Multiple steps merging:
  {
    auto const [out5_keys, out5_vals] =
      merge_M2(vcol_views{*out1_keys, *out2_keys}, vcol_views{*out1_vals, *out2_vals});
    auto const [out6_keys, out6_vals] =
      merge_M2(vcol_views{*out3_keys, *out4_keys}, vcol_views{*out3_vals, *out4_vals});
    auto const [final_keys, final_vals] =
      merge_M2(vcol_views{*out5_keys, *out6_keys}, vcol_views{*out5_vals, *out6_vals});

    auto const out_M2s = final_vals->child(2);
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected_keys, *final_keys, verbosity);
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected_M2s, out_M2s, verbosity);
  }

  // One step merging:
  {
    auto const [final_keys, final_vals] =
      merge_M2(vcol_views{*out1_keys, *out2_keys, *out3_keys, *out4_keys},
               vcol_views{*out1_vals, *out2_vals, *out3_vals, *out4_vals});

    auto const out_M2s = final_vals->child(2);
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected_keys, *final_keys, verbosity);
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected_M2s, out_M2s, verbosity);
  }
}
