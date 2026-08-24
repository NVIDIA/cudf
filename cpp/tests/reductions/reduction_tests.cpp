/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/iterator_utilities.hpp>
#include <cudf_test/table_utilities.hpp>
#include <cudf_test/testing_main.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/copying.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/dictionary/dictionary_column_view.hpp>
#include <cudf/dictionary/update_keys.hpp>
#include <cudf/fixed_point/fixed_point.hpp>
#include <cudf/reduction.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/sorting.hpp>
#include <cudf/types.hpp>
#include <cudf/wrappers/timestamps.hpp>

#include <cuda/iterator>

#include <algorithm>
#include <iterator>
#include <numeric>
#include <vector>

using aggregation        = cudf::aggregation;
using reduce_aggregation = cudf::reduce_aggregation;

template <typename T>
auto convert_int(int value)
{
  if (std::is_unsigned_v<T>) value = std::abs(value);
  if constexpr (cudf::is_timestamp_t<T>::value) {
    return T{typename T::duration(value)};
  } else {
    return static_cast<T>(value);
  }
}

template <typename T>
auto convert_values(std::vector<int> const& int_values)
{
  std::vector<T> v(int_values.size());
  std::transform(
    int_values.begin(), int_values.end(), v.begin(), [](int x) { return convert_int<T>(x); });
  return v;
}

template <typename T>
cudf::test::fixed_width_column_wrapper<T> construct_null_column(std::vector<T> const& values,
                                                                std::vector<bool> const& bools)
{
  if (values.size() > bools.size()) { throw std::logic_error("input vector size mismatch."); }
  return cudf::test::fixed_width_column_wrapper<T>(values.begin(), values.end(), bools.begin());
}

template <typename T>
std::vector<T> replace_nulls(std::vector<T> const& values,
                             std::vector<bool> const& bools,
                             T identity)
{
  std::vector<T> v(values.size());
  std::transform(values.begin(), values.end(), bools.begin(), v.begin(), [identity](T x, bool b) {
    return (b) ? x : identity;
  });
  return v;
}

// ------------------------------------------------------------------------

// This is the main test feature
template <typename T>
struct ReductionTest : public cudf::test::BaseFixture {
  // Sum/Prod/SumOfSquare never support non arithmetics
  static constexpr bool ret_non_arithmetic = (std::is_arithmetic_v<T> || std::is_same_v<T, bool>);

  ReductionTest() {}

  ~ReductionTest() override {}

  template <typename T_out>
  std::pair<T_out, bool> reduction_test(cudf::column_view const& underlying_column,
                                        reduce_aggregation const& agg,
                                        std::optional<cudf::data_type> _output_dtype = {})
  {
    auto const output_dtype                 = _output_dtype.value_or(underlying_column.type());
    std::unique_ptr<cudf::scalar> reduction = cudf::reduce(underlying_column, agg, output_dtype);
    using ScalarType                        = cudf::scalar_type_t<T_out>;
    auto result                             = static_cast<ScalarType*>(reduction.get());
    return {result->value(), result->is_valid()};
  }

  // Test with initial value
  template <typename T_out>
  std::pair<T_out, bool> reduction_test(cudf::column_view const& underlying_column,
                                        cudf::scalar const& initial_value,
                                        reduce_aggregation const& agg,
                                        std::optional<cudf::data_type> _output_dtype = {})
  {
    auto const output_dtype = _output_dtype.value_or(underlying_column.type());
    std::unique_ptr<cudf::scalar> reduction =
      cudf::reduce(underlying_column, agg, output_dtype, initial_value);
    using ScalarType = cudf::scalar_type_t<T_out>;
    auto result      = static_cast<ScalarType*>(reduction.get());
    return {result->value(), result->is_valid()};
  }
};

template <typename T>
struct MinMaxReductionTest : public ReductionTest<T> {};

using MinMaxTypes = cudf::test::Types<int32_t, int64_t, float, double>;
TYPED_TEST_SUITE(MinMaxReductionTest, MinMaxTypes);

// ------------------------------------------------------------------------
TYPED_TEST(MinMaxReductionTest, MinMaxReductions)
{
  using T = TypeParam;
  std::vector<int> int_values({5, 0, -120, -111, 0, 64, 63, 99, 123, -16});
  std::vector<bool> host_bools({true, true, false, true, true, true, false, true, false, true});
  std::vector<bool> all_null(
    {false, false, false, false, false, false, false, false, false, false});
  std::vector<T> v       = convert_values<T>(int_values);
  T init_value           = convert_int<T>(100);
  auto const init_scalar = cudf::make_fixed_width_scalar<T>(init_value);

  // Min/Max succeeds for any gdf types including
  // non-arithmetic types (date32, date64, timestamp, category)

  // test without nulls
  cudf::test::fixed_width_column_wrapper<T> col(v.begin(), v.end());

  T expected_min_result      = *(std::min_element(v.begin(), v.end()));
  T expected_max_result      = *(std::max_element(v.begin(), v.end()));
  T expected_min_init_result = std::accumulate(
    v.begin(), v.end(), init_value, [](T const& a, T const& b) { return std::min<T>(a, b); });
  T expected_max_init_result = std::accumulate(
    v.begin(), v.end(), init_value, [](T const& a, T const& b) { return std::max<T>(a, b); });

  EXPECT_EQ(
    this->template reduction_test<T>(col, *cudf::make_min_aggregation<reduce_aggregation>()).first,
    expected_min_result);
  EXPECT_EQ(
    this->template reduction_test<T>(col, *cudf::make_max_aggregation<reduce_aggregation>()).first,
    expected_max_result);
  EXPECT_EQ(this
              ->template reduction_test<T>(
                col, *init_scalar, *cudf::make_min_aggregation<reduce_aggregation>())
              .first,
            expected_min_init_result);
  EXPECT_EQ(this
              ->template reduction_test<T>(
                col, *init_scalar, *cudf::make_max_aggregation<reduce_aggregation>())
              .first,
            expected_max_init_result);

  auto res = cudf::minmax(col);

  using ScalarType = cudf::scalar_type_t<T>;
  auto min_result  = static_cast<ScalarType*>(res.first.get());
  auto max_result  = static_cast<ScalarType*>(res.second.get());
  EXPECT_EQ(T{min_result->value()}, expected_min_result);
  EXPECT_EQ(T{max_result->value()}, expected_max_result);

  // test with some nulls
  cudf::test::fixed_width_column_wrapper<T> col_nulls = construct_null_column(v, host_bools);

  auto r_min = replace_nulls(v, host_bools, std::numeric_limits<T>::max());
  auto r_max = replace_nulls(v, host_bools, std::numeric_limits<T>::lowest());

  T expected_min_null_result = *(std::min_element(r_min.begin(), r_min.end()));
  T expected_max_null_result = *(std::max_element(r_max.begin(), r_max.end()));
  T expected_min_init_null_result =
    std::accumulate(r_min.begin(), r_min.end(), init_value, [](T const& a, T const& b) {
      return std::min<T>(a, b);
    });
  T expected_max_init_null_result =
    std::accumulate(r_max.begin(), r_max.end(), init_value, [](T const& a, T const& b) {
      return std::max<T>(a, b);
    });

  EXPECT_EQ(
    this->template reduction_test<T>(col_nulls, *cudf::make_min_aggregation<reduce_aggregation>())
      .first,
    expected_min_null_result);
  EXPECT_EQ(
    this->template reduction_test<T>(col_nulls, *cudf::make_max_aggregation<reduce_aggregation>())
      .first,
    expected_max_null_result);
  EXPECT_EQ(this
              ->template reduction_test<T>(
                col_nulls, *init_scalar, *cudf::make_min_aggregation<reduce_aggregation>())
              .first,
            expected_min_init_null_result);
  EXPECT_EQ(this
              ->template reduction_test<T>(
                col_nulls, *init_scalar, *cudf::make_max_aggregation<reduce_aggregation>())
              .first,
            expected_max_init_null_result);

  auto null_res = cudf::minmax(col_nulls);

  using ScalarType     = cudf::scalar_type_t<T>;
  auto min_null_result = static_cast<ScalarType*>(null_res.first.get());
  auto max_null_result = static_cast<ScalarType*>(null_res.second.get());
  EXPECT_EQ(T{min_null_result->value()}, expected_min_null_result);
  EXPECT_EQ(T{max_null_result->value()}, expected_max_null_result);

  // test with all null
  cudf::test::fixed_width_column_wrapper<T> col_all_nulls = construct_null_column(v, all_null);
  init_scalar->set_valid_async(false);

  EXPECT_FALSE(
    this
      ->template reduction_test<T>(col_all_nulls, *cudf::make_min_aggregation<reduce_aggregation>())
      .second);
  EXPECT_FALSE(
    this
      ->template reduction_test<T>(col_all_nulls, *cudf::make_max_aggregation<reduce_aggregation>())
      .second);
  EXPECT_FALSE(this
                 ->template reduction_test<T>(
                   col_all_nulls, *init_scalar, *cudf::make_min_aggregation<reduce_aggregation>())
                 .second);
  EXPECT_FALSE(this
                 ->template reduction_test<T>(
                   col_all_nulls, *init_scalar, *cudf::make_max_aggregation<reduce_aggregation>())
                 .second);

  auto all_null_res = cudf::minmax(col_all_nulls);

  using ScalarType         = cudf::scalar_type_t<T>;
  auto min_all_null_result = static_cast<ScalarType*>(all_null_res.first.get());
  auto max_all_null_result = static_cast<ScalarType*>(all_null_res.second.get());
  EXPECT_EQ(min_all_null_result->is_valid(), false);
  EXPECT_EQ(max_all_null_result->is_valid(), false);
}

TYPED_TEST(MinMaxReductionTest, ArgMinMaxReductions)
{
  using T = TypeParam;

  // Data copied from MinMaxReductions test above.
  std::vector<int> int_values({5, 0, -120, -111, 0, 64, 63, 99, 123, -16});
  std::vector<bool> host_bools({true, true, false, true, true, true, false, true, false, true});
  std::vector<bool> all_null(
    {false, false, false, false, false, false, false, false, false, false});
  std::vector<T> v = convert_values<T>(int_values);

  auto const argmin = [](std::vector<T> const& input) -> int {
    if (input.empty()) { return -1; }
    auto const it = std::min_element(input.begin(), input.end());
    return static_cast<int>(std::distance(input.begin(), it));
  };
  auto const argmax = [](std::vector<T> const& input) -> int {
    if (input.empty()) { return -1; }
    auto const it = std::max_element(input.begin(), input.end());
    return static_cast<int>(std::distance(input.begin(), it));
  };

  // Test without nulls.
  {
    cudf::test::fixed_width_column_wrapper<T> col(v.begin(), v.end());
    auto const expected_argmin = argmin(v);
    auto const expected_argmax = argmax(v);
    EXPECT_EQ(this
                ->template reduction_test<int>(col,
                                               *cudf::make_argmin_aggregation<reduce_aggregation>(),
                                               cudf::data_type{cudf::type_id::INT32})
                .first,
              expected_argmin);
    EXPECT_EQ(this
                ->template reduction_test<int>(col,
                                               *cudf::make_argmax_aggregation<reduce_aggregation>(),
                                               cudf::data_type{cudf::type_id::INT32})
                .first,
              expected_argmax);
  }

  // Test with some nulls.
  {
    cudf::test::fixed_width_column_wrapper<T> col_nulls = construct_null_column(v, host_bools);
    auto r_min                 = replace_nulls(v, host_bools, std::numeric_limits<T>::max());
    auto r_max                 = replace_nulls(v, host_bools, std::numeric_limits<T>::lowest());
    auto const expected_argmin = argmin(r_min);
    auto const expected_argmax = argmax(r_max);
    EXPECT_EQ(this
                ->template reduction_test<int>(col_nulls,
                                               *cudf::make_argmin_aggregation<reduce_aggregation>(),
                                               cudf::data_type{cudf::type_id::INT32})
                .first,
              expected_argmin);
    EXPECT_EQ(this
                ->template reduction_test<int>(col_nulls,
                                               *cudf::make_argmax_aggregation<reduce_aggregation>(),
                                               cudf::data_type{cudf::type_id::INT32})
                .first,
              expected_argmax);
  }

  // Test with all nulls.
  {
    cudf::test::fixed_width_column_wrapper<T> col_all_nulls = construct_null_column(v, all_null);
    EXPECT_FALSE(
      this
        ->template reduction_test<int>(col_all_nulls,
                                       *cudf::make_argmin_aggregation<reduce_aggregation>(),
                                       cudf::data_type{cudf::type_id::INT32})
        .second);
    EXPECT_FALSE(
      this
        ->template reduction_test<int>(col_all_nulls,
                                       *cudf::make_argmax_aggregation<reduce_aggregation>(),
                                       cudf::data_type{cudf::type_id::INT32})
        .second);
  }
}

template <typename T>
struct SumReductionTest : public ReductionTest<T> {};
using SumTypes = cudf::test::Types<int16_t, int32_t, float, double>;
TYPED_TEST_SUITE(SumReductionTest, SumTypes);

TYPED_TEST(SumReductionTest, Sum)
{
  using T = TypeParam;
  std::vector<int> int_values({6, -14, 13, 64, 0, -13, -20, 45});
  std::vector<bool> host_bools({true, true, false, false, true, true, true, true});
  std::vector<T> v       = convert_values<T>(int_values);
  T init_value           = convert_int<T>(100);
  auto const init_scalar = cudf::make_fixed_width_scalar<T>(init_value);

  // test without nulls
  cudf::test::fixed_width_column_wrapper<T> col(v.begin(), v.end());
  T expected_value      = std::accumulate(v.begin(), v.end(), T{0});
  T expected_value_init = std::accumulate(v.begin(), v.end(), init_value);

  EXPECT_EQ(
    this->template reduction_test<T>(col, *cudf::make_sum_aggregation<reduce_aggregation>()).first,
    expected_value);
  EXPECT_EQ(this
              ->template reduction_test<T>(
                col, *init_scalar, *cudf::make_sum_aggregation<reduce_aggregation>())
              .first,
            expected_value_init);

  // test with nulls
  cudf::test::fixed_width_column_wrapper<T> col_nulls = construct_null_column(v, host_bools);
  auto r                                              = replace_nulls(v, host_bools, T{0});
  T expected_null_value                               = std::accumulate(r.begin(), r.end(), T{0});
  init_scalar->set_valid_async(false);

  EXPECT_EQ(
    this->template reduction_test<T>(col_nulls, *cudf::make_sum_aggregation<reduce_aggregation>())
      .first,
    expected_null_value);
  EXPECT_FALSE(this
                 ->template reduction_test<T>(
                   col_nulls, *init_scalar, *cudf::make_sum_aggregation<reduce_aggregation>())
                 .second);
}

TYPED_TEST_SUITE(ReductionTest, cudf::test::NumericTypes);

TYPED_TEST(ReductionTest, Product)
{
  using T = TypeParam;

  std::vector<int> int_values({5, -1, 1, 0, 3, 2, 4});
  std::vector<bool> host_bools({true, true, false, false, true, true, true});
  std::vector<TypeParam> v = convert_values<TypeParam>(int_values);
  T init_value             = convert_int<T>(4);
  auto const init_scalar   = cudf::make_fixed_width_scalar<T>(init_value);

  auto calc_prod = [](std::vector<T>& v) {
    T expected_value = std::accumulate(v.begin(), v.end(), T{1}, std::multiplies<T>());
    return expected_value;
  };

  auto calc_prod_init = [](std::vector<T>& v, T init) {
    T expected_value = std::accumulate(v.begin(), v.end(), init, std::multiplies<T>());
    return expected_value;
  };

  // test without nulls
  cudf::test::fixed_width_column_wrapper<T> col(v.begin(), v.end());
  TypeParam expected_value      = calc_prod(v);
  TypeParam expected_value_init = calc_prod_init(v, init_value);

  EXPECT_EQ(
    this->template reduction_test<T>(col, *cudf::make_product_aggregation<reduce_aggregation>())
      .first,
    expected_value);
  EXPECT_EQ(this
              ->template reduction_test<T>(
                col, *init_scalar, *cudf::make_product_aggregation<reduce_aggregation>())
              .first,
            expected_value_init);

  // test with nulls
  cudf::test::fixed_width_column_wrapper<T> col_nulls = construct_null_column(v, host_bools);
  auto r                                              = replace_nulls(v, host_bools, T{1});
  TypeParam expected_null_value                       = calc_prod(r);
  init_scalar->set_valid_async(false);

  EXPECT_EQ(
    this
      ->template reduction_test<T>(col_nulls, *cudf::make_product_aggregation<reduce_aggregation>())
      .first,
    expected_null_value);
  EXPECT_FALSE(this
                 ->template reduction_test<T>(
                   col_nulls, *init_scalar, *cudf::make_product_aggregation<reduce_aggregation>())
                 .second);
}

TYPED_TEST(ReductionTest, SumOfSquare)
{
  using T = TypeParam;
  std::vector<int> int_values({-3, 2, 1, 0, 5, -3, -2});
  std::vector<bool> host_bools({true, true, false, false, true, true, true, true});
  std::vector<T> v = convert_values<T>(int_values);

  auto calc_reduction = [](std::vector<T>& v) {
    T value = std::accumulate(v.begin(), v.end(), T{0}, [](T acc, T i) { return acc + i * i; });
    return value;
  };

  // test without nulls
  cudf::test::fixed_width_column_wrapper<T> col(v.begin(), v.end());
  T expected_value = calc_reduction(v);

  EXPECT_EQ(this
              ->template reduction_test<T>(
                col, *cudf::make_sum_of_squares_aggregation<reduce_aggregation>())
              .first,
            expected_value);

  // test with nulls
  cudf::test::fixed_width_column_wrapper<T> col_nulls = construct_null_column(v, host_bools);
  auto r                                              = replace_nulls(v, host_bools, T{0});
  T expected_null_value                               = calc_reduction(r);

  EXPECT_EQ(this
              ->template reduction_test<T>(
                col_nulls, *cudf::make_sum_of_squares_aggregation<reduce_aggregation>())
              .first,
            expected_null_value);
}

auto histogram_reduction(cudf::column_view const& input,
                         std::unique_ptr<cudf::reduce_aggregation> const& agg)
{
  CUDF_EXPECTS(
    agg->kind == cudf::aggregation::HISTOGRAM || agg->kind == cudf::aggregation::MERGE_HISTOGRAM,
    "Aggregation must be either HISTOGRAM or MERGE_HISTOGRAM.");

  auto const result_scalar = cudf::reduce(input, *agg, cudf::data_type{cudf::type_id::INT64});
  EXPECT_EQ(result_scalar->is_valid(), true);

  auto const result_list_scalar = dynamic_cast<cudf::list_scalar*>(result_scalar.get());
  EXPECT_NE(result_list_scalar, nullptr);

  auto const histogram = result_list_scalar->view();
  EXPECT_EQ(histogram.num_children(), 2);
  EXPECT_EQ(histogram.null_count(), 0);
  EXPECT_EQ(histogram.child(1).null_count(), 0);

  // Sort the histogram based on the first column (unique input values).
  auto const sort_order = cudf::sorted_order(cudf::table_view{{histogram.child(0)}}, {}, {});
  return std::move(cudf::gather(cudf::table_view{{histogram}}, *sort_order)->release().front());
}

template <typename T>
struct ReductionHistogramTest : public cudf::test::BaseFixture {};

// Avoid unsigned types, as the tests below have negative values in their input.
using HistogramTestTypes = cudf::test::Concat<cudf::test::Types<int8_t, int16_t, int32_t, int64_t>,
                                              cudf::test::FloatingPointTypes,
                                              cudf::test::FixedPointTypes,
                                              cudf::test::ChronoTypes>;
TYPED_TEST_SUITE(ReductionHistogramTest, HistogramTestTypes);

TYPED_TEST(ReductionHistogramTest, Histogram)
{
  using data_col    = cudf::test::fixed_width_column_wrapper<TypeParam, int>;
  using int64_col   = cudf::test::fixed_width_column_wrapper<int64_t>;
  using structs_col = cudf::test::structs_column_wrapper;

  auto const agg = cudf::make_histogram_aggregation<reduce_aggregation>();

  // Empty input.
  {
    auto const input    = data_col{};
    auto const expected = [] {
      auto child1 = data_col{};
      auto child2 = int64_col{};
      return structs_col{{child1, child2}};
    }();
    auto const result = histogram_reduction(input, agg);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *result);
  }

  {
    auto const input    = data_col{-3, 2, 1, 2, 0, 5, 2, -3, -2, 2, 1};
    auto const expected = [] {
      auto child1 = data_col{-3, -2, 0, 1, 2, 5};
      auto child2 = int64_col{2, 1, 1, 2, 4, 1};
      return structs_col{{child1, child2}};
    }();
    auto const result = histogram_reduction(input, agg);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *result);
  }

  // Test without nulls, sliced input.
  {
    auto const input_original = data_col{-3, 2, 1, 2, 0, 5, 2, -3, -2, 2, 1};
    auto const input          = cudf::slice(input_original, {0, 7})[0];
    auto const expected       = [] {
      auto child1 = data_col{-3, 0, 1, 2, 5};
      auto child2 = int64_col{1, 1, 1, 3, 1};
      return structs_col{{child1, child2}};
    }();
    auto const result = histogram_reduction(input, agg);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *result);
  }

  // Test with nulls.
  using namespace cudf::test::iterators;
  auto constexpr null{0};
  {
    auto const input    = data_col{{null, -3, 2, 1, 2, 0, null, 5, 2, null, -3, -2, null, 2, 1},
                                nulls_at({0, 6, 9, 12})};
    auto const expected = [] {
      auto child1 = data_col{{null, -3, -2, 0, 1, 2, 5}, null_at(0)};
      auto child2 = int64_col{4, 2, 1, 1, 2, 4, 1};
      return structs_col{{child1, child2}};
    }();
    auto const result = histogram_reduction(input, agg);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *result);
  }

  // Test with nulls, sliced input.
  {
    auto const input_original = data_col{
      {null, -3, 2, 1, 2, 0, null, 5, 2, null, -3, -2, null, 2, 1}, nulls_at({0, 6, 9, 12})};
    auto const input    = cudf::slice(input_original, {0, 9})[0];
    auto const expected = [] {
      auto child1 = data_col{{null, -3, 0, 1, 2, 5}, null_at(0)};
      auto child2 = int64_col{2, 1, 1, 1, 3, 1};
      return structs_col{{child1, child2}};
    }();
    auto const result = histogram_reduction(input, agg);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *result);
  }
}

TYPED_TEST(ReductionHistogramTest, MergeHistogram)
{
  using data_col    = cudf::test::fixed_width_column_wrapper<TypeParam>;
  using int64_col   = cudf::test::fixed_width_column_wrapper<int64_t>;
  using structs_col = cudf::test::structs_column_wrapper;

  auto const agg = cudf::make_merge_histogram_aggregation<reduce_aggregation>();

  // Empty input.
  {
    auto const input = [] {
      auto child1 = data_col{};
      auto child2 = int64_col{};
      return structs_col{{child1, child2}};
    }();
    auto const expected = [] {
      auto child1 = data_col{};
      auto child2 = int64_col{};
      return structs_col{{child1, child2}};
    }();
    auto const result = histogram_reduction(input, agg);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *result);
  }

  // Test without nulls.
  {
    auto const input = [] {
      auto child1 = data_col{-3, 2, 1, 2, 0, 5, 2, -3, -2, 2, 1};
      auto child2 = int64_col{2, 1, 1, 2, 4, 1, 2, 3, 5, 3, 4};
      return structs_col{{child1, child2}};
    }();

    auto const expected = [] {
      auto child1 = data_col{-3, -2, 0, 1, 2, 5};
      auto child2 = int64_col{5, 5, 4, 5, 8, 1};
      return structs_col{{child1, child2}};
    }();
    auto const result = histogram_reduction(input, agg);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *result);
  }

  // Test without nulls, sliced input.
  {
    auto const input_original = [] {
      auto child1 = data_col{-3, 2, 1, 2, 0, 5, 2, -3, -2, 2, 1};
      auto child2 = int64_col{2, 1, 1, 2, 4, 1, 2, 3, 5, 3, 4};
      return structs_col{{child1, child2}};
    }();
    auto const input = cudf::slice(input_original, {0, 7})[0];

    auto const expected = [] {
      auto child1 = data_col{-3, 0, 1, 2, 5};
      auto child2 = int64_col{2, 4, 1, 5, 1};
      return structs_col{{child1, child2}};
    }();
    auto const result = histogram_reduction(input, agg);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *result);
  }

  // Test with nulls.
  using namespace cudf::test::iterators;
  auto constexpr null{0};
  {
    auto const input = [] {
      auto child1 = data_col{{-3, 2, null, 1, 2, null, 0, 5, null, 2, -3, null, -2, 2, 1, null},
                             nulls_at({2, 5, 8, 11, 15})};
      auto child2 = int64_col{2, 1, 12, 1, 2, 11, 4, 1, 10, 2, 3, 15, 5, 3, 4, 19};
      return structs_col{{child1, child2}};
    }();

    auto const expected = [] {
      auto child1 = data_col{{null, -3, -2, 0, 1, 2, 5}, null_at(0)};
      auto child2 = int64_col{67, 5, 5, 4, 5, 8, 1};
      return structs_col{{child1, child2}};
    }();
    auto const result = histogram_reduction(input, agg);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *result);
  }

  // Test with nulls, sliced input.
  {
    auto const input_original = [] {
      auto child1 = data_col{{-3, 2, null, 1, 2, null, 0, 5, null, 2, -3, null, -2, 2, 1, null},
                             nulls_at({2, 5, 8, 11, 15})};
      auto child2 = int64_col{2, 1, 12, 1, 2, 11, 4, 1, 10, 2, 3, 15, 5, 3, 4, 19};
      return structs_col{{child1, child2}};
    }();
    auto const input = cudf::slice(input_original, {0, 9})[0];

    auto const expected = [] {
      auto child1 = data_col{{null, -3, 0, 1, 2, 5}, null_at(0)};
      auto child2 = int64_col{33, 2, 4, 1, 3, 1};
      return structs_col{{child1, child2}};
    }();
    auto const result = histogram_reduction(input, agg);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *result);
  }
}

template <typename T>
struct ReductionAnyAllTest : public ReductionTest<bool> {};
using AnyAllTypes = cudf::test::Types<int32_t, float, bool>;
TYPED_TEST_SUITE(ReductionAnyAllTest, AnyAllTypes);

TYPED_TEST(ReductionAnyAllTest, AnyAllTrueTrue)
{
  using T = TypeParam;
  std::vector<int> int_values({true, true, true, true});
  std::vector<bool> host_bools({true, true, false, true});
  std::vector<T> v       = convert_values<T>(int_values);
  auto const init_scalar = cudf::make_fixed_width_scalar<T>(convert_int<T>(true));

  // Min/Max succeeds for any gdf types including
  // non-arithmetic types (date32, date64, timestamp, category)
  bool expected = true;
  cudf::data_type output_dtype(cudf::type_id::BOOL8);

  // test without nulls
  cudf::test::fixed_width_column_wrapper<T> col(v.begin(), v.end());

  EXPECT_EQ(this
              ->template reduction_test<bool>(
                col, *cudf::make_any_aggregation<reduce_aggregation>(), output_dtype)
              .first,
            expected);
  EXPECT_EQ(this
              ->template reduction_test<bool>(
                col, *cudf::make_all_aggregation<reduce_aggregation>(), output_dtype)
              .first,
            expected);
  EXPECT_EQ(this
              ->template reduction_test<bool>(
                col, *init_scalar, *cudf::make_any_aggregation<reduce_aggregation>(), output_dtype)
              .first,
            expected);
  EXPECT_EQ(this
              ->template reduction_test<bool>(
                col, *init_scalar, *cudf::make_all_aggregation<reduce_aggregation>(), output_dtype)
              .first,
            expected);

  // test with nulls
  cudf::test::fixed_width_column_wrapper<T> col_nulls = construct_null_column(v, host_bools);
  init_scalar->set_valid_async(false);

  EXPECT_EQ(this
              ->template reduction_test<bool>(
                col_nulls, *cudf::make_any_aggregation<reduce_aggregation>(), output_dtype)
              .first,
            expected);
  EXPECT_EQ(this
              ->template reduction_test<bool>(
                col_nulls, *cudf::make_all_aggregation<reduce_aggregation>(), output_dtype)
              .first,
            expected);
  EXPECT_FALSE(
    this
      ->template reduction_test<bool>(
        col_nulls, *init_scalar, *cudf::make_any_aggregation<reduce_aggregation>(), output_dtype)
      .second);
  EXPECT_FALSE(
    this
      ->template reduction_test<bool>(
        col_nulls, *init_scalar, *cudf::make_all_aggregation<reduce_aggregation>(), output_dtype)
      .second);
}

TYPED_TEST(ReductionAnyAllTest, AnyAllFalseFalse)
{
  using T = TypeParam;
  std::vector<int> int_values({false, false, false, false});
  std::vector<bool> host_bools({true, true, false, true});
  std::vector<T> v       = convert_values<T>(int_values);
  auto const init_scalar = cudf::make_fixed_width_scalar<T>(convert_int<T>(false));

  // Min/Max succeeds for any gdf types including
  // non-arithmetic types (date32, date64, timestamp, category)
  bool expected = false;
  cudf::data_type output_dtype(cudf::type_id::BOOL8);

  // test without nulls
  cudf::test::fixed_width_column_wrapper<T> col(v.begin(), v.end());

  EXPECT_EQ(this
              ->template reduction_test<bool>(
                col, *cudf::make_any_aggregation<reduce_aggregation>(), output_dtype)
              .first,
            expected);
  EXPECT_EQ(this
              ->template reduction_test<bool>(
                col, *cudf::make_all_aggregation<reduce_aggregation>(), output_dtype)
              .first,
            expected);
  EXPECT_EQ(this
              ->template reduction_test<bool>(
                col, *init_scalar, *cudf::make_any_aggregation<reduce_aggregation>(), output_dtype)
              .first,
            expected);
  EXPECT_EQ(this
              ->template reduction_test<bool>(
                col, *init_scalar, *cudf::make_all_aggregation<reduce_aggregation>(), output_dtype)
              .first,
            expected);

  // test with nulls
  cudf::test::fixed_width_column_wrapper<T> col_nulls = construct_null_column(v, host_bools);
  init_scalar->set_valid_async(false);

  EXPECT_EQ(this
              ->template reduction_test<bool>(
                col_nulls, *cudf::make_any_aggregation<reduce_aggregation>(), output_dtype)
              .first,
            expected);
  EXPECT_EQ(this
              ->template reduction_test<bool>(
                col_nulls, *cudf::make_all_aggregation<reduce_aggregation>(), output_dtype)
              .first,
            expected);
  EXPECT_FALSE(
    this
      ->template reduction_test<bool>(
        col_nulls, *init_scalar, *cudf::make_any_aggregation<reduce_aggregation>(), output_dtype)
      .second);
  EXPECT_FALSE(
    this
      ->template reduction_test<bool>(
        col_nulls, *init_scalar, *cudf::make_all_aggregation<reduce_aggregation>(), output_dtype)
      .second);
}

// ----------------------------------------------------------------------------

template <typename T>
struct MultiStepReductionTest : public ReductionTest<T> {};
using MultiStepReductionTypes = cudf::test::Types<int16_t, int32_t, float, double>;
TYPED_TEST_SUITE(MultiStepReductionTest, MultiStepReductionTypes);

TYPED_TEST(MultiStepReductionTest, Mean)
{
  using T = TypeParam;
  std::vector<int> int_values({-3, 2, 1, 0, 5, -3, -2, 28});
  std::vector<bool> host_bools({true, true, false, true, true, true, false, true});

  auto calc_mean = [](std::vector<T>& v, cudf::size_type valid_count) {
    double sum = std::accumulate(v.begin(), v.end(), double{0});
    return sum / valid_count;
  };

  // test without nulls
  std::vector<T> v = convert_values<T>(int_values);
  cudf::test::fixed_width_column_wrapper<T> col(v.begin(), v.end());
  double expected_value = calc_mean(v, v.size());

  EXPECT_EQ(this
              ->template reduction_test<double>(col,
                                                *cudf::make_mean_aggregation<reduce_aggregation>(),
                                                cudf::data_type(cudf::type_id::FLOAT64))
              .first,
            expected_value);

  // test with nulls
  cudf::test::fixed_width_column_wrapper<T> col_nulls = construct_null_column(v, host_bools);
  cudf::size_type valid_count =
    cudf::column_view(col_nulls).size() - cudf::column_view(col_nulls).null_count();
  auto replaced_array = replace_nulls(v, host_bools, T{0});

  double expected_value_nulls = calc_mean(replaced_array, valid_count);

  EXPECT_EQ(this
              ->template reduction_test<double>(col_nulls,
                                                *cudf::make_mean_aggregation<reduce_aggregation>(),
                                                cudf::data_type(cudf::type_id::FLOAT64))
              .first,
            expected_value_nulls);
}

template <typename T>
double calc_var(std::vector<T> const& v, int ddof, std::vector<bool> const& mask = {})
{
  auto const values = [&]() {
    if (mask.empty()) { return v; }
    std::vector<T> masked{};
    thrust::copy_if(
      v.begin(), v.end(), mask.begin(), std::back_inserter(masked), [](auto m) { return m; });
    return masked;
  }();
  auto const valid_count = values.size();
  double const mean      = std::accumulate(values.cbegin(), values.cend(), double{0}) / valid_count;
  double const sq_sum_of_differences =
    std::accumulate(values.cbegin(), values.cend(), double{0}, [mean](double acc, auto const v) {
      return acc + std::pow(v - mean, 2);
    });
  return sq_sum_of_differences / (valid_count - ddof);
}

TYPED_TEST(MultiStepReductionTest, var_std)
{
  using T = TypeParam;
  std::vector<int> int_values({-3, 2, 1, 0, 5, -3, -2, 28});
  std::vector<bool> host_bools({true, true, false, true, true, true, false, true});

  // test without nulls
  std::vector<T> v = convert_values<T>(int_values);
  cudf::test::fixed_width_column_wrapper<T> col(v.begin(), v.end());

  auto const ddof = 1;
  double var      = calc_var(v, ddof);
  double std      = std::sqrt(var);
  auto var_agg    = cudf::make_variance_aggregation<reduce_aggregation>(ddof);
  auto std_agg    = cudf::make_std_aggregation<reduce_aggregation>(ddof);

  EXPECT_EQ(
    this->template reduction_test<double>(col, *var_agg, cudf::data_type(cudf::type_id::FLOAT64))
      .first,
    var);
  EXPECT_EQ(
    this->template reduction_test<double>(col, *std_agg, cudf::data_type(cudf::type_id::FLOAT64))
      .first,
    std);

  // test with nulls
  cudf::test::fixed_width_column_wrapper<T> col_nulls = construct_null_column(v, host_bools);
  double var_nulls                                    = calc_var(v, ddof, host_bools);
  double std_nulls                                    = std::sqrt(var_nulls);

  EXPECT_DOUBLE_EQ(this
                     ->template reduction_test<double>(
                       col_nulls, *var_agg, cudf::data_type(cudf::type_id::FLOAT64))
                     .first,
                   var_nulls);
  EXPECT_DOUBLE_EQ(this
                     ->template reduction_test<double>(
                       col_nulls, *std_agg, cudf::data_type(cudf::type_id::FLOAT64))
                     .first,
                   std_nulls);
}

// ----------------------------------------------------------------------------

template <typename T>
struct ReductionMultiStepErrorCheck : public ReductionTest<T> {
  void reduction_error_check(cudf::test::fixed_width_column_wrapper<T>& col,
                             bool succeeded_condition,
                             reduce_aggregation const& agg,
                             cudf::data_type output_dtype)
  {
    cudf::column_view const underlying_column = col;
    auto statement = [&]() { cudf::reduce(underlying_column, agg, output_dtype); };

    if (succeeded_condition) {
      CUDF_EXPECT_NO_THROW(statement());
    } else {
      EXPECT_ANY_THROW(statement());
    }
  }
};

TYPED_TEST_SUITE(ReductionMultiStepErrorCheck, cudf::test::AllTypes);

TYPED_TEST(ReductionMultiStepErrorCheck, ErrorHandling)
{
  using T = TypeParam;
  std::vector<int> int_values({-3, 2});
  std::vector<bool> host_bools({true, false});

  std::vector<T> v = convert_values<T>(int_values);
  cudf::test::fixed_width_column_wrapper<T> col(v.begin(), v.end());
  cudf::test::fixed_width_column_wrapper<T> col_nulls = construct_null_column(v, host_bools);

  bool is_input_acceptable = this->ret_non_arithmetic;

  std::vector<cudf::data_type> dtypes(static_cast<int32_t>(cudf::type_id::NUM_TYPE_IDS) + 1);
  int i = 0;
  std::generate(dtypes.begin(), dtypes.end(), [&]() {
    return cudf::data_type(static_cast<cudf::type_id>(i++));
  });

  auto is_supported_outdtype = [](cudf::data_type dtype) {
    if (dtype == cudf::data_type(cudf::type_id::FLOAT32)) return true;
    if (dtype == cudf::data_type(cudf::type_id::FLOAT64)) return true;
    return false;
  };

  auto evaluate = [&](cudf::data_type dtype) mutable {
    bool expect_succeed = is_input_acceptable & is_supported_outdtype(dtype);
    auto const ddof     = 1;
    auto var_agg        = cudf::make_variance_aggregation<reduce_aggregation>(ddof);
    auto std_agg        = cudf::make_std_aggregation<reduce_aggregation>(ddof);
    this->reduction_error_check(
      col, expect_succeed, *cudf::make_mean_aggregation<reduce_aggregation>(), dtype);
    this->reduction_error_check(col, expect_succeed, *var_agg, dtype);
    this->reduction_error_check(col, expect_succeed, *std_agg, dtype);

    this->reduction_error_check(
      col_nulls, expect_succeed, *cudf::make_mean_aggregation<reduce_aggregation>(), dtype);
    this->reduction_error_check(col_nulls, expect_succeed, *var_agg, dtype);
    this->reduction_error_check(col_nulls, expect_succeed, *std_agg, dtype);
    return;
  };

  std::for_each(dtypes.begin(), dtypes.end(), evaluate);
}

// ----------------------------------------------------------------------------

struct ReductionDtypeTest : public cudf::test::BaseFixture {
  template <typename T_in, typename T_out>
  void reduction_test(std::vector<int>& int_values,
                      T_out expected_value,
                      bool succeeded_condition,
                      reduce_aggregation const& agg,
                      cudf::data_type out_dtype,
                      bool expected_overflow = false)
  {
    std::vector<T_in> input_values = convert_values<T_in>(int_values);
    cudf::test::fixed_width_column_wrapper<T_in> const col(input_values.begin(),
                                                           input_values.end());

    auto statement = [&]() {
      std::unique_ptr<cudf::scalar> result = cudf::reduce(col, agg, out_dtype);
      using ScalarType                     = cudf::scalar_type_t<T_out>;
      auto result1                         = static_cast<ScalarType*>(result.get());
      if (result1->is_valid() && !expected_overflow) {
        EXPECT_EQ(expected_value, result1->value());
      }
    };

    if (succeeded_condition) {
      CUDF_EXPECT_NO_THROW(statement());
    } else {
      EXPECT_ANY_THROW(statement());
    }
  }
};

TEST_F(ReductionDtypeTest, all_null_output)
{
  auto sum_agg = cudf::make_sum_aggregation<reduce_aggregation>();

  auto const col =
    cudf::test::fixed_point_column_wrapper<int32_t>{
      {0, 0, 0}, {false, false, false}, numeric::scale_type{-2}}
      .release();

  std::unique_ptr<cudf::scalar> result = cudf::reduce(*col, *sum_agg, col->type());
  EXPECT_EQ(result->is_valid(), false);
  EXPECT_EQ(result->type().id(), col->type().id());
  EXPECT_EQ(result->type().scale(), col->type().scale());
}

// test case for different output precision
TEST_F(ReductionDtypeTest, different_precision)
{
  constexpr bool expected_overflow = true;
  std::vector<int> int_values({6, -14, 13, 109, -13, -20, 0, 98, 122, 123});
  int expected_value = std::accumulate(int_values.begin(), int_values.end(), 0);
  auto sum_agg       = cudf::make_sum_aggregation<reduce_aggregation>();

  // over flow
  this->reduction_test<int8_t, int8_t>(int_values,
                                       static_cast<int8_t>(expected_value),
                                       true,
                                       *sum_agg,
                                       cudf::data_type(cudf::type_id::INT8),
                                       expected_overflow);

  this->reduction_test<int8_t, int64_t>(int_values,
                                        static_cast<int64_t>(expected_value),
                                        true,
                                        *sum_agg,
                                        cudf::data_type(cudf::type_id::INT64));

  this->reduction_test<int8_t, double>(int_values,
                                       static_cast<double>(expected_value),
                                       true,
                                       *sum_agg,
                                       cudf::data_type(cudf::type_id::FLOAT64));

  // down cast (over flow)
  this->reduction_test<double, int8_t>(int_values,
                                       static_cast<int8_t>(expected_value),
                                       true,
                                       *sum_agg,
                                       cudf::data_type(cudf::type_id::INT8),
                                       expected_overflow);

  // down cast (no over flow)
  this->reduction_test<double, int16_t>(int_values,
                                        static_cast<int16_t>(expected_value),
                                        true,
                                        *sum_agg,
                                        cudf::data_type(cudf::type_id::INT16));

  // not supported case:
  // wrapper classes other than bool are not convertible
  this->reduction_test<cudf::timestamp_D, cudf::timestamp_s>(
    int_values,
    cudf::timestamp_s{cudf::duration_s(expected_value)},
    false,
    *sum_agg,
    cudf::data_type(cudf::type_id::TIMESTAMP_SECONDS));

  this->reduction_test<cudf::timestamp_s, cudf::timestamp_ns>(
    int_values,
    cudf::timestamp_ns{cudf::duration_ns(expected_value)},
    false,
    *sum_agg,
    cudf::data_type(cudf::type_id::TIMESTAMP_NANOSECONDS));

  this->reduction_test<int8_t, cudf::timestamp_us>(
    int_values,
    cudf::timestamp_us{cudf::duration_us(expected_value)},
    false,
    *sum_agg,
    cudf::data_type(cudf::type_id::TIMESTAMP_MICROSECONDS));

  std::vector<bool> v = convert_values<bool>(int_values);

  // When summing bool values into an non-bool arithmetic type,
  // it's an integer/float sum of ones and zeros.
  int expected = std::accumulate(v.begin(), v.end(), int{0});

  this->reduction_test<bool, int8_t>(int_values,
                                     static_cast<int8_t>(expected),
                                     true,
                                     *sum_agg,
                                     cudf::data_type(cudf::type_id::INT8));
  this->reduction_test<bool, int16_t>(int_values,
                                      static_cast<int16_t>(expected),
                                      true,
                                      *sum_agg,
                                      cudf::data_type(cudf::type_id::INT16));
  this->reduction_test<bool, int32_t>(int_values,
                                      static_cast<int32_t>(expected),
                                      true,
                                      *sum_agg,
                                      cudf::data_type(cudf::type_id::INT32));
  this->reduction_test<bool, int64_t>(int_values,
                                      static_cast<int64_t>(expected),
                                      true,
                                      *sum_agg,
                                      cudf::data_type(cudf::type_id::INT64));
  this->reduction_test<bool, float>(int_values,
                                    static_cast<float>(expected),
                                    true,
                                    *sum_agg,
                                    cudf::data_type(cudf::type_id::FLOAT32));
  this->reduction_test<bool, double>(int_values,
                                     static_cast<double>(expected),
                                     true,
                                     *sum_agg,
                                     cudf::data_type(cudf::type_id::FLOAT64));

  // make sure boolean arithmetic semantics are obeyed when reducing to a bool
  this->reduction_test<bool, bool>(
    int_values, true, true, *sum_agg, cudf::data_type(cudf::type_id::BOOL8));

  this->reduction_test<int32_t, bool>(
    int_values, true, true, *sum_agg, cudf::data_type(cudf::type_id::BOOL8));

  // cudf::timestamp_s and int64_t are not convertible types.
  this->reduction_test<cudf::timestamp_s, int64_t>(int_values,
                                                   static_cast<int64_t>(expected_value),
                                                   false,
                                                   *sum_agg,
                                                   cudf::data_type(cudf::type_id::INT64));
}

struct ReductionEmptyTest : public cudf::test::BaseFixture {};

// test case for empty input cases
TEST_F(ReductionEmptyTest, empty_column)
{
  using T        = int32_t;
  auto statement = [](cudf::column_view const& col) {
    std::unique_ptr<cudf::scalar> result =
      cudf::reduce(col,
                   *cudf::make_sum_aggregation<reduce_aggregation>(),
                   cudf::data_type(cudf::type_id::INT64));
    EXPECT_EQ(result->is_valid(), false);
  };

  // test if the size of input column is zero
  // expect result.is_valid() is false
  std::vector<T> empty_data(0);
  cudf::test::fixed_width_column_wrapper<T> const col0(empty_data.begin(), empty_data.end());
  CUDF_EXPECT_NO_THROW(statement(col0));

  // test if null count is equal or greater than size of input
  // expect result.is_valid() is false
  int col_size = 5;
  std::vector<T> col_data(col_size);
  std::vector<bool> valids(col_size, false);

  cudf::test::fixed_width_column_wrapper<T> col_nulls = construct_null_column(col_data, valids);
  CUDF_EXPECT_NO_THROW(statement(col_nulls));

  auto any_agg   = cudf::make_any_aggregation<cudf::reduce_aggregation>();
  auto all_agg   = cudf::make_all_aggregation<cudf::reduce_aggregation>();
  auto bool_type = cudf::data_type{cudf::type_id::BOOL8};

  auto result = cudf::reduce(col0, *any_agg, bool_type);
  EXPECT_EQ(result->is_valid(), true);
  EXPECT_EQ(dynamic_cast<cudf::numeric_scalar<bool>*>(result.get())->value(), false);
  result = cudf::reduce(col_nulls, *any_agg, bool_type);
  EXPECT_EQ(result->is_valid(), true);
  EXPECT_EQ(dynamic_cast<cudf::numeric_scalar<bool>*>(result.get())->value(), false);

  result = cudf::reduce(col0, *all_agg, bool_type);
  EXPECT_EQ(result->is_valid(), true);
  EXPECT_EQ(dynamic_cast<cudf::numeric_scalar<bool>*>(result.get())->value(), true);
  result = cudf::reduce(col_nulls, *all_agg, bool_type);
  EXPECT_EQ(result->is_valid(), true);
  EXPECT_EQ(dynamic_cast<cudf::numeric_scalar<bool>*>(result.get())->value(), true);

  auto size_data_type = cudf::data_type(cudf::type_to_id<cudf::size_type>());
  auto nunique_agg =
    cudf::make_nunique_aggregation<cudf::reduce_aggregation>(cudf::null_policy::INCLUDE);
  result = cudf::reduce(col0, *nunique_agg, size_data_type);
  EXPECT_EQ(result->is_valid(), false);
  result = cudf::reduce(col_nulls, *nunique_agg, size_data_type);
  EXPECT_EQ(result->is_valid(), true);
  EXPECT_EQ(dynamic_cast<cudf::numeric_scalar<cudf::size_type>*>(result.get())->value(), 1);

  auto double_type = cudf::data_type{cudf::type_id::FLOAT64};
  auto median_agg  = cudf::make_median_aggregation<cudf::reduce_aggregation>();
  result           = cudf::reduce(col0, *median_agg, double_type);
  EXPECT_EQ(result->is_valid(), false);
  result = cudf::reduce(col_nulls, *median_agg, double_type);
  EXPECT_EQ(result->is_valid(), false);

  auto quantile_agg =
    cudf::make_quantile_aggregation<cudf::reduce_aggregation>({0.0}, cudf::interpolation::LINEAR);
  result = cudf::reduce(col0, *quantile_agg, double_type);
  EXPECT_EQ(result->is_valid(), false);
  result = cudf::reduce(col_nulls, *quantile_agg, double_type);
  EXPECT_EQ(result->is_valid(), false);

  auto count_agg =
    cudf::make_count_aggregation<cudf::reduce_aggregation>(cudf::null_policy::INCLUDE);
  result = cudf::reduce(col0, *count_agg, size_data_type);
  EXPECT_EQ(result->is_valid(), true);
  EXPECT_EQ(dynamic_cast<cudf::numeric_scalar<cudf::size_type>*>(result.get())->value(), 0);
  result = cudf::reduce(col_nulls, *count_agg, size_data_type);
  EXPECT_EQ(result->is_valid(), true);
  EXPECT_EQ(dynamic_cast<cudf::numeric_scalar<cudf::size_type>*>(result.get())->value(), col_size);
}

TEST_F(ReductionEmptyTest, Errors)
{
  auto input     = cudf::test::fixed_width_column_wrapper<int32_t>({7, 8, 9});
  auto agg       = cudf::make_nunique_aggregation<cudf::reduce_aggregation>();
  auto bool_type = cudf::data_type{cudf::type_id::BOOL8};
  EXPECT_THROW(cudf::reduce(input, *agg, bool_type), std::invalid_argument);
}

// ----------------------------------------------------------------------------

struct ReductionParamTest : public ReductionTest<double>,
                            public ::testing::WithParamInterface<cudf::size_type> {};

INSTANTIATE_TEST_CASE_P(ddofParam, ReductionParamTest, ::testing::Range(1, 5));

TEST_P(ReductionParamTest, std_var)
{
  int ddof = GetParam();
  std::vector<double> int_values({-3, 2, 1, 0, 5, -3, -2, 28});
  std::vector<bool> host_bools({true, true, false, true, true, true, false, true});

  // test without nulls
  cudf::test::fixed_width_column_wrapper<double> col(int_values.begin(), int_values.end());

  double var   = calc_var(int_values, ddof);
  double std   = std::sqrt(var);
  auto var_agg = cudf::make_variance_aggregation<reduce_aggregation>(ddof);
  auto std_agg = cudf::make_std_aggregation<reduce_aggregation>(ddof);

  EXPECT_EQ(
    this->template reduction_test<double>(col, *var_agg, cudf::data_type(cudf::type_id::FLOAT64))
      .first,
    var);
  EXPECT_EQ(
    this->template reduction_test<double>(col, *std_agg, cudf::data_type(cudf::type_id::FLOAT64))
      .first,
    std);

  // test with nulls
  cudf::test::fixed_width_column_wrapper<double> col_nulls =
    construct_null_column(int_values, host_bools);
  double var_nulls = calc_var(int_values, ddof, host_bools);
  double std_nulls = std::sqrt(var_nulls);

  EXPECT_DOUBLE_EQ(this
                     ->template reduction_test<double>(
                       col_nulls, *var_agg, cudf::data_type(cudf::type_id::FLOAT64))
                     .first,
                   var_nulls);
  EXPECT_DOUBLE_EQ(this
                     ->template reduction_test<double>(
                       col_nulls, *std_agg, cudf::data_type(cudf::type_id::FLOAT64))
                     .first,
                   std_nulls);
}

//-------------------------------------------------------------------
struct StringReductionTest : public cudf::test::BaseFixture,
                             public testing::WithParamInterface<std::vector<std::string>> {
  // Min/Max
  void reduction_test(cudf::column_view const& underlying_column,
                      std::string expected_value,
                      bool succeeded_condition,
                      reduce_aggregation const& agg,
                      cudf::data_type output_dtype = cudf::data_type{})
  {
    if (cudf::data_type{} == output_dtype) output_dtype = underlying_column.type();

    auto statement = [&]() {
      std::unique_ptr<cudf::scalar> result = cudf::reduce(underlying_column, agg, output_dtype);
      using ScalarType                     = cudf::scalar_type_t<cudf::string_view>;
      auto result1                         = static_cast<ScalarType*>(result.get());
      EXPECT_TRUE(result1->is_valid());
      if (result1->is_valid()) {
        EXPECT_EQ(expected_value, result1->to_string())
          << (agg.kind == aggregation::MIN ? "MIN" : "MAX");
      }
    };

    if (succeeded_condition) {
      CUDF_EXPECT_NO_THROW(statement());
    } else {
      EXPECT_ANY_THROW(statement());
    }
  }

  void reduction_test(cudf::column_view const& underlying_column,
                      std::string initial_value,
                      std::string expected_value,
                      bool succeeded_condition,
                      reduce_aggregation const& agg,
                      cudf::data_type output_dtype = cudf::data_type{})
  {
    if (cudf::data_type{} == output_dtype) output_dtype = underlying_column.type();
    auto string_scalar = cudf::make_string_scalar(initial_value);

    auto statement = [&]() {
      std::unique_ptr<cudf::scalar> result =
        cudf::reduce(underlying_column, agg, output_dtype, *string_scalar);
      using ScalarType = cudf::scalar_type_t<cudf::string_view>;
      auto result1     = static_cast<ScalarType*>(result.get());
      EXPECT_TRUE(result1->is_valid());
      if (result1->is_valid()) {
        EXPECT_EQ(expected_value, result1->to_string())
          << (agg.kind == aggregation::MIN ? "MIN" : "MAX");
      }
    };

    if (succeeded_condition) {
      CUDF_EXPECT_NO_THROW(statement());
    } else {
      EXPECT_ANY_THROW(statement());
    }
  }

  // ARGMIN/ARGMAX
  template <typename OutputType>
  void reduction_test(cudf::column_view const& underlying_column,
                      OutputType expected,
                      reduce_aggregation const& agg,
                      cudf::data_type output_dtype = cudf::data_type{cudf::type_id::INT32})
  {
    std::unique_ptr<cudf::scalar> reduction = cudf::reduce(underlying_column, agg, output_dtype);
    using ScalarType                        = cudf::scalar_type_t<OutputType>;
    auto const result                       = static_cast<ScalarType*>(reduction.get());
    EXPECT_EQ(result->value(), expected);
  }
};

// ------------------------------------------------------------------------
std::vector<std::vector<std::string>> string_list{{
  {"one", "two", "three", "four", "five", "six", "seven", "eight", "nine"},
  {"", "two", "three", "four", "five", "six", "seven", "eight", "nine"},
  {"one", "", "three", "four", "five", "six", "seven", "eight", "nine"},
  {"", "", "", "four", "five", "six", "seven", "eight", "nine"},
  {"", "", "", "", "", "", "", "", ""},
  // DeviceMin identity sentinel test cases
  {"\xF7\xBF\xBF\xBF", "", "", "", "", "", "", "", ""},
  {"one", "two", "three", "four", "\xF7\xBF\xBF\xBF", "six", "seven", "eight", "nine"},
  {"one", "two", "\xF7\xBF\xBF\xBF", "four", "five", "six", "seven", "eight", "nine"},
}};
INSTANTIATE_TEST_CASE_P(string_cases, StringReductionTest, testing::ValuesIn(string_list));
TEST_P(StringReductionTest, MinMax)
{
  // data and valid arrays
  std::vector<std::string> host_strings(GetParam());
  std::vector<bool> host_bools({true, false, true, true, true, true, false, false, true});
  std::transform(cuda::counting_iterator<std::size_t>{0},
                 cuda::counting_iterator<std::size_t>{host_strings.size()},
                 host_strings.begin(),
                 [host_strings, host_bools](auto idx) {
                   return host_bools[idx] ? host_strings[idx] : std::string{};
                 });
  bool succeed(true);
  std::string initial_value = "init";

  // all valid string column
  cudf::test::strings_column_wrapper col(host_strings.begin(), host_strings.end());

  std::string expected_min_result = *(std::min_element(host_strings.begin(), host_strings.end()));
  std::string expected_max_result = *(std::max_element(host_strings.begin(), host_strings.end()));
  std::string expected_min_init_result = std::min(expected_min_result, initial_value);
  std::string expected_max_init_result = std::max(expected_max_result, initial_value);

  // string column with nulls
  cudf::test::strings_column_wrapper col_nulls(
    host_strings.begin(), host_strings.end(), host_bools.begin());

  std::vector<std::string> r_strings;
  std::copy_if(host_strings.begin(),
               host_strings.end(),
               std::back_inserter(r_strings),
               [host_bools, i = 0](auto s) mutable { return host_bools[i++]; });

  std::string expected_min_null_result = *(std::min_element(r_strings.begin(), r_strings.end()));
  std::string expected_max_null_result = *(std::max_element(r_strings.begin(), r_strings.end()));
  std::string expected_min_init_null_result = std::min(expected_min_null_result, initial_value);
  std::string expected_max_init_null_result = std::max(expected_max_null_result, initial_value);

  // MIN
  this->reduction_test(
    col, expected_min_result, succeed, *cudf::make_min_aggregation<reduce_aggregation>());
  this->reduction_test(col_nulls,
                       expected_min_null_result,
                       succeed,
                       *cudf::make_min_aggregation<reduce_aggregation>());
  this->reduction_test(col,
                       initial_value,
                       expected_min_init_result,
                       succeed,
                       *cudf::make_min_aggregation<reduce_aggregation>());
  this->reduction_test(col_nulls,
                       initial_value,
                       expected_min_init_null_result,
                       succeed,
                       *cudf::make_min_aggregation<reduce_aggregation>());
  // MAX
  this->reduction_test(
    col, expected_max_result, succeed, *cudf::make_max_aggregation<reduce_aggregation>());
  this->reduction_test(col_nulls,
                       expected_max_null_result,
                       succeed,
                       *cudf::make_max_aggregation<reduce_aggregation>());
  this->reduction_test(col,
                       initial_value,
                       expected_max_init_result,
                       succeed,
                       *cudf::make_max_aggregation<reduce_aggregation>());
  this->reduction_test(col_nulls,
                       initial_value,
                       expected_max_init_null_result,
                       succeed,
                       *cudf::make_max_aggregation<reduce_aggregation>());

  // MINMAX
  auto result = cudf::minmax(col);
  EXPECT_EQ(static_cast<cudf::string_scalar*>(result.first.get())->to_string(),
            expected_min_result);
  EXPECT_EQ(static_cast<cudf::string_scalar*>(result.second.get())->to_string(),
            expected_max_result);
  result = cudf::minmax(col_nulls);
  EXPECT_EQ(static_cast<cudf::string_scalar*>(result.first.get())->to_string(),
            expected_min_null_result);
  EXPECT_EQ(static_cast<cudf::string_scalar*>(result.second.get())->to_string(),
            expected_max_null_result);
}

// Copied from the MinMax test above.
TEST_P(StringReductionTest, ArgMinMax)
{
  // Data and valid arrays.
  std::vector<std::string> host_strings(GetParam());
  std::vector<bool> host_bools({true, false, true, true, true, true, false, false, true});
  std::transform(cuda::counting_iterator<std::size_t>{0},
                 cuda::counting_iterator<std::size_t>{host_strings.size()},
                 host_strings.begin(),
                 [host_strings, host_bools](auto idx) {
                   return host_bools[idx] ? host_strings[idx] : std::string{};
                 });

  auto const argmin = [](std::vector<std::string> const& input) -> int {
    if (input.empty()) { return -1; }
    auto const it = std::min_element(input.begin(), input.end());
    return static_cast<int>(std::distance(input.begin(), it));
  };
  auto const argmax = [](std::vector<std::string> const& input) -> int {
    if (input.empty()) { return -1; }
    auto const it = std::max_element(input.begin(), input.end());
    return static_cast<int>(std::distance(input.begin(), it));
  };

  // All valid string column.
  {
    cudf::test::strings_column_wrapper col(host_strings.begin(), host_strings.end());
    auto const expected_argmin = argmin(host_strings);
    auto const expected_argmax = argmax(host_strings);
    this->reduction_test<int>(
      col, expected_argmin, *cudf::make_argmin_aggregation<reduce_aggregation>());
    this->reduction_test<int>(
      col, expected_argmax, *cudf::make_argmax_aggregation<reduce_aggregation>());
  }

  // String column with nulls.
  {
    cudf::test::strings_column_wrapper col_nulls(
      host_strings.begin(), host_strings.end(), host_bools.begin());

    std::string const min_string_sentinel{"\x0\x0\x0\x0"};
    std::string const max_string_sentinel{"\xF7\xBF\xBF\xBF"};
    std::vector<std::string> argmin_strings;
    std::vector<std::string> argmax_strings;
    std::transform(host_strings.begin(),
                   host_strings.end(),
                   std::back_inserter(argmin_strings),
                   [&, i = 0](auto const& s) mutable {
                     auto const valid = host_bools[i++];
                     return valid ? s : max_string_sentinel;
                   });
    std::transform(host_strings.begin(),
                   host_strings.end(),
                   std::back_inserter(argmax_strings),
                   [&, i = 0](auto const& s) mutable {
                     auto const valid = host_bools[i++];
                     return valid ? s : min_string_sentinel;
                   });

    auto const expected_argmin = argmin(argmin_strings);
    auto const expected_argmax = argmax(argmax_strings);
    this->reduction_test<int>(
      col_nulls, expected_argmin, *cudf::make_argmin_aggregation<reduce_aggregation>());
    this->reduction_test<int>(
      col_nulls, expected_argmax, *cudf::make_argmax_aggregation<reduce_aggregation>());
  }
}

TEST_P(StringReductionTest, DictionaryMinMax)
{
  // data and valid arrays
  std::vector<std::string> host_strings(GetParam());
  cudf::test::dictionary_column_wrapper<std::string> col(host_strings.begin(), host_strings.end());

  std::string expected_min_result = *(std::min_element(host_strings.begin(), host_strings.end()));
  std::string expected_max_result = *(std::max_element(host_strings.begin(), host_strings.end()));

  auto result = cudf::minmax(col);
  EXPECT_EQ(static_cast<cudf::string_scalar*>(result.first.get())->to_string(),
            expected_min_result);
  EXPECT_EQ(static_cast<cudf::string_scalar*>(result.second.get())->to_string(),
            expected_max_result);

  // column with nulls
  std::vector<bool> validity({true, false, true, true, true, true, false, false, true});
  cudf::test::dictionary_column_wrapper<std::string> col_nulls(
    host_strings.begin(), host_strings.end(), validity.begin());

  std::vector<std::string> r_strings;
  std::copy_if(host_strings.begin(),
               host_strings.end(),
               std::back_inserter(r_strings),
               [validity, i = 0](auto s) mutable { return validity[i++]; });

  expected_min_result = *(std::min_element(r_strings.begin(), r_strings.end()));
  expected_max_result = *(std::max_element(r_strings.begin(), r_strings.end()));

  result = cudf::minmax(col_nulls);
  EXPECT_EQ(static_cast<cudf::string_scalar*>(result.first.get())->to_string(),
            expected_min_result);
  EXPECT_EQ(static_cast<cudf::string_scalar*>(result.second.get())->to_string(),
            expected_max_result);

  // test sliced column
  result = cudf::minmax(cudf::slice(col_nulls, {3, 7}).front());
  // 3->2 and 7->5 because r_strings contains no null entries
  expected_min_result = *(std::min_element(r_strings.begin() + 2, r_strings.begin() + 5));
  expected_max_result = *(std::max_element(r_strings.begin() + 2, r_strings.begin() + 5));
  EXPECT_EQ(static_cast<cudf::string_scalar*>(result.first.get())->to_string(),
            expected_min_result);
  EXPECT_EQ(static_cast<cudf::string_scalar*>(result.second.get())->to_string(),
            expected_max_result);
}

TEST_F(StringReductionTest, AllNull)
{
  // data and all null arrays
  std::vector<std::string> host_strings(
    {"one", "two", "three", "four", "five", "six", "seven", "eight", "nine"});
  std::vector<bool> host_bools(host_strings.size(), false);
  auto initial_value = cudf::make_string_scalar("");
  initial_value->set_valid_async(false);

  // string column with nulls
  cudf::test::strings_column_wrapper col_nulls(
    host_strings.begin(), host_strings.end(), host_bools.begin());
  cudf::data_type output_dtype = cudf::column_view(col_nulls).type();

  // MIN
  auto result =
    cudf::reduce(col_nulls, *cudf::make_min_aggregation<reduce_aggregation>(), output_dtype);
  EXPECT_FALSE(result->is_valid());
  result = cudf::reduce(
    col_nulls, *cudf::make_min_aggregation<reduce_aggregation>(), output_dtype, *initial_value);
  EXPECT_FALSE(result->is_valid());
  // MAX
  result = cudf::reduce(col_nulls, *cudf::make_max_aggregation<reduce_aggregation>(), output_dtype);
  EXPECT_FALSE(result->is_valid());
  result = cudf::reduce(
    col_nulls, *cudf::make_max_aggregation<reduce_aggregation>(), output_dtype, *initial_value);
  EXPECT_FALSE(result->is_valid());
  // MINMAX
  auto mm_result = cudf::minmax(col_nulls);
  EXPECT_FALSE(mm_result.first->is_valid());
  EXPECT_FALSE(mm_result.second->is_valid());
  // ARGMIN
  result = cudf::reduce(col_nulls,
                        *cudf::make_argmin_aggregation<reduce_aggregation>(),
                        cudf::data_type{cudf::type_id::INT32});
  EXPECT_FALSE(result->is_valid());
  // ARGMAX
  result = cudf::reduce(col_nulls,
                        *cudf::make_argmax_aggregation<reduce_aggregation>(),
                        cudf::data_type{cudf::type_id::INT32});
  EXPECT_FALSE(result->is_valid());
}

TYPED_TEST(ReductionTest, Median)
{
  using T = TypeParam;
  //{-20, -14, -13,  0, 6, 13, 45, 64/None} =  3.0, 0.0
  std::vector<int> int_values({6, -14, 13, 64, 0, -13, -20, 45});
  std::vector<bool> host_bools({true, true, true, false, true, true, true, true});
  std::vector<T> v = convert_values<T>(int_values);
  auto output_type = cudf::data_type{cudf::type_to_id<double>()};

  // test without nulls
  cudf::test::fixed_width_column_wrapper<T> col(v.begin(), v.end());
  double expected_value = [] {
    if (std::is_same_v<T, bool>) return 1.0;
    if (std::is_signed_v<T>) return 3.0;
    return 13.5;
  }();
  EXPECT_EQ(this
              ->template reduction_test<double>(
                col, *cudf::make_median_aggregation<reduce_aggregation>(), output_type)
              .first,
            expected_value);

  auto col_odd              = cudf::split(col, {1})[1];
  double expected_value_odd = [] {
    if (std::is_same_v<T, bool>) return 1.0;
    if (std::is_signed_v<T>) return 0.0;
    return 14.0;
  }();
  EXPECT_EQ(this
              ->template reduction_test<double>(
                col_odd, *cudf::make_median_aggregation<reduce_aggregation>(), output_type)
              .first,
            expected_value_odd);

  // test with nulls
  cudf::test::fixed_width_column_wrapper<T> col_nulls = construct_null_column(v, host_bools);
  double expected_null_value                          = [] {
    if (std::is_same_v<T, bool>) return 1.0;
    if (std::is_signed_v<T>) return 0.0;
    return 13.0;
  }();

  EXPECT_EQ(this
              ->template reduction_test<double>(
                col_nulls, *cudf::make_median_aggregation<reduce_aggregation>(), output_type)
              .first,
            expected_null_value);

  auto col_nulls_odd             = cudf::split(col_nulls, {1})[1];
  double expected_null_value_odd = [] {
    if (std::is_same_v<T, bool>) return 1.0;
    if (std::is_signed_v<T>) return -6.5;
    return 13.5;
  }();
  EXPECT_EQ(this
              ->template reduction_test<double>(
                col_nulls_odd, *cudf::make_median_aggregation<reduce_aggregation>(), output_type)
              .first,
            expected_null_value_odd);
}

TYPED_TEST(ReductionTest, Quantile)
{
  using T = TypeParam;
  //{-20, -14, -13,  0, 6, 13, 45, 64/None}
  std::vector<int> int_values({6, -14, 13, 64, 0, -13, -20, 45});
  std::vector<bool> host_bools({true, true, true, false, true, true, true, true});
  std::vector<T> v = convert_values<T>(int_values);
  cudf::interpolation interp{cudf::interpolation::LINEAR};
  auto output_type = cudf::data_type{cudf::type_to_id<double>()};

  // test without nulls
  cudf::test::fixed_width_column_wrapper<T> col(v.begin(), v.end());
  double expected_value0 = std::is_same_v<T, bool> || std::is_unsigned_v<T> ? v[4] : v[6];
  EXPECT_EQ(
    this
      ->template reduction_test<double>(
        col, *cudf::make_quantile_aggregation<reduce_aggregation>({0.0}, interp), output_type)
      .first,
    expected_value0);

  double expected_value1 = v[3];
  EXPECT_EQ(
    this
      ->template reduction_test<double>(
        col, *cudf::make_quantile_aggregation<reduce_aggregation>({1.0}, interp), output_type)
      .first,
    expected_value1);

  // test with nulls
  cudf::test::fixed_width_column_wrapper<T> col_nulls = construct_null_column(v, host_bools);
  double expected_null_value1                         = v[7];

  EXPECT_EQ(
    this
      ->template reduction_test<double>(
        col_nulls, *cudf::make_quantile_aggregation<reduce_aggregation>({0}, interp), output_type)
      .first,
    expected_value0);
  EXPECT_EQ(
    this
      ->template reduction_test<double>(
        col_nulls, *cudf::make_quantile_aggregation<reduce_aggregation>({1}, interp), output_type)
      .first,
    expected_null_value1);
}

TYPED_TEST(ReductionTest, UniqueCount)
{
  using T = TypeParam;
  std::vector<int> int_values({1, -3, 1, 2, 0, 2, -4, 45});  // 6 unique values
  std::vector<bool> host_bools({true, true, true, false, true, true, true, true});
  std::vector<T> v = convert_values<T>(int_values);

  auto const output_type = cudf::data_type{cudf::type_to_id<cudf::size_type>()};

  // test without nulls
  cudf::test::fixed_width_column_wrapper<T> col(v.begin(), v.end());
  cudf::size_type expected_value = std::is_same_v<T, bool> ? 2 : 6;
  EXPECT_EQ(this
              ->template reduction_test<cudf::size_type>(
                col,
                *cudf::make_nunique_aggregation<reduce_aggregation>(cudf::null_policy::INCLUDE),
                output_type)
              .first,
            expected_value);
  EXPECT_EQ(this
              ->template reduction_test<cudf::size_type>(
                col,
                *cudf::make_nunique_aggregation<reduce_aggregation>(cudf::null_policy::EXCLUDE),
                output_type)
              .first,
            expected_value);

  // test with nulls
  cudf::test::fixed_width_column_wrapper<T> col_nulls = construct_null_column(v, host_bools);
  cudf::size_type expected_null_value0                = std::is_same_v<T, bool> ? 3 : 7;
  cudf::size_type expected_null_value1                = std::is_same_v<T, bool> ? 2 : 6;

  EXPECT_EQ(this
              ->template reduction_test<cudf::size_type>(
                col_nulls,
                *cudf::make_nunique_aggregation<reduce_aggregation>(cudf::null_policy::INCLUDE),
                output_type)
              .first,
            expected_null_value0);
  EXPECT_EQ(this
              ->template reduction_test<cudf::size_type>(
                col_nulls,
                *cudf::make_nunique_aggregation<reduce_aggregation>(cudf::null_policy::EXCLUDE),
                output_type)
              .first,
            expected_null_value1);

  EXPECT_EQ(this
              ->template reduction_test<uint32_t>(
                col_nulls,
                *cudf::make_nunique_aggregation<reduce_aggregation>(cudf::null_policy::EXCLUDE),
                cudf::data_type{cudf::type_to_id<uint32_t>()})
              .first,
            expected_null_value1);
}

TYPED_TEST(ReductionTest, Count)
{
  using T = TypeParam;
  std::vector<int> int_values({1, -3, 1, 2, 0, 2, -4, 45});
  std::vector<T> v = convert_values<T>(int_values);

  auto const output_type = cudf::data_type{cudf::type_to_id<cudf::size_type>()};

  // test without nulls
  auto col            = cudf::test::fixed_width_column_wrapper<T>(v.begin(), v.end());
  auto expected_value = static_cast<cudf::size_type>(v.size());
  auto count_agg = cudf::make_count_aggregation<reduce_aggregation>(cudf::null_policy::INCLUDE);
  auto count_agg_exclude =
    cudf::make_count_aggregation<reduce_aggregation>(cudf::null_policy::EXCLUDE);
  EXPECT_EQ(this->template reduction_test<cudf::size_type>(col, *count_agg, output_type).first,
            expected_value);
  EXPECT_EQ(
    this->template reduction_test<cudf::size_type>(col, *count_agg_exclude, output_type).first,
    expected_value);

  // test with nulls
  auto validity = cudf::test::iterators::null_at(3);
  col           = cudf::test::fixed_width_column_wrapper<T>(v.begin(), v.end(), validity);
  EXPECT_EQ(this->template reduction_test<cudf::size_type>(col, *count_agg, output_type).first,
            expected_value);

  expected_value = static_cast<cudf::size_type>(v.size() - 1);
  EXPECT_EQ(
    this->template reduction_test<cudf::size_type>(col, *count_agg_exclude, output_type).first,
    expected_value);
}

template <typename T>
struct FixedPointTestAllReps : public cudf::test::BaseFixture {};

TYPED_TEST_SUITE(FixedPointTestAllReps, cudf::test::FixedPointTypes);

TYPED_TEST(FixedPointTestAllReps, FixedPointReductionProductZeroScale)
{
  using namespace numeric;
  using decimalXX = TypeParam;

  auto const ONE   = decimalXX{1, scale_type{0}};
  auto const TWO   = decimalXX{2, scale_type{0}};
  auto const THREE = decimalXX{3, scale_type{0}};
  auto const FOUR  = decimalXX{4, scale_type{0}};
  auto const _24   = decimalXX{24, scale_type{0}};
  auto const _48   = decimalXX{48, scale_type{0}};

  auto const in       = std::vector<decimalXX>{ONE, TWO, THREE, FOUR};
  auto const column   = cudf::test::fixed_width_column_wrapper<decimalXX>(in.cbegin(), in.cend());
  auto const expected = std::accumulate(in.cbegin(), in.cend(), ONE, std::multiplies<decimalXX>());
  auto const out_type = static_cast<cudf::column_view>(column).type();

  auto const result =
    cudf::reduce(column, *cudf::make_product_aggregation<reduce_aggregation>(), out_type);
  auto const result_scalar = static_cast<cudf::scalar_type_t<decimalXX>*>(result.get());
  auto const result_fp     = decimalXX{result_scalar->value()};

  EXPECT_EQ(result_fp, expected);
  EXPECT_EQ(result_fp, _24);

  // Test with initial value
  auto const init_expected =
    std::accumulate(in.cbegin(), in.cend(), TWO, std::multiplies<decimalXX>());
  auto const init_scalar = cudf::make_fixed_point_scalar<decimalXX>(2, scale_type{0});

  auto const init_result = cudf::reduce(
    column, *cudf::make_product_aggregation<reduce_aggregation>(), out_type, *init_scalar);
  auto const init_result_scalar = static_cast<cudf::scalar_type_t<decimalXX>*>(init_result.get());
  auto const init_result_fp     = decimalXX{init_result_scalar->value()};

  EXPECT_EQ(init_result_fp, init_expected);
  EXPECT_EQ(init_result_fp, _48);
}

TYPED_TEST(FixedPointTestAllReps, FixedPointReductionProduct)
{
  using namespace numeric;
  using decimalXX  = TypeParam;
  using RepType    = cudf::device_storage_type_t<decimalXX>;
  using fp_wrapper = cudf::test::fixed_point_column_wrapper<RepType>;

  for (auto const i : {0, -1}) {
    auto const scale    = scale_type{i};
    auto const column   = fp_wrapper{{1, 2, 3, 1, 2, 3}, scale};
    auto const out_type = static_cast<cudf::column_view>(column).type();
    auto const expected = decimalXX{scaled_integer<RepType>{36, scale_type{i * 6}}};

    auto const result =
      cudf::reduce(column, *cudf::make_product_aggregation<reduce_aggregation>(), out_type);
    auto const result_scalar = static_cast<cudf::scalar_type_t<decimalXX>*>(result.get());

    EXPECT_EQ(result_scalar->fixed_point_value(), expected);

    // Test with initial value
    auto const init_expected = decimalXX{scaled_integer<RepType>{72, scale_type{i * 7}}};
    auto const init_scalar   = cudf::make_fixed_point_scalar<decimalXX>(2, scale);

    auto const init_result = cudf::reduce(
      column, *cudf::make_product_aggregation<reduce_aggregation>(), out_type, *init_scalar);
    auto const init_result_scalar = static_cast<cudf::scalar_type_t<decimalXX>*>(init_result.get());

    EXPECT_EQ(init_result_scalar->fixed_point_value(), init_expected);
  }
}

TYPED_TEST(FixedPointTestAllReps, FixedPointReductionProductWithNulls)
{
  using namespace numeric;
  using decimalXX  = TypeParam;
  using RepType    = cudf::device_storage_type_t<decimalXX>;
  using fp_wrapper = cudf::test::fixed_point_column_wrapper<RepType>;

  for (auto const i : {0, -1}) {
    auto const scale    = scale_type{i};
    auto const column   = fp_wrapper{{1, 2, 3, 1, 2, 3}, {1, 1, 1, 0, 0, 0}, scale};
    auto const out_type = static_cast<cudf::column_view>(column).type();
    auto const expected = decimalXX{scaled_integer<RepType>{6, scale_type{i * 3}}};

    auto const result =
      cudf::reduce(column, *cudf::make_product_aggregation<reduce_aggregation>(), out_type);
    auto const result_scalar = static_cast<cudf::scalar_type_t<decimalXX>*>(result.get());

    EXPECT_EQ(result_scalar->fixed_point_value(), expected);

    // Test with initial value
    auto const init_expected = decimalXX{scaled_integer<RepType>{12, scale_type{i * 4}}};
    auto const init_scalar   = cudf::make_fixed_point_scalar<decimalXX>(2, scale);

    auto const init_result = cudf::reduce(
      column, *cudf::make_product_aggregation<reduce_aggregation>(), out_type, *init_scalar);
    auto const init_result_scalar = static_cast<cudf::scalar_type_t<decimalXX>*>(init_result.get());

    EXPECT_EQ(init_result_scalar->fixed_point_value(), init_expected);
  }
}

TYPED_TEST(FixedPointTestAllReps, FixedPointReductionSum)
{
  using namespace numeric;
  using decimalXX  = TypeParam;
  using RepType    = cudf::device_storage_type_t<decimalXX>;
  using fp_wrapper = cudf::test::fixed_point_column_wrapper<RepType>;

  for (auto const i : {0, -1, -2, -3}) {
    auto const scale = scale_type{i};

    auto const column   = fp_wrapper{{1, 2, 3, 4}, scale};
    auto const expected = decimalXX{scaled_integer<RepType>{10, scale}};
    auto const out_type = static_cast<cudf::column_view>(column).type();

    auto const result =
      cudf::reduce(column, *cudf::make_sum_aggregation<reduce_aggregation>(), out_type);
    auto const result_scalar = static_cast<cudf::scalar_type_t<decimalXX>*>(result.get());

    EXPECT_EQ(result_scalar->fixed_point_value(), expected);

    // Test with initial value
    auto const init_expected = decimalXX{scaled_integer<RepType>{12, scale}};
    auto const init_scalar   = cudf::make_fixed_point_scalar<decimalXX>(2, scale);

    auto const init_result = cudf::reduce(
      column, *cudf::make_sum_aggregation<reduce_aggregation>(), out_type, *init_scalar);
    auto const init_result_scalar = static_cast<cudf::scalar_type_t<decimalXX>*>(init_result.get());

    EXPECT_EQ(init_result_scalar->fixed_point_value(), init_expected);
  }
}

TYPED_TEST(FixedPointTestAllReps, FixedPointReductionSumAlternate)
{
  using namespace numeric;
  using decimalXX = TypeParam;

  auto const ZERO   = decimalXX{0, scale_type{0}};
  auto const ONE    = decimalXX{1, scale_type{0}};
  auto const TWO    = decimalXX{2, scale_type{0}};
  auto const THREE  = decimalXX{3, scale_type{0}};
  auto const FOUR   = decimalXX{4, scale_type{0}};
  auto const TEN    = decimalXX{10, scale_type{0}};
  auto const TWELVE = decimalXX{12, scale_type{0}};

  auto const in       = std::vector<decimalXX>{ONE, TWO, THREE, FOUR};
  auto const column   = cudf::test::fixed_width_column_wrapper<decimalXX>(in.cbegin(), in.cend());
  auto const expected = std::accumulate(in.cbegin(), in.cend(), ZERO, std::plus<decimalXX>());
  auto const out_type = static_cast<cudf::column_view>(column).type();

  auto const result =
    cudf::reduce(column, *cudf::make_sum_aggregation<reduce_aggregation>(), out_type);
  auto const result_scalar = static_cast<cudf::scalar_type_t<decimalXX>*>(result.get());

  EXPECT_EQ(result_scalar->fixed_point_value(), expected);
  EXPECT_EQ(result_scalar->fixed_point_value(), TEN);

  // Test with initial value
  auto const init_expected = std::accumulate(in.cbegin(), in.cend(), TWO, std::plus<decimalXX>());
  auto const init_scalar   = cudf::make_fixed_point_scalar<decimalXX>(2, scale_type{0});

  auto const init_result =
    cudf::reduce(column, *cudf::make_sum_aggregation<reduce_aggregation>(), out_type, *init_scalar);
  auto const init_result_scalar = static_cast<cudf::scalar_type_t<decimalXX>*>(init_result.get());

  EXPECT_EQ(init_result_scalar->fixed_point_value(), init_expected);
  EXPECT_EQ(init_result_scalar->fixed_point_value(), TWELVE);
}

TYPED_TEST(FixedPointTestAllReps, FixedPointReductionSumFractional)
{
  using namespace numeric;
  using decimalXX  = TypeParam;
  using RepType    = cudf::device_storage_type_t<decimalXX>;
  using fp_wrapper = cudf::test::fixed_point_column_wrapper<RepType>;

  for (auto const i : {0, -1, -2, -3}) {
    auto const scale    = scale_type{i};
    auto const column   = fp_wrapper{{111, 222, 333}, scale};
    auto const out_type = static_cast<cudf::column_view>(column).type();
    auto const expected = decimalXX{scaled_integer<RepType>{666, scale}};

    auto const result =
      cudf::reduce(column, *cudf::make_sum_aggregation<reduce_aggregation>(), out_type);
    auto const result_scalar = static_cast<cudf::scalar_type_t<decimalXX>*>(result.get());

    EXPECT_EQ(result_scalar->fixed_point_value(), expected);

    // Test with initial value
    auto const init_expected = decimalXX{scaled_integer<RepType>{668, scale}};
    auto const init_scalar   = cudf::make_fixed_point_scalar<decimalXX>(2, scale);

    auto const init_result = cudf::reduce(
      column, *cudf::make_sum_aggregation<reduce_aggregation>(), out_type, *init_scalar);
    auto const init_result_scalar = static_cast<cudf::scalar_type_t<decimalXX>*>(init_result.get());

    EXPECT_EQ(init_result_scalar->fixed_point_value(), init_expected);
  }
}

TYPED_TEST(FixedPointTestAllReps, FixedPointReductionSumLarge)
{
  using namespace numeric;
  using decimalXX  = TypeParam;
  using RepType    = cudf::device_storage_type_t<decimalXX>;
  using fp_wrapper = cudf::test::fixed_point_column_wrapper<RepType>;

  for (auto const i : {0, -1, -2}) {
    auto const scale = scale_type{i};
    auto f = cudf::detail::make_counting_transform_iterator(0, [](int i) { return RepType{i}; });
    auto const values         = std::vector<RepType>(f, f + 1000);
    auto const column         = fp_wrapper{values.cbegin(), values.cend(), scale};
    auto const out_type       = static_cast<cudf::column_view>(column).type();
    auto const expected_value = std::accumulate(values.cbegin(), values.cend(), RepType{0});
    auto const expected       = decimalXX{scaled_integer<RepType>{expected_value, scale}};

    auto const result =
      cudf::reduce(column, *cudf::make_sum_aggregation<reduce_aggregation>(), out_type);
    auto const result_scalar = static_cast<cudf::scalar_type_t<decimalXX>*>(result.get());

    EXPECT_EQ(result_scalar->fixed_point_value(), expected);

    // Test with initial value
    int const init_value = 2;
    auto const init_expected_value =
      std::accumulate(values.cbegin(), values.cend(), RepType{init_value});
    auto const init_expected = decimalXX{scaled_integer<RepType>{init_expected_value, scale}};
    auto const init_scalar   = cudf::make_fixed_point_scalar<decimalXX>(init_value, scale);

    auto const init_result = cudf::reduce(
      column, *cudf::make_sum_aggregation<reduce_aggregation>(), out_type, *init_scalar);
    auto const init_result_scalar = static_cast<cudf::scalar_type_t<decimalXX>*>(init_result.get());

    EXPECT_EQ(init_result_scalar->fixed_point_value(), init_expected);
  }
}

TYPED_TEST(FixedPointTestAllReps, FixedPointReductionMinAndArgMin)
{
  using namespace numeric;
  using decimalXX  = TypeParam;
  using RepType    = cudf::device_storage_type_t<decimalXX>;
  using fp_wrapper = cudf::test::fixed_point_column_wrapper<RepType>;

  for (auto const i : {0, -1, -2, -3}) {
    auto const scale        = scale_type{i};
    auto const ONE          = decimalXX{scaled_integer<RepType>{1, scale}};
    auto const column       = fp_wrapper{{2, 3, 1, 4}, scale};
    auto const min_out_type = static_cast<cudf::column_view>(column).type();

    {
      auto const result =
        cudf::reduce(column, *cudf::make_min_aggregation<reduce_aggregation>(), min_out_type);
      auto const result_scalar = static_cast<cudf::scalar_type_t<decimalXX>*>(result.get());
      EXPECT_EQ(result_scalar->fixed_point_value(), ONE);
    }

    // Test with initial value
    {
      auto const init_expected = decimalXX{scaled_integer<RepType>{0, scale}};
      auto const init_scalar   = cudf::make_fixed_point_scalar<decimalXX>(0, scale);
      auto const init_result   = cudf::reduce(
        column, *cudf::make_min_aggregation<reduce_aggregation>(), min_out_type, *init_scalar);
      auto const init_result_scalar =
        static_cast<cudf::scalar_type_t<decimalXX>*>(init_result.get());
      EXPECT_EQ(init_result_scalar->fixed_point_value(), init_expected);
    }

    // Test ARGMIN
    {
      auto const argmin_out_type = cudf::data_type{cudf::type_id::INT32};
      auto const result =
        cudf::reduce(column, *cudf::make_argmin_aggregation<reduce_aggregation>(), argmin_out_type);
      auto const result_scalar = static_cast<cudf::scalar_type_t<int>*>(result.get());
      EXPECT_EQ(result_scalar->value(), 2);
    }
  }
}

TYPED_TEST(FixedPointTestAllReps, FixedPointReductionMinLarge)
{
  using namespace numeric;
  using decimalXX  = TypeParam;
  using RepType    = cudf::device_storage_type_t<decimalXX>;
  using fp_wrapper = cudf::test::fixed_point_column_wrapper<RepType>;

  for (auto const i : {0, -1, -2, -3}) {
    auto const scale = scale_type{i};
    auto f = cudf::detail::make_counting_transform_iterator(0, [](auto e) { return e % 43; });
    auto const column   = fp_wrapper{f, f + 5000, scale};
    auto const out_type = static_cast<cudf::column_view>(column).type();
    auto const expected = decimalXX{0, scale};

    auto const result =
      cudf::reduce(column, *cudf::make_min_aggregation<reduce_aggregation>(), out_type);
    auto const result_scalar = static_cast<cudf::scalar_type_t<decimalXX>*>(result.get());

    EXPECT_EQ(result_scalar->fixed_point_value(), expected);

    // Test with initial value
    auto const init_expected = decimalXX{scaled_integer<RepType>{0, scale}};
    auto const init_scalar   = cudf::make_fixed_point_scalar<decimalXX>(0, scale);

    auto const init_result = cudf::reduce(
      column, *cudf::make_min_aggregation<reduce_aggregation>(), out_type, *init_scalar);
    auto const init_result_scalar = static_cast<cudf::scalar_type_t<decimalXX>*>(init_result.get());

    EXPECT_EQ(init_result_scalar->fixed_point_value(), init_expected);
  }
}

TYPED_TEST(FixedPointTestAllReps, FixedPointReductionMaxAndArgMax)
{
  using namespace numeric;
  using decimalXX  = TypeParam;
  using RepType    = cudf::device_storage_type_t<decimalXX>;
  using fp_wrapper = cudf::test::fixed_point_column_wrapper<RepType>;

  for (auto const i : {0, -1, -2, -3}) {
    auto const scale        = scale_type{i};
    auto const FOUR         = decimalXX{scaled_integer<RepType>{4, scale}};
    auto const column       = fp_wrapper{{2, 3, 1, 4}, scale};
    auto const max_out_type = static_cast<cudf::column_view>(column).type();

    {
      auto const result =
        cudf::reduce(column, *cudf::make_max_aggregation<reduce_aggregation>(), max_out_type);
      auto const result_scalar = static_cast<cudf::scalar_type_t<decimalXX>*>(result.get());
      EXPECT_EQ(result_scalar->fixed_point_value(), FOUR);
    }

    // Test with initial value
    {
      auto const init_expected = decimalXX{scaled_integer<RepType>{5, scale}};
      auto const init_scalar   = cudf::make_fixed_point_scalar<decimalXX>(5, scale);
      auto const init_result   = cudf::reduce(
        column, *cudf::make_max_aggregation<reduce_aggregation>(), max_out_type, *init_scalar);
      auto const init_result_scalar =
        static_cast<cudf::scalar_type_t<decimalXX>*>(init_result.get());
      EXPECT_EQ(init_result_scalar->fixed_point_value(), init_expected);
    }

    // Test ARGMAX
    {
      auto const argmax_out_type = cudf::data_type{cudf::type_id::INT32};
      auto const result =
        cudf::reduce(column, *cudf::make_argmax_aggregation<reduce_aggregation>(), argmax_out_type);
      auto const result_scalar = static_cast<cudf::scalar_type_t<int>*>(result.get());
      EXPECT_EQ(result_scalar->value(), 3);
    }
  }
}

TYPED_TEST(FixedPointTestAllReps, FixedPointReductionMaxLarge)
{
  using namespace numeric;
  using decimalXX  = TypeParam;
  using RepType    = cudf::device_storage_type_t<decimalXX>;
  using fp_wrapper = cudf::test::fixed_point_column_wrapper<RepType>;

  for (auto const i : {0, -1, -2, -3}) {
    auto const scale = scale_type{i};
    auto f = cudf::detail::make_counting_transform_iterator(0, [](auto e) { return e % 43; });
    auto const column   = fp_wrapper{f, f + 5000, scale};
    auto const out_type = static_cast<cudf::column_view>(column).type();
    auto const expected = decimalXX{scaled_integer<RepType>{42, scale}};

    auto const result =
      cudf::reduce(column, *cudf::make_max_aggregation<reduce_aggregation>(), out_type);
    auto const result_scalar = static_cast<cudf::scalar_type_t<decimalXX>*>(result.get());

    EXPECT_EQ(result_scalar->fixed_point_value(), expected);

    // Test with initial value
    auto const init_expected = decimalXX{scaled_integer<RepType>{43, scale}};
    auto const init_scalar   = cudf::make_fixed_point_scalar<decimalXX>(43, scale);

    auto const init_result = cudf::reduce(
      column, *cudf::make_max_aggregation<reduce_aggregation>(), out_type, *init_scalar);
    auto const init_result_scalar = static_cast<cudf::scalar_type_t<decimalXX>*>(init_result.get());

    EXPECT_EQ(init_result_scalar->fixed_point_value(), init_expected);
  }
}

TYPED_TEST(FixedPointTestAllReps, FixedPointReductionNUnique)
{
  using namespace numeric;
  using decimalXX  = TypeParam;
  using RepType    = cudf::device_storage_type_t<decimalXX>;
  using fp_wrapper = cudf::test::fixed_point_column_wrapper<RepType>;
  auto output_type = cudf::data_type{cudf::type_to_id<cudf::size_type>()};

  for (auto const i : {0, -1, -2, -3}) {
    auto const scale  = scale_type{i};
    auto const column = fp_wrapper{{1, 1, 2, 2, 3, 3, 4, 4}, scale};

    auto const result =
      cudf::reduce(column, *cudf::make_nunique_aggregation<reduce_aggregation>(), output_type);
    auto const result_scalar = static_cast<cudf::scalar_type_t<cudf::size_type>*>(result.get());

    EXPECT_EQ(result_scalar->value(), 4);
  }
}

TYPED_TEST(FixedPointTestAllReps, FixedPointReductionSumOfSquares)
{
  using namespace numeric;
  using decimalXX  = TypeParam;
  using RepType    = cudf::device_storage_type_t<decimalXX>;
  using fp_wrapper = cudf::test::fixed_point_column_wrapper<RepType>;

  for (auto const i : {0, -1, -2}) {
    auto const scale    = scale_type{i};
    auto const column   = fp_wrapper{{1, 2, 3, 4}, scale};
    auto const out_type = static_cast<cudf::column_view>(column).type();
    auto const expected = decimalXX{scaled_integer<RepType>{30, scale_type{i * 2}}};

    auto const result =
      cudf::reduce(column, *cudf::make_sum_of_squares_aggregation<reduce_aggregation>(), out_type);
    auto const result_scalar = static_cast<cudf::scalar_type_t<decimalXX>*>(result.get());

    EXPECT_EQ(result_scalar->fixed_point_value(), expected);
  }
}

TYPED_TEST(FixedPointTestAllReps, FixedPointReductionMedianOddNumberOfElements)
{
  using namespace numeric;
  using decimalXX  = TypeParam;
  using RepType    = cudf::device_storage_type_t<decimalXX>;
  using fp_wrapper = cudf::test::fixed_point_column_wrapper<RepType>;

  for (auto const s : {0, -1, -2, -3, -4}) {
    auto const scale    = scale_type{s};
    auto const column   = fp_wrapper{{1, 2, 2, 3, 4}, scale};
    auto const out_type = static_cast<cudf::column_view>(column).type();
    auto const expected = decimalXX{scaled_integer<RepType>{2, scale}};

    auto const result =
      cudf::reduce(column, *cudf::make_median_aggregation<reduce_aggregation>(), out_type);
    auto const result_scalar = static_cast<cudf::scalar_type_t<decimalXX>*>(result.get());

    EXPECT_EQ(result_scalar->fixed_point_value(), expected);
  }
}

TYPED_TEST(FixedPointTestAllReps, FixedPointReductionMedianEvenNumberOfElements)
{
  using namespace numeric;
  using decimalXX  = TypeParam;
  using RepType    = cudf::device_storage_type_t<decimalXX>;
  using fp_wrapper = cudf::test::fixed_point_column_wrapper<RepType>;

  for (auto const s : {0, -1, -2, -3, -4}) {
    auto const scale    = scale_type{s};
    auto const column   = fp_wrapper{{10, 20, 20, 30, 30, 40}, scale};
    auto const out_type = static_cast<cudf::column_view>(column).type();
    auto const expected = decimalXX{scaled_integer<RepType>{25, scale}};

    auto const result =
      cudf::reduce(column, *cudf::make_median_aggregation<reduce_aggregation>(), out_type);
    auto const result_scalar = static_cast<cudf::scalar_type_t<decimalXX>*>(result.get());

    EXPECT_EQ(result_scalar->fixed_point_value(), expected);
  }
}

TYPED_TEST(FixedPointTestAllReps, FixedPointReductionQuantile)
{
  using namespace numeric;
  using decimalXX  = TypeParam;
  using RepType    = cudf::device_storage_type_t<decimalXX>;
  using fp_wrapper = cudf::test::fixed_point_column_wrapper<RepType>;

  for (auto const s : {0, -1, -2, -3, -4}) {
    auto const scale    = scale_type{s};
    auto const column   = fp_wrapper{{1, 2, 3, 4, 5}, scale};
    auto const out_type = static_cast<cudf::column_view>(column).type();

    for (auto const i : {0, 1, 2, 3, 4}) {
      auto const expected      = decimalXX{scaled_integer<RepType>{i + 1, scale}};
      auto const result        = cudf::reduce(column,
                                       *cudf::make_quantile_aggregation<reduce_aggregation>(
                                         {i / 4.0}, cudf::interpolation::LINEAR),
                                       out_type);
      auto const result_scalar = static_cast<cudf::scalar_type_t<decimalXX>*>(result.get());
      EXPECT_EQ(result_scalar->fixed_point_value(), expected);
    }
  }
}

TYPED_TEST(FixedPointTestAllReps, FixedPointReductionNthElement)
{
  using namespace numeric;
  using decimalXX  = TypeParam;
  using RepType    = cudf::device_storage_type_t<decimalXX>;
  using fp_wrapper = cudf::test::fixed_point_column_wrapper<RepType>;

  for (auto const s : {0, -1, -2, -3, -4}) {
    auto const scale    = scale_type{s};
    auto const values   = std::vector<RepType>{4104, 42, 1729, 55};
    auto const column   = fp_wrapper{values.cbegin(), values.cend(), scale};
    auto const out_type = static_cast<cudf::column_view>(column).type();

    for (auto const i : {0, 1, 2, 3}) {
      auto const expected = decimalXX{scaled_integer<RepType>{values[i], scale}};
      auto const result   = cudf::reduce(
        column,
        *cudf::make_nth_element_aggregation<reduce_aggregation>(i, cudf::null_policy::INCLUDE),
        out_type);
      auto const result_scalar = static_cast<cudf::scalar_type_t<decimalXX>*>(result.get());
      EXPECT_EQ(result_scalar->fixed_point_value(), expected);
    }
  }
}

struct Decimal128Only : public cudf::test::BaseFixture {};

TEST_F(Decimal128Only, Decimal128ProductReduction)
{
  using namespace numeric;
  using RepType    = cudf::device_storage_type_t<decimal128>;
  using fp_wrapper = cudf::test::fixed_point_column_wrapper<RepType>;

  for (auto const i : {0, -1, -2, -3}) {
    auto const scale    = scale_type{i};
    auto const column   = fp_wrapper{{2, 2, 2, 2, 2, 2, 2, 2, 2}, scale};
    auto const expected = decimal128{scaled_integer<RepType>{512, scale_type{i * 9}}};

    auto const out_type = cudf::data_type{cudf::type_id::DECIMAL128, scale};
    auto const result =
      cudf::reduce(column, *cudf::make_product_aggregation<reduce_aggregation>(), out_type);
    auto const result_scalar = static_cast<cudf::scalar_type_t<decimal128>*>(result.get());

    EXPECT_EQ(result_scalar->fixed_point_value(), expected);

    // Test with initial value
    auto const init_expected = decimal128{scaled_integer<RepType>{1024, scale_type{i * 10}}};
    auto const init_scalar   = cudf::make_fixed_point_scalar<decimal128>(2, scale);

    auto const init_result = cudf::reduce(
      column, *cudf::make_product_aggregation<reduce_aggregation>(), out_type, *init_scalar);
    auto const init_result_scalar =
      static_cast<cudf::scalar_type_t<decimal128>*>(init_result.get());

    EXPECT_EQ(init_result_scalar->fixed_point_value(), init_expected);
  }
}

TEST_F(Decimal128Only, Decimal128ProductReduction2)
{
  using namespace numeric;
  using RepType    = cudf::device_storage_type_t<decimal128>;
  using fp_wrapper = cudf::test::fixed_point_column_wrapper<RepType>;

  for (auto const i : {0, -1, -2, -3, -4, -5, -6}) {
    auto const scale    = scale_type{i};
    auto const column   = fp_wrapper{{1, 2, 3, 4, 5, 6}, scale};
    auto const expected = decimal128{scaled_integer<RepType>{720, scale_type{i * 6}}};

    auto const out_type = cudf::data_type{cudf::type_id::DECIMAL128, scale};
    auto const result =
      cudf::reduce(column, *cudf::make_product_aggregation<reduce_aggregation>(), out_type);
    auto const result_scalar = static_cast<cudf::scalar_type_t<decimal128>*>(result.get());

    EXPECT_EQ(result_scalar->fixed_point_value(), expected);

    // Test with initial value
    auto const init_expected = decimal128{scaled_integer<RepType>{2160, scale_type{i * 7}}};
    auto const init_scalar   = cudf::make_fixed_point_scalar<decimal128>(3, scale);

    auto const init_result = cudf::reduce(
      column, *cudf::make_product_aggregation<reduce_aggregation>(), out_type, *init_scalar);
    auto const init_result_scalar =
      static_cast<cudf::scalar_type_t<decimal128>*>(init_result.get());

    EXPECT_EQ(init_result_scalar->fixed_point_value(), init_expected);
  }
}

TEST_F(Decimal128Only, Decimal128ProductReduction3)
{
  using namespace numeric;
  using RepType    = cudf::device_storage_type_t<decimal128>;
  using fp_wrapper = cudf::test::fixed_point_column_wrapper<RepType>;

  auto const values   = std::vector(127, -2);
  auto const scale    = scale_type{0};
  auto const column   = fp_wrapper{values.cbegin(), values.cend(), scale};
  auto const lowest   = cuda::std::numeric_limits<RepType>::lowest();
  auto const expected = decimal128{scaled_integer<RepType>{lowest, scale}};

  auto const out_type = cudf::data_type{cudf::type_id::DECIMAL128, scale};
  auto const result =
    cudf::reduce(column, *cudf::make_product_aggregation<reduce_aggregation>(), out_type);
  auto const result_scalar = static_cast<cudf::scalar_type_t<decimal128>*>(result.get());

  EXPECT_EQ(result_scalar->fixed_point_value(), expected);

  // Test with initial value
  auto const init_scalar = cudf::make_fixed_point_scalar<decimal128>(5, scale);

  auto const init_result = cudf::reduce(
    column, *cudf::make_product_aggregation<reduce_aggregation>(), out_type, *init_scalar);
  auto const init_result_scalar = static_cast<cudf::scalar_type_t<decimal128>*>(init_result.get());

  EXPECT_EQ(init_result_scalar->fixed_point_value(), expected);
}

TYPED_TEST(ReductionTest, NthElement)
{
  using T = TypeParam;
  std::vector<int> int_values(4000);
  std::iota(int_values.begin(), int_values.end(), 0);
  std::vector<bool> host_bools(int_values.size());
  auto valid_condition = [](auto i) { return (i % 3 and i % 7); };
  std::transform(int_values.begin(), int_values.end(), host_bools.begin(), valid_condition);

  cudf::size_type valid_count = std::count(host_bools.begin(), host_bools.end(), true);
  std::vector<int> int_values_valid(valid_count);
  std::copy_if(int_values.begin(), int_values.end(), int_values_valid.begin(), valid_condition);

  std::vector<T> v           = convert_values<T>(int_values);
  std::vector<T> v_valid     = convert_values<T>(int_values_valid);
  cudf::size_type input_size = v.size();

  auto mod = [](int a, int b) { return (a % b + b) % b; };
  cudf::test::fixed_width_column_wrapper<T> col(v.begin(), v.end());
  cudf::test::fixed_width_column_wrapper<T> col_nulls = construct_null_column(v, host_bools);
  // without nulls
  for (cudf::size_type n :
       {-input_size, -input_size / 2, -2, -1, 0, 1, 2, input_size / 2, input_size - 1}) {
    auto const index         = mod(n, v.size());
    T expected_value_nonull  = v[index];
    bool const expected_null = host_bools[index];
    EXPECT_EQ(
      this
        ->template reduction_test<T>(
          col,
          *cudf::make_nth_element_aggregation<reduce_aggregation>(n, cudf::null_policy::INCLUDE))
        .first,
      expected_value_nonull);
    EXPECT_EQ(
      this
        ->template reduction_test<T>(
          col,
          *cudf::make_nth_element_aggregation<reduce_aggregation>(n, cudf::null_policy::EXCLUDE))
        .first,
      expected_value_nonull);
    auto res = this->template reduction_test<T>(
      col_nulls,
      *cudf::make_nth_element_aggregation<reduce_aggregation>(n, cudf::null_policy::INCLUDE));
    EXPECT_EQ(res.first, expected_value_nonull);
    EXPECT_EQ(res.second, expected_null);
  }
  // valid only
  for (cudf::size_type n :
       {-valid_count, -valid_count / 2, -2, -1, 0, 1, 2, valid_count / 2, valid_count - 1}) {
    T expected_value_null = v_valid[mod(n, v_valid.size())];
    EXPECT_EQ(
      this
        ->template reduction_test<T>(
          col_nulls,
          *cudf::make_nth_element_aggregation<reduce_aggregation>(n, cudf::null_policy::EXCLUDE))
        .first,
      expected_value_null);
  }
  // error cases
  for (cudf::size_type n : {-input_size - 1, input_size}) {
    EXPECT_ANY_THROW(this->template reduction_test<T>(
      col, *cudf::make_nth_element_aggregation<reduce_aggregation>(n, cudf::null_policy::INCLUDE)));
    EXPECT_ANY_THROW(this->template reduction_test<T>(
      col_nulls,
      *cudf::make_nth_element_aggregation<reduce_aggregation>(n, cudf::null_policy::INCLUDE)));
    EXPECT_ANY_THROW(this->template reduction_test<T>(
      col, *cudf::make_nth_element_aggregation<reduce_aggregation>(n, cudf::null_policy::EXCLUDE)));
    EXPECT_ANY_THROW(this->template reduction_test<T>(
      col_nulls,
      *cudf::make_nth_element_aggregation<reduce_aggregation>(n, cudf::null_policy::EXCLUDE)));
  }
}

struct DictionaryStringReductionTest : public StringReductionTest {};

std::vector<std::vector<std::string>> data_list = {
  {"nine", "two", "five", "three", "five", "six", "two", "eight", "nine"},
};
INSTANTIATE_TEST_CASE_P(dictionary_cases,
                        DictionaryStringReductionTest,
                        testing::ValuesIn(data_list));
TEST_P(DictionaryStringReductionTest, MinMax)
{
  std::vector<std::string> host_strings(GetParam());
  cudf::data_type output_type{cudf::type_id::STRING};

  cudf::test::dictionary_column_wrapper<std::string> col(host_strings.begin(), host_strings.end());

  // MIN
  this->reduction_test(col,
                       *(std::min_element(host_strings.begin(), host_strings.end())),
                       true,
                       *cudf::make_min_aggregation<reduce_aggregation>(),
                       output_type);
  // sliced
  this->reduction_test(cudf::slice(col, {1, 7}).front(),
                       *(std::min_element(host_strings.begin() + 1, host_strings.begin() + 7)),
                       true,
                       *cudf::make_min_aggregation<reduce_aggregation>(),
                       output_type);
  // MAX
  this->reduction_test(col,
                       *(std::max_element(host_strings.begin(), host_strings.end())),
                       true,
                       *cudf::make_max_aggregation<reduce_aggregation>(),
                       output_type);
  // sliced
  this->reduction_test(cudf::slice(col, {1, 7}).front(),
                       *(std::max_element(host_strings.begin() + 1, host_strings.begin() + 7)),
                       true,
                       *cudf::make_max_aggregation<reduce_aggregation>(),
                       output_type);
}

TEST_P(DictionaryStringReductionTest, MinMaxUnsortedDuplicateKeys)
{
  std::vector<std::string> host_strings(GetParam());
  cudf::data_type output_type{cudf::type_id::STRING};

  cudf::test::dictionary_column_wrapper<std::string> col(host_strings.begin(), host_strings.end());
  cudf::test::strings_column_wrapper new_keys{
    "nine", "two", "eight", "five", "six", "three", "two"};
  auto const dict = cudf::dictionary::set_keys(col, new_keys);

  auto const min_it = std::min_element(host_strings.begin(), host_strings.end());
  auto const max_it = std::max_element(host_strings.begin(), host_strings.end());

  // MIN/MAX
  this->reduction_test(
    dict->view(), *min_it, true, *cudf::make_min_aggregation<reduce_aggregation>(), output_type);
  this->reduction_test(
    dict->view(), *max_it, true, *cudf::make_max_aggregation<reduce_aggregation>(), output_type);

  // ARGMIN/ARGMAX
  this->reduction_test<int>(dict->view(),
                            static_cast<int>(std::distance(host_strings.begin(), min_it)),
                            *cudf::make_argmin_aggregation<reduce_aggregation>());
  this->reduction_test<int>(dict->view(),
                            static_cast<int>(std::distance(host_strings.begin(), max_it)),
                            *cudf::make_argmax_aggregation<reduce_aggregation>());
}

TEST_P(DictionaryStringReductionTest, MinMaxUnsortedDuplicateKeysWithNulls)
{
  // Rows 1 and 6 are null; "two" is the duplicate key (appears twice in new_keys).
  // "two" would be the lexicographic max, so this verifies null rows are excluded.
  // Valid rows: "nine"(0), "five"(2), "three"(3), "five"(4), "six"(5), "eight"(7), "nine"(8)
  // => min="eight" at row 7, max="three" at row 3
  std::vector<std::string> host_strings(GetParam());
  std::vector<bool> valid(host_strings.size(), true);
  valid[1] = false;
  valid[6] = false;
  cudf::data_type output_type{cudf::type_id::STRING};

  cudf::test::dictionary_column_wrapper<std::string> col(
    host_strings.begin(), host_strings.end(), valid.begin());
  cudf::test::strings_column_wrapper new_keys{
    "nine", "two", "eight", "five", "six", "three", "two"};
  auto const dict = cudf::dictionary::set_keys(col, new_keys);

  std::vector<std::string> valid_strings;
  for (size_t i = 0; i < host_strings.size(); ++i) {
    if (valid[i]) { valid_strings.push_back(host_strings[i]); }
  }
  auto const expected_min = *std::min_element(valid_strings.begin(), valid_strings.end());
  auto const expected_max = *std::max_element(valid_strings.begin(), valid_strings.end());

  this->reduction_test(dict->view(),
                       expected_min,
                       true,
                       *cudf::make_min_aggregation<reduce_aggregation>(),
                       output_type);
  this->reduction_test(dict->view(),
                       expected_max,
                       true,
                       *cudf::make_max_aggregation<reduce_aggregation>(),
                       output_type);

  // argmin/argmax: expect the first valid row index that holds the min/max value
  auto first_valid_idx = [&](std::string const& target) {
    for (size_t i = 0; i < host_strings.size(); ++i) {
      if (valid[i] && host_strings[i] == target) { return static_cast<int>(i); }
    }
    return -1;
  };
  this->reduction_test<int>(dict->view(),
                            first_valid_idx(expected_min),
                            *cudf::make_argmin_aggregation<reduce_aggregation>());
  this->reduction_test<int>(dict->view(),
                            first_valid_idx(expected_max),
                            *cudf::make_argmax_aggregation<reduce_aggregation>());
}

template <typename T>
struct DictionaryAnyAllTest : public ReductionTest<bool> {};
using DictionaryAnyAllTypes = cudf::test::Types<int32_t, int64_t, float, double, bool>;
TYPED_TEST_SUITE(DictionaryAnyAllTest, cudf::test::NumericTypes);
TYPED_TEST(DictionaryAnyAllTest, AnyAll)
{
  using T = TypeParam;
  std::vector<int> all_values({true, true, true, true});
  std::vector<T> v_all = convert_values<T>(all_values);
  std::vector<int> none_values({false, false, false, false});
  std::vector<T> v_none = convert_values<T>(none_values);
  std::vector<int> some_values({false, true, false, true});
  std::vector<T> v_some = convert_values<T>(some_values);
  cudf::data_type output_dtype(cudf::type_id::BOOL8);

  auto any_agg = cudf::make_any_aggregation<reduce_aggregation>();
  auto all_agg = cudf::make_all_aggregation<reduce_aggregation>();

  // without nulls
  {
    cudf::test::dictionary_column_wrapper<T> all_col(v_all.begin(), v_all.end());
    EXPECT_TRUE(this->template reduction_test<bool>(all_col, *any_agg, output_dtype).first);
    EXPECT_TRUE(this->template reduction_test<bool>(all_col, *all_agg, output_dtype).first);
    cudf::test::dictionary_column_wrapper<T> none_col(v_none.begin(), v_none.end());
    EXPECT_FALSE(this->template reduction_test<bool>(none_col, *any_agg, output_dtype).first);
    EXPECT_FALSE(this->template reduction_test<bool>(none_col, *all_agg, output_dtype).first);
    cudf::test::dictionary_column_wrapper<T> some_col(v_some.begin(), v_some.end());
    EXPECT_TRUE(this->template reduction_test<bool>(some_col, *any_agg, output_dtype).first);
    EXPECT_FALSE(this->template reduction_test<bool>(some_col, *all_agg, output_dtype).first);
    // sliced test
    auto slice1 = cudf::slice(some_col, {1, 3}).front();
    auto slice2 = cudf::slice(some_col, {1, 2}).front();
    EXPECT_TRUE(this->template reduction_test<bool>(slice1, *any_agg, output_dtype).first);
    EXPECT_TRUE(this->template reduction_test<bool>(slice2, *all_agg, output_dtype).first);
  }
  // with nulls
  {
    std::vector<bool> valid({true, true, false, true});
    cudf::test::dictionary_column_wrapper<T> all_col(v_all.begin(), v_all.end(), valid.begin());
    EXPECT_TRUE(this->template reduction_test<bool>(all_col, *any_agg, output_dtype).first);
    EXPECT_TRUE(this->template reduction_test<bool>(all_col, *all_agg, output_dtype).first);
    cudf::test::dictionary_column_wrapper<T> none_col(v_none.begin(), v_none.end(), valid.begin());
    EXPECT_FALSE(this->template reduction_test<bool>(none_col, *any_agg, output_dtype).first);
    EXPECT_FALSE(this->template reduction_test<bool>(none_col, *all_agg, output_dtype).first);
    cudf::test::dictionary_column_wrapper<T> some_col(v_some.begin(), v_some.end(), valid.begin());
    EXPECT_TRUE(this->template reduction_test<bool>(some_col, *any_agg, output_dtype).first);
    EXPECT_FALSE(this->template reduction_test<bool>(some_col, *all_agg, output_dtype).first);
    // sliced test
    auto slice1 = cudf::slice(some_col, {0, 3}).front();
    auto slice2 = cudf::slice(some_col, {1, 4}).front();
    EXPECT_TRUE(this->template reduction_test<bool>(slice1, *any_agg, output_dtype).first);
    EXPECT_TRUE(this->template reduction_test<bool>(slice2, *all_agg, output_dtype).first);
  }
}

template <typename T>
struct DictionaryReductionTest : public ReductionTest<T> {};

using DictionaryTypes = cudf::test::Types<int32_t, int64_t, float, double>;
TYPED_TEST_SUITE(DictionaryReductionTest, DictionaryTypes);
TYPED_TEST(DictionaryReductionTest, Sum)
{
  using T = TypeParam;
  std::vector<int> int_values({6, -14, 13, 64, 0, -13, -20, 45});
  std::vector<T> v = convert_values<T>(int_values);
  cudf::data_type output_type{cudf::type_to_id<T>()};

  cudf::test::dictionary_column_wrapper<T> col(v.begin(), v.end());

  T expected_value = std::accumulate(v.begin(), v.end(), T{0});
  EXPECT_EQ(this
              ->template reduction_test<T>(
                col, *cudf::make_sum_aggregation<reduce_aggregation>(), output_type)
              .first,
            expected_value);

  // test with nulls
  std::vector<bool> validity({true, true, false, false, true, true, true, true});
  cudf::test::dictionary_column_wrapper<T> col_nulls(v.begin(), v.end(), validity.begin());
  expected_value = [v, validity] {
    auto const r = replace_nulls(v, validity, T{0});
    return std::accumulate(r.begin(), r.end(), T{0});
  }();
  EXPECT_EQ(this
              ->template reduction_test<T>(
                col_nulls, *cudf::make_sum_aggregation<reduce_aggregation>(), output_type)
              .first,
            expected_value);
}

TYPED_TEST(DictionaryReductionTest, MinMaxUnsortedDuplicateKeys)
{
  using T = TypeParam;
  std::vector<int> int_values({50, 10, 40, 20, 30, 10, 50});
  std::vector<T> v = convert_values<T>(int_values);
  cudf::data_type output_type{cudf::type_to_id<T>()};

  cudf::test::dictionary_column_wrapper<T> col(v.begin(), v.end());
  std::vector<int> key_values({30, 50, 10, 40, 20, 10});
  std::vector<T> new_key_values = convert_values<T>(key_values);
  cudf::test::fixed_width_column_wrapper<T> new_keys(new_key_values.begin(), new_key_values.end());
  auto const dict = cudf::dictionary::set_keys(col, new_keys);

  auto const min_it = std::min_element(v.begin(), v.end());
  auto const max_it = std::max_element(v.begin(), v.end());

  EXPECT_EQ(this
              ->template reduction_test<T>(
                dict->view(), *cudf::make_min_aggregation<reduce_aggregation>(), output_type)
              .first,
            *min_it);
  EXPECT_EQ(this
              ->template reduction_test<T>(
                dict->view(), *cudf::make_max_aggregation<reduce_aggregation>(), output_type)
              .first,
            *max_it);

  auto const res        = cudf::minmax(dict->view());
  using ScalarType      = cudf::scalar_type_t<T>;
  auto const min_result = static_cast<ScalarType*>(res.first.get());
  auto const max_result = static_cast<ScalarType*>(res.second.get());
  EXPECT_EQ(T{min_result->value()}, *min_it);
  EXPECT_EQ(T{max_result->value()}, *max_it);

  EXPECT_EQ(
    this
      ->template reduction_test<int32_t>(dict->view(),
                                         *cudf::make_argmin_aggregation<reduce_aggregation>(),
                                         cudf::data_type{cudf::type_id::INT32})
      .first,
    static_cast<int32_t>(std::distance(v.begin(), min_it)));
  EXPECT_EQ(
    this
      ->template reduction_test<int32_t>(dict->view(),
                                         *cudf::make_argmax_aggregation<reduce_aggregation>(),
                                         cudf::data_type{cudf::type_id::INT32})
      .first,
    static_cast<int32_t>(std::distance(v.begin(), max_it)));
}

TYPED_TEST(DictionaryReductionTest, MinMaxAllNulls)
{
  using T = TypeParam;
  std::vector<int> int_values({5, 0, -14, 64});
  std::vector<T> v = convert_values<T>(int_values);
  cudf::data_type output_type{cudf::type_to_id<T>()};

  std::vector<bool> all_null(v.size(), false);
  cudf::test::dictionary_column_wrapper<T> col(v.begin(), v.end(), all_null.begin());

  // reduce min/max on an all-null dictionary must return an invalid (null) scalar
  EXPECT_FALSE(this
                 ->template reduction_test<T>(
                   col, *cudf::make_min_aggregation<reduce_aggregation>(), output_type)
                 .second);
  EXPECT_FALSE(this
                 ->template reduction_test<T>(
                   col, *cudf::make_max_aggregation<reduce_aggregation>(), output_type)
                 .second);

  // cudf::minmax must also return invalid scalars
  auto const res        = cudf::minmax(col);
  using ScalarType      = cudf::scalar_type_t<T>;
  auto const min_result = static_cast<ScalarType*>(res.first.get());
  auto const max_result = static_cast<ScalarType*>(res.second.get());
  EXPECT_FALSE(min_result->is_valid());
  EXPECT_FALSE(max_result->is_valid());
}

TYPED_TEST(DictionaryReductionTest, MinMaxUnsortedDuplicateKeysWithNulls)
{
  using T = TypeParam;
  // Row 2 and row 5 are null; the duplicate key 10 appears at key positions 2 and 5.
  // Null rows must be excluded from min/max even when their key value is extreme.
  std::vector<int> int_values({50, 10, 40, 20, 30, 10, 50});
  std::vector<T> v        = convert_values<T>(int_values);
  std::vector<bool> valid = {true, true, false, true, true, false, true};
  cudf::data_type output_type{cudf::type_to_id<T>()};

  cudf::test::dictionary_column_wrapper<T> col(v.begin(), v.end(), valid.begin());
  std::vector<int> key_values({30, 50, 10, 40, 20, 10});
  std::vector<T> new_key_values = convert_values<T>(key_values);
  cudf::test::fixed_width_column_wrapper<T> new_keys(new_key_values.begin(), new_key_values.end());
  auto const dict = cudf::dictionary::set_keys(col, new_keys);

  // Expected min/max considering only valid rows: {50, 10, -, 20, 30, -, 50}
  std::vector<T> valid_v;
  for (size_t i = 0; i < v.size(); ++i) {
    if (valid[i]) { valid_v.push_back(v[i]); }
  }
  auto const expected_min = *std::min_element(valid_v.begin(), valid_v.end());
  auto const expected_max = *std::max_element(valid_v.begin(), valid_v.end());

  EXPECT_EQ(this
              ->template reduction_test<T>(
                dict->view(), *cudf::make_min_aggregation<reduce_aggregation>(), output_type)
              .first,
            expected_min);
  EXPECT_EQ(this
              ->template reduction_test<T>(
                dict->view(), *cudf::make_max_aggregation<reduce_aggregation>(), output_type)
              .first,
            expected_max);

  auto const res        = cudf::minmax(dict->view());
  using ScalarType      = cudf::scalar_type_t<T>;
  auto const min_result = static_cast<ScalarType*>(res.first.get());
  auto const max_result = static_cast<ScalarType*>(res.second.get());
  EXPECT_EQ(T{min_result->value()}, expected_min);
  EXPECT_EQ(T{max_result->value()}, expected_max);

  // argmin/argmax must also skip null rows
  auto const argmin_idx =
    this
      ->template reduction_test<int32_t>(dict->view(),
                                         *cudf::make_argmin_aggregation<reduce_aggregation>(),
                                         cudf::data_type{cudf::type_id::INT32})
      .first;
  auto const argmax_idx =
    this
      ->template reduction_test<int32_t>(dict->view(),
                                         *cudf::make_argmax_aggregation<reduce_aggregation>(),
                                         cudf::data_type{cudf::type_id::INT32})
      .first;
  EXPECT_EQ(v[argmin_idx], expected_min);
  EXPECT_EQ(v[argmax_idx], expected_max);
  EXPECT_TRUE(valid[argmin_idx]);
  EXPECT_TRUE(valid[argmax_idx]);
}

TYPED_TEST(DictionaryReductionTest, Product)
{
  using T = TypeParam;
  std::vector<int> int_values({5, -1, 1, 0, 3, 2, 4});
  std::vector<TypeParam> v = convert_values<TypeParam>(int_values);
  cudf::data_type output_type{cudf::type_to_id<T>()};

  auto calc_prod = [](std::vector<T> const& v) {
    return std::accumulate(v.cbegin(), v.cend(), T{1}, std::multiplies<T>());
  };

  // test without nulls
  cudf::test::dictionary_column_wrapper<T> col(v.begin(), v.end());

  EXPECT_EQ(this
              ->template reduction_test<T>(
                col, *cudf::make_product_aggregation<reduce_aggregation>(), output_type)
              .first,
            calc_prod(v));

  // test with nulls
  std::vector<bool> validity({true, true, false, false, true, true, true});
  cudf::test::dictionary_column_wrapper<T> col_nulls(v.begin(), v.end(), validity.begin());

  EXPECT_EQ(this
              ->template reduction_test<T>(
                col_nulls, *cudf::make_product_aggregation<reduce_aggregation>(), output_type)
              .first,
            calc_prod(replace_nulls(v, validity, T{1})));
}

TYPED_TEST(DictionaryReductionTest, SumOfSquare)
{
  using T = TypeParam;
  std::vector<int> int_values({-3, 2, 1, 0, 5, -3, -2});
  std::vector<T> v = convert_values<T>(int_values);
  cudf::data_type output_type{cudf::type_to_id<T>()};

  auto calc_reduction = [](std::vector<T> const& v) {
    return std::accumulate(v.cbegin(), v.cend(), T{0}, [](T acc, T i) { return acc + i * i; });
  };

  // test without nulls
  cudf::test::dictionary_column_wrapper<T> col(v.begin(), v.end());

  EXPECT_EQ(this
              ->template reduction_test<T>(
                col, *cudf::make_sum_of_squares_aggregation<reduce_aggregation>(), output_type)
              .first,
            calc_reduction(v));

  // test with nulls
  std::vector<bool> validity({true, true, false, false, true, true, true, true});
  cudf::test::dictionary_column_wrapper<T> col_nulls(v.begin(), v.end(), validity.begin());

  EXPECT_EQ(
    this
      ->template reduction_test<T>(
        col_nulls, *cudf::make_sum_of_squares_aggregation<reduce_aggregation>(), output_type)
      .first,
    calc_reduction(replace_nulls(v, validity, T{0})));
}

TYPED_TEST(DictionaryReductionTest, Mean)
{
  using T = TypeParam;
  std::vector<int> int_values({-3, 2, 1, 0, 5, -3, -2, 28});
  std::vector<T> v = convert_values<T>(int_values);
  cudf::data_type output_type{cudf::type_to_id<double>()};

  auto calc_mean = [](std::vector<T> const& v, cudf::size_type valid_count) {
    double sum = std::accumulate(v.cbegin(), v.cend(), double{0});
    return sum / valid_count;
  };

  // test without nulls
  cudf::test::dictionary_column_wrapper<T> col(v.begin(), v.end());

  EXPECT_EQ(this
              ->template reduction_test<double>(
                col, *cudf::make_mean_aggregation<reduce_aggregation>(), output_type)
              .first,
            calc_mean(v, v.size()));

  // test with nulls
  std::vector<bool> validity({true, true, false, true, true, true, false, true});
  cudf::test::dictionary_column_wrapper<T> col_nulls(v.begin(), v.end(), validity.begin());

  cudf::size_type valid_count = std::count(validity.begin(), validity.end(), true);

  EXPECT_EQ(this
              ->template reduction_test<double>(
                col_nulls, *cudf::make_mean_aggregation<reduce_aggregation>(), output_type)
              .first,
            calc_mean(replace_nulls(v, validity, T{0}), valid_count));
}

TYPED_TEST(DictionaryReductionTest, VarStd)
{
  using T = TypeParam;
  std::vector<int> int_values({-3, 2, 1, 0, 5, -3, -2, 28});
  std::vector<T> v = convert_values<T>(int_values);
  cudf::data_type output_type{cudf::type_to_id<double>()};

  // test without nulls
  cudf::test::dictionary_column_wrapper<T> col(v.begin(), v.end());

  cudf::size_type const ddof = 1;
  double var                 = calc_var(v, ddof);
  double std                 = std::sqrt(var);
  auto var_agg               = cudf::make_variance_aggregation<reduce_aggregation>(ddof);
  auto std_agg               = cudf::make_std_aggregation<reduce_aggregation>(ddof);

  EXPECT_EQ(this->template reduction_test<double>(col, *var_agg, output_type).first, var);
  EXPECT_EQ(this->template reduction_test<double>(col, *std_agg, output_type).first, std);

  // test with nulls
  std::vector<bool> validity({true, true, false, true, true, true, false, true});
  cudf::test::dictionary_column_wrapper<T> col_nulls(v.begin(), v.end(), validity.begin());

  double var_nulls = calc_var(v, ddof, validity);
  double std_nulls = std::sqrt(var_nulls);

  EXPECT_DOUBLE_EQ(this->template reduction_test<double>(col_nulls, *var_agg, output_type).first,
                   var_nulls);
  EXPECT_DOUBLE_EQ(this->template reduction_test<double>(col_nulls, *std_agg, output_type).first,
                   std_nulls);
}

TYPED_TEST(DictionaryReductionTest, NthElement)
{
  using T = TypeParam;
  std::vector<int> int_values({-3, 2, 1, 0, 5, -3, -2, 28});
  std::vector<T> v = convert_values<T>(int_values);
  cudf::data_type output_type{cudf::type_to_id<T>()};

  // test without nulls
  cudf::test::dictionary_column_wrapper<T> col(v.begin(), v.end());
  cudf::size_type n = 5;
  EXPECT_EQ(this
              ->template reduction_test<T>(col,
                                           *cudf::make_nth_element_aggregation<reduce_aggregation>(
                                             n, cudf::null_policy::INCLUDE),
                                           output_type)
              .first,
            v[n]);

  // test with nulls
  std::vector<bool> validity({true, true, false, true, true, true, false, true});
  cudf::test::dictionary_column_wrapper<T> col_nulls(v.begin(), v.end(), validity.begin());

  EXPECT_EQ(this
              ->template reduction_test<T>(col_nulls,
                                           *cudf::make_nth_element_aggregation<reduce_aggregation>(
                                             n, cudf::null_policy::INCLUDE),
                                           output_type)
              .first,
            v[n]);
  EXPECT_FALSE(
    this
      ->template reduction_test<T>(
        col_nulls,
        *cudf::make_nth_element_aggregation<reduce_aggregation>(2, cudf::null_policy::INCLUDE),
        output_type)
      .second);
}

TYPED_TEST(DictionaryReductionTest, UniqueCount)
{
  using T = TypeParam;
  std::vector<int> int_values({1, -3, 1, 2, 0, 2, -4, 45});  // 6 unique values
  std::vector<T> v = convert_values<T>(int_values);
  cudf::data_type output_type{cudf::type_to_id<cudf::size_type>()};

  // test without nulls
  cudf::test::dictionary_column_wrapper<T> col(v.begin(), v.end());
  EXPECT_EQ(this
              ->template reduction_test<int>(
                col,
                *cudf::make_nunique_aggregation<reduce_aggregation>(cudf::null_policy::INCLUDE),
                output_type)
              .first,
            6);

  // test with nulls
  std::vector<bool> validity({true, true, true, false, true, true, true, true});
  cudf::test::dictionary_column_wrapper<T> col_nulls(v.begin(), v.end(), validity.begin());

  EXPECT_EQ(this
              ->template reduction_test<int>(
                col_nulls,
                *cudf::make_nunique_aggregation<reduce_aggregation>(cudf::null_policy::INCLUDE),
                output_type)
              .first,
            7);
  EXPECT_EQ(this
              ->template reduction_test<int>(
                col_nulls,
                *cudf::make_nunique_aggregation<reduce_aggregation>(cudf::null_policy::EXCLUDE),
                output_type)
              .first,
            6);
}

TYPED_TEST(DictionaryReductionTest, Median)
{
  using T = TypeParam;
  std::vector<int> int_values({6, -14, 13, 64, 0, -13, -20, 45});
  std::vector<T> v = convert_values<T>(int_values);
  auto output_type = cudf::data_type{cudf::type_to_id<double>()};

  // test without nulls
  cudf::test::dictionary_column_wrapper<T> col(v.begin(), v.end());
  EXPECT_EQ(this
              ->template reduction_test<double>(
                col, *cudf::make_median_aggregation<reduce_aggregation>(), output_type)
              .first,
            (std::is_signed_v<T>) ? 3.0 : 13.5);

  // test with nulls
  std::vector<bool> validity({true, true, true, false, true, true, true, true});
  cudf::test::dictionary_column_wrapper<T> col_nulls(v.begin(), v.end(), validity.begin());
  EXPECT_EQ(this
              ->template reduction_test<double>(
                col_nulls, *cudf::make_median_aggregation<reduce_aggregation>(), output_type)
              .first,
            (std::is_signed_v<T>) ? 0.0 : 13.0);
}

TYPED_TEST(DictionaryReductionTest, Quantile)
{
  using T = TypeParam;
  std::vector<int> int_values({6, -14, 13, 64, 0, -13, -20, 45});
  std::vector<T> v = convert_values<T>(int_values);
  cudf::interpolation interp{cudf::interpolation::LINEAR};
  auto output_type = cudf::data_type{cudf::type_to_id<double>()};

  // test without nulls
  cudf::test::dictionary_column_wrapper<T> col(v.begin(), v.end());
  double expected_value = std::is_same_v<T, bool> || std::is_unsigned_v<T> ? 0.0 : -20.0;
  EXPECT_EQ(
    this
      ->template reduction_test<double>(
        col, *cudf::make_quantile_aggregation<reduce_aggregation>({0.0}, interp), output_type)
      .first,
    expected_value);
  EXPECT_EQ(
    this
      ->template reduction_test<double>(
        col, *cudf::make_quantile_aggregation<reduce_aggregation>({1.0}, interp), output_type)
      .first,
    64.0);

  // test with nulls
  std::vector<bool> validity({true, true, true, false, true, true, true, true});
  cudf::test::dictionary_column_wrapper<T> col_nulls(v.begin(), v.end(), validity.begin());

  EXPECT_EQ(
    this
      ->template reduction_test<double>(
        col_nulls, *cudf::make_quantile_aggregation<reduce_aggregation>({0}, interp), output_type)
      .first,
    expected_value);
  EXPECT_EQ(
    this
      ->template reduction_test<double>(
        col_nulls, *cudf::make_quantile_aggregation<reduce_aggregation>({1}, interp), output_type)
      .first,
    45.0);
}

struct ListReductionTest : public cudf::test::BaseFixture {
  void reduction_test(cudf::column_view const& input_data,
                      cudf::column_view const& expected_value,
                      bool succeeded_condition,
                      bool is_valid,
                      reduce_aggregation const& agg)
  {
    auto statement = [&]() {
      std::unique_ptr<cudf::scalar> result =
        cudf::reduce(input_data, agg, cudf::data_type(cudf::type_id::LIST));
      auto list_result = dynamic_cast<cudf::list_scalar*>(result.get());
      EXPECT_EQ(is_valid, list_result->is_valid());
      if (is_valid) {
        CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_value, list_result->view());
      } else {
        CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected_value, list_result->view());
      }
    };

    if (succeeded_condition) {
      CUDF_EXPECT_NO_THROW(statement());
    } else {
      EXPECT_ANY_THROW(statement());
    }
  }
};

TEST_F(ListReductionTest, ListReductionNthElement)
{
  using LCW        = cudf::test::lists_column_wrapper<int>;
  using ElementCol = cudf::test::fixed_width_column_wrapper<int>;

  // test without nulls
  LCW col{{-3}, {2, 1}, {0, 5, -3}, {-2}, {}, {28}};
  this->reduction_test(
    col,
    ElementCol{0, 5, -3},  // expected_value,
    true,
    true,
    *cudf::make_nth_element_aggregation<reduce_aggregation>(2, cudf::null_policy::INCLUDE));

  // test with null-exclude
  std::vector<bool> validity{true, false, false, true, true, false};
  LCW col_nulls({{-3}, {2, 1}, {0, 5, -3}, {-2}, {}, {28}}, validity.begin());
  this->reduction_test(
    col_nulls,
    ElementCol{-2},  // expected_value,
    true,
    true,
    *cudf::make_nth_element_aggregation<reduce_aggregation>(1, cudf::null_policy::EXCLUDE));

  // test with null-include
  this->reduction_test(
    col_nulls,
    ElementCol{},  // expected_value,
    true,
    false,
    *cudf::make_nth_element_aggregation<reduce_aggregation>(1, cudf::null_policy::INCLUDE));
}

TEST_F(ListReductionTest, NestedListReductionNthElement)
{
  using LCW = cudf::test::lists_column_wrapper<int>;

  // test without nulls
  auto validity    = std::vector<bool>{true, false, false, true, true};
  auto nested_list = LCW(
    {{LCW{}, LCW{2, 3, 4}}, {}, {LCW{5}, LCW{6}, LCW{7, 8}}, {LCW{9, 10}}, {LCW{11}, LCW{12, 13}}},
    validity.begin());
  this->reduction_test(
    nested_list,
    LCW{{}, {2, 3, 4}},  // expected_value,
    true,
    true,
    *cudf::make_nth_element_aggregation<reduce_aggregation>(0, cudf::null_policy::INCLUDE));

  // test with null-include
  this->reduction_test(
    nested_list,
    LCW{},  // expected_value,
    true,
    false,
    *cudf::make_nth_element_aggregation<reduce_aggregation>(2, cudf::null_policy::INCLUDE));

  // test with null-exclude
  this->reduction_test(
    nested_list,
    LCW{{11}, {12, 13}},  // expected_value,
    true,
    true,
    *cudf::make_nth_element_aggregation<reduce_aggregation>(2, cudf::null_policy::EXCLUDE));
}

TEST_F(ListReductionTest, NonValidListReductionNthElement)
{
  using LCW        = cudf::test::lists_column_wrapper<int>;
  using ElementCol = cudf::test::fixed_width_column_wrapper<int>;

  // test against col.size() <= col.null_count()
  std::vector<bool> validity{false};
  this->reduction_test(
    LCW{{{1, 2}}, validity.begin()},
    ElementCol{},  // expected_value,
    true,
    false,
    *cudf::make_nth_element_aggregation<reduce_aggregation>(0, cudf::null_policy::INCLUDE));

  // test against empty input
  this->reduction_test(
    LCW{},
    ElementCol{},  // expected_value,
    true,
    false,
    *cudf::make_nth_element_aggregation<reduce_aggregation>(0, cudf::null_policy::INCLUDE));
}

TEST_F(ListReductionTest, ReductionMinMaxAndArgMinMaxNoNull)
{
  using INTS_CW          = cudf::test::fixed_width_column_wrapper<int>;
  using LISTS_CW         = cudf::test::lists_column_wrapper<int>;
  using STRINGS_CW       = cudf::test::strings_column_wrapper;
  using LISTS_STRINGS_CW = cudf::test::lists_column_wrapper<cudf::string_view>;

  {
    auto const input = LISTS_CW{{3, 4}, {1, 2}, {5, 6, 7}, {0, 8}, {9, 10}, {1, 0}};
    this->reduction_test(
      input, INTS_CW{0, 8}, true, true, *cudf::make_min_aggregation<reduce_aggregation>());
    this->reduction_test(
      input, INTS_CW{9, 10}, true, true, *cudf::make_max_aggregation<reduce_aggregation>());

    using ScalarType  = cudf::scalar_type_t<int>;
    auto const argmin = cudf::reduce(input,
                                     *cudf::make_argmin_aggregation<reduce_aggregation>(),
                                     cudf::data_type{cudf::type_id::INT32});
    auto const argmax = cudf::reduce(input,
                                     *cudf::make_argmax_aggregation<reduce_aggregation>(),
                                     cudf::data_type{cudf::type_id::INT32});
    EXPECT_EQ(static_cast<ScalarType*>(argmin.get())->value(), 3);
    EXPECT_EQ(static_cast<ScalarType*>(argmax.get())->value(), 4);
  }

  {
    auto const input = LISTS_STRINGS_CW{
      {"34", "43"}, {"12", "21"}, {"567", "6", "765"}, {"08", "8"}, {"109", "10"}, {"10", "00"}};
    this->reduction_test(
      input, STRINGS_CW{"08", "8"}, true, true, *cudf::make_min_aggregation<reduce_aggregation>());
    this->reduction_test(input,
                         STRINGS_CW{"567", "6", "765"},
                         true,
                         true,
                         *cudf::make_max_aggregation<reduce_aggregation>());

    using ScalarType  = cudf::scalar_type_t<int>;
    auto const argmin = cudf::reduce(input,
                                     *cudf::make_argmin_aggregation<reduce_aggregation>(),
                                     cudf::data_type{cudf::type_id::INT32});
    auto const argmax = cudf::reduce(input,
                                     *cudf::make_argmax_aggregation<reduce_aggregation>(),
                                     cudf::data_type{cudf::type_id::INT32});
    EXPECT_EQ(static_cast<ScalarType*>(argmin.get())->value(), 3);
    EXPECT_EQ(static_cast<ScalarType*>(argmax.get())->value(), 2);
  }
}

TEST_F(ListReductionTest, ReductionMinMaxAndArgMinMaxSlicedInput)
{
  using INTS_CW          = cudf::test::fixed_width_column_wrapper<int>;
  using LISTS_CW         = cudf::test::lists_column_wrapper<int>;
  using STRINGS_CW       = cudf::test::strings_column_wrapper;
  using LISTS_STRINGS_CW = cudf::test::lists_column_wrapper<cudf::string_view>;

  {
    auto const input_original = LISTS_CW{{9, 9} /*don't care*/,
                                         {0, 0} /*don't care*/,
                                         {3, 4},
                                         {1, 2},
                                         {5, 6, 7},
                                         {0, 8},
                                         {9, 10},
                                         {1, 0},
                                         {0, 7} /*don't care*/};
    auto const input          = cudf::slice(input_original, {2, 8})[0];
    this->reduction_test(
      input, INTS_CW{0, 8}, true, true, *cudf::make_min_aggregation<reduce_aggregation>());
    this->reduction_test(
      input, INTS_CW{9, 10}, true, true, *cudf::make_max_aggregation<reduce_aggregation>());
  }
  {
    auto const input_original = LISTS_STRINGS_CW{{"08", "8"} /*don't care*/,
                                                 {"999", "8"} /*don't care*/,
                                                 {"34", "43"},
                                                 {"12", "21"},
                                                 {"567", "6", "765"},
                                                 {"08", "8"},
                                                 {"109", "10"},
                                                 {"10", "00"}};
    auto const input          = cudf::slice(input_original, {2, 8})[0];
    this->reduction_test(
      input, STRINGS_CW{"08", "8"}, true, true, *cudf::make_min_aggregation<reduce_aggregation>());
    this->reduction_test(input,
                         STRINGS_CW{"567", "6", "765"},
                         true,
                         true,
                         *cudf::make_max_aggregation<reduce_aggregation>());
  }
}

TEST_F(ListReductionTest, ReductionMinMaxWithNulls)
{
  using INTS_CW  = cudf::test::fixed_width_column_wrapper<int>;
  using LISTS_CW = cudf::test::lists_column_wrapper<int>;
  using cudf::test::iterators::null_at;
  using cudf::test::iterators::nulls_at;
  constexpr int null{0};

  auto const input = LISTS_CW{{LISTS_CW{3, 4},
                               LISTS_CW{1, 2},
                               LISTS_CW{{1, null}, null_at(1)},
                               LISTS_CW{} /*null*/,
                               LISTS_CW{5, 6, 7},
                               LISTS_CW{1, 8},
                               LISTS_CW{{9, null}, null_at(1)},
                               LISTS_CW{} /*null*/},
                              nulls_at({3, 7})};
  this->reduction_test(input,
                       INTS_CW{{1, null}, null_at(1)},
                       true,
                       true,
                       *cudf::make_min_aggregation<reduce_aggregation>());
  this->reduction_test(input,
                       INTS_CW{{9, null}, null_at(1)},
                       true,
                       true,
                       *cudf::make_max_aggregation<reduce_aggregation>());

  using ScalarType  = cudf::scalar_type_t<int>;
  auto const argmin = cudf::reduce(input,
                                   *cudf::make_argmin_aggregation<reduce_aggregation>(),
                                   cudf::data_type{cudf::type_id::INT32});
  auto const argmax = cudf::reduce(input,
                                   *cudf::make_argmax_aggregation<reduce_aggregation>(),
                                   cudf::data_type{cudf::type_id::INT32});
  EXPECT_EQ(static_cast<ScalarType*>(argmin.get())->value(), 2);
  EXPECT_EQ(static_cast<ScalarType*>(argmax.get())->value(), 6);
}

struct StructReductionTest : public cudf::test::BaseFixture {
  using SCW = cudf::test::structs_column_wrapper;

  void reduction_test(cudf::column_view const& struct_column,
                      cudf::table_view const& expected_value,
                      bool succeeded_condition,
                      bool is_valid,
                      reduce_aggregation const& agg)
  {
    auto statement = [&]() {
      std::unique_ptr<cudf::scalar> result =
        cudf::reduce(struct_column, agg, cudf::data_type(cudf::type_id::STRUCT));
      auto struct_result = dynamic_cast<cudf::struct_scalar*>(result.get());
      EXPECT_EQ(is_valid, struct_result->is_valid());
      if (is_valid) { CUDF_TEST_EXPECT_TABLES_EQUIVALENT(expected_value, struct_result->view()); }
    };

    if (succeeded_condition) {
      CUDF_EXPECT_NO_THROW(statement());
    } else {
      EXPECT_ANY_THROW(statement());
    }
  }
};

TEST_F(StructReductionTest, StructReductionNthElement)
{
  using ICW = cudf::test::fixed_width_column_wrapper<int>;

  // test without nulls
  auto child0 = *ICW{-3, 2, 1, 0, 5, -3, -2, 28}.release();
  auto child1 = *ICW{0, 1, 2, 3, 4, 5, 6, 7}.release();
  auto child2 = *ICW{{-10, 10, -100, 100, -1000, 1000, -10000, 10000},
                     {true, false, false, true, true, true, false, true}}
                   .release();
  std::vector<std::unique_ptr<cudf::column>> input_vector;
  input_vector.push_back(std::make_unique<cudf::column>(child0));
  input_vector.push_back(std::make_unique<cudf::column>(child1));
  input_vector.push_back(std::make_unique<cudf::column>(child2));
  auto struct_col  = SCW(std::move(input_vector));
  auto result_col0 = ICW{1};
  auto result_col1 = ICW{2};
  auto result_col2 = ICW{{0}, {false}};
  this->reduction_test(
    struct_col,
    cudf::table_view{{result_col0, result_col1, result_col2}},  // expected_value,
    true,
    true,
    *cudf::make_nth_element_aggregation<reduce_aggregation>(2, cudf::null_policy::INCLUDE));

  // test with null-include
  std::vector<bool> validity{true, true, true, false, true, false, false, true};
  input_vector.clear();
  input_vector.push_back(std::make_unique<cudf::column>(child0));
  input_vector.push_back(std::make_unique<cudf::column>(child1));
  input_vector.push_back(std::make_unique<cudf::column>(child2));
  struct_col  = SCW(std::move(input_vector), validity);
  result_col0 = ICW{{0}, {false}};
  result_col1 = ICW{{0}, {false}};
  result_col2 = ICW{{0}, {false}};
  this->reduction_test(
    struct_col,
    cudf::table_view{{result_col0, result_col1, result_col2}},  // expected_value,
    true,
    false,
    *cudf::make_nth_element_aggregation<reduce_aggregation>(6, cudf::null_policy::INCLUDE));

  // test with null-exclude
  result_col0 = ICW{{28}, {true}};
  result_col1 = ICW{{7}, {true}};
  result_col2 = ICW{{10000}, {true}};
  this->reduction_test(
    struct_col,
    cudf::table_view{{result_col0, result_col1, result_col2}},  // expected_value,
    true,
    true,
    *cudf::make_nth_element_aggregation<reduce_aggregation>(4, cudf::null_policy::EXCLUDE));
}

TEST_F(StructReductionTest, NestedStructReductionNthElement)
{
  using ICW = cudf::test::fixed_width_column_wrapper<int>;
  using LCW = cudf::test::lists_column_wrapper<int>;

  auto int_col0    = ICW{-4, -3, -2, -1, 0};
  auto struct_col0 = SCW({int_col0}, std::vector<bool>{true, false, false, true, true});
  auto int_col1    = ICW{0, 1, 2, 3, 4};
  auto list_col    = LCW{{0}, {}, {1, 2}, {3}, {4}};
  auto struct_col1 =
    SCW({struct_col0, int_col1, list_col}, std::vector<bool>{true, true, true, false, true});
  auto result_child0 = ICW{0};
  auto result_col0   = SCW({result_child0}, std::vector<bool>{false});
  auto result_col1   = ICW{{1}, {true}};
  auto result_col2   = LCW({LCW{}}, std::vector<bool>{true}.begin());
  // test without nulls
  this->reduction_test(
    struct_col1,
    cudf::table_view{{result_col0, result_col1, result_col2}},  // expected_value,
    true,
    true,
    *cudf::make_nth_element_aggregation<reduce_aggregation>(1, cudf::null_policy::INCLUDE));

  // test with null-include
  result_child0 = ICW{0};
  result_col0   = SCW({result_child0}, std::vector<bool>{false});
  result_col1   = ICW{{0}, {false}};
  result_col2   = LCW({LCW{3}}, std::vector<bool>{false}.begin());
  this->reduction_test(
    struct_col1,
    cudf::table_view{{result_col0, result_col1, result_col2}},  // expected_value,
    true,
    false,
    *cudf::make_nth_element_aggregation<reduce_aggregation>(3, cudf::null_policy::INCLUDE));

  // test with null-exclude
  result_child0 = ICW{0};
  result_col0   = SCW({result_child0}, std::vector<bool>{true});
  result_col1   = ICW{{4}, {true}};
  result_col2   = LCW({LCW{4}}, std::vector<bool>{true}.begin());
  this->reduction_test(
    struct_col1,
    cudf::table_view{{result_col0, result_col1, result_col2}},  // expected_value,
    true,
    true,
    *cudf::make_nth_element_aggregation<reduce_aggregation>(3, cudf::null_policy::EXCLUDE));
}

TEST_F(StructReductionTest, NonValidStructReductionNthElement)
{
  using ICW = cudf::test::fixed_width_column_wrapper<int>;

  // test against col.size() <= col.null_count()
  auto child0     = ICW{-3, 3};
  auto child1     = ICW{0, 0};
  auto child2     = ICW{{-10, 10}, {false, true}};
  auto struct_col = SCW{{child0, child1, child2}, {false, false}};
  auto ret_col0   = ICW{{0}, {false}};
  auto ret_col1   = ICW{{0}, {false}};
  auto ret_col2   = ICW{{0}, {false}};
  this->reduction_test(
    struct_col,
    cudf::table_view{{ret_col0, ret_col1, ret_col2}},  // expected_value,
    true,
    false,
    *cudf::make_nth_element_aggregation<reduce_aggregation>(0, cudf::null_policy::INCLUDE));

  // test against empty input (would fail because we can not create empty struct scalar)
  child0     = ICW{};
  child1     = ICW{};
  child2     = ICW{};
  struct_col = SCW{{child0, child1, child2}};
  ret_col0   = ICW{};
  ret_col1   = ICW{};
  ret_col2   = ICW{};
  this->reduction_test(
    struct_col,
    cudf::table_view{{ret_col0, ret_col1, ret_col2}},  // expected_value,
    false,
    false,
    *cudf::make_nth_element_aggregation<reduce_aggregation>(0, cudf::null_policy::INCLUDE));
}

TEST_F(StructReductionTest, StructReductionMinMaxAndArgMinMaxNoNull)
{
  using INTS_CW    = cudf::test::fixed_width_column_wrapper<int>;
  using STRINGS_CW = cudf::test::strings_column_wrapper;
  using STRUCTS_CW = cudf::test::structs_column_wrapper;

  auto const input = [] {
    auto child1 = STRINGS_CW{"año", "bit", "₹1", "aaa", "zit", "bat", "aab", "$1", "€1", "wut"};
    auto child2 = INTS_CW{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    return STRUCTS_CW{{child1, child2}};
  }();

  {
    auto const expected_child1 = STRINGS_CW{"$1"};
    auto const expected_child2 = INTS_CW{8};
    this->reduction_test(input,
                         cudf::table_view{{expected_child1, expected_child2}},
                         true,
                         true,
                         *cudf::make_min_aggregation<reduce_aggregation>());
  }

  {
    auto const expected_child1 = STRINGS_CW{"₹1"};
    auto const expected_child2 = INTS_CW{3};
    this->reduction_test(input,
                         cudf::table_view{{expected_child1, expected_child2}},
                         true,
                         true,
                         *cudf::make_max_aggregation<reduce_aggregation>());
  }

  {
    using ScalarType  = cudf::scalar_type_t<int>;
    auto const argmin = cudf::reduce(input,
                                     *cudf::make_argmin_aggregation<reduce_aggregation>(),
                                     cudf::data_type{cudf::type_id::INT32});
    auto const argmax = cudf::reduce(input,
                                     *cudf::make_argmax_aggregation<reduce_aggregation>(),
                                     cudf::data_type{cudf::type_id::INT32});
    EXPECT_EQ(static_cast<ScalarType*>(argmin.get())->value(), 7);
    EXPECT_EQ(static_cast<ScalarType*>(argmax.get())->value(), 2);
  }
}

TEST_F(StructReductionTest, StructReductionMinMaxAndArgMinMaxSlicedInput)
{
  using INTS_CW    = cudf::test::fixed_width_column_wrapper<int>;
  using STRINGS_CW = cudf::test::strings_column_wrapper;
  using STRUCTS_CW = cudf::test::structs_column_wrapper;
  constexpr int32_t dont_care{1};

  auto const input_original = [] {
    auto child1 = STRINGS_CW{"$dont_care",
                             "$dont_care",
                             "año",
                             "bit",
                             "₹1",
                             "aaa",
                             "zit",
                             "bat",
                             "aab",
                             "$1",
                             "€1",
                             "wut",
                             "₹dont_care"};
    auto child2 = INTS_CW{dont_care, dont_care, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, dont_care};
    return STRUCTS_CW{{child1, child2}};
  }();

  auto const input = cudf::slice(input_original, {2, 12})[0];

  {
    auto const expected_child1 = STRINGS_CW{"$1"};
    auto const expected_child2 = INTS_CW{8};
    this->reduction_test(input,
                         cudf::table_view{{expected_child1, expected_child2}},
                         true,
                         true,
                         *cudf::make_min_aggregation<reduce_aggregation>());
  }

  {
    auto const expected_child1 = STRINGS_CW{"₹1"};
    auto const expected_child2 = INTS_CW{3};
    this->reduction_test(input,
                         cudf::table_view{{expected_child1, expected_child2}},
                         true,
                         true,
                         *cudf::make_max_aggregation<reduce_aggregation>());
  }

  {
    using ScalarType  = cudf::scalar_type_t<int>;
    auto const argmin = cudf::reduce(input,
                                     *cudf::make_argmin_aggregation<reduce_aggregation>(),
                                     cudf::data_type{cudf::type_id::INT32});
    auto const argmax = cudf::reduce(input,
                                     *cudf::make_argmax_aggregation<reduce_aggregation>(),
                                     cudf::data_type{cudf::type_id::INT32});
    EXPECT_EQ(static_cast<ScalarType*>(argmin.get())->value(), 7);
    EXPECT_EQ(static_cast<ScalarType*>(argmax.get())->value(), 2);
  }
}

TEST_F(StructReductionTest, StructReductionMinMaxAndArgMinMaxWithNulls)
{
  using INTS_CW    = cudf::test::fixed_width_column_wrapper<int>;
  using STRINGS_CW = cudf::test::strings_column_wrapper;
  using STRUCTS_CW = cudf::test::structs_column_wrapper;
  using cudf::test::iterators::null_at;
  using cudf::test::iterators::nulls_at;

  auto const input = [] {
    auto child1 = STRINGS_CW{{"año",
                              "bit",
                              "",     // child null
                              "aaa",  // parent null
                              "zit",
                              "bat",
                              "aab",
                              "",    // child null
                              "€1",  // parent null
                              "wut"},
                             nulls_at({2, 7})};
    auto child2 = INTS_CW{{1,
                           2,
                           0,  // child null
                           4,  // parent null
                           5,
                           6,
                           7,
                           0,  // child null
                           9,  // parent NULL
                           10},
                          nulls_at({2, 7})};
    return STRUCTS_CW{{child1, child2}, nulls_at({3, 8})};
  }();

  {
    // In the structs column, the min struct is {null, null}.
    auto const expected_child1 = STRINGS_CW{{""}, null_at(0)};
    auto const expected_child2 = INTS_CW{{8}, null_at(0)};
    this->reduction_test(input,
                         cudf::table_view{{expected_child1, expected_child2}},
                         true,
                         true,
                         *cudf::make_min_aggregation<reduce_aggregation>());
  }

  {
    auto const expected_child1 = STRINGS_CW{"zit"};
    auto const expected_child2 = INTS_CW{5};
    this->reduction_test(input,
                         cudf::table_view{{expected_child1, expected_child2}},
                         true,
                         true,
                         *cudf::make_max_aggregation<reduce_aggregation>());
  }

  {
    using ScalarType  = cudf::scalar_type_t<int>;
    auto const argmin = cudf::reduce(input,
                                     *cudf::make_argmin_aggregation<reduce_aggregation>(),
                                     cudf::data_type{cudf::type_id::INT32});
    auto const argmax = cudf::reduce(input,
                                     *cudf::make_argmax_aggregation<reduce_aggregation>(),
                                     cudf::data_type{cudf::type_id::INT32});
    EXPECT_EQ(static_cast<ScalarType*>(argmin.get())->value(), 2);
    EXPECT_EQ(static_cast<ScalarType*>(argmax.get())->value(), 4);
  }
}

// Helpers to map a parameter type to its storage rep (T for integers, T::rep for decimals).
template <typename T, bool = cudf::is_fixed_point<T>()>
struct sum_overflow_rep {
  using type = T;
};
template <typename T>
struct sum_overflow_rep<T, true> {
  using type = typename T::rep;
};
template <typename T>
using sum_overflow_rep_t = typename sum_overflow_rep<T>::type;

// Test for SUM_OVERFLOW aggregation using regular reduce() function.
// Parametrized on signed integer types AND decimal types.
template <typename T>
struct ReduceOverflowTest : public cudf::test::BaseFixture {
  using Rep = sum_overflow_rep_t<T>;
  static constexpr numeric::scale_type scale{0};

  cudf::test::fixed_width_column_wrapper<T> make_col(std::initializer_list<Rep> values)
    requires(!cudf::is_fixed_point<T>())
  {
    return cudf::test::fixed_width_column_wrapper<T>(values);
  }
  cudf::test::fixed_point_column_wrapper<Rep> make_col(std::initializer_list<Rep> values)
    requires(cudf::is_fixed_point<T>())
  {
    return cudf::test::fixed_point_column_wrapper<Rep>(values, scale);
  }

  cudf::test::fixed_width_column_wrapper<T> make_null_col(std::initializer_list<Rep> values,
                                                          std::initializer_list<bool> validity)
    requires(!cudf::is_fixed_point<T>())
  {
    return cudf::test::fixed_width_column_wrapper<T>(values, std::cbegin(validity));
  }
  cudf::test::fixed_point_column_wrapper<Rep> make_null_col(std::initializer_list<Rep> values,
                                                            std::initializer_list<bool> validity)
    requires(cudf::is_fixed_point<T>())
  {
    return cudf::test::fixed_point_column_wrapper<Rep>(values, std::cbegin(validity), scale);
  }

  cudf::test::fixed_width_column_wrapper<T> make_col_from_vec(std::vector<Rep> const& values)
    requires(!cudf::is_fixed_point<T>())
  {
    return cudf::test::fixed_width_column_wrapper<T>(values.begin(), values.end());
  }
  cudf::test::fixed_point_column_wrapper<Rep> make_col_from_vec(std::vector<Rep> const& values)
    requires(cudf::is_fixed_point<T>())
  {
    return cudf::test::fixed_point_column_wrapper<Rep>(values.begin(), values.end(), scale);
  }

  std::unique_ptr<cudf::scalar> make_init_scalar(Rep value)
  {
    if constexpr (cudf::is_fixed_point<T>()) {
      return cudf::make_fixed_point_scalar<T>(value, scale);
    } else {
      return cudf::make_fixed_width_scalar<T>(static_cast<T>(value));
    }
  }

  Rep get_sum_value(std::unique_ptr<cudf::scalar> const& sum_result)
  {
    if constexpr (cudf::is_fixed_point<T>()) {
      return static_cast<cudf::fixed_point_scalar<T> const*>(sum_result.get())->value();
    } else {
      return static_cast<cudf::numeric_scalar<T> const*>(sum_result.get())->value();
    }
  }

  std::pair<std::unique_ptr<cudf::scalar>, std::unique_ptr<cudf::scalar>> extract_sum_overflow(
    std::unique_ptr<cudf::scalar> const& result)
  {
    EXPECT_TRUE(result->is_valid());
    EXPECT_EQ(result->type().id(), cudf::type_id::STRUCT);

    auto struct_scalar_ptr = static_cast<cudf::struct_scalar const*>(result.get());
    auto table_view        = struct_scalar_ptr->view();

    EXPECT_EQ(table_view.num_columns(), 2);
    EXPECT_EQ(table_view.column(0).size(), 1);
    EXPECT_EQ(table_view.column(1).size(), 1);

    return std::make_pair(cudf::get_element(table_view.column(0), 0),
                          cudf::get_element(table_view.column(1), 0));
  }
};
using ReduceOverflowTypes = ::testing::Types<int8_t,
                                             int16_t,
                                             int32_t,
                                             int64_t,
                                             numeric::decimal32,
                                             numeric::decimal64,
                                             numeric::decimal128>;
TYPED_TEST_SUITE(ReduceOverflowTest, ReduceOverflowTypes);

TYPED_TEST(ReduceOverflowTest, SumWithoutOverflow)
{
  using Rep = typename TestFixture::Rep;
  auto col  = this->make_col({Rep{1}, Rep{2}, Rep{3}, Rep{4}, Rep{5}});

  auto result = cudf::reduce(col,
                             *cudf::make_sum_overflow_aggregation<reduce_aggregation>(),
                             cudf::data_type{cudf::type_id::STRUCT});

  auto [sum_result, overflow_flag] = this->extract_sum_overflow(result);

  EXPECT_TRUE(sum_result->is_valid());
  EXPECT_EQ(sum_result->type().id(), cudf::type_to_id<TypeParam>());
  EXPECT_EQ(this->get_sum_value(sum_result), Rep{15});
  EXPECT_FALSE(static_cast<cudf::numeric_scalar<bool> const*>(overflow_flag.get())->value());
}

TYPED_TEST(ReduceOverflowTest, PositiveOverflow)
{
  using Rep = typename TestFixture::Rep;
  auto col  = this->make_col({std::numeric_limits<Rep>::max(), Rep{1}});

  auto result = cudf::reduce(col,
                             *cudf::make_sum_overflow_aggregation<reduce_aggregation>(),
                             cudf::data_type{cudf::type_id::STRUCT});

  auto [sum_result, overflow_flag] = this->extract_sum_overflow(result);
  EXPECT_TRUE(static_cast<cudf::numeric_scalar<bool> const*>(overflow_flag.get())->value());
}

TYPED_TEST(ReduceOverflowTest, NegativeOverflow)
{
  using Rep = typename TestFixture::Rep;
  auto col  = this->make_col({std::numeric_limits<Rep>::min(), Rep{-1}});

  auto result = cudf::reduce(col,
                             *cudf::make_sum_overflow_aggregation<reduce_aggregation>(),
                             cudf::data_type{cudf::type_id::STRUCT});

  auto [sum_result, overflow_flag] = this->extract_sum_overflow(result);
  EXPECT_TRUE(static_cast<cudf::numeric_scalar<bool> const*>(overflow_flag.get())->value());
}

TYPED_TEST(ReduceOverflowTest, AccumulatingOverflow)
{
  using Rep      = typename TestFixture::Rep;
  auto const big = static_cast<Rep>(std::numeric_limits<Rep>::max() / Rep{3});
  auto col       = this->make_col({big, big, big, big});  // 4 * (max/3) > max

  auto result = cudf::reduce(col,
                             *cudf::make_sum_overflow_aggregation<reduce_aggregation>(),
                             cudf::data_type{cudf::type_id::STRUCT});

  auto [sum_result, overflow_flag] = this->extract_sum_overflow(result);
  EXPECT_TRUE(static_cast<cudf::numeric_scalar<bool> const*>(overflow_flag.get())->value());
}

TYPED_TEST(ReduceOverflowTest, EmptyColumn)
{
  auto empty_col = this->make_col({});

  auto result = cudf::reduce(empty_col,
                             *cudf::make_sum_overflow_aggregation<reduce_aggregation>(),
                             cudf::data_type{cudf::type_id::STRUCT});

  auto [sum_result, overflow_flag] = this->extract_sum_overflow(result);
  EXPECT_FALSE(sum_result->is_valid());
  EXPECT_EQ(sum_result->type().id(), cudf::type_to_id<TypeParam>());
  EXPECT_FALSE(static_cast<cudf::numeric_scalar<bool> const*>(overflow_flag.get())->value());
}

TYPED_TEST(ReduceOverflowTest, AllNullColumn)
{
  using Rep     = typename TestFixture::Rep;
  auto null_col = this->make_null_col({Rep{1}, Rep{2}, Rep{3}}, {false, false, false});

  auto result = cudf::reduce(null_col,
                             *cudf::make_sum_overflow_aggregation<reduce_aggregation>(),
                             cudf::data_type{cudf::type_id::STRUCT});

  auto [sum_result, overflow_flag] = this->extract_sum_overflow(result);
  EXPECT_FALSE(sum_result->is_valid());
  EXPECT_FALSE(static_cast<cudf::numeric_scalar<bool> const*>(overflow_flag.get())->value());
}

TYPED_TEST(ReduceOverflowTest, WithInitialValue)
{
  using Rep        = typename TestFixture::Rep;
  auto col         = this->make_col({Rep{1}, Rep{2}, Rep{3}});
  auto init_scalar = this->make_init_scalar(Rep{10});

  auto result = cudf::reduce(col,
                             *cudf::make_sum_overflow_aggregation<reduce_aggregation>(),
                             cudf::data_type{cudf::type_id::STRUCT},
                             *init_scalar);

  auto [sum_result, overflow_flag] = this->extract_sum_overflow(result);
  EXPECT_TRUE(sum_result->is_valid());
  EXPECT_EQ(this->get_sum_value(sum_result), Rep{16});  // 10 + 1 + 2 + 3
  EXPECT_FALSE(static_cast<cudf::numeric_scalar<bool> const*>(overflow_flag.get())->value());
}

TYPED_TEST(ReduceOverflowTest, InitialValuePositiveOverflow)
{
  using Rep = typename TestFixture::Rep;
  auto col  = this->make_col({Rep{1}, Rep{2}, Rep{3}});
  // (max - 3) + 1 + 2 + 3 = max + 3 overflows
  auto init_scalar =
    this->make_init_scalar(static_cast<Rep>(std::numeric_limits<Rep>::max() - Rep{3}));

  auto result = cudf::reduce(col,
                             *cudf::make_sum_overflow_aggregation<reduce_aggregation>(),
                             cudf::data_type{cudf::type_id::STRUCT},
                             *init_scalar);

  auto [sum_result, overflow_flag] = this->extract_sum_overflow(result);
  EXPECT_TRUE(static_cast<cudf::numeric_scalar<bool> const*>(overflow_flag.get())->value());
}

TYPED_TEST(ReduceOverflowTest, InitialValueNegativeOverflow)
{
  using Rep = typename TestFixture::Rep;
  auto col  = this->make_col({Rep{-1}, Rep{-2}, Rep{-3}});
  // (min + 3) + (-1) + (-2) + (-3) = min - 3 overflows
  auto init_scalar =
    this->make_init_scalar(static_cast<Rep>(std::numeric_limits<Rep>::min() + Rep{3}));

  auto result = cudf::reduce(col,
                             *cudf::make_sum_overflow_aggregation<reduce_aggregation>(),
                             cudf::data_type{cudf::type_id::STRUCT},
                             *init_scalar);

  auto [sum_result, overflow_flag] = this->extract_sum_overflow(result);
  EXPECT_TRUE(static_cast<cudf::numeric_scalar<bool> const*>(overflow_flag.get())->value());
}

TYPED_TEST(ReduceOverflowTest, SlicedColumn)
{
  using Rep = typename TestFixture::Rep;
  auto full =
    this->make_col({Rep{50}, Rep{60}, Rep{1}, Rep{2}, Rep{3}, Rep{4}, Rep{5}, Rep{70}, Rep{80}});
  auto sliced = cudf::slice(full, {2, 7});
  ASSERT_EQ(sliced.size(), 1);

  auto result = cudf::reduce(sliced.front(),
                             *cudf::make_sum_overflow_aggregation<reduce_aggregation>(),
                             cudf::data_type{cudf::type_id::STRUCT});

  auto [sum_result, overflow_flag] = this->extract_sum_overflow(result);
  EXPECT_TRUE(sum_result->is_valid());
  EXPECT_EQ(this->get_sum_value(sum_result), Rep{15});
  EXPECT_FALSE(static_cast<cudf::numeric_scalar<bool> const*>(overflow_flag.get())->value());
}

TYPED_TEST(ReduceOverflowTest, SlicedColumnWithNulls)
{
  using Rep = typename TestFixture::Rep;
  auto full = this->make_null_col(
    {Rep{50}, Rep{60}, Rep{1}, Rep{2}, Rep{3}, Rep{4}, Rep{5}, Rep{70}, Rep{80}},
    {true, true, true, false, true, false, true, true, true});
  auto sliced = cudf::slice(full, {2, 7});
  ASSERT_EQ(sliced.size(), 1);

  auto result = cudf::reduce(sliced.front(),
                             *cudf::make_sum_overflow_aggregation<reduce_aggregation>(),
                             cudf::data_type{cudf::type_id::STRUCT});

  auto [sum_result, overflow_flag] = this->extract_sum_overflow(result);
  EXPECT_TRUE(sum_result->is_valid());
  EXPECT_EQ(this->get_sum_value(sum_result), Rep{9});  // 1 + 3 + 5
  EXPECT_FALSE(static_cast<cudf::numeric_scalar<bool> const*>(overflow_flag.get())->value());
}

TYPED_TEST(ReduceOverflowTest, MultiBlockInputNoOverflow)
{
  using Rep = typename TestFixture::Rep;
  // Alternating +1/-1 exercises both pairwise-overflow branches on every combine; the
  // closed-form sum (-1 for odd N, 0 for even N) fits in every supported DeviceType.
  constexpr cudf::size_type N = 100'001;
  std::vector<Rep> data;
  data.reserve(N);
  for (cudf::size_type i = 0; i < N; ++i) {
    data.push_back(static_cast<Rep>(i % 2 == 0 ? 1 : -1));
  }
  auto col = this->make_col_from_vec(data);

  auto result = cudf::reduce(col,
                             *cudf::make_sum_overflow_aggregation<reduce_aggregation>(),
                             cudf::data_type{cudf::type_id::STRUCT});

  auto [sum_result, overflow_flag] = this->extract_sum_overflow(result);
  EXPECT_TRUE(sum_result->is_valid());
  EXPECT_EQ(this->get_sum_value(sum_result), Rep{1});
  EXPECT_FALSE(static_cast<cudf::numeric_scalar<bool> const*>(overflow_flag.get())->value());
}

TYPED_TEST(ReduceOverflowTest, InvalidInit)
{
  using Rep = typename TestFixture::Rep;
  auto col  = this->make_col({Rep{1}, Rep{2}, Rep{3}});

  auto init_scalar = this->make_init_scalar(Rep{0});
  init_scalar->set_valid_async(false, cudf::get_default_stream());

  auto result = cudf::reduce(col,
                             *cudf::make_sum_overflow_aggregation<reduce_aggregation>(),
                             cudf::data_type{cudf::type_id::STRUCT},
                             *init_scalar);

  auto [sum_result, overflow_flag] = this->extract_sum_overflow(result);
  EXPECT_FALSE(sum_result->is_valid());
  EXPECT_FALSE(static_cast<cudf::numeric_scalar<bool> const*>(overflow_flag.get())->value());
}

// Non-typed fixture used by the error-handling test.
struct ReduceOverflowErrorTest : public cudf::test::BaseFixture {};

TEST_F(ReduceOverflowErrorTest, UnsupportedTypes)
{
  // Unsigned, floating-point, and string columns must be rejected.
  cudf::test::fixed_width_column_wrapper<uint32_t> uint_col{1u, 2u, 3u};
  EXPECT_THROW(cudf::reduce(uint_col,
                            *cudf::make_sum_overflow_aggregation<reduce_aggregation>(),
                            cudf::data_type{cudf::type_id::STRUCT}),
               std::invalid_argument);

  cudf::test::fixed_width_column_wrapper<float> float_col{1.0f, 2.0f, 3.0f};
  EXPECT_THROW(cudf::reduce(float_col,
                            *cudf::make_sum_overflow_aggregation<reduce_aggregation>(),
                            cudf::data_type{cudf::type_id::STRUCT}),
               std::invalid_argument);

  cudf::test::strings_column_wrapper string_col{"a", "b", "c"};
  EXPECT_THROW(cudf::reduce(string_col,
                            *cudf::make_sum_overflow_aggregation<reduce_aggregation>(),
                            cudf::data_type{cudf::type_id::STRUCT}),
               std::invalid_argument);
}

struct ReductionIsValidTest : public cudf::test::BaseFixture {};

TEST_F(ReductionIsValidTest, IsValidAggregation)
{
  auto const int64_type   = cudf::data_type{cudf::type_id::INT64};
  auto const float_type   = cudf::data_type{cudf::type_id::FLOAT32};
  auto const string_type  = cudf::data_type{cudf::type_id::STRING};
  auto const struct_type  = cudf::data_type{cudf::type_id::STRUCT};
  auto const decimal_type = cudf::data_type{cudf::type_id::DECIMAL32, 2};

  EXPECT_TRUE(cudf::reduction::is_valid_aggregation(int64_type, cudf::aggregation::ARGMAX));
  EXPECT_TRUE(cudf::reduction::is_valid_aggregation(int64_type, cudf::aggregation::ARGMIN));
  EXPECT_TRUE(cudf::reduction::is_valid_aggregation(int64_type, cudf::aggregation::SUM));
  EXPECT_TRUE(cudf::reduction::is_valid_aggregation(int64_type, cudf::aggregation::SUM_OVERFLOW));
  EXPECT_TRUE(cudf::reduction::is_valid_aggregation(int64_type, cudf::aggregation::PRODUCT));
  EXPECT_TRUE(cudf::reduction::is_valid_aggregation(int64_type, cudf::aggregation::MIN));
  EXPECT_TRUE(cudf::reduction::is_valid_aggregation(int64_type, cudf::aggregation::MAX));
  EXPECT_TRUE(cudf::reduction::is_valid_aggregation(int64_type, cudf::aggregation::ANY));
  EXPECT_TRUE(cudf::reduction::is_valid_aggregation(int64_type, cudf::aggregation::ALL));
  EXPECT_TRUE(cudf::reduction::is_valid_aggregation(int64_type, cudf::aggregation::SUM_OF_SQUARES));
  EXPECT_TRUE(cudf::reduction::is_valid_aggregation(int64_type, cudf::aggregation::MEAN));
  EXPECT_TRUE(cudf::reduction::is_valid_aggregation(int64_type, cudf::aggregation::VARIANCE));
  EXPECT_TRUE(cudf::reduction::is_valid_aggregation(int64_type, cudf::aggregation::STD));
  EXPECT_TRUE(cudf::reduction::is_valid_aggregation(int64_type, cudf::aggregation::MEDIAN));
  EXPECT_TRUE(cudf::reduction::is_valid_aggregation(int64_type, cudf::aggregation::QUANTILE));
  EXPECT_TRUE(cudf::reduction::is_valid_aggregation(int64_type, cudf::aggregation::NUNIQUE));
  EXPECT_TRUE(cudf::reduction::is_valid_aggregation(int64_type, cudf::aggregation::NTH_ELEMENT));
  EXPECT_TRUE(cudf::reduction::is_valid_aggregation(int64_type, cudf::aggregation::COUNT_ALL));
  EXPECT_TRUE(cudf::reduction::is_valid_aggregation(int64_type, cudf::aggregation::COUNT_VALID));
  EXPECT_TRUE(cudf::reduction::is_valid_aggregation(int64_type, cudf::aggregation::HISTOGRAM));
  EXPECT_TRUE(
    cudf::reduction::is_valid_aggregation(int64_type, cudf::aggregation::MERGE_HISTOGRAM));
  EXPECT_TRUE(cudf::reduction::is_valid_aggregation(int64_type, cudf::aggregation::BITWISE_AGG));
  EXPECT_TRUE(cudf::reduction::is_valid_aggregation(int64_type, cudf::aggregation::COLLECT_LIST));
  EXPECT_TRUE(cudf::reduction::is_valid_aggregation(int64_type, cudf::aggregation::COLLECT_SET));
  EXPECT_TRUE(cudf::reduction::is_valid_aggregation(int64_type, cudf::aggregation::MERGE_LISTS));
  EXPECT_TRUE(cudf::reduction::is_valid_aggregation(int64_type, cudf::aggregation::MERGE_SETS));
  EXPECT_TRUE(cudf::reduction::is_valid_aggregation(int64_type, cudf::aggregation::TDIGEST));
  EXPECT_TRUE(cudf::reduction::is_valid_aggregation(int64_type, cudf::aggregation::MERGE_TDIGEST));

  EXPECT_TRUE(cudf::reduction::is_valid_aggregation(decimal_type, cudf::aggregation::SUM));
  EXPECT_TRUE(cudf::reduction::is_valid_aggregation(decimal_type, cudf::aggregation::SUM_OVERFLOW));
  EXPECT_TRUE(
    cudf::reduction::is_valid_aggregation(decimal_type, cudf::aggregation::SUM_OF_SQUARES));
  EXPECT_TRUE(cudf::reduction::is_valid_aggregation(decimal_type, cudf::aggregation::MEDIAN));
  EXPECT_TRUE(cudf::reduction::is_valid_aggregation(decimal_type, cudf::aggregation::QUANTILE));
  EXPECT_TRUE(cudf::reduction::is_valid_aggregation(decimal_type, cudf::aggregation::TDIGEST));
  EXPECT_TRUE(
    cudf::reduction::is_valid_aggregation(decimal_type, cudf::aggregation::MERGE_TDIGEST));

  EXPECT_TRUE(cudf::reduction::is_valid_aggregation(string_type, cudf::aggregation::MIN));
  EXPECT_TRUE(cudf::reduction::is_valid_aggregation(string_type, cudf::aggregation::MAX));

  EXPECT_FALSE(cudf::reduction::is_valid_aggregation(string_type, cudf::aggregation::SUM));
  EXPECT_FALSE(cudf::reduction::is_valid_aggregation(string_type, cudf::aggregation::PRODUCT));
  EXPECT_FALSE(cudf::reduction::is_valid_aggregation(string_type, cudf::aggregation::ANY));
  EXPECT_FALSE(cudf::reduction::is_valid_aggregation(string_type, cudf::aggregation::ALL));

  EXPECT_FALSE(cudf::reduction::is_valid_aggregation(float_type, cudf::aggregation::SUM_OVERFLOW));

  EXPECT_FALSE(cudf::reduction::is_valid_aggregation(int64_type, cudf::aggregation::ROW_NUMBER));
  EXPECT_FALSE(cudf::reduction::is_valid_aggregation(int64_type, cudf::aggregation::RANK));
  EXPECT_FALSE(cudf::reduction::is_valid_aggregation(int64_type, cudf::aggregation::LAG));
  EXPECT_FALSE(cudf::reduction::is_valid_aggregation(int64_type, cudf::aggregation::LEAD));
  EXPECT_FALSE(cudf::reduction::is_valid_aggregation(int64_type, cudf::aggregation::M2));
  EXPECT_FALSE(cudf::reduction::is_valid_aggregation(int64_type, cudf::aggregation::MERGE_M2));
}

CUDF_TEST_PROGRAM_MAIN()
