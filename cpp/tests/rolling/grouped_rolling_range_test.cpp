/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/iterator_utilities.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/aggregation.hpp>
#include <cudf/column/column.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/null_mask.hpp>
#include <cudf/rolling.hpp>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/sorting.hpp>
#include <cudf/table/table_view.hpp>

#include <cuda/iterator>

#include <algorithm>
#include <set>
#include <vector>

template <typename T>
using fwcw = cudf::test::fixed_width_column_wrapper<T>;
template <typename T>
using decimals_column = cudf::test::fixed_point_column_wrapper<T>;
using ints_column     = fwcw<int32_t>;
using bigints_column  = fwcw<int64_t>;
using strings_column  = cudf::test::strings_column_wrapper;
using column_ptr      = std::unique_ptr<cudf::column>;

template <typename T>
struct BaseGroupedRollingRangeOrderByTest : cudf::test::BaseFixture {
  // Stand-in for std::pow(10, n), but for integral return.
  static constexpr std::array<int32_t, 6> pow10{1, 10, 100, 1000, 10000, 100000};

  // Test data.
  column_ptr const grouping_keys = ints_column{0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2}.release();
  column_ptr const agg_values    = ints_column{1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3}.release();
  cudf::size_type const num_rows = grouping_keys->size();

  /**
   * @brief Get grouped rolling results for specified order-by column and range bounds.
   */
  [[nodiscard]] column_ptr get_grouped_range_rolling_result(
    cudf::range_window_bounds const& preceding,
    cudf::range_window_bounds const& following,
    cudf::column_view const& order_by_column,
    cudf::rolling_aggregation const& agg,
    cudf::order const& order = cudf::order::ASCENDING) const
  {
    return cudf::grouped_range_rolling_window(cudf::table_view{{grouping_keys->view()}},
                                              order_by_column,
                                              order,
                                              agg_values->view(),
                                              preceding,
                                              following,
                                              1,  // min_periods
                                              agg);
  }

  [[nodiscard]] column_ptr get_grouped_range_rolling_sum_result(
    cudf::range_window_bounds const& preceding,
    cudf::range_window_bounds const& following,
    cudf::column_view const& order_by_column,
    cudf::order const& order = cudf::order::ASCENDING) const
  {
    return get_grouped_range_rolling_result(
      preceding,
      following,
      order_by_column,
      *cudf::make_sum_aggregation<cudf::rolling_aggregation>(),
      order);
  }
};

template <typename T>
struct GroupedRollingRangeOrderByNumericTest : public BaseGroupedRollingRangeOrderByTest<T> {
  using base = BaseGroupedRollingRangeOrderByTest<T>;

  using base::agg_values;
  using base::get_grouped_range_rolling_sum_result;
  using base::grouping_keys;
  using base::num_rows;

  static auto constexpr inf = std::numeric_limits<T>::infinity();
  static auto constexpr nan = std::numeric_limits<T>::quiet_NaN();

  [[nodiscard]] auto make_range_bounds(T const& value) const
  {
    return cudf::range_window_bounds::get(*cudf::make_fixed_width_scalar(value));
  }

  [[nodiscard]] auto make_unbounded_range_bounds() const
  {
    return cudf::range_window_bounds::unbounded(cudf::data_type{cudf::type_to_id<T>()});
  }

  /// Generate order-by column with values: [0, 100,   200,   300,   ... 1100,   1200,   1300]
  [[nodiscard]] column_ptr generate_order_by_column() const
  {
    auto const begin = cudf::detail::make_counting_transform_iterator(
      cudf::size_type{0}, [&](T const& i) -> T { return i * 100; });

    return fwcw<T>(begin, begin + num_rows).release();
  }

  /// Generate order-by column with values: [-1400, -1300, -1200 ... -300, -200, -100]
  [[nodiscard]] column_ptr generate_negative_order_by_column() const
  {
    auto const begin = cudf::detail::make_counting_transform_iterator(
      cudf::size_type{0}, [&](T const& i) -> T { return (i - num_rows) * 100; });

    return fwcw<T>(begin, begin + num_rows).release();
  }

  /**
   * @brief Run grouped_rolling test with no nulls in the order-by column
   */
  void run_test_no_null_oby() const
  {
    auto const preceding = make_range_bounds(T{200});
    auto const following = make_range_bounds(T{100});
    auto const order_by  = generate_order_by_column();
    auto const results   = get_grouped_range_rolling_sum_result(preceding, following, *order_by);
    auto const expected_results = bigints_column{{2, 3, 4, 4, 4, 3, 4, 6, 8, 6, 6, 9, 12, 9},
                                                 cudf::test::iterators::no_nulls()};
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected_results);
  }

  /**
   * @brief Run grouped_rolling test with no nulls in the order-by column
   */
  void run_test_negative_oby() const
  {
    auto const preceding = make_range_bounds(T{200});
    auto const following = make_range_bounds(T{100});
    auto const order_by  = generate_negative_order_by_column();
    auto const results   = get_grouped_range_rolling_sum_result(preceding, following, *order_by);
    auto const expected_results = bigints_column{{2, 3, 4, 4, 4, 3, 4, 6, 8, 6, 6, 9, 12, 9},
                                                 cudf::test::iterators::no_nulls()};
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected_results);
  }

  /**
   * @brief Run grouped_rolling test with nulls in the order-by column
   * (i.e. 2 nulls at the beginning of each group)
   *
   */
  void run_test_nulls_in_oby() const
  {
    auto const preceding = make_range_bounds(T{200});
    auto const following = make_range_bounds(T{100});

    // Nullify the first two rows of each group in the order_by column.
    auto const nulled_order_by = [&] {
      auto col           = generate_order_by_column();
      auto new_null_mask = create_null_mask(col->size(), cudf::mask_state::ALL_VALID);
      cudf::set_null_mask(static_cast<cudf::bitmask_type*>(new_null_mask.data()),
                          0,
                          2,
                          false);  // Nulls in first group.
      cudf::set_null_mask(static_cast<cudf::bitmask_type*>(new_null_mask.data()),
                          6,
                          8,
                          false);  // Nulls in second group.
      cudf::set_null_mask(static_cast<cudf::bitmask_type*>(new_null_mask.data()),
                          10,
                          12,
                          false);  // Nulls in third group.
      col->set_null_mask(std::move(new_null_mask), 6);
      return col;
    }();

    auto const results =
      get_grouped_range_rolling_sum_result(preceding, following, *nulled_order_by);
    auto const expected_results =
      bigints_column{{2, 2, 2, 3, 4, 3, 4, 4, 4, 4, 6, 6, 6, 6}, cudf::test::iterators::no_nulls()};
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected_results);
  }

  /**
   * @brief Run grouped_rolling test with unbounded preceding and unbounded following.
   */
  void run_test_unbounded_preceding_to_unbounded_following()
  {
    auto const order_by  = generate_order_by_column();
    auto const preceding = make_unbounded_range_bounds();
    auto const following = make_unbounded_range_bounds();
    auto const results   = get_grouped_range_rolling_sum_result(preceding, following, *order_by);

    auto const expected_results = bigints_column{6, 6, 6, 6, 6, 6, 8, 8, 8, 8, 12, 12, 12, 12};
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected_results);
  }

  /**
   * @brief Run grouped_rolling test with unbounded preceding and current row.
   */
  void run_test_unbounded_preceding_to_current_row()
  {
    auto const order_by            = generate_order_by_column();
    auto const unbounded_preceding = make_unbounded_range_bounds();
    auto const current_row         = make_range_bounds(T{0});
    auto const results =
      get_grouped_range_rolling_sum_result(unbounded_preceding, current_row, *order_by);

    auto const expected_results = bigints_column{{1, 2, 3, 4, 5, 6, 2, 4, 6, 8, 3, 6, 9, 12},
                                                 cudf::test::iterators::no_nulls()};
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected_results);
  }

  /**
   * @brief Run grouped_rolling test with current row and unbounded following.
   */
  void run_test_current_row_to_unbounded_following()
  {
    auto const order_by            = generate_order_by_column();
    auto const unbounded_following = make_unbounded_range_bounds();

    auto const current_row = make_range_bounds(T{0});
    auto const results =
      get_grouped_range_rolling_sum_result(current_row, unbounded_following, *order_by);

    auto const expected_results = bigints_column{{6, 5, 4, 3, 2, 1, 8, 6, 4, 2, 12, 9, 6, 3},
                                                 cudf::test::iterators::no_nulls()};
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected_results);
  }

  [[nodiscard]] column_ptr generate_ascending_order_by_NaNs_infinity()
  {
    auto const vec =
      std::vector<T>{-inf, -inf, -50, 0, 7, nan, -inf, 0, inf, nan, 0, inf, nan, nan};
    return fwcw<T>(vec.begin(), vec.end()).release();
  }

  void run_test_bounded_ascending_order_by_NaNs_infinity()
  {
    auto const order_by  = generate_ascending_order_by_NaNs_infinity();
    auto const preceding = make_range_bounds(T{200});
    auto const following = make_range_bounds(T{100});

    auto const results = get_grouped_range_rolling_sum_result(preceding, following, *order_by);

    auto const expected_results =
      bigints_column{{2, 2, 3, 3, 3, 1, 2, 2, 2, 2, 3, 3, 6, 6}, cudf::test::iterators::no_nulls()};
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected_results);
  }

  void run_test_unbounded_ascending_order_by_NaNs_infinity()
  {
    auto const order_by = generate_ascending_order_by_NaNs_infinity();

    {
      // UNBOUNDED PRECEDING to CURRENT ROW.
      auto const preceding = make_unbounded_range_bounds();
      auto const following = make_range_bounds(T{0});

      auto const results = get_grouped_range_rolling_sum_result(preceding, following, *order_by);

      auto const expected_results = bigints_column{{2, 2, 3, 4, 5, 6, 2, 4, 6, 8, 3, 6, 12, 12},
                                                   cudf::test::iterators::no_nulls()};
      CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected_results);
    }
    {
      // CURRENT ROW to UNBOUNDED FOLLOWING
      auto const preceding = make_range_bounds(T{0});
      auto const following = make_unbounded_range_bounds();

      auto const results = get_grouped_range_rolling_sum_result(preceding, following, *order_by);

      auto const expected_results = bigints_column{{6, 6, 4, 3, 2, 1, 8, 6, 4, 2, 12, 9, 6, 6},
                                                   cudf::test::iterators::no_nulls()};
      CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected_results);
    }
    {
      // UNBOUNDED PRECEDING to UNBOUNDED FOLLOWING
      auto const preceding = make_unbounded_range_bounds();
      auto const following = make_unbounded_range_bounds();

      auto const results = get_grouped_range_rolling_sum_result(preceding, following, *order_by);

      auto const expected_results = bigints_column{6, 6, 6, 6, 6, 6, 8, 8, 8, 8, 12, 12, 12, 12};

      CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected_results);
    }
  }

  [[nodiscard]] column_ptr generate_descending_order_by_NaNs_infinity()
  {
    auto const vec =
      std::vector<T>{nan, 7, 0, -50, -inf, -inf, nan, inf, 0, -inf, nan, nan, inf, 0};
    return fwcw<T>(vec.begin(), vec.end()).release();
  }

  void run_test_bounded_descending_order_by_NaNs_infinity()
  {
    auto const order_by  = generate_descending_order_by_NaNs_infinity();
    auto const preceding = make_range_bounds(T{200});
    auto const following = make_range_bounds(T{100});

    auto const results = get_grouped_range_rolling_sum_result(
      preceding, following, *order_by, cudf::order::DESCENDING);

    auto const expected_results =
      bigints_column{{1, 3, 3, 3, 2, 2, 2, 2, 2, 2, 6, 6, 3, 3}, cudf::test::iterators::no_nulls()};
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected_results);
  }

  void run_test_unbounded_descending_order_by_NaNs_infinity()
  {
    auto const order_by = generate_descending_order_by_NaNs_infinity();

    {
      // UNBOUNDED PRECEDING to CURRENT ROW.
      auto const preceding = make_unbounded_range_bounds();
      auto const following = make_range_bounds(T{0});

      auto const results = get_grouped_range_rolling_sum_result(
        preceding, following, *order_by, cudf::order::DESCENDING);

      auto const expected_results = bigints_column{{1, 2, 3, 4, 6, 6, 2, 4, 6, 8, 6, 6, 9, 12},
                                                   cudf::test::iterators::no_nulls()};
      CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected_results);
    }
    {
      // CURRENT ROW to UNBOUNDED FOLLOWING
      auto const preceding = make_range_bounds(T{0});
      auto const following = make_unbounded_range_bounds();

      auto const results = get_grouped_range_rolling_sum_result(
        preceding, following, *order_by, cudf::order::DESCENDING);

      auto const expected_results = bigints_column{{6, 5, 4, 3, 2, 2, 8, 6, 4, 2, 12, 12, 6, 3},
                                                   cudf::test::iterators::no_nulls()};
      CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected_results);
    }
    {
      // UNBOUNDED PRECEDING to UNBOUNDED FOLLOWING
      auto const preceding = make_unbounded_range_bounds();
      auto const following = make_unbounded_range_bounds();

      auto const results = get_grouped_range_rolling_sum_result(
        preceding, following, *order_by, cudf::order::DESCENDING);

      auto const expected_results = bigints_column{6, 6, 6, 6, 6, 6, 8, 8, 8, 8, 12, 12, 12, 12};
      CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected_results);
    }
  }
};

template <typename FloatingPointType>
struct GroupedRollingRangeOrderByFloatingPointTest
  : GroupedRollingRangeOrderByNumericTest<FloatingPointType> {};

TYPED_TEST_SUITE(GroupedRollingRangeOrderByFloatingPointTest, cudf::test::FloatingPointTypes);

TYPED_TEST(GroupedRollingRangeOrderByFloatingPointTest, BoundedRanges)
{
  this->run_test_no_null_oby();
  this->run_test_negative_oby();
  this->run_test_nulls_in_oby();
  this->run_test_bounded_ascending_order_by_NaNs_infinity();
  this->run_test_bounded_descending_order_by_NaNs_infinity();
}

TYPED_TEST(GroupedRollingRangeOrderByFloatingPointTest, UnboundedRanges)
{
  this->run_test_unbounded_preceding_to_unbounded_following();
  this->run_test_unbounded_preceding_to_current_row();
  this->run_test_current_row_to_unbounded_following();
  this->run_test_unbounded_ascending_order_by_NaNs_infinity();
  this->run_test_unbounded_descending_order_by_NaNs_infinity();
}

template <typename DecimalT>
struct GroupedRollingRangeOrderByDecimalTypedTest
  : BaseGroupedRollingRangeOrderByTest<typename DecimalT::rep> {
  using Rep  = typename DecimalT::rep;
  using base = BaseGroupedRollingRangeOrderByTest<Rep>;

  using base::agg_values;
  using base::grouping_keys;
  using base::num_rows;

  [[nodiscard]] auto make_fixed_point_range_bounds(typename DecimalT::rep value,
                                                   numeric::scale_type scale) const
  {
    return cudf::range_window_bounds::get(*cudf::make_fixed_point_scalar<DecimalT>(value, scale));
  }

  [[nodiscard]] auto make_unbounded_fixed_point_range_bounds() const
  {
    return cudf::range_window_bounds::unbounded(cudf::data_type{cudf::type_to_id<DecimalT>()});
  }

  /// For different scales, generate order_by column with
  /// the same effective values:           [0, 100,   200,   300,   ... 1100,   1200,   1300]
  /// For scale == -2, the rep values are: [0, 10000, 20000, 30000, ... 110000, 120000, 130000]
  /// For scale ==  2, the rep values are: [0, 1,     2,     3,     ... 11,     12,     13]
  [[nodiscard]] column_ptr generate_order_by_column(numeric::scale_type scale) const
  {
    auto const begin = thrust::make_transform_iterator(
      cuda::counting_iterator<Rep>{0},
      [&](auto i) -> Rep { return (i * 10000) / base::pow10[scale + 2]; });

    return decimals_column<Rep>{begin, begin + num_rows, numeric::scale_type{scale}}.release();
  }

  /**
   * @brief Scale the range bounds value to new scale, so that effective
   * value remains identical.
   *
   * Keeping the effective range bounds value identical ensures that
   * the expected result from grouped_rolling remains the same.
   */
  [[nodiscard]] Rep rescale_range_value(Rep const& value_at_scale_0,
                                        numeric::scale_type new_scale) const
  {
    // Scale  ->   Rep (for value == 200)
    //  -2    ->       20000
    //  -1    ->       2000
    //   0    ->       200
    //   1    ->       20
    //   2    ->       2
    return (value_at_scale_0 * 100) / base::pow10[new_scale + 2];
  }

  /**
   * @brief Get grouped rolling results for specified order-by column and range scale
   *
   */
  [[nodiscard]] column_ptr get_grouped_range_rolling_result(
    cudf::column_view const& order_by_column, numeric::scale_type const& range_scale) const
  {
    auto const preceding =
      this->make_fixed_point_range_bounds(rescale_range_value(Rep{200}, range_scale), range_scale);
    auto const following =
      this->make_fixed_point_range_bounds(rescale_range_value(Rep{100}, range_scale), range_scale);

    return base::get_grouped_range_rolling_sum_result(preceding, following, order_by_column);
  }

  /**
   * @brief Run grouped_rolling test for specified order-by column scale with
   * no nulls in the order-by column
   *
   */
  void run_test_no_null_oby(numeric::scale_type const& order_by_column_scale) const
  {
    auto const order_by = generate_order_by_column(order_by_column_scale);
    // Run tests for range bounds generated for all scales >= oby_column_scale.
    for (int32_t range_scale = order_by_column_scale; range_scale <= 2; ++range_scale) {
      auto const results =
        get_grouped_range_rolling_result(*order_by, numeric::scale_type{range_scale});
      auto const expected_results = bigints_column{{2, 3, 4, 4, 4, 3, 4, 6, 8, 6, 6, 9, 12, 9},
                                                   cudf::test::iterators::no_nulls()};
      CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected_results);
    }
  }

  /**
   * @brief Run grouped_rolling test for specified order-by column scale with
   * nulls in the order-by column (i.e. 2 nulls at the beginning of each group)
   *
   */
  void run_test_nulls_in_oby(numeric::scale_type const& order_by_column_scale) const
  {
    // Nullify the first two rows of each group in the order_by column.
    auto const nulled_order_by = [&] {
      auto col           = generate_order_by_column(order_by_column_scale);
      auto new_null_mask = create_null_mask(col->size(), cudf::mask_state::ALL_VALID);
      cudf::set_null_mask(static_cast<cudf::bitmask_type*>(new_null_mask.data()),
                          0,
                          2,
                          false);  // Nulls in first group.
      cudf::set_null_mask(static_cast<cudf::bitmask_type*>(new_null_mask.data()),
                          6,
                          8,
                          false);  // Nulls in second group.
      cudf::set_null_mask(static_cast<cudf::bitmask_type*>(new_null_mask.data()),
                          10,
                          12,
                          false);  // Nulls in third group.
      col->set_null_mask(std::move(new_null_mask), 6);
      return col;
    }();

    // Run tests for range bounds generated for all scales >= oby_column_scale.
    for (auto range_scale = int32_t{order_by_column_scale}; range_scale <= 2; ++range_scale) {
      auto const results =
        get_grouped_range_rolling_result(*nulled_order_by, numeric::scale_type{range_scale});
      auto const expected_results = bigints_column{{2, 2, 2, 3, 4, 3, 4, 4, 4, 4, 6, 6, 6, 6},
                                                   cudf::test::iterators::no_nulls()};
      CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected_results);
    }
  }

  /**
   * @brief Run grouped_rolling test for specified order-by column scale with
   * unbounded preceding and unbounded following.
   *
   */
  void run_test_unbounded_preceding_to_unbounded_following(numeric::scale_type oby_column_scale)
  {
    auto const order_by  = generate_order_by_column(oby_column_scale);
    auto const preceding = make_unbounded_fixed_point_range_bounds();
    auto const following = make_unbounded_fixed_point_range_bounds();
    auto results =
      cudf::grouped_range_rolling_window(cudf::table_view{{grouping_keys->view()}},
                                         order_by->view(),
                                         cudf::order::ASCENDING,
                                         agg_values->view(),
                                         preceding,
                                         following,
                                         1,  // min_periods
                                         *cudf::make_sum_aggregation<cudf::rolling_aggregation>());

    auto expected_results = bigints_column{6, 6, 6, 6, 6, 6, 8, 8, 8, 8, 12, 12, 12, 12};
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected_results);
  }

  /**
   * @brief Run grouped_rolling test for specified order-by column scale with
   * unbounded preceding and current row.
   *
   */
  void run_test_unbounded_preceding_to_current_row(numeric::scale_type oby_column_scale)
  {
    auto const order_by            = generate_order_by_column(oby_column_scale);
    auto const unbounded_preceding = make_unbounded_fixed_point_range_bounds();

    for (int32_t range_scale = oby_column_scale; range_scale <= 2; ++range_scale) {
      auto const current_row =
        make_fixed_point_range_bounds(rescale_range_value(Rep{0}, numeric::scale_type{range_scale}),
                                      numeric::scale_type{range_scale});
      auto const results = cudf::grouped_range_rolling_window(
        cudf::table_view{{grouping_keys->view()}},
        order_by->view(),
        cudf::order::ASCENDING,
        agg_values->view(),
        unbounded_preceding,
        current_row,
        1,  // min_periods
        *cudf::make_sum_aggregation<cudf::rolling_aggregation>());

      auto expected_results = bigints_column{{1, 2, 3, 4, 5, 6, 2, 4, 6, 8, 3, 6, 9, 12},
                                             cudf::test::iterators::no_nulls()};
      CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected_results);
    }
  }

  /**
   * @brief Run grouped_rolling test for specified order-by column scale with
   * current row and unbounded following.
   *
   */
  void run_test_current_row_to_unbounded_following(numeric::scale_type oby_column_scale)
  {
    auto const order_by            = generate_order_by_column(oby_column_scale);
    auto const unbounded_following = make_unbounded_fixed_point_range_bounds();

    for (int32_t range_scale = oby_column_scale; range_scale <= 2; ++range_scale) {
      auto const current_row =
        make_fixed_point_range_bounds(rescale_range_value(Rep{0}, numeric::scale_type{range_scale}),
                                      numeric::scale_type{range_scale});
      auto const results = cudf::grouped_range_rolling_window(
        cudf::table_view{{grouping_keys->view()}},
        order_by->view(),
        cudf::order::ASCENDING,
        agg_values->view(),
        current_row,
        unbounded_following,
        1,  // min_periods
        *cudf::make_sum_aggregation<cudf::rolling_aggregation>());

      auto expected_results = bigints_column{{6, 5, 4, 3, 2, 1, 8, 6, 4, 2, 12, 9, 6, 3},
                                             cudf::test::iterators::no_nulls()};
      CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected_results);
    }
  }
};

TYPED_TEST_SUITE(GroupedRollingRangeOrderByDecimalTypedTest, cudf::test::FixedPointTypes);

TYPED_TEST(GroupedRollingRangeOrderByDecimalTypedTest, BoundedRanges)
{
  for (auto const order_by_column_scale : {-2, -1, 0, 1, 2}) {
    auto const oby_scale = numeric::scale_type{order_by_column_scale};
    this->run_test_no_null_oby(oby_scale);
    this->run_test_nulls_in_oby(oby_scale);
  }
}

TYPED_TEST(GroupedRollingRangeOrderByDecimalTypedTest, UnboundedRanges)
{
  for (auto const order_by_scale : {-2, -1, 0, 1, 2}) {
    auto const order_by_column_scale = numeric::scale_type{order_by_scale};
    this->run_test_unbounded_preceding_to_unbounded_following(order_by_column_scale);
    this->run_test_unbounded_preceding_to_current_row(order_by_column_scale);
    this->run_test_current_row_to_unbounded_following(order_by_column_scale);
  }
}

struct GroupedRollingRangeOrderByStringTest : public cudf::test::BaseFixture {
  // Test data.
  column_ptr const grouping_keys = ints_column{0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2}.release();
  column_ptr const agg_values    = ints_column{1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3}.release();
  cudf::size_type const num_rows = grouping_keys->size();
  cudf::size_type const min_periods = 1;

  cudf::data_type const string_type{cudf::type_id::STRING};
  cudf::range_window_bounds const unbounded_preceding =
    cudf::range_window_bounds::unbounded(string_type);
  cudf::range_window_bounds const unbounded_following =
    cudf::range_window_bounds::unbounded(string_type);
  cudf::range_window_bounds const current_row = cudf::range_window_bounds::current_row(string_type);

  [[nodiscard]] static auto nullable_ints_column(std::initializer_list<int> const& ints)
  {
    return ints_column{ints, cudf::test::iterators::no_nulls()};
  }

  [[nodiscard]] auto get_count_over_partitioned_window(
    cudf::column_view const& order_by,
    cudf::order const& order,
    cudf::range_window_bounds const& preceding,
    cudf::range_window_bounds const& following) const
  {
    return cudf::grouped_range_rolling_window(
      cudf::table_view{{*grouping_keys}},
      order_by,
      order,
      *agg_values,
      preceding,
      following,
      min_periods,
      *cudf::make_count_aggregation<cudf::rolling_aggregation>());
  }

  [[nodiscard]] auto get_count_over_unpartitioned_window(
    cudf::column_view const& order_by,
    cudf::order const& order,
    cudf::range_window_bounds const& preceding,
    cudf::range_window_bounds const& following) const
  {
    return cudf::grouped_range_rolling_window(
      cudf::table_view{std::vector<cudf::column_view>{}},
      order_by,
      order,
      *agg_values,
      preceding,
      following,
      min_periods,
      *cudf::make_count_aggregation<cudf::rolling_aggregation>());
  }
};

TEST(GroupedRollingRangeMultiOrderByTest, CurrentRowPeerFrameNullOrderCombinations)
{
  auto const grouping_keys = ints_column{0, 0, 0, 0, 0, 0, 1, 1, 1, 1}.release();
  auto const orderby0 =
    ints_column{{0, 0, 1, 1, 0, 2, 1, 1, 0, 2}, cudf::test::iterators::nulls_at({0, 1, 4, 8})}
      .release();
  auto const orderby1 =
    ints_column{{0, 0, 0, 1, 1, 2, 1, 1, 2, 0}, cudf::test::iterators::nulls_at({0, 1, 2, 9})}
      .release();
  auto const values = bigints_column{1, 2, 3, 4, 5, 6, 10, 20, 30, 40}.release();
  std::multiset<int64_t> const expected_sums{3, 3, 3, 4, 5, 6, 30, 30, 30, 40};
  std::multiset<int64_t> const expected_mins{1, 1, 3, 4, 5, 6, 10, 10, 30, 40};

  // grouping_keys has no nulls and is not part of the order-by spec under test, so its sort order
  // is fixed; only the two order-by columns vary across the loop below.
  auto constexpr group_key_order      = cudf::order::ASCENDING;
  auto constexpr group_key_null_order = cudf::null_order::BEFORE;

  for (auto const order0 : {cudf::order::ASCENDING, cudf::order::DESCENDING}) {
    for (auto const null_order0 : {cudf::null_order::BEFORE, cudf::null_order::AFTER}) {
      for (auto const order1 : {cudf::order::ASCENDING, cudf::order::DESCENDING}) {
        for (auto const null_order1 : {cudf::null_order::BEFORE, cudf::null_order::AFTER}) {
          std::vector<cudf::order> const sort_orders{group_key_order, order0, order1};
          std::vector<cudf::null_order> const sort_null_orders{
            group_key_null_order, null_order0, null_order1};
          auto const sorted = cudf::sort_by_key(
            cudf::table_view{
              {grouping_keys->view(), orderby0->view(), orderby1->view(), values->view()}},
            cudf::table_view{{grouping_keys->view(), orderby0->view(), orderby1->view()}},
            sort_orders,
            sort_null_orders);
          auto const sorted_cols = sorted->view();

          std::vector<cudf::order> const orders{order0, order1};
          std::vector<cudf::null_order> const null_orders{null_order0, null_order1};
          std::vector<cudf::rolling_request> requests;
          requests.push_back(
            {sorted_cols.column(3), 1, cudf::make_sum_aggregation<cudf::rolling_aggregation>()});
          requests.push_back(
            {sorted_cols.column(3), 1, cudf::make_min_aggregation<cudf::rolling_aggregation>()});

          auto const result = cudf::grouped_range_rolling_window(
            cudf::table_view{{sorted_cols.column(0)}},
            cudf::table_view{{sorted_cols.column(1), sorted_cols.column(2)}},
            cudf::host_span<cudf::order const>{orders},
            cudf::host_span<cudf::null_order const>{null_orders},
            cudf::current_row{},
            cudf::current_row{},
            cudf::host_span<cudf::rolling_request const>{requests});
          auto const result_cols = result->view();

          auto const [host_sums, _s_valid] = cudf::test::to_host<int64_t>(result_cols.column(0));
          auto const [host_mins, _m_valid] = cudf::test::to_host<int64_t>(result_cols.column(1));
          std::multiset<int64_t> const actual_sums(host_sums.begin(), host_sums.end());
          std::multiset<int64_t> const actual_mins(host_mins.begin(), host_mins.end());
          auto const context = ::testing::Message()
                               << "orders=(" << static_cast<int>(order0) << ", "
                               << static_cast<int>(order1) << "), null_orders=("
                               << static_cast<int>(null_order0) << ", "
                               << static_cast<int>(null_order1) << ")";
          EXPECT_EQ(actual_sums, expected_sums) << "SUM mismatch for " << context;
          EXPECT_EQ(actual_mins, expected_mins) << "MIN mismatch for " << context;
        }
      }
    }
  }
}

TEST(GroupedRollingRangeMultiOrderByTest, CurrentRowPeerFrameWithStringOrderBy)
{
  auto const grouping_keys = ints_column{0, 0, 0, 0, 0, 1, 1, 1}.release();
  auto const orderby0      = ints_column{1, 1, 1, 2, 2, 1, 1, 2}.release();
  auto const orderby1      = strings_column{"α", "α", "世", "α", "α", "α", "世", "α"}.release();
  auto const values        = bigints_column{10, 20, 30, 40, 50, 100, 200, 300}.release();

  std::vector<cudf::order> orders{cudf::order::ASCENDING, cudf::order::ASCENDING};
  std::vector<cudf::null_order> null_orders{cudf::null_order::BEFORE, cudf::null_order::BEFORE};
  std::vector<cudf::rolling_request> requests;
  requests.push_back({values->view(), 1, cudf::make_sum_aggregation<cudf::rolling_aggregation>()});

  auto const result =
    cudf::grouped_range_rolling_window(cudf::table_view{{grouping_keys->view()}},
                                       cudf::table_view{{orderby0->view(), orderby1->view()}},
                                       cudf::host_span<cudf::order const>{orders},
                                       cudf::host_span<cudf::null_order const>{null_orders},
                                       cudf::current_row{},
                                       cudf::current_row{},
                                       cudf::host_span<cudf::rolling_request const>{requests});
  auto columns = result->release();

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    *columns[0],
    bigints_column{{30, 30, 30, 90, 90, 100, 200, 300}, cudf::test::iterators::no_nulls()});
}

TEST(GroupedRollingRangeMultiOrderByTest, CurrentRowPeerFrameHonorsMinPeriods)
{
  auto const grouping_keys = ints_column{0, 0, 0, 0, 0, 1, 1, 1}.release();
  auto const orderby0      = ints_column{1, 1, 1, 2, 2, 1, 1, 2}.release();
  auto const orderby1      = ints_column{1, 1, 2, 1, 1, 1, 2, 1}.release();
  auto const values =
    bigints_column{{10, 0, 30, 40, 50, 0, 200, 300}, cudf::test::iterators::nulls_at({1, 5})}
      .release();

  std::vector<cudf::order> orders{cudf::order::ASCENDING, cudf::order::ASCENDING};
  std::vector<cudf::null_order> null_orders{cudf::null_order::BEFORE, cudf::null_order::BEFORE};
  std::vector<cudf::rolling_request> requests;
  requests.push_back({values->view(), 2, cudf::make_sum_aggregation<cudf::rolling_aggregation>()});

  auto const result =
    cudf::grouped_range_rolling_window(cudf::table_view{{grouping_keys->view()}},
                                       cudf::table_view{{orderby0->view(), orderby1->view()}},
                                       cudf::host_span<cudf::order const>{orders},
                                       cudf::host_span<cudf::null_order const>{null_orders},
                                       cudf::current_row{},
                                       cudf::current_row{},
                                       cudf::host_span<cudf::rolling_request const>{requests});
  auto columns = result->release();

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    *columns[0],
    bigints_column{{0, 0, 0, 90, 90, 0, 0, 0},
                   cudf::test::iterators::nulls_at({0, 1, 2, 5, 6, 7})});
}

TEST(GroupedRollingRangeMultiOrderByTest, UnpartitionedUnboundedPrecedingCurrentRowCount)
{
  auto const orderby0 = ints_column{1, 1, 1, 2, 2}.release();
  auto const orderby1 = ints_column{1, 1, 2, 1, 1}.release();
  auto const values   = ints_column{1, 2, 3, 4, 5}.release();

  std::vector<cudf::order> orders{cudf::order::ASCENDING, cudf::order::ASCENDING};
  std::vector<cudf::null_order> null_orders{cudf::null_order::BEFORE, cudf::null_order::BEFORE};
  std::vector<cudf::rolling_request> requests;
  requests.push_back(
    {values->view(), 1, cudf::make_count_aggregation<cudf::rolling_aggregation>()});

  auto const result =
    cudf::grouped_range_rolling_window(cudf::table_view{std::vector<cudf::column_view>{}},
                                       cudf::table_view{{orderby0->view(), orderby1->view()}},
                                       cudf::host_span<cudf::order const>{orders},
                                       cudf::host_span<cudf::null_order const>{null_orders},
                                       cudf::unbounded{},
                                       cudf::current_row{},
                                       cudf::host_span<cudf::rolling_request const>{requests});
  auto columns = result->release();

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*columns[0],
                                 ints_column{{2, 2, 3, 5, 5}, cudf::test::iterators::no_nulls()});
}

TEST_F(GroupedRollingRangeOrderByStringTest, Ascending_Partitioned_NoNulls)
{
  // clang-format off
  auto const orderby =
      strings_column{
          "A", "A", "A", "B", "B", "B", // Group 0.
          "C", "C", "C", "C",           // Group 1.
          "D", "D", "E", "E"            // Group 2.
      }.release();
  // clang-format on

  // Partitioned cases.
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    *get_count_over_partitioned_window(
      *orderby, cudf::order::ASCENDING, unbounded_preceding, current_row),
    nullable_ints_column({3, 3, 3, 6, 6, 6, 4, 4, 4, 4, 2, 2, 4, 4}));

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    *get_count_over_partitioned_window(
      *orderby, cudf::order::ASCENDING, current_row, unbounded_following),
    nullable_ints_column({6, 6, 6, 3, 3, 3, 4, 4, 4, 4, 4, 4, 2, 2}));

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    *get_count_over_partitioned_window(
      *orderby, cudf::order::ASCENDING, unbounded_preceding, unbounded_following),
    ints_column{6, 6, 6, 6, 6, 6, 4, 4, 4, 4, 4, 4, 4, 4});

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    *get_count_over_partitioned_window(*orderby, cudf::order::ASCENDING, current_row, current_row),
    nullable_ints_column({3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 2, 2, 2, 2}));
}

TEST_F(GroupedRollingRangeOrderByStringTest, Ascending_NoParts_NoNulls)
{
  // clang-format off
  auto const orderby =
      strings_column{
          "A", "A", "A", "B", "B", "B",
          "C", "C", "C", "C",
          "D", "D", "E", "E"
      }.release();
  // clang-format on

  // Un-partitioned cases.
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    *get_count_over_unpartitioned_window(
      *orderby, cudf::order::ASCENDING, unbounded_preceding, current_row),
    nullable_ints_column({3, 3, 3, 6, 6, 6, 10, 10, 10, 10, 12, 12, 14, 14}));

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    *get_count_over_unpartitioned_window(
      *orderby, cudf::order::ASCENDING, current_row, unbounded_following),
    nullable_ints_column({14, 14, 14, 11, 11, 11, 8, 8, 8, 8, 4, 4, 2, 2}));

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    *get_count_over_unpartitioned_window(
      *orderby, cudf::order::ASCENDING, unbounded_preceding, unbounded_following),
    ints_column{14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14});

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*get_count_over_unpartitioned_window(
                                   *orderby, cudf::order::ASCENDING, current_row, current_row),
                                 nullable_ints_column({3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 2, 2, 2, 2}));
}

TEST_F(GroupedRollingRangeOrderByStringTest, Ascending_Partitioned_WithNulls)
{
  // clang-format off
  auto const orderby =
      strings_column{
          {
              "X", "X", "X", "B", "B", "B", // Group 0.
              "C", "C", "C", "C",           // Group 1.
              "X", "X", "E", "E"            // Group 2.
          },
          cudf::test::iterators::nulls_at({0, 1, 2, 10, 11})
      }.release();
  // clang-format on

  // Partitioned cases.
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    *get_count_over_partitioned_window(
      *orderby, cudf::order::ASCENDING, unbounded_preceding, current_row),
    nullable_ints_column({3, 3, 3, 6, 6, 6, 4, 4, 4, 4, 2, 2, 4, 4}));

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    *get_count_over_partitioned_window(
      *orderby, cudf::order::ASCENDING, current_row, unbounded_following),
    nullable_ints_column({6, 6, 6, 3, 3, 3, 4, 4, 4, 4, 4, 4, 2, 2}));

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    *get_count_over_partitioned_window(
      *orderby, cudf::order::ASCENDING, unbounded_preceding, unbounded_following),
    ints_column{6, 6, 6, 6, 6, 6, 4, 4, 4, 4, 4, 4, 4, 4});

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    *get_count_over_partitioned_window(*orderby, cudf::order::ASCENDING, current_row, current_row),
    nullable_ints_column({3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 2, 2, 2, 2}));
}

TEST_F(GroupedRollingRangeOrderByStringTest, Ascending_NoParts_WithNulls)
{
  // Un-partitioned cases. Null values have to be clustered together.
  // clang-format off
  auto const orderby =
      strings_column{
          {
              "X", "X", "X", "B", "B", "B",
              "C", "C", "C", "C",
              "D", "D", "E", "E"
          },
          cudf::test::iterators::nulls_at({0, 1, 2})
      }.release();
  // clang-format on

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    *get_count_over_unpartitioned_window(
      *orderby, cudf::order::ASCENDING, unbounded_preceding, current_row),
    nullable_ints_column({3, 3, 3, 6, 6, 6, 10, 10, 10, 10, 12, 12, 14, 14}));

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    *get_count_over_unpartitioned_window(
      *orderby, cudf::order::ASCENDING, current_row, unbounded_following),
    nullable_ints_column({14, 14, 14, 11, 11, 11, 8, 8, 8, 8, 4, 4, 2, 2}));

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    *get_count_over_unpartitioned_window(
      *orderby, cudf::order::ASCENDING, unbounded_preceding, unbounded_following),
    ints_column{14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14});

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*get_count_over_unpartitioned_window(
                                   *orderby, cudf::order::ASCENDING, current_row, current_row),
                                 nullable_ints_column({3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 2, 2, 2, 2}));
}

TEST_F(GroupedRollingRangeOrderByStringTest, Descending_Partitioned_NoNulls)
{
  // clang-format off
  auto const orderby =
      strings_column{
          "B", "B", "B", "A", "A", "A", // Group 0.
          "C", "C", "C", "C",           // Group 1.
          "E", "E", "D", "D"            // Group 2.
      }.release();
  // clang-format on

  // Partitioned cases.
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    *get_count_over_partitioned_window(
      *orderby, cudf::order::DESCENDING, unbounded_preceding, current_row),
    nullable_ints_column({3, 3, 3, 6, 6, 6, 4, 4, 4, 4, 2, 2, 4, 4}));

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    *get_count_over_partitioned_window(
      *orderby, cudf::order::DESCENDING, current_row, unbounded_following),
    nullable_ints_column({6, 6, 6, 3, 3, 3, 4, 4, 4, 4, 4, 4, 2, 2}));

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    *get_count_over_partitioned_window(
      *orderby, cudf::order::DESCENDING, unbounded_preceding, unbounded_following),
    ints_column{6, 6, 6, 6, 6, 6, 4, 4, 4, 4, 4, 4, 4, 4});

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    *get_count_over_partitioned_window(*orderby, cudf::order::DESCENDING, current_row, current_row),
    nullable_ints_column({3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 2, 2, 2, 2}));
}

TEST_F(GroupedRollingRangeOrderByStringTest, Descending_NoParts_NoNulls)
{
  // Un-partitioned cases. Order-by column must be entirely in descending order.
  // clang-format off
  auto const orderby =
      strings_column{
          "E", "E", "E", "D", "D", "D",
          "C", "C", "C", "C",
          "B", "B", "A", "A"
      }.release();
  // clang-format on

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    *get_count_over_unpartitioned_window(
      *orderby, cudf::order::DESCENDING, unbounded_preceding, current_row),
    nullable_ints_column({3, 3, 3, 6, 6, 6, 10, 10, 10, 10, 12, 12, 14, 14}));

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    *get_count_over_unpartitioned_window(
      *orderby, cudf::order::DESCENDING, current_row, unbounded_following),
    nullable_ints_column({14, 14, 14, 11, 11, 11, 8, 8, 8, 8, 4, 4, 2, 2}));

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    *get_count_over_unpartitioned_window(
      *orderby, cudf::order::DESCENDING, unbounded_preceding, unbounded_following),
    ints_column{14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14});

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*get_count_over_unpartitioned_window(
                                   *orderby, cudf::order::DESCENDING, current_row, current_row),
                                 nullable_ints_column({3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 2, 2, 2, 2}));
}

TEST_F(GroupedRollingRangeOrderByStringTest, Descending_Partitioned_WithNulls)
{
  // clang-format off
  auto const orderby =
      strings_column{
          {
              "X", "X", "X", "A", "A", "A", // Group 0.
              "C", "C", "C", "C",           // Group 1.
              "X", "X", "D", "D"            // Group 2.
          },
          cudf::test::iterators::nulls_at({0, 1, 2, 10, 11})
      }.release();
  // clang-format on

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    *get_count_over_partitioned_window(
      *orderby, cudf::order::DESCENDING, unbounded_preceding, current_row),
    nullable_ints_column({3, 3, 3, 6, 6, 6, 4, 4, 4, 4, 2, 2, 4, 4}));

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    *get_count_over_partitioned_window(
      *orderby, cudf::order::DESCENDING, current_row, unbounded_following),
    nullable_ints_column({6, 6, 6, 3, 3, 3, 4, 4, 4, 4, 4, 4, 2, 2}));

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    *get_count_over_partitioned_window(
      *orderby, cudf::order::DESCENDING, unbounded_preceding, unbounded_following),
    ints_column{6, 6, 6, 6, 6, 6, 4, 4, 4, 4, 4, 4, 4, 4});

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    *get_count_over_partitioned_window(*orderby, cudf::order::DESCENDING, current_row, current_row),
    nullable_ints_column({3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 2, 2, 2, 2}));
}

TEST_F(GroupedRollingRangeOrderByStringTest, Descending_NoParts_WithNulls)
{
  // Order-by column must be ordered completely in descending.
  // clang-format off
  auto const orderby =
      strings_column{
          {
              "X", "X", "X", "D", "D", "D",
              "C", "C", "C", "C",
              "B", "B", "A", "A"
          },
          cudf::test::iterators::nulls_at({0, 1, 2})
      }.release();
  // clang-format on

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    *get_count_over_unpartitioned_window(
      *orderby, cudf::order::DESCENDING, unbounded_preceding, current_row),
    nullable_ints_column({3, 3, 3, 6, 6, 6, 10, 10, 10, 10, 12, 12, 14, 14}));

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    *get_count_over_unpartitioned_window(
      *orderby, cudf::order::DESCENDING, current_row, unbounded_following),
    nullable_ints_column({14, 14, 14, 11, 11, 11, 8, 8, 8, 8, 4, 4, 2, 2}));

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    *get_count_over_unpartitioned_window(
      *orderby, cudf::order::DESCENDING, unbounded_preceding, unbounded_following),
    ints_column{14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14});

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*get_count_over_unpartitioned_window(
                                   *orderby, cudf::order::DESCENDING, current_row, current_row),
                                 nullable_ints_column({3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 2, 2, 2, 2}));
}
