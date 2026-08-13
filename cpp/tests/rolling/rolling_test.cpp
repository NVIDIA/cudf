/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "rolling_test.hpp"

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/iterator_utilities.hpp>
#include <cudf_test/random.hpp>
#include <cudf_test/testing_main.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/aggregation.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/dictionary/encode.hpp>
#include <cudf/rolling.hpp>
#include <cudf/utilities/bit.hpp>
#include <cudf/utilities/traits.hpp>

#include <cuda/iterator>
#include <thrust/host_vector.h>

#include <src/rolling/detail/rolling.hpp>

#include <algorithm>
#include <limits>
#include <type_traits>
#include <vector>

class RollingStringTest : public cudf::test::BaseFixture {};

TEST_F(RollingStringTest, NoNullStringMinMaxCount)
{
  cudf::test::strings_column_wrapper input(
    {"This", "is", "rolling", "test", "being", "operated", "on", "string", "column"});
  std::vector<cudf::size_type> window{2};
  cudf::test::strings_column_wrapper expected_min(
    {"This", "This", "being", "being", "being", "being", "column", "column", "column"},
    {1, 1, 1, 1, 1, 1, 1, 1, 1});
  cudf::test::strings_column_wrapper expected_max(
    {"rolling", "test", "test", "test", "test", "string", "string", "string", "string"},
    {1, 1, 1, 1, 1, 1, 1, 1, 1});
  cudf::test::fixed_width_column_wrapper<cudf::size_type> expected_count(
    {3, 4, 4, 4, 4, 4, 4, 3, 2}, {1, 1, 1, 1, 1, 1, 1, 1, 1});

  auto got_min = cudf::rolling_window(
    input, window[0], window[0], 1, *cudf::make_min_aggregation<cudf::rolling_aggregation>());
  auto got_max = cudf::rolling_window(
    input, window[0], window[0], 1, *cudf::make_max_aggregation<cudf::rolling_aggregation>());
  auto got_count_valid = cudf::rolling_window(
    input, window[0], window[0], 1, *cudf::make_count_aggregation<cudf::rolling_aggregation>());
  auto got_count_all = cudf::rolling_window(
    input,
    window[0],
    window[0],
    1,
    *cudf::make_count_aggregation<cudf::rolling_aggregation>(cudf::null_policy::INCLUDE));

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_min, got_min->view());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_max, got_max->view());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_count, got_count_valid->view());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_count, got_count_all->view());
}

TEST_F(RollingStringTest, NullStringMinMaxCount)
{
  cudf::test::strings_column_wrapper input(
    {"This", "is", "rolling", "test", "being", "operated", "on", "string", "column"},
    {1, 0, 0, 1, 0, 1, 1, 1, 0});
  std::vector<cudf::size_type> window{2};
  cudf::test::strings_column_wrapper expected_min(
    {"This", "This", "test", "operated", "on", "on", "on", "on", "string"},
    {1, 1, 1, 1, 1, 1, 1, 1, 1});
  cudf::test::strings_column_wrapper expected_max(
    {"This", "test", "test", "test", "test", "string", "string", "string", "string"},
    {1, 1, 1, 1, 1, 1, 1, 1, 1});
  cudf::test::fixed_width_column_wrapper<cudf::size_type> expected_count_val(
    {1, 2, 1, 2, 3, 3, 3, 2, 1}, {1, 1, 1, 1, 1, 1, 1, 1, 1});
  cudf::test::fixed_width_column_wrapper<cudf::size_type> expected_count_all(
    {3, 4, 4, 4, 4, 4, 4, 3, 2}, {1, 1, 1, 1, 1, 1, 1, 1, 1});

  auto got_min = cudf::rolling_window(
    input, window[0], window[0], 1, *cudf::make_min_aggregation<cudf::rolling_aggregation>());
  auto got_max = cudf::rolling_window(
    input, window[0], window[0], 1, *cudf::make_max_aggregation<cudf::rolling_aggregation>());
  auto got_count_valid = cudf::rolling_window(
    input, window[0], window[0], 1, *cudf::make_count_aggregation<cudf::rolling_aggregation>());
  auto got_count_all = cudf::rolling_window(
    input,
    window[0],
    window[0],
    1,
    *cudf::make_count_aggregation<cudf::rolling_aggregation>(cudf::null_policy::INCLUDE));

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_min, got_min->view());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_max, got_max->view());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_count_val, got_count_valid->view());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_count_all, got_count_all->view());
}

TEST_F(RollingStringTest, MinPeriods)
{
  cudf::test::strings_column_wrapper input(
    {"This", "is", "rolling", "test", "being", "operated", "on", "string", "column"},
    {1, 0, 0, 1, 0, 1, 1, 1, 0});
  std::vector<cudf::size_type> window{2};
  cudf::test::strings_column_wrapper expected_min(
    {"This", "This", "This", "operated", "on", "on", "on", "on", "on"},
    {0, 0, 0, 0, 1, 1, 1, 0, 0});
  cudf::test::strings_column_wrapper expected_max(
    {"This", "test", "test", "test", "test", "string", "string", "string", "string"},
    {0, 0, 0, 0, 1, 1, 1, 0, 0});
  cudf::test::fixed_width_column_wrapper<cudf::size_type> expected_count_val(
    {1, 2, 1, 2, 3, 3, 3, 2, 2}, {1, 1, 1, 1, 1, 1, 1, 1, 0});
  cudf::test::fixed_width_column_wrapper<cudf::size_type> expected_count_all(
    {3, 4, 4, 4, 4, 4, 4, 3, 2}, {0, 1, 1, 1, 1, 1, 1, 0, 0});

  auto got_min = cudf::rolling_window(
    input, window[0], window[0], 3, *cudf::make_min_aggregation<cudf::rolling_aggregation>());
  auto got_max = cudf::rolling_window(
    input, window[0], window[0], 3, *cudf::make_max_aggregation<cudf::rolling_aggregation>());
  auto got_count_valid = cudf::rolling_window(
    input, window[0], window[0], 3, *cudf::make_count_aggregation<cudf::rolling_aggregation>());
  auto got_count_all = cudf::rolling_window(
    input,
    window[0],
    window[0],
    4,
    *cudf::make_count_aggregation<cudf::rolling_aggregation>(cudf::null_policy::INCLUDE));

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_min, got_min->view());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_max, got_max->view());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_count_val, got_count_valid->view());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_count_all, got_count_all->view());
}

// =========================================================================================
class RollingStructTest : public cudf::test::BaseFixture {};

TEST_F(RollingStructTest, NoNullStructsMinMaxCount)
{
  using strings_col = cudf::test::strings_column_wrapper;
  using ints_col    = cudf::test::fixed_width_column_wrapper<int32_t>;
  using structs_col = cudf::test::structs_column_wrapper;

  auto const do_test = [](auto const& input) {
    auto const expected_min = [] {
      auto child1 = strings_col{
        "This", "This", "being", "being", "being", "being", "column", "column", "column"};
      auto child2 = ints_col{1, 1, 5, 5, 5, 5, 9, 9, 9};
      return structs_col{{child1, child2}, cudf::test::iterators::no_nulls()};
    }();

    auto const expected_max = [] {
      auto child1 = strings_col{
        "rolling", "test", "test", "test", "test", "string", "string", "string", "string"};
      auto child2 = ints_col{3, 4, 4, 4, 4, 8, 8, 8, 8};
      return structs_col{{child1, child2}, cudf::test::iterators::no_nulls()};
    }();

    auto const expected_count =
      ints_col{{3, 4, 4, 4, 4, 4, 4, 3, 2}, cudf::test::iterators::no_nulls()};
    auto constexpr preceding  = 2;
    auto constexpr following  = 2;
    auto constexpr min_period = 1;

    auto const result_min =
      cudf::rolling_window(input,
                           preceding,
                           following,
                           min_period,
                           *cudf::make_min_aggregation<cudf::rolling_aggregation>());
    auto const result_max =
      cudf::rolling_window(input,
                           preceding,
                           following,
                           min_period,
                           *cudf::make_max_aggregation<cudf::rolling_aggregation>());
    auto const result_count_valid =
      cudf::rolling_window(input,
                           preceding,
                           following,
                           min_period,
                           *cudf::make_count_aggregation<cudf::rolling_aggregation>());
    auto const result_count_all = cudf::rolling_window(
      input,
      preceding,
      following,
      min_period,
      *cudf::make_count_aggregation<cudf::rolling_aggregation>(cudf::null_policy::INCLUDE));

    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_min, result_min->view());
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_max, result_max->view());
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_count, result_count_valid->view());
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_count, result_count_all->view());
  };

  auto const input_no_sliced = [] {
    auto child1 =
      strings_col{"This", "is", "rolling", "test", "being", "operated", "on", "string", "column"};
    auto child2 = ints_col{1, 2, 3, 4, 5, 6, 7, 8, 9};
    return structs_col{{child1, child2}};
  }();

  auto const input_before_sliced = [] {
    auto constexpr dont_care{0};
    auto child1 = strings_col{"1dont_care",
                              "1dont_care",
                              "@dont_care",
                              "This",
                              "is",
                              "rolling",
                              "test",
                              "being",
                              "operated",
                              "on",
                              "string",
                              "column",
                              "1dont_care",
                              "1dont_care",
                              "@dont_care"};
    auto child2 = ints_col{
      dont_care, dont_care, dont_care, 1, 2, 3, 4, 5, 6, 7, 8, 9, dont_care, dont_care, dont_care};
    return structs_col{{child1, child2}};
  }();
  auto const input_sliced = cudf::slice(input_before_sliced, {3, 12})[0];

  do_test(input_no_sliced);
  do_test(input_sliced);
}

TEST_F(RollingStructTest, NullChildrenMinMaxCount)
{
  using strings_col = cudf::test::strings_column_wrapper;
  using ints_col    = cudf::test::fixed_width_column_wrapper<int32_t>;
  using structs_col = cudf::test::structs_column_wrapper;

  auto const input = [] {
    auto child1 = strings_col{
      {"This", "" /*NULL*/, "" /*NULL*/, "test", "" /*NULL*/, "operated", "on", "string", "column"},
      cudf::test::iterators::nulls_at({1, 2, 4})};
    auto child2 = ints_col{1, 2, 3, 4, 5, 6, 7, 8, 9};
    return structs_col{{child1, child2}};
  }();

  auto const expected_min = [] {
    auto child1 = strings_col{{"" /*NULL*/,
                               "" /*NULL*/,
                               "" /*NULL*/,
                               "" /*NULL*/,
                               "" /*NULL*/,
                               "" /*NULL*/,
                               "column",
                               "column",
                               "column"},
                              cudf::test::iterators::nulls_at({0, 1, 2, 3, 4, 5})};
    auto child2 = ints_col{2, 2, 2, 3, 5, 5, 9, 9, 9};
    return structs_col{{child1, child2}, cudf::test::iterators::no_nulls()};
  }();

  auto const expected_max = [] {
    auto child1 =
      strings_col{"This", "test", "test", "test", "test", "string", "string", "string", "string"};
    auto child2 = ints_col{1, 4, 4, 4, 4, 8, 8, 8, 8};
    return structs_col{{child1, child2}, cudf::test::iterators::no_nulls()};
  }();

  auto const expected_count =
    ints_col{{3, 4, 4, 4, 4, 4, 4, 3, 2}, cudf::test::iterators::no_nulls()};
  auto constexpr preceding  = 2;
  auto constexpr following  = 2;
  auto constexpr min_period = 1;

  auto const result_min =
    cudf::rolling_window(input,
                         preceding,
                         following,
                         min_period,
                         *cudf::make_min_aggregation<cudf::rolling_aggregation>());

  auto const result_max =
    cudf::rolling_window(input,
                         preceding,
                         following,
                         min_period,
                         *cudf::make_max_aggregation<cudf::rolling_aggregation>());

  auto const result_count_valid =
    cudf::rolling_window(input,
                         preceding,
                         following,
                         min_period,
                         *cudf::make_count_aggregation<cudf::rolling_aggregation>());
  auto const result_count_all = cudf::rolling_window(
    input,
    preceding,
    following,
    min_period,
    *cudf::make_count_aggregation<cudf::rolling_aggregation>(cudf::null_policy::INCLUDE));

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_min, result_min->view());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_max, result_max->view());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_count, result_count_valid->view());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_count, result_count_all->view());
}

TEST_F(RollingStructTest, NullParentMinMaxCount)
{
  using strings_col = cudf::test::strings_column_wrapper;
  using ints_col    = cudf::test::fixed_width_column_wrapper<int32_t>;
  using structs_col = cudf::test::structs_column_wrapper;

  auto constexpr null{0};
  auto const input = [] {
    auto child1 = strings_col{"This",
                              "" /*NULL*/,
                              "" /*NULL*/,
                              "test",
                              "" /*NULL*/,
                              "operated",
                              "on",
                              "string",
                              "" /*NULL*/};
    auto child2 = ints_col{1, null, null, 4, null, 6, 7, 8, null};
    return structs_col{{child1, child2}, cudf::test::iterators::nulls_at({1, 2, 4, 8})};
  }();

  auto const expected_min = [] {
    auto child1 = strings_col{"This", "This", "test", "operated", "on", "on", "on", "on", "string"};
    auto child2 = ints_col{1, 1, 4, 6, 7, 7, 7, 7, 8};
    return structs_col{{child1, child2}, cudf::test::iterators::no_nulls()};
  }();

  auto const expected_max = [] {
    auto child1 =
      strings_col{"This", "test", "test", "test", "test", "string", "string", "string", "string"};
    auto child2 = ints_col{1, 4, 4, 4, 4, 8, 8, 8, 8};
    return structs_col{{child1, child2}, cudf::test::iterators::no_nulls()};
  }();

  auto const expected_count_valid =
    ints_col{{1, 2, 1, 2, 3, 3, 3, 2, 1}, cudf::test::iterators::no_nulls()};
  auto const expected_count_all =
    ints_col{{3, 4, 4, 4, 4, 4, 4, 3, 2}, cudf::test::iterators::no_nulls()};
  auto constexpr preceding  = 2;
  auto constexpr following  = 2;
  auto constexpr min_period = 1;

  auto const result_min =
    cudf::rolling_window(input,
                         preceding,
                         following,
                         min_period,
                         *cudf::make_min_aggregation<cudf::rolling_aggregation>());

  auto const result_max =
    cudf::rolling_window(input,
                         preceding,
                         following,
                         min_period,
                         *cudf::make_max_aggregation<cudf::rolling_aggregation>());

  auto const result_count_valid =
    cudf::rolling_window(input,
                         preceding,
                         following,
                         min_period,
                         *cudf::make_count_aggregation<cudf::rolling_aggregation>());
  auto const result_count_all = cudf::rolling_window(
    input,
    preceding,
    following,
    min_period,
    *cudf::make_count_aggregation<cudf::rolling_aggregation>(cudf::null_policy::INCLUDE));

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_min, result_min->view());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_max, result_max->view());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_count_valid, result_count_valid->view());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_count_all, result_count_all->view());
}

// =========================================================================================
template <typename T>
class RollingTest : public cudf::test::BaseFixture {
 protected:
  // input as column_wrapper
  void run_test_col(cudf::column_view const& input,
                    std::vector<cudf::size_type> const& preceding_window,
                    std::vector<cudf::size_type> const& following_window,
                    cudf::size_type min_periods,
                    cudf::rolling_aggregation const& op)
  {
    std::unique_ptr<cudf::column> output;

    // wrap windows
    if (preceding_window.size() > 1) {
      cudf::test::fixed_width_column_wrapper<cudf::size_type> preceding_window_wrapper(
        preceding_window.begin(), preceding_window.end());
      cudf::test::fixed_width_column_wrapper<cudf::size_type> following_window_wrapper(
        following_window.begin(), following_window.end());

      EXPECT_NO_THROW(
        output = cudf::rolling_window(
          input, preceding_window_wrapper, following_window_wrapper, min_periods, op));
    } else {
      EXPECT_NO_THROW(output = cudf::rolling_window(
                        input, preceding_window[0], following_window[0], min_periods, op));
    }

    auto reference =
      create_reference_output(op, input, preceding_window, following_window, min_periods);

    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*output, *reference);
  }

  // helper function to test all aggregators
  void run_test_col_agg(cudf::column_view const& input,
                        std::vector<cudf::size_type> const& preceding_window,
                        std::vector<cudf::size_type> const& following_window,
                        cudf::size_type min_periods)
  {
    // test all supported aggregators
    run_test_col(input,
                 preceding_window,
                 following_window,
                 min_periods,
                 *cudf::make_min_aggregation<cudf::rolling_aggregation>());
    run_test_col(input,
                 preceding_window,
                 following_window,
                 min_periods,
                 *cudf::make_count_aggregation<cudf::rolling_aggregation>());
    run_test_col(
      input,
      preceding_window,
      following_window,
      min_periods,
      *cudf::make_count_aggregation<cudf::rolling_aggregation>(cudf::null_policy::INCLUDE));
    run_test_col(input,
                 preceding_window,
                 following_window,
                 min_periods,
                 *cudf::make_max_aggregation<cudf::rolling_aggregation>());

    if (not cudf::is_timestamp(input.type())) {
      run_test_col(input,
                   preceding_window,
                   following_window,
                   min_periods,
                   *cudf::make_sum_aggregation<cudf::rolling_aggregation>());
      run_test_col(input,
                   preceding_window,
                   following_window,
                   min_periods,
                   *cudf::make_mean_aggregation<cudf::rolling_aggregation>());
    }
  }

 private:
  // use SFINAE to only instantiate for supported combinations

  // specialization for COUNT_VALID, COUNT_ALL
  template <bool include_nulls>
  std::unique_ptr<cudf::column> create_count_reference_output(
    cudf::column_view const& input,
    std::vector<cudf::size_type> const& preceding_window_col,
    std::vector<cudf::size_type> const& following_window_col,
    cudf::size_type min_periods)
  {
    cudf::size_type num_rows = input.size();
    std::vector<cudf::size_type> ref_data(num_rows);
    std::vector<bool> ref_valid(num_rows);

    // input data and mask

    std::vector<cudf::bitmask_type> in_valid = cudf::test::bitmask_to_host(input);
    cudf::bitmask_type* valid_mask           = in_valid.data();

    for (cudf::size_type i = 0; i < num_rows; i++) {
      // load sizes
      min_periods = std::max(min_periods, 1);  // at least one observation is required

      // compute bounds
      auto preceding_window       = preceding_window_col[i % preceding_window_col.size()];
      auto following_window       = following_window_col[i % following_window_col.size()];
      cudf::size_type start       = std::min(num_rows, std::max(0, i - preceding_window + 1));
      cudf::size_type end         = std::min(num_rows, std::max(0, i + following_window + 1));
      cudf::size_type start_index = std::min(start, end);
      cudf::size_type end_index   = std::max(start, end);

      // aggregate
      cudf::size_type count = 0;
      for (cudf::size_type j = start_index; j < end_index; j++) {
        if (include_nulls || !input.nullable() || cudf::bit_is_set(valid_mask, j)) count++;
      }

      ref_valid[i] = ((end_index - start_index) >= min_periods);
      if (ref_valid[i]) ref_data[i] = count;
    }

    cudf::test::fixed_width_column_wrapper<cudf::size_type> col(
      ref_data.begin(), ref_data.end(), ref_valid.begin());
    return col.release();
  }

  template <typename agg_op, cudf::aggregation::Kind k, typename OutputType, bool is_mean>
  std::unique_ptr<cudf::column> create_reference_output(
    cudf::column_view const& input,
    std::vector<cudf::size_type> const& preceding_window_col,
    std::vector<cudf::size_type> const& following_window_col,
    cudf::size_type min_periods)
    requires(is_rolling_supported<T, k>())
  {
    cudf::size_type num_rows = input.size();
    thrust::host_vector<OutputType> ref_data(num_rows);
    thrust::host_vector<bool> ref_valid(num_rows);

    // input data and mask
    auto [in_col, in_valid]        = cudf::test::to_host<T>(input);
    cudf::bitmask_type* valid_mask = in_valid.data();

    agg_op op;
    for (cudf::size_type i = 0; i < num_rows; i++) {
      auto val = agg_op::template identity<OutputType>();

      // load sizes
      min_periods = std::max(min_periods, 1);  // at least one observation is required

      // compute bounds
      auto preceding_window       = preceding_window_col[i % preceding_window_col.size()];
      auto following_window       = following_window_col[i % following_window_col.size()];
      cudf::size_type start       = std::min(num_rows, std::max(0, i - preceding_window + 1));
      cudf::size_type end         = std::min(num_rows, std::max(0, i + following_window + 1));
      cudf::size_type start_index = std::min(start, end);
      cudf::size_type end_index   = std::max(start, end);

      // aggregate
      cudf::size_type count = 0;
      for (cudf::size_type j = start_index; j < end_index; j++) {
        if (!input.nullable() || cudf::bit_is_set(valid_mask, j)) {
          val = op(static_cast<OutputType>(in_col[j]), val);
          count++;
        }
      }

      ref_valid[i] = (count >= min_periods);
      if (ref_valid[i]) {
        cudf::detail::rolling_store_output_functor<OutputType, is_mean>{}(ref_data[i], val, count);
      }
    }

    cudf::test::fixed_width_column_wrapper<OutputType> col(
      ref_data.begin(), ref_data.end(), ref_valid.begin());
    return col.release();
  }

  template <typename agg_op, cudf::aggregation::Kind k, typename OutputType, bool is_mean>
  std::unique_ptr<cudf::column> create_reference_output(
    cudf::column_view const& input,
    std::vector<cudf::size_type> const& preceding_window_col,
    std::vector<cudf::size_type> const& following_window_col,
    cudf::size_type min_periods)
    requires(!is_rolling_supported<T, k>())
  {
    CUDF_FAIL("Unsupported combination of type and aggregation");
  }

  std::unique_ptr<cudf::column> create_reference_output(
    cudf::rolling_aggregation const& op,
    cudf::column_view const& input,
    std::vector<cudf::size_type> const& preceding_window,
    std::vector<cudf::size_type> const& following_window,
    cudf::size_type min_periods)
  {
    // unroll aggregation types
    switch (op.kind) {
      case cudf::aggregation::SUM:
        return create_reference_output<cudf::DeviceSum,
                                       cudf::aggregation::SUM,
                                       std::conditional_t<std::is_integral_v<T>, int64_t, T>,
                                       false>(
          input, preceding_window, following_window, min_periods);
      case cudf::aggregation::MIN:
        return create_reference_output<cudf::DeviceMin, cudf::aggregation::MIN, T, false>(
          input, preceding_window, following_window, min_periods);
      case cudf::aggregation::MAX:
        return create_reference_output<cudf::DeviceMax, cudf::aggregation::MAX, T, false>(
          input, preceding_window, following_window, min_periods);
      case cudf::aggregation::COUNT_VALID:
        return create_count_reference_output<false>(
          input, preceding_window, following_window, min_periods);
      case cudf::aggregation::COUNT_ALL:
        return create_count_reference_output<true>(
          input, preceding_window, following_window, min_periods);
      case cudf::aggregation::MEAN:
        return create_reference_output<cudf::DeviceSum,
                                       cudf::aggregation::MEAN,
                                       std::conditional_t<cudf::is_duration<T>(), T, double>,
                                       true>(
          input, preceding_window, following_window, min_periods);
      default: return cudf::test::fixed_width_column_wrapper<T>({}).release();
    }
  }
};

template <typename T>
class RollingVarStdTest : public cudf::test::BaseFixture {};

TYPED_TEST_SUITE(RollingVarStdTest, cudf::test::FixedWidthTypesWithoutChrono);

class RollingtVarStdTestUntyped : public cudf::test::BaseFixture {};

class RollingErrorTest : public cudf::test::BaseFixture {};

class RollingSumEdgeCaseTest : public cudf::test::BaseFixture {};

// negative sizes
TEST_F(RollingErrorTest, NegativeMinPeriods)
{
  std::vector<cudf::size_type> const col_data = {0, 1, 2, 0, 4};
  std::vector<bool> const col_valid           = {1, 1, 1, 0, 1};
  cudf::test::fixed_width_column_wrapper<cudf::size_type> input(
    col_data.begin(), col_data.end(), col_valid.begin());

  EXPECT_THROW(
    cudf::rolling_window(input, 2, 2, -2, *cudf::make_sum_aggregation<cudf::rolling_aggregation>()),
    cudf::logic_error);
}

// window array size mismatch
TEST_F(RollingErrorTest, WindowArraySizeMismatch)
{
  std::vector<cudf::size_type> const col_data = {0, 1, 2, 0, 4};
  std::vector<bool> const col_valid           = {1, 1, 1, 0, 1};
  cudf::test::fixed_width_column_wrapper<cudf::size_type> input(
    col_data.begin(), col_data.end(), col_valid.begin());

  std::vector<cudf::size_type> five({1, 1, 2, 1, 0});
  std::vector<cudf::size_type> four({1, 2, 3, 4});
  cudf::test::fixed_width_column_wrapper<cudf::size_type> five_elements(five.begin(), five.end());
  cudf::test::fixed_width_column_wrapper<cudf::size_type> four_elements(four.begin(), four.end());

  // this runs ok
  EXPECT_NO_THROW(cudf::rolling_window(input,
                                       five_elements,
                                       five_elements,
                                       1,
                                       *cudf::make_sum_aggregation<cudf::rolling_aggregation>()));

  // mismatch for the window array
  EXPECT_THROW(cudf::rolling_window(input,
                                    four_elements,
                                    five_elements,
                                    1,
                                    *cudf::make_sum_aggregation<cudf::rolling_aggregation>()),
               cudf::logic_error);

  // mismatch for the forward window array
  EXPECT_THROW(cudf::rolling_window(input,
                                    five_elements,
                                    four_elements,
                                    1,
                                    *cudf::make_sum_aggregation<cudf::rolling_aggregation>()),
               cudf::logic_error);
}

TEST_F(RollingErrorTest, EmptyInput)
{
  cudf::test::fixed_width_column_wrapper<int32_t> empty_col{};
  std::unique_ptr<cudf::column> output;
  EXPECT_NO_THROW(output = cudf::rolling_window(
                    empty_col, 2, 0, 2, *cudf::make_sum_aggregation<cudf::rolling_aggregation>()));
  EXPECT_EQ(output->size(), 0);

  cudf::test::fixed_width_column_wrapper<int32_t> preceding_window{};
  cudf::test::fixed_width_column_wrapper<int32_t> following_window{};
  EXPECT_NO_THROW(output =
                    cudf::rolling_window(empty_col,
                                         preceding_window,
                                         following_window,
                                         2,
                                         *cudf::make_sum_aggregation<cudf::rolling_aggregation>()));
  EXPECT_EQ(output->size(), 0);

  cudf::test::fixed_width_column_wrapper<int32_t> nonempty_col{{1, 2, 3}};
  EXPECT_NO_THROW(output =
                    cudf::rolling_window(nonempty_col,
                                         preceding_window,
                                         following_window,
                                         2,
                                         *cudf::make_sum_aggregation<cudf::rolling_aggregation>()));
  EXPECT_EQ(output->size(), 0);
}

TEST_F(RollingErrorTest, SizeMismatch)
{
  cudf::test::fixed_width_column_wrapper<int32_t> nonempty_col{{1, 2, 3}};
  std::unique_ptr<cudf::column> output;

  {
    cudf::test::fixed_width_column_wrapper<int32_t> preceding_window{{1, 1}};  // wrong size
    cudf::test::fixed_width_column_wrapper<int32_t> following_window{{1, 1, 1}};
    EXPECT_THROW(
      output = cudf::rolling_window(nonempty_col,
                                    preceding_window,
                                    following_window,
                                    2,
                                    *cudf::make_sum_aggregation<cudf::rolling_aggregation>()),
      cudf::logic_error);
  }
  {
    cudf::test::fixed_width_column_wrapper<int32_t> preceding_window{{1, 1, 1}};
    cudf::test::fixed_width_column_wrapper<int32_t> following_window{{1, 2}};  // wrong size
    EXPECT_THROW(
      output = cudf::rolling_window(nonempty_col,
                                    preceding_window,
                                    following_window,
                                    2,
                                    *cudf::make_sum_aggregation<cudf::rolling_aggregation>()),
      cudf::logic_error);
  }
}

TEST_F(RollingErrorTest, WindowWrongDtype)
{
  cudf::test::fixed_width_column_wrapper<int32_t> nonempty_col{{1, 2, 3}};
  std::unique_ptr<cudf::column> output;

  cudf::test::fixed_width_column_wrapper<float> preceding_window{{1.0f, 1.0f, 1.0f}};
  cudf::test::fixed_width_column_wrapper<float> following_window{{1.0f, 1.0f, 1.0f}};
  EXPECT_THROW(
    output = cudf::rolling_window(nonempty_col,
                                  preceding_window,
                                  following_window,
                                  2,
                                  *cudf::make_sum_aggregation<cudf::rolling_aggregation>()),
    cudf::logic_error);
}

// incorrect type/aggregation combo: sum of timestamps
TEST_F(RollingErrorTest, SumTimestampNotSupported)
{
  constexpr cudf::size_type size{10};
  auto const d_iter  = cuda::counting_iterator<cudf::timestamp_D::rep>{0};
  auto const ns_iter = cuda::counting_iterator<cudf::timestamp_s::rep>{0};
  cudf::test::fixed_width_column_wrapper<cudf::timestamp_D, cudf::timestamp_D::rep> input_D(
    d_iter, d_iter + size);
  cudf::test::fixed_width_column_wrapper<cudf::timestamp_s, cudf::timestamp_s::rep> input_s(
    ns_iter, ns_iter + size);
  cudf::test::fixed_width_column_wrapper<cudf::timestamp_ms, cudf::timestamp_ms::rep> input_ms(
    ns_iter, ns_iter + size);
  cudf::test::fixed_width_column_wrapper<cudf::timestamp_us, cudf::timestamp_us::rep> input_us(
    ns_iter, ns_iter + size);
  cudf::test::fixed_width_column_wrapper<cudf::timestamp_ns, cudf::timestamp_ns::rep> input_ns(
    ns_iter, ns_iter + size);

  EXPECT_THROW(cudf::rolling_window(
                 input_D, 2, 2, 0, *cudf::make_sum_aggregation<cudf::rolling_aggregation>()),
               cudf::logic_error);
  EXPECT_THROW(cudf::rolling_window(
                 input_s, 2, 2, 0, *cudf::make_sum_aggregation<cudf::rolling_aggregation>()),
               cudf::logic_error);
  EXPECT_THROW(cudf::rolling_window(
                 input_ms, 2, 2, 0, *cudf::make_sum_aggregation<cudf::rolling_aggregation>()),
               cudf::logic_error);
  EXPECT_THROW(cudf::rolling_window(
                 input_us, 2, 2, 0, *cudf::make_sum_aggregation<cudf::rolling_aggregation>()),
               cudf::logic_error);
  EXPECT_THROW(cudf::rolling_window(
                 input_ns, 2, 2, 0, *cudf::make_sum_aggregation<cudf::rolling_aggregation>()),
               cudf::logic_error);
}

// incorrect type/aggregation combo: mean of timestamps
TEST_F(RollingErrorTest, MeanTimestampNotSupported)
{
  constexpr cudf::size_type size{10};
  auto const d_iter  = cuda::counting_iterator<cudf::timestamp_D::rep>{0};
  auto const ns_iter = cuda::counting_iterator<cudf::timestamp_s::rep>{0};
  cudf::test::fixed_width_column_wrapper<cudf::timestamp_D, cudf::timestamp_D::rep> input_D(
    d_iter, d_iter + size);
  cudf::test::fixed_width_column_wrapper<cudf::timestamp_s, cudf::timestamp_s::rep> input_s(
    ns_iter, ns_iter + size);
  cudf::test::fixed_width_column_wrapper<cudf::timestamp_ms, cudf::timestamp_ms::rep> input_ms(
    ns_iter, ns_iter + size);
  cudf::test::fixed_width_column_wrapper<cudf::timestamp_us, cudf::timestamp_us::rep> input_us(
    ns_iter, ns_iter + size);
  cudf::test::fixed_width_column_wrapper<cudf::timestamp_ns, cudf::timestamp_ns::rep> input_ns(
    ns_iter, ns_iter + size);

  EXPECT_THROW(cudf::rolling_window(
                 input_D, 2, 2, 0, *cudf::make_mean_aggregation<cudf::rolling_aggregation>()),
               cudf::logic_error);
  EXPECT_THROW(cudf::rolling_window(
                 input_s, 2, 2, 0, *cudf::make_mean_aggregation<cudf::rolling_aggregation>()),
               cudf::logic_error);
  EXPECT_THROW(cudf::rolling_window(
                 input_ms, 2, 2, 0, *cudf::make_mean_aggregation<cudf::rolling_aggregation>()),
               cudf::logic_error);
  EXPECT_THROW(cudf::rolling_window(
                 input_us, 2, 2, 0, *cudf::make_mean_aggregation<cudf::rolling_aggregation>()),
               cudf::logic_error);
  EXPECT_THROW(cudf::rolling_window(
                 input_ns, 2, 2, 0, *cudf::make_mean_aggregation<cudf::rolling_aggregation>()),
               cudf::logic_error);
}

TYPED_TEST_SUITE(RollingTest, cudf::test::FixedWidthTypesWithoutFixedPoint);

// simple example from Pandas docs
TYPED_TEST(RollingTest, SimpleStatic)
{
  // https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.rolling.html
  auto const col_data              = cudf::test::make_type_param_vector<TypeParam>({0, 1, 2, 0, 4});
  std::vector<bool> const col_mask = {1, 1, 1, 0, 1};

  cudf::test::fixed_width_column_wrapper<TypeParam> input(
    col_data.begin(), col_data.end(), col_mask.begin());
  std::vector<cudf::size_type> window{2};

  // static sizes
  this->run_test_col_agg(input, window, window, 1);
}

TEST_F(RollingSumEdgeCaseTest, SumFloatNonFinite)
{
  auto const nan = std::numeric_limits<double>::quiet_NaN();
  auto const inf = std::numeric_limits<double>::infinity();

  auto const input =
    cudf::test::fixed_width_column_wrapper<double>{{nan, 1.0, 2.0, inf, -inf, 4.0, 5.0}};
  auto const expected =
    cudf::test::fixed_width_column_wrapper<double>{{nan, nan, 3.0, inf, nan, -inf, 9.0}};

  auto const result =
    cudf::rolling_window(input, 2, 0, 1, *cudf::make_sum_aggregation<cudf::rolling_aggregation>());

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected, result->view());
}

TEST_F(RollingSumEdgeCaseTest, SumFloatFiniteCancellation)
{
  auto const input    = cudf::test::fixed_width_column_wrapper<double>{{1e20, 1.0, 2.0, 3.0}};
  auto const expected = cudf::test::fixed_width_column_wrapper<double>{{1e20, 1e20, 3.0, 5.0}};

  auto const result =
    cudf::rolling_window(input, 2, 0, 1, *cudf::make_sum_aggregation<cudf::rolling_aggregation>());

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected, result->view());
}

TEST_F(RollingSumEdgeCaseTest, SumInt64PrefixOverflow)
{
  auto constexpr max = std::numeric_limits<int64_t>::max();

  auto const input =
    cudf::test::fixed_width_column_wrapper<int64_t>{{max, int64_t{-1}, int64_t{2}}};
  auto const expected =
    cudf::test::fixed_width_column_wrapper<int64_t>{{max, max - int64_t{1}, int64_t{1}}};

  auto const result =
    cudf::rolling_window(input, 2, 0, 1, *cudf::make_sum_aggregation<cudf::rolling_aggregation>());

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected, result->view());
}

TYPED_TEST(RollingVarStdTest, SimpleStaticVarianceStd)
{
#define XXX 0  // NULL stub

  using ResultType = double;

  double const nan = std::numeric_limits<double>::signaling_NaN();

  cudf::size_type const ddof = 1, min_periods = 0, preceding_window = 2, following_window = 1;

  auto const col_data =
    cudf::test::make_type_param_vector<TypeParam>({XXX, XXX, 9, 5, XXX, XXX, XXX, 0, 8, 5, 8});
  std::vector<bool> const col_mask = {0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1};

  auto const expected_var =
    cudf::is_boolean<TypeParam>()
      ? std::vector<ResultType>{XXX, nan, 0, 0, nan, XXX, nan, 0.5, 0.3333333333333333, 0, 0}
      : std::vector<ResultType>{XXX, nan, 8, 8, nan, XXX, nan, 32, 16.33333333333333, 3, 4.5};
  std::vector<ResultType> expected_std(expected_var.size());
  std::transform(expected_var.begin(), expected_var.end(), expected_std.begin(), [](auto const& x) {
    return std::sqrt(x);
  });

  std::vector<bool> const expected_mask = {0, /* all null window */
                                           1, /* 0 div 0, nan */
                                           1,
                                           1,
                                           1, /* 0 div 0, nan */
                                           0, /* all null window */
                                           1, /* 0 div 0, nan */
                                           1,
                                           1,
                                           1,
                                           1};

  cudf::test::fixed_width_column_wrapper<TypeParam> input(
    col_data.begin(), col_data.end(), col_mask.begin());
  cudf::test::fixed_width_column_wrapper<ResultType> var_expect(
    expected_var.begin(), expected_var.end(), expected_mask.begin());
  cudf::test::fixed_width_column_wrapper<ResultType> std_expect(
    expected_std.begin(), expected_std.end(), expected_mask.begin());

  std::unique_ptr<cudf::column> var_result, std_result;
  // static sizes
  EXPECT_NO_THROW(var_result = cudf::rolling_window(input,
                                                    preceding_window,
                                                    following_window,
                                                    min_periods,
                                                    dynamic_cast<cudf::rolling_aggregation const&>(
                                                      *cudf::make_variance_aggregation(ddof))););
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*var_result, var_expect);

  EXPECT_NO_THROW(std_result = cudf::rolling_window(input,
                                                    preceding_window,
                                                    following_window,
                                                    min_periods,
                                                    dynamic_cast<cudf::rolling_aggregation const&>(
                                                      *cudf::make_std_aggregation(ddof))););
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*std_result, std_expect);

#undef XXX
}

TEST_F(RollingtVarStdTestUntyped, SimpleStaticVarianceStdInfNaN)
{
#define XXX 0.  // NULL stub

  using ResultType = double;

  double const inf           = std::numeric_limits<double>::infinity();
  double const nan           = std::numeric_limits<double>::signaling_NaN();
  cudf::size_type const ddof = 1, min_periods = 1, preceding_window = 3, following_window = 0;

  auto const col_data =
    cudf::test::make_type_param_vector<double>({5., 4., XXX, inf, 4., 8., 0., nan, XXX, 5.});
  std::vector<bool> const col_mask = {1, 1, 0, 1, 1, 1, 1, 1, 0, 1};

  auto const expected_var =
    std::vector<ResultType>{nan, 0.5, 0.5, nan, nan, nan, 16, nan, nan, nan};
  std::vector<ResultType> expected_std(expected_var.size());
  std::transform(expected_var.begin(), expected_var.end(), expected_std.begin(), [](auto const& x) {
    return std::sqrt(x);
  });

  std::vector<bool> const expected_mask = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1};

  cudf::test::fixed_width_column_wrapper<double> input(
    col_data.begin(), col_data.end(), col_mask.begin());
  cudf::test::fixed_width_column_wrapper<ResultType> var_expect(
    expected_var.begin(), expected_var.end(), expected_mask.begin());
  cudf::test::fixed_width_column_wrapper<ResultType> std_expect(
    expected_std.begin(), expected_std.end(), expected_mask.begin());

  std::unique_ptr<cudf::column> var_result, std_result;
  // static sizes
  EXPECT_NO_THROW(var_result = cudf::rolling_window(input,
                                                    preceding_window,
                                                    following_window,
                                                    min_periods,
                                                    dynamic_cast<cudf::rolling_aggregation const&>(
                                                      *cudf::make_variance_aggregation(ddof))););
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*var_result, var_expect);

  EXPECT_NO_THROW(std_result = cudf::rolling_window(input,
                                                    preceding_window,
                                                    following_window,
                                                    min_periods,
                                                    dynamic_cast<cudf::rolling_aggregation const&>(
                                                      *cudf::make_std_aggregation(ddof))););
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*std_result, std_expect);

#undef XXX
}

/*
// negative sizes
TYPED_TEST(RollingTest, NegativeWindowSizes)
{
  auto const col_data  = cudf::test::make_type_param_vector<TypeParam>({0, 1, 2, 0, 4});
  auto const col_valid = std::vector<bool>{1, 1, 1, 0, 1};
  cudf::test::fixed_width_column_wrapper<TypeParam> input(
    col_data.begin(), col_data.end(), col_valid.begin());
  std::vector<cudf::size_type> window{3};
  std::vector<cudf::size_type> negative_window{-2};


  this->run_test_col_agg(input, negative_window, window, 1);
  this->run_test_col_agg(input, window, negative_window, 1);
  this->run_test_col_agg(input, negative_window, negative_window, 1);
}
 */

// simple example from Pandas docs:
TYPED_TEST(RollingTest, SimpleDynamic)
{
  // https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.rolling.html
  auto const col_data              = cudf::test::make_type_param_vector<TypeParam>({0, 1, 2, 0, 4});
  std::vector<bool> const col_mask = {1, 1, 1, 0, 1};

  cudf::test::fixed_width_column_wrapper<TypeParam> input(
    col_data.begin(), col_data.end(), col_mask.begin());
  std::vector<cudf::size_type> preceding_window({1, 2, 3, 4, 2});
  std::vector<cudf::size_type> following_window({2, 1, 2, 1, 0});

  // dynamic sizes
  this->run_test_col_agg(input, preceding_window, following_window, 1);
}

// all rows are invalid
TYPED_TEST(RollingTest, AllInvalid)
{
  cudf::size_type num_rows = 1000;

  std::vector<TypeParam> col_data(num_rows);
  std::vector<bool> col_mask(num_rows, 0);

  cudf::test::fixed_width_column_wrapper<TypeParam> input(
    col_data.begin(), col_data.end(), col_mask.begin());
  std::vector<cudf::size_type> window({100});
  cudf::size_type periods = 100;

  this->run_test_col_agg(input, window, window, periods);
}

// window = following_window = 0
// Note: Preceding includes current row, so its value is set to 1.
TYPED_TEST(RollingTest, ZeroWindow)
{
  cudf::size_type num_rows = 1000;

  std::vector<int> col_data(num_rows, 1);
  std::vector<bool> col_mask(num_rows, 1);

  cudf::test::fixed_width_column_wrapper<TypeParam, int> input(
    col_data.begin(), col_data.end(), col_mask.begin());
  std::vector<cudf::size_type> preceding({0});
  std::vector<cudf::size_type> following({1});
  cudf::size_type periods = num_rows;

  this->run_test_col_agg(input, preceding, following, periods);
}

// min_periods = 0
TYPED_TEST(RollingTest, ZeroPeriods)
{
  cudf::size_type num_rows = 1000;

  std::vector<int> col_data(num_rows, 1);
  std::vector<bool> col_mask(num_rows, 1);

  cudf::test::fixed_width_column_wrapper<TypeParam, int> input(
    col_data.begin(), col_data.end(), col_mask.begin());

  std::vector<cudf::size_type> window({num_rows});
  cudf::size_type periods = 0;

  this->run_test_col_agg(input, window, window, periods);
}

// window in one direction is not large enough to collect enough samples,
//   but if using both directions we should get == min_periods,
// also tests out of boundary accesses
TYPED_TEST(RollingTest, BackwardForwardWindow)
{
  cudf::size_type num_rows = 1000;

  std::vector<int> col_data(num_rows, 1);
  std::vector<bool> col_mask(num_rows, 1);

  cudf::test::fixed_width_column_wrapper<TypeParam, int> input(
    col_data.begin(), col_data.end(), col_mask.begin());

  std::vector<cudf::size_type> window({num_rows});
  cudf::size_type periods = num_rows;

  this->run_test_col_agg(input, window, window, periods);
}

// random input data, static parameters, no nulls
TYPED_TEST(RollingTest, RandomStaticAllValid)
{
  cudf::size_type num_rows = 10000;

  // random input
  std::vector<TypeParam> col_data(num_rows);
  cudf::test::UniformRandomGenerator<TypeParam> rng;
  std::generate(col_data.begin(), col_data.end(), [&rng]() { return rng.generate(); });
  cudf::test::fixed_width_column_wrapper<TypeParam> input(col_data.begin(), col_data.end());

  std::vector<cudf::size_type> window({50});
  cudf::size_type periods = 50;

  this->run_test_col_agg(input, window, window, periods);
}

// random input data, static parameters, with nulls
TYPED_TEST(RollingTest, RandomStaticWithInvalid)
{
  cudf::size_type num_rows = 10000;

  // random input
  std::vector<TypeParam> col_data(num_rows);
  std::vector<bool> col_valid(num_rows);
  cudf::test::UniformRandomGenerator<TypeParam> rng;
  cudf::test::UniformRandomGenerator<bool> rbg;
  std::generate(col_data.begin(), col_data.end(), [&rng]() { return rng.generate(); });
  std::generate(col_valid.begin(), col_valid.end(), [&rbg]() { return rbg.generate(); });
  cudf::test::fixed_width_column_wrapper<TypeParam> input(
    col_data.begin(), col_data.end(), col_valid.begin());

  std::vector<cudf::size_type> window({50});
  cudf::size_type periods = 50;

  this->run_test_col_agg(input, window, window, periods);
}

// random input data, dynamic parameters, no nulls
TYPED_TEST(RollingTest, RandomDynamicAllValid)
{
  cudf::size_type num_rows        = 50000;
  cudf::size_type max_window_size = 50;

  // random input
  std::vector<TypeParam> col_data(num_rows);
  cudf::test::UniformRandomGenerator<TypeParam> rng;
  std::generate(col_data.begin(), col_data.end(), [&rng]() { return rng.generate(); });
  cudf::test::fixed_width_column_wrapper<TypeParam> input(col_data.begin(), col_data.end());

  // random parameters
  cudf::test::UniformRandomGenerator<cudf::size_type> window_rng(0, max_window_size);

  std::vector<cudf::size_type> preceding_window(num_rows);
  std::vector<cudf::size_type> following_window(num_rows);

  auto it = cuda::counting_iterator<cudf::size_type>{0};
  std::transform(it, it + num_rows, preceding_window.begin(), [&window_rng, num_rows](auto i) {
    auto p = window_rng.generate();
    return std::min(i + 1, std::max(p, i + 1 - num_rows));
  });
  std::transform(it, it + num_rows, following_window.begin(), [&window_rng, num_rows](auto i) {
    auto f = window_rng.generate();
    return std::max(-i - 1, std::min(f, num_rows - i - 1));
  });
  this->run_test_col_agg(input, preceding_window, following_window, max_window_size);
}

// random input data, dynamic parameters, with nulls
TYPED_TEST(RollingTest, RandomDynamicWithInvalid)
{
  cudf::size_type num_rows        = 50000;
  cudf::size_type max_window_size = 50;

  // random input with nulls
  std::vector<TypeParam> col_data(num_rows);
  std::vector<bool> col_valid(num_rows);
  cudf::test::UniformRandomGenerator<TypeParam> rng;
  cudf::test::UniformRandomGenerator<bool> rbg;
  std::generate(col_data.begin(), col_data.end(), [&rng]() { return rng.generate(); });
  std::generate(col_valid.begin(), col_valid.end(), [&rbg]() { return rbg.generate(); });
  cudf::test::fixed_width_column_wrapper<TypeParam> input(
    col_data.begin(), col_data.end(), col_valid.begin());

  // random parameters
  cudf::test::UniformRandomGenerator<cudf::size_type> window_rng(0, max_window_size);

  std::vector<cudf::size_type> preceding_window(num_rows);
  std::vector<cudf::size_type> following_window(num_rows);

  auto it = cuda::counting_iterator<cudf::size_type>{0};
  std::transform(it, it + num_rows, preceding_window.begin(), [&window_rng, num_rows](auto i) {
    auto p = window_rng.generate();
    return std::min(i + 1, std::max(p, i + 1 - num_rows));
  });
  std::transform(it, it + num_rows, following_window.begin(), [&window_rng, num_rows](auto i) {
    auto f = window_rng.generate();
    return std::max(-i - 1, std::min(f, num_rows - i - 1));
  });
  this->run_test_col_agg(input, preceding_window, following_window, max_window_size);
}

// ------------- non-fixed-width types --------------------

using RollingTestStrings = RollingTest<cudf::string_view>;

TEST_F(RollingTestStrings, StringsUnsupportedOperators)
{
  cudf::test::strings_column_wrapper input{{"This", "is", "not", "", "string", ""},
                                           {1, 1, 1, 0, 1, 0}};

  std::vector<cudf::size_type> window{1};

  EXPECT_THROW(
    cudf::rolling_window(input, 2, 2, 0, *cudf::make_sum_aggregation<cudf::rolling_aggregation>()),
    cudf::logic_error);
  EXPECT_THROW(
    cudf::rolling_window(input, 2, 2, 0, *cudf::make_mean_aggregation<cudf::rolling_aggregation>()),
    cudf::logic_error);
  EXPECT_THROW(
    cudf::rolling_window(input,
                         2,
                         2,
                         0,
                         *cudf::make_udf_aggregation<cudf::rolling_aggregation>(
                           cudf::udf_source_type::PTX, std::string{}, cudf::data_type{})),
    cudf::logic_error);
  EXPECT_THROW(
    cudf::rolling_window(input,
                         2,
                         2,
                         0,
                         *cudf::make_udf_aggregation<cudf::rolling_aggregation>(
                           cudf::udf_source_type::CUDA, std::string{}, cudf::data_type{})),
    cudf::logic_error);
}

/*TEST_F(RollingTestStrings, SimpleStatic)
{
  cudf::test::strings_column_wrapper input{{"This", "is", "not", "a", "string", "type"},
                                           {1, 1, 1, 0, 1, 0}};

  std::vector<cudf::size_type> window{1};

  EXPECT_NO_THROW(this->run_test_col(input, window, window, 0, rolling_operator::MIN));
  EXPECT_NO_THROW(this->run_test_col(input, window, window, 0, rolling_operator::MAX));
  EXPECT_NO_THROW(this->run_test_col(input, window, window, 0, rolling_operator::COUNT_VALID));
  EXPECT_NO_THROW(this->run_test_col(input, window, window, 0, rolling_operator::COUNT_ALL));
}*/

struct RollingTestUdf : public cudf::test::BaseFixture {
  std::string const cuda_func{
    R"***(
      template <typename OutType, typename InType>
      __device__ void CUDA_GENERIC_AGGREGATOR(OutType *ret, InType *in_col, cudf::size_type start,
                                              cudf::size_type count) {
        OutType val = 0;
        for (cudf::size_type i = 0; i < count; i++) {
          val += in_col[start + i];
        }
        *ret = val;
      }
    )***"};

  std::string const ptx_func{
    R"***(
    //
    // Generated by NVIDIA NVVM Compiler
    //
    // Compiler Build ID: CL-24817639
    // Cuda compilation tools, release 10.0, V10.0.130
    // Based on LLVM 3.4svn
    //

    .version 6.3
    .target sm_70
    .address_size 64

    // .globl	_ZN8__main__7add$241E5ArrayIiLi1E1A7mutable7alignedE
    .common .global .align 8 .u64 _ZN08NumbaEnv8__main__7add$241E5ArrayIiLi1E1A7mutable7alignedE;

    .visible .func  (.param .b32 func_retval0) _ZN8__main__7add$241E5ArrayIiLi1E1A7mutable7alignedE(
    .param .b64 _ZN8__main__7add$241E5ArrayIiLi1E1A7mutable7alignedE_paam_0,
    .param .b64 _ZN8__main__7add$241E5ArrayIiLi1E1A7mutable7alignedE_paam_1,
    .param .b64 _ZN8__main__7add$241E5ArrayIiLi1E1A7mutable7alignedE_paam_2,
    .param .b64 _ZN8__main__7add$241E5ArrayIiLi1E1A7mutable7alignedE_paam_3,
    .param .b64 _ZN8__main__7add$241E5ArrayIiLi1E1A7mutable7alignedE_paam_4,
    .param .b64 _ZN8__main__7add$241E5ArrayIiLi1E1A7mutable7alignedE_paam_5,
    .param .b64 _ZN8__main__7add$241E5ArrayIiLi1E1A7mutable7alignedE_paam_6,
    .param .b64 _ZN8__main__7add$241E5ArrayIiLi1E1A7mutable7alignedE_paam_7
    )
    {
    .reg .pred 	%p<3>;
    .reg .b32 	%r<6>;
    .reg .b64 	%rd<18>;


    ld.param.u64 	%rd6, [_ZN8__main__7add$241E5ArrayIiLi1E1A7mutable7alignedE_paam_0];
    ld.param.u64 	%rd7, [_ZN8__main__7add$241E5ArrayIiLi1E1A7mutable7alignedE_paam_5];
    ld.param.u64 	%rd8, [_ZN8__main__7add$241E5ArrayIiLi1E1A7mutable7alignedE_paam_6];
    ld.param.u64 	%rd9, [_ZN8__main__7add$241E5ArrayIiLi1E1A7mutable7alignedE_paam_7];
    mov.u64 	%rd15, 0;
    mov.u64 	%rd16, %rd15;

    BB0_1:
    mov.u64 	%rd2, %rd16;
    mov.u32 	%r5, 0;
    setp.ge.s64	%p1, %rd15, %rd8;
    mov.u64 	%rd17, %rd15;
    @%p1 bra 	BB0_3;

    mul.lo.s64 	%rd12, %rd15, %rd9;
    add.s64 	%rd13, %rd12, %rd7;
    ld.u32 	%r5, [%rd13];
    add.s64 	%rd17, %rd15, 1;

    BB0_3:
    cvt.s64.s32	%rd14, %r5;
    add.s64 	%rd16, %rd14, %rd2;
    setp.lt.s64	%p2, %rd15, %rd8;
    mov.u64 	%rd15, %rd17;
    @%p2 bra 	BB0_1;

    st.u64 	[%rd6], %rd2;
    mov.u32 	%r4, 0;
    st.param.b32	[func_retval0+0], %r4;
    ret;
    }
    )***"};
};

TEST_F(RollingTestUdf, StaticWindow)
{
  cudf::size_type size = 1000;

  cudf::test::fixed_width_column_wrapper<int32_t> input(cuda::counting_iterator<int32_t>{0},
                                                        cuda::counting_iterator{size},
                                                        cuda::make_constant_iterator(true));

  std::unique_ptr<cudf::column> output;

  auto start = cudf::detail::make_counting_transform_iterator(0, [size](cudf::size_type row) {
    return std::accumulate(cuda::counting_iterator{std::max(0, row - 2 + 1)},
                           cuda::counting_iterator{std::min(size, row + 2 + 1)},
                           0);
  });

  auto valid = cudf::detail::make_counting_transform_iterator(
    0, [size](cudf::size_type row) { return (row != 0 && row != size - 2 && row != size - 1); });

  cudf::test::fixed_width_column_wrapper<int64_t> expected{start, start + size, valid};

  // Test CUDA UDF
  auto cuda_udf_agg = cudf::make_udf_aggregation<cudf::rolling_aggregation>(
    cudf::udf_source_type::CUDA, this->cuda_func, cudf::data_type{cudf::type_id::INT64});

  output = cudf::rolling_window(input, 2, 2, 4, *cuda_udf_agg);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*output, expected);

  // Test NUMBA UDF
  auto ptx_udf_agg = cudf::make_udf_aggregation<cudf::rolling_aggregation>(
    cudf::udf_source_type::PTX, this->ptx_func, cudf::data_type{cudf::type_id::INT64});

  output = cudf::rolling_window(input, 2, 2, 4, *ptx_udf_agg);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*output, expected);
}

TEST_F(RollingTestUdf, DynamicWindow)
{
  cudf::size_type size = 1000;

  cudf::test::fixed_width_column_wrapper<int32_t> input(cuda::counting_iterator<int32_t>{0},
                                                        cuda::counting_iterator{size},
                                                        cuda::make_constant_iterator(true));

  auto prec = cudf::detail::make_counting_transform_iterator(
    0, [] __device__(cudf::size_type row) { return row % 2 + 2; });

  auto follow = cudf::detail::make_counting_transform_iterator(
    0, [] __device__(cudf::size_type row) { return row % 2; });

  cudf::test::fixed_width_column_wrapper<int32_t> preceding(prec, prec + size);
  cudf::test::fixed_width_column_wrapper<int32_t> following(follow, follow + size);
  std::unique_ptr<cudf::column> output;

  auto start =
    cudf::detail::make_counting_transform_iterator(0, [size] __device__(cudf::size_type row) {
      return std::accumulate(cuda::counting_iterator{std::max(0, row - (row % 2 + 2) + 1)},
                             cuda::counting_iterator{std::min(size, row + (row % 2) + 1)},
                             0);
    });

  auto valid = cudf::detail::make_counting_transform_iterator(
    0, [] __device__(cudf::size_type row) { return row != 0; });

  cudf::test::fixed_width_column_wrapper<int64_t> expected{start, start + size, valid};

  // Test CUDA UDF
  auto cuda_udf_agg = cudf::make_udf_aggregation<cudf::rolling_aggregation>(
    cudf::udf_source_type::CUDA, this->cuda_func, cudf::data_type{cudf::type_id::INT64});

  output = cudf::rolling_window(input, preceding, following, 2, *cuda_udf_agg);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*output, expected);

  // Test PTX UDF
  auto ptx_udf_agg = cudf::make_udf_aggregation<cudf::rolling_aggregation>(
    cudf::udf_source_type::PTX, this->ptx_func, cudf::data_type{cudf::type_id::INT64});

  output = cudf::rolling_window(input, preceding, following, 2, *ptx_udf_agg);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*output, expected);
}

template <typename T>
struct FixedPointTests : public cudf::test::BaseFixture {};

TYPED_TEST_SUITE(FixedPointTests, cudf::test::FixedPointTypes);

TYPED_TEST(FixedPointTests, MinMaxCountLagLead)
{
  using decimalXX  = TypeParam;
  using RepType    = cudf::device_storage_type_t<decimalXX>;
  using fp_wrapper = cudf::test::fixed_point_column_wrapper<RepType>;
  using fw_wrapper = cudf::test::fixed_width_column_wrapper<cudf::size_type>;

  auto const scale         = numeric::scale_type{-1};
  auto const input         = fp_wrapper{{42, 1729, 55, 3, 1, 2}, {1, 1, 1, 1, 1, 1}, scale};
  auto const expected_min  = fp_wrapper{{42, 42, 3, 1, 1, 1}, {1, 1, 1, 1, 1, 1}, scale};
  auto const expected_max  = fp_wrapper{{1729, 1729, 1729, 55, 3, 2}, {1, 1, 1, 1, 1, 1}, scale};
  auto const expected_lag  = fp_wrapper{{0, 42, 1729, 55, 3, 1}, {0, 1, 1, 1, 1, 1}, scale};
  auto const expected_lead = fp_wrapper{{1729, 55, 3, 1, 2, 0}, {1, 1, 1, 1, 1, 0}, scale};
  auto const expected_count_val = fw_wrapper{{2, 3, 3, 3, 3, 2}, {1, 1, 1, 1, 1, 1}};
  auto const expected_count_all = fw_wrapper{{2, 3, 3, 3, 3, 2}, {1, 1, 1, 1, 1, 1}};
  auto const expected_rowno     = fw_wrapper{{1, 2, 2, 2, 2, 2}, {1, 1, 1, 1, 1, 1}};
  auto const expected_rowno1    = fw_wrapper{{1, 1, 1, 1, 1, 1}, {1, 1, 1, 1, 1, 1}};

  auto const min =
    cudf::rolling_window(input, 2, 1, 1, *cudf::make_min_aggregation<cudf::rolling_aggregation>());
  auto const max =
    cudf::rolling_window(input, 2, 1, 1, *cudf::make_max_aggregation<cudf::rolling_aggregation>());
  auto const lag =
    cudf::rolling_window(input, 2, 1, 1, *cudf::make_lag_aggregation<cudf::rolling_aggregation>(1));
  auto const lead = cudf::rolling_window(
    input, 2, 1, 1, *cudf::make_lead_aggregation<cudf::rolling_aggregation>(1));
  auto const valid = cudf::rolling_window(
    input, 2, 1, 1, *cudf::make_count_aggregation<cudf::rolling_aggregation>());
  auto const all = cudf::rolling_window(
    input,
    2,
    1,
    1,
    *cudf::make_count_aggregation<cudf::rolling_aggregation>(cudf::null_policy::INCLUDE));
  auto const rowno = cudf::rolling_window(
    input, 2, 1, 1, *cudf::make_row_number_aggregation<cudf::rolling_aggregation>());

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_min, min->view());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_max, max->view());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_lag, lag->view());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_lead, lead->view());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_count_val, valid->view());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_count_all, all->view());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_rowno, rowno->view());

  // ROW_NUMBER will always return row 1 if the preceding window is set to a constant 1
  for (int following = 1; following < 5; ++following) {
    auto const rowno1 = cudf::rolling_window(
      input, 1, following, 1, *cudf::make_row_number_aggregation<cudf::rolling_aggregation>());
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_rowno1, rowno1->view());
  }
}

TYPED_TEST(FixedPointTests, MinMaxCountLagLeadNulls)
{
  using decimalXX  = TypeParam;
  using RepType    = cudf::device_storage_type_t<decimalXX>;
  using fp_wrapper = cudf::test::fixed_point_column_wrapper<RepType>;
  using fw_wrapper = cudf::test::fixed_width_column_wrapper<cudf::size_type>;

  auto const scale              = numeric::scale_type{-1};
  auto const input              = fp_wrapper{{42, 1729, 55, 343, 1, 2}, {1, 0, 1, 0, 1, 1}, scale};
  auto const expected_sum       = fp_wrapper{{42, 97, 55, 56, 3, 3}, {1, 1, 1, 1, 1, 1}, scale};
  auto const expected_min       = fp_wrapper{{42, 42, 55, 1, 1, 1}, {1, 1, 1, 1, 1, 1}, scale};
  auto const expected_max       = fp_wrapper{{42, 55, 55, 55, 2, 2}, {1, 1, 1, 1, 1, 1}, scale};
  auto const expected_lag       = fp_wrapper{{0, 42, 1729, 55, 343, 1}, {0, 1, 0, 1, 0, 1}, scale};
  auto const expected_lead      = fp_wrapper{{1729, 55, 343, 1, 2, 0}, {0, 1, 0, 1, 1, 0}, scale};
  auto const expected_count_val = fw_wrapper{{1, 2, 1, 2, 2, 2}, {1, 1, 1, 1, 1, 1}};
  auto const expected_count_all = fw_wrapper{{2, 3, 3, 3, 3, 2}, {1, 1, 1, 1, 1, 1}};
  auto const expected_rowno     = fw_wrapper{{1, 2, 2, 2, 2, 2}, {1, 1, 1, 1, 1, 1}};

  auto const sum =
    cudf::rolling_window(input, 2, 1, 1, *cudf::make_sum_aggregation<cudf::rolling_aggregation>());
  auto const min =
    cudf::rolling_window(input, 2, 1, 1, *cudf::make_min_aggregation<cudf::rolling_aggregation>());
  auto const max =
    cudf::rolling_window(input, 2, 1, 1, *cudf::make_max_aggregation<cudf::rolling_aggregation>());
  auto const lag =
    cudf::rolling_window(input, 2, 1, 1, *cudf::make_lag_aggregation<cudf::rolling_aggregation>(1));
  auto const lead = cudf::rolling_window(
    input, 2, 1, 1, *cudf::make_lead_aggregation<cudf::rolling_aggregation>(1));
  auto const valid = cudf::rolling_window(
    input, 2, 1, 1, *cudf::make_count_aggregation<cudf::rolling_aggregation>());
  auto const all = cudf::rolling_window(
    input,
    2,
    1,
    1,
    *cudf::make_count_aggregation<cudf::rolling_aggregation>(cudf::null_policy::INCLUDE));
  auto const rowno = cudf::rolling_window(
    input, 2, 1, 1, *cudf::make_row_number_aggregation<cudf::rolling_aggregation>());

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_sum, sum->view());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_min, min->view());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_max, max->view());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_lag, lag->view());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_lead, lead->view());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_count_val, valid->view());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_count_all, all->view());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_rowno, rowno->view());
}

TYPED_TEST(FixedPointTests, VarStd)
{
  using decimalXX  = TypeParam;
  using RepType    = cudf::device_storage_type_t<decimalXX>;
  using fp_wrapper = cudf::test::fixed_point_column_wrapper<RepType>;
  using fw_wrapper = cudf::test::fixed_width_column_wrapper<double>;

  double const nan = std::numeric_limits<double>::signaling_NaN();
  double const inf = std::numeric_limits<double>::infinity();
  cudf::size_type preceding_window{3}, following_window{0}, min_periods{1}, ddof{2};

  // The variance of `input` given `scale` == 0
  std::vector<double> result_base_v{
    nan, inf, 1882804.66666666667, 1928018.666666666667, 1874.6666666666667, 2.0};
  std::vector<bool> result_mask_v{1, 1, 1, 1, 1, 1};

  // var tests
  for (int32_t s = -2; s <= 2; s++) {
    auto const scale = numeric::scale_type{s};
    auto const input = fp_wrapper{{42, 1729, 55, 3, 1, 2}, {1, 1, 1, 1, 1, 1}, scale};

    auto got = cudf::rolling_window(
      input,
      preceding_window,
      following_window,
      min_periods,
      dynamic_cast<cudf::rolling_aggregation const&>(*cudf::make_variance_aggregation(ddof)));

    std::vector<double> result_scaled_v(result_base_v.size());
    std::transform(
      result_base_v.begin(), result_base_v.end(), result_scaled_v.begin(), [&s](auto x) {
        // When values are scaled by 10^n, the variance is scaled by 10^2n.
        return x * exp10(s) * exp10(s);
      });
    fw_wrapper expect(result_scaled_v.begin(), result_scaled_v.end(), result_mask_v.begin());

    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expect, *got);
  }

  // std tests
  for (int32_t s = -2; s <= 2; s++) {
    auto const scale = numeric::scale_type{s};
    auto const input = fp_wrapper{{42, 1729, 55, 3, 1, 2}, {1, 1, 1, 1, 1, 1}, scale};

    auto got = cudf::rolling_window(
      input,
      preceding_window,
      following_window,
      min_periods,
      dynamic_cast<cudf::rolling_aggregation const&>(*cudf::make_std_aggregation(ddof)));

    std::vector<double> result_scaled_v(result_base_v.size());
    std::transform(
      result_base_v.begin(), result_base_v.end(), result_scaled_v.begin(), [&s](auto x) {
        // When values are scaled by 10^n, the variance is scaled by 10^2n.
        return std::sqrt(x * exp10(s) * exp10(s));
      });
    fw_wrapper expect(result_scaled_v.begin(), result_scaled_v.end(), result_mask_v.begin());

    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expect, *got);
  }
}

class RollingDictionaryTest : public cudf::test::BaseFixture {};

TEST_F(RollingDictionaryTest, Count)
{
  cudf::test::dictionary_column_wrapper<std::string> input(
    {"This", "is", "rolling", "test", "being", "operated", "on", "string", "column"},
    {1, 0, 0, 1, 0, 1, 1, 1, 0});
  cudf::test::fixed_width_column_wrapper<cudf::size_type> expected_count_val(
    {1, 2, 1, 2, 3, 3, 3, 2, 1}, {1, 1, 1, 1, 1, 1, 1, 1, 1});
  cudf::test::fixed_width_column_wrapper<cudf::size_type> expected_count_all(
    {3, 4, 4, 4, 4, 4, 4, 3, 2}, {1, 1, 1, 1, 1, 1, 1, 1, 1});
  cudf::test::fixed_width_column_wrapper<cudf::size_type> expected_row_number(
    {1, 2, 2, 2, 2, 2, 2, 2, 2}, {1, 1, 1, 1, 1, 1, 1, 1, 1});

  auto got_count_valid = cudf::rolling_window(
    input, 2, 2, 1, *cudf::make_count_aggregation<cudf::rolling_aggregation>());
  auto got_count_all = cudf::rolling_window(
    input,
    2,
    2,
    1,
    *cudf::make_count_aggregation<cudf::rolling_aggregation>(cudf::null_policy::INCLUDE));
  auto got_row_number = cudf::rolling_window(
    input, 2, 2, 1, *cudf::make_row_number_aggregation<cudf::rolling_aggregation>());

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_count_val, got_count_valid->view());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_count_all, got_count_all->view());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_row_number, got_row_number->view());
}

TEST_F(RollingDictionaryTest, MinMax)
{
  cudf::test::dictionary_column_wrapper<std::string> input(
    {"This", "is", "rolling", "test", "being", "operated", "on", "string", "column"},
    {1, 0, 0, 1, 0, 1, 1, 1, 0});
  cudf::test::strings_column_wrapper expected_min(
    {"This", "This", "test", "operated", "on", "on", "on", "on", "string"},
    {1, 1, 1, 1, 1, 1, 1, 1, 1});
  cudf::test::strings_column_wrapper expected_max(
    {"This", "test", "test", "test", "test", "string", "string", "string", "string"},
    {1, 1, 1, 1, 1, 1, 1, 1, 1});

  auto got_min_dict =
    cudf::rolling_window(input, 2, 2, 1, *cudf::make_min_aggregation<cudf::rolling_aggregation>());
  auto got_min = cudf::dictionary::decode(cudf::dictionary_column_view(got_min_dict->view()));

  auto got_max_dict =
    cudf::rolling_window(input, 2, 2, 1, *cudf::make_max_aggregation<cudf::rolling_aggregation>());
  auto got_max = cudf::dictionary::decode(cudf::dictionary_column_view(got_max_dict->view()));

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_min, got_min->view());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_max, got_max->view());
}

TEST_F(RollingDictionaryTest, LeadLag)
{
  cudf::test::dictionary_column_wrapper<std::string> input(
    {"This", "is", "rolling", "test", "being", "operated", "on", "string", "column"},
    {1, 0, 0, 1, 0, 1, 1, 1, 0});
  cudf::test::strings_column_wrapper expected_lead(
    {"", "", "test", "", "operated", "on", "string", "", ""}, {0, 0, 1, 0, 1, 1, 1, 0, 0});
  cudf::test::strings_column_wrapper expected_lag(
    {"", "This", "", "", "test", "", "operated", "on", "string"}, {0, 1, 0, 0, 1, 0, 1, 1, 1});

  auto got_lead_dict = cudf::rolling_window(
    input, 2, 1, 1, *cudf::make_lead_aggregation<cudf::rolling_aggregation>(1));
  auto got_lead = cudf::dictionary::decode(cudf::dictionary_column_view(got_lead_dict->view()));

  auto got_lag_dict =
    cudf::rolling_window(input, 2, 2, 1, *cudf::make_lag_aggregation<cudf::rolling_aggregation>(1));
  auto got_lag = cudf::dictionary::decode(cudf::dictionary_column_view(got_lag_dict->view()));

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_lead, got_lead->view());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_lag, got_lag->view());
}

CUDF_TEST_PROGRAM_MAIN()
