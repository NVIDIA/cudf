/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/aggregation.hpp>
#include <cudf/copying.hpp>
#include <cudf/reduction.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/utilities/span.hpp>
#include <cudf/utilities/traits.hpp>

#include <cmath>
#include <limits>
#include <numeric>
#include <vector>

// min/max follow the IEEE 754-2019 minimum/maximum operations for floating-point values:
// NaN propagates and -0 is less than +0. argmin/argmax select the first NaN.

template <typename T>
struct MinMaxNaNTest : public cudf::test::BaseFixture {
  static constexpr T NaN = std::numeric_limits<T>::quiet_NaN();

  static T reduce_value(cudf::column_view const& input, cudf::reduce_aggregation const& agg)
  {
    auto const result = cudf::reduce(input, agg, cudf::data_type{cudf::type_to_id<T>()});
    EXPECT_TRUE(result->is_valid());
    return static_cast<cudf::numeric_scalar<T> const*>(result.get())->value();
  }

  static cudf::size_type reduce_index(cudf::column_view const& input,
                                      cudf::reduce_aggregation const& agg)
  {
    auto const result = cudf::reduce(input, agg, cudf::data_type{cudf::type_id::INT32});
    EXPECT_TRUE(result->is_valid());
    return static_cast<cudf::numeric_scalar<cudf::size_type> const*>(result.get())->value();
  }

  struct expected {
    bool is_nan;  // min/max result is NaN
    T min;        // checked when !is_nan
    T max;        // checked when !is_nan
    cudf::size_type argmin;
    cudf::size_type argmax;
  };

  static void check(cudf::column_view const& input, expected const& exp)
  {
    auto const min_agg    = cudf::make_min_aggregation<cudf::reduce_aggregation>();
    auto const max_agg    = cudf::make_max_aggregation<cudf::reduce_aggregation>();
    auto const argmin_agg = cudf::make_argmin_aggregation<cudf::reduce_aggregation>();
    auto const argmax_agg = cudf::make_argmax_aggregation<cudf::reduce_aggregation>();

    auto const min              = reduce_value(input, *min_agg);
    auto const max              = reduce_value(input, *max_agg);
    auto const [mm_min, mm_max] = cudf::minmax(input);
    auto const mm_min_value = static_cast<cudf::numeric_scalar<T> const*>(mm_min.get())->value();
    auto const mm_max_value = static_cast<cudf::numeric_scalar<T> const*>(mm_max.get())->value();
    if (exp.is_nan) {
      EXPECT_TRUE(std::isnan(min));
      EXPECT_TRUE(std::isnan(max));
      EXPECT_TRUE(std::isnan(mm_min_value));
      EXPECT_TRUE(std::isnan(mm_max_value));
    } else {
      EXPECT_EQ(min, exp.min);
      EXPECT_EQ(max, exp.max);
      EXPECT_EQ(mm_min_value, exp.min);
      EXPECT_EQ(mm_max_value, exp.max);
    }
    EXPECT_EQ(reduce_index(input, *argmin_agg), exp.argmin);
    EXPECT_EQ(reduce_index(input, *argmax_agg), exp.argmax);
  }
};

TYPED_TEST_SUITE(MinMaxNaNTest, cudf::test::FloatingPointTypes);

TYPED_TEST(MinMaxNaNTest, Reduce)
{
  using T            = TypeParam;
  auto constexpr NaN = TestFixture::NaN;
  using wrapper      = cudf::test::fixed_width_column_wrapper<T>;

  // The result must not depend on where the NaN is
  this->check(wrapper{T{1}, NaN, T{3}}, {true, {}, {}, 1, 1});
  this->check(wrapper{NaN, T{1}, T{3}}, {true, {}, {}, 0, 0});
  this->check(wrapper{T{1}, T{3}, NaN}, {true, {}, {}, 2, 2});
  this->check(wrapper{NaN, NaN}, {true, {}, {}, 0, 0});
  this->check(wrapper{T{1}, T{3}, T{2}}, {false, T{1}, T{3}, 0, 1});

  // A null row is ignored, including when its underlying value is NaN
  this->check(wrapper{{T{0}, T{1}, NaN, T{2}}, {0, 1, 1, 1}}, {true, {}, {}, 2, 2});
  this->check(wrapper{{NaN, T{1}, T{2}}, {0, 1, 1}}, {false, T{1}, T{2}, 1, 2});
}

TYPED_TEST(MinMaxNaNTest, ReduceLarge)
{
  using T            = TypeParam;
  auto constexpr NaN = TestFixture::NaN;

  // Large enough for the reduction to run in parallel across many blocks
  cudf::size_type constexpr size = 100'000;
  for (auto const& nan_rows : std::vector<std::vector<cudf::size_type>>{
         {0}, {size / 2}, {size - 1}, {700, 90'000}, {5, 50'000, size - 1}}) {
    SCOPED_TRACE("first NaN row " + std::to_string(nan_rows.front()));
    std::vector<T> values(size);
    std::iota(values.begin(), values.end(), T{0});
    for (auto row : nan_rows) {
      values[row] = NaN;
    }
    auto const input = cudf::test::fixed_width_column_wrapper<T>(values.begin(), values.end());
    this->check(input, {true, {}, {}, nan_rows.front(), nan_rows.front()});
  }
}

TYPED_TEST(MinMaxNaNTest, Sliced)
{
  using T            = TypeParam;
  auto constexpr NaN = TestFixture::NaN;

  // NaNs and a null just outside each slice must not affect the result, and argmin/argmax are
  // relative to the slice
  auto const input = cudf::test::fixed_width_column_wrapper<T>{
    {NaN, T{0}, T{1}, T{3}, T{2}, NaN, T{4}}, {1, 0, 1, 1, 1, 1, 1}};
  auto const slices = cudf::slice(input, {1, 5, 2, 6, 0, 2});
  this->check(slices[0], {false, T{1}, T{3}, 1, 2});  // [null, 1, 3, 2]
  this->check(slices[1], {true, {}, {}, 3, 3});       // [1, 3, 2, NaN]
  this->check(slices[2], {true, {}, {}, 0, 0});       // [NaN, null]
}

TYPED_TEST(MinMaxNaNTest, SignedZeros)
{
  using T       = TypeParam;
  using wrapper = cudf::test::fixed_width_column_wrapper<T>;

  auto const min_agg = cudf::make_min_aggregation<cudf::reduce_aggregation>();
  auto const max_agg = cudf::make_max_aggregation<cudf::reduce_aggregation>();
  for (auto const& input : {wrapper{T{0}, -T{0}, T{0}}, wrapper{-T{0}, T{0}, -T{0}}}) {
    auto const min = this->reduce_value(input, *min_agg);
    auto const max = this->reduce_value(input, *max_agg);
    EXPECT_EQ(min, T{0});
    EXPECT_TRUE(std::signbit(min));
    EXPECT_EQ(max, T{0});
    EXPECT_FALSE(std::signbit(max));
  }
}

TYPED_TEST(MinMaxNaNTest, Scan)
{
  using T            = TypeParam;
  auto constexpr NaN = TestFixture::NaN;
  using wrapper      = cudf::test::fixed_width_column_wrapper<T>;

  auto const input    = wrapper{T{1}, NaN, T{3}, T{0}};
  auto const expected = wrapper{T{1}, NaN, NaN, NaN};
  for (auto const& agg : {cudf::make_min_aggregation<cudf::scan_aggregation>(),
                          cudf::make_max_aggregation<cudf::scan_aggregation>()}) {
    auto const result = cudf::scan(input, *agg, cudf::scan_type::INCLUSIVE);
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected, result->view());
  }
}

TYPED_TEST(MinMaxNaNTest, SegmentedReduce)
{
  using T            = TypeParam;
  auto constexpr NaN = TestFixture::NaN;
  using wrapper      = cudf::test::fixed_width_column_wrapper<T>;

  // segments: [1, NaN, 3], [2, 4], [NaN], []
  auto const input     = wrapper{T{1}, NaN, T{3}, T{2}, T{4}, NaN};
  auto const offsets   = cudf::test::fixed_width_column_wrapper<cudf::size_type>{0, 3, 5, 6, 6};
  auto const d_offsets = cudf::device_span<cudf::size_type const>(
    cudf::column_view(offsets).data<cudf::size_type>(), cudf::column_view(offsets).size());
  auto const output_type = cudf::data_type{cudf::type_to_id<T>()};

  auto constexpr nulls = cudf::null_policy::EXCLUDE;
  auto const min_agg   = cudf::make_min_aggregation<cudf::segmented_reduce_aggregation>();
  auto const max_agg   = cudf::make_max_aggregation<cudf::segmented_reduce_aggregation>();

  auto const min = cudf::segmented_reduce(input, d_offsets, *min_agg, output_type, nulls);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(wrapper({NaN, T{2}, NaN, T{0}}, {1, 1, 1, 0}), min->view());

  auto const max = cudf::segmented_reduce(input, d_offsets, *max_agg, output_type, nulls);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(wrapper({NaN, T{4}, NaN, T{0}}, {1, 1, 1, 0}), max->view());
}

// With nan_semantics::TOTALLY_ORDERED, NaN is greater than every other value: min/argmin ignore NaN
// unless every valid value is NaN, while max/argmax match UNORDERED (NaN is the maximum).

template <typename T>
struct MinMaxNaNTotallyOrderedTest : public MinMaxNaNTest<T> {
  static constexpr auto ordered = cudf::nan_semantics::TOTALLY_ORDERED;

  struct ordered_expected {
    bool min_is_nan;  // min result is NaN
    T min;            // checked when !min_is_nan
    cudf::size_type argmin;
    cudf::size_type argmax;  // max is always NaN in these tests when a NaN is present
  };

  static void check_ordered(cudf::column_view const& input, ordered_expected const& exp)
  {
    auto const min_agg    = cudf::make_min_aggregation<cudf::reduce_aggregation>(ordered);
    auto const argmin_agg = cudf::make_argmin_aggregation<cudf::reduce_aggregation>(ordered);
    auto const argmax_agg = cudf::make_argmax_aggregation<cudf::reduce_aggregation>(ordered);

    auto const min = MinMaxNaNTest<T>::reduce_value(input, *min_agg);
    if (exp.min_is_nan) {
      EXPECT_TRUE(std::isnan(min));
    } else {
      EXPECT_EQ(min, exp.min);
    }
    EXPECT_EQ(MinMaxNaNTest<T>::reduce_index(input, *argmin_agg), exp.argmin);
    EXPECT_EQ(MinMaxNaNTest<T>::reduce_index(input, *argmax_agg), exp.argmax);
  }
};

TYPED_TEST_SUITE(MinMaxNaNTotallyOrderedTest, cudf::test::FloatingPointTypes);

TYPED_TEST(MinMaxNaNTotallyOrderedTest, Reduce)
{
  using T            = TypeParam;
  auto constexpr NaN = TestFixture::NaN;
  using wrapper      = cudf::test::fixed_width_column_wrapper<T>;

  this->check_ordered(wrapper{T{1}, NaN, T{3}}, {false, T{1}, 0, 1});
  this->check_ordered(wrapper{NaN, T{3}, T{1}}, {false, T{1}, 2, 0});
  this->check_ordered(wrapper{T{3}, T{1}, NaN}, {false, T{1}, 1, 2});
  this->check_ordered(wrapper{NaN, -NaN, NaN}, {true, {}, 0, 0});
  this->check_ordered(wrapper{T{2}, T{1}, T{1}}, {false, T{1}, 1, 0});

  // Infinity is less than NaN
  auto constexpr inf = std::numeric_limits<T>::infinity();
  this->check_ordered(wrapper{NaN, inf}, {false, inf, 1, 0});

  // A null row is never selected, including when it is the only non-NaN row
  this->check_ordered(wrapper{{T{0}, NaN, T{2}}, {0, 1, 1}}, {false, T{2}, 2, 1});
  this->check_ordered(wrapper{{T{0}, NaN}, {0, 1}}, {true, {}, 1, 1});
  this->check_ordered(wrapper{{NaN, T{0}}, {1, 0}}, {true, {}, 0, 0});

  // TOTALLY_ORDERED and UNORDERED max agree
  auto const max_agg   = cudf::make_max_aggregation<cudf::reduce_aggregation>(TestFixture::ordered);
  auto const input     = wrapper{T{1}, NaN, T{3}};
  auto const max_value = this->reduce_value(input, *max_agg);
  EXPECT_TRUE(std::isnan(max_value));

  // All nulls produce a null result
  auto const nulls   = wrapper{{NaN, T{1}}, {0, 0}};
  auto const min_agg = cudf::make_min_aggregation<cudf::reduce_aggregation>(TestFixture::ordered);
  EXPECT_FALSE(cudf::reduce(nulls, *min_agg, cudf::data_type{cudf::type_to_id<T>()})->is_valid());
}

TYPED_TEST(MinMaxNaNTotallyOrderedTest, ReduceLarge)
{
  using T            = TypeParam;
  auto constexpr NaN = TestFixture::NaN;

  // Large enough for the reduction to run in parallel across many blocks. Every other row is NaN
  // and the minimum is placed at different positions.
  cudf::size_type constexpr size = 100'000;
  for (auto min_row : std::vector<cudf::size_type>{1, size / 2 + 1, size - 1}) {
    SCOPED_TRACE("min row " + std::to_string(min_row));
    std::vector<T> values(size);
    for (cudf::size_type i = 0; i < size; ++i) {
      values[i] = (i % 2 == 0) ? NaN : T{10} + static_cast<T>(i % 1000);
    }
    values[min_row]  = T{-1};
    auto const input = cudf::test::fixed_width_column_wrapper<T>(values.begin(), values.end());
    this->check_ordered(input, {false, T{-1}, min_row, 0});
  }

  // Every row is NaN
  std::vector<T> nans(size, NaN);
  auto const input = cudf::test::fixed_width_column_wrapper<T>(nans.begin(), nans.end());
  this->check_ordered(input, {true, {}, 0, 0});
}

TYPED_TEST(MinMaxNaNTotallyOrderedTest, Sliced)
{
  using T            = TypeParam;
  auto constexpr NaN = TestFixture::NaN;

  // Rows just outside each slice must not affect the result, and argmin/argmax are relative to
  // the slice
  auto const input = cudf::test::fixed_width_column_wrapper<T>{{T{-5}, NaN, T{3}, T{2}, NaN, T{-6}},
                                                               {1, 1, 1, 1, 1, 1}};
  auto const slices = cudf::slice(input, {1, 5, 4, 5});
  this->check_ordered(slices[0], {false, T{2}, 2, 0});  // [NaN, 3, 2, NaN]
  this->check_ordered(slices[1], {true, {}, 0, 0});     // [NaN]
}

TYPED_TEST(MinMaxNaNTotallyOrderedTest, SignedZeros)
{
  using T            = TypeParam;
  auto constexpr NaN = TestFixture::NaN;
  using wrapper      = cudf::test::fixed_width_column_wrapper<T>;

  auto const min_agg = cudf::make_min_aggregation<cudf::reduce_aggregation>(TestFixture::ordered);
  auto const argmin_agg =
    cudf::make_argmin_aggregation<cudf::reduce_aggregation>(TestFixture::ordered);
  for (auto const& input : {wrapper{T{0}, NaN, -T{0}, T{0}}, wrapper{-T{0}, T{0}, NaN, -T{0}}}) {
    auto const min = this->reduce_value(input, *min_agg);
    EXPECT_EQ(min, T{0});
    EXPECT_TRUE(std::signbit(min));
    // -0 and +0 are equal values for argmin, so the first zero is selected
    EXPECT_EQ(this->reduce_index(input, *argmin_agg), 0);
  }
}

TYPED_TEST(MinMaxNaNTotallyOrderedTest, InitialValue)
{
  using T            = TypeParam;
  auto constexpr NaN = TestFixture::NaN;
  using wrapper      = cudf::test::fixed_width_column_wrapper<T>;

  auto const min_agg = cudf::make_min_aggregation<cudf::reduce_aggregation>(TestFixture::ordered);
  auto const type    = cudf::data_type{cudf::type_to_id<T>()};
  auto const value   = [&](cudf::column_view const& input, T init) {
    auto const init_scalar = cudf::numeric_scalar<T>(init);
    auto const result      = cudf::reduce(input, *min_agg, type, std::cref(init_scalar));
    return static_cast<cudf::numeric_scalar<T> const*>(result.get())->value();
  };
  EXPECT_EQ(value(wrapper{NaN, T{2}}, NaN), T{2});
  EXPECT_EQ(value(wrapper{NaN, NaN}, T{2}), T{2});
  EXPECT_TRUE(std::isnan(value(wrapper{NaN, NaN}, NaN)));
}

TYPED_TEST(MinMaxNaNTotallyOrderedTest, Dictionary)
{
  using T            = TypeParam;
  auto constexpr NaN = TestFixture::NaN;

  auto const input =
    cudf::test::dictionary_column_wrapper<T>({NaN, T{3}, NaN, T{1}, T{3}}, {1, 1, 1, 1, 0});
  this->check_ordered(input, {false, T{1}, 3, 0});
  auto const nans = cudf::test::dictionary_column_wrapper<T>({T{1}, NaN, NaN}, {0, 1, 1});
  this->check_ordered(nans, {true, {}, 1, 1});
}

TYPED_TEST(MinMaxNaNTotallyOrderedTest, Scan)
{
  using T            = TypeParam;
  auto constexpr NaN = TestFixture::NaN;
  using wrapper      = cudf::test::fixed_width_column_wrapper<T>;

  auto const min_agg = cudf::make_min_aggregation<cudf::scan_aggregation>(TestFixture::ordered);
  auto const input   = wrapper{{NaN, T{3}, T{0}, T{1}, NaN}, {1, 1, 0, 1, 1}};

  auto const inclusive = cudf::scan(input, *min_agg, cudf::scan_type::INCLUSIVE);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(wrapper({NaN, T{3}, T{0}, T{1}, T{1}}, {1, 1, 0, 1, 1}),
                                      inclusive->view());

  // An exclusive scan starts from the identity, which is NaN with TOTALLY_ORDERED semantics
  auto const exclusive = cudf::scan(input, *min_agg, cudf::scan_type::EXCLUSIVE);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(wrapper({NaN, NaN, T{0}, T{3}, T{1}}, {1, 1, 0, 1, 1}),
                                      exclusive->view());

  // TOTALLY_ORDERED and UNORDERED max agree
  auto const max_agg = cudf::make_max_aggregation<cudf::scan_aggregation>(TestFixture::ordered);
  auto const max     = cudf::scan(wrapper{T{1}, NaN, T{3}}, *max_agg, cudf::scan_type::INCLUSIVE);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(wrapper{T{1}, NaN, NaN}, max->view());
}

TYPED_TEST(MinMaxNaNTotallyOrderedTest, SegmentedReduce)
{
  using T            = TypeParam;
  auto constexpr NaN = TestFixture::NaN;
  using wrapper      = cudf::test::fixed_width_column_wrapper<T>;

  // segments: [1, NaN, 3], [NaN, null], [NaN], [], [null, 2]
  auto const input =
    wrapper{{T{1}, NaN, T{3}, NaN, T{0}, NaN, T{0}, T{2}}, {1, 1, 1, 1, 0, 1, 0, 1}};
  auto const offsets   = cudf::test::fixed_width_column_wrapper<cudf::size_type>{0, 3, 5, 6, 6, 8};
  auto const d_offsets = cudf::device_span<cudf::size_type const>(
    cudf::column_view(offsets).data<cudf::size_type>(), cudf::column_view(offsets).size());
  auto const output_type = cudf::data_type{cudf::type_to_id<T>()};
  auto const min_agg =
    cudf::make_min_aggregation<cudf::segmented_reduce_aggregation>(TestFixture::ordered);

  auto const exclude =
    cudf::segmented_reduce(input, d_offsets, *min_agg, output_type, cudf::null_policy::EXCLUDE);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(wrapper({T{1}, NaN, NaN, T{0}, T{2}}, {1, 1, 1, 0, 1}),
                                      exclude->view());

  auto const include =
    cudf::segmented_reduce(input, d_offsets, *min_agg, output_type, cudf::null_policy::INCLUDE);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(wrapper({T{1}, T{0}, NaN, T{0}, T{0}}, {1, 0, 1, 0, 0}),
                                      include->view());
}
