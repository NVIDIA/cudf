/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/iterator_utilities.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/aggregation.hpp>
#include <cudf/rolling.hpp>

#include <cuda/iterator>

#include <memory>
#include <optional>

auto constexpr X = int32_t{0};  // Placeholder for null.

template <typename T>
using fwcw                 = cudf::test::fixed_width_column_wrapper<T>;
using grouping_keys_column = fwcw<int32_t>;

/// Rolling test executor with fluent interface.
class rolling_exec {
  cudf::size_type _preceding{1};
  cudf::size_type _following{0};
  cudf::size_type _min_periods{1};
  cudf::column_view _grouping;
  cudf::column_view _input;
  cudf::null_policy _null_handling = cudf::null_policy::INCLUDE;

 public:
  rolling_exec& preceding(cudf::size_type preceding)
  {
    _preceding = preceding;
    return *this;
  }
  rolling_exec& following(cudf::size_type following)
  {
    _following = following;
    return *this;
  }
  rolling_exec& min_periods(cudf::size_type min_periods)
  {
    _min_periods = min_periods;
    return *this;
  }
  rolling_exec& grouping(cudf::column_view grouping)
  {
    _grouping = grouping;
    return *this;
  }
  rolling_exec& input(cudf::column_view input)
  {
    _input = input;
    return *this;
  }
  rolling_exec& null_handling(cudf::null_policy null_handling)
  {
    _null_handling = null_handling;
    return *this;
  }

  [[nodiscard]] std::unique_ptr<cudf::column> test_grouped_nth_element(
    cudf::size_type n, std::optional<cudf::null_policy> null_handling = std::nullopt) const
  {
    return cudf::grouped_rolling_window(
      cudf::table_view{{_grouping}},
      _input,
      _preceding,
      _following,
      _min_periods,
      *cudf::make_nth_element_aggregation<cudf::rolling_aggregation>(
        n, null_handling.value_or(_null_handling)));
  }

  [[nodiscard]] std::unique_ptr<cudf::column> test_nth_element(
    cudf::size_type n, std::optional<cudf::null_policy> null_handling = std::nullopt) const
  {
    return cudf::rolling_window(_input,
                                _preceding,
                                _following,
                                _min_periods,
                                *cudf::make_nth_element_aggregation<cudf::rolling_aggregation>(
                                  n, null_handling.value_or(_null_handling)));
  }
};

struct NthElementTest : public cudf::test::BaseFixture {};

template <typename T>
struct NthElementTypedTest : public NthElementTest {};

using TypesForTest = cudf::test::Concat<cudf::test::IntegralTypes,
                                        cudf::test::FloatingPointTypes,
                                        cudf::test::DurationTypes,
                                        cudf::test::TimestampTypes>;

TYPED_TEST_SUITE(NthElementTypedTest, TypesForTest);

TYPED_TEST(NthElementTypedTest, RollingWindow)
{
  using T = TypeParam;

  auto const input_col =
    fwcw<T>{{0, 1, 2, 3, 4, X, 10, 11, 12, 13, 14, 15, 16, 20}, cudf::test::iterators::null_at(5)};
  auto tester = rolling_exec{}.input(input_col);
  {
    // Window of 5 elements, min-periods == 1.
    tester.preceding(3).following(2).min_periods(1);

    auto const first_element = tester.test_nth_element(0);
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(
      *first_element,
      fwcw<T>{{0, 0, 0, 1, 2, 3, 4, X, 10, 11, 12, 13, 14, 15}, cudf::test::iterators::null_at(7)});
    auto const last_element = tester.test_nth_element(-1);
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(
      *last_element,
      fwcw<T>{{2, 3, 4, X, 10, 11, 12, 13, 14, 15, 16, 20, 20, 20},
              cudf::test::iterators::null_at(3)});
    auto const third_element = tester.test_nth_element(2);
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*third_element,
                                        fwcw<T>{{2, 2, 2, 3, 4, X, 10, 11, 12, 13, 14, 15, 16, 20},
                                                cudf::test::iterators::null_at(5)});
    auto const second_last_element = tester.test_nth_element(-2);
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*second_last_element,
                                        fwcw<T>{{1, 2, 3, 4, X, 10, 11, 12, 13, 14, 15, 16, 16, 16},
                                                cudf::test::iterators::null_at(4)});
  }
  {
    // Window of 3 elements, min-periods == 3. Expect null elements at column margins.
    tester.preceding(2).following(1).min_periods(3);
    auto const first_element = tester.test_nth_element(0);
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*first_element,
                                        fwcw<T>{{X, 0, 1, 2, 3, 4, X, 10, 11, 12, 13, 14, 15, X},
                                                cudf::test::iterators::nulls_at({0, 6, 13})});
    auto const last_element = tester.test_nth_element(-1);
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*last_element,
                                        fwcw<T>{{X, 2, 3, 4, X, 10, 11, 12, 13, 14, 15, 16, 20, X},
                                                cudf::test::iterators::nulls_at({0, 4, 13})});
    auto const second_element = tester.test_nth_element(1);
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*second_element,
                                        fwcw<T>{{X, 1, 2, 3, 4, X, 10, 11, 12, 13, 14, 15, 16, X},
                                                cudf::test::iterators::nulls_at({0, 5, 13})});
    auto const second_last_element = tester.test_nth_element(-2);
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*second_last_element,
                                        fwcw<T>{{X, 1, 2, 3, 4, X, 10, 11, 12, 13, 14, 15, 16, X},
                                                cudf::test::iterators::nulls_at({0, 5, 13})});
  }
  {
    // Too large values for `min_periods`. No window has enough periods.
    tester.preceding(2).following(1).min_periods(4);
    auto const all_null_values =
      fwcw<T>{{X, X, X, X, X, X, X, X, X, X, X, X, X, X}, cudf::test::iterators::all_nulls()};
    auto const first_element = tester.test_nth_element(0);
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*first_element, all_null_values);
    auto const last_element = tester.test_nth_element(-1);
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*last_element, all_null_values);
    auto const second_element = tester.test_nth_element(1);
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*second_element, all_null_values);
    auto const second_last_element = tester.test_nth_element(-2);
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*second_last_element, all_null_values);
  }
}

TYPED_TEST(NthElementTypedTest, RollingWindowExcludeNulls)
{
  using T = TypeParam;

  auto const input_col =
    fwcw<T>{{0, X, X, X, 4, X, 6, 7}, cudf::test::iterators::nulls_at({1, 2, 3, 5})};
  auto tester = rolling_exec{}.input(input_col);

  {
    // Window of 5 elements, min-periods == 2.
    tester.preceding(3).following(2).min_periods(1).null_handling(cudf::null_policy::EXCLUDE);

    auto const first_element = tester.test_nth_element(0);
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(
      *first_element, fwcw<T>{{0, 0, 0, 4, 4, 4, 4, 6}, cudf::test::iterators::no_nulls()});
    auto const last_element = tester.test_nth_element(-1);
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(
      *last_element, fwcw<T>{{0, 0, 4, 4, 6, 7, 7, 7}, cudf::test::iterators::no_nulls()});
    auto const second_element = tester.test_nth_element(1);
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(
      *second_element,
      fwcw<T>{{X, X, 4, X, 6, 6, 6, 7}, cudf::test::iterators::nulls_at({0, 1, 3})});
    auto const second_last_element = tester.test_nth_element(-2);
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(
      *second_last_element,
      fwcw<T>{{X, X, 0, X, 4, 6, 6, 6}, cudf::test::iterators::nulls_at({0, 1, 3})});
  }
  {
    // Window of 3 elements, min-periods == 1.
    tester.preceding(2).following(1).min_periods(1).null_handling(cudf::null_policy::EXCLUDE);

    auto const first_element = tester.test_nth_element(0);
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(
      *first_element, fwcw<T>{{0, 0, X, 4, 4, 4, 6, 6}, cudf::test::iterators::null_at(2)});
    auto const last_element = tester.test_nth_element(-1);
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(
      *last_element, fwcw<T>{{0, 0, X, 4, 4, 6, 7, 7}, cudf::test::iterators::null_at(2)});
    auto const second_element = tester.test_nth_element(1);
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(
      *second_element,
      fwcw<T>{{X, X, X, X, X, 6, 7, 7}, cudf::test::iterators::nulls_at({0, 1, 2, 3, 4})});
    auto const second_last_element = tester.test_nth_element(-2);
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(
      *second_last_element,
      fwcw<T>{{X, X, X, X, X, 4, 6, 6}, cudf::test::iterators::nulls_at({0, 1, 2, 3, 4})});
  }
  {
    // Too large values for `min_periods`. No window has enough periods.
    tester.preceding(2).following(1).min_periods(4);
    auto const all_null_values =
      fwcw<T>{{X, X, X, X, X, X, X, X}, cudf::test::iterators::all_nulls()};

    auto const first_element = tester.test_nth_element(0);
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*first_element, all_null_values);
    auto const last_element = tester.test_nth_element(-1);
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*last_element, all_null_values);
    auto const second_element = tester.test_nth_element(1);
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*second_element, all_null_values);
    auto const second_last_element = tester.test_nth_element(-2);
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*second_last_element, all_null_values);
  }
}

TYPED_TEST(NthElementTypedTest, GroupedRollingWindow)
{
  using T = TypeParam;

  // clang-format off
  auto const group_col = fwcw<int32_t>{0, 0, 0, 0, 0, 0,
                                       10, 10, 10, 10, 10, 10, 10,
                                       20};
  auto const input_col = fwcw<T> {0, 1, 2, 3, 4, 5,           // Group 0
                                  10, 11, 12, 13, 14, 15, 16, // Group 10
                                  20};                        // Group 20
  // clang-format on
  auto tester = rolling_exec{}.grouping(group_col).input(input_col);

  {
    // Window of 5 elements, min-periods == 1.
    tester.preceding(3).following(2).min_periods(1);
    auto const first_element = tester.test_grouped_nth_element(0);
    // clang-format off
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*first_element,
                                        fwcw<T>{{0, 0, 0, 1, 2, 3,           // Group 0
                                                 10, 10, 10, 11, 12, 13, 14, // Group 10
                                                 20},                        // Group 20
                                                cudf::test::iterators::no_nulls()});
    auto const last_element = tester.test_grouped_nth_element(-1);
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*last_element,
                                        fwcw<T>{{2, 3, 4, 5, 5, 5,           // Group 0
                                                 12, 13, 14, 15, 16, 16, 16, // Group 10
                                                 20},                        // Group 20
                                                cudf::test::iterators::no_nulls()});
    auto const third_element = tester.test_grouped_nth_element(2);
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*third_element,
                                        fwcw<T>{{2, 2, 2, 3, 4, 5,           // Group 0
                                                 12, 12, 12, 13, 14, 15, 16, // Group 10
                                                 X},                         // Group 20
                                                cudf::test::iterators::null_at(13)});
    auto const second_last_element = tester.test_grouped_nth_element(-2);
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*second_last_element,
                                        fwcw<T>{{1, 2, 3, 4, 4, 4,           // Group 0
                                                 11, 12, 13, 14, 15, 15, 15, // Group 10
                                                 X},                         // Group 20
                                                cudf::test::iterators::null_at(13)});
    // clang-format on
  }
  {
    // Window of 3 elements, min-periods == 3. Expect null elements at group margins.
    tester.preceding(2).following(1).min_periods(3);
    auto const first_element = tester.test_grouped_nth_element(0);
    // clang-format off
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*first_element,
                                        fwcw<T>{{X, 0, 1, 2, 3, X,         // Group 0
                                                 X, 10, 11, 12, 13, 14, X, // Group 10
                                                 X},                       // Group 20
                                                cudf::test::iterators::nulls_at({0, 5, 6, 12, 13})});
    auto const last_element = tester.test_grouped_nth_element(-1);
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*last_element,
                                        fwcw<T>{{X, 2, 3, 4, 5, X,         // Group 0
                                                 X, 12, 13, 14, 15, 16, X, // Group 10
                                                 X},                       // Group 20
                                                cudf::test::iterators::nulls_at({0, 5, 6, 12, 13})});
    auto const second_element = tester.test_grouped_nth_element(1);
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*second_element,
                                        fwcw<T>{{X, 1, 2, 3, 4, X,         // Group 0
                                                 X, 11, 12, 13, 14, 15, X, // Group 10
                                                 X},                       // Group 20
                                                cudf::test::iterators::nulls_at({0, 5, 6, 12, 13})});
    auto const second_last_element = tester.test_grouped_nth_element(-2);
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*second_last_element,
                                        fwcw<T>{{X, 1, 2, 3, 4, X,         // Group 0
                                                 X, 11, 12, 13, 14, 15, X, // Group 10
                                                 X},                       // Group 20
                                                cudf::test::iterators::nulls_at({0, 5, 6, 12, 13})});
    // clang-format on
  }
  {
    // Too large values for `min_periods`. No window has enough periods.
    tester.preceding(2).following(1).min_periods(4);
    auto const all_null_values =
      fwcw<T>{{X, X, X, X, X, X, X, X, X, X, X, X, X, X}, cudf::test::iterators::all_nulls()};

    auto const first_element = tester.test_grouped_nth_element(0);
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*first_element, all_null_values);
    auto const last_element = tester.test_grouped_nth_element(-1);
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*last_element, all_null_values);
    auto const second_element = tester.test_grouped_nth_element(1);
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*second_element, all_null_values);
    auto const second_last_element = tester.test_grouped_nth_element(-2);
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*second_last_element, all_null_values);
  }
}

TYPED_TEST(NthElementTypedTest, GroupedRollingWindowExcludeNulls)
{
  using T = TypeParam;

  // clang-format off
  auto const group_col = fwcw<int32_t>{0, 0, 0, 0, 0, 0,
                                       10, 10, 10, 10, 10, 10, 10,
                                       20,
                                       30};
  auto const input_col = fwcw<T> {{0, 1, X, 3, X, 5,         // Group 0
                                   10, X, X, 13, 14, 15, 16, // Group 10
                                   20,                       // Group 20
                                   X},                       // Group 30
                                  cudf::test::iterators::nulls_at({2, 4, 7, 8, 14})};
  // clang-format on
  auto tester = rolling_exec{}.grouping(group_col).input(input_col);

  {
    // Window of 5 elements, min-periods == 1.
    tester.preceding(3).following(2).min_periods(1).null_handling(cudf::null_policy::EXCLUDE);
    auto const first_element = tester.test_grouped_nth_element(0);
    // clang-format off
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*first_element,
                                        fwcw<T>{{0, 0, 0, 1, 3, 3,           // Group 0
                                                 10, 10, 10, 13, 13, 13, 14, // Group 10
                                                 20,                         // Group 20
                                                 X},                         // Group 30
                                                cudf::test::iterators::null_at(14)});
    auto const last_element = tester.test_grouped_nth_element(-1);
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*last_element,
                                        fwcw<T>{{1, 3, 3, 5, 5, 5,           // Group 0
                                                 10, 13, 14, 15, 16, 16, 16, // Group 10
                                                 20,                         // Group 20
                                                 X},                         // Group 30
                                                cudf::test::iterators::null_at(14)});
    auto const third_element = tester.test_grouped_nth_element(2);
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*third_element,
                                        fwcw<T>{{X, 3, 3, 5, X, X,          // Group 0
                                                 X, X, 14, 15, 15, 15, 16,  // Group 10
                                                 X,                         // Group 20
                                                 X},                        // Group 30
                                                cudf::test::iterators::nulls_at({0, 4, 5, 6, 7, 13, 14})});
    auto const second_last_element = tester.test_grouped_nth_element(-2);
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*second_last_element,
                                        fwcw<T>{{0, 1, 1, 3, 3, 3,          // Group 0
                                                 X, 10, 13, 14, 15, 15, 15, // Group 10
                                                 X,                         // Group 20
                                                 X},                        // Group 30
                                                cudf::test::iterators::nulls_at({6, 13, 14})});
    // clang-format on
  }
  {
    // Window of 3 elements, min-periods == 3. Expect null elements at group margins.
    tester.preceding(2).following(1).min_periods(3);
    auto const first_element = tester.test_grouped_nth_element(0);
    // clang-format off
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*first_element,
                                        fwcw<T>{{X, 0, 1, 3, 3, X,         // Group 0
                                                 X, 10, 13, 13, 13, 14, X, // Group 10
                                                 X,                        // Group 20
                                                 X},                       // Group 30
                                                cudf::test::iterators::nulls_at({0, 5, 6, 12, 13, 14})});
    auto const last_element = tester.test_grouped_nth_element(-1);
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*last_element,
                                        fwcw<T>{{X, 1, 3, 3, 5, X,         // Group 0
                                                 X, 10, 13, 14, 15, 16, X, // Group 10
                                                 X,                        // Group 20
                                                 X},                       // Group 30
                                                cudf::test::iterators::nulls_at({0, 5, 6, 12, 13, 14})});
    auto const second_element = tester.test_grouped_nth_element(1);
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*second_element,
                                        fwcw<T>{{X, 1, 3, X, 5, X,       // Group 0
                                                 X, X, X, 14, 14, 15, X, // Group 10
                                                 X,                      // Group 20
                                                 X},                     // Group 30
                                                cudf::test::iterators::nulls_at({0, 3, 5, 6, 7, 8, 12, 13, 14})});
    auto const second_last_element = tester.test_grouped_nth_element(-2);
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*second_last_element,
                                        fwcw<T>{{X, 0, 1, X, 3, X,       // Group 0
                                                 X, X, X, 13, 14, 15, X, // Group 10
                                                 X,                      // Group 20
                                                 X},                     // Group 30
                                                cudf::test::iterators::nulls_at({0, 3, 5, 6, 7, 8, 12, 13, 14})});
    // clang-format on
  }
  {
    // Too large values for `min_periods`. No window has enough periods.
    tester.preceding(2).following(1).min_periods(4);
    auto const all_null_values =
      fwcw<T>{{X, X, X, X, X, X, X, X, X, X, X, X, X, X, X}, cudf::test::iterators::all_nulls()};

    auto const first_element = tester.test_grouped_nth_element(0);
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*first_element, all_null_values);
    auto const last_element = tester.test_grouped_nth_element(-1);
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*last_element, all_null_values);
    auto const second_element = tester.test_grouped_nth_element(1);
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*second_element, all_null_values);
    auto const second_last_element = tester.test_grouped_nth_element(-2);
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*second_last_element, all_null_values);
  }
}

TYPED_TEST(NthElementTypedTest, EmptyInput)
{
  using T = TypeParam;

  auto const group_col = fwcw<int32_t>{};
  auto const input_col = fwcw<T>{};
  auto tester = rolling_exec{}.grouping(group_col).input(input_col).preceding(3).following(1);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*tester.test_grouped_nth_element(0), fwcw<T>{});
}

TEST_F(NthElementTest, RollingWindowOnStrings)
{
  using strings = cudf::test::strings_column_wrapper;

  auto constexpr X = "";  // Placeholder for null string.

  auto const input_col =
    strings{{"", "1", "22", "333", "4444", "", "10", "11", "12", "13", "14", "15", "16", "20"},
            cudf::test::iterators::null_at(5)};
  auto tester = rolling_exec{}.input(input_col);

  {
    // Window of 5 elements, min-periods == 1.
    tester.preceding(3).following(2).min_periods(1);

    auto const first_element = tester.test_nth_element(0);
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(
      *first_element,
      strings{{"", "", "", "1", "22", "333", "4444", X, "10", "11", "12", "13", "14", "15"},
              cudf::test::iterators::null_at(7)});
    auto const last_element = tester.test_nth_element(-1);
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(
      *last_element,
      strings{{"22", "333", "4444", X, "10", "11", "12", "13", "14", "15", "16", "20", "20", "20"},
              cudf::test::iterators::null_at(3)});
    auto const third_element = tester.test_nth_element(2);
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(
      *third_element,
      strings{{"22", "22", "22", "333", "4444", X, "10", "11", "12", "13", "14", "15", "16", "20"},
              cudf::test::iterators::null_at(5)});
    auto const second_last_element = tester.test_nth_element(-2);
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(
      *second_last_element,
      strings{{"1", "22", "333", "4444", X, "10", "11", "12", "13", "14", "15", "16", "16", "16"},
              cudf::test::iterators::null_at(4)});
  }
  {
    // Window of 3 elements, min-periods == 3. Expect null elements at column margins.
    tester.preceding(2).following(1).min_periods(3);

    auto const first_element = tester.test_nth_element(0);
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(
      *first_element,
      strings{{X, "", "1", "22", "333", "4444", X, "10", "11", "12", "13", "14", "15", X},
              cudf::test::iterators::nulls_at({0, 6, 13})});
    auto const last_element = tester.test_nth_element(-1);
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(
      *last_element,
      strings{{X, "22", "333", "4444", X, "10", "11", "12", "13", "14", "15", "16", "20", X},
              cudf::test::iterators::nulls_at({0, 4, 13})});
    auto const second_element = tester.test_nth_element(1);
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(
      *second_element,
      strings{{X, "1", "22", "333", "4444", X, "10", "11", "12", "13", "14", "15", "16", X},
              cudf::test::iterators::nulls_at({0, 5, 13})});
    auto const second_last_element = tester.test_nth_element(-2);
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(
      *second_last_element,
      strings{{X, "1", "22", "333", "4444", X, "10", "11", "12", "13", "14", "15", "16", X},
              cudf::test::iterators::nulls_at({0, 5, 13})});
  }
  {
    // Too large values for `min_periods`. No window has enough periods.
    tester.preceding(2).following(1).min_periods(4);
    auto const all_null_values =
      strings{{X, X, X, X, X, X, X, X, X, X, X, X, X, X}, cudf::test::iterators::all_nulls()};

    auto const first_element = tester.test_nth_element(0);
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*first_element, all_null_values);
    auto const last_element = tester.test_nth_element(-1);
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*last_element, all_null_values);
    auto const second_element = tester.test_nth_element(1);
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*second_element, all_null_values);
    auto const second_last_element = tester.test_nth_element(-2);
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*second_last_element, all_null_values);
  }
}

TEST_F(NthElementTest, GroupedRollingWindowForStrings)
{
  using strings    = cudf::test::strings_column_wrapper;
  auto constexpr X = "";  // Placeholder for null strings.

  // clang-format off
  auto const group_col = fwcw<int32_t>{0, 0, 0, 0, 0, 0,
                                       10, 10, 10, 10, 10, 10, 10,
                                       20};
  auto const input_col = strings{{"", "1", "22", "333", "4444", X,          // Group 0
                                  "10", "11", "12", "13", "14", "15", "16", // Group 10
                                  "20"},                                    // Group 20
                                 cudf::test::iterators::null_at(5)};
  // clang-format on
  auto tester = rolling_exec{}.grouping(group_col).input(input_col);

  {
    // Window of 5 elements, min-periods == 1.
    tester.preceding(3).following(2).min_periods(1);

    auto const first_element = tester.test_grouped_nth_element(0);
    // clang-format off
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(
      *first_element,
      strings{{"", "", "", "1", "22", "333",             // Group 0
               "10", "10", "10", "11", "12", "13", "14", // Group 10
               "20"},                                    // Group 20
              cudf::test::iterators::no_nulls()});
    auto const last_element = tester.test_grouped_nth_element(-1);
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(
      *last_element,
      strings{{"22", "333", "4444", X, X, X,             // Group 0
               "12", "13", "14", "15", "16", "16", "16", // Group 10
               "20"},                                    // Group 20
              cudf::test::iterators::nulls_at({3, 4, 5})});
    auto const third_element = tester.test_grouped_nth_element(2);
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(
      *third_element,
      strings{{"22", "22", "22", "333", "4444", X,       // Group 0
               "12", "12", "12", "13", "14", "15", "16", // Group 10
               X},                                       // Group 20
              cudf::test::iterators::nulls_at({5, 13})});
    auto const second_last_element = tester.test_grouped_nth_element(-2);
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(
      *second_last_element,
      strings{{"1", "22", "333", "4444", "4444", "4444", // Group 0
               "11", "12", "13", "14", "15", "15", "15", // Group 10
               X},                                       // Group 20
              cudf::test::iterators::null_at(13)});
    // clang-format on
  }
  {
    // Window of 3 elements, min-periods == 3. Expect null elements at group margins.
    tester.preceding(2).following(1).min_periods(3);
    auto const first_element = tester.test_grouped_nth_element(0);
    // clang-format off
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(
      *first_element,
      strings{{X, "", "1", "22", "333", X,         // Group 0
               X, "10", "11", "12", "13", "14", X, // Group 10
               X},                                 // Group 20
              cudf::test::iterators::nulls_at({0, 5, 6, 12, 13})});
    auto const last_element = tester.test_grouped_nth_element(-1);
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(
      *last_element,
      strings{{X, "22", "333", "4444", X, X,       // Group 0
               X, "12", "13", "14", "15", "16", X, // Group 10
               X},                                 // Group 20
              cudf::test::iterators::nulls_at({0, 4, 5, 6, 12, 13})});
    auto const second_element = tester.test_grouped_nth_element(1);
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(
      *second_element,
      strings{{X, "1", "22", "333", "4444", X,     // Group 0
               X, "11", "12", "13", "14", "15", X, // Group 10
               X},                                 // Group 20
              cudf::test::iterators::nulls_at({0, 5, 6, 12, 13})});
    auto const second_last_element = tester.test_grouped_nth_element(-2);
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(
      *second_last_element,
      strings{{X, "1", "22", "333", "4444", X,     // Group 0
               X, "11", "12", "13", "14", "15", X, // Group 10
               X},                                 // Group 20
              cudf::test::iterators::nulls_at({0, 5, 6, 12, 13})});
    // clang-format on
  }
  {
    // Too large values for `min_periods`. No window has enough periods.
    tester.preceding(2).following(1).min_periods(4);
    auto const all_null_strings =
      strings{{X, X, X, X, X, X, X, X, X, X, X, X, X, X}, cudf::test::iterators::all_nulls()};

    auto const first_element = tester.test_grouped_nth_element(0);
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*first_element, all_null_strings);
    auto const last_element = tester.test_grouped_nth_element(-1);
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*last_element, all_null_strings);
    auto const second_element = tester.test_grouped_nth_element(1);
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*second_element, all_null_strings);
    auto const second_last_element = tester.test_grouped_nth_element(-2);
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*second_last_element, all_null_strings);
  }
}
