/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/table_utilities.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/column/column_view.hpp>
#include <cudf/quantiles.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>

#include <stdexcept>

template <typename T>
struct QuantilesTest : public cudf::test::BaseFixture {};

using TestTypes = cudf::test::AllTypes;

TYPED_TEST_SUITE(QuantilesTest, TestTypes);

TYPED_TEST(QuantilesTest, TestZeroColumns)
{
  auto input = cudf::table_view(std::vector<cudf::column_view>{});

  EXPECT_THROW(cudf::quantiles(input, {0.0f}), cudf::logic_error);
}

TYPED_TEST(QuantilesTest, TestMultiColumnZeroRows)
{
  using T = TypeParam;

  cudf::test::fixed_width_column_wrapper<T> input_a({});
  auto input = cudf::table_view({input_a});

  EXPECT_THROW(cudf::quantiles(input, {0.0f}), cudf::logic_error);
}

TYPED_TEST(QuantilesTest, TestZeroRequestedQuantiles)
{
  using T = TypeParam;

  cudf::test::fixed_width_column_wrapper<T, int32_t> input_a({1}, {1});
  auto input = cudf::table_view(std::vector<cudf::column_view>{input_a});

  auto actual   = cudf::quantiles(input, {});
  auto expected = cudf::empty_like(input);

  CUDF_TEST_EXPECT_TABLES_EQUAL(expected->view(), actual->view());
}

TYPED_TEST(QuantilesTest, TestMultiColumnOrderCountMismatch)
{
  using T = TypeParam;

  cudf::test::fixed_width_column_wrapper<T> input_a({});
  cudf::test::fixed_width_column_wrapper<T> input_b({});
  auto input = cudf::table_view({input_a});

  EXPECT_THROW(cudf::quantiles(input,
                               {0.0f},
                               cudf::interpolation::NEAREST,
                               cudf::sorted::NO,
                               {cudf::order::ASCENDING},
                               {cudf::null_order::AFTER, cudf::null_order::AFTER}),
               cudf::logic_error);
}

TYPED_TEST(QuantilesTest, TestMultiColumnNullOrderCountMismatch)
{
  using T = TypeParam;

  cudf::test::fixed_width_column_wrapper<T> input_a({});
  cudf::test::fixed_width_column_wrapper<T> input_b({});
  auto input = cudf::table_view({input_a});

  EXPECT_THROW(cudf::quantiles(input,
                               {0.0f},
                               cudf::interpolation::NEAREST,
                               cudf::sorted::NO,
                               {cudf::order::ASCENDING, cudf::order::ASCENDING},
                               {cudf::null_order::AFTER}),
               cudf::logic_error);
}

TYPED_TEST(QuantilesTest, TestMultiColumnArithmeticInterpolation)
{
  using T = TypeParam;

  cudf::test::fixed_width_column_wrapper<T> input_a({});
  cudf::test::fixed_width_column_wrapper<T> input_b({});
  auto input = cudf::table_view({input_a});

  EXPECT_THROW(cudf::quantiles(input, {0.0f}, cudf::interpolation::LINEAR), std::invalid_argument);

  EXPECT_THROW(cudf::quantiles(input, {0.0f}, cudf::interpolation::MIDPOINT),
               std::invalid_argument);
}

TYPED_TEST(QuantilesTest, TestMultiColumnUnsorted)
{
  using T = TypeParam;

  auto input_a = cudf::test::strings_column_wrapper(
    {"C", "B", "A", "A", "D", "B", "D", "B", "D", "C", "C", "C",
     "D", "B", "D", "B", "C", "C", "A", "D", "B", "A", "A", "A"},
    {true, true, true, true, true, true, true, true, true, true, true, true,
     true, true, true, true, true, true, true, true, true, true, true, true});

  cudf::test::fixed_width_column_wrapper<T, int32_t> input_b(
    {4, 3, 5, 0, 1, 0, 4, 1, 5, 3, 0, 5, 2, 4, 3, 2, 1, 2, 3, 0, 5, 1, 4, 2},
    {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1});

  auto input = cudf::table_view({input_a, input_b});

  auto actual = cudf::quantiles(input,
                                {0.0f, 0.5f, 0.7f, 0.25f, 1.0f},
                                cudf::interpolation::NEAREST,
                                cudf::sorted::NO,
                                {cudf::order::ASCENDING, cudf::order::DESCENDING});

  auto expected_a =
    cudf::test::strings_column_wrapper({"A", "C", "C", "B", "D"}, {true, true, true, true, true});

  cudf::test::fixed_width_column_wrapper<T, int32_t> expected_b({5, 5, 1, 5, 0}, {1, 1, 1, 1, 1});

  auto expected = cudf::table_view({expected_a, expected_b});

  CUDF_TEST_EXPECT_TABLES_EQUAL(expected, actual->view());
}

TYPED_TEST(QuantilesTest, TestMultiColumnAssumedSorted)
{
  using T = TypeParam;

  auto input_a = cudf::test::strings_column_wrapper(
    {"C", "B", "A", "A", "D", "B", "D", "B", "D", "C", "C", "C",
     "D", "B", "D", "B", "C", "C", "A", "D", "B", "A", "A", "A"},
    {true, true, true, true, true, true, true, true, true, true, true, true,
     true, true, true, true, true, true, true, true, true, true, true, true});

  cudf::test::fixed_width_column_wrapper<T, int32_t> input_b(
    {4, 3, 5, 0, 1, 0, 4, 1, 5, 3, 0, 5, 2, 4, 3, 2, 1, 2, 3, 0, 5, 1, 4, 2},
    {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1});

  auto input = cudf::table_view({input_a, input_b});

  auto actual = cudf::quantiles(
    input, {0.0f, 0.5f, 0.7f, 0.25f, 1.0f}, cudf::interpolation::NEAREST, cudf::sorted::YES);

  auto expected_a =
    cudf::test::strings_column_wrapper({"C", "D", "C", "D", "A"}, {true, true, true, true, true});

  cudf::test::fixed_width_column_wrapper<T, int32_t> expected_b({4, 2, 1, 4, 2}, {1, 1, 1, 1, 1});

  auto expected = cudf::table_view({expected_a, expected_b});

  CUDF_TEST_EXPECT_TABLES_EQUAL(expected, actual->view());
}
