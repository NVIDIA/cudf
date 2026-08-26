/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/iterator_utilities.hpp>
#include <cudf_test/lists_column_wrapper.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/lists/filling.hpp>
#include <cudf/utilities/error.hpp>

using namespace cudf::test::iterators;

namespace {
template <typename T, typename U = int32_t>
using ListsCol = cudf::test::lists_column_wrapper<T, U>;

template <typename T, typename U = int32_t>
using FWDCol = cudf::test::fixed_width_column_wrapper<T, U>;

using IntsCol = cudf::test::fixed_width_column_wrapper<int32_t>;
}  // namespace

/*-----------------------------------------------------------------------------------------------*/
template <typename T>
class NumericSequencesTypedTest : public cudf::test::BaseFixture {};
using NumericTypes =
  cudf::test::Concat<cudf::test::IntegralTypesNotBool, cudf::test::FloatingPointTypes>;
TYPED_TEST_SUITE(NumericSequencesTypedTest, NumericTypes);

TYPED_TEST(NumericSequencesTypedTest, SimpleTestNoNull)
{
  using T = TypeParam;

  auto const starts = FWDCol<T>{1, 2, 3};
  auto const sizes  = IntsCol{5, 3, 4};

  // Sequences with step == 1.
  {
    auto const expected =
      ListsCol<T>{ListsCol<T>{1, 2, 3, 4, 5}, ListsCol<T>{2, 3, 4}, ListsCol<T>{3, 4, 5, 6}};
    auto const result = cudf::lists::sequences(starts, sizes);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *result);
  }

  // Sequences with various steps.
  {
    auto const steps = FWDCol<T>{1, 3, 2};
    auto const expected =
      ListsCol<T>{ListsCol<T>{1, 2, 3, 4, 5}, ListsCol<T>{2, 5, 8}, ListsCol<T>{3, 5, 7, 9}};
    auto const result = cudf::lists::sequences(starts, steps, sizes);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *result);
  }
}

TYPED_TEST(NumericSequencesTypedTest, ZeroSizesTest)
{
  using T = TypeParam;

  auto const starts = FWDCol<T>{1, 2, 3};
  auto const sizes  = IntsCol{0, 3, 0};

  // Sequences with step == 1.
  {
    auto const expected = ListsCol<T>{ListsCol<T>{}, ListsCol<T>{2, 3, 4}, ListsCol<T>{}};
    auto const result   = cudf::lists::sequences(starts, sizes);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *result);
  }

  // Sequences with various steps.
  {
    auto const steps    = FWDCol<T>{1, 3, 2};
    auto const expected = ListsCol<T>{ListsCol<T>{}, ListsCol<T>{2, 5, 8}, ListsCol<T>{}};
    auto const result   = cudf::lists::sequences(starts, steps, sizes);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *result);
  }
}

TYPED_TEST(NumericSequencesTypedTest, SlicedInputTestNoNulls)
{
  using T = TypeParam;
  constexpr int32_t dont_care{123};

  auto const starts_original =
    FWDCol<T>{dont_care, dont_care, dont_care, 1, 2, 3, 4, 5, dont_care, dont_care};
  auto const sizes_original = IntsCol{dont_care, 5, 3, 4, 1, 2, dont_care, dont_care};

  auto const starts = cudf::slice(starts_original, {3, 8})[0];
  auto const sizes  = cudf::slice(sizes_original, {1, 6})[0];

  // Sequences with step == 1.
  {
    auto const expected = ListsCol<T>{ListsCol<T>{1, 2, 3, 4, 5},
                                      ListsCol<T>{2, 3, 4},
                                      ListsCol<T>{3, 4, 5, 6},
                                      ListsCol<T>{4},
                                      ListsCol<T>{5, 6}

    };
    auto const result   = cudf::lists::sequences(starts, sizes);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *result);
  }

  // Sequences with various steps.
  {
    auto const steps_original = FWDCol<T>{dont_care, dont_care, 1, 3, 2, 2, 3, dont_care};
    auto const steps          = cudf::slice(steps_original, {2, 7})[0];

    auto const expected = ListsCol<T>{ListsCol<T>{1, 2, 3, 4, 5},
                                      ListsCol<T>{2, 5, 8},
                                      ListsCol<T>{3, 5, 7, 9},
                                      ListsCol<T>{4},
                                      ListsCol<T>{5, 8}

    };
    auto const result   = cudf::lists::sequences(starts, steps, sizes);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *result);
  }
}

/*-----------------------------------------------------------------------------------------------*/
// Data generated using https://www.epochconverter.com/
template <typename T>
class DurationSequencesTypedTest : public cudf::test::BaseFixture {};
TYPED_TEST_SUITE(DurationSequencesTypedTest, cudf::test::DurationTypes);

// Start time is 1638477473L - Thursday, December 2, 2021 8:37:53 PM.
constexpr int64_t start_time = 1638477473L;

TYPED_TEST(DurationSequencesTypedTest, SequencesNoNull)
{
  using T = TypeParam;

  auto const starts = FWDCol<T, int64_t>{start_time, start_time, start_time};
  auto const sizes  = IntsCol{1, 2, 3};

  // Sequences with step == 1.
  {
    auto const expected_h = std::vector<int64_t>{start_time, start_time + 1L, start_time + 2L};
    auto const expected =
      ListsCol<T, int64_t>{ListsCol<T, int64_t>{expected_h.begin(), expected_h.begin() + 1},
                           ListsCol<T, int64_t>{expected_h.begin(), expected_h.begin() + 2},
                           ListsCol<T, int64_t>{expected_h.begin(), expected_h.begin() + 3}};
    auto const result = cudf::lists::sequences(starts, sizes);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *result);
  }

  // Sequences with various steps, including negative.
  {
    auto const steps    = FWDCol<T, int64_t>{10L, -155L, -13L};
    auto const expected = ListsCol<T, int64_t>{
      ListsCol<T, int64_t>{start_time},
      ListsCol<T, int64_t>{start_time, start_time - 155L},
      ListsCol<T, int64_t>{start_time, start_time - 13L, start_time - 13L * 2L}};
    auto const result = cudf::lists::sequences(starts, steps, sizes);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *result);
  }
}

/*-----------------------------------------------------------------------------------------------*/
class NumericSequencesTest : public cudf::test::BaseFixture {};

TEST_F(NumericSequencesTest, EmptyInput)
{
  auto const starts   = IntsCol{};
  auto const sizes    = IntsCol{};
  auto const steps    = IntsCol{};
  auto const expected = ListsCol<int32_t>{};

  // Sequences with step == 1.
  {
    auto const result = cudf::lists::sequences(starts, sizes);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *result);
  }

  // Sequences with given steps.
  {
    auto const result = cudf::lists::sequences(starts, steps, sizes);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *result);
  }
}

TEST_F(NumericSequencesTest, InvalidSizesInput)
{
  auto const starts = IntsCol{};
  auto const steps  = IntsCol{};
  auto const sizes  = FWDCol<float>{};

  EXPECT_THROW(cudf::lists::sequences(starts, sizes), cudf::data_type_error);
  EXPECT_THROW(cudf::lists::sequences(starts, steps, sizes), cudf::data_type_error);
}

TEST_F(NumericSequencesTest, MismatchedColumnSizesInput)
{
  auto const starts = IntsCol{1, 2, 3};
  auto const steps  = IntsCol{1, 2};
  auto const sizes  = IntsCol{1, 2, 3, 4};

  EXPECT_THROW(cudf::lists::sequences(starts, sizes), cudf::logic_error);
  EXPECT_THROW(cudf::lists::sequences(starts, steps, sizes), cudf::logic_error);
}

TEST_F(NumericSequencesTest, MismatchedColumnTypesInput)
{
  auto const starts = IntsCol{1, 2, 3};
  auto const steps  = FWDCol<float>{1, 2, 3};
  auto const sizes  = IntsCol{1, 2, 3};

  EXPECT_THROW(cudf::lists::sequences(starts, steps, sizes), cudf::data_type_error);
}

TEST_F(NumericSequencesTest, InputHasNulls)
{
  constexpr int32_t null{0};

  {
    auto const starts = IntsCol{{null, 2, 3}, null_at(0)};
    auto const sizes  = IntsCol{1, 2, 3};
    EXPECT_THROW(cudf::lists::sequences(starts, sizes), cudf::logic_error);
  }

  {
    auto const starts = IntsCol{1, 2, 3};
    auto const sizes  = IntsCol{{null, 2, 3}, null_at(0)};
    EXPECT_THROW(cudf::lists::sequences(starts, sizes), cudf::logic_error);
  }

  {
    auto const starts = IntsCol{1, 2, 3};
    auto const steps  = IntsCol{{null, 2, 3}, null_at(0)};
    auto const sizes  = IntsCol{1, 2, 3};
    EXPECT_THROW(cudf::lists::sequences(starts, steps, sizes), cudf::logic_error);
  }
}
