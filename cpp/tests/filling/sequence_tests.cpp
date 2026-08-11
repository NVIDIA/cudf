/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/filling.hpp>
#include <cudf/scalar/scalar.hpp>

template <typename T>
class SequenceTypedTestFixture : public cudf::test::BaseFixture {};

class SequenceTestFixture : public cudf::test::BaseFixture {};

using NumericTypesNoBool = cudf::test::Types<int8_t, int16_t, int32_t, int64_t, float, double>;

TYPED_TEST_SUITE(SequenceTypedTestFixture, NumericTypesNoBool);

TYPED_TEST(SequenceTypedTestFixture, Incrementing)
{
  using T = TypeParam;

  cudf::numeric_scalar<T> init(0);
  cudf::numeric_scalar<T> step(1);

  cudf::size_type num_els = 10;

  cudf::test::fixed_width_column_wrapper<T> expected_w({0, 1, 2, 3, 4, 5, 6, 7, 8, 9});

  auto result = cudf::sequence(num_els, init, step);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result, expected_w);
}

TYPED_TEST(SequenceTypedTestFixture, Decrementing)
{
  using T = TypeParam;

  cudf::numeric_scalar<T> init(0);
  cudf::numeric_scalar<T> step(-5);

  cudf::size_type num_els = 10;

  cudf::test::fixed_width_column_wrapper<T> expected_w(
    {0, -5, -10, -15, -20, -25, -30, -35, -40, -45});

  auto result = cudf::sequence(num_els, init, step);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result, expected_w);
}

TYPED_TEST(SequenceTypedTestFixture, EmptyOutput)
{
  using T = TypeParam;

  cudf::numeric_scalar<T> init(0);
  cudf::numeric_scalar<T> step(-5);

  cudf::size_type num_els = 0;

  cudf::test::fixed_width_column_wrapper<T> expected_w({});

  auto result = cudf::sequence(num_els, init, step);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result, expected_w);
}

TEST_F(SequenceTestFixture, BadTypes)
{
  cudf::string_scalar string_init("zero");
  cudf::string_scalar string_step("???");
  EXPECT_THROW(cudf::sequence(10, string_init, string_step), std::invalid_argument);

  cudf::numeric_scalar<bool> bool_init(true);
  cudf::numeric_scalar<bool> bool_step(false);
  EXPECT_THROW(cudf::sequence(10, bool_init, bool_step), cudf::data_type_error);

  cudf::timestamp_scalar<cudf::timestamp_s> ts_init(cudf::duration_s{10}, true);
  cudf::timestamp_scalar<cudf::timestamp_s> ts_step(cudf::duration_s{10}, true);
  EXPECT_THROW(cudf::sequence(10, ts_init, ts_step), std::invalid_argument);
}

TEST_F(SequenceTestFixture, MismatchedInputs)
{
  cudf::numeric_scalar<int> init(0);
  cudf::numeric_scalar<float> step(-5);
  EXPECT_THROW(cudf::sequence(10, init, step), cudf::data_type_error);

  cudf::numeric_scalar<int> init2(0);
  cudf::numeric_scalar<int8_t> step2(-5);
  EXPECT_THROW(cudf::sequence(10, init2, step2), cudf::data_type_error);

  cudf::numeric_scalar<float> init3(0);
  cudf::numeric_scalar<double> step3(-5);
  EXPECT_THROW(cudf::sequence(10, init3, step3), cudf::data_type_error);
}

TEST_F(SequenceTestFixture, InvalidInput)
{
  cudf::numeric_scalar<int> init(0, false);
  EXPECT_THROW(cudf::sequence(10, init), std::invalid_argument);
  cudf::numeric_scalar<int8_t> step1(1);
  EXPECT_THROW(cudf::sequence(10, init, step1), std::invalid_argument);

  cudf::numeric_scalar<int> zero(0);
  cudf::numeric_scalar<float> step(1, false);
  EXPECT_THROW(cudf::sequence(10, zero, step), std::invalid_argument);
  EXPECT_THROW(cudf::sequence(10, init, step), std::invalid_argument);
}

TYPED_TEST(SequenceTypedTestFixture, DefaultStep)
{
  using T = TypeParam;

  cudf::numeric_scalar<T> init(0);

  cudf::size_type num_els = 10;

  cudf::test::fixed_width_column_wrapper<T> expected_w({0, 1, 2, 3, 4, 5, 6, 7, 8, 9});

  auto result = cudf::sequence(num_els, init);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result, expected_w);
}

TEST_F(SequenceTestFixture, DateSequenceBasic)
{
  // Timestamp generated using https://www.epochconverter.com/
  cudf::timestamp_scalar<cudf::timestamp_s> init(1629852896L, true);  // 2021-08-25 00:54:56 GMT
  cudf::size_type size{5};
  cudf::size_type months{1};

  cudf::test::fixed_width_column_wrapper<cudf::timestamp_s, int64_t> expected{
    1629852896L,  // 2021-08-25 00:54:56 GMT
    1632531296L,  // 2021-09-25 00:54:56 GMT
    1635123296L,  // 2021-10-25 00:54:56 GMT
    1637801696L,  // 2021-11-25 00:54:56 GMT
    1640393696L,  // 2021-12-25 00:54:56 GMT
  };

  auto got = calendrical_month_sequence(size, init, months);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *got);
}

TEST_F(SequenceTestFixture, DateSequenceLeapYear)
{
  // Timestamp generated using https://www.epochconverter.com/
  cudf::timestamp_scalar<cudf::timestamp_s> init(951876379L, true);  // 2000-02-29 02:06:19 GMT
  cudf::size_type size{5};
  cudf::size_type months{12};

  cudf::test::fixed_width_column_wrapper<cudf::timestamp_s, int64_t> expected{
    951876379L,   // 2000-02-29 02:06:19 GMT Leap Year
    983412379L,   // 2001-02-28 02:06:19 GMT
    1014948379L,  // 2002-02-28 02:06:19 GMT
    1046484379L,  // 2003-02-28 02:06:19 GMT
    1078106779L,  // 2004-02-29 02:06:19 GMT Leap Year
  };

  auto got = calendrical_month_sequence(size, init, months);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *got);
}

TEST_F(SequenceTestFixture, DateSequenceBadTypes)
{
  cudf::numeric_scalar<int64_t> init(951876379, true);
  cudf::size_type size   = 5;
  cudf::size_type months = 12;

  EXPECT_THROW(calendrical_month_sequence(size, init, months), cudf::logic_error);
}
