/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>

#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/transform.hpp>

template <typename T>
struct NaNsToNullTest : public cudf::test::BaseFixture {
  void run_test(cudf::column_view const& input, cudf::column_view const& expected)
  {
    auto got = cudf::column_nans_to_nulls(input);

    EXPECT_EQ(expected.null_count(), got->null_count());

    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, got->view());
  }

  std::unique_ptr<cudf::column> create_expected(std::vector<T> const& input,
                                                std::vector<bool> const& mask = {})
  {
    std::vector<T> expected(input);
    std::vector<bool> expected_mask;

    if (mask.size() > 0) {
      std::transform(input.begin(),
                     input.end(),
                     mask.begin(),
                     std::back_inserter(expected_mask),
                     [](T val, bool validity) { return validity and not std::isnan(val); });
    } else {
      std::transform(input.begin(), input.end(), std::back_inserter(expected_mask), [](T val) {
        return not std::isnan(val);
      });
    }

    return cudf::test::fixed_width_column_wrapper<T>(
             expected.begin(), expected.end(), expected_mask.begin())
      .release();
  }
};

using test_types = ::testing::Types<float, double>;

TYPED_TEST_SUITE(NaNsToNullTest, test_types);

TYPED_TEST(NaNsToNullTest, WithMask)
{
  using T = TypeParam;

  std::vector<T> input   = {1, NAN, 3, NAN, 5, NAN};
  std::vector<bool> mask = {true, true, true, true, false, false};
  auto input_column =
    cudf::test::fixed_width_column_wrapper<T>(input.begin(), input.end(), mask.begin());
  auto expected_column = this->create_expected(input, mask);
  this->run_test(input_column, expected_column->view());
}

TYPED_TEST(NaNsToNullTest, WithNoMask)
{
  using T = TypeParam;

  std::vector<T> input = {1, NAN, 3, NAN, 5, NAN};
  auto input_column    = cudf::test::fixed_width_column_wrapper<T>(input.begin(), input.end());
  auto expected_column = this->create_expected(input);
  this->run_test(input_column, expected_column->view());
}

TYPED_TEST(NaNsToNullTest, NoNANWithMask)
{
  using T = TypeParam;

  std::vector<T> input   = {1, 2, 3, 4, 5, 6};
  std::vector<bool> mask = {true, true, true, true, false, false};
  auto input_column =
    cudf::test::fixed_width_column_wrapper<T>(input.begin(), input.end(), mask.begin());
  auto expected_column = this->create_expected(input, mask);
  this->run_test(input_column, expected_column->view());
}

TYPED_TEST(NaNsToNullTest, NoNANNoMask)
{
  using T = TypeParam;

  std::vector<T> input = {1, 2, 3, 4, 5, 6};
  auto input_column    = cudf::test::fixed_width_column_wrapper<T>(input.begin(), input.end());
  auto expected_column = this->create_expected(input);
  this->run_test(input_column, expected_column->view());
}

TYPED_TEST(NaNsToNullTest, WithOffset)
{
  using T                = TypeParam;
  std::vector<T> input   = {1, NAN, 3, NAN, 0, NAN, 7, NAN, 9};
  std::vector<bool> mask = {true, true, true, true, false, false, true, true, true};

  auto input_column =
    cudf::test::fixed_width_column_wrapper<T>(input.begin(), input.end(), mask.begin());
  auto sliced_column = cudf::slice(input_column, {1, 5}).front();

  std::vector<T> expected         = {0, 3, 0, 0};
  std::vector<bool> expected_mask = {false, true, false, false};

  auto expected_column = cudf::test::fixed_width_column_wrapper<T>(
    expected.begin(), expected.end(), expected_mask.begin());
  this->run_test(sliced_column, expected_column);
}

TYPED_TEST(NaNsToNullTest, EmptyColumn)
{
  using T = TypeParam;

  auto input_column = cudf::test::fixed_width_column_wrapper<T>({});
  this->run_test(input_column, input_column);
}

struct NaNsToNullFailTest : public cudf::test::BaseFixture {};

TEST_F(NaNsToNullFailTest, StringType)
{
  std::vector<std::string> strings{
    "", "this", "is", "a", "column", "of", "strings", "with", "in", "valid"};
  cudf::test::strings_column_wrapper input(strings.begin(), strings.end());

  EXPECT_THROW(cudf::column_nans_to_nulls(input), std::invalid_argument);
}

TEST_F(NaNsToNullFailTest, IntegerType)
{
  std::vector<int32_t> input = {1, 2, 3, 4, 5, 6};
  auto input_column = cudf::test::fixed_width_column_wrapper<int32_t>(input.begin(), input.end());

  EXPECT_THROW(cudf::column_nans_to_nulls(input_column), std::invalid_argument);
}

TEST_F(NaNsToNullFailTest, EmptyColumn)
{
  auto input_column = cudf::test::fixed_width_column_wrapper<int32_t>({});
  EXPECT_THROW(cudf::column_nans_to_nulls(input_column), std::invalid_argument);
}
