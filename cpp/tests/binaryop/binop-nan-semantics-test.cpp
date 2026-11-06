/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/cudf_gtest.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/binaryop.hpp>
#include <cudf/scalar/scalar.hpp>

#include <limits>
#include <vector>

namespace {
using bools          = cudf::test::fixed_width_column_wrapper<bool>;
auto const bool_type = cudf::data_type{cudf::type_id::BOOL8};

/**
 * @brief Builds a floating-point column from `values`, which may mix integers with NaN and
 * infinity, and optional `valids`
 */
template <typename T>
cudf::test::fixed_width_column_wrapper<T> make_column(std::vector<T> const& values,
                                                      std::vector<bool> const& valids = {})
{
  using wrapper = cudf::test::fixed_width_column_wrapper<T>;
  if (valids.empty()) { return wrapper(values.begin(), values.end()); }
  return wrapper(values.begin(), values.end(), valids.begin());
}
}  // namespace

template <typename T>
struct BinaryOperationNanSemanticsTest : public cudf::test::BaseFixture {
  static constexpr T NaN = std::numeric_limits<T>::quiet_NaN();
  static constexpr T Inf = std::numeric_limits<T>::infinity();
};

TYPED_TEST_SUITE(BinaryOperationNanSemanticsTest, cudf::test::FloatingPointTypes);

TYPED_TEST(BinaryOperationNanSemanticsTest, Comparisons)
{
  using T            = TypeParam;
  auto constexpr NaN = TestFixture::NaN;
  auto constexpr Inf = TestFixture::Inf;

  auto const lhs = make_column<T>({1, NaN, NaN, 3, -Inf, Inf, -NaN});
  auto const rhs = make_column<T>({NaN, NaN, 1, 3, NaN, NaN, NaN});

  struct test_case {
    cudf::binary_operator op;
    bools ieee754;
    bools etl;
  };
  std::vector<test_case> cases;
  cases.push_back({cudf::binary_operator::EQUAL, {0, 0, 0, 1, 0, 0, 0}, {0, 1, 0, 1, 0, 0, 1}});
  cases.push_back({cudf::binary_operator::NOT_EQUAL, {1, 1, 1, 0, 1, 1, 1}, {1, 0, 1, 0, 1, 1, 0}});
  cases.push_back({cudf::binary_operator::LESS, {0, 0, 0, 0, 0, 0, 0}, {1, 0, 0, 0, 1, 1, 0}});
  cases.push_back({cudf::binary_operator::GREATER, {0, 0, 0, 0, 0, 0, 0}, {0, 0, 1, 0, 0, 0, 0}});
  cases.push_back(
    {cudf::binary_operator::LESS_EQUAL, {0, 0, 0, 1, 0, 0, 0}, {1, 1, 0, 1, 1, 1, 1}});
  cases.push_back(
    {cudf::binary_operator::GREATER_EQUAL, {0, 0, 0, 1, 0, 0, 0}, {0, 1, 1, 1, 0, 0, 1}});

  for (auto const& c : cases) {
    SCOPED_TRACE(static_cast<int>(c.op));
    // The default is IEEE 754
    auto result = cudf::binary_operation(lhs, rhs, c.op, bool_type);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(c.ieee754, *result);
    result = cudf::binary_operation(lhs, rhs, c.op, bool_type, cudf::nan_semantics::DEFAULT);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(c.ieee754, *result);
    result = cudf::binary_operation(lhs, rhs, c.op, bool_type, cudf::nan_semantics::IEEE754);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(c.ieee754, *result);
    result = cudf::binary_operation(lhs, rhs, c.op, bool_type, cudf::nan_semantics::ETL);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(c.etl, *result);
  }
}

TYPED_TEST(BinaryOperationNanSemanticsTest, ScalarOperands)
{
  using T            = TypeParam;
  auto constexpr NaN = TestFixture::NaN;

  auto const col    = make_column<T>({1, NaN, 0}, {1, 1, 0});
  auto const nan    = cudf::numeric_scalar<T>(NaN);
  auto const etl    = cudf::nan_semantics::ETL;
  auto const ieee   = cudf::nan_semantics::IEEE754;
  auto const less   = cudf::binary_operator::LESS;
  auto const equal  = cudf::binary_operator::EQUAL;
  auto const valids = std::vector<bool>{1, 1, 0};

  // col < NaN
  auto result = cudf::binary_operation(col, nan, less, bool_type, etl);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(bools({1, 0, 0}, valids.begin()), *result);
  result = cudf::binary_operation(col, nan, less, bool_type, ieee);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(bools({0, 0, 0}, valids.begin()), *result);
  // NaN == col
  result = cudf::binary_operation(nan, col, equal, bool_type, etl);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(bools({0, 1, 0}, valids.begin()), *result);
  result = cudf::binary_operation(nan, col, equal, bool_type, ieee);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(bools({0, 0, 0}, valids.begin()), *result);
  // NaN > col
  result = cudf::binary_operation(nan, col, cudf::binary_operator::GREATER, bool_type, etl);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(bools({1, 0, 0}, valids.begin()), *result);
}

TYPED_TEST(BinaryOperationNanSemanticsTest, NullAwareEquality)
{
  using T            = TypeParam;
  auto constexpr NaN = TestFixture::NaN;

  auto const lhs = make_column<T>({NaN, NaN, 0, 1, 0}, {1, 1, 0, 1, 0});
  auto const rhs = make_column<T>({NaN, 0, NaN, NaN, 0}, {1, 0, 1, 1, 0});

  auto result = cudf::binary_operation(
    lhs, rhs, cudf::binary_operator::NULL_EQUALS, bool_type, cudf::nan_semantics::ETL);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(bools{1, 0, 0, 0, 1}, *result);
  result = cudf::binary_operation(lhs, rhs, cudf::binary_operator::NULL_EQUALS, bool_type);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(bools{0, 0, 0, 0, 1}, *result);

  result = cudf::binary_operation(
    lhs, rhs, cudf::binary_operator::NULL_NOT_EQUALS, bool_type, cudf::nan_semantics::ETL);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(bools{0, 1, 1, 1, 0}, *result);
  result = cudf::binary_operation(lhs, rhs, cudf::binary_operator::NULL_NOT_EQUALS, bool_type);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(bools{1, 1, 1, 1, 0}, *result);
}

TYPED_TEST(BinaryOperationNanSemanticsTest, NullMaxMin)
{
  using T            = TypeParam;
  auto constexpr NaN = TestFixture::NaN;

  auto const lhs  = make_column<T>({NaN, 1, NaN, 0, 2}, {1, 1, 1, 0, 1});
  auto const rhs  = make_column<T>({1, NaN, NaN, NaN, 0}, {1, 1, 1, 1, 0});
  auto const type = cudf::data_type{cudf::type_to_id<T>()};

  // A NaN is the largest value with ETL semantics, regardless of the operand order
  auto result = cudf::binary_operation(
    lhs, rhs, cudf::binary_operator::NULL_MAX, type, cudf::nan_semantics::ETL);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(make_column<T>({NaN, NaN, NaN, NaN, 2}), *result);
  result = cudf::binary_operation(
    lhs, rhs, cudf::binary_operator::NULL_MIN, type, cudf::nan_semantics::ETL);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(make_column<T>({1, 1, NaN, NaN, 2}), *result);

  // With IEEE 754 semantics, the result depends on the operand order
  result = cudf::binary_operation(lhs, rhs, cudf::binary_operator::NULL_MAX, type);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(make_column<T>({1, NaN, NaN, NaN, 2}), *result);
  result = cudf::binary_operation(lhs, rhs, cudf::binary_operator::NULL_MIN, type);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(make_column<T>({1, NaN, NaN, NaN, 2}), *result);
}

struct BinaryOperationNanSemanticsMixedTest : public cudf::test::BaseFixture {};

TEST_F(BinaryOperationNanSemanticsMixedTest, IntegerAndFloatingPoint)
{
  // The integer operand is compared in the common floating-point type
  auto const col = cudf::test::fixed_width_column_wrapper<int32_t>{1, 2};
  auto const nan = cudf::numeric_scalar<double>(std::numeric_limits<double>::quiet_NaN());

  auto result = cudf::binary_operation(
    col, nan, cudf::binary_operator::LESS, bool_type, cudf::nan_semantics::ETL);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(bools{1, 1}, *result);
  result = cudf::binary_operation(col, nan, cudf::binary_operator::LESS, bool_type);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(bools{0, 0}, *result);
}

TEST_F(BinaryOperationNanSemanticsMixedTest, NonFloatingPointUnaffected)
{
  auto const lhs = cudf::test::fixed_width_column_wrapper<int32_t>{1, 2, 3};
  auto const rhs = cudf::test::fixed_width_column_wrapper<int32_t>{2, 2, 2};

  for (auto op : {cudf::binary_operator::LESS,
                  cudf::binary_operator::EQUAL,
                  cudf::binary_operator::NULL_EQUALS}) {
    auto const ieee = cudf::binary_operation(lhs, rhs, op, bool_type);
    auto const etl  = cudf::binary_operation(lhs, rhs, op, bool_type, cudf::nan_semantics::ETL);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*ieee, *etl);
  }
}
