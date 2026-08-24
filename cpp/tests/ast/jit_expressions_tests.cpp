/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/iterator_utilities.hpp>
#include <cudf_test/testing_main.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/ast/expressions.hpp>
#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/filling.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/transform.hpp>

#include <rmm/cuda_stream.hpp>

#include <cuda/iterator>

#include <limits>
#include <vector>

constexpr cudf::test::debug_output_level VERBOSITY{cudf::test::debug_output_level::ALL_ERRORS};

template <typename T>
using column_wrapper = cudf::test::fixed_width_column_wrapper<T>;

template <typename T>
using decimal_column_wrapper = cudf::test::fixed_point_column_wrapper<typename T::rep>;

struct JITExpressionTest : public cudf::test::BaseFixture {};

template <typename T>
struct JITIntegerArithmeticTest : public cudf::test::BaseFixture {
  static constexpr T MAX = std::numeric_limits<T>::max();
  static constexpr T MIN = std::numeric_limits<T>::min();
};

template <typename T>
struct JITSignedIntegerArithmeticTest : public JITIntegerArithmeticTest<T> {};

template <typename T>
struct JITDecimalArithmeticTest : public JITIntegerArithmeticTest<typename T::rep> {};

using SignedIntegralTypesNotBool = cudf::test::Types<int8_t, int16_t, int32_t, int64_t>;

TYPED_TEST_SUITE(JITIntegerArithmeticTest, cudf::test::IntegralTypesNotBool);
TYPED_TEST_SUITE(JITSignedIntegerArithmeticTest, SignedIntegralTypesNotBool);
TYPED_TEST_SUITE(JITDecimalArithmeticTest, cudf::test::FixedPointTypes);

TEST_F(JITExpressionTest, Coalesce)
{
  auto a         = column_wrapper<int32_t>{{1, 3, 5, 7, 9, 11}, {1, 0, 0, 1, 0, 0}};
  auto b         = column_wrapper<int32_t>{{2, 4, 6, 8, 10, 12}, {1, 1, 1, 0, 1, 0}};
  auto expected  = column_wrapper<int32_t>{{1, 4, 6, 7, 10, 0}, {1, 1, 1, 1, 1, 0}};
  auto table     = cudf::table_view{{a, b}};
  auto tree      = cudf::ast::tree{};
  auto a_ref     = cudf::ast::column_reference(0);
  auto b_ref     = cudf::ast::column_reference(1);
  auto& coalesce = cudf::ast::jit::operation(tree, cudf::ast::jit::op::COALESCE, {a_ref, b_ref});
  auto result    = cudf::compute_column_jit(table, coalesce);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view(), VERBOSITY);
}

TYPED_TEST(JITIntegerArithmeticTest, AddOverflow)
{
  using T            = TypeParam;
  auto a             = column_wrapper<T>{{3, 20, 1, 50}};
  auto b             = column_wrapper<T>{{10, 7, 20, 0}};
  auto b_fail        = column_wrapper<T>{{T{10}, this->MAX, T{20}, T{0}}};
  auto expected      = column_wrapper<T>{{13, 27, 21, 50}};
  auto expected_fail = column_wrapper<T>{{13, 0, 21, 50}, {1, 0, 1, 1}};
  auto table         = cudf::table_view{{a, b, b_fail}};
  auto tree          = cudf::ast::tree{};
  auto a_ref         = cudf::ast::column_reference(0);
  auto b_ref         = cudf::ast::column_reference(1);
  auto b_fail_ref    = cudf::ast::column_reference(2);

  auto& add = cudf::ast::jit::operation(tree, cudf::ast::jit::op::ADD_OVERFLOW, {a_ref, b_ref});
  auto& add_fail =
    cudf::ast::jit::operation(tree, cudf::ast::jit::op::ADD_OVERFLOW, {a_ref, b_fail_ref});
  auto& try_add_fail = cudf::ast::jit::operation(
    tree, cudf::ast::jit::op::ADD_OVERFLOW, {a_ref, b_fail_ref}, cudf::error_policy::NULLIFY);
  auto result      = cudf::compute_column_jit(table, add);
  auto result_fail = cudf::compute_column_jit(table, try_add_fail);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view(), VERBOSITY);

  EXPECT_THROW(result = cudf::compute_column_jit(table, add_fail), cudf::evaluation_error);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_fail, result_fail->view(), VERBOSITY);
}

TYPED_TEST(JITDecimalArithmeticTest, AddOverflow)
{
  using T     = TypeParam;
  using R     = typename T::rep;
  auto a      = decimal_column_wrapper<T>{{3, 20, 1, 50}, numeric::scale_type{0}};
  auto b      = decimal_column_wrapper<T>{{10, 7, 20, 0}, numeric::scale_type{0}};
  auto b_fail = decimal_column_wrapper<T>{{R{10}, this->MAX, R{20}, R{0}}, numeric::scale_type{0}};
  auto expected = decimal_column_wrapper<T>{{13, 27, 21, 50}, numeric::scale_type{0}};
  auto expected_fail =
    decimal_column_wrapper<T>{{13, 0, 21, 50}, {1, 0, 1, 1}, numeric::scale_type{0}};
  auto table      = cudf::table_view{{a, b, b_fail}};
  auto tree       = cudf::ast::tree{};
  auto a_ref      = cudf::ast::column_reference(0);
  auto b_ref      = cudf::ast::column_reference(1);
  auto b_fail_ref = cudf::ast::column_reference(2);
  auto& add = cudf::ast::jit::operation(tree, cudf::ast::jit::op::ADD_OVERFLOW, {a_ref, b_ref});
  auto& add_fail =
    cudf::ast::jit::operation(tree, cudf::ast::jit::op::ADD_OVERFLOW, {a_ref, b_fail_ref});
  auto& try_add_fail = cudf::ast::jit::operation(
    tree, cudf::ast::jit::op::ADD_OVERFLOW, {a_ref, b_fail_ref}, cudf::error_policy::NULLIFY);
  auto result      = cudf::compute_column_jit(table, add);
  auto result_fail = cudf::compute_column_jit(table, try_add_fail);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view(), VERBOSITY);

  EXPECT_THROW(result = cudf::compute_column_jit(table, add_fail), cudf::evaluation_error);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_fail, result_fail->view(), VERBOSITY);
}

TYPED_TEST(JITSignedIntegerArithmeticTest, SubOverflow)
{
  using T            = TypeParam;
  auto a             = column_wrapper<T>{{3, 20, 1, 50}};
  auto b             = column_wrapper<T>{{10, 7, 20, 0}};
  auto b_fail        = column_wrapper<T>{{T{10}, T{this->MIN}, T{20}, T{0}}};
  auto expected      = column_wrapper<T>{{-7, 13, -19, 50}};
  auto expected_fail = column_wrapper<T>{{-7, 0, -19, 50}, {1, 0, 1, 1}};
  auto table         = cudf::table_view{{a, b, b_fail}};
  auto tree          = cudf::ast::tree{};
  auto a_ref         = cudf::ast::column_reference(0);
  auto b_ref         = cudf::ast::column_reference(1);
  auto b_fail_ref    = cudf::ast::column_reference(2);
  auto& sub = cudf::ast::jit::operation(tree, cudf::ast::jit::op::SUB_OVERFLOW, {a_ref, b_ref});
  auto& sub_fail =
    cudf::ast::jit::operation(tree, cudf::ast::jit::op::SUB_OVERFLOW, {a_ref, b_fail_ref});
  auto& try_sub_fail = cudf::ast::jit::operation(
    tree, cudf::ast::jit::op::SUB_OVERFLOW, {a_ref, b_fail_ref}, cudf::error_policy::NULLIFY);

  auto result      = cudf::compute_column_jit(table, sub);
  auto result_fail = cudf::compute_column_jit(table, try_sub_fail);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view(), VERBOSITY);

  EXPECT_THROW(result = cudf::compute_column_jit(table, sub_fail), cudf::evaluation_error);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_fail, result_fail->view(), VERBOSITY);
}

TYPED_TEST(JITDecimalArithmeticTest, SubOverflow)
{
  using T = TypeParam;
  using R = typename T::rep;
  auto a  = decimal_column_wrapper<T>{{3, 20, 1, 50}, numeric::scale_type{0}};
  auto b  = decimal_column_wrapper<T>{{10, 7, 20, 0}, numeric::scale_type{0}};
  auto b_fail =
    decimal_column_wrapper<T>{{R{10}, R{this->MIN}, R{20}, R{0}}, numeric::scale_type{0}};
  auto expected = decimal_column_wrapper<T>{{-7, 13, -19, 50}, numeric::scale_type{0}};
  auto expected_fail =
    decimal_column_wrapper<T>{{-7, 0, -19, 50}, {1, 0, 1, 1}, numeric::scale_type{0}};
  auto table      = cudf::table_view{{a, b, b_fail}};
  auto tree       = cudf::ast::tree{};
  auto a_ref      = cudf::ast::column_reference(0);
  auto b_ref      = cudf::ast::column_reference(1);
  auto b_fail_ref = cudf::ast::column_reference(2);
  auto& sub = cudf::ast::jit::operation(tree, cudf::ast::jit::op::SUB_OVERFLOW, {a_ref, b_ref});
  auto& sub_fail =
    cudf::ast::jit::operation(tree, cudf::ast::jit::op::SUB_OVERFLOW, {a_ref, b_fail_ref});
  auto& try_sub_fail = cudf::ast::jit::operation(
    tree, cudf::ast::jit::op::SUB_OVERFLOW, {a_ref, b_fail_ref}, cudf::error_policy::NULLIFY);
  auto result      = cudf::compute_column_jit(table, sub);
  auto result_fail = cudf::compute_column_jit(table, try_sub_fail);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view(), VERBOSITY);

  EXPECT_THROW(result = cudf::compute_column_jit(table, sub_fail), cudf::evaluation_error);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_fail, result_fail->view(), VERBOSITY);
}

TYPED_TEST(JITIntegerArithmeticTest, MulOverflow)
{
  using T            = TypeParam;
  auto a             = column_wrapper<T>{{3, 20, 2, 50}};
  auto b             = column_wrapper<T>{{10, 2, 1, 0}};
  auto b_fail        = column_wrapper<T>{{T{10}, T{this->MAX}, T{1}, T{0}}};
  auto expected      = column_wrapper<T>{{30, 40, 2, 0}};
  auto expected_fail = column_wrapper<T>{{30, 0, 2, 0}, {1, 0, 1, 1}};
  auto table         = cudf::table_view{{a, b, b_fail}};
  auto a_ref         = cudf::ast::column_reference(0);
  auto b_ref         = cudf::ast::column_reference(1);
  auto b_fail_ref    = cudf::ast::column_reference(2);
  auto tree          = cudf::ast::tree{};
  auto& mul = cudf::ast::jit::operation(tree, cudf::ast::jit::op::MUL_OVERFLOW, {a_ref, b_ref});
  auto& mul_fail =
    cudf::ast::jit::operation(tree, cudf::ast::jit::op::MUL_OVERFLOW, {a_ref, b_fail_ref});
  auto& try_mul_fail = cudf::ast::jit::operation(
    tree, cudf::ast::jit::op::MUL_OVERFLOW, {a_ref, b_fail_ref}, cudf::error_policy::NULLIFY);
  auto result      = cudf::compute_column_jit(table, mul);
  auto result_fail = cudf::compute_column_jit(table, try_mul_fail);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view(), VERBOSITY);

  EXPECT_THROW(result = cudf::compute_column_jit(table, mul_fail), cudf::evaluation_error);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_fail, result_fail->view(), VERBOSITY);
}

TYPED_TEST(JITDecimalArithmeticTest, MulOverflow)
{
  using T = TypeParam;
  using R = typename T::rep;
  auto a  = decimal_column_wrapper<T>{{3, 20, 2, 50}, numeric::scale_type{0}};
  auto b  = decimal_column_wrapper<T>{{10, 7, 1, 0}, numeric::scale_type{0}};
  auto b_fail =
    decimal_column_wrapper<T>{{R{10}, R{this->MAX}, R{1}, R{0}}, numeric::scale_type{0}};
  auto expected = decimal_column_wrapper<T>{{30, 140, 2, 0}, numeric::scale_type{0}};
  auto expected_fail =
    decimal_column_wrapper<T>{{30, 0, 2, 0}, {1, 0, 1, 1}, numeric::scale_type{0}};
  auto table      = cudf::table_view{{a, b, b_fail}};
  auto a_ref      = cudf::ast::column_reference(0);
  auto b_ref      = cudf::ast::column_reference(1);
  auto b_fail_ref = cudf::ast::column_reference(2);
  auto tree       = cudf::ast::tree{};
  auto& mul = cudf::ast::jit::operation(tree, cudf::ast::jit::op::MUL_OVERFLOW, {a_ref, b_ref});
  auto& mul_fail =
    cudf::ast::jit::operation(tree, cudf::ast::jit::op::MUL_OVERFLOW, {a_ref, b_fail_ref});
  auto& try_mul_fail = cudf::ast::jit::operation(
    tree, cudf::ast::jit::op::MUL_OVERFLOW, {a_ref, b_fail_ref}, cudf::error_policy::NULLIFY);

  // This fails on CI CUDA 12.2, driver 535, V100
  if constexpr (std::is_same_v<T, numeric::decimal128>) {
    int driver_version{0};
    auto const err = cudaDriverGetVersion(&driver_version);
    if (err != cudaSuccess or driver_version < 12090) {
      std::cout
        << "Skipping JITDecimalArithmeticTest.MulOverflow/decimal128 test, driver earlier than 12.9"
        << std::endl;
      GTEST_SKIP();
    }
  }

  auto result      = cudf::compute_column_jit(table, mul);
  auto result_fail = cudf::compute_column_jit(table, try_mul_fail);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view(), VERBOSITY);

  EXPECT_THROW(result = cudf::compute_column_jit(table, mul_fail), cudf::evaluation_error);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_fail, result_fail->view(), VERBOSITY);
}

TYPED_TEST(JITIntegerArithmeticTest, DivOverflow)
{
  using T            = TypeParam;
  auto a             = column_wrapper<T>{{3, 20, 1, 50}};
  auto b             = column_wrapper<T>{{10, 7, 2, 1}};
  auto b_fail        = column_wrapper<T>{{10, 1, 20, 0}};
  auto expected      = column_wrapper<T>{{0, 2, 0, 50}};
  auto expected_fail = column_wrapper<T>{{0, 20, 0, 50}, {1, 1, 1, 0}};
  auto table         = cudf::table_view{{a, b, b_fail}};
  auto a_ref         = cudf::ast::column_reference(0);
  auto b_ref         = cudf::ast::column_reference(1);
  auto b_fail_ref    = cudf::ast::column_reference(2);
  auto tree          = cudf::ast::tree{};
  auto& div = cudf::ast::jit::operation(tree, cudf::ast::jit::op::DIV_OVERFLOW, {a_ref, b_ref});
  auto& div_fail =
    cudf::ast::jit::operation(tree, cudf::ast::jit::op::DIV_OVERFLOW, {a_ref, b_fail_ref});
  auto& try_div_fail = cudf::ast::jit::operation(
    tree, cudf::ast::jit::op::DIV_OVERFLOW, {a_ref, b_fail_ref}, cudf::error_policy::NULLIFY);
  auto result      = cudf::compute_column_jit(table, div);
  auto result_fail = cudf::compute_column_jit(table, try_div_fail);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view(), VERBOSITY);

  EXPECT_THROW(result = cudf::compute_column_jit(table, div_fail), cudf::evaluation_error);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_fail, result_fail->view(), VERBOSITY);
}

TYPED_TEST(JITDecimalArithmeticTest, DivOverflow)
{
  using T       = TypeParam;
  auto a        = decimal_column_wrapper<T>{{3, 20, 1, 50}, numeric::scale_type{0}};
  auto b        = decimal_column_wrapper<T>{{10, 7, 2, 1}, numeric::scale_type{0}};
  auto b_fail   = decimal_column_wrapper<T>{{10, 1, 20, 0}, numeric::scale_type{0}};
  auto expected = decimal_column_wrapper<T>{{0, 2, 0, 50}, numeric::scale_type{0}};
  auto expected_fail =
    decimal_column_wrapper<T>{{0, 20, 0, 50}, {1, 1, 1, 0}, numeric::scale_type{0}};
  auto table      = cudf::table_view{{a, b, b_fail}};
  auto a_ref      = cudf::ast::column_reference(0);
  auto b_ref      = cudf::ast::column_reference(1);
  auto b_fail_ref = cudf::ast::column_reference(2);
  auto tree       = cudf::ast::tree{};
  auto& div = cudf::ast::jit::operation(tree, cudf::ast::jit::op::DIV_OVERFLOW, {a_ref, b_ref});
  auto& div_fail =
    cudf::ast::jit::operation(tree, cudf::ast::jit::op::DIV_OVERFLOW, {a_ref, b_fail_ref});
  auto& try_div_fail = cudf::ast::jit::operation(
    tree, cudf::ast::jit::op::DIV_OVERFLOW, {a_ref, b_fail_ref}, cudf::error_policy::NULLIFY);
  auto result      = cudf::compute_column_jit(table, div);
  auto result_fail = cudf::compute_column_jit(table, try_div_fail);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view(), VERBOSITY);

  EXPECT_THROW(result = cudf::compute_column_jit(table, div_fail), cudf::evaluation_error);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_fail, result_fail->view(), VERBOSITY);
}

TYPED_TEST(JITIntegerArithmeticTest, ModOverflow)
{
  using T            = TypeParam;
  auto a             = column_wrapper<T>{{3, 20, 1, 50}};
  auto b             = column_wrapper<T>{{10, 7, 2, 1}};
  auto b_fail        = column_wrapper<T>{{10, 1, 20, 0}};
  auto expected      = column_wrapper<T>{{3, 6, 1, 0}};
  auto expected_fail = column_wrapper<T>{{3, 0, 1, 0}, {1, 1, 1, 0}};
  auto table         = cudf::table_view{{a, b, b_fail}};
  auto a_ref         = cudf::ast::column_reference(0);
  auto b_ref         = cudf::ast::column_reference(1);
  auto b_fail_ref    = cudf::ast::column_reference(2);
  auto tree          = cudf::ast::tree{};
  auto& mod = cudf::ast::jit::operation(tree, cudf::ast::jit::op::MOD_OVERFLOW, {a_ref, b_ref});
  auto& mod_fail =
    cudf::ast::jit::operation(tree, cudf::ast::jit::op::MOD_OVERFLOW, {a_ref, b_fail_ref});
  auto& try_mod_fail = cudf::ast::jit::operation(
    tree, cudf::ast::jit::op::MOD_OVERFLOW, {a_ref, b_fail_ref}, cudf::error_policy::NULLIFY);
  auto result      = cudf::compute_column_jit(table, mod);
  auto result_fail = cudf::compute_column_jit(table, try_mod_fail);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view(), VERBOSITY);

  EXPECT_THROW(result = cudf::compute_column_jit(table, mod_fail), cudf::evaluation_error);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_fail, result_fail->view(), VERBOSITY);
}

TYPED_TEST(JITDecimalArithmeticTest, ModOverflow)
{
  using T       = TypeParam;
  auto a        = decimal_column_wrapper<T>{{3, 20, 1, 50}, numeric::scale_type{0}};
  auto b        = decimal_column_wrapper<T>{{10, 7, 2, 1}, numeric::scale_type{0}};
  auto b_fail   = decimal_column_wrapper<T>{{10, 1, 20, 0}, numeric::scale_type{0}};
  auto expected = decimal_column_wrapper<T>{{3, 6, 1, 0}, numeric::scale_type{0}};
  auto expected_fail =
    decimal_column_wrapper<T>{{3, 0, 1, 0}, {1, 1, 1, 0}, numeric::scale_type{0}};
  auto table      = cudf::table_view{{a, b, b_fail}};
  auto a_ref      = cudf::ast::column_reference(0);
  auto b_ref      = cudf::ast::column_reference(1);
  auto b_fail_ref = cudf::ast::column_reference(2);
  auto tree       = cudf::ast::tree{};
  auto& mod = cudf::ast::jit::operation(tree, cudf::ast::jit::op::MOD_OVERFLOW, {a_ref, b_ref});
  auto& mod_fail =
    cudf::ast::jit::operation(tree, cudf::ast::jit::op::MOD_OVERFLOW, {a_ref, b_fail_ref});
  auto& try_mod_fail = cudf::ast::jit::operation(
    tree, cudf::ast::jit::op::MOD_OVERFLOW, {a_ref, b_fail_ref}, cudf::error_policy::NULLIFY);
  auto result      = cudf::compute_column_jit(table, mod);
  auto result_fail = cudf::compute_column_jit(table, try_mod_fail);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view(), VERBOSITY);

  EXPECT_THROW(result = cudf::compute_column_jit(table, mod_fail), cudf::evaluation_error);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_fail, result_fail->view(), VERBOSITY);
}

TYPED_TEST(JITSignedIntegerArithmeticTest, AbsOverflow)
{
  using T     = TypeParam;
  auto a      = column_wrapper<T>{{T{3}, T{-20}, T{1}, T{-50}, this->MAX, T{this->MIN + 1}, T{0}}};
  auto a_fail = column_wrapper<T>{{T{3}, T{-20}, T{1}, T{-50}, this->MIN, T{1}, T{0}}};
  auto expected =
    column_wrapper<T>{{T{3}, T{20}, T{1}, T{50}, this->MAX, T{std::abs(this->MIN + 1)}, T{0}}};
  auto expected_fail = column_wrapper<T>{{3, 20, 1, 50, 0, 1, 0}, {1, 1, 1, 1, 0, 1, 1}};
  auto table         = cudf::table_view{{a, a_fail}};
  auto a_ref         = cudf::ast::column_reference(0);
  auto a_fail_ref    = cudf::ast::column_reference(1);
  auto tree          = cudf::ast::tree{};
  auto& abs          = cudf::ast::jit::operation(tree, cudf::ast::jit::op::ABS_OVERFLOW, {a_ref});
  auto& abs_fail = cudf::ast::jit::operation(tree, cudf::ast::jit::op::ABS_OVERFLOW, {a_fail_ref});
  auto& try_abs_fail = cudf::ast::jit::operation(
    tree, cudf::ast::jit::op::ABS_OVERFLOW, {a_fail_ref}, cudf::error_policy::NULLIFY);
  auto result      = cudf::compute_column_jit(table, abs);
  auto result_fail = cudf::compute_column_jit(table, try_abs_fail);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view(), VERBOSITY);

  EXPECT_THROW(result = cudf::compute_column_jit(table, abs_fail), cudf::evaluation_error);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_fail, result_fail->view(), VERBOSITY);
}

TYPED_TEST(JITDecimalArithmeticTest, AbsOverflow)
{
  using T = TypeParam;
  using R = typename T::rep;
  auto a  = decimal_column_wrapper<T>{
    {R{3}, R{-20}, R{1}, R{-50}, this->MAX, R{this->MIN + 1}, R{0}}, numeric::scale_type{0}};
  auto a_fail   = decimal_column_wrapper<T>{{R{3}, R{-20}, R{1}, R{-50}, this->MIN, R{1}, R{0}},
                                            numeric::scale_type{0}};
  auto expected = decimal_column_wrapper<T>{
    {R{3}, R{20}, R{1}, R{50}, this->MAX, R{std::abs(this->MIN + 1)}, R{0}},
    numeric::scale_type{0}};
  auto expected_fail = decimal_column_wrapper<T>{
    {3, 20, 1, 50, 0, 1, 0}, {1, 1, 1, 1, 0, 1, 1}, numeric::scale_type{0}};
  auto table      = cudf::table_view{{a, a_fail}};
  auto a_ref      = cudf::ast::column_reference(0);
  auto a_fail_ref = cudf::ast::column_reference(1);
  auto tree       = cudf::ast::tree{};
  auto& abs       = cudf::ast::jit::operation(tree, cudf::ast::jit::op::ABS_OVERFLOW, {a_ref});
  auto& abs_fail  = cudf::ast::jit::operation(tree, cudf::ast::jit::op::ABS_OVERFLOW, {a_fail_ref});
  auto& try_abs_fail = cudf::ast::jit::operation(
    tree, cudf::ast::jit::op::ABS_OVERFLOW, {a_fail_ref}, cudf::error_policy::NULLIFY);
  auto result      = cudf::compute_column_jit(table, abs);
  auto result_fail = cudf::compute_column_jit(table, try_abs_fail);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view(), VERBOSITY);

  EXPECT_THROW(result = cudf::compute_column_jit(table, abs_fail), cudf::evaluation_error);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_fail, result_fail->view(), VERBOSITY);
}

TYPED_TEST(JITSignedIntegerArithmeticTest, NegOverflow)
{
  using T       = TypeParam;
  auto a        = column_wrapper<T>{{T{3}, T{-20}, T{1}, T{-50}, this->MAX, T{-this->MAX}, T{0}}};
  auto a_fail   = column_wrapper<T>{{T{3}, T{-20}, T{1}, T{-50}, this->MIN, T{1}, T{0}}};
  auto expected = column_wrapper<T>{{T{-3}, T{20}, T{-1}, T{50}, T{-this->MAX}, this->MAX, T{0}}};
  auto expected_fail = column_wrapper<T>{{-3, 20, -1, 50, 0, -1, 0}, {1, 1, 1, 1, 0, 1, 1}};
  auto table         = cudf::table_view{{a, a_fail}};
  auto a_ref         = cudf::ast::column_reference(0);
  auto a_fail_ref    = cudf::ast::column_reference(1);
  auto tree          = cudf::ast::tree{};
  auto& neg          = cudf::ast::jit::operation(tree, cudf::ast::jit::op::NEG_OVERFLOW, {a_ref});
  auto& neg_fail = cudf::ast::jit::operation(tree, cudf::ast::jit::op::NEG_OVERFLOW, {a_fail_ref});
  auto& try_neg_fail = cudf::ast::jit::operation(
    tree, cudf::ast::jit::op::NEG_OVERFLOW, {a_fail_ref}, cudf::error_policy::NULLIFY);
  auto result      = cudf::compute_column_jit(table, neg);
  auto result_fail = cudf::compute_column_jit(table, try_neg_fail);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view(), VERBOSITY);

  EXPECT_THROW(result = cudf::compute_column_jit(table, neg_fail), cudf::evaluation_error);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_fail, result_fail->view(), VERBOSITY);
}

TYPED_TEST(JITDecimalArithmeticTest, NegOverflow)
{
  using T = TypeParam;
  using R = typename T::rep;
  auto a  = decimal_column_wrapper<T>{{R{3}, R{-20}, R{1}, R{-50}, this->MAX, R{-this->MAX}, R{0}},
                                      numeric::scale_type{0}};
  auto a_fail   = decimal_column_wrapper<T>{{R{3}, R{-20}, R{1}, R{-50}, this->MIN, R{1}, R{0}},
                                            numeric::scale_type{0}};
  auto expected = decimal_column_wrapper<T>{
    {R{-3}, R{20}, R{-1}, R{50}, R{-this->MAX}, this->MAX, R{0}}, numeric::scale_type{0}};
  auto expected_fail = decimal_column_wrapper<T>{
    {-3, 20, -1, 50, 0, -1, 0}, {1, 1, 1, 1, 0, 1, 1}, numeric::scale_type{0}};
  auto table      = cudf::table_view{{a, a_fail}};
  auto a_ref      = cudf::ast::column_reference(0);
  auto a_fail_ref = cudf::ast::column_reference(1);
  auto tree       = cudf::ast::tree{};
  auto& neg       = cudf::ast::jit::operation(tree, cudf::ast::jit::op::NEG_OVERFLOW, {a_ref});
  auto& neg_fail  = cudf::ast::jit::operation(tree, cudf::ast::jit::op::NEG_OVERFLOW, {a_fail_ref});
  auto& try_neg_fail = cudf::ast::jit::operation(
    tree, cudf::ast::jit::op::NEG_OVERFLOW, {a_fail_ref}, cudf::error_policy::NULLIFY);
  auto result      = cudf::compute_column_jit(table, neg);
  auto result_fail = cudf::compute_column_jit(table, try_neg_fail);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view(), VERBOSITY);

  EXPECT_THROW(result = cudf::compute_column_jit(table, neg_fail), cudf::evaluation_error);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_fail, result_fail->view(), VERBOSITY);
}

TYPED_TEST(JITDecimalArithmeticTest, CheckPrecision)
{
  using T       = TypeParam;
  auto a        = decimal_column_wrapper<T>{{3, 200, 250, 200}, numeric::scale_type{0}};
  auto a_fail   = decimal_column_wrapper<T>{{3, 200, 250, 20000}, numeric::scale_type{0}};
  auto expected = decimal_column_wrapper<T>{{3, 200, 250, 200}, numeric::scale_type{0}};
  auto expected_fail =
    decimal_column_wrapper<T>{{3, 200, 250, 200}, {1, 1, 1, 0}, numeric::scale_type{0}};
  auto max_precision = cudf::numeric_scalar<int32_t>(3);
  auto table         = cudf::table_view{{a, a_fail}};
  auto a_ref         = cudf::ast::column_reference(0);
  auto a_fail_ref    = cudf::ast::column_reference(1);
  auto tree          = cudf::ast::tree{};
  auto precision     = cudf::ast::literal(max_precision);
  auto& check_precision =
    cudf::ast::jit::operation(tree, cudf::ast::jit::op::CHECK_PRECISION, {a_ref, precision});
  auto& check_precision_fail =
    cudf::ast::jit::operation(tree, cudf::ast::jit::op::CHECK_PRECISION, {a_fail_ref, precision});
  auto& try_check_precision = cudf::ast::jit::operation(tree,
                                                        cudf::ast::jit::op::CHECK_PRECISION,
                                                        {a_fail_ref, precision},
                                                        cudf::error_policy::NULLIFY);
  auto result               = cudf::compute_column_jit(table, check_precision);
  auto result_fail          = cudf::compute_column_jit(table, try_check_precision);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view(), VERBOSITY);

  EXPECT_THROW(result = cudf::compute_column_jit(table, check_precision_fail),
               cudf::evaluation_error);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_fail, result_fail->view(), VERBOSITY);
}

TEST_F(JITExpressionTest, BitShiftLeft)
{
  auto a             = column_wrapper<uint32_t>{0b111111, 0b111110, 0b101111, 0b1100};
  auto expected      = column_wrapper<uint32_t>{0b11111100, 0b11111000, 0b10111100, 0b110000};
  auto shift         = cudf::numeric_scalar<uint32_t>(2);
  auto table         = cudf::table_view{{a}};
  auto a_ref         = cudf::ast::column_reference(0);
  auto tree          = cudf::ast::tree{};
  auto shift_literal = cudf::ast::literal(shift);
  auto& shift_left =
    cudf::ast::jit::operation(tree, cudf::ast::jit::op::BITWISE_SHIFT_LEFT, {a_ref, shift_literal});
  auto result = cudf::compute_column_jit(table, shift_left);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view(), VERBOSITY);
}

TEST_F(JITExpressionTest, BitShiftRight)
{
  auto a             = column_wrapper<uint32_t>{0b1111, 0b10111, 0b11100, 0b11110011};
  auto expected      = column_wrapper<uint32_t>{0b11, 0b101, 0b111, 0b111100};
  auto shift         = cudf::numeric_scalar<uint32_t>(2);
  auto table         = cudf::table_view{{a}};
  auto a_ref         = cudf::ast::column_reference(0);
  auto tree          = cudf::ast::tree{};
  auto shift_literal = cudf::ast::literal(shift);
  auto& shift_right  = cudf::ast::jit::operation(
    tree, cudf::ast::jit::op::BITWISE_SHIFT_RIGHT, {a_ref, shift_literal});
  auto result = cudf::compute_column_jit(table, shift_right);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view(), VERBOSITY);
}

template <typename To>
constexpr cudf::ast::jit::op get_cast_op()
{
  using enum cudf::ast::jit::op;
  if constexpr (std::is_same_v<To, bool>) {
    return CAST_TO_BOOL8;
  } else if constexpr (std::is_same_v<To, int8_t>) {
    return CAST_TO_INT8;
  } else if constexpr (std::is_same_v<To, int16_t>) {
    return CAST_TO_INT16;
  } else if constexpr (std::is_same_v<To, int32_t>) {
    return CAST_TO_INT32;
  } else if constexpr (std::is_same_v<To, int64_t>) {
    return CAST_TO_INT64;
  } else if constexpr (std::is_same_v<To, uint8_t>) {
    return CAST_TO_UINT8;
  } else if constexpr (std::is_same_v<To, uint16_t>) {
    return CAST_TO_UINT16;
  } else if constexpr (std::is_same_v<To, uint32_t>) {
    return CAST_TO_UINT32;
  } else if constexpr (std::is_same_v<To, uint64_t>) {
    return CAST_TO_UINT64;
  } else if constexpr (std::is_same_v<To, float>) {
    return CAST_TO_FLOAT32;
  } else if constexpr (std::is_same_v<To, double>) {
    return CAST_TO_FLOAT64;
  } else if constexpr (std::is_same_v<To, numeric::decimal32>) {
    return CAST_TO_DECIMAL32;
  } else if constexpr (std::is_same_v<To, numeric::decimal64>) {
    return CAST_TO_DECIMAL64;
  } else if constexpr (std::is_same_v<To, numeric::decimal128>) {
    static_assert(std::is_same_v<To, numeric::decimal128>);
    return CAST_TO_DECIMAL128;
  }
}

template <typename From, typename To>
void test_cast()
{
  auto a        = column_wrapper<From>{{0, 1, 2, 3, 4, 5}};
  auto expected = column_wrapper<To>{{0, 1, 2, 3, 4, 5}};
  auto table    = cudf::table_view{{a}};
  auto a_ref    = cudf::ast::column_reference(0);
  auto tree     = cudf::ast::tree{};
  auto result =
    cudf::compute_column_jit(table, cudf::ast::jit::operation(tree, get_cast_op<To>(), {a_ref}));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view(), VERBOSITY);
}

template <typename From, typename To>
void test_from_decimal_cast()
{
  auto a        = decimal_column_wrapper<From>{{0, 1, 2, 3, 4, 5}, numeric::scale_type{0}};
  auto expected = column_wrapper<To>{0, 1, 2, 3, 4, 5};
  auto table    = cudf::table_view{{a}};
  auto a_ref    = cudf::ast::column_reference(0);
  auto tree     = cudf::ast::tree{};
  auto result =
    cudf::compute_column_jit(table, cudf::ast::jit::operation(tree, get_cast_op<To>(), {a_ref}));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view(), VERBOSITY);
}

template <typename To>
void test_cast_to()
{
  test_cast<uint8_t, To>();
  test_cast<uint16_t, To>();
  test_cast<uint32_t, To>();
  test_cast<uint64_t, To>();
  test_cast<int8_t, To>();
  test_cast<int16_t, To>();
  test_cast<int32_t, To>();
  test_cast<int64_t, To>();
  test_cast<float, To>();
  test_cast<double, To>();
  test_from_decimal_cast<numeric::decimal32, To>();
  test_from_decimal_cast<numeric::decimal64, To>();
  test_from_decimal_cast<numeric::decimal128, To>();
}

TEST_F(JITExpressionTest, Cast)
{
  test_cast_to<bool>();
  test_cast_to<int8_t>();
  test_cast_to<int16_t>();
  test_cast_to<int32_t>();
  test_cast_to<int64_t>();
  test_cast_to<uint8_t>();
  test_cast_to<uint16_t>();
  test_cast_to<uint32_t>();
  test_cast_to<uint64_t>();
  test_cast_to<float>();
  test_cast_to<double>();
}

template <typename From, typename To>
void test_decimal_cast()
{
  auto a        = decimal_column_wrapper<From>{{0, 1, 2, 3, 4, 5}, numeric::scale_type{0}};
  auto expected = decimal_column_wrapper<To>{{0, 1, 2, 3, 4, 5}, numeric::scale_type{0}};
  auto table    = cudf::table_view{{a}};
  auto a_ref    = cudf::ast::column_reference(0);
  auto tree     = cudf::ast::tree{};
  auto result =
    cudf::compute_column_jit(table, cudf::ast::jit::operation(tree, get_cast_op<To>(), {a_ref}));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view(), VERBOSITY);
}

TYPED_TEST(JITDecimalArithmeticTest, CastTo)
{
  using T = TypeParam;
  test_decimal_cast<T, numeric::decimal32>();
  test_decimal_cast<T, numeric::decimal64>();
  test_decimal_cast<T, numeric::decimal128>();
}

TEST_F(JITExpressionTest, Rescale)
{
  auto a = cudf::test::fixed_point_column_wrapper<int32_t>{{123, 1234, 12345, 123456, 1234567},
                                                           numeric::scale_type{0}};
  auto expected = cudf::test::fixed_point_column_wrapper<int32_t>{
    {12300, 123400, 1234500, 12345600, 123456700}, numeric::scale_type{-2}};
  auto table     = cudf::table_view{{a}};
  auto a_ref     = cudf::ast::column_reference(0);
  auto tree      = cudf::ast::tree{};
  auto& rescaled = cudf::ast::jit::operation(
    tree, cudf::ast::jit::op::RESCALE, {a_ref}, cudf::error_policy::PROPAGATE, -2);
  auto result = cudf::compute_column_jit(table, rescaled);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view(), VERBOSITY);
}

TEST_F(JITExpressionTest, OverflowFused)
{
  constexpr auto I32_MAX = std::numeric_limits<int32_t>::max();
  auto a                 = column_wrapper<int32_t>{{1, 3, 20, 1, 50, 10}};
  auto b                 = column_wrapper<int32_t>{{1, 10, 7, 20, I32_MAX, 2}};
  auto c                 = column_wrapper<int32_t>{{1, 5, 4, I32_MAX, 2, 5}};
  auto d                 = column_wrapper<int32_t>{{0, 1, 0, 0, 1, 5}};
  auto expected          = column_wrapper<int32_t>{{0, 65, 0, 0, 0, 12}, {0, 1, 0, 0, 0, 1}};
  auto table             = cudf::table_view{{a, b, c, d}};
  auto tree              = cudf::ast::tree{};
  auto a_ref             = cudf::ast::column_reference(0);
  auto b_ref             = cudf::ast::column_reference(1);
  auto c_ref             = cudf::ast::column_reference(2);
  auto d_ref             = cudf::ast::column_reference(3);
  auto& add              = cudf::ast::jit::operation(
    tree, cudf::ast::jit::op::ADD_OVERFLOW, {a_ref, b_ref}, cudf::error_policy::NULLIFY);
  auto& mul = cudf::ast::jit::operation(
    tree, cudf::ast::jit::op::MUL_OVERFLOW, {add, c_ref}, cudf::error_policy::NULLIFY);
  auto& div = cudf::ast::jit::operation(
    tree, cudf::ast::jit::op::DIV_OVERFLOW, {mul, d_ref}, cudf::error_policy::NULLIFY);
  auto result = cudf::compute_column_jit(table, div);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view(), VERBOSITY);
}
