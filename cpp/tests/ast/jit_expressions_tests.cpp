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
#include <cudf/copying.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/filling.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/transform.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/cuda_stream.hpp>

#include <cuda/iterator>
#include <cuda_runtime_api.h>

#include <array>
#include <functional>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

constexpr cudf::test::debug_output_level VERBOSITY{cudf::test::debug_output_level::ALL_ERRORS};

template <typename T>
using column_wrapper = cudf::test::fixed_width_column_wrapper<T>;

template <typename T>
using decimal_column_wrapper = cudf::test::fixed_point_column_wrapper<typename T::rep>;

struct JITExpressionTest : public cudf::test::BaseFixture {};

TEST_F(JITExpressionTest, LtoKernelCacheIdentity)
{
  constexpr auto max = std::numeric_limits<int32_t>::max();
  auto a             = column_wrapper<int32_t>{{1, max, 3}, {1, 1, 0}};
  auto b             = column_wrapper<int32_t>{2, 1, 4};
  auto expected_sub  = column_wrapper<int32_t>{{-1, max - 1, 0}, {1, 1, 0}};
  auto expected_null = column_wrapper<int32_t>{{3, 0, 0}, {1, 0, 0}};
  auto table         = cudf::table_view{{a, b}};
  auto tree          = cudf::ast::tree{};
  auto a_ref         = cudf::ast::column_reference(0);
  auto b_ref         = cudf::ast::column_reference(1);
  auto& subtract     = cudf::ast::jit::operation(tree, cudf::ast::jit::op::SUB, {a_ref, b_ref});
  auto& nullified    = cudf::ast::jit::operation(
    tree, cudf::ast::jit::op::ADD_OVERFLOW, {a_ref, b_ref}, cudf::error_policy::NULLIFY);

  std::array<std::reference_wrapper<cudf::ast::expression const>, 2> expressions{subtract,
                                                                                 nullified};
  auto result = cudf::compute_table_jit(table, expressions);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_sub, result->view().column(0), VERBOSITY);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_null, result->view().column(1), VERBOSITY);

  auto expected_mul = column_wrapper<int32_t>{{2, max, 0}, {1, 1, 0}};
  auto& multiply    = cudf::ast::jit::operation(tree, cudf::ast::jit::op::MUL, {a_ref, b_ref});
  std::array<std::reference_wrapper<cudf::ast::expression const>, 2> second_expressions{multiply,
                                                                                        subtract};
  auto second_result = cudf::compute_table_jit(table, second_expressions);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_mul, second_result->view().column(0), VERBOSITY);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_sub, second_result->view().column(1), VERBOSITY);

  // Revisiting an earlier operation after another kernel is cached catches identity collisions.
  auto repeated_result = cudf::compute_table_jit(table, expressions);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_sub, repeated_result->view().column(0), VERBOSITY);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_null, repeated_result->view().column(1), VERBOSITY);
}

TEST_F(JITExpressionTest, LtoIntegerOverflow)
{
  cudf::ast::tree tree;
  std::vector<std::unique_ptr<cudf::column>> inputs;
  std::vector<std::reference_wrapper<cudf::ast::expression const>> throwing;
  auto append = [&]<typename T>() {
    auto const base = static_cast<cudf::size_type>(inputs.size());
    inputs.push_back(column_wrapper<T>{{1, 2, 3}, {1, 1, 0}}.release());
    inputs.push_back(
      column_wrapper<T>{{T{1}, std::numeric_limits<T>::max(), T{3}}, {1, 1, 0}}.release());
    inputs.push_back(column_wrapper<T>{2, 1, 4}.release());
    auto& fail = tree.push(cudf::ast::column_reference(base + 1));
    auto& rhs  = tree.push(cudf::ast::column_reference(base + 2));
    throwing.emplace_back(cudf::ast::jit::operation(
      tree, cudf::ast::jit::op::ADD_OVERFLOW, {fail, rhs}, cudf::error_policy::PROPAGATE));
  };
  [&]<typename... T>(cudf::test::Types<T...>) {
    (append.template operator()<T>(), ...);
  }(cudf::test::IntegralTypesNotBool{});
  std::vector<cudf::column_view> views;
  for (auto const& column : inputs) {
    views.push_back(column->view());
  }
  // An earlier PROPAGATE failure must not mask an unchecked later type. Substituting safe
  // inputs preserves the nullable schema and expression graph, so every check reuses one kernel.
  for (size_t base = 0; base < views.size(); base += 3) {
    views[base + 1] = views[base];
  }
  ASSERT_NO_THROW(cudf::compute_table_jit(cudf::table_view{views}, throwing));
  for (size_t base = 0; base < views.size(); base += 3) {
    SCOPED_TRACE(cudf::type_to_name(views[base].type()));
    views[base + 1] = inputs[base + 1]->view();
    EXPECT_THROW(cudf::compute_table_jit(cudf::table_view{views}, throwing),
                 cudf::evaluation_error);
    views[base + 1] = views[base];
  }
}

TEST_F(JITExpressionTest, LtoNullableGridStrideAndTails)
{
  int device{}, multiprocessors{}, threads_per_multiprocessor{};
  ASSERT_EQ(cudaSuccess, cudaGetDevice(&device));
  ASSERT_EQ(cudaSuccess,
            cudaDeviceGetAttribute(&multiprocessors, cudaDevAttrMultiProcessorCount, device));
  ASSERT_EQ(cudaSuccess,
            cudaDeviceGetAttribute(
              &threads_per_multiprocessor, cudaDevAttrMaxThreadsPerMultiProcessor, device));
  // Exceed the occupancy grid's resident-thread capacity to exercise repeated loop iterations.
  auto const large_rows = 2 * multiprocessors * threads_per_multiprocessor + 31;
  auto values           = cudf::detail::make_counting_transform_iterator(0, [](auto i) -> int32_t {
    return i % 67 == 0 ? std::numeric_limits<int32_t>::max() : i % 251 - 100;
  });
  auto validity =
    cudf::detail::make_counting_transform_iterator(0, [](auto i) { return i % 7 != 0; });
  auto expected_sub =
    cudf::detail::make_counting_transform_iterator(0, [values](auto i) { return values[i] - 1; });
  auto expected_add = cudf::detail::make_counting_transform_iterator(
    0, [values](auto i) { return i % 67 == 0 ? 0 : values[i] + 1; });
  auto add_validity = cudf::detail::make_counting_transform_iterator(
    0, [](auto i) { return i % 7 != 0 && i % 67 != 0; });
  auto scalar     = cudf::numeric_scalar<int32_t>(1);
  auto tree       = cudf::ast::tree{};
  auto input_ref  = cudf::ast::column_reference(0);
  auto& literal   = tree.push(cudf::ast::literal(scalar));
  auto& subtract  = cudf::ast::jit::operation(tree, cudf::ast::jit::op::SUB, {input_ref, literal});
  auto& nullified = cudf::ast::jit::operation(
    tree, cudf::ast::jit::op::ADD_OVERFLOW, {input_ref, literal}, cudf::error_policy::NULLIFY);
  std::array<std::reference_wrapper<cudf::ast::expression const>, 2> expressions{subtract,
                                                                                 nullified};
  for (auto const rows : std::array{31, 32, 33, large_rows}) {
    SCOPED_TRACE(rows);
    auto input               = column_wrapper<int32_t>(values, values + rows, validity);
    auto expected_sub_column = column_wrapper<int32_t>(expected_sub, expected_sub + rows, validity);
    auto expected_add_column =
      column_wrapper<int32_t>(expected_add, expected_add + rows, add_validity);
    auto result = cudf::compute_table_jit(cudf::table_view{{input}}, expressions);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_sub_column, result->view().column(0), VERBOSITY);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_add_column, result->view().column(1), VERBOSITY);
  }
}

TEST_F(JITExpressionTest, LtoSlicedInputsAndScalars)
{
  std::vector<float> values(70);
  std::vector<double> doubles(70);
  std::vector<bool> flags(70);
  std::vector<bool> validity(70);
  for (size_t i = 0; i < values.size(); ++i) {
    values[i]   = static_cast<float>(i);
    doubles[i]  = static_cast<double>(i);
    flags[i]    = i % 2 == 0;
    validity[i] = i % 7 != 0;
  }
  auto floats        = column_wrapper<float>(values.begin(), values.end(), validity.begin());
  auto wide          = column_wrapper<double>(doubles.begin(), doubles.end(), validity.begin());
  auto booleans      = column_wrapper<bool>(flags.begin(), flags.end(), validity.begin());
  auto input         = cudf::slice(cudf::table_view{{floats, wide, booleans}}, {3, 68}).front();
  auto scalar        = cudf::numeric_scalar<float>(2);
  auto wide_scalar   = cudf::numeric_scalar<double>(2);
  auto tree          = cudf::ast::tree{};
  auto float_ref     = cudf::ast::column_reference(0);
  auto double_ref    = cudf::ast::column_reference(1);
  auto bool_ref      = cudf::ast::column_reference(2);
  auto& literal      = tree.push(cudf::ast::literal(scalar));
  auto& wide_literal = tree.push(cudf::ast::literal(wide_scalar));
  auto& add = cudf::ast::jit::operation(tree, cudf::ast::jit::op::ADD, {float_ref, literal});
  auto& wide_add =
    cudf::ast::jit::operation(tree, cudf::ast::jit::op::ADD, {double_ref, wide_literal});
  std::array<std::reference_wrapper<cudf::ast::expression const>, 3> expressions{
    add, wide_add, bool_ref};
  auto result = cudf::compute_table_jit(input, expressions);
  for (auto& value : values) {
    value += 2;
  }
  for (auto& value : doubles) {
    value += 2;
  }
  auto expected_float =
    column_wrapper<float>(values.begin() + 3, values.begin() + 68, validity.begin() + 3);
  auto expected_double =
    column_wrapper<double>(doubles.begin() + 3, doubles.begin() + 68, validity.begin() + 3);
  auto expected_bool =
    column_wrapper<bool>(flags.begin() + 3, flags.begin() + 68, validity.begin() + 3);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_float, result->view().column(0), VERBOSITY);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_double, result->view().column(1), VERBOSITY);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_bool, result->view().column(2), VERBOSITY);

  scalar.set_valid_async(false);
  auto null_result = cudf::compute_table_jit(input, expressions);
  EXPECT_EQ(input.num_rows(), null_result->view().column(0).null_count());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_double, null_result->view().column(1), VERBOSITY);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_bool, null_result->view().column(2), VERBOSITY);

  auto empty        = cudf::slice(input, {0, 0}).front();
  auto empty_result = cudf::compute_table_jit(empty, expressions);
  EXPECT_EQ(0, empty_result->num_rows());
  EXPECT_EQ(3, empty_result->num_columns());
}

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

struct overflow_expressions {
  cudf::ast::expression const& success;
  cudf::ast::expression const& throwing;
  cudf::ast::expression const& nullified;
};

void expect_overflow_results(cudf::table_view const& table,
                             overflow_expressions ops,
                             cudf::column_view expected,
                             cudf::column_view expected_fail)
{
  // Successful and NULLIFY expressions can share one JIT kernel. The throwing path must be
  // evaluated independently to verify its error policy without discarding the successful outputs.
  auto expressions = std::to_array<std::reference_wrapper<cudf::ast::expression const>>(
    {ops.success, ops.nullified});
  auto result = cudf::compute_table_jit(table, expressions);

  ASSERT_EQ(result->num_columns(), static_cast<cudf::size_type>(expressions.size()));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view().column(0), VERBOSITY);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_fail, result->view().column(1), VERBOSITY);
  EXPECT_THROW(cudf::compute_column_jit(table, ops.throwing), cudf::evaluation_error);
}

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
  expect_overflow_results(table,
                          {.success = add, .throwing = add_fail, .nullified = try_add_fail},
                          expected,
                          expected_fail);
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
  expect_overflow_results(table,
                          {.success = add, .throwing = add_fail, .nullified = try_add_fail},
                          expected,
                          expected_fail);
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

  expect_overflow_results(table,
                          {.success = sub, .throwing = sub_fail, .nullified = try_sub_fail},
                          expected,
                          expected_fail);
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
  expect_overflow_results(table,
                          {.success = sub, .throwing = sub_fail, .nullified = try_sub_fail},
                          expected,
                          expected_fail);
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
  expect_overflow_results(table,
                          {.success = mul, .throwing = mul_fail, .nullified = try_mul_fail},
                          expected,
                          expected_fail);
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

  expect_overflow_results(table,
                          {.success = mul, .throwing = mul_fail, .nullified = try_mul_fail},
                          expected,
                          expected_fail);
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
  expect_overflow_results(table,
                          {.success = div, .throwing = div_fail, .nullified = try_div_fail},
                          expected,
                          expected_fail);
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
  expect_overflow_results(table,
                          {.success = div, .throwing = div_fail, .nullified = try_div_fail},
                          expected,
                          expected_fail);
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
  expect_overflow_results(table,
                          {.success = mod, .throwing = mod_fail, .nullified = try_mod_fail},
                          expected,
                          expected_fail);
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
  expect_overflow_results(table,
                          {.success = mod, .throwing = mod_fail, .nullified = try_mod_fail},
                          expected,
                          expected_fail);
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
  expect_overflow_results(table,
                          {.success = abs, .throwing = abs_fail, .nullified = try_abs_fail},
                          expected,
                          expected_fail);
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
  expect_overflow_results(table,
                          {.success = abs, .throwing = abs_fail, .nullified = try_abs_fail},
                          expected,
                          expected_fail);
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
  expect_overflow_results(table,
                          {.success = neg, .throwing = neg_fail, .nullified = try_neg_fail},
                          expected,
                          expected_fail);
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
  expect_overflow_results(table,
                          {.success = neg, .throwing = neg_fail, .nullified = try_neg_fail},
                          expected,
                          expected_fail);
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
  expect_overflow_results(table,
                          {.success   = check_precision,
                           .throwing  = check_precision_fail,
                           .nullified = try_check_precision},
                          expected,
                          expected_fail);
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

template <typename T, typename Values>
std::unique_ptr<cudf::column> make_cast_input(Values const& values)
{
  if constexpr (cudf::is_fixed_point<T>()) {
    return decimal_column_wrapper<T>(values.begin(), values.end(), numeric::scale_type{0})
      .release();
  } else {
    return column_wrapper<T>(values.begin(), values.end()).release();
  }
}

template <typename ToTypes, typename FromTypes>
struct cast_test;

template <typename... To, typename... From>
struct cast_test<cudf::test::Types<To...>, cudf::test::Types<From...>> {
  static void run()
  {
    auto const values = std::array{0, 1, 2, 3, 4, 5};

    auto columns = std::vector<std::unique_ptr<cudf::column>>{};
    columns.reserve(sizeof...(From));
    (columns.push_back(make_cast_input<From>(values)), ...);
    auto table = cudf::table{std::move(columns)};

    auto tree = cudf::ast::tree{};
    auto refs = std::vector<std::reference_wrapper<cudf::ast::expression const>>{};
    refs.reserve(table.num_columns());
    for (cudf::size_type i = 0; i < table.num_columns(); ++i) {
      refs.emplace_back(tree.push(cudf::ast::column_reference(i)));
    }
    auto expressions = std::vector<std::reference_wrapper<cudf::ast::expression const>>{};
    expressions.reserve(sizeof...(To) * sizeof...(From));
    (append_expressions<To>(tree, refs, expressions), ...);
    auto result = cudf::compute_table_jit(table.view(), expressions);

    ASSERT_EQ(result->num_columns(), static_cast<cudf::size_type>(expressions.size()));
    auto output_index = cudf::size_type{0};
    (expect_results<To>(result->view(), values, output_index), ...);
  }

 private:
  template <typename ToType>
  static void append_expressions(
    cudf::ast::tree& tree,
    std::vector<std::reference_wrapper<cudf::ast::expression const>> const& refs,
    std::vector<std::reference_wrapper<cudf::ast::expression const>>& expressions)
  {
    auto const op = get_cast_op<ToType>();
    for (auto const& ref : refs) {
      expressions.emplace_back(cudf::ast::jit::operation(tree, op, {ref}));
    }
  }

  template <typename ToType, typename Values>
  static void expect_results(cudf::table_view const& result,
                             Values const& values,
                             cudf::size_type& output_index)
  {
    static auto const from_names =
      std::array{cudf::type_to_name(cudf::data_type{cudf::type_to_id<From>()})...};
    static auto const to_name = cudf::type_to_name(cudf::data_type{cudf::type_to_id<ToType>()});
    auto expected             = make_cast_input<ToType>(values);
    for (auto const& from_name : from_names) {
      SCOPED_TRACE(std::to_string(output_index) + ": " + from_name + " -> " + to_name);
      CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected->view(), result.column(output_index), VERBOSITY);
      ++output_index;
    }
  }
};

template <typename ToTypes, typename FromTypes>
void test_casts()
{
  cast_test<ToTypes, FromTypes>::run();
}

using standard_cast_sources = cudf::test::Types<uint8_t,
                                                uint16_t,
                                                uint32_t,
                                                uint64_t,
                                                int8_t,
                                                int16_t,
                                                int32_t,
                                                int64_t,
                                                float,
                                                double,
                                                numeric::decimal32,
                                                numeric::decimal64,
                                                numeric::decimal128>;
using decimal_cast_sources =
  cudf::test::Types<numeric::decimal32, numeric::decimal64, numeric::decimal128>;

TEST_F(JITExpressionTest, Cast)
{
  test_casts<cudf::test::Types<bool, int8_t, int16_t>, standard_cast_sources>();
  test_casts<cudf::test::Types<int32_t, int64_t, uint8_t>, standard_cast_sources>();
  test_casts<cudf::test::Types<uint16_t, uint32_t, uint64_t>, standard_cast_sources>();
  test_casts<cudf::test::Types<float, double>, standard_cast_sources>();
}

TEST_F(JITExpressionTest, DecimalCast)
{
  test_casts<cudf::test::Types<numeric::decimal32, numeric::decimal64, numeric::decimal128>,
             decimal_cast_sources>();
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
