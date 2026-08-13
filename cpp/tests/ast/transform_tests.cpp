/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/iterator_utilities.hpp>
#include <cudf_test/testing_main.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/ast/expressions.hpp>
#include <cudf/binaryop.hpp>
#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/filling.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/transform.hpp>

#include <rmm/cuda_stream.hpp>

#include <cuda/iterator>

#include <algorithm>
#include <limits>
#include <random>
#include <vector>

// NOTE: each test in this file must be run twice - once with the AST Interpreter executor
// (`executor_ast`) and once with the JIT executor (`executor_jit`). This is intended to ensure
// behavioural compatibility between the two executors.

template <typename T>
using column_wrapper = cudf::test::fixed_width_column_wrapper<T>;

constexpr cudf::test::debug_output_level verbosity{cudf::test::debug_output_level::ALL_ERRORS};

template <typename T>
struct TransformTest : public cudf::test::BaseFixture {};

struct executor_ast {
  static std::unique_ptr<cudf::column> compute_column(
    cudf::table_view const& table,
    cudf::ast::expression const& expr,
    rmm::cuda_stream_view stream      = cudf::get_default_stream(),
    rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref())
  {
    return cudf::compute_column(table, expr, stream, mr);
  }
};

struct executor_jit {
  static std::unique_ptr<cudf::column> compute_column(
    cudf::table_view const& table,
    cudf::ast::expression const& expr,
    rmm::cuda_stream_view stream      = cudf::get_default_stream(),
    rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref())
  {
    return cudf::compute_column_jit(table, expr, stream, mr);
  }
};

using Executors = cudf::test::Types<executor_ast, executor_jit>;

TYPED_TEST_SUITE(TransformTest, Executors);

TYPED_TEST(TransformTest, ColumnReference)
{
  using Executor = TypeParam;

  auto c_0   = column_wrapper<int32_t>{3, 20, 1, 50};
  auto c_1   = column_wrapper<int32_t>{10, 7, 20, 0};
  auto table = cudf::table_view{{c_0, c_1}};

  auto col_ref_0 = cudf::ast::column_reference(0);

  auto const& expected = c_0;
  auto result          = Executor::compute_column(table, col_ref_0);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view(), verbosity);
}

TYPED_TEST(TransformTest, BasicAdditionDoubleCast)
{
  using Executor = TypeParam;

  auto c_0 = column_wrapper<double>{3, 20, 1, 50};
  std::vector<__int128_t> data1{10, 7, 20, 0};
  auto c_1 = cudf::test::fixed_point_column_wrapper<__int128_t>(
    data1.begin(), data1.end(), numeric::scale_type{0});
  auto table      = cudf::table_view{{c_0, c_1}};
  auto col_ref_0  = cudf::ast::column_reference(0);
  auto col_ref_1  = cudf::ast::column_reference(1);
  auto cast       = cudf::ast::operation(cudf::ast::ast_operator::CAST_TO_FLOAT64, col_ref_1);
  auto expression = cudf::ast::operation(cudf::ast::ast_operator::ADD, col_ref_0, cast);
  auto expected   = column_wrapper<double>{13, 27, 21, 50};
  auto result     = Executor::compute_column(table, expression);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view(), verbosity);
}

TYPED_TEST(TransformTest, Literal)
{
  using Executor = TypeParam;

  auto c_0   = column_wrapper<int32_t>{3, 20, 1, 50};
  auto c_1   = column_wrapper<int32_t>{10, 7, 20, 0};
  auto table = cudf::table_view{{c_0, c_1}};

  auto literal_value = cudf::numeric_scalar<int32_t>(42);
  auto literal       = cudf::ast::literal(literal_value);

  auto expected = column_wrapper<int32_t>{42, 42, 42, 42};
  auto result   = Executor::compute_column(table, literal);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view(), verbosity);
}

TYPED_TEST(TransformTest, NullLiteral)
{
  using Executor = TypeParam;

  auto c_0   = column_wrapper<int32_t>{0, 0, 0, 0};
  auto table = cudf::table_view{{c_0}};

  auto literal_value = cudf::numeric_scalar<int32_t>(-123);
  literal_value.set_valid_async(false);
  auto literal = cudf::ast::literal(literal_value);

  auto expression = cudf::ast::operation(cudf::ast::ast_operator::IDENTITY, literal);

  auto result   = Executor::compute_column(table, expression);
  auto expected = column_wrapper<int32_t>({-123, -123, -123, -123}, {0, 0, 0, 0});

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view(), verbosity);
}

TYPED_TEST(TransformTest, IsNull)
{
  using Executor = TypeParam;

  auto c_0   = column_wrapper<int32_t>{{0, 1, 2, 0}, {0, 1, 1, 0}};
  auto table = cudf::table_view{{c_0}};

  // result of IS_NULL on literal, will be a column of table size, with all values set to
  // !literal.is_valid(). The table values are irrelevant.
  auto literal_value = cudf::numeric_scalar<int32_t>(-123);
  auto literal       = cudf::ast::literal(literal_value);
  auto expression    = cudf::ast::operation(cudf::ast::ast_operator::IS_NULL, literal);

  auto result    = Executor::compute_column(table, expression);
  auto expected1 = column_wrapper<bool>({0, 0, 0, 0});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected1, result->view(), verbosity);

  literal_value.set_valid_async(false);
  result         = Executor::compute_column(table, expression);
  auto expected2 = column_wrapper<bool>({1, 1, 1, 1}, cudf::test::iterators::no_nulls());
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected2, result->view(), verbosity);

  auto col_ref_0   = cudf::ast::column_reference(0);
  auto expression2 = cudf::ast::operation(cudf::ast::ast_operator::IS_NULL, col_ref_0);
  result           = Executor::compute_column(table, expression2);
  auto expected3   = column_wrapper<bool>({1, 0, 0, 1}, cudf::test::iterators::no_nulls());
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected3, result->view(), verbosity);
}

TYPED_TEST(TransformTest, BasicAddition)
{
  using Executor = TypeParam;

  auto c_0   = column_wrapper<int32_t>{3, 20, 1, 50};
  auto c_1   = column_wrapper<int32_t>{10, 7, 20, 0};
  auto table = cudf::table_view{{c_0, c_1}};

  auto col_ref_0  = cudf::ast::column_reference(0);
  auto col_ref_1  = cudf::ast::column_reference(1);
  auto expression = cudf::ast::operation(cudf::ast::ast_operator::ADD, col_ref_0, col_ref_1);

  auto expected = column_wrapper<int32_t>{13, 27, 21, 50};
  auto result   = Executor::compute_column(table, expression);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view(), verbosity);
}

TYPED_TEST(TransformTest, BasicAdditionEmptyTable)
{
  using Executor = TypeParam;

  auto c_0   = column_wrapper<int32_t>{};
  auto c_1   = column_wrapper<int32_t>{};
  auto table = cudf::table_view{{c_0, c_1}};

  auto col_ref_0  = cudf::ast::column_reference(0);
  auto col_ref_1  = cudf::ast::column_reference(1);
  auto expression = cudf::ast::operation(cudf::ast::ast_operator::ADD, col_ref_0, col_ref_1);

  auto expected = column_wrapper<int32_t>{};
  auto result   = Executor::compute_column(table, expression);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view(), verbosity);
}

TYPED_TEST(TransformTest, BasicAdditionCast)
{
  using Executor = TypeParam;

  auto c_0   = column_wrapper<int64_t>{3, 20, 1, 50};
  auto c_1   = column_wrapper<int8_t>{10, 7, 20, 0};
  auto table = cudf::table_view{{c_0, c_1}};

  auto col_ref_0  = cudf::ast::column_reference(0);
  auto col_ref_1  = cudf::ast::column_reference(1);
  auto cast       = cudf::ast::operation(cudf::ast::ast_operator::CAST_TO_INT64, col_ref_1);
  auto expression = cudf::ast::operation(cudf::ast::ast_operator::ADD, col_ref_0, cast);

  auto expected = column_wrapper<int64_t>{13, 27, 21, 50};
  auto result   = Executor::compute_column(table, expression);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view(), verbosity);
}

TYPED_TEST(TransformTest, BasicEquality)
{
  using Executor = TypeParam;

  auto c_0   = column_wrapper<int32_t>{3, 20, 1, 50};
  auto c_1   = column_wrapper<int32_t>{3, 7, 1, 0};
  auto table = cudf::table_view{{c_0, c_1}};

  auto col_ref_0  = cudf::ast::column_reference(0);
  auto col_ref_1  = cudf::ast::column_reference(1);
  auto expression = cudf::ast::operation(cudf::ast::ast_operator::EQUAL, col_ref_0, col_ref_1);

  auto expected = column_wrapper<bool>{true, false, true, false};
  auto result   = Executor::compute_column(table, expression);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view(), verbosity);
}

TYPED_TEST(TransformTest, ScalarBroadcast)
{
  using Executor = TypeParam;

  auto c_0 = column_wrapper<int32_t>{{3, 20, 1, 50, 60}, {0, 0, 1, 0, 1}};
  auto c_1 = column_wrapper<int32_t>{{3, 20, 1, 50, 60}, {0, 0, 0, 0, 0}};
  auto c_2 = column_wrapper<int32_t>{{3, 20, 1, 50, 60}, {1, 1, 1, 1, 1}};
  auto c_3 = column_wrapper<int32_t>{3, 20, 1, 50, 60};

  auto t0 = cudf::table_view{{c_0}};

  auto col_ref_0 = cudf::ast::column_reference(0);
  auto scalar    = cudf::numeric_scalar<int32_t>(42);
  auto literal   = cudf::ast::literal(scalar);

  auto expected = column_wrapper<int32_t>{{42, 42, 42, 42, 42}};

  auto result = Executor::compute_column(t0, literal);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view(), verbosity);

  auto t1 = cudf::table_view{{c_1}};
  result  = Executor::compute_column(t1, literal);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view(), verbosity);

  auto t2 = cudf::table_view{{c_2}};
  result  = Executor::compute_column(t2, literal);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view(), verbosity);

  auto t3 = cudf::table_view{{c_3}};
  result  = Executor::compute_column(t3, literal);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view(), verbosity);
}

TYPED_TEST(TransformTest, BasicAdditionLarge)
{
  using Executor = TypeParam;

  auto zero = cudf::numeric_scalar<int32_t>(0);
  auto two  = cudf::numeric_scalar<int32_t>(2);

  auto col   = cudf::sequence(2000, zero);
  auto table = cudf::table_view{{col->view(), col->view()}};

  auto col_ref    = cudf::ast::column_reference(0);
  auto expression = cudf::ast::operation(cudf::ast::ast_operator::ADD, col_ref, col_ref);

  auto expected = cudf::sequence(2000, zero, two);
  auto result   = Executor::compute_column(table, expression);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected->view(), result->view(), verbosity);
}

TYPED_TEST(TransformTest, LessComparator)
{
  using Executor = TypeParam;

  auto c_0   = column_wrapper<int32_t>{3, 20, 1, 50};
  auto c_1   = column_wrapper<int32_t>{10, 7, 20, 0};
  auto table = cudf::table_view{{c_0, c_1}};

  auto col_ref_0  = cudf::ast::column_reference(0);
  auto col_ref_1  = cudf::ast::column_reference(1);
  auto expression = cudf::ast::operation(cudf::ast::ast_operator::LESS, col_ref_0, col_ref_1);

  auto expected = column_wrapper<bool>{true, false, true, false};
  auto result   = Executor::compute_column(table, expression);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view(), verbosity);
}

TYPED_TEST(TransformTest, LessComparatorLarge)
{
  auto zero   = cudf::numeric_scalar<int32_t>(0);
  auto two    = cudf::numeric_scalar<int32_t>(2);
  auto five_h = cudf::numeric_scalar<int32_t>(500);

  using Executor = TypeParam;
  auto c_0       = cudf::sequence(2000, zero, two);
  auto c_1       = cudf::sequence(2000, five_h);
  auto table     = cudf::table_view{{c_0->view(), c_1->view()}};

  auto col_ref_0  = cudf::ast::column_reference(0);
  auto col_ref_1  = cudf::ast::column_reference(1);
  auto expression = cudf::ast::operation(cudf::ast::ast_operator::LESS, col_ref_0, col_ref_1);

  auto c        = cudf::detail::make_counting_transform_iterator(0, [](auto i) { return i < 500; });
  auto expected = column_wrapper<bool>(c, c + 2000);
  auto result   = Executor::compute_column(table, expression);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view(), verbosity);
}

TYPED_TEST(TransformTest, MultiLevelTreeArithmetic)
{
  using Executor = TypeParam;

  auto c_0   = column_wrapper<int32_t>{3, 20, 1, 50};
  auto c_1   = column_wrapper<int32_t>{10, 7, 20, 0};
  auto c_2   = column_wrapper<int32_t>{-3, 66, 2, -99};
  auto table = cudf::table_view{{c_0, c_1, c_2}};

  auto col_ref_0 = cudf::ast::column_reference(0);
  auto col_ref_1 = cudf::ast::column_reference(1);
  auto col_ref_2 = cudf::ast::column_reference(2);

  auto expression_left_subtree =
    cudf::ast::operation(cudf::ast::ast_operator::ADD, col_ref_0, col_ref_1);

  auto expression_right_subtree =
    cudf::ast::operation(cudf::ast::ast_operator::SUB, col_ref_2, col_ref_0);

  auto expression_tree = cudf::ast::operation(
    cudf::ast::ast_operator::ADD, expression_left_subtree, expression_right_subtree);

  auto result   = Executor::compute_column(table, expression_tree);
  auto expected = column_wrapper<int32_t>{7, 73, 22, -99};

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view(), verbosity);
}

TYPED_TEST(TransformTest, MultiLevelTreeArithmeticLarge)
{
  auto zero = cudf::numeric_scalar<int32_t>(0);
  auto one  = cudf::numeric_scalar<int32_t>(1);
  auto two  = cudf::numeric_scalar<int32_t>(2);

  using Executor = TypeParam;
  auto c_0       = cudf::sequence(2000, zero);
  auto c_1       = cudf::sequence(2000, one);
  auto c_2       = cudf::sequence(2000, zero, two);
  auto table     = cudf::table_view{{c_0->view(), c_1->view(), c_2->view()}};

  auto col_ref_0 = cudf::ast::column_reference(0);
  auto col_ref_1 = cudf::ast::column_reference(1);
  auto col_ref_2 = cudf::ast::column_reference(2);

  auto expr_left_subtree = cudf::ast::operation(cudf::ast::ast_operator::MUL, col_ref_0, col_ref_1);
  auto expr_right_subtree =
    cudf::ast::operation(cudf::ast::ast_operator::ADD, col_ref_2, col_ref_0);
  auto expr_tree =
    cudf::ast::operation(cudf::ast::ast_operator::SUB, expr_left_subtree, expr_right_subtree);

  auto result = Executor::compute_column(table, expr_tree);
  auto calc   = [](auto i) { return (i * (i + 1)) - (i + (i * 2)); };
  auto d      = cudf::detail::make_counting_transform_iterator(0, [&](auto i) { return calc(i); });
  auto expected = column_wrapper<int32_t>(d, d + 2000);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view(), verbosity);
}

TYPED_TEST(TransformTest, ImbalancedTreeArithmetic)
{
  using Executor = TypeParam;

  auto c_0   = column_wrapper<double>{0.15, 0.37, 4.2, 21.3};
  auto c_1   = column_wrapper<double>{0.0, -42.0, 1.0, 98.6};
  auto c_2   = column_wrapper<double>{0.6, std::numeric_limits<double>::infinity(), 0.999, 1.0};
  auto table = cudf::table_view{{c_0, c_1, c_2}};

  auto col_ref_0 = cudf::ast::column_reference(0);
  auto col_ref_1 = cudf::ast::column_reference(1);
  auto col_ref_2 = cudf::ast::column_reference(2);

  auto expression_right_subtree =
    cudf::ast::operation(cudf::ast::ast_operator::MUL, col_ref_0, col_ref_1);

  auto expression_tree =
    cudf::ast::operation(cudf::ast::ast_operator::SUB, col_ref_2, expression_right_subtree);

  auto result = Executor::compute_column(table, expression_tree);
  auto expected =
    column_wrapper<double>{0.6, std::numeric_limits<double>::infinity(), -3.201, -2099.18};

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view(), verbosity);
}

TYPED_TEST(TransformTest, ImbalancedTreeArithmeticDeep)
{
  using Executor = TypeParam;

  auto c_0   = column_wrapper<int64_t>{4, 5, 6};
  auto table = cudf::table_view{{c_0}};

  auto col_ref_0 = cudf::ast::column_reference(0);

  // expression: (c0 < c0) == (c0 < (c0 + c0))
  //              {false, false, false} == (c0 < {8, 10, 12})
  //              {false, false, false} == {true, true, true}
  //              {false, false, false}
  auto expression_left_subtree =
    cudf::ast::operation(cudf::ast::ast_operator::LESS, col_ref_0, col_ref_0);
  auto expression_right_inner_subtree =
    cudf::ast::operation(cudf::ast::ast_operator::ADD, col_ref_0, col_ref_0);
  auto expression_right_subtree =
    cudf::ast::operation(cudf::ast::ast_operator::LESS, col_ref_0, expression_right_inner_subtree);

  auto expression_tree = cudf::ast::operation(
    cudf::ast::ast_operator::EQUAL, expression_left_subtree, expression_right_subtree);

  auto result   = Executor::compute_column(table, expression_tree);
  auto expected = column_wrapper<bool>{false, false, false};

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view(), verbosity);
}

// TODO: This fails on CI nightlies for CUDA 12.2, driver 535, V100
TYPED_TEST(TransformTest, DISABLED_DeeplyNestedArithmeticLogicalExpression)
{
  using Executor = TypeParam;

  if constexpr (std::is_same_v<Executor, executor_jit>) {
    int driver_version{0};
    auto const err = cudaDriverGetVersion(&driver_version);
    if (err != cudaSuccess or driver_version < 12090) {
      std::cout << "Skipping executor_jit test, driver earlier than 12.9" << std::endl;
      GTEST_SKIP();
    }
  }

  // Test logic for deeply nested arithmetic and logical expressions.
  constexpr int64_t left_depth_level  = 100;
  constexpr int64_t right_depth_level = 75;

  cudf::ast::tree tree;

  auto generate_ast_expr = [&](int64_t depth_level,
                               cudf::ast::column_reference const& col_ref,
                               cudf::ast::ast_operator root_operator,
                               cudf::ast::ast_operator arithmetic_operator,
                               bool nested_left_tree) -> cudf::ast::expression const& {
    auto op = arithmetic_operator;
    tree.push(cudf::ast::operation{op, col_ref, col_ref});

    for (int64_t i = 0; i < depth_level - 1; i++) {
      if (i == depth_level - 2) {
        op = root_operator;
      } else {
        op = arithmetic_operator;
      }
      if (nested_left_tree) {
        tree.push(cudf::ast::operation{op, tree.back(), col_ref});
      } else {
        tree.push(cudf::ast::operation{op, col_ref, tree.back()});
      }
    }

    return tree.back();
  };

  auto c_0   = column_wrapper<int64_t>{0, 0, 0};
  auto c_1   = column_wrapper<int32_t>{0, 0, 0};
  auto table = cudf::table_view{{c_0, c_1}};

  auto const& col_ref_0 = tree.push(cudf::ast::column_reference(0));
  auto const& col_ref_1 = tree.push(cudf::ast::column_reference(1));

  auto const& left_expression  = generate_ast_expr(left_depth_level,
                                                  col_ref_0,
                                                  cudf::ast::ast_operator::LESS,
                                                  cudf::ast::ast_operator::ADD,
                                                  false);
  auto const& right_expression = generate_ast_expr(right_depth_level,
                                                   col_ref_1,
                                                   cudf::ast::ast_operator::EQUAL,
                                                   cudf::ast::ast_operator::SUB,
                                                   true);

  auto const& expression = tree.push(
    cudf::ast::operation(cudf::ast::ast_operator::LOGICAL_OR, left_expression, right_expression));

  // Expression:
  // OR(<(+(+(+(+($0, $0), $0), $0), $0), $0), ==($1, -($1, -($1, -($1, -($1, $1))))))
  // ...
  // OR(<($L, $0), ==($1, $R))
  // true
  //
  // Breakdown:
  // - Left Operand ($L): (+(+(+(+($0, $0), $0), $0), $0), $0)
  // - Right Operand ($R): -($1, -($1, -($1, -($1, $1))))
  // Explanation:
  // If all $1 values and $R values are zeros, the result is true because of the equality check
  // combined with the OR operator in OR(<($L, $0), ==($1, $R)).

  auto result   = Executor::compute_column(table, expression);
  auto expected = column_wrapper<bool>{true, true, true};

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view(), verbosity);
}

TYPED_TEST(TransformTest, MultiLevelTreeComparator)
{
  using Executor = TypeParam;

  auto c_0   = column_wrapper<int32_t>{3, 20, 1, 50};
  auto c_1   = column_wrapper<int32_t>{10, 7, 20, 0};
  auto c_2   = column_wrapper<int32_t>{-3, 66, 2, -99};
  auto table = cudf::table_view{{c_0, c_1, c_2}};

  auto col_ref_0 = cudf::ast::column_reference(0);
  auto col_ref_1 = cudf::ast::column_reference(1);
  auto col_ref_2 = cudf::ast::column_reference(2);

  auto expression_left_subtree =
    cudf::ast::operation(cudf::ast::ast_operator::GREATER_EQUAL, col_ref_0, col_ref_1);

  auto expression_right_subtree =
    cudf::ast::operation(cudf::ast::ast_operator::GREATER, col_ref_2, col_ref_0);

  auto expression_tree = cudf::ast::operation(
    cudf::ast::ast_operator::LOGICAL_AND, expression_left_subtree, expression_right_subtree);

  auto result   = Executor::compute_column(table, expression_tree);
  auto expected = column_wrapper<bool>{false, true, false, false};

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view(), verbosity);
}

TYPED_TEST(TransformTest, MultiTypeOperationFailure)
{
  using Executor = TypeParam;

  auto c_0   = column_wrapper<int32_t>{3, 20, 1, 50};
  auto c_1   = column_wrapper<double>{0.15, 0.77, 4.2, 21.3};
  auto table = cudf::table_view{{c_0, c_1}};

  auto col_ref_0 = cudf::ast::column_reference(0);
  auto col_ref_1 = cudf::ast::column_reference(1);

  auto expression_0_plus_1 =
    cudf::ast::operation(cudf::ast::ast_operator::ADD, col_ref_0, col_ref_1);
  auto expression_1_plus_0 =
    cudf::ast::operation(cudf::ast::ast_operator::ADD, col_ref_1, col_ref_0);

  // Operations on different types are not allowed
  EXPECT_THROW(Executor::compute_column(table, expression_0_plus_1), cudf::logic_error);
  EXPECT_THROW(Executor::compute_column(table, expression_1_plus_0), cudf::logic_error);
}

TYPED_TEST(TransformTest, ColumnReferenceExceed)
{
  using Executor = TypeParam;

  auto c_0   = column_wrapper<int32_t>{3, 20, 1, 50};
  auto c_1   = column_wrapper<int32_t>{10, 7, 20, 0};
  auto table = cudf::table_view{{c_0, c_1}};

  auto col_ref_0 = cudf::ast::column_reference(2);

  EXPECT_THROW(Executor::compute_column(table, col_ref_0), std::out_of_range);
}

TYPED_TEST(TransformTest, LiteralComparison)
{
  using Executor = TypeParam;

  auto c_0   = column_wrapper<int32_t>{3, 20, 1, 50};
  auto table = cudf::table_view{{c_0}};

  auto col_ref_0     = cudf::ast::column_reference(0);
  auto literal_value = cudf::numeric_scalar<int32_t>(41);
  auto literal       = cudf::ast::literal(literal_value);

  auto expression = cudf::ast::operation(cudf::ast::ast_operator::GREATER, col_ref_0, literal);

  auto result   = Executor::compute_column(table, expression);
  auto expected = column_wrapper<bool>{false, false, false, true};

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view(), verbosity);
}

TYPED_TEST(TransformTest, UnaryNot)
{
  using Executor = TypeParam;

  auto c_0   = column_wrapper<int32_t>{3, 0, 1, 50};
  auto table = cudf::table_view{{c_0}};

  auto col_ref_0 = cudf::ast::column_reference(0);

  auto expression = cudf::ast::operation(cudf::ast::ast_operator::NOT, col_ref_0);

  auto result   = Executor::compute_column(table, expression);
  auto expected = column_wrapper<bool>{false, true, false, false};

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view(), verbosity);
}

TYPED_TEST(TransformTest, UnaryTrigonometry)
{
  using Executor = TypeParam;

  auto c_0   = column_wrapper<double>{0.0, M_PI / 4, M_PI / 3};
  auto table = cudf::table_view{{c_0}};

  auto col_ref_0 = cudf::ast::column_reference(0);

  auto expected_sin   = column_wrapper<double>{0.0, std::sqrt(2) / 2, std::sqrt(3.0) / 2.0};
  auto expression_sin = cudf::ast::operation(cudf::ast::ast_operator::SIN, col_ref_0);
  auto result_sin     = Executor::compute_column(table, expression_sin);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected_sin, result_sin->view(), verbosity);

  auto expected_cos   = column_wrapper<double>{1.0, std::sqrt(2) / 2, 0.5};
  auto expression_cos = cudf::ast::operation(cudf::ast::ast_operator::COS, col_ref_0);
  auto result_cos     = Executor::compute_column(table, expression_cos);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected_cos, result_cos->view(), verbosity);

  auto expected_tan   = column_wrapper<double>{0.0, 1.0, std::sqrt(3.0)};
  auto expression_tan = cudf::ast::operation(cudf::ast::ast_operator::TAN, col_ref_0);
  auto result_tan     = Executor::compute_column(table, expression_tan);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected_tan, result_tan->view(), verbosity);
}

TYPED_TEST(TransformTest, ArityCheckFailure)
{
  auto col_ref_0 = cudf::ast::column_reference(0);
  EXPECT_THROW(cudf::ast::operation(cudf::ast::ast_operator::ADD, col_ref_0),
               std::invalid_argument);
  EXPECT_THROW(cudf::ast::operation(cudf::ast::ast_operator::ABS, col_ref_0, col_ref_0),
               std::invalid_argument);
}

TYPED_TEST(TransformTest, StringComparison)
{
  using Executor = TypeParam;

  auto c_0   = cudf::test::strings_column_wrapper({"a", "bb", "ccc", "dddd"});
  auto c_1   = cudf::test::strings_column_wrapper({"aa", "b", "cccc", "ddd"});
  auto table = cudf::table_view{{c_0, c_1}};

  auto col_ref_0  = cudf::ast::column_reference(0);
  auto col_ref_1  = cudf::ast::column_reference(1);
  auto expression = cudf::ast::operation(cudf::ast::ast_operator::LESS, col_ref_0, col_ref_1);

  auto expected = column_wrapper<bool>{true, false, true, false};
  auto result   = Executor::compute_column(table, expression);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view(), verbosity);
}

TYPED_TEST(TransformTest, StringScalarComparison)
{
  using Executor = TypeParam;

  auto c_0   = cudf::test::strings_column_wrapper({"1", "12", "", "23"}, {true, true, false, true});
  auto table = cudf::table_view{{c_0}};

  auto literal_value = cudf::string_scalar("2");
  auto literal       = cudf::ast::literal(literal_value);

  auto col_ref_0  = cudf::ast::column_reference(0);
  auto expression = cudf::ast::operation(cudf::ast::ast_operator::LESS, col_ref_0, literal);

  auto expected = column_wrapper<bool>{{true, true, true, false}, {true, true, false, true}};
  auto result   = Executor::compute_column(table, expression);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view(), verbosity);

  // compare with null literal
  literal_value.set_valid_async(false);
  auto expected2 = column_wrapper<bool>{{false, false, false, false}, {false, false, false, false}};
  auto result2   = Executor::compute_column(table, expression);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view(), verbosity);
}

TYPED_TEST(TransformTest, NumericScalarComparison)
{
  using Executor = TypeParam;

  auto c_0   = column_wrapper<int32_t>{1, 12, 123, 23};
  auto table = cudf::table_view{{c_0}};

  auto literal_value = cudf::numeric_scalar<int32_t>(2);
  auto literal       = cudf::ast::literal(literal_value);

  auto col_ref_0  = cudf::ast::column_reference(0);
  auto expression = cudf::ast::operation(cudf::ast::ast_operator::LESS, col_ref_0, literal);

  auto expected = column_wrapper<bool>{true, false, false, false};
  auto result   = Executor::compute_column(table, expression);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view(), verbosity);
}

TYPED_TEST(TransformTest, CopyColumn)
{
  using Executor = TypeParam;

  auto c_0   = column_wrapper<int32_t>{3, 0, 1, 50};
  auto table = cudf::table_view{{c_0}};

  auto col_ref_0  = cudf::ast::column_reference(0);
  auto expression = cudf::ast::operation(cudf::ast::ast_operator::IDENTITY, col_ref_0);

  auto result   = Executor::compute_column(table, expression);
  auto expected = column_wrapper<int32_t>{3, 0, 1, 50};

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view(), verbosity);
}

TYPED_TEST(TransformTest, CopyLiteral)
{
  using Executor = TypeParam;

  auto c_0   = column_wrapper<int32_t>{0, 0, 0, 0};
  auto table = cudf::table_view{{c_0}};

  auto literal_value = cudf::numeric_scalar<int32_t>(-123);
  auto literal       = cudf::ast::literal(literal_value);

  auto expression = cudf::ast::operation(cudf::ast::ast_operator::IDENTITY, literal);

  auto result   = Executor::compute_column(table, expression);
  auto expected = column_wrapper<int32_t>{-123, -123, -123, -123};

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view(), verbosity);
}

TYPED_TEST(TransformTest, TrueDiv)
{
  using Executor = TypeParam;

  auto c_0   = column_wrapper<int32_t>{3, 0, 1, 50};
  auto table = cudf::table_view{{c_0}};

  auto col_ref_0     = cudf::ast::column_reference(0);
  auto literal_value = cudf::numeric_scalar<int32_t>(2);
  auto literal       = cudf::ast::literal(literal_value);

  auto expression = cudf::ast::operation(cudf::ast::ast_operator::TRUE_DIV, col_ref_0, literal);

  auto result   = Executor::compute_column(table, expression);
  auto expected = column_wrapper<double>{1.5, 0.0, 0.5, 25.0};

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view(), verbosity);
}

TYPED_TEST(TransformTest, FloorDiv)
{
  using Executor = TypeParam;

  auto c_0   = column_wrapper<double>{3.0, 0.0, 1.0, 50.0};
  auto table = cudf::table_view{{c_0}};

  auto col_ref_0     = cudf::ast::column_reference(0);
  auto literal_value = cudf::numeric_scalar<double>(2.0);
  auto literal       = cudf::ast::literal(literal_value);

  auto expression = cudf::ast::operation(cudf::ast::ast_operator::FLOOR_DIV, col_ref_0, literal);

  auto result   = Executor::compute_column(table, expression);
  auto expected = column_wrapper<double>{1.0, 0.0, 0.0, 25.0};

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view(), verbosity);
}

TYPED_TEST(TransformTest, Mod)
{
  using Executor = TypeParam;

  auto c_0   = column_wrapper<double>{3.0, 0.0, -1.0, -50.0};
  auto table = cudf::table_view{{c_0}};

  auto col_ref_0     = cudf::ast::column_reference(0);
  auto literal_value = cudf::numeric_scalar<double>(2.0);
  auto literal       = cudf::ast::literal(literal_value);

  auto expression = cudf::ast::operation(cudf::ast::ast_operator::MOD, col_ref_0, literal);

  auto result   = Executor::compute_column(table, expression);
  auto expected = column_wrapper<double>{1.0, 0.0, -1.0, 0.0};

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view(), verbosity);
}

TYPED_TEST(TransformTest, PyMod)
{
  using Executor = TypeParam;

  auto c_0   = column_wrapper<double>{3.0, 0.0, -1.0, -50.0};
  auto table = cudf::table_view{{c_0}};

  auto col_ref_0     = cudf::ast::column_reference(0);
  auto literal_value = cudf::numeric_scalar<double>(2.0);
  auto literal       = cudf::ast::literal(literal_value);

  auto expression = cudf::ast::operation(cudf::ast::ast_operator::PYMOD, col_ref_0, literal);

  auto result   = Executor::compute_column(table, expression);
  auto expected = column_wrapper<double>{1.0, 0.0, 1.0, 0.0};

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view(), verbosity);
}

TYPED_TEST(TransformTest, BasicEqualityNullEqualNoNulls)
{
  using Executor = TypeParam;

  auto c_0   = column_wrapper<int32_t>{3, 20, 1, 50};
  auto c_1   = column_wrapper<int32_t>{3, 7, 1, 0};
  auto table = cudf::table_view{{c_0, c_1}};

  auto col_ref_0  = cudf::ast::column_reference(0);
  auto col_ref_1  = cudf::ast::column_reference(1);
  auto expression = cudf::ast::operation(cudf::ast::ast_operator::NULL_EQUAL, col_ref_0, col_ref_1);

  auto expected = column_wrapper<bool>{true, false, true, false};
  auto result   = Executor::compute_column(table, expression);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view(), verbosity);
}

TYPED_TEST(TransformTest, BasicEqualityNormalEqualWithNulls)
{
  using Executor = TypeParam;

  auto c_0   = column_wrapper<int32_t>{{3, 20, 1, 50}, {1, 1, 0, 0}};
  auto c_1   = column_wrapper<int32_t>{{3, 7, 1, 0}, {1, 1, 0, 0}};
  auto table = cudf::table_view{{c_0, c_1}};

  auto col_ref_0  = cudf::ast::column_reference(0);
  auto col_ref_1  = cudf::ast::column_reference(1);
  auto expression = cudf::ast::operation(cudf::ast::ast_operator::EQUAL, col_ref_0, col_ref_1);

  auto expected = column_wrapper<bool>{{true, false, true, true}, {1, 1, 0, 0}};
  auto result   = Executor::compute_column(table, expression);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view(), verbosity);
}

TYPED_TEST(TransformTest, BasicEqualityNulls)
{
  using Executor = TypeParam;

  auto c_0   = column_wrapper<int32_t>{{3, 20, 1, 2, 50}, {1, 1, 0, 1, 0}};
  auto c_1   = column_wrapper<int32_t>{{3, 7, 1, 2, 0}, {1, 1, 1, 0, 0}};
  auto table = cudf::table_view{{c_0, c_1}};

  auto col_ref_0  = cudf::ast::column_reference(0);
  auto col_ref_1  = cudf::ast::column_reference(1);
  auto expression = cudf::ast::operation(cudf::ast::ast_operator::NULL_EQUAL, col_ref_0, col_ref_1);

  auto expected = column_wrapper<bool>{{true, false, false, false, true}, {1, 1, 1, 1, 1}};
  auto result   = Executor::compute_column(table, expression);

  // uses EQUIVALENT because null masks are not generated when not needed in JIT. as is the case
  // with NULL_EQUAL.
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected, result->view(), verbosity);
}

TYPED_TEST(TransformTest, UnaryNotNulls)
{
  using Executor = TypeParam;

  auto c_0   = column_wrapper<int32_t>{{3, 0, 0, 50}, {0, 0, 1, 1}};
  auto table = cudf::table_view{{c_0}};

  auto col_ref_0 = cudf::ast::column_reference(0);

  auto expression = cudf::ast::operation(cudf::ast::ast_operator::NOT, col_ref_0);

  auto result   = Executor::compute_column(table, expression);
  auto expected = column_wrapper<bool>{{false, true, true, false}, {0, 0, 1, 1}};

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view(), verbosity);
}

TYPED_TEST(TransformTest, BasicAdditionNulls)
{
  using Executor = TypeParam;

  auto c_0   = column_wrapper<int32_t>{{3, 20, 1, 50}, {0, 0, 1, 1}};
  auto c_1   = column_wrapper<int32_t>{{10, 7, 20, 0}, {0, 1, 0, 1}};
  auto table = cudf::table_view{{c_0, c_1}};

  auto col_ref_0  = cudf::ast::column_reference(0);
  auto col_ref_1  = cudf::ast::column_reference(1);
  auto expression = cudf::ast::operation(cudf::ast::ast_operator::ADD, col_ref_0, col_ref_1);

  auto expected = column_wrapper<int32_t>{{0, 0, 0, 50}, {0, 0, 0, 1}};
  auto result   = Executor::compute_column(table, expression);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view(), verbosity);
}

TYPED_TEST(TransformTest, BasicAdditionLargeNulls)
{
  using Executor = TypeParam;

  auto N = 2000;
  auto a = cuda::counting_iterator<int32_t>{0};

  auto validities = std::vector<int32_t>(N);
  std::fill(validities.begin(), validities.begin() + N / 2, 0);
  std::fill(validities.begin() + (N / 2), validities.end(), 0);

  std::random_device rd;
  std::mt19937 gen(rd());
  std::shuffle(validities.begin(), validities.end(), gen);

  auto col   = column_wrapper<int32_t>(a, a + N, validities.begin());
  auto table = cudf::table_view{{col}};

  auto col_ref    = cudf::ast::column_reference(0);
  auto expression = cudf::ast::operation(cudf::ast::ast_operator::ADD, col_ref, col_ref);

  auto b        = cudf::detail::make_counting_transform_iterator(0, [](auto i) { return i * 2; });
  auto expected = column_wrapper<int32_t>(b, b + N, validities.begin());
  auto result   = Executor::compute_column(table, expression);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view(), verbosity);
}

TYPED_TEST(TransformTest, NullLogicalAnd)
{
  using Executor = TypeParam;

  auto c_0   = column_wrapper<bool>{{false, false, true, true, false, false, true, true},
                                    {1, 1, 1, 1, 1, 0, 0, 0}};
  auto c_1   = column_wrapper<bool>{{false, true, false, true, true, true, false, true},
                                    {1, 1, 1, 1, 0, 1, 1, 0}};
  auto table = cudf::table_view{{c_0, c_1}};

  auto col_ref_0 = cudf::ast::column_reference(0);
  auto col_ref_1 = cudf::ast::column_reference(1);
  auto expression =
    cudf::ast::operation(cudf::ast::ast_operator::NULL_LOGICAL_AND, col_ref_0, col_ref_1);

  auto expected = column_wrapper<bool>{{false, false, false, true, false, false, false, true},
                                       {1, 1, 1, 1, 1, 0, 1, 0}};
  auto result   = Executor::compute_column(table, expression);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view(), verbosity);
}

TYPED_TEST(TransformTest, NullLogicalOr)
{
  using Executor = TypeParam;

  auto c_0   = column_wrapper<bool>{{false, false, true, true, false, false, true, true},
                                    {1, 1, 1, 1, 1, 0, 1, 0}};
  auto c_1   = column_wrapper<bool>{{false, true, false, true, true, true, false, true},
                                    {1, 1, 1, 1, 0, 1, 0, 0}};
  auto table = cudf::table_view{{c_0, c_1}};

  auto col_ref_0 = cudf::ast::column_reference(0);
  auto col_ref_1 = cudf::ast::column_reference(1);
  auto expression =
    cudf::ast::operation(cudf::ast::ast_operator::NULL_LOGICAL_OR, col_ref_0, col_ref_1);

  auto expected = column_wrapper<bool>{{false, true, true, true, false, true, true, true},
                                       {1, 1, 1, 1, 0, 1, 1, 0}};
  auto result   = Executor::compute_column(table, expression);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view(), verbosity);
}

TYPED_TEST(TransformTest, ScalarOnly)
{
  using Executor = TypeParam;

  auto column = column_wrapper<int>{1, 2, 3, 4, 5};
  auto table  = cudf::table_view{{column}};

  cudf::ast::tree tree{};

  auto first_value  = cudf::numeric_scalar(0);
  auto second_value = cudf::numeric_scalar(1);
  auto first        = cudf::ast::literal(first_value);
  auto second       = cudf::ast::literal(second_value);

  auto const& neq =
    tree.push(cudf::ast::operation{cudf::ast::ast_operator::NOT_EQUAL, first, second});

  auto result = Executor::compute_column(table, neq);

  auto expected = column_wrapper<bool>{true, true, true, true, true};

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
}

TYPED_TEST(TransformTest, ComplexScalarOnly)
{
  using Executor = TypeParam;

  auto column = column_wrapper<int>{1, 2, 3, 4, 5};
  auto table  = cudf::table_view{{column}};

  cudf::ast::tree tree{};

  auto first_value  = cudf::string_scalar("first");
  auto second_value = cudf::string_scalar("second");
  auto first        = cudf::ast::literal(first_value);
  auto second       = cudf::ast::literal(second_value);

  auto const& neq =
    tree.push(cudf::ast::operation{cudf::ast::ast_operator::NOT_EQUAL, first, second});

  auto result = Executor::compute_column(table, neq);

  auto expected = column_wrapper<bool>{true, true, true, true, true};

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
}

template <typename T>
struct DecimalComparisonTest : public cudf::test::BaseFixture {};

using DecimalComparisonParams = cudf::test::CrossProduct<Executors, cudf::test::FixedPointTypes>;

TYPED_TEST_SUITE(DecimalComparisonTest, DecimalComparisonParams);

TYPED_TEST(DecimalComparisonTest, DecimalComparison)
{
  using Executor  = cudf::test::GetType<TypeParam, 0>;
  using decimalXX = cudf::test::GetType<TypeParam, 1>;
  using RepType   = cudf::device_storage_type_t<decimalXX>;

  auto const scale = numeric::scale_type{-4};
  auto c_a         = cudf::test::fixed_point_column_wrapper<RepType>({1, 2, 3, 4}, scale);
  auto table       = cudf::table_view{{c_a}};

  auto literal_value = cudf::fixed_point_scalar<decimalXX>(2, scale, true);
  auto literal       = cudf::ast::literal(literal_value);
  auto col_ref       = cudf::ast::column_reference(0);
  auto expression    = cudf::ast::operation(cudf::ast::ast_operator::EQUAL, col_ref, literal);
  auto result        = Executor::compute_column(table, expression);

  auto expected = column_wrapper<bool>({false, true, false, false});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view(), verbosity);
}

TYPED_TEST(TransformTest, FloorDivIntegerEqualComparison)
{
  using Executor = TypeParam;

  auto col   = column_wrapper<int64_t>{300'964, 300'972, 500'000, 26};
  auto table = cudf::table_view{{col}};

  auto col_ref       = cudf::ast::column_reference(0);
  auto divisor_value = cudf::numeric_scalar<int64_t>(100'000);
  auto divisor       = cudf::ast::literal(divisor_value);
  auto floor_div     = cudf::ast::operation(cudf::ast::ast_operator::FLOOR_DIV, col_ref, divisor);

  auto zero_value = cudf::numeric_scalar<int64_t>(0);
  auto zero       = cudf::ast::literal(zero_value);
  auto eq_expr    = cudf::ast::operation(cudf::ast::ast_operator::EQUAL, floor_div, zero);

  auto result   = Executor::compute_column(table, eq_expr);
  auto expected = column_wrapper<bool>{false, false, false, true};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view(), verbosity);
}

TYPED_TEST(TransformTest, TrueDivIntegerEqualComparison)
{
  using Executor = TypeParam;

  auto col   = column_wrapper<int64_t>{10, 8, 7, 4};
  auto table = cudf::table_view{{col}};

  auto col_ref       = cudf::ast::column_reference(0);
  auto divisor_value = cudf::numeric_scalar<int64_t>(4);
  auto divisor       = cudf::ast::literal(divisor_value);
  auto true_div      = cudf::ast::operation(cudf::ast::ast_operator::TRUE_DIV, col_ref, divisor);

  auto two_value = cudf::numeric_scalar<double>(2.0);
  auto two       = cudf::ast::literal(two_value);
  auto eq_expr   = cudf::ast::operation(cudf::ast::ast_operator::EQUAL, true_div, two);

  auto result   = Executor::compute_column(table, eq_expr);
  auto expected = column_wrapper<bool>{false, true, false, false};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view(), verbosity);
}

TYPED_TEST(TransformTest, FloorDivIntegerNegativeOperands)
{
  using Executor = TypeParam;

  auto col   = column_wrapper<int64_t>{-7, 7, -6, 6};
  auto table = cudf::table_view{{col}};

  auto col_ref           = cudf::ast::column_reference(0);
  auto divisor_value_pos = cudf::numeric_scalar<int64_t>(2);
  auto divisor_pos       = cudf::ast::literal(divisor_value_pos);
  auto floor_div_pos =
    cudf::ast::operation(cudf::ast::ast_operator::FLOOR_DIV, col_ref, divisor_pos);
  auto result_pos   = Executor::compute_column(table, floor_div_pos);
  auto expected_pos = column_wrapper<int64_t>{-4, 3, -3, 3};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_pos, result_pos->view(), verbosity);

  auto divisor_value_neg = cudf::numeric_scalar<int64_t>(-2);
  auto divisor_neg       = cudf::ast::literal(divisor_value_neg);
  auto floor_div_neg =
    cudf::ast::operation(cudf::ast::ast_operator::FLOOR_DIV, col_ref, divisor_neg);
  auto result_neg   = Executor::compute_column(table, floor_div_neg);
  auto expected_neg = column_wrapper<int64_t>{3, -4, 3, -3};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_neg, result_neg->view(), verbosity);
}

TYPED_TEST(TransformTest, PowIntegerEqualComparison)
{
  using Executor = TypeParam;

  auto col   = column_wrapper<int64_t>{2, 3, 4, 2};
  auto table = cudf::table_view{{col}};

  auto col_ref    = cudf::ast::column_reference(0);
  auto exp_value  = cudf::numeric_scalar<int64_t>(2);
  auto exp_scalar = cudf::ast::literal(exp_value);
  auto pow_expr   = cudf::ast::operation(cudf::ast::ast_operator::POW, col_ref, exp_scalar);

  auto sixteen_value = cudf::numeric_scalar<int64_t>(16);
  auto sixteen       = cudf::ast::literal(sixteen_value);
  auto eq_expr       = cudf::ast::operation(cudf::ast::ast_operator::EQUAL, pow_expr, sixteen);

  auto result   = Executor::compute_column(table, eq_expr);
  auto expected = column_wrapper<bool>{false, false, true, false};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view(), verbosity);
}

TYPED_TEST(TransformTest, PowIntegerZeroExponent)
{
  using Executor = TypeParam;

  auto col   = column_wrapper<int64_t>{2, 3, 4, 5};
  auto table = cudf::table_view{{col}};

  auto col_ref    = cudf::ast::column_reference(0);
  auto exp_value  = cudf::numeric_scalar<int64_t>(0);
  auto exp_scalar = cudf::ast::literal(exp_value);
  auto pow_expr   = cudf::ast::operation(cudf::ast::ast_operator::POW, col_ref, exp_scalar);

  auto result   = Executor::compute_column(table, pow_expr);
  auto expected = column_wrapper<int64_t>{1, 1, 1, 1};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view(), verbosity);
}

TYPED_TEST(TransformTest, PowIntegerZeroBase)
{
  using Executor = TypeParam;

  auto col   = column_wrapper<int64_t>{0, 0, 0, 0};
  auto table = cudf::table_view{{col}};

  auto col_ref    = cudf::ast::column_reference(0);
  auto exp_value  = cudf::numeric_scalar<int64_t>(3);
  auto exp_scalar = cudf::ast::literal(exp_value);
  auto pow_expr   = cudf::ast::operation(cudf::ast::ast_operator::POW, col_ref, exp_scalar);

  auto result   = Executor::compute_column(table, pow_expr);
  auto expected = column_wrapper<int64_t>{0, 0, 0, 0};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view(), verbosity);
}

TYPED_TEST(TransformTest, PowIntegerNegativeExponent)
{
  using Executor = TypeParam;

  auto col   = column_wrapper<int64_t>{2, 3, 4, 5};
  auto table = cudf::table_view{{col}};

  auto col_ref    = cudf::ast::column_reference(0);
  auto exp_value  = cudf::numeric_scalar<int64_t>(-1);
  auto exp_scalar = cudf::ast::literal(exp_value);
  auto pow_expr   = cudf::ast::operation(cudf::ast::ast_operator::POW, col_ref, exp_scalar);

  auto result   = Executor::compute_column(table, pow_expr);
  auto expected = column_wrapper<int64_t>{0, 0, 0, 0};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view(), verbosity);
}

template <typename T>
struct DecimalTests : public cudf::test::BaseFixture {};

// decimal128 intermediates exceed the 8-byte IntermediateDataType limit, so
// only test decimal32 and decimal64 here.
using DecimalArithmeticParams =
  cudf::test::CrossProduct<Executors, cudf::test::Types<numeric::decimal32, numeric::decimal64>>;
TYPED_TEST_SUITE(DecimalTests, DecimalArithmeticParams);

// Regression test for https://github.com/rapidsai/cudf/issues/21980
// Nested decimal expressions lose scale in intermediate return types,
// causing "non-matching operand types" at parse time.
TYPED_TEST(DecimalTests, NestedDecimalArithmetic)
{
  using Executor  = cudf::test::GetType<TypeParam, 0>;
  using decimalXX = cudf::test::GetType<TypeParam, 1>;
  using RepType   = cudf::device_storage_type_t<decimalXX>;

  auto const scale = numeric::scale_type{-2};

  // col0 = {10.00, 20.00, 30.00, 40.00}  (rep values: 1000, 2000, 3000, 4000)
  // col1 = {0.05,  0.10,  0.15,  0.20}   (rep values: 5, 10, 15, 20)
  auto c_0   = cudf::test::fixed_point_column_wrapper<RepType>({1000, 2000, 3000, 4000}, scale);
  auto c_1   = cudf::test::fixed_point_column_wrapper<RepType>({5, 10, 15, 20}, scale);
  auto table = cudf::table_view{{c_0, c_1}};

  // AST: col0 * (literal - col1)
  // literal = 1.00 (rep=100, scale=-2)
  // SUB result scale = min(-2, -2) = -2
  // MUL result scale = -2 + -2 = -4
  auto literal_value = cudf::fixed_point_scalar<decimalXX>(RepType{100}, scale, true);
  auto literal       = cudf::ast::literal(literal_value);
  auto col_ref_0     = cudf::ast::column_reference(0);
  auto col_ref_1     = cudf::ast::column_reference(1);

  cudf::ast::tree tree{};
  auto const& sub_expr =
    tree.push(cudf::ast::operation(cudf::ast::ast_operator::SUB, literal, col_ref_1));
  auto const& mul_expr =
    tree.push(cudf::ast::operation(cudf::ast::ast_operator::MUL, col_ref_0, sub_expr));

  auto result = Executor::compute_column(table, mul_expr);

  // Expected: col0 * (1.00 - col1)
  // Row 0: 10.00 * (1.00 - 0.05) = 10.00 * 0.95 = 9.5000  => rep 95000 at scale -4
  // Row 1: 20.00 * (1.00 - 0.10) = 20.00 * 0.90 = 18.0000 => rep 180000 at scale -4
  // Row 2: 30.00 * (1.00 - 0.15) = 30.00 * 0.85 = 25.5000 => rep 255000 at scale -4
  // Row 3: 40.00 * (1.00 - 0.20) = 40.00 * 0.80 = 32.0000 => rep 320000 at scale -4
  auto const expected_scale = numeric::scale_type{-4};
  auto expected             = cudf::test::fixed_point_column_wrapper<RepType>(
    {RepType{95000}, RepType{180000}, RepType{255000}, RepType{320000}}, expected_scale);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view(), verbosity);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view(), verbosity);
}

TYPED_TEST(DecimalTests, DecimalDivide)
{
  using Executor  = cudf::test::GetType<TypeParam, 0>;
  using decimalXX = cudf::test::GetType<TypeParam, 1>;
  using RepType   = cudf::device_storage_type_t<decimalXX>;

  auto const scale = numeric::scale_type{-2};

  // col0 = {10.00, 20.00, 30.00, 40.00}  (rep values: 1000, 2000, 3000, 4000)
  // col1 = {0.05,  0.10,  0.15,  0.20}   (rep values: 5, 10, 15, 20)
  auto c_0   = cudf::test::fixed_point_column_wrapper<RepType>({1000, 2000, 3000, 4000}, scale);
  auto c_1   = cudf::test::fixed_point_column_wrapper<RepType>({5, 10, 15, 20}, scale);
  auto table = cudf::table_view{{c_0, c_1}};

  // AST: col0 / (literal + col1)
  // literal = 10.00 (rep=1000, scale=-2)
  // ADD result scale = min(-2, -2) = -2
  // DIV result scale = -2 - -2 = 0
  auto literal_value = cudf::fixed_point_scalar<decimalXX>(RepType{1000}, scale, true);  // 10
  auto literal       = cudf::ast::literal(literal_value);
  auto col_ref_0     = cudf::ast::column_reference(0);
  auto col_ref_1     = cudf::ast::column_reference(1);

  cudf::ast::tree tree{};
  auto const& add_expr =
    tree.push(cudf::ast::operation(cudf::ast::ast_operator::ADD, literal, col_ref_1));
  auto const& div_expr =
    tree.push(cudf::ast::operation(cudf::ast::ast_operator::DIV, col_ref_0, add_expr));

  auto result = Executor::compute_column(table, div_expr);

  // Expected: col0 * (10.00 + col1)
  // Row 0: 10.00 / (10.00 + 0.05) = 10.00 / 10.05 = 0 at scale 0
  // Row 1: 20.00 / (10.00 + 0.10) = 20.00 / 10.10 = 1 at scale 0
  // Row 2: 30.00 / (10.00 + 0.15) = 30.00 / 10.15 = 2 at scale 0
  // Row 3: 40.00 / (10.00 + 0.20) = 40.00 / 10.20 = 3 at scale 0
  auto const expected_scale = numeric::scale_type{0};
  auto expected             = cudf::test::fixed_point_column_wrapper<RepType>(
    {RepType{0}, RepType{1}, RepType{2}, RepType{3}}, expected_scale);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view(), verbosity);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view(), verbosity);
}

TYPED_TEST(DecimalTests, DecimalComparator)
{
  using Executor  = cudf::test::GetType<TypeParam, 0>;
  using decimalXX = cudf::test::GetType<TypeParam, 1>;
  using RepType   = cudf::device_storage_type_t<decimalXX>;

  auto const scale = numeric::scale_type{-2};

  // col0 = {10.00, 20.00, 30.00, 40.00}  (rep values: 1000, 2000, 3000, 4000)
  // col1 = {0.05,  0.10,  0.15,  0.20}   (rep values: 5, 10, 15, 20)
  auto c_0   = cudf::test::fixed_point_column_wrapper<RepType>({1000, 2000, 3000, 4000}, scale);
  auto c_1   = cudf::test::fixed_point_column_wrapper<RepType>({5, 10, 15, 20}, scale);
  auto table = cudf::table_view{{c_0, c_1}};

  auto value0    = cudf::fixed_point_scalar<decimalXX>(RepType{1000}, scale, true);  // 10
  auto literal0  = cudf::ast::literal(value0);
  auto col_ref_0 = cudf::ast::column_reference(0);
  auto value1    = cudf::fixed_point_scalar<decimalXX>(RepType{20}, scale, true);  // 0.2
  auto literal1  = cudf::ast::literal(value1);
  auto col_ref_1 = cudf::ast::column_reference(1);

  {
    cudf::ast::tree tree{};
    auto const& expr =
      tree.push(cudf::ast::operation(cudf::ast::ast_operator::LESS, literal0, col_ref_0));
    auto expected = cudf::test::fixed_width_column_wrapper<bool>({0, 1, 1, 1});
    auto result   = Executor::compute_column(table, expr);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
  }
  {
    cudf::ast::tree tree{};
    auto const& expr =
      tree.push(cudf::ast::operation(cudf::ast::ast_operator::GREATER, literal1, col_ref_1));
    auto expected = cudf::test::fixed_width_column_wrapper<bool>({1, 1, 1, 0});
    auto result   = Executor::compute_column(table, expr);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
  }
}

TYPED_TEST(DecimalTests, DecimalDifferentScales)
{
  using Executor  = cudf::test::GetType<TypeParam, 0>;
  using decimalXX = cudf::test::GetType<TypeParam, 1>;
  using RepType   = cudf::device_storage_type_t<decimalXX>;

  auto scale1 = numeric::scale_type{0};
  auto scale2 = numeric::scale_type{-2};

  auto value0   = cudf::fixed_point_scalar<decimalXX>(RepType{10}, scale1, true);
  auto literal0 = cudf::ast::literal(value0);
  auto c_0 = cudf::test::fixed_point_column_wrapper<RepType>({500, 1000, 2000, 3000, 4000}, scale2);
  auto table     = cudf::table_view({c_0});
  auto col_ref_0 = cudf::ast::column_reference(0);

  {
    cudf::ast::tree tree{};
    auto const& expr =
      tree.push(cudf::ast::operation(cudf::ast::ast_operator::EQUAL, col_ref_0, literal0));
    auto expected = cudf::test::fixed_width_column_wrapper<bool>({0, 1, 0, 0, 0});
    auto result   = Executor::compute_column(table, expr);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
  }
  {
    cudf::ast::tree tree{};
    auto const& expr =
      tree.push(cudf::ast::operation(cudf::ast::ast_operator::DIV, col_ref_0, literal0));
    auto expected =
      cudf::test::fixed_point_column_wrapper<RepType>({50, 100, 200, 300, 400}, scale2);
    auto result = Executor::compute_column(table, expr);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
  }
  {
    cudf::ast::tree tree{};
    auto const& expr =
      tree.push(cudf::ast::operation(cudf::ast::ast_operator::SUB, col_ref_0, literal0));
    auto expected =
      cudf::test::fixed_point_column_wrapper<RepType>({-500, 0, 1000, 2000, 3000}, scale2);
    auto result = Executor::compute_column(table, expr);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
  }
}

TYPED_TEST(DecimalTests, LessConsumesMulOfDecimals)
{
  using Executor  = cudf::test::GetType<TypeParam, 0>;
  using decimalXX = cudf::test::GetType<TypeParam, 1>;
  using RepType   = cudf::device_storage_type_t<decimalXX>;

  auto const scale = numeric::scale_type{-2};
  // Values at scale=-2:  col_a in {1, 2, 5, 10, 25, 50}.00,
  // col_thresh = 0.20, col_avg = 25.00.  Expected: col_a < 5.00 → [1,1,0,0,0,0].
  auto col_a =
    cudf::test::fixed_point_column_wrapper<RepType>({100, 200, 500, 1000, 2500, 5000}, scale);
  auto col_thresh =
    cudf::test::fixed_point_column_wrapper<RepType>({20, 20, 20, 20, 20, 20}, scale);
  auto col_avg =
    cudf::test::fixed_point_column_wrapper<RepType>({2500, 2500, 2500, 2500, 2500, 2500}, scale);
  auto table = cudf::table_view{{col_a, col_thresh, col_avg}};

  // Path 1: single fused AST -- col_a < (col_thresh * col_avg)
  auto ra     = cudf::ast::column_reference(0);
  auto rt     = cudf::ast::column_reference(1);
  auto rv     = cudf::ast::column_reference(2);
  auto mul    = cudf::ast::operation(cudf::ast::ast_operator::MUL, rt, rv);
  auto lt     = cudf::ast::operation(cudf::ast::ast_operator::LESS, ra, mul);
  auto result = Executor::compute_column(table, lt);

  // Path 2: chained binary_operation, no AST intermediate
  auto decimal_type = cudf::data_type{cudf::type_to_id<decimalXX>(), numeric::scale_type{-4}};
  auto tmp = cudf::binary_operation(col_thresh, col_avg, cudf::binary_operator::MUL, decimal_type);
  auto ref_result = cudf::binary_operation(
    col_a, tmp->view(), cudf::binary_operator::LESS, cudf::data_type{cudf::type_id::BOOL8});

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result->view(), ref_result->view());
}

TYPED_TEST(TransformTest, NullPropagatesViaSmallTypeIntermediate)
{
  using Executor = TypeParam;

  auto a     = cudf::test::fixed_width_column_wrapper<int32_t>{{1, 2, 3, 4}, {1, 0, 1, 1}};
  auto b     = cudf::test::fixed_width_column_wrapper<int32_t>{{10, 10, 10, 10}, {1, 1, 1, 1}};
  auto c     = cudf::test::fixed_width_column_wrapper<int32_t>{{5, 5, 5, 5}, {1, 1, 0, 1}};
  auto table = cudf::table_view{{a, b, c}};

  auto ra  = cudf::ast::column_reference(0);
  auto rb  = cudf::ast::column_reference(1);
  auto rc  = cudf::ast::column_reference(2);
  auto mul = cudf::ast::operation(cudf::ast::ast_operator::MUL, ra, rb);
  auto add = cudf::ast::operation(cudf::ast::ast_operator::ADD, mul, rc);

  // row 0: 1*10+5  = 15 (valid)
  // row 1: null*10 = null → null+5 = null
  // row 2: 3*10    = 30  → 30+null = null
  // row 3: 4*10+5  = 45 (valid)
  auto expected = column_wrapper<int32_t>{{15, 0, 0, 45}, {1, 0, 0, 1}};
  auto result   = Executor::compute_column(table, add);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view(), verbosity);
}

TYPED_TEST(DecimalTests, LessConsumesMulOfDecimalsWithNulls)
{
  using Executor  = cudf::test::GetType<TypeParam, 0>;
  using decimalXX = cudf::test::GetType<TypeParam, 1>;
  using RepType   = cudf::device_storage_type_t<decimalXX>;

  auto const scale = numeric::scale_type{-2};
  // col_a  at scale=-2: {1.00, null, 5.00, 10.00}
  // thresh at scale=-2: {0.20, 0.20, null, 0.20}
  // avg    at scale=-2: {25.00, 25.00, 25.00, 25.00} (no nulls)
  auto col_a =
    cudf::test::fixed_point_column_wrapper<RepType>({100, 200, 500, 1000}, {1, 0, 1, 1}, scale);
  auto col_thresh =
    cudf::test::fixed_point_column_wrapper<RepType>({20, 20, 20, 20}, {1, 1, 0, 1}, scale);
  auto col_avg = cudf::test::fixed_point_column_wrapper<RepType>({2500, 2500, 2500, 2500}, scale);
  auto table   = cudf::table_view{{col_a, col_thresh, col_avg}};

  auto ra  = cudf::ast::column_reference(0);
  auto rt  = cudf::ast::column_reference(1);
  auto rv  = cudf::ast::column_reference(2);
  auto mul = cudf::ast::operation(cudf::ast::ast_operator::MUL, rt, rv);
  auto lt  = cudf::ast::operation(cudf::ast::ast_operator::LESS, ra, mul);

  // row 0: 1.00 < (0.20*25.00 = 5.00) → true
  // row 1: col_a=null                  → null
  // row 2: col_thresh=null → MUL=null  → null
  // row 3: 10.00 < 5.00                → false
  auto expected = cudf::test::fixed_width_column_wrapper<bool>({1, 0, 0, 0}, {1, 0, 0, 1});
  auto result   = Executor::compute_column(table, lt);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view(), verbosity);
}

TYPED_TEST(TransformTest, NonDefaultStream)
{
  // This test ensures that the algorithm is stream safe when a nondefault
  // custom stream is passed. See #21920.

  using Executor = TypeParam;

  rmm::cuda_stream stream;

  auto c_0   = column_wrapper<int32_t>{3, 20, 1, 50};
  auto c_1   = column_wrapper<int32_t>{10, 7, 20, 0};
  auto table = cudf::table_view{{c_0, c_1}};

  auto col_ref_0  = cudf::ast::column_reference(0);
  auto col_ref_1  = cudf::ast::column_reference(1);
  auto expression = cudf::ast::operation(cudf::ast::ast_operator::ADD, col_ref_0, col_ref_1);

  auto expected = column_wrapper<int32_t>{13, 27, 21, 50};
  auto result   = Executor::compute_column(table, expression, stream);
  stream.synchronize();

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view(), verbosity);
}

TYPED_TEST(TransformTest, Decimal128Unsupported)
{
  using Executor = TypeParam;

  // column = {2000.00, 2000.00}  (rep 200000 @ scale -2)
  auto const scale = numeric::scale_type{-2};
  auto const col   = cudf::test::fixed_point_column_wrapper<__int128>{{200000, 200000}, scale};
  auto table       = cudf::table_view{{col}};

  // literal = 0.5 @ scale -2  (rep 50)
  auto half = numeric::decimal128{numeric::scaled_integer<numeric::decimal128::rep>{50, scale}};
  auto lit  = cudf::fixed_point_scalar<numeric::decimal128>(half, true);

  auto lr = cudf::ast::literal(lit);
  auto cr = cudf::ast::column_reference(0);

  auto ast = cudf::ast::operation(cudf::ast::ast_operator::MUL, lr, cr);
  if constexpr (std::is_same_v<Executor, executor_ast>) {
    EXPECT_THROW(Executor::compute_column(table, ast), cudf::data_type_error);
  } else {
    auto result = Executor::compute_column(table, ast);
    // Expected: 0.5 * 2000.00 = 1000.00  => rep 10000000 @ scale -4
    EXPECT_EQ(result->type().id(), cudf::type_id::DECIMAL128);
    EXPECT_EQ(result->type().scale(), numeric::scale_type{-4});
    auto expected = cudf::test::fixed_point_column_wrapper<__int128>{{10000000, 10000000},
                                                                     numeric::scale_type{-4}};
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result, expected);
  }
}

TYPED_TEST(TransformTest, Decimal128IdentityOutput)
{
  using Executor = TypeParam;

  auto const input   = column_wrapper<int32_t>{0, 0};
  auto const table   = cudf::table_view{{input}};
  auto const scale   = numeric::scale_type{-2};
  auto literal_value = cudf::fixed_point_scalar<numeric::decimal128>(12345, scale, true);
  auto literal       = cudf::ast::literal(literal_value);
  auto expression    = cudf::ast::operation(cudf::ast::ast_operator::IDENTITY, literal);

  if constexpr (std::is_same_v<Executor, executor_ast>) {
    EXPECT_THROW(Executor::compute_column(table, expression), cudf::data_type_error);
  } else {
    auto result   = Executor::compute_column(table, expression);
    auto expected = cudf::test::fixed_point_column_wrapper<__int128>({12345, 12345}, scale);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view(), verbosity);
  }
}

CUDF_TEST_PROGRAM_MAIN()
