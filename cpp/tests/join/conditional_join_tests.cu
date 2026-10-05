/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/ast/expressions.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/join/conditional_join.hpp>
#include <cudf/join/filtered_join.hpp>
#include <cudf/join/join.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <rmm/exec_policy.hpp>

#include <thrust/equal.h>
#include <thrust/sort.h>
#include <thrust/transform.h>

#include <algorithm>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

namespace {
using PairJoinReturn   = std::pair<std::unique_ptr<rmm::device_uvector<cudf::size_type>>,
                                   std::unique_ptr<rmm::device_uvector<cudf::size_type>>>;
using SingleJoinReturn = std::unique_ptr<rmm::device_uvector<cudf::size_type>>;
using NullMaskVector   = std::vector<bool>;

template <typename T>
using ColumnVector = std::vector<std::vector<T>>;

template <typename T>
using NullableColumnVector = std::vector<std::pair<std::vector<T>, NullMaskVector>>;

// Common column references.
auto const col_ref_left_0  = cudf::ast::column_reference(0, cudf::ast::table_reference::LEFT);
auto const col_ref_right_0 = cudf::ast::column_reference(0, cudf::ast::table_reference::RIGHT);

// Common expressions.
auto left_zero_eq_right_zero =
  cudf::ast::operation(cudf::ast::ast_operator::EQUAL, col_ref_left_0, col_ref_right_0);

// Generate a single pair of left/right non-nullable columns of random data
// suitable for testing a join against a reference join implementation.
template <typename T>
std::pair<std::vector<T>, std::vector<T>> gen_random_repeated_columns(
  unsigned int N_left            = 10000,
  unsigned int num_repeats_left  = 10,
  unsigned int N_right           = 10000,
  unsigned int num_repeats_right = 10)
{
  // Generate columns of num_repeats repeats of the integer range [0, num_unique),
  // then merge a shuffled version and compare to hash join.
  unsigned int num_unique_left  = N_left / num_repeats_left;
  unsigned int num_unique_right = N_right / num_repeats_right;

  std::vector<T> left(N_left);
  std::vector<T> right(N_right);

  for (unsigned int i = 0; i < num_repeats_left; ++i) {
    std::iota(std::next(left.begin(), num_unique_left * i),
              std::next(left.begin(), num_unique_left * (i + 1)),
              0);
  }
  for (unsigned int i = 0; i < num_repeats_right; ++i) {
    std::iota(std::next(right.begin(), num_unique_right * i),
              std::next(right.begin(), num_unique_right * (i + 1)),
              0);
  }

  std::random_device rd;
  std::mt19937 gen(rd());
  std::shuffle(left.begin(), left.end(), gen);
  std::shuffle(right.begin(), right.end(), gen);
  return std::pair(std::move(left), std::move(right));
}

// Generate a single pair of left/right nullable columns of random data
// suitable for testing a join against a reference join implementation.
template <typename T>
std::pair<std::pair<std::vector<T>, std::vector<bool>>,
          std::pair<std::vector<T>, std::vector<bool>>>
gen_random_nullable_repeated_columns(unsigned int N = 10000, unsigned int num_repeats = 10)
{
  auto [left, right] = gen_random_repeated_columns<T>(N, num_repeats);

  std::vector<bool> left_nulls(N);
  std::vector<bool> right_nulls(N);

  // Seed with a real random value, if available
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> uniform_dist(0, 1);

  std::generate(left_nulls.begin(), left_nulls.end(), [&uniform_dist, &gen]() {
    return uniform_dist(gen) > 0.5;
  });
  std::generate(right_nulls.begin(), right_nulls.end(), [&uniform_dist, &gen]() {
    return uniform_dist(gen) > 0.5;
  });

  return std::pair(std::pair(std::move(left), std::move(left_nulls)),
                   std::pair(std::move(right), std::move(right_nulls)));
}

// `rmm::device_uvector<T>` requires that T be trivially copyable. `thrust::pair` does
// not satisfy this requirement because it defines nontrivial copy/move
// constructors. Therefore, we need a simple, trivially copyable pair-like
// object. `index_pair` is a minimal implementation suitable for use in the
// tests in this file.
struct index_pair {
  cudf::size_type first{};
  cudf::size_type second{};
  __device__ index_pair() {};
  __device__ index_pair(cudf::size_type const& first, cudf::size_type const& second)
    : first(first), second(second) {};
};

__device__ inline bool operator<(index_pair const& lhs, index_pair const& rhs)
{
  if (lhs.first > rhs.first) return false;
  return (lhs.first < rhs.first) || (lhs.second < rhs.second);
}

__device__ inline bool operator==(index_pair const& lhs, index_pair const& rhs)
{ return lhs.first == rhs.first && lhs.second == rhs.second; }

}  // namespace

/**
 * Fixture for all nested loop conditional joins.
 */
template <typename T>
struct ConditionalJoinTest : public cudf::test::BaseFixture {
  /**
   * Convenience utility for parsing initializer lists of input data into
   * suitable inputs for tables.
   */
  template <typename U>
  std::tuple<std::vector<cudf::test::fixed_width_column_wrapper<T>>,
             std::vector<cudf::test::fixed_width_column_wrapper<T>>,
             std::vector<cudf::column_view>,
             std::vector<cudf::column_view>,
             cudf::table_view,
             cudf::table_view>
  parse_input(std::vector<U> left_data, std::vector<U> right_data)
  {
    auto wrapper_generator = [](U& v) {
      if constexpr (std::is_same_v<U, std::vector<T>>) {
        return cudf::test::fixed_width_column_wrapper<T>(v.begin(), v.end());
      } else if constexpr (std::is_same_v<U, std::pair<std::vector<T>, std::vector<bool>>>) {
        return cudf::test::fixed_width_column_wrapper<T>(
          v.first.begin(), v.first.end(), v.second.begin());
      }
      throw std::runtime_error("Invalid input to parse_input.");
      return cudf::test::fixed_width_column_wrapper<T>();
    };

    // Note that we need to maintain the column wrappers otherwise the
    // resulting column views will be referencing potentially invalid memory.
    std::vector<cudf::test::fixed_width_column_wrapper<T>> left_wrappers;
    std::vector<cudf::column_view> left_columns;
    for (auto v : left_data) {
      left_wrappers.push_back(wrapper_generator(v));
      left_columns.push_back(left_wrappers.back());
    }

    std::vector<cudf::test::fixed_width_column_wrapper<T>> right_wrappers;
    std::vector<cudf::column_view> right_columns;
    for (auto v : right_data) {
      right_wrappers.push_back(wrapper_generator(v));
      right_columns.push_back(right_wrappers.back());
    }

    return std::make_tuple(std::move(left_wrappers),
                           std::move(right_wrappers),
                           std::move(left_columns),
                           std::move(right_columns),
                           cudf::table_view(left_columns),
                           cudf::table_view(right_columns));
  }
};

/**
 * Fixture for join types that return both left and right indices (inner, left,
 * and full joins).
 */
template <typename T>
struct ConditionalJoinPairReturnTest : public ConditionalJoinTest<T> {
  /*
   * Perform a join of tables constructed from two input data sets according to
   * the provided predicate and verify that the outputs match the expected
   * outputs (up to order).
   */
  void _test(cudf::table_view left,
             cudf::table_view right,
             cudf::ast::operation predicate,
             std::vector<std::pair<cudf::size_type, cudf::size_type>> expected_outputs)
  {
    auto result_size = this->join_size(left, right, predicate);
    EXPECT_EQ(result_size, expected_outputs.size());

    auto result     = this->join(left, right, predicate);
    auto lhs_result = cudf::detail::make_std_vector(*result.first, cudf::get_default_stream());
    auto rhs_result = cudf::detail::make_std_vector(*result.second, cudf::get_default_stream());
    std::vector<std::pair<cudf::size_type, cudf::size_type>> result_pairs(lhs_result.size());
    std::transform(lhs_result.begin(),
                   lhs_result.end(),
                   rhs_result.begin(),
                   result_pairs.begin(),
                   [](cudf::size_type lhs, cudf::size_type rhs) { return std::pair{lhs, rhs}; });
    std::sort(result_pairs.begin(), result_pairs.end());
    std::sort(expected_outputs.begin(), expected_outputs.end());

    EXPECT_TRUE(std::equal(
      expected_outputs.begin(), expected_outputs.end(), result_pairs.begin(), result_pairs.end()));
  }

  /*
   * Perform a join of tables constructed from two input data sets according to
   * the provided predicate and verify that the outputs match the expected
   * outputs (up to order).
   */
  void test(ColumnVector<T> left_data,
            ColumnVector<T> right_data,
            cudf::ast::operation predicate,
            std::vector<std::pair<cudf::size_type, cudf::size_type>> expected_outputs)
  {
    // Note that we need to maintain the column wrappers otherwise the
    // resulting column views will be referencing potentially invalid memory.
    auto [left_wrappers, right_wrappers, left_columns, right_columns, left, right] =
      this->parse_input(left_data, right_data);
    this->_test(left, right, predicate, expected_outputs);
  }

  void test_nulls(NullableColumnVector<T> left_data,
                  NullableColumnVector<T> right_data,
                  cudf::ast::operation predicate,
                  std::vector<std::pair<cudf::size_type, cudf::size_type>> expected_outputs)
  {
    // Note that we need to maintain the column wrappers otherwise the
    // resulting column views will be referencing potentially invalid memory.
    auto [left_wrappers, right_wrappers, left_columns, right_columns, left, right] =
      this->parse_input(left_data, right_data);
    this->_test(left, right, predicate, expected_outputs);
  }

  /*
   * Perform a join of tables constructed from two input data sets according to
   * an equality predicate on all corresponding columns and verify that the outputs match the
   * expected outputs (up to order).
   */
  void _compare_to_hash_join(PairJoinReturn const& result, PairJoinReturn const& reference)
  {
    auto result_pairs =
      rmm::device_uvector<index_pair>(result.first->size(), cudf::get_default_stream());
    auto reference_pairs =
      rmm::device_uvector<index_pair>(reference.first->size(), cudf::get_default_stream());

    thrust::transform(rmm::exec_policy_nosync(cudf::get_default_stream()),
                      result.first->begin(),
                      result.first->end(),
                      result.second->begin(),
                      result_pairs.begin(),
                      [] __device__(cudf::size_type first, cudf::size_type second) {
                        return index_pair{first, second};
                      });
    thrust::transform(rmm::exec_policy_nosync(cudf::get_default_stream()),
                      reference.first->begin(),
                      reference.first->end(),
                      reference.second->begin(),
                      reference_pairs.begin(),
                      [] __device__(cudf::size_type first, cudf::size_type second) {
                        return index_pair{first, second};
                      });

    thrust::sort(rmm::exec_policy_nosync(cudf::get_default_stream()),
                 result_pairs.begin(),
                 result_pairs.end());
    thrust::sort(rmm::exec_policy_nosync(cudf::get_default_stream()),
                 reference_pairs.begin(),
                 reference_pairs.end());

    EXPECT_TRUE(thrust::equal(rmm::exec_policy_nosync(cudf::get_default_stream()),
                              reference_pairs.begin(),
                              reference_pairs.end(),
                              result_pairs.begin()));
  }

  void compare_to_hash_join(ColumnVector<T> left_data, ColumnVector<T> right_data)
  {
    // Note that we need to maintain the column wrappers otherwise the
    // resulting column views will be referencing potentially invalid memory.
    auto [left_wrappers, right_wrappers, left_columns, right_columns, left, right] =
      this->parse_input(left_data, right_data);
    auto result    = this->join(left, right, left_zero_eq_right_zero);
    auto reference = this->reference_join(left, right);
    this->_compare_to_hash_join(result, reference);
  }

  void compare_to_hash_join_nulls(NullableColumnVector<T> left_data,
                                  NullableColumnVector<T> right_data)
  {
    // Note that we need to maintain the column wrappers otherwise the
    // resulting column views will be referencing potentially invalid memory.
    auto [left_wrappers, right_wrappers, left_columns, right_columns, left, right] =
      this->parse_input(left_data, right_data);

    // Test comparing nulls as equal (the default for ref joins, uses NULL_EQUAL for AST
    // expression).
    auto predicate =
      cudf::ast::operation(cudf::ast::ast_operator::NULL_EQUAL, col_ref_left_0, col_ref_right_0);
    auto result    = this->join(left, right, predicate);
    auto reference = this->reference_join(left, right);
    this->_compare_to_hash_join(result, reference);

    // Test comparing nulls as equal (null_equality::UNEQUAL for ref joins, uses EQUAL for AST
    // expression).
    result    = this->join(left, right, left_zero_eq_right_zero);
    reference = this->reference_join(left, right, cudf::null_equality::UNEQUAL);
    this->_compare_to_hash_join(result, reference);
  }

  /**
   * This method must be implemented by subclasses for specific types of joins.
   * It should be a simply forwarding of arguments to the appropriate cudf
   * conditional join API.
   */
  virtual PairJoinReturn join(cudf::table_view left,
                              cudf::table_view right,
                              cudf::ast::operation predicate) = 0;

  /**
   * This method must be implemented by subclasses for specific types of joins.
   * It should be a simply forwarding of arguments to the appropriate cudf
   * conditional join size computation API.
   */
  virtual std::size_t join_size(cudf::table_view left,
                                cudf::table_view right,
                                cudf::ast::operation predicate) = 0;

  /**
   * This method must be implemented by subclasses for specific types of joins.
   * It should be a simply forwarding of arguments to the appropriate cudf
   * hash join API for comparison with conditional joins.
   */
  virtual PairJoinReturn reference_join(
    cudf::table_view left,
    cudf::table_view right,
    cudf::null_equality compare_nulls = cudf::null_equality::EQUAL) = 0;
};

/**
 * Tests of conditional inner joins.
 */
template <typename T>
struct ConditionalInnerJoinTest : public ConditionalJoinPairReturnTest<T> {
  PairJoinReturn join(cudf::table_view left,
                      cudf::table_view right,
                      cudf::ast::operation predicate) override
  { return cudf::conditional_inner_join(left, right, predicate); }

  std::size_t join_size(cudf::table_view left,
                        cudf::table_view right,
                        cudf::ast::operation predicate) override
  { return cudf::conditional_inner_join_size(left, right, predicate); }

  PairJoinReturn reference_join(
    cudf::table_view left,
    cudf::table_view right,
    cudf::null_equality compare_nulls = cudf::null_equality::EQUAL) override
  { return cudf::inner_join(left, right, compare_nulls); }
};

TYPED_TEST_SUITE(ConditionalInnerJoinTest, cudf::test::IntegralTypesNotBool);

TYPED_TEST(ConditionalInnerJoinTest, TestOneColumnOneRowAllEqual)
{ this->test({{0}}, {{0}}, left_zero_eq_right_zero, {{0, 0}}); };

TYPED_TEST(ConditionalInnerJoinTest, TestOneColumnLeftEmpty)
{ this->test({{}}, {{3, 4, 5}}, left_zero_eq_right_zero, {}); };

TYPED_TEST(ConditionalInnerJoinTest, TestOneColumnRightEmpty)
{ this->test({{3, 4, 5}}, {{}}, left_zero_eq_right_zero, {}); };

TYPED_TEST(ConditionalInnerJoinTest, TestOneColumnTwoRowAllEqual)
{ this->test({{0, 1}}, {{0, 0}}, left_zero_eq_right_zero, {{0, 0}, {0, 1}}); };

TYPED_TEST(ConditionalInnerJoinTest, TestTwoColumnOneRowAllEqual)
{ this->test({{0}, {0}}, {{0}, {0}}, left_zero_eq_right_zero, {{0, 0}}); };

TYPED_TEST(ConditionalInnerJoinTest, TestTwoColumnThreeRowAllEqual)
{
  this->test({{0, 1, 2}, {10, 20, 30}},
             {{0, 1, 2}, {30, 40, 50}},
             left_zero_eq_right_zero,
             {{0, 0}, {1, 1}, {2, 2}});
};

TYPED_TEST(ConditionalInnerJoinTest, TestTwoColumnThreeRowSomeEqual)
{
  this->test({{0, 1, 2}, {10, 20, 30}},
             {{0, 1, 3}, {30, 40, 50}},
             left_zero_eq_right_zero,
             {{0, 0}, {1, 1}});
};

TYPED_TEST(ConditionalInnerJoinTest, TestNotComparison)
{
  auto col_ref_0  = cudf::ast::column_reference(0);
  auto expression = cudf::ast::operation(cudf::ast::ast_operator::NOT, col_ref_0);

  this->test({{0, 1, 2}}, {{3, 4, 5}}, expression, {{0, 0}, {0, 1}, {0, 2}});
};

TYPED_TEST(ConditionalInnerJoinTest, TestGreaterComparison)
{
  auto col_ref_0  = cudf::ast::column_reference(0);
  auto col_ref_1  = cudf::ast::column_reference(0, cudf::ast::table_reference::RIGHT);
  auto expression = cudf::ast::operation(cudf::ast::ast_operator::GREATER, col_ref_0, col_ref_1);

  this->test({{0, 1, 2}}, {{1, 0, 0}}, expression, {{1, 1}, {1, 2}, {2, 0}, {2, 1}, {2, 2}});
};

TYPED_TEST(ConditionalInnerJoinTest, TestGreaterTwoColumnComparison)
{
  auto col_ref_0  = cudf::ast::column_reference(0);
  auto col_ref_1  = cudf::ast::column_reference(1, cudf::ast::table_reference::RIGHT);
  auto expression = cudf::ast::operation(cudf::ast::ast_operator::GREATER, col_ref_0, col_ref_1);

  this->test({{0, 1, 2}, {0, 0, 0}},
             {{0, 0, 0}, {1, 0, 0}},
             expression,
             {{1, 1}, {1, 2}, {2, 0}, {2, 1}, {2, 2}});
};

TYPED_TEST(ConditionalInnerJoinTest, TestGreaterDifferentNumberColumnComparison)
{
  auto col_ref_0  = cudf::ast::column_reference(0);
  auto col_ref_1  = cudf::ast::column_reference(1, cudf::ast::table_reference::RIGHT);
  auto expression = cudf::ast::operation(cudf::ast::ast_operator::GREATER, col_ref_0, col_ref_1);

  this->test(
    {{0, 1, 2}}, {{0, 0, 0}, {1, 0, 0}}, expression, {{1, 1}, {1, 2}, {2, 0}, {2, 1}, {2, 2}});
};

TYPED_TEST(ConditionalInnerJoinTest, TestGreaterDifferentNumberColumnDifferentSizeComparison)
{
  auto col_ref_0  = cudf::ast::column_reference(0);
  auto col_ref_1  = cudf::ast::column_reference(1, cudf::ast::table_reference::RIGHT);
  auto expression = cudf::ast::operation(cudf::ast::ast_operator::GREATER, col_ref_0, col_ref_1);

  this->test({{0, 1}}, {{0, 0, 0}, {1, 0, 0}}, expression, {{1, 1}, {1, 2}});
};

TYPED_TEST(ConditionalInnerJoinTest, TestComplexConditionMultipleColumns)
{
  // LEFT is implicit, but specifying explicitly to validate that it works.
  auto col_ref_0      = cudf::ast::column_reference(0, cudf::ast::table_reference::LEFT);
  auto scalar_1       = cudf::numeric_scalar<TypeParam>(1);
  auto literal_1      = cudf::ast::literal(scalar_1);
  auto left_0_equal_1 = cudf::ast::operation(cudf::ast::ast_operator::EQUAL, col_ref_0, literal_1);

  auto col_ref_1 = cudf::ast::column_reference(1, cudf::ast::table_reference::RIGHT);
  auto comparison_filter =
    cudf::ast::operation(cudf::ast::ast_operator::LESS, col_ref_1, col_ref_0);

  auto expression =
    cudf::ast::operation(cudf::ast::ast_operator::LOGICAL_AND, left_0_equal_1, comparison_filter);

  this->test({{0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2}, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}},
             {{0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2},
              {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11},
              {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
             expression,
             {{4, 0}, {5, 0}, {6, 0}, {7, 0}});
};

TYPED_TEST(ConditionalInnerJoinTest, TestSymmetry)
{
  auto col_ref_0  = cudf::ast::column_reference(0);
  auto col_ref_1  = cudf::ast::column_reference(0, cudf::ast::table_reference::RIGHT);
  auto expression = cudf::ast::operation(cudf::ast::ast_operator::GREATER, col_ref_1, col_ref_0);
  auto expression_reverse =
    cudf::ast::operation(cudf::ast::ast_operator::LESS, col_ref_0, col_ref_1);

  this->test(
    {{0, 1, 2}}, {{1, 2, 3}}, expression, {{0, 0}, {0, 1}, {0, 2}, {1, 1}, {1, 2}, {2, 2}});
  this->test(
    {{0, 1, 2}}, {{1, 2, 3}}, expression_reverse, {{0, 0}, {0, 1}, {0, 2}, {1, 1}, {1, 2}, {2, 2}});
};

TYPED_TEST(ConditionalInnerJoinTest, TestCompareRandomToHash)
{
  auto [left, right] = gen_random_repeated_columns<TypeParam>();
  this->compare_to_hash_join({left}, {right});
};

TYPED_TEST(ConditionalInnerJoinTest, TestCompareRandomToHashNulls)
{
  auto [left, right] = gen_random_nullable_repeated_columns<TypeParam>();
  this->compare_to_hash_join_nulls({left}, {right});
};

TYPED_TEST(ConditionalInnerJoinTest, TestCompareRandomToHashNullsLargerLeft)
{
  auto [left, right] = gen_random_repeated_columns<TypeParam>(2000, 10, 1000, 10);
  this->compare_to_hash_join({left}, {right});
};

TYPED_TEST(ConditionalInnerJoinTest, TestCompareRandomToHashNullsLargerRight)
{
  auto [left, right] = gen_random_repeated_columns<TypeParam>(1000, 10, 2000, 10);
  this->compare_to_hash_join({left}, {right});
};

TYPED_TEST(ConditionalInnerJoinTest, TestOneColumnTwoNullsRowAllEqual)
{
  this->test_nulls(
    {{{0, 1}, {1, 0}}}, {{{0, 0}, {1, 1}}}, left_zero_eq_right_zero, {{0, 0}, {0, 1}});
};

TYPED_TEST(ConditionalInnerJoinTest, TestOneColumnTwoNullsNoOutputRowAllEqual)
{ this->test_nulls({{{0, 1}, {0, 1}}}, {{{0, 0}, {1, 1}}}, left_zero_eq_right_zero, {}); };

/**
 * Tests of conditional left joins.
 */
template <typename T>
struct ConditionalLeftJoinTest : public ConditionalJoinPairReturnTest<T> {
  PairJoinReturn join(cudf::table_view left,
                      cudf::table_view right,
                      cudf::ast::operation predicate) override
  { return cudf::conditional_left_join(left, right, predicate); }

  std::size_t join_size(cudf::table_view left,
                        cudf::table_view right,
                        cudf::ast::operation predicate) override
  { return cudf::conditional_left_join_size(left, right, predicate); }

  PairJoinReturn reference_join(
    cudf::table_view left,
    cudf::table_view right,
    cudf::null_equality compare_nulls = cudf::null_equality::EQUAL) override
  { return cudf::left_join(left, right, compare_nulls); }
};

TYPED_TEST_SUITE(ConditionalLeftJoinTest, cudf::test::IntegralTypesNotBool);

TYPED_TEST(ConditionalLeftJoinTest, TestTwoColumnThreeRowSomeEqual)
{
  this->test({{0, 1, 2}, {10, 20, 30}},
             {{0, 1, 3}, {30, 40, 50}},
             left_zero_eq_right_zero,
             {{0, 0}, {1, 1}, {2, cudf::JoinNoMatch}});
};

TYPED_TEST(ConditionalLeftJoinTest, TestOneColumnLeftEmpty)
{ this->test({{}}, {{3, 4, 5}}, left_zero_eq_right_zero, {}); };

TYPED_TEST(ConditionalLeftJoinTest, TestOneColumnRightEmpty)
{
  this->test({{3, 4, 5}},
             {{}},
             left_zero_eq_right_zero,
             {{0, cudf::JoinNoMatch}, {1, cudf::JoinNoMatch}, {2, cudf::JoinNoMatch}});
};

TYPED_TEST(ConditionalLeftJoinTest, TestCompareRandomToHash)
{
  auto [left, right] = gen_random_repeated_columns<TypeParam>();
  this->compare_to_hash_join({left}, {right});
};

TYPED_TEST(ConditionalLeftJoinTest, TestCompareRandomToHashNulls)
{
  auto [left, right] = gen_random_repeated_columns<TypeParam>();
  this->compare_to_hash_join({left}, {right});
};

/**
 * Tests of conditional full joins.
 */
template <typename T>
struct ConditionalFullJoinTest : public ConditionalJoinPairReturnTest<T> {
  PairJoinReturn join(cudf::table_view left,
                      cudf::table_view right,
                      cudf::ast::operation predicate) override
  { return cudf::conditional_full_join(left, right, predicate); }

  std::size_t join_size(cudf::table_view left,
                        cudf::table_view right,
                        cudf::ast::operation predicate) override
  {
    // Full joins don't actually support size calculations, but to support a
    // uniform testing framework we just calculate it from the result of doing
    // the join.
    return cudf::conditional_full_join(left, right, predicate).first->size();
  }

  PairJoinReturn reference_join(
    cudf::table_view left,
    cudf::table_view right,
    cudf::null_equality compare_nulls = cudf::null_equality::EQUAL) override
  { return cudf::full_join(left, right, compare_nulls); }
};

TYPED_TEST_SUITE(ConditionalFullJoinTest, cudf::test::IntegralTypesNotBool);

TYPED_TEST(ConditionalFullJoinTest, TestOneColumnNoneEqual)
{
  this->test({{0, 1, 2}},
             {{3, 4, 5}},
             left_zero_eq_right_zero,
             {{0, cudf::JoinNoMatch},
              {1, cudf::JoinNoMatch},
              {2, cudf::JoinNoMatch},
              {cudf::JoinNoMatch, 0},
              {cudf::JoinNoMatch, 1},
              {cudf::JoinNoMatch, 2}});
};

TYPED_TEST(ConditionalFullJoinTest, TestOneColumnLeftEmpty)
{
  this->test({{}},
             {{3, 4, 5}},
             left_zero_eq_right_zero,
             {{cudf::JoinNoMatch, 0}, {cudf::JoinNoMatch, 1}, {cudf::JoinNoMatch, 2}});
};

TYPED_TEST(ConditionalFullJoinTest, TestOneColumnRightEmpty)
{
  this->test({{3, 4, 5}},
             {{}},
             left_zero_eq_right_zero,
             {{0, cudf::JoinNoMatch}, {1, cudf::JoinNoMatch}, {2, cudf::JoinNoMatch}});
};

TYPED_TEST(ConditionalFullJoinTest, TestTwoColumnThreeRowSomeEqual)
{
  this->test({{0, 1, 2}, {10, 20, 30}},
             {{0, 1, 3}, {30, 40, 50}},
             left_zero_eq_right_zero,
             {{0, 0}, {1, 1}, {2, cudf::JoinNoMatch}, {cudf::JoinNoMatch, 2}});
};

TYPED_TEST(ConditionalFullJoinTest, TestCompareRandomToHash)
{
  auto [left, right] = gen_random_repeated_columns<TypeParam>();
  this->compare_to_hash_join({left}, {right});
};

TYPED_TEST(ConditionalFullJoinTest, TestCompareRandomToHashNulls)
{
  auto [left, right] = gen_random_repeated_columns<TypeParam>();
  this->compare_to_hash_join({left}, {right});
};

/**
 * Fixture for join types that return both only left indices (left semi and
 * left anti).
 */
template <typename T>
struct ConditionalJoinSingleReturnTest : public ConditionalJoinTest<T> {
  /*
   * Perform a join of tables constructed from two input data sets according to
   * the provided predicate and verify that the outputs match the expected
   * outputs (up to order).
   */
  void test(ColumnVector<T> left_data,
            ColumnVector<T> right_data,
            cudf::ast::operation predicate,
            std::vector<cudf::size_type> expected_outputs)
  {
    auto [left_wrappers, right_wrappers, left_columns, right_columns, left, right] =
      this->parse_input(left_data, right_data);
    auto result_size = this->join_size(left, right, predicate);
    EXPECT_EQ(result_size, expected_outputs.size());

    auto result         = this->join(left, right, predicate);
    auto result_indices = cudf::detail::make_std_vector(*result, cudf::get_default_stream());
    std::sort(result_indices.begin(), result_indices.end());
    std::sort(expected_outputs.begin(), expected_outputs.end());
    EXPECT_TRUE(std::equal(result_indices.begin(),
                           result_indices.end(),
                           expected_outputs.begin(),
                           expected_outputs.end()));
  }

  void _compare_to_hash_join(std::unique_ptr<rmm::device_uvector<cudf::size_type>> const& result,
                             std::unique_ptr<rmm::device_uvector<cudf::size_type>> const& reference)
  {
    thrust::sort(
      rmm::exec_policy_nosync(cudf::get_default_stream()), result->begin(), result->end());
    thrust::sort(
      rmm::exec_policy_nosync(cudf::get_default_stream()), reference->begin(), reference->end());
    EXPECT_TRUE(thrust::equal(rmm::exec_policy_nosync(cudf::get_default_stream()),
                              result->begin(),
                              result->end(),
                              reference->begin()));
  }

  /*
   * Perform a join of tables constructed from two input data sets according to
   * an equality predicate on all corresponding columns and verify that the outputs match the
   * expected outputs (up to order).
   */
  void compare_to_hash_join(ColumnVector<T> left_data, ColumnVector<T> right_data)
  {
    // Note that we need to maintain the column wrappers otherwise the
    // resulting column views will be referencing potentially invalid memory.
    auto [left_wrappers, right_wrappers, left_columns, right_columns, left, right] =
      this->parse_input(left_data, right_data);
    auto result    = this->join(left, right, left_zero_eq_right_zero);
    auto reference = this->reference_join(left, right);
    this->_compare_to_hash_join(result, reference);
  }

  void compare_to_hash_join_nulls(NullableColumnVector<T> left_data,
                                  NullableColumnVector<T> right_data)
  {
    // Note that we need to maintain the column wrappers otherwise the
    // resulting column views will be referencing potentially invalid memory.
    auto [left_wrappers, right_wrappers, left_columns, right_columns, left, right] =
      this->parse_input(left_data, right_data);
    auto predicate =
      cudf::ast::operation(cudf::ast::ast_operator::NULL_EQUAL, col_ref_left_0, col_ref_right_0);
    auto result    = this->join(left, right, predicate);
    auto reference = this->reference_join(left, right);
    this->_compare_to_hash_join(result, reference);

    result    = this->join(left, right, left_zero_eq_right_zero);
    reference = this->reference_join(left, right, cudf::null_equality::UNEQUAL);
    this->_compare_to_hash_join(result, reference);
  }

  /**
   * This method must be implemented by subclasses for specific types of joins.
   * It should be a simply forwarding of arguments to the appropriate cudf
   * conditional join API.
   */
  virtual SingleJoinReturn join(cudf::table_view left,
                                cudf::table_view right,
                                cudf::ast::operation predicate) = 0;

  /**
   * This method must be implemented by subclasses for specific types of joins.
   * It should be a simply forwarding of arguments to the appropriate cudf
   * conditional join size computation API.
   */
  virtual std::size_t join_size(cudf::table_view left,
                                cudf::table_view right,
                                cudf::ast::operation predicate) = 0;

  /**
   * This method must be implemented by subclasses for specific types of joins.
   * It should be a simply forwarding of arguments to the appropriate cudf
   * hash join API for comparison with conditional joins.
   */
  virtual SingleJoinReturn reference_join(
    cudf::table_view left,
    cudf::table_view right,
    cudf::null_equality compare_nulls = cudf::null_equality::EQUAL) = 0;
};

/**
 * Tests of conditional left semi joins.
 */
template <typename T>
struct ConditionalLeftSemiJoinTest : public ConditionalJoinSingleReturnTest<T> {
  SingleJoinReturn join(cudf::table_view left,
                        cudf::table_view right,
                        cudf::ast::operation predicate) override
  { return cudf::conditional_left_semi_join(left, right, predicate); }

  std::size_t join_size(cudf::table_view left,
                        cudf::table_view right,
                        cudf::ast::operation predicate) override
  { return cudf::conditional_left_semi_join_size(left, right, predicate); }

  SingleJoinReturn reference_join(
    cudf::table_view left,
    cudf::table_view right,
    cudf::null_equality compare_nulls = cudf::null_equality::EQUAL) override
  {
    cudf::filtered_join obj(right, compare_nulls, cudf::get_default_stream());
    return obj.semi_join(left);
  }
};

TYPED_TEST_SUITE(ConditionalLeftSemiJoinTest, cudf::test::IntegralTypesNotBool);

TYPED_TEST(ConditionalLeftSemiJoinTest, TestOneColumnLeftEmpty)
{ this->test({{}}, {{3, 4, 5}}, left_zero_eq_right_zero, {}); };

TYPED_TEST(ConditionalLeftSemiJoinTest, TestOneColumnRightEmpty)
{ this->test({{3, 4, 5}}, {{}}, left_zero_eq_right_zero, {}); };

TYPED_TEST(ConditionalLeftSemiJoinTest, TestTwoColumnThreeRowSomeEqual)
{
  this->test({{0, 1, 2}, {10, 20, 30}}, {{0, 1, 3}, {30, 40, 50}}, left_zero_eq_right_zero, {0, 1});
};

TYPED_TEST(ConditionalLeftSemiJoinTest, TestCompareRandomToHash)
{
  auto [left, right] = gen_random_repeated_columns<TypeParam>();
  this->compare_to_hash_join({left}, {right});
};

TYPED_TEST(ConditionalLeftSemiJoinTest, TestCompareRandomToHashNulls)
{
  auto [left, right] = gen_random_nullable_repeated_columns<TypeParam>();
  this->compare_to_hash_join_nulls({left}, {right});
};

/**
 * Tests of conditional left anti joins.
 */
template <typename T>
struct ConditionalLeftAntiJoinTest : public ConditionalJoinSingleReturnTest<T> {
  SingleJoinReturn join(cudf::table_view left,
                        cudf::table_view right,
                        cudf::ast::operation predicate) override
  { return cudf::conditional_left_anti_join(left, right, predicate); }

  std::size_t join_size(cudf::table_view left,
                        cudf::table_view right,
                        cudf::ast::operation predicate) override
  { return cudf::conditional_left_anti_join_size(left, right, predicate); }

  SingleJoinReturn reference_join(
    cudf::table_view left,
    cudf::table_view right,
    cudf::null_equality compare_nulls = cudf::null_equality::EQUAL) override
  {
    cudf::filtered_join obj(right, compare_nulls, cudf::get_default_stream());
    return obj.anti_join(left);
  }
};

TYPED_TEST_SUITE(ConditionalLeftAntiJoinTest, cudf::test::IntegralTypesNotBool);

TYPED_TEST(ConditionalLeftAntiJoinTest, TestOneColumnLeftEmpty)
{ this->test({{}}, {{3, 4, 5}}, left_zero_eq_right_zero, {}); };

TYPED_TEST(ConditionalLeftAntiJoinTest, TestOneColumnRightEmpty)
{ this->test({{3, 4, 5}}, {{}}, left_zero_eq_right_zero, {0, 1, 2}); };

TYPED_TEST(ConditionalLeftAntiJoinTest, TestTwoColumnThreeRowSomeEqual)
{ this->test({{0, 1, 2}, {10, 20, 30}}, {{0, 1, 3}, {30, 40, 50}}, left_zero_eq_right_zero, {2}); };

TYPED_TEST(ConditionalLeftAntiJoinTest, TestCompareRandomToHash)
{
  auto [left, right] = gen_random_repeated_columns<TypeParam>();
  this->compare_to_hash_join({left}, {right});
};

TYPED_TEST(ConditionalLeftAntiJoinTest, TestCompareRandomToHashNulls)
{
  auto [left, right] = gen_random_nullable_repeated_columns<TypeParam>();
  this->compare_to_hash_join_nulls({left}, {right});
};
