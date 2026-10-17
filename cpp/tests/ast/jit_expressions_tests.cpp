/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/iterator_utilities.hpp>
#include <cudf_test/testing_main.hpp>
#include <cudf_test/type_list_utilities.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/ast/expressions.hpp>
#include <cudf/ast/jit/udf.hpp>
#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/filling.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/transform.hpp>
#include <cudf/utilities/type_dispatcher.hpp>
#include <cudf/wrappers/timestamps.hpp>

#include <rmm/cuda_stream.hpp>

#include <cuda/iterator>

#include <cudf_test_fragments.hpp>

#include <algorithm>
#include <array>
#include <functional>
#include <initializer_list>
#include <limits>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

constexpr cudf::test::debug_output_level VERBOSITY{cudf::test::debug_output_level::ALL_ERRORS};

template <typename T>
using column_wrapper = cudf::test::fixed_width_column_wrapper<T>;

template <typename T>
using decimal_column_wrapper = cudf::test::fixed_point_column_wrapper<typename T::rep>;

struct JITExpressionTest : public cudf::test::BaseFixture {};

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

namespace {

// Device functions compiled ahead of time to LTO-IR (transform/fragments/ast_udf_callees.cu).
std::span<uint8_t const> ast_udf_callees()
{
  auto const range = cudf_test_fragments::file_ranges[cudf_test_fragments::ast_udf_callees];
  return cudf_test_fragments::files.subspan(range[0], range[1]);
}

cudf::ast::jit::device_binary callee(char const* symbol, bool is_pure = true)
{
  return cudf::ast::jit::device_binary{
    ast_udf_callees(), cudf::lto_binary_type::FATBIN, symbol, is_pure};
}

auto const int32_type = cudf::data_type{cudf::type_id::INT32};

}  // namespace

TEST_F(JITExpressionTest, UdfCall)
{
  auto a        = column_wrapper<int32_t>{{1, 2, 3, -4}};
  auto expected = column_wrapper<int32_t>{{2, 3, 4, -3}};
  auto table    = cudf::table_view{{a}};
  auto tree     = cudf::ast::tree{};
  auto a_ref    = cudf::ast::column_reference(0);
  auto& call    = cudf::ast::jit::call(tree, callee("lto_add_one"), int32_type, {a_ref});
  auto result   = cudf::compute_column_jit(table, call);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view(), VERBOSITY);
}

TEST_F(JITExpressionTest, UdfCallFusesWithOperators)
{
  auto a        = column_wrapper<int32_t>{{1, 2, 3}};
  auto b        = column_wrapper<int32_t>{{10, 20, 30}};
  auto expected = column_wrapper<int32_t>{{22, 66, 132}};
  auto table    = cudf::table_view{{a, b}};
  auto tree     = cudf::ast::tree{};
  auto a_ref    = cudf::ast::column_reference(0);
  auto b_ref    = cudf::ast::column_reference(1);
  auto& sum     = cudf::ast::jit::operation(tree, cudf::ast::jit::op::ADD, {a_ref, b_ref});
  auto& plus_1  = cudf::ast::jit::call(tree, callee("lto_add_one"), int32_type, {a_ref});
  // (a + 1) * (a + b): two expressions over the same column, one kernel.
  auto& product = cudf::ast::jit::operation(tree, cudf::ast::jit::op::MUL, {plus_1, sum});
  auto result   = cudf::compute_column_jit(table, product);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view(), VERBOSITY);
}

TEST_F(JITExpressionTest, UdfCallNestsCallees)
{
  // Two functions from one fragment, one calling the result of the other: 2(a + 1).
  auto a        = column_wrapper<int32_t>{{1, 2, 3}};
  auto expected = column_wrapper<int32_t>{{4, 6, 8}};
  auto table    = cudf::table_view{{a}};
  auto tree     = cudf::ast::tree{};
  auto a_ref    = cudf::ast::column_reference(0);
  auto& plus_1  = cudf::ast::jit::call(tree, callee("lto_add_one"), int32_type, {a_ref});
  auto& doubled = cudf::ast::jit::call(tree, callee("lto_twice"), int32_type, {plus_1});
  auto result   = cudf::compute_column_jit(table, doubled);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view(), VERBOSITY);
}

TEST_F(JITExpressionTest, UdfCallCommonSubexpressions)
{
  // Equal pure calls across outputs share one evaluation, calls of different functions over the
  // same argument stay apart, and separate impure calls are never merged. The generated code is
  // checked in ROW_IR_TEST; this checks the results.
  auto a        = column_wrapper<int32_t>{{1, 2, 3}};
  auto table    = cudf::table_view{{a}};
  auto tree     = cudf::ast::tree{};
  auto a_ref    = cudf::ast::column_reference(0);
  auto& one_a   = cudf::ast::jit::call(tree, callee("lto_add_one"), int32_type, {a_ref});
  auto& one_b   = cudf::ast::jit::call(tree, callee("lto_add_one"), int32_type, {a_ref});
  auto& two     = cudf::ast::jit::call(tree, callee("lto_twice"), int32_type, {a_ref});
  auto& sum     = cudf::ast::jit::operation(tree, cudf::ast::jit::op::ADD, {one_b, two});
  auto& twice_a = cudf::ast::jit::call(tree, callee("lto_twice", false), int32_type, {a_ref});
  auto& twice_b = cudf::ast::jit::call(tree, callee("lto_twice", false), int32_type, {a_ref});
  auto& impure  = cudf::ast::jit::operation(tree, cudf::ast::jit::op::ADD, {twice_a, twice_b});
  auto expressions =
    std::to_array<std::reference_wrapper<cudf::ast::expression const>>({one_a, sum, two});
  auto result = cudf::compute_table_jit(table, expressions);

  ASSERT_EQ(result->num_columns(), 3);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    column_wrapper<int32_t>{{2, 3, 4}}, result->view().column(0), VERBOSITY);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    column_wrapper<int32_t>{{4, 7, 10}}, result->view().column(1), VERBOSITY);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    column_wrapper<int32_t>{{2, 4, 6}}, result->view().column(2), VERBOSITY);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(column_wrapper<int32_t>{{4, 8, 12}},
                                 cudf::compute_column_jit(table, impure)->view(),
                                 VERBOSITY);
}

TEST_F(JITExpressionTest, UdfCallImpureCalls)
{
  // lto_next_id returns a new number on every call. One call referred to twice is evaluated once
  // per row, so subtracting it from itself gives zero. Two separate calls are never merged, so
  // their difference is never zero.
  auto a         = column_wrapper<int32_t>{{1, 2, 3}};
  auto table     = cudf::table_view{{a}};
  auto tree      = cudf::ast::tree{};
  auto a_ref     = cudf::ast::column_reference(0);
  auto int64     = cudf::data_type{cudf::type_id::INT64};
  auto& id       = cudf::ast::jit::call(tree, callee("lto_next_id", false), int64, {a_ref});
  auto& other_id = cudf::ast::jit::call(tree, callee("lto_next_id", false), int64, {a_ref});
  auto& same     = cudf::ast::jit::operation(tree, cudf::ast::jit::op::SUB, {id, id});
  auto& apart    = cudf::ast::jit::operation(tree, cudf::ast::jit::op::SUB, {id, other_id});

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    column_wrapper<int64_t>{{0, 0, 0}}, cudf::compute_column_jit(table, same)->view(), VERBOSITY);
  auto const differences =
    cudf::test::to_host<int64_t>(cudf::compute_column_jit(table, apart)->view()).first;
  EXPECT_TRUE(std::none_of(
    differences.begin(), differences.end(), [](int64_t difference) { return difference == 0; }));
}

TEST_F(JITExpressionTest, UdfCallPropagatesNulls)
{
  auto a        = column_wrapper<int32_t>{{1, 2, 3}, {1, 0, 1}};
  auto expected = column_wrapper<int32_t>{{2, 0, 4}, {1, 0, 1}};
  auto table    = cudf::table_view{{a}};
  auto tree     = cudf::ast::tree{};
  auto a_ref    = cudf::ast::column_reference(0);
  auto& call    = cudf::ast::jit::call(tree, callee("lto_add_one"), int32_type, {a_ref});
  auto result   = cudf::compute_column_jit(table, call);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view(), VERBOSITY);
}

TEST_F(JITExpressionTest, UdfCallErrorPolicies)
{
  auto a          = column_wrapper<int32_t>{{8, -2, 6}};
  auto table      = cudf::table_view{{a}};
  auto tree       = cudf::ast::tree{};
  auto a_ref      = cudf::ast::column_reference(0);
  auto& halve     = cudf::ast::jit::call(tree, callee("lto_checked_halve"), int32_type, {a_ref});
  auto& try_halve = cudf::ast::jit::call(
    tree, callee("lto_checked_halve"), int32_type, {a_ref}, cudf::error_policy::NULLIFY);

  try {
    std::ignore = cudf::compute_column_jit(table, halve);
    FAIL() << "expected cudf::evaluation_error";
  } catch (cudf::evaluation_error const& e) {
    EXPECT_EQ(e.error_code(), cudf::errc::ARITHMETIC_OVERFLOW);
  }

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(column_wrapper<int32_t>{{4, 0, 3}, {1, 0, 1}},
                                 cudf::compute_column_jit(table, try_halve)->view(),
                                 VERBOSITY);
}

TEST_F(JITExpressionTest, UdfCallNullifyWithInfallibleCallee)
{
  // NULLIFY adds no nulls when no row fails.
  auto a        = column_wrapper<int32_t>{{1, 2, 3}};
  auto nullable = column_wrapper<int32_t>{{1, 2, 3}, {1, 0, 1}};
  auto tree     = cudf::ast::tree{};
  auto a_ref    = cudf::ast::column_reference(0);
  auto& call    = cudf::ast::jit::call(
    tree, callee("lto_add_one"), int32_type, {a_ref}, cudf::error_policy::NULLIFY);

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(column_wrapper<int32_t>{{2, 3, 4}},
                                      cudf::compute_column_jit(cudf::table_view{{a}}, call)->view(),
                                      VERBOSITY);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    column_wrapper<int32_t>{{2, 0, 4}, {1, 0, 1}},
    cudf::compute_column_jit(cudf::table_view{{nullable}}, call)->view(),
    VERBOSITY);
}

TEST_F(JITExpressionTest, UdfCallHighestErrorCodeWins)
{
  // Rows fail with ARITHMETIC_OVERFLOW before and after the row that fails with DIVISION_BY_ZERO.
  auto a     = column_wrapper<int32_t>{{-1, 4, 0, -3}};
  auto table = cudf::table_view{{a}};
  auto tree  = cudf::ast::tree{};
  auto a_ref = cudf::ast::column_reference(0);
  auto& call = cudf::ast::jit::call(tree, callee("lto_checked_hundred_over"), int32_type, {a_ref});

  try {
    std::ignore = cudf::compute_column_jit(table, call);
    FAIL() << "expected cudf::evaluation_error";
  } catch (cudf::evaluation_error const& e) {
    EXPECT_EQ(e.error_code(), cudf::errc::DIVISION_BY_ZERO);
  }
}

TEST_F(JITExpressionTest, UdfCallUserErrorCodes)
{
  // lto_fail_with fails each row with the code it holds. The codes libcudf reserves for
  // user-defined functions reach the caller unchanged, and as the highest codes they are reported
  // over libcudf's own.
  auto const code     = [](cudf::errc error) { return static_cast<int32_t>(error); };
  auto const error_of = [](std::initializer_list<int32_t> codes) {
    auto column = column_wrapper<int32_t>(codes.begin(), codes.end());
    auto tree   = cudf::ast::tree{};
    auto ref    = cudf::ast::column_reference(0);
    auto& call  = cudf::ast::jit::call(tree, callee("lto_fail_with"), int32_type, {ref});
    try {
      std::ignore = cudf::compute_column_jit(cudf::table_view{{column}}, call);
    } catch (cudf::evaluation_error const& e) {
      EXPECT_NE(std::string_view{e.what()}.find(cudf::to_string(e.error_code())),
                std::string_view::npos);
      return e.error_code();
    }
    return cudf::errc::SUCCESS;
  };

  EXPECT_EQ(error_of({0, code(cudf::errc::USER_ERROR_0), 0}), cudf::errc::USER_ERROR_0);
  EXPECT_EQ(error_of({code(cudf::errc::USER_ERROR_31), 0, code(cudf::errc::USER_ERROR_5)}),
            cudf::errc::USER_ERROR_31);
  EXPECT_EQ(error_of({code(cudf::errc::DIVISION_BY_ZERO),
                      code(cudf::errc::USER_ERROR_0),
                      code(cudf::errc::ARITHMETIC_OVERFLOW)}),
            cudf::errc::USER_ERROR_0);
  EXPECT_STREQ(cudf::to_string(cudf::errc::USER_ERROR_17), "USER_ERROR_17");
}

TEST_F(JITExpressionTest, UdfCallOverTimestamps)
{
  using cudf::timestamp_ns;
  using rep         = timestamp_ns::rep;
  constexpr rep day = 86'400'000'000'000;
  // Row 0: Monday 1970-01-05 and a day in its week; row 1: 1969-12-31 and 1969-12-29, both in the
  // week starting Monday 1969-12-29.
  auto a     = cudf::test::fixed_width_column_wrapper<timestamp_ns, rep>{{4 * day + 5, rep{-1}}};
  auto b     = cudf::test::fixed_width_column_wrapper<timestamp_ns, rep>{{10 * day + 7, -3 * day}};
  auto table = cudf::table_view{{a, b}};
  auto tree  = cudf::ast::tree{};
  auto a_ref = cudf::ast::column_reference(0);
  auto b_ref = cudf::ast::column_reference(1);
  auto type  = cudf::data_type{cudf::type_id::TIMESTAMP_NANOSECONDS};
  auto& week_a = cudf::ast::jit::call(tree, callee("lto_week_start_ns"), type, {a_ref});
  auto& week_b = cudf::ast::jit::call(tree, callee("lto_week_start_ns"), type, {b_ref});
  auto& same   = cudf::ast::jit::operation(tree, cudf::ast::jit::op::EQUAL, {week_a, week_b});
  auto result  = cudf::compute_column_jit(table, same);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(column_wrapper<bool>{{true, true}}, result->view(), VERBOSITY);
}

TEST_F(JITExpressionTest, UdfCallInTransformProgram)
{
  auto a     = column_wrapper<int32_t>{{1, 2, 3}};
  auto table = cudf::table_view{{a}};
  auto tree  = cudf::ast::tree{};
  auto a_ref = cudf::ast::column_reference(0);
  auto& call = cudf::ast::jit::call(tree, callee("lto_add_one"), int32_type, {a_ref});
  std::reference_wrapper<cudf::ast::expression const> expressions[] = {call};
  auto program = cudf::transform_program{table, expressions};

  auto other = column_wrapper<int32_t>{{10, 20}};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    column_wrapper<int32_t>{{2, 3, 4}}, program.run(table)->view().column(0), VERBOSITY);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(column_wrapper<int32_t>{{11, 21}},
                                 program.run(cudf::table_view{{other}})->view().column(0),
                                 VERBOSITY);
}

TEST_F(JITExpressionTest, UdfCallInterpreterRejectsCall)
{
  auto a     = column_wrapper<int32_t>{{1}};
  auto table = cudf::table_view{{a}};
  auto tree  = cudf::ast::tree{};
  auto a_ref = cudf::ast::column_reference(0);
  auto& call = cudf::ast::jit::call(tree, callee("lto_add_one"), int32_type, {a_ref});

  EXPECT_THROW(std::ignore = cudf::compute_column(table, call), std::invalid_argument);
}

TEST_F(JITExpressionTest, UdfCallValidatesArguments)
{
  auto tree  = cudf::ast::tree{};
  auto a_ref = cudf::ast::column_reference(0);
  EXPECT_THROW(std::ignore = cudf::ast::jit::call(tree, callee("lto_add_one"), int32_type, {}),
               std::invalid_argument);
  EXPECT_THROW(std::ignore = cudf::ast::jit::call(
                 tree, callee("lto_add_one"), cudf::data_type{cudf::type_id::STRING}, {a_ref}),
               std::invalid_argument);
  EXPECT_THROW(std::ignore = cudf::ast::jit::call(
                 tree,
                 cudf::ast::jit::device_binary{{}, cudf::lto_binary_type::FATBIN, "f"},
                 int32_type,
                 {a_ref}),
               std::invalid_argument);
  for (auto const* symbol : {"", "1f", "ns::f", "f(int)", "f g", "f;"}) {
    EXPECT_THROW(std::ignore = cudf::ast::jit::call(tree, callee(symbol), int32_type, {a_ref}),
                 std::invalid_argument)
      << symbol;
  }
}

TEST_F(JITExpressionTest, UdfCallRejectsTwoDefinitionsForOneSymbol)
{
  auto a     = column_wrapper<int32_t>{{1}};
  auto table = cudf::table_view{{a}};
  auto a_ref = cudf::ast::column_reference(0);
  auto copy  = std::vector<uint8_t>(ast_udf_callees().begin(), ast_udf_callees().end());
  auto other_fragment =
    cudf::ast::jit::device_binary{copy, cudf::lto_binary_type::FATBIN, "lto_add_one", true};
  for (auto const& other : {other_fragment, callee("lto_add_one", false)}) {
    auto tree = cudf::ast::tree{};
    auto& one = cudf::ast::jit::call(tree, callee("lto_add_one"), int32_type, {a_ref});
    auto& two = cudf::ast::jit::call(tree, other, int32_type, {a_ref});
    auto& sum = cudf::ast::jit::operation(tree, cudf::ast::jit::op::ADD, {one, two});

    EXPECT_THROW(std::ignore = cudf::compute_column_jit(table, sum), std::invalid_argument);
  }
}

TEST_F(JITExpressionTest, UdfCallRejectsTwoSignaturesForOneSymbol)
{
  // The symbol is declared once, from the types of its first call.
  auto a           = column_wrapper<int32_t>{{1}};
  auto b           = column_wrapper<int64_t>{{1}};
  auto table       = cudf::table_view{{a, b}};
  auto a_ref       = cudf::ast::column_reference(0);
  auto b_ref       = cudf::ast::column_reference(1);
  auto int64       = cudf::data_type{cudf::type_id::INT64};
  auto by_argument = cudf::ast::tree{};
  auto& narrow     = cudf::ast::jit::call(by_argument, callee("lto_add_one"), int32_type, {a_ref});
  auto& wide       = cudf::ast::jit::call(by_argument, callee("lto_add_one"), int32_type, {b_ref});
  auto& sum = cudf::ast::jit::operation(by_argument, cudf::ast::jit::op::ADD, {narrow, wide});
  EXPECT_THROW(std::ignore = cudf::compute_column_jit(table, sum), std::invalid_argument);

  auto by_output = cudf::ast::tree{};
  auto& as_int32 = cudf::ast::jit::call(by_output, callee("lto_add_one"), int32_type, {a_ref});
  auto& as_int64 = cudf::ast::jit::call(by_output, callee("lto_add_one"), int64, {a_ref});
  auto& as_int64s =
    cudf::ast::jit::operation(by_output, cudf::ast::jit::op::CAST_TO_INT64, {as_int32});
  auto& sum64 =
    cudf::ast::jit::operation(by_output, cudf::ast::jit::op::ADD, {as_int64s, as_int64});
  EXPECT_THROW(std::ignore = cudf::compute_column_jit(table, sum64), std::invalid_argument);
}

/// @brief Selects the fixture value type so integer and decimal cases share input factories.
template <typename T, bool = cudf::is_fixed_point<T>()>
struct overflow_rep {
  using type = T;
};

/// @brief Uses decimal storage values while column construction preserves the logical type.
template <typename T>
struct overflow_rep<T, true> {
  using type = typename T::rep;
};

/// @brief Provides the storage value type used by overflow fixture arrays.
template <typename T>
using overflow_rep_t = typename overflow_rep<T>::type;

/// @brief Creates typed fixture columns, retaining decimal semantics with scale zero.
template <typename T, std::size_t N>
std::unique_ptr<cudf::column> make_overflow_column(std::array<overflow_rep_t<T>, N> const& values)
{
  if constexpr (cudf::is_fixed_point<T>()) {
    return decimal_column_wrapper<T>(values.begin(), values.end(), numeric::scale_type{0})
      .release();
  } else {
    return column_wrapper<T>(values.begin(), values.end()).release();
  }
}

/// @brief Creates nullable expected columns so NULLIFY results retain type and validity coverage.
template <typename T, std::size_t N>
std::unique_ptr<cudf::column> make_overflow_column(std::array<overflow_rep_t<T>, N> const& values,
                                                   std::array<bool, N> const& validity)
{
  if constexpr (cudf::is_fixed_point<T>()) {
    return decimal_column_wrapper<T>(
             values.begin(), values.end(), validity.begin(), numeric::scale_type{0})
      .release();
  } else {
    return column_wrapper<T>(values.begin(), values.end(), validity.begin()).release();
  }
}

/// @brief Expands type coverage within one batch instead of separate typed GTest nodes.
template <typename... T, typename F>
void for_each_overflow_type(cudf::test::Types<T...>, F&& f)
{
  (f.template operator()<T>(), ...);
}

/// @brief Keeps binary boundary inputs and normal/NULLIFY expectations together for each type.
template <typename R>
struct binary_overflow_inputs {
  std::array<R, 4> a;
  std::array<R, 4> b;
  std::array<R, 4> b_fail;
  std::array<R, 4> expected;
  std::array<R, 4> expected_fail;
  std::array<bool, 4> validity;
};

/// @brief Keeps unary boundary inputs and normal/NULLIFY expectations together for each type.
template <typename R>
struct unary_overflow_inputs {
  std::array<R, 7> a;
  std::array<R, 7> a_fail;
  std::array<R, 7> expected;
  std::array<R, 7> expected_fail;
  std::array<bool, 7> validity;
};

/// @brief Covers integer arithmetic boundaries without treating booleans as numeric operands.
using integral_overflow_types = cudf::test::IntegralTypesNotBool;
/// @brief Restricts signed-only boundary cases to the four signed integer widths.
using signed_overflow_types = cudf::test::Types<int8_t, int16_t, int32_t, int64_t>;
/// @brief Shares binary fixtures across integer and decimal arithmetic with identical inputs.
using binary_overflow_types =
  cudf::test::Concat<integral_overflow_types, cudf::test::FixedPointTypes>;
/// @brief Shares signed boundary fixtures across signed integers and decimal representations.
using signed_decimal_overflow_types =
  cudf::test::Concat<signed_overflow_types, cudf::test::FixedPointTypes>;

/// @brief Owns fixture data and expressions to batch type coverage into fewer JIT compilations.
class overflow_batch {
  /// @brief Identifies interchangeable operands so each throwing case can be isolated without
  /// changing the expression graph or input schema.
  struct failure_case {
    cudf::size_type failing_input;
    cudf::size_type safe_input;
    std::string label;
  };

  cudf::ast::tree tree{};
  std::vector<std::unique_ptr<cudf::column>> inputs{};
  std::vector<std::unique_ptr<cudf::column>> expected{};
  std::vector<std::unique_ptr<cudf::scalar>> literals{};
  std::vector<std::reference_wrapper<cudf::ast::expression const>> outputs{};
  std::vector<std::reference_wrapper<cudf::ast::expression const>> throwing{};
  std::vector<std::string> labels{};
  std::vector<failure_case> failures{};

  /// @brief Retains column ownership for the borrowed views used during batch evaluation.
  cudf::size_type add_input(std::unique_ptr<cudf::column> input)
  {
    auto const index = static_cast<cudf::size_type>(inputs.size());
    inputs.push_back(std::move(input));
    return index;
  }

  /// @brief Gives operand references tree-owned lifetimes for the batch's expression graph.
  cudf::ast::column_reference const& add_reference(cudf::size_type column)
  {
    return tree.push(cudf::ast::column_reference(column));
  }

  /// @brief Registers normal and NULLIFY outputs together with isolated THROW expectations.
  template <typename T, std::size_t N>
  void append_case(
    cudf::ast::jit::op op,
    std::initializer_list<std::reference_wrapper<cudf::ast::expression const>> success_args,
    std::initializer_list<std::reference_wrapper<cudf::ast::expression const>> failure_args,
    std::array<overflow_rep_t<T>, N> const& expected_values,
    std::array<overflow_rep_t<T>, N> const& expected_fail_values,
    std::array<bool, N> const& validity,
    failure_case failure)
  {
    auto label = std::move(failure.label);
    if (!label.empty()) { label += ' '; }
    label += cudf::type_to_name(cudf::data_type{cudf::type_to_id<T>()});
    outputs.emplace_back(cudf::ast::jit::operation(tree, op, success_args));
    expected.push_back(make_overflow_column<T>(expected_values));
    labels.push_back(label + " success");
    outputs.emplace_back(
      cudf::ast::jit::operation(tree, op, failure_args, cudf::error_policy::NULLIFY));
    expected.push_back(make_overflow_column<T>(expected_fail_values, validity));
    labels.push_back(label + " NULLIFY");
    throwing.emplace_back(cudf::ast::jit::operation(tree, op, failure_args));
    failures.push_back({failure.failing_input, failure.safe_input, label + " THROW"});
  }

 public:
  /// @brief Adds one type's binary boundary coverage while keeping its failing operand replaceable.
  template <typename T>
  void append_binary(cudf::ast::jit::op op,
                     binary_overflow_inputs<overflow_rep_t<T>> const& values,
                     std::string_view operation_name = {})
  {
    auto const a_index      = add_input(make_overflow_column<T>(values.a));
    auto const b_index      = add_input(make_overflow_column<T>(values.b));
    auto const b_fail_index = add_input(make_overflow_column<T>(values.b_fail));
    auto const& a           = add_reference(a_index);
    auto const& b           = add_reference(b_index);
    auto const& b_fail      = add_reference(b_fail_index);
    append_case<T>(op,
                   {a, b},
                   {a, b_fail},
                   values.expected,
                   values.expected_fail,
                   values.validity,
                   {b_fail_index, b_index, std::string{operation_name}});
  }

  /// @brief Applies a shared binary fixture factory across types, preserving driver exclusions.
  template <cudf::ast::jit::op Op, typename... T, typename F>
  void append_binary_types(cudf::test::Types<T...> types,
                           F&& make_inputs,
                           std::string_view operation_name = {})
  {
    for_each_overflow_type(types, [&]<typename U>() {
      if constexpr (Op == cudf::ast::jit::op::MUL_OVERFLOW &&
                    std::is_same_v<U, numeric::decimal128>) {
        int driver_version{0};
        if (cudaDriverGetVersion(&driver_version) != cudaSuccess || driver_version < 12090) {
          return;
        }
      }
      append_binary<U>(Op, make_inputs.template operator()<U>(), operation_name);
    });
  }

  /// @brief Adds one type's unary boundary coverage while keeping its failing operand replaceable.
  template <typename T>
  void append_unary(cudf::ast::jit::op op, unary_overflow_inputs<overflow_rep_t<T>> const& values)
  {
    auto const a_index      = add_input(make_overflow_column<T>(values.a));
    auto const a_fail_index = add_input(make_overflow_column<T>(values.a_fail));
    auto const& a           = add_reference(a_index);
    auto const& a_fail      = add_reference(a_fail_index);
    append_case<T>(op,
                   {a},
                   {a_fail},
                   values.expected,
                   values.expected_fail,
                   values.validity,
                   {a_fail_index, a_index, {}});
  }

  /// @brief Applies a shared unary fixture factory across types within the same expression batch.
  template <cudf::ast::jit::op Op, typename... T, typename F>
  void append_unary_types(cudf::test::Types<T...> types, F&& make_inputs)
  {
    for_each_overflow_type(
      types, [&]<typename U>() { append_unary<U>(Op, make_inputs.template operator()<U>()); });
  }

  /// @brief Adds decimal precision boundaries and owns the scalar borrowed by their expressions.
  template <typename T>
  void append_precision()
  {
    using R            = overflow_rep_t<T>;
    auto const a_index = add_input(make_overflow_column<T>(std::array<R, 4>{3, 200, 250, 200}));
    auto const a_fail_index =
      add_input(make_overflow_column<T>(std::array<R, 4>{3, 200, 250, 20000}));
    auto const& a      = add_reference(a_index);
    auto const& a_fail = add_reference(a_fail_index);
    auto max_precision = std::make_unique<cudf::numeric_scalar<int32_t>>(3);
    auto& precision    = tree.push(cudf::ast::literal(*max_precision));
    literals.push_back(std::move(max_precision));
    append_case<T>(cudf::ast::jit::op::CHECK_PRECISION,
                   {a, precision},
                   {a_fail, precision},
                   std::array<R, 4>{3, 200, 250, 200},
                   std::array<R, 4>{3, 200, 250, 200},
                   {1, 1, 1, 0},
                   {a_fail_index, a_index, {}});
  }

  /// @brief Checks batched normal/NULLIFY outputs and each isolated THROW case with type
  /// diagnostics.
  void expect_results() const
  {
    std::vector<cudf::column_view> input_views;
    input_views.reserve(inputs.size());
    for (auto const& input : inputs) {
      input_views.push_back(input->view());
    }
    auto const table = cudf::table_view{input_views};

    auto result = cudf::compute_table_jit(table, outputs);
    ASSERT_EQ(result->num_columns(), static_cast<cudf::size_type>(expected.size()));
    for (cudf::size_type i = 0; i < result->num_columns(); ++i) {
      SCOPED_TRACE(labels[i]);
      CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected[i]->view(), result->view().column(i), VERBOSITY);
    }

    // PROPAGATE returns on the first error, so isolate each failure while keeping the expression
    // graph and input schema unchanged to reuse the compiled kernel.
    for (auto const& failure : failures) {
      input_views[failure.failing_input] = input_views[failure.safe_input];
    }
    ASSERT_NO_THROW(cudf::compute_table_jit(cudf::table_view{input_views}, throwing));
    for (auto const& failure : failures) {
      SCOPED_TRACE(failure.label);
      input_views[failure.failing_input] = inputs[failure.failing_input]->view();
      EXPECT_THROW(cudf::compute_table_jit(cudf::table_view{input_views}, throwing),
                   cudf::evaluation_error);
      input_views[failure.failing_input] = input_views[failure.safe_input];
    }
  }
};

TEST_F(JITExpressionTest, BinaryOverflow)
{
  overflow_batch batch;
  batch.append_binary_types<cudf::ast::jit::op::ADD_OVERFLOW>(
    binary_overflow_types{},
    []<typename T>() {
      using R = overflow_rep_t<T>;
      return binary_overflow_inputs<R>{{3, 20, 1, 50},
                                       {10, 7, 20, 0},
                                       {10, std::numeric_limits<R>::max(), 20, 0},
                                       {13, 27, 21, 50},
                                       {13, 0, 21, 50},
                                       {1, 0, 1, 1}};
    },
    "add");
  batch.append_binary_types<cudf::ast::jit::op::MUL_OVERFLOW>(
    integral_overflow_types{},
    []<typename T>() {
      using R = overflow_rep_t<T>;
      return binary_overflow_inputs<R>{{3, 20, 2, 50},
                                       {10, 2, 1, 0},
                                       {10, std::numeric_limits<R>::max(), 1, 0},
                                       {30, 40, 2, 0},
                                       {30, 0, 2, 0},
                                       {1, 0, 1, 1}};
    },
    "multiply");
  batch.append_binary_types<cudf::ast::jit::op::MUL_OVERFLOW>(
    cudf::test::FixedPointTypes{},
    []<typename T>() {
      using R = overflow_rep_t<T>;
      return binary_overflow_inputs<R>{{3, 20, 2, 50},
                                       {10, 7, 1, 0},
                                       {10, std::numeric_limits<R>::max(), 1, 0},
                                       {30, 140, 2, 0},
                                       {30, 0, 2, 0},
                                       {1, 0, 1, 1}};
    },
    "multiply");
  batch.append_binary_types<cudf::ast::jit::op::DIV_OVERFLOW>(
    binary_overflow_types{},
    []<typename T>() {
      using R = overflow_rep_t<T>;
      return binary_overflow_inputs<R>{
        {3, 20, 1, 50}, {10, 7, 2, 1}, {10, 1, 20, 0}, {0, 2, 0, 50}, {0, 20, 0, 50}, {1, 1, 1, 0}};
    },
    "divide");
  batch.append_binary_types<cudf::ast::jit::op::MOD_OVERFLOW>(
    binary_overflow_types{},
    []<typename T>() {
      using R = overflow_rep_t<T>;
      return binary_overflow_inputs<R>{
        {3, 20, 1, 50}, {10, 7, 2, 1}, {10, 1, 20, 0}, {3, 6, 1, 0}, {3, 0, 1, 0}, {1, 1, 1, 0}};
    },
    "modulo");
  batch.append_binary_types<cudf::ast::jit::op::SUB_OVERFLOW>(
    signed_decimal_overflow_types{},
    []<typename T>() {
      using R = overflow_rep_t<T>;
      return binary_overflow_inputs<R>{{3, 20, 1, 50},
                                       {10, 7, 20, 0},
                                       {10, std::numeric_limits<R>::min(), 20, 0},
                                       {-7, 13, -19, 50},
                                       {-7, 0, -19, 50},
                                       {1, 0, 1, 1}};
    },
    "subtract");
  batch.expect_results();
}

TEST_F(JITExpressionTest, AbsOverflow)
{
  overflow_batch batch;
  auto make_inputs = []<typename T>() {
    using R = overflow_rep_t<T>;
    return unary_overflow_inputs<R>{
      {R{3},
       R{-20},
       R{1},
       R{-50},
       std::numeric_limits<R>::max(),
       R{std::numeric_limits<R>::min() + 1},
       R{0}},
      {R{3}, R{-20}, R{1}, R{-50}, std::numeric_limits<R>::min(), R{1}, R{0}},
      {R{3},
       R{20},
       R{1},
       R{50},
       std::numeric_limits<R>::max(),
       R{std::abs(std::numeric_limits<R>::min() + 1)},
       R{0}},
      {R{3}, R{20}, R{1}, R{50}, R{0}, R{1}, R{0}},
      {1, 1, 1, 1, 0, 1, 1}};
  };
  batch.append_unary_types<cudf::ast::jit::op::ABS_OVERFLOW>(signed_decimal_overflow_types{},
                                                             make_inputs);
  batch.expect_results();
}

TEST_F(JITExpressionTest, NegOverflow)
{
  overflow_batch batch;
  auto make_inputs = []<typename T>() {
    using R = overflow_rep_t<T>;
    return unary_overflow_inputs<R>{
      {R{3},
       R{-20},
       R{1},
       R{-50},
       std::numeric_limits<R>::max(),
       R{-std::numeric_limits<R>::max()},
       R{0}},
      {R{3}, R{-20}, R{1}, R{-50}, std::numeric_limits<R>::min(), R{1}, R{0}},
      {R{-3},
       R{20},
       R{-1},
       R{50},
       R{-std::numeric_limits<R>::max()},
       std::numeric_limits<R>::max(),
       R{0}},
      {R{-3}, R{20}, R{-1}, R{50}, R{0}, R{-1}, R{0}},
      {1, 1, 1, 1, 0, 1, 1}};
  };
  batch.append_unary_types<cudf::ast::jit::op::NEG_OVERFLOW>(signed_decimal_overflow_types{},
                                                             make_inputs);
  batch.expect_results();
}

TEST_F(JITExpressionTest, CheckPrecision)
{
  overflow_batch batch;
  for_each_overflow_type(cudf::test::FixedPointTypes{},
                         [&]<typename T>() { batch.append_precision<T>(); });
  batch.expect_results();
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
