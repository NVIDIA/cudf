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
#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/filling.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/transform.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/cuda_stream.hpp>

#include <cuda/iterator>

#include <array>
#include <functional>
#include <initializer_list>
#include <limits>
#include <memory>
#include <string>
#include <string_view>
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
