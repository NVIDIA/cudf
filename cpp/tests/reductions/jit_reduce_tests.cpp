/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/iterator_utilities.hpp>

#include <cudf/ast/expressions.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/dictionary/encode.hpp>
#include <cudf/errc.hpp>
#include <cudf/reduction.hpp>
#include <cudf/table/table.hpp>

#include <cudf_reduce_test_fragments.hpp>

#include <algorithm>
#include <array>
#include <limits>
#include <numeric>
#include <sstream>

template <typename T>
using wrapper = cudf::test::fixed_width_column_wrapper<T>;

struct JitReduceTest : cudf::test::BaseFixture {};

namespace {
std::string source()
{
  auto range =
    cudf_reduce_test_fragments::file_ranges[cudf_reduce_test_fragments::sum_count_source];
  auto bytes = cudf_reduce_test_fragments::files.subspan(range[0], range[1]);
  return {reinterpret_cast<char const*>(bytes.data()), bytes.size()};
}

std::span<uint8_t const> binary(bool ir)
{
  auto id = ir ? cudf_reduce_test_fragments::sum_count_ir : cudf_reduce_test_fragments::sum_count;
  auto range = cudf_reduce_test_fragments::file_ranges[id];
  return cudf_reduce_test_fragments::files.subspan(range[0], range[1]);
}

auto const states =
  std::array{cudf::data_type{cudf::type_id::INT64}, cudf::data_type{cudf::type_id::INT32}};
auto const outputs = std::array{cudf::reduce_output{cudf::data_type{cudf::type_id::INT64}},
                                cudf::reduce_output{cudf::data_type{cudf::type_id::FLOAT64}}};

void expect_results(std::vector<std::unique_ptr<cudf::column>> const& result,
                    int64_t sum,
                    double mean,
                    bool valid_mean = true)
{
  ASSERT_EQ(result.size(), 2);
  wrapper<int64_t> expected_sum{sum};
  wrapper<double> expected_mean{{mean}, {valid_mean}};
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*result[0], expected_sum);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*result[1], expected_mean);
}
}  // namespace

TEST_F(JitReduceTest, CudaAndLto)
{
  wrapper<int32_t> input{1, 2, 3, 4};
  std::array<cudf::reduce_input, 1> inputs{input};
  expect_results(cudf::reduce(source(), inputs, states, outputs), 10, 2.5);
  for (bool ir : {false, true}) {
    expect_results(
      cudf::reduce_lto(binary(ir),
                       ir ? cudf::lto_binary_type::LTO_IR : cudf::lto_binary_type::FATBIN,
                       inputs,
                       states,
                       outputs),
      10,
      2.5);
  }
}

TEST_F(JitReduceTest, UnequalLengthsAndLargeFinalization)
{
  wrapper<int32_t> input{1, 2, 3, 4};
  std::array<cudf::reduce_input, 1> inputs{input};
  cudf::reduce_program program(source(), inputs, states, outputs);
  auto varied    = outputs;
  varied[0].size = 1031;
  varied[1].size = 35;
  auto result    = program.run(inputs, varied);
  auto seq =
    cudf::detail::make_counting_transform_iterator(0, [](auto i) { return int64_t{10} + i; });
  wrapper<int64_t> sums(seq, seq + 1031);
  std::vector<double> means(35, 2.5);
  wrapper<double> expected_means(means.begin(), means.end());
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*result[0], sums);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*result[1], expected_means);
  varied[0].size = 0;
  varied[1].size = 1;
  result         = program.run(inputs, varied);
  EXPECT_EQ(result[0]->size(), 0);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*result[1], wrapper<double>{2.5});
  varied[1].size = 0;
  result         = program.run(inputs, varied);
  EXPECT_EQ(result[0]->size(), 0);
  EXPECT_EQ(result[1]->size(), 0);
}

TEST_F(JitReduceTest, EmptyNullsAndSlices)
{
  wrapper<int32_t> input{{9, 1, 2, 3, 4, 9}, {true, true, false, true, true, true}};
  auto sliced = cudf::slice(input, {1, 5});
  std::array<cudf::reduce_input, 1> inputs{sliced[0]};
  cudf::reduce_program program(source(), inputs, states, outputs);
  expect_results(program.run(inputs), 8, 8.0 / 3);
  wrapper<int32_t> empty{};
  inputs[0] = empty;
  expect_results(program.run(inputs), 0, 0, false);
  wrapper<int32_t> nulls{{1, 2, 3}, {false, false, false}};
  inputs[0] = nulls;
  expect_results(program.run(inputs), 0, 0, false);
}

TEST_F(JitReduceTest, MultiBlockMergeAndReuse)
{
  for (int32_t n : {1, 31, 256, 1031, 1'000'003}) {
    std::vector<int32_t> data(n);
    std::iota(data.begin(), data.end(), 0);
    for (auto& x : data) {
      x %= 17;
    }
    wrapper<int32_t> input(data.begin(), data.end());
    std::array<cudf::reduce_input, 1> inputs{input};
    auto sum = std::accumulate(data.begin(), data.end(), int64_t{0});
    cudf::reduce_program program(source(), inputs, states, outputs);
    auto moved = std::move(program);
    expect_results(moved.run(inputs), sum, static_cast<double>(sum) / n);
    expect_results(moved.run(inputs), sum, static_cast<double>(sum) / n);
    EXPECT_THROW(program.run(inputs), std::invalid_argument);
  }
}

TEST_F(JitReduceTest, ExplicitSchemaAndValidation)
{
  std::array specs{cudf::reduce_input_spec{cudf::data_type{cudf::type_id::INT32}}};
  cudf::reduce_program program(source(), specs, states, outputs);
  wrapper<int32_t> input{1, 2};
  std::array<cudf::reduce_input, 1> inputs{input};
  expect_results(program.run(inputs), 3, 1.5);
  EXPECT_THROW(program.run(inputs, {}, 3), std::invalid_argument);
  EXPECT_THROW(program.run(inputs, {}, -1), std::invalid_argument);
  auto bad_outputs    = outputs;
  bad_outputs[0].size = -1;
  EXPECT_THROW(program.run(inputs, bad_outputs), std::invalid_argument);
  bad_outputs         = outputs;
  bad_outputs[0].type = cudf::data_type{cudf::type_id::INT32};
  EXPECT_THROW(program.run(inputs, bad_outputs), std::invalid_argument);
  wrapper<int64_t> wrong{1, 2};
  inputs[0] = wrong;
  EXPECT_THROW(program.run(inputs), std::invalid_argument);
  EXPECT_THROW(cudf::reduce_program(source(), specs, std::span<cudf::data_type const>{}, outputs),
               std::invalid_argument);
  EXPECT_THROW(
    cudf::reduce_program(source(), specs, states, std::span<cudf::reduce_output const>{}),
    std::invalid_argument);
}

TEST_F(JitReduceTest, BroadcastAndNullAware)
{
  auto code = source();
  auto pos  = code.find("cuda::std::int32_t value)");
  ASSERT_NE(pos, std::string::npos);
  code.replace(pos,
               std::string("cuda::std::int32_t value)").size(),
               "cuda::std::optional<cuda::std::int32_t> value)");
  pos = code.find("*sum += value;");
  code.replace(pos, std::string("*sum += value;").size(), "*sum += value.value_or(100);");
  wrapper<int32_t> scalar{5};
  std::array<cudf::reduce_input, 1> inputs{cudf::scalar_column_view{scalar}};
  cudf::reduce_program program(
    code, inputs, states, outputs, {.is_null_aware = cudf::null_aware::YES});
  EXPECT_THROW(program.run(inputs), std::invalid_argument);
  expect_results(program.run(inputs, {}, 4), 20, 5);
  wrapper<int32_t> null{{0}, {false}};
  inputs[0] = cudf::scalar_column_view{null};
  expect_results(program.run(inputs, {}, 4), 400, 100);
  expect_results(cudf::reduce(source(), inputs, states, outputs, {}, 4), 0, 0, false);
  for (bool ir : {false, true}) {
    auto id    = ir ? cudf_reduce_test_fragments::nullable_sum_count_ir
                    : cudf_reduce_test_fragments::nullable_sum_count;
    auto range = cudf_reduce_test_fragments::file_ranges[id];
    auto bytes = cudf_reduce_test_fragments::files.subspan(range[0], range[1]);
    cudf::reduce_program lto(bytes,
                             ir ? cudf::lto_binary_type::LTO_IR : cudf::lto_binary_type::FATBIN,
                             inputs,
                             states,
                             outputs,
                             {.is_null_aware = cudf::null_aware::YES});
    expect_results(lto.run(inputs, {}, 4), 400, 100);
    inputs[0] = cudf::scalar_column_view{scalar};
    expect_results(lto.run(inputs, {}, 4), 20, 5);
    inputs[0] = cudf::scalar_column_view{null};
  }
}

TEST_F(JitReduceTest, AstMappingAndLto)
{
  wrapper<int32_t> input{1, 2, 3, 4};
  cudf::table_view table{{input}};
  cudf::ast::column_reference x(0);
  cudf::ast::operation square(cudf::ast::ast_operator::MUL, x, x);
  std::array<std::reference_wrapper<cudf::ast::expression const>, 1> expressions{square};
  expect_results(cudf::reduce(source(), table, expressions, states, outputs), 30, 7.5);
  for (bool ir : {false, true}) {
    cudf::reduce_program program(binary(ir),
                                 ir ? cudf::lto_binary_type::LTO_IR : cudf::lto_binary_type::FATBIN,
                                 table,
                                 expressions,
                                 states,
                                 outputs);
    expect_results(program.run(table), 30, 7.5);
    wrapper<int32_t> other{2, 3};
    expect_results(program.run(cudf::table_view{{other}}), 13, 6.5);
  }
}

TEST_F(JitReduceTest, AstLiteralLifetimeAndNullPropagation)
{
  wrapper<int32_t> input{{1, 2, 3}, {true, false, true}};
  cudf::table_view table{{input}};
  std::unique_ptr<cudf::reduce_program> program;
  {
    cudf::numeric_scalar<int32_t> five(5);
    cudf::ast::literal literal(five);
    cudf::ast::column_reference x(0);
    cudf::ast::operation plus(cudf::ast::ast_operator::ADD, x, literal);
    std::array<std::reference_wrapper<cudf::ast::expression const>, 1> expressions{plus};
    program = std::make_unique<cudf::reduce_program>(source(), table, expressions, states, outputs);
  }
  expect_results(program->run(table), 14, 7);
  wrapper<int32_t> valid{1, 2, 3};
  expect_results(program->run(cudf::table_view{{valid}}), 21, 7);
}

TEST_F(JitReduceTest, AstCoalesceAndCheckedErrors)
{
  wrapper<int32_t> input{{1, 2, 3}, {true, false, true}};
  cudf::table_view table{{input}};
  cudf::numeric_scalar<int32_t> six(6);
  cudf::ast::literal literal(six);
  cudf::ast::column_reference x(0);
  cudf::ast::tree tree;
  auto& coalesce = cudf::ast::jit::operation(tree, cudf::ast::jit::op::COALESCE, {x, literal});
  std::array<std::reference_wrapper<cudf::ast::expression const>, 1> expressions{coalesce};
  auto run = [&](int engine, cudf::error_policy policy = cudf::error_policy::PROPAGATE) {
    cudf::reduce_options options{.errors = policy};
    if (engine == 0) {
      return cudf::reduce(source(), table, expressions, states, outputs, options);
    }
    return cudf::reduce_lto(
      binary(engine == 2),
      engine == 2 ? cudf::lto_binary_type::LTO_IR : cudf::lto_binary_type::FATBIN,
      table,
      expressions,
      states,
      outputs,
      options);
  };
  for (int engine : {0, 1, 2}) {
    expect_results(run(engine), 10, 10.0 / 3);
  }

  wrapper<int32_t> overflow{std::numeric_limits<int32_t>::max(), 1};
  table = cudf::table_view{{overflow}};
  cudf::numeric_scalar<int32_t> one(1);
  cudf::ast::literal one_literal(one);
  auto& checked =
    cudf::ast::jit::operation(tree, cudf::ast::jit::op::ADD_OVERFLOW, {x, one_literal});
  expressions[0] = checked;
  for (int engine : {0, 1, 2}) {
    EXPECT_THROW(run(engine), cudf::evaluation_error);
    auto nullified = run(engine, cudf::error_policy::NULLIFY);
    EXPECT_EQ(nullified[0]->null_count(), 1);
    EXPECT_EQ(nullified[1]->null_count(), 1);
  }
  auto& try_add = cudf::ast::jit::operation(
    tree, cudf::ast::jit::op::ADD_OVERFLOW, {x, one_literal}, cudf::error_policy::NULLIFY);
  expressions[0] = try_add;
  for (int engine : {0, 1, 2}) {
    expect_results(run(engine), 2, 2);
  }
}

TEST_F(JitReduceTest, MultipleExpressionsAndRepeatedColumn)
{
  wrapper<int32_t> input{1, 2, 3, 4};
  cudf::table_view table{{input}};
  cudf::ast::column_reference x(0);
  cudf::ast::operation square(cudf::ast::ast_operator::MUL, x, x);
  cudf::ast::operation cube(cudf::ast::ast_operator::MUL, square, x);
  std::array<std::reference_wrapper<cudf::ast::expression const>, 2> expressions{square, cube};
  auto code = source();
  auto pos  = code.find("cuda::std::int32_t value)");
  code.replace(pos,
               std::string("cuda::std::int32_t value)").size(),
               "cuda::std::int32_t value, cuda::std::int32_t other)");
  pos = code.find("*sum += value;");
  code.replace(pos, std::string("*sum += value;").size(), "*sum += value + other;");
  expect_results(cudf::reduce(code, table, expressions, states, outputs), 130, 32.5);
}

TEST_F(JitReduceTest, IndependentOutputValidity)
{
  wrapper<int32_t> input{1, 2, 3, 4};
  std::array<cudf::reduce_input, 1> inputs{input};
  auto code = source();
  auto pos  = code.find("mean_out != nullptr && count != 0");
  code.replace(pos,
               std::string("mean_out != nullptr && count != 0").size(),
               "mean_out != nullptr && count != 0 && index % 2 == 0");
  auto varied    = outputs;
  varied[0].size = 2;
  varied[1].size = 67;
  auto result    = cudf::reduce(code, inputs, states, varied);
  std::vector<double> values(67, 2.5);
  std::vector<bool> valid(67);
  for (size_t i = 0; i < valid.size(); ++i) {
    valid[i] = i % 2 == 0;
  }
  wrapper<double> expected(values.begin(), values.end(), valid.begin());
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*result[0], wrapper<int64_t>{10, 11});
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*result[1], expected);
  EXPECT_EQ(result[1]->null_count(), 33);
}

TEST_F(JitReduceTest, VoidCallbacksAndDurationState)
{
  std::string code = R"(
__device__ void cudf_reduce_init(cudf::duration_ms* s) { *s = cudf::duration_ms{0}; }
__device__ cudf::errc cudf_reduce_update(cudf::size_type, cudf::duration_ms* s, cudf::duration_ms x) {
  *s += x; return cudf::errc::SUCCESS;
}
__device__ void cudf_reduce_merge(cudf::duration_ms* s, cudf::duration_ms x) { *s += x; }
__device__ void cudf_reduce_finalize(cudf::size_type,
  cuda::std::span<cudf::size_type const>, cuda::std::optional<cudf::duration_ms>* out,
  cudf::duration_ms s) { *out = s; }
)";
  using T          = cudf::duration_ms;
  wrapper<T> input{T{1}, T{2}, T{3}};
  std::array<cudf::reduce_input, 1> inputs{input};
  std::array state{cudf::data_type{cudf::type_id::DURATION_MILLISECONDS}};
  std::array out{cudf::reduce_output{state[0]}};
  auto result = cudf::reduce(code, inputs, state, out);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*result[0], wrapper<T>{T{6}});
}

TEST_F(JitReduceTest, DecimalStateAndMetadata)
{
  std::string code = R"(
__device__ void cudf_reduce_init(numeric::decimal32* s) {
  *s = numeric::decimal32{0, numeric::scale_type{-2}};
}
__device__ void cudf_reduce_update(cudf::size_type, numeric::decimal32* s, numeric::decimal32 x) {
  *s = *s + x;
}
__device__ void cudf_reduce_merge(numeric::decimal32* s, numeric::decimal32 x) { *s = *s + x; }
__device__ void cudf_reduce_finalize(cudf::size_type,
  cuda::std::span<cudf::size_type const>, cuda::std::optional<numeric::decimal32>* out,
  numeric::decimal32 s) { *out = s; }
)";
  cudf::test::fixed_point_column_wrapper<int32_t> input{{125, 250}, numeric::scale_type{-2}};
  std::array<cudf::reduce_input, 1> inputs{input};
  std::array state{cudf::data_type{cudf::type_id::DECIMAL32, numeric::scale_type{-2}}};
  std::array out{cudf::reduce_output{state[0]}};
  cudf::reduce_program program(code, inputs, state, out);
  auto result = program.run(inputs);
  cudf::test::fixed_point_column_wrapper<int32_t> expected{{375}, numeric::scale_type{-2}};
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*result[0], expected);
  cudf::test::fixed_point_column_wrapper<int32_t> wrong_scale{{125, 250}, numeric::scale_type{-3}};
  inputs[0] = wrong_scale;
  EXPECT_THROW(program.run(inputs), std::invalid_argument);
}

TEST_F(JitReduceTest, MixedWidthStateAndDecimal128)
{
  // Sub-word fields, padding and multi-word values must survive both warp and block merging.
  std::string code = R"(
__device__ void cudf_reduce_init(int8_t* hi, int16_t* lo, bool* seen,
                               numeric::decimal128* sum) {
  *hi = -128; *lo = 32767; *seen = false;
  *sum = numeric::decimal128{0, numeric::scale_type{-2}};
}
__device__ void cudf_reduce_update(cudf::size_type, int8_t* hi, int16_t* lo, bool* seen,
                                 numeric::decimal128* sum, int32_t x) {
  *hi = cuda::std::max(*hi, int8_t(x));
  *lo = cuda::std::min(*lo, int16_t(x - 1000));
  *seen = *seen || x == 17;
  *sum = *sum + numeric::decimal128{numeric::scaled_integer<__int128_t>{
    (__int128_t{1} << 70) * x, numeric::scale_type{-2}}};
}
__device__ void cudf_reduce_merge(int8_t* hi, int16_t* lo, bool* seen,
                                numeric::decimal128* sum, int8_t h, int16_t l, bool s,
                                numeric::decimal128 v) {
  *hi = cuda::std::max(*hi, h); *lo = cuda::std::min(*lo, l);
  *seen = *seen || s; *sum = *sum + v;
}
__device__ void cudf_reduce_finalize(cudf::size_type,
  cuda::std::span<cudf::size_type const>, cuda::std::optional<int8_t>* hi,
  cuda::std::optional<int16_t>* lo, cuda::std::optional<bool>* seen,
  cuda::std::optional<numeric::decimal128>* sum, int8_t h, int16_t l, bool s,
  numeric::decimal128 v) { *hi = h; *lo = l; *seen = s; *sum = v; }
)";
  std::vector<int32_t> data(1031);
  std::generate(data.begin(), data.end(), [i = 0]() mutable { return i++ % 128; });
  wrapper<int32_t> input(data.begin(), data.end());
  std::array<cudf::reduce_input, 1> inputs{input};
  std::array state{cudf::data_type{cudf::type_id::INT8},
                   cudf::data_type{cudf::type_id::INT16},
                   cudf::data_type{cudf::type_id::BOOL8},
                   cudf::data_type{cudf::type_id::DECIMAL128, numeric::scale_type{-2}}};
  std::array out{cudf::reduce_output{state[0]},
                 cudf::reduce_output{state[1]},
                 cudf::reduce_output{state[2]},
                 cudf::reduce_output{state[3]}};
  auto result = cudf::reduce(code, inputs, state, out);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*result[0], wrapper<int8_t>{127});
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*result[1], wrapper<int16_t>{-1000});
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*result[2], wrapper<bool>{true});
  auto const sum = (__int128_t{1} << 70) * std::accumulate(data.begin(), data.end(), int64_t{0});
  cudf::test::fixed_point_column_wrapper<__int128_t> expected{{sum}, numeric::scale_type{-2}};
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*result[3], expected);
}

TEST_F(JitReduceTest, StringsAndDictionaryInputs)
{
  auto code = source();
  auto pos  = code.find("cuda::std::int32_t value)");
  code.replace(pos, std::string("cuda::std::int32_t value)").size(), "cudf::string_view value)");
  pos = code.find("*sum += value;");
  code.replace(pos, std::string("*sum += value;").size(), "*sum += value.size_bytes();");
  cudf::test::strings_column_wrapper strings{"a", "bcd", "", "ef"};
  std::array<cudf::reduce_input, 1> inputs{strings};
  std::array schema{cudf::reduce_input_spec{cudf::data_type{cudf::type_id::STRING}}};
  cudf::reduce_program strings_program(code, schema, states, outputs);
  expect_results(strings_program.run(inputs), 6, 1.5);
  cudf::test::strings_column_wrapper empty_strings{};
  inputs[0] = empty_strings;
  expect_results(strings_program.run(inputs), 0, 0, false);
  inputs[0] = strings;
  expect_results(strings_program.run(inputs), 6, 1.5);

  code = source();
  wrapper<int32_t> values{{4, 1, 4, 3}, {true, true, false, true}};
  auto dictionary = cudf::dictionary::encode(values);
  inputs[0]       = dictionary->view();
  expect_results(cudf::reduce(code, inputs, states, outputs), 8, 8.0 / 3);
  for (bool ir : {false, true}) {
    expect_results(
      cudf::reduce_lto(binary(ir),
                       ir ? cudf::lto_binary_type::LTO_IR : cudf::lto_binary_type::FATBIN,
                       inputs,
                       states,
                       outputs),
      8,
      8.0 / 3);
    auto id    = ir ? cudf_reduce_test_fragments::nullable_sum_count_ir
                    : cudf_reduce_test_fragments::nullable_sum_count;
    auto range = cudf_reduce_test_fragments::file_ranges[id];
    auto bytes = cudf_reduce_test_fragments::files.subspan(range[0], range[1]);
    expect_results(
      cudf::reduce_lto(bytes,
                       ir ? cudf::lto_binary_type::LTO_IR : cudf::lto_binary_type::FATBIN,
                       inputs,
                       states,
                       outputs,
                       {.is_null_aware = cudf::null_aware::YES}),
      108,
      27);
  }
}

TEST_F(JitReduceTest, NoInputsWithExplicitRows)
{
  auto code = source();
  auto pos  = code.find("cuda::std::int32_t value)");
  // The fixture's preceding comma and whitespace are removed along with the input parameter.
  auto comma = code.rfind(',', pos);
  code.erase(comma, pos + std::string("cuda::std::int32_t value").size() - comma);
  pos = code.find("cuda::std::int32_t,");
  code.replace(pos, std::string("cuda::std::int32_t,").size(), "cuda::std::int32_t row,");
  pos = code.find("*sum += value;");
  code.replace(pos, std::string("*sum += value;").size(), "*sum += row;");
  std::array<cudf::reduce_input, 0> inputs{};
  cudf::reduce_program program(code, inputs, states, outputs);
  EXPECT_THROW(program.run(inputs), std::invalid_argument);
  expect_results(program.run(inputs, {}, 5), 10, 2);
  expect_results(program.run(inputs, {}, 0), 0, 0, false);
}

TEST_F(JitReduceTest, WideStateGlobalScratchFallback)
{
  constexpr int fields = 200;
  std::ostringstream code;
  auto args = [&](bool values) {
    for (int i = 0; i < fields; ++i) {
      if (i != 0) { code << ","; }
      code << "int64_t" << (values ? " v" : "* s") << i;
    }
  };
  code << "__device__ void cudf_reduce_init(";
  args(false);
  code << ") {";
  for (int i = 0; i < fields; ++i) {
    code << "*s" << i << "=0;";
  }
  code << "}\n__device__ void cudf_reduce_update(cudf::size_type,";
  args(false);
  code << ",int32_t x){";
  for (int i = 0; i < fields; ++i) {
    code << "*s" << i << "+=x;";
  }
  code << "}\n__device__ void cudf_reduce_merge(";
  args(false);
  code << ",";
  args(true);
  code << "){";
  for (int i = 0; i < fields; ++i) {
    code << "*s" << i << "+=v" << i << ";";
  }
  code << "}\n__device__ void cudf_reduce_finalize(cudf::size_type,"
          "cuda::std::span<cudf::size_type const>,cuda::std::optional<int64_t>* out,";
  args(true);
  code << "){*out=v0+v199;}\n";
  std::vector<cudf::data_type> state(fields, cudf::data_type{cudf::type_id::INT64});
  std::array out{cudf::reduce_output{state[0]}};
  std::vector<int32_t> data(257, 1);
  wrapper<int32_t> input(data.begin(), data.end());
  std::array<cudf::reduce_input, 1> inputs{input};
  auto result = cudf::reduce(code.str(), inputs, state, out);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*result[0], wrapper<int64_t>{514});
}

TEST_F(JitReduceTest, LifecycleErrorsAndRecovery)
{
  auto source_range =
    cudf_reduce_test_fragments::file_ranges[cudf_reduce_test_fragments::errors_source];
  auto source_bytes = cudf_reduce_test_fragments::files.subspan(source_range[0], source_range[1]);
  std::string code(reinterpret_cast<char const*>(source_bytes.data()), source_bytes.size());
  std::vector<int32_t> data(10003, 1);
  wrapper<int32_t> input(data.begin(), data.end());
  std::array<cudf::reduce_input, 1> inputs{input};
  std::array state{cudf::data_type{cudf::type_id::INT64}};
  std::array out{cudf::reduce_output{state[0], 35}};
  auto stream = cudf::test::get_default_stream();
  auto flag   = cudf::detail::make_zeroed_device_uvector_async<int>(
    1, stream, cudf::get_current_device_resource_ref());
  for (int engine : {0, 1, 2}) {
    for (auto policy : {cudf::error_policy::PROPAGATE, cudf::error_policy::NULLIFY}) {
      auto program = [&] {
        cudf::reduce_options options{.errors = policy, .user_data = flag.data()};
        if (engine == 0) { return cudf::reduce_program(code, inputs, state, out, options); }
        auto id =
          engine == 1 ? cudf_reduce_test_fragments::errors : cudf_reduce_test_fragments::errors_ir;
        auto range = cudf_reduce_test_fragments::file_ranges[id];
        auto bytes = cudf_reduce_test_fragments::files.subspan(range[0], range[1]);
        return cudf::reduce_program(
          bytes,
          engine == 1 ? cudf::lto_binary_type::FATBIN : cudf::lto_binary_type::LTO_IR,
          inputs,
          state,
          out,
          options);
      }();
      for (int phase : {1, 2, 3, 4}) {
        CUDF_CUDA_TRY(cudaMemcpyAsync(
          flag.data(), &phase, sizeof(phase), cudaMemcpyHostToDevice, stream.get()));
        if (policy == cudf::error_policy::PROPAGATE) {
          try {
            program.run(inputs, {}, std::nullopt, stream);
            FAIL() << "Expected lifecycle error";
          } catch (cudf::evaluation_error const& e) {
            EXPECT_EQ(e.error_code(),
                      phase == 1 || phase == 3 ? cudf::errc::ARITHMETIC_OVERFLOW
                                               : cudf::errc::DIVISION_BY_ZERO);
          }
        } else {
          auto result = program.run(inputs, {}, std::nullopt, stream);
          EXPECT_EQ(result[0]->size(), 35);
          EXPECT_EQ(result[0]->null_count(), 35);
        }
      }
      CUDF_CUDA_TRY(cudaMemsetAsync(flag.data(), 0, sizeof(int), stream.get()));
      auto result = program.run(inputs, {}, std::nullopt, stream);
      EXPECT_EQ(result[0]->null_count(), 0);
      EXPECT_EQ(cudf::test::to_host<int64_t>(*result[0]).first.front(), 10003);
    }
  }
}

TEST_F(JitReduceTest, InvalidSourceAndBinary)
{
  wrapper<int32_t> input{1};
  std::array<cudf::reduce_input, 1> inputs{input};
  EXPECT_THROW(cudf::reduce_program("this is not CUDA", inputs, states, outputs), std::exception);
  EXPECT_THROW(
    cudf::reduce_program(
      std::span<uint8_t const>{}, cudf::lto_binary_type::LTO_IR, inputs, states, outputs),
    std::invalid_argument);
  EXPECT_THROW(cudf::reduce_program(
                 binary(false), static_cast<cudf::lto_binary_type>(99), inputs, states, outputs),
               std::invalid_argument);
}
