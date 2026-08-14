/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "io/json/nested_json.hpp"
#include "io/utilities/hostdevice_vector.hpp"

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/cudf_gtest.hpp>
#include <cudf_test/random.hpp>
#include <cudf_test/testing_main.hpp>

#include <cudf/io/json.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/span.hpp>

#include <cuda/iterator>
#include <cuda/std/tuple>

#include <algorithm>
#include <limits>
#include <string>

namespace cuio_json = cudf::io::json;

namespace {
// Forward declaration
void print_column(std::string const& input,
                  cuio_json::json_column const& column,
                  uint32_t indent = 0);

/**
 * @brief Helper to generate indentation
 */
std::string pad(uint32_t indent = 0)
{
  std::string pad{};
  if (indent > 0) pad.insert(pad.begin(), indent, ' ');
  return pad;
}

/**
 * @brief Prints a string column.
 */
void print_json_string_col(std::string const& input,
                           cuio_json::json_column const& column,
                           uint32_t indent = 0)
{
  for (std::size_t i = 0; i < column.string_offsets.size(); i++) {
    std::cout << pad(indent) << i << ": [" << (column.validity[i] ? "1" : "0") << "] '"
              << input.substr(column.string_offsets[i], column.string_lengths[i]) << "'\n";
  }
}

/**
 * @brief Prints a list column.
 */
void print_json_list_col(std::string const& input,
                         cuio_json::json_column const& column,
                         uint32_t indent = 0)
{
  std::cout << pad(indent) << " [LIST]\n";
  std::cout << pad(indent) << " -> num. child-columns: " << column.child_columns.size() << "\n";
  std::cout << pad(indent) << " -> num. rows: " << column.current_offset << "\n";
  std::cout << pad(indent) << " -> num. valid: " << column.valid_count << "\n";
  std::cout << pad(indent) << " offsets[]: "
            << "\n";
  for (std::size_t i = 0; i < column.child_offsets.size() - 1; i++) {
    std::cout << pad(indent + 2) << i << ": [" << (column.validity[i] ? "1" : "0") << "] ["
              << column.child_offsets[i] << ", " << column.child_offsets[i + 1] << ")\n";
  }
  if (column.child_columns.size() > 0) {
    std::cout << pad(indent) << column.child_columns.begin()->first << "[]: "
              << "\n";
    print_column(input, column.child_columns.begin()->second, indent + 2);
  }
}

/**
 * @brief Prints a struct column.
 */
void print_json_struct_col(std::string const& input,
                           cuio_json::json_column const& column,
                           uint32_t indent = 0)
{
  std::cout << pad(indent) << " [STRUCT]\n";
  std::cout << pad(indent) << " -> num. child-columns: " << column.child_columns.size() << "\n";
  std::cout << pad(indent) << " -> num. rows: " << column.current_offset << "\n";
  std::cout << pad(indent) << " -> num. valid: " << column.valid_count << "\n";
  std::cout << pad(indent) << " -> validity[]: "
            << "\n";
  for (decltype(column.current_offset) i = 0; i < column.current_offset; i++) {
    std::cout << pad(indent + 2) << i << ": [" << (column.validity[i] ? "1" : "0") << "]\n";
  }
  auto it = std::begin(column.child_columns);
  for (std::size_t i = 0; i < column.child_columns.size(); i++) {
    std::cout << pad(indent + 2) << "child #" << i << " '" << it->first << "'[] \n";
    print_column(input, it->second, indent + 2);
    it++;
  }
}

/**
 * @brief Prints the column's data and recurses through and prints all the child columns.
 */
void print_column(std::string const& input, cuio_json::json_column const& column, uint32_t indent)
{
  switch (column.type) {
    case cuio_json::json_col_t::StringColumn: print_json_string_col(input, column, indent); break;
    case cuio_json::json_col_t::ListColumn: print_json_list_col(input, column, indent); break;
    case cuio_json::json_col_t::StructColumn: print_json_struct_col(input, column, indent); break;
    case cuio_json::json_col_t::Unknown: std::cout << pad(indent) << "[UNKNOWN]\n"; break;
    default: break;
  }
}
}  // namespace

// Base test fixture for tests
struct JsonTest : public cudf::test::BaseFixture {};

TEST_F(JsonTest, StackContext)
{
  // Type used to represent the atomic symbol type used within the finite-state machine
  using SymbolT      = char;
  using StackSymbolT = char;

  // Prepare cuda stream for data transfers & kernels
  auto const stream = cudf::get_default_stream();

  // Test input
  char const delimiter    = 'h';
  std::string const input = R"(  [{)"
                            R"("category": "reference",)"
                            R"("index:": [4,12,42],)"
                            R"("author": "Nigel Rees",)"
                            R"("title": "[Sayings of the Century]",)"
                            R"("price": 8.95)"
                            R"(},  )"
                            R"({)"
                            R"("category": "reference",)"
                            R"("index": [4,{},null,{"a":[{ }, {}] } ],)"
                            R"("author": "Nigel Rees",)"
                            R"("title": "{}\\\"[], <=semantic-symbols-string\\\\",)"
                            R"("price": 8.95)"
                            R"(}] )";

  // Prepare input & output buffers
  cudf::string_scalar const d_scalar(input, true, stream);
  auto const d_input =
    cudf::device_span<SymbolT const>{d_scalar.data(), static_cast<size_t>(d_scalar.size())};
  cudf::detail::hostdevice_vector<StackSymbolT> stack_context(input.size(), stream);

  // Run algorithm
  constexpr auto stack_behavior = cuio_json::stack_behavior_t::PushPopWithoutReset;
  cuio_json::detail::get_stack_context(
    d_input, stack_context.device_ptr(), stack_behavior, delimiter, stream);

  // Copy back the results
  stack_context.device_to_host_async(stream);

  // Make sure we copied back the stack context
  stream.synchronize();

  std::vector<char> const golden_stack_context{
    '_', '_', '_', '[', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{',
    '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{',
    '{', '[', '[', '[', '[', '[', '[', '[', '[', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{',
    '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{',
    '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{',
    '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{',
    '{', '{', '{', '{', '{', '{', '{', '[', '[', '[', '[', '{', '{', '{', '{', '{', '{', '{', '{',
    '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{',
    '{', '{', '{', '{', '{', '{', '{', '[', '[', '[', '{', '[', '[', '[', '[', '[', '[', '[', '{',
    '{', '{', '{', '{', '[', '{', '{', '[', '[', '[', '{', '[', '{', '{', '[', '[', '{', '{', '{',
    '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{',
    '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{',
    '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{',
    '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{',
    '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '[', '_'};

  ASSERT_EQ(golden_stack_context.size(), stack_context.size());
  CUDF_TEST_EXPECT_VECTOR_EQUAL(golden_stack_context, stack_context, stack_context.size());
}

TEST_F(JsonTest, StackContextUtf8)
{
  // Type used to represent the atomic symbol type used within the finite-state machine
  using SymbolT      = char;
  using StackSymbolT = char;

  // Prepare cuda stream for data transfers & kernels
  auto const stream = cudf::get_default_stream();

  // Test input
  char const delimiter    = 'h';
  std::string const input = R"([{"a":{"year":1882,"author": "Bharathi"}, {"a":"filip ʒakotɛ"}}])";

  // Prepare input & output buffers
  cudf::string_scalar const d_scalar(input, true, stream);
  auto const d_input =
    cudf::device_span<SymbolT const>{d_scalar.data(), static_cast<size_t>(d_scalar.size())};
  cudf::detail::hostdevice_vector<StackSymbolT> stack_context(input.size(), stream);

  // Run algorithm
  constexpr auto stack_behavior = cuio_json::stack_behavior_t::PushPopWithoutReset;
  cuio_json::detail::get_stack_context(
    d_input, stack_context.device_ptr(), stack_behavior, delimiter, stream);

  // Copy back the results
  stack_context.device_to_host_async(stream);

  // Make sure we copied back the stack context
  stream.synchronize();

  std::vector<char> const golden_stack_context{
    '_', '[', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{',
    '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{',
    '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{',
    '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '{', '['};

  ASSERT_EQ(golden_stack_context.size(), stack_context.size());
  CUDF_TEST_EXPECT_VECTOR_EQUAL(golden_stack_context, stack_context, stack_context.size());
}

/**
 * @brief Test fixture for parametrized JSON reader tests
 */
struct JsonDelimiterParamTest : public cudf::test::BaseFixture,
                                public testing::WithParamInterface<char> {};

// Parametrize qualifying JSON tests for multiple delimiters
INSTANTIATE_TEST_SUITE_P(JsonDelimiterParamTest,
                         JsonDelimiterParamTest,
                         ::testing::Values('\n', '\b', '\v', '\f', 'h'));

TEST_P(JsonDelimiterParamTest, StackContextRecovering)
{
  // Type used to represent the atomic symbol type used within the finite-state machine
  using SymbolT      = char;
  using StackSymbolT = char;

  // Prepare cuda stream for data transfers & kernels
  auto const stream = cudf::get_default_stream();

  // JSON lines input that recovers on invalid lines
  char const delimiter = GetParam();
  std::string input    = R"({"a":-2},
  {"a":
  {"a":{"a":[321
  {"a":[1]}

  {"b":123}
  )";
  std::replace(input.begin(), input.end(), '\n', delimiter);

  // Expected stack context (including stack context of the newline characters)
  std::string const golden_stack_context =
    "_{{{{{{{__"
    "___{{{{{"
    "___{{{{{{{{{{[[[["
    "___{{{{{[[{_"
    "_"
    "___{{{{{{{{_"
    "__";

  // Prepare input & output buffers
  cudf::string_scalar const d_scalar(input, true, stream);
  auto const d_input =
    cudf::device_span<SymbolT const>{d_scalar.data(), static_cast<size_t>(d_scalar.size())};
  cudf::detail::hostdevice_vector<StackSymbolT> stack_context(input.size(), stream);

  // Run algorithm
  constexpr auto stack_behavior = cuio_json::stack_behavior_t::ResetOnDelimiter;
  cuio_json::detail::get_stack_context(
    d_input, stack_context.device_ptr(), stack_behavior, delimiter, stream);

  // Copy back the results
  stack_context.device_to_host_async(stream);

  // Make sure we copied back the stack context
  stream.synchronize();

  // Verify results
  ASSERT_EQ(golden_stack_context.size(), stack_context.size());
  CUDF_TEST_EXPECT_VECTOR_EQUAL(golden_stack_context, stack_context, stack_context.size());
}

TEST_P(JsonDelimiterParamTest, StackContextRecoveringFuzz)
{
  // Type used to represent the atomic symbol type used within the finite-state machine
  using SymbolT      = char;
  using StackSymbolT = char;

  char const delimiter = GetParam();
  std::mt19937 gen(42);
  std::uniform_int_distribution<int> distribution(0, 4);

  constexpr std::size_t input_length = 1024 * 1024;
  std::string input{};
  input.reserve(input_length);

  bool inside_quotes = false;
  std::stack<StackSymbolT> host_stack{};
  for (std::size_t i = 0; i < input_length; ++i) {
    bool is_ok = true;
    char current{};
    do {
      int rand_char = distribution(gen);
      is_ok         = true;
      switch (rand_char) {
        case 0: current = '{'; break;
        case 1: current = '['; break;
        case 2: current = '}'; break;
        case 3: current = '"'; break;
        case 4: current = delimiter; break;
      }
      if (current == '"')
        inside_quotes = !inside_quotes;
      else if (current == '{' && !inside_quotes)
        host_stack.push('{');
      else if (current == '[' && !inside_quotes)
        host_stack.push('[');
      else if (current == '}' && !inside_quotes) {
        if (host_stack.size() > 0) {
          // Get the proper 'pop' stack symbol
          current = (host_stack.top() == '{' ? '}' : ']');
          host_stack.pop();
        } else
          is_ok = false;
      } else if (current == delimiter) {
        // Increase chance to have longer lines
        if (distribution(gen) == 0) {
          is_ok = false;
        } else {
          host_stack    = {};
          inside_quotes = false;
        }
      }
    } while (!is_ok);
    input += current;
  }

  std::string expected_stack_context{};
  expected_stack_context.reserve(input_length);
  inside_quotes = false;
  host_stack    = std::stack<StackSymbolT>{};
  for (auto const current : input) {
    // Write the stack context for the current input symbol
    if (host_stack.empty()) {
      expected_stack_context += '_';
    } else {
      expected_stack_context += host_stack.top();
    }

    if (current == '"')
      inside_quotes = !inside_quotes;
    else if (current == '{' && !inside_quotes)
      host_stack.push('{');
    else if (current == '[' && !inside_quotes)
      host_stack.push('[');
    else if (current == '}' && !inside_quotes && host_stack.size() > 0)
      host_stack.pop();
    else if (current == ']' && !inside_quotes && host_stack.size() > 0)
      host_stack.pop();
    else if (current == delimiter) {
      host_stack    = {};
      inside_quotes = false;
    }
  }

  // Prepare cuda stream for data transfers & kernels
  auto const stream = cudf::get_default_stream();

  // Prepare input & output buffers
  cudf::string_scalar const d_scalar(input, true, stream);
  auto const d_input =
    cudf::device_span<SymbolT const>{d_scalar.data(), static_cast<size_t>(d_scalar.size())};
  cudf::detail::hostdevice_vector<StackSymbolT> stack_context(input.size(), stream);

  // Run algorithm
  constexpr auto stack_behavior = cuio_json::stack_behavior_t::ResetOnDelimiter;
  cuio_json::detail::get_stack_context(
    d_input, stack_context.device_ptr(), stack_behavior, delimiter, stream);

  // Copy back the results
  stack_context.device_to_host_async(stream);

  // Make sure we copied back the stack context
  stream.synchronize();

  ASSERT_EQ(expected_stack_context.size(), stack_context.size());
  CUDF_TEST_EXPECT_VECTOR_EQUAL(expected_stack_context, stack_context, stack_context.size());
}

struct JsonNewlineDelimiterTest : public cudf::test::BaseFixture {};

TEST_F(JsonNewlineDelimiterTest, TokenStream)
{
  using cuio_json::PdaTokenT;
  using cuio_json::SymbolOffsetT;
  using cuio_json::SymbolT;
  // Test input
  std::string const input = R"(  [{)"
                            R"("category": "reference",)"
                            R"("index:": [4,12,42],)"
                            R"("author": "Nigel Rees",)"
                            R"("title": "[Sayings of the Century]",)"
                            R"("price": 8.95)"
                            R"(},  )"
                            R"({)"
                            R"("category": "reference",)"
                            R"("index": [4,{},null,{"a":[{ }, {}] } ],)"
                            R"("author": "Nigel Rees",)"
                            R"("title": "{}[], <=semantic-symbols-string",)"
                            R"("price": 8.95)"
                            R"(}] )";

  auto const stream = cudf::get_default_stream();

  // Default parsing options
  cudf::io::json_reader_options default_options{};

  // Prepare input & output buffers
  cudf::string_scalar const d_scalar(input, true, stream);
  auto const d_input =
    cudf::device_span<SymbolT const>{d_scalar.data(), static_cast<size_t>(d_scalar.size())};

  // Parse the JSON and get the token stream
  auto [d_tokens_gpu, d_token_indices_gpu] = cuio_json::detail::get_token_stream(
    d_input, default_options, stream, cudf::get_current_device_resource_ref());
  // Copy back the number of tokens that were written
  auto const tokens_gpu        = cudf::detail::make_std_vector_async(d_tokens_gpu, stream);
  auto const token_indices_gpu = cudf::detail::make_std_vector_async(d_token_indices_gpu, stream);
  stream.synchronize();

  // Golden token stream sample
  using token_t = cuio_json::token_t;
  std::vector<std::pair<std::size_t, cuio_json::PdaTokenT>> const golden_token_stream = {
    {2, token_t::ListBegin},
    {3, token_t::StructBegin},
    {4, token_t::StructMemberBegin},
    {4, token_t::FieldNameBegin},
    {13, token_t::FieldNameEnd},
    {16, token_t::StringBegin},
    {26, token_t::StringEnd},
    {27, token_t::StructMemberEnd},
    {28, token_t::StructMemberBegin},
    {28, token_t::FieldNameBegin},
    {35, token_t::FieldNameEnd},
    {38, token_t::ListBegin},
    {39, token_t::ValueBegin},
    {40, token_t::ValueEnd},
    {41, token_t::ValueBegin},
    {43, token_t::ValueEnd},
    {44, token_t::ValueBegin},
    {46, token_t::ValueEnd},
    {46, token_t::ListEnd},
    {47, token_t::StructMemberEnd},
    {48, token_t::StructMemberBegin},
    {48, token_t::FieldNameBegin},
    {55, token_t::FieldNameEnd},
    {58, token_t::StringBegin},
    {69, token_t::StringEnd},
    {70, token_t::StructMemberEnd},
    {71, token_t::StructMemberBegin},
    {71, token_t::FieldNameBegin},
    {77, token_t::FieldNameEnd},
    {80, token_t::StringBegin},
    {105, token_t::StringEnd},
    {106, token_t::StructMemberEnd},
    {107, token_t::StructMemberBegin},
    {107, token_t::FieldNameBegin},
    {113, token_t::FieldNameEnd},
    {116, token_t::ValueBegin},
    {120, token_t::ValueEnd},
    {120, token_t::StructMemberEnd},
    {120, token_t::StructEnd},
    {124, token_t::StructBegin},
    {125, token_t::StructMemberBegin},
    {125, token_t::FieldNameBegin},
    {134, token_t::FieldNameEnd},
    {137, token_t::StringBegin},
    {147, token_t::StringEnd},
    {148, token_t::StructMemberEnd},
    {149, token_t::StructMemberBegin},
    {149, token_t::FieldNameBegin},
    {155, token_t::FieldNameEnd},
    {158, token_t::ListBegin},
    {159, token_t::ValueBegin},
    {160, token_t::ValueEnd},
    {161, token_t::StructBegin},
    {162, token_t::StructEnd},
    {164, token_t::ValueBegin},
    {168, token_t::ValueEnd},
    {169, token_t::StructBegin},
    {170, token_t::StructMemberBegin},
    {170, token_t::FieldNameBegin},
    {172, token_t::FieldNameEnd},
    {174, token_t::ListBegin},
    {175, token_t::StructBegin},
    {177, token_t::StructEnd},
    {180, token_t::StructBegin},
    {181, token_t::StructEnd},
    {182, token_t::ListEnd},
    {184, token_t::StructMemberEnd},
    {184, token_t::StructEnd},
    {186, token_t::ListEnd},
    {187, token_t::StructMemberEnd},
    {188, token_t::StructMemberBegin},
    {188, token_t::FieldNameBegin},
    {195, token_t::FieldNameEnd},
    {198, token_t::StringBegin},
    {209, token_t::StringEnd},
    {210, token_t::StructMemberEnd},
    {211, token_t::StructMemberBegin},
    {211, token_t::FieldNameBegin},
    {217, token_t::FieldNameEnd},
    {220, token_t::StringBegin},
    {252, token_t::StringEnd},
    {253, token_t::StructMemberEnd},
    {254, token_t::StructMemberBegin},
    {254, token_t::FieldNameBegin},
    {260, token_t::FieldNameEnd},
    {263, token_t::ValueBegin},
    {267, token_t::ValueEnd},
    {267, token_t::StructMemberEnd},
    {267, token_t::StructEnd},
    {268, token_t::ListEnd}};

  // Verify the number of tokens matches
  ASSERT_EQ(golden_token_stream.size(), tokens_gpu.size());
  ASSERT_EQ(golden_token_stream.size(), token_indices_gpu.size());

  for (std::size_t i = 0; i < tokens_gpu.size(); i++) {
    // Ensure the index the tokens are pointing to do match
    EXPECT_EQ(golden_token_stream[i].first, token_indices_gpu[i]) << "Mismatch at #" << i;

    // Ensure the token category is correct
    EXPECT_EQ(golden_token_stream[i].second, tokens_gpu[i]) << "Mismatch at #" << i;
  }
}

TEST_F(JsonNewlineDelimiterTest, TokenStream2)
{
  using cuio_json::PdaTokenT;
  using cuio_json::SymbolOffsetT;
  using cuio_json::SymbolT;
  // value end with comma, space, close-brace ", }"
  std::string const input =
    R"([ {}, { "a": { "y" : 6, "z": [] }}, { "a" : { "x" : 8, "y": 9}, "b" : {"x": 10 , "z": 11)"
    "\n}}]";

  auto const stream = cudf::get_default_stream();

  // Default parsing options
  cudf::io::json_reader_options default_options{};

  // Prepare input & output buffers
  cudf::string_scalar const d_scalar(input, true, stream);
  auto const d_input =
    cudf::device_span<SymbolT const>{d_scalar.data(), static_cast<size_t>(d_scalar.size())};

  // Parse the JSON and get the token stream
  auto [d_tokens_gpu, d_token_indices_gpu] = cuio_json::detail::get_token_stream(
    d_input, default_options, stream, cudf::get_current_device_resource_ref());
  // Copy back the number of tokens that were written
  auto const tokens_gpu        = cudf::detail::make_std_vector_async(d_tokens_gpu, stream);
  auto const token_indices_gpu = cudf::detail::make_std_vector_async(d_token_indices_gpu, stream);
  stream.synchronize();

  // Golden token stream sample
  using token_t = cuio_json::token_t;
  // clang-format off
  std::vector<std::pair<std::size_t, cuio_json::PdaTokenT>> const golden_token_stream = {
    {0, token_t::ListBegin},
    {2, token_t::StructBegin}, {3, token_t::StructEnd}, //{}
    {6, token_t::StructBegin},
        {8, token_t::StructMemberBegin}, {8, token_t::FieldNameBegin}, {10, token_t::FieldNameEnd}, //a
            {13, token_t::StructBegin},
                {15, token_t::StructMemberBegin}, {15, token_t::FieldNameBegin}, {17, token_t::FieldNameEnd}, {21, token_t::ValueBegin}, {22, token_t::ValueEnd}, {22, token_t::StructMemberEnd}, //a.y
                {24, token_t::StructMemberBegin}, {24, token_t::FieldNameBegin},  {26, token_t::FieldNameEnd},  {29, token_t::ListBegin}, {30, token_t::ListEnd}, {32, token_t::StructMemberEnd}, //a.z
            {32, token_t::StructEnd},
        {33, token_t::StructMemberEnd},
    {33, token_t::StructEnd},
    {36, token_t::StructBegin},
        {38, token_t::StructMemberBegin}, {38, token_t::FieldNameBegin}, {40, token_t::FieldNameEnd}, //a
            {44, token_t::StructBegin},
                {46, token_t::StructMemberBegin}, {46, token_t::FieldNameBegin}, {48, token_t::FieldNameEnd}, {52, token_t::ValueBegin}, {53, token_t::ValueEnd}, {53, token_t::StructMemberEnd}, //a.x
                {55, token_t::StructMemberBegin}, {55, token_t::FieldNameBegin}, {57, token_t::FieldNameEnd}, {60, token_t::ValueBegin}, {61, token_t::ValueEnd}, {61, token_t::StructMemberEnd}, //a.y
            {61, token_t::StructEnd},
        {62, token_t::StructMemberEnd},
        {64, token_t::StructMemberBegin}, {64, token_t::FieldNameBegin}, {66, token_t::FieldNameEnd}, //b
            {70, token_t::StructBegin},
                {71, token_t::StructMemberBegin}, {71, token_t::FieldNameBegin}, {73, token_t::FieldNameEnd}, {76, token_t::ValueBegin}, {78, token_t::ValueEnd}, {79, token_t::StructMemberEnd}, //b.x
                {81, token_t::StructMemberBegin}, {81, token_t::FieldNameBegin}, {83, token_t::FieldNameEnd}, {86, token_t::ValueBegin}, {88, token_t::ValueEnd}, {89, token_t::StructMemberEnd}, //b.z
            {89, token_t::StructEnd},
        {90, token_t::StructMemberEnd},
    {90, token_t::StructEnd},
    {91, token_t::ListEnd}};
  // clang-format on

  // Verify the number of tokens matches
  ASSERT_EQ(golden_token_stream.size(), tokens_gpu.size());
  ASSERT_EQ(golden_token_stream.size(), token_indices_gpu.size());

  for (std::size_t i = 0; i < tokens_gpu.size(); i++) {
    // Ensure the index the tokens are pointing to do match
    EXPECT_EQ(golden_token_stream[i].first, token_indices_gpu[i]) << "Mismatch at #" << i;

    // Ensure the token category is correct
    EXPECT_EQ(golden_token_stream[i].second, tokens_gpu[i]) << "Mismatch at #" << i;
  }
}

struct JsonParserTest : public cudf::test::BaseFixture {};

TEST_F(JsonParserTest, ExtractColumn)
{
  using cuio_json::SymbolT;
  auto json_parser = cuio_json::detail::device_parse_nested_json;

  // Prepare cuda stream for data transfers & kernels
  auto const stream = cudf::get_default_stream();
  auto mr           = cudf::get_current_device_resource_ref();

  // Default parsing options
  cudf::io::json_reader_options default_options{};

  std::string const input = R"( [{"a":0.0, "b":1.0}, {"a":0.1, "b":1.1}, {"a":0.2, "b":1.2}] )";
  auto const d_input      = cudf::detail::make_device_uvector_async(
    cudf::host_span<char const>{input.c_str(), input.size()},
    stream,
    cudf::get_current_device_resource_ref());
  // Get the JSON's tree representation
  auto const cudf_table = json_parser(d_input, default_options, stream, mr);

  auto const expected_col_count = 2;
  EXPECT_EQ(cudf_table.tbl->num_columns(), expected_col_count);

  auto expected_col1            = cudf::test::fixed_width_column_wrapper<double>({0.0, 0.1, 0.2});
  auto expected_col2            = cudf::test::fixed_width_column_wrapper<double>({1.0, 1.1, 1.2});
  cudf::column_view parsed_col1 = cudf_table.tbl->get_column(0);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_col1, parsed_col1);
  cudf::column_view parsed_col2 = cudf_table.tbl->get_column(1);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_col2, parsed_col2);
}

TEST_P(JsonDelimiterParamTest, RecoveringTokenStream)
{
  // Test input. Inline comments used to indicate character indexes
  //                           012345678 <= line 0
  char const delimiter = GetParam();

  std::string input = R"({"a":2 {})"
                      // 9
                      "\n"
                      // 01234 <= line 1
                      R"({"a":)"
                      // 5
                      "\n"
                      // 67890123456789 <= line 2
                      R"({"a":{"a":[321)"
                      // 0
                      "\n"
                      // 123456789 <= line 3
                      R"({"a":[1]})"
                      // 0
                      "\n"
                      // 1  <= line 4
                      "\n"
                      // 23456789 <= line 5
                      R"({"b":123})";
  std::replace(input.begin(), input.end(), '\n', delimiter);

  // Golden token stream sample
  using token_t = cuio_json::token_t;
  std::vector<std::pair<std::size_t, cuio_json::PdaTokenT>> const golden_token_stream = {
    // Line 0 (invalid)
    {0, token_t::StructBegin},
    {0, token_t::StructEnd},
    // Line 1 (invalid)
    {0, token_t::StructBegin},
    {0, token_t::StructEnd},
    // Line 2 (invalid)
    {0, token_t::StructBegin},
    {0, token_t::StructEnd},
    // Line 3 (valid)
    {31, token_t::StructBegin},
    {32, token_t::StructMemberBegin},
    {32, token_t::FieldNameBegin},
    {34, token_t::FieldNameEnd},
    {36, token_t::ListBegin},
    {37, token_t::ValueBegin},
    {38, token_t::ValueEnd},
    {38, token_t::ListEnd},
    {39, token_t::StructMemberEnd},
    {39, token_t::StructEnd},
    // Line 4 (empty)
    // Line 5 (valid)
    {42, token_t::StructBegin},
    {43, token_t::StructMemberBegin},
    {43, token_t::FieldNameBegin},
    {45, token_t::FieldNameEnd},
    {47, token_t::ValueBegin},
    {50, token_t::ValueEnd},
    {50, token_t::StructMemberEnd},
    {50, token_t::StructEnd}};

  auto const stream = cudf::get_default_stream();

  // Default parsing options
  cudf::io::json_reader_options default_options{};
  default_options.set_recovery_mode(cudf::io::json_recovery_mode_t::RECOVER_WITH_NULL);
  default_options.enable_lines(true);
  default_options.set_delimiter(delimiter);

  // Prepare input & output buffers
  cudf::string_scalar const d_scalar(input, true, stream);
  auto const d_input = cudf::device_span<cuio_json::SymbolT const>{
    d_scalar.data(), static_cast<size_t>(d_scalar.size())};

  // Parse the JSON and get the token stream
  auto [d_tokens_gpu, d_token_indices_gpu] = cuio_json::detail::get_token_stream(
    d_input, default_options, stream, cudf::get_current_device_resource_ref());
  // Copy back the number of tokens that were written
  auto const tokens_gpu        = cudf::detail::make_std_vector_async(d_tokens_gpu, stream);
  auto const token_indices_gpu = cudf::detail::make_std_vector_async(d_token_indices_gpu, stream);

  stream.synchronize();
  // Verify the number of tokens matches
  ASSERT_EQ(golden_token_stream.size(), tokens_gpu.size());
  ASSERT_EQ(golden_token_stream.size(), token_indices_gpu.size());

  for (std::size_t i = 0; i < tokens_gpu.size(); i++) {
    // Ensure the index the tokens are pointing to do match
    EXPECT_EQ(golden_token_stream[i].first, token_indices_gpu[i]) << "Mismatch at #" << i;
    // Ensure the token category is correct
    EXPECT_EQ(golden_token_stream[i].second, tokens_gpu[i]) << "Mismatch at #" << i;
  }
}

TEST_F(JsonTest, PostProcessTokenStream)
{
  // Golden token stream sample
  using token_t       = cuio_json::token_t;
  using token_index_t = cuio_json::SymbolOffsetT;
  using tuple_t       = cuda::std::tuple<token_index_t, cuio_json::PdaTokenT>;

  std::vector<tuple_t> const input = {// Line 0 (invalid)
                                      {0, token_t::LineEnd},
                                      {0, token_t::StructBegin},
                                      {1, token_t::StructMemberBegin},
                                      {1, token_t::FieldNameBegin},
                                      {3, token_t::FieldNameEnd},
                                      {5, token_t::ValueBegin},
                                      {7, token_t::ValueEnd},
                                      {7, token_t::StructMemberEnd},
                                      {7, token_t::StructEnd},
                                      {8, token_t::ErrorBegin},
                                      {9, token_t::LineEnd},
                                      // Line 1
                                      {10, token_t::StructBegin},
                                      {11, token_t::StructMemberBegin},
                                      {11, token_t::FieldNameBegin},
                                      {13, token_t::FieldNameEnd},
                                      {15, token_t::LineEnd},
                                      // Line 2 (invalid)
                                      {16, token_t::StructBegin},
                                      {17, token_t::StructMemberBegin},
                                      {17, token_t::FieldNameBegin},
                                      {19, token_t::FieldNameEnd},
                                      {21, token_t::StructBegin},
                                      {22, token_t::StructMemberBegin},
                                      {22, token_t::FieldNameBegin},
                                      {24, token_t::FieldNameEnd},
                                      {26, token_t::ListBegin},
                                      {27, token_t::ValueBegin},
                                      {29, token_t::ErrorBegin},
                                      {30, token_t::LineEnd},
                                      // Line 3 (invalid)
                                      {31, token_t::StructBegin},
                                      {32, token_t::StructMemberBegin},
                                      {32, token_t::FieldNameBegin},
                                      {34, token_t::FieldNameEnd},
                                      {36, token_t::ListBegin},
                                      {37, token_t::ValueBegin},
                                      {38, token_t::ValueEnd},
                                      {38, token_t::ListEnd},
                                      {39, token_t::StructMemberEnd},
                                      {39, token_t::StructEnd},
                                      {40, token_t::ErrorBegin},
                                      {40, token_t::LineEnd},
                                      // Line 4
                                      {41, token_t::LineEnd},
                                      // Line 5
                                      {42, token_t::StructBegin},
                                      {43, token_t::StructMemberBegin},
                                      {43, token_t::FieldNameBegin},
                                      {45, token_t::FieldNameEnd},
                                      {47, token_t::ValueBegin},
                                      {50, token_t::ValueEnd},
                                      {50, token_t::StructMemberEnd},
                                      {50, token_t::StructEnd}};

  std::vector<tuple_t> const expected_output = {// Line 0 (invalid)
                                                {0, token_t::StructBegin},
                                                {0, token_t::StructEnd},
                                                // Line 1
                                                {10, token_t::StructBegin},
                                                {11, token_t::StructMemberBegin},
                                                {11, token_t::FieldNameBegin},
                                                {13, token_t::FieldNameEnd},
                                                // Line 2 (invalid)
                                                {0, token_t::StructBegin},
                                                {0, token_t::StructEnd},
                                                // Line 3 (invalid)
                                                {0, token_t::StructBegin},
                                                {0, token_t::StructEnd},
                                                // Line 4 (empty)
                                                // Line 5
                                                {42, token_t::StructBegin},
                                                {43, token_t::StructMemberBegin},
                                                {43, token_t::FieldNameBegin},
                                                {45, token_t::FieldNameEnd},
                                                {47, token_t::ValueBegin},
                                                {50, token_t::ValueEnd},
                                                {50, token_t::StructMemberEnd},
                                                {50, token_t::StructEnd}};

  // Decompose tuples
  auto const stream = cudf::get_default_stream();
  std::vector<token_index_t> offsets(input.size());
  std::vector<cuio_json::PdaTokenT> tokens(input.size());
  auto token_tuples = cuda::make_zip_iterator(offsets.begin(), tokens.begin());
  thrust::copy(input.cbegin(), input.cend(), token_tuples);

  // Initialize device-side test data
  auto const d_offsets = cudf::detail::make_device_uvector_async(
    cudf::host_span<token_index_t const>{offsets.data(), offsets.size()},
    stream,
    cudf::get_current_device_resource_ref());
  auto const d_tokens = cudf::detail::make_device_uvector_async(
    tokens, stream, cudf::get_current_device_resource_ref());

  // Run system-under-test
  auto [d_filtered_tokens, d_filtered_indices] =
    cuio_json::detail::process_token_stream(d_tokens, d_offsets, stream);

  auto const filtered_tokens  = cudf::detail::make_std_vector_async(d_filtered_tokens, stream);
  auto const filtered_indices = cudf::detail::make_std_vector_async(d_filtered_indices, stream);
  stream.synchronize();

  // Verify the number of tokens matches
  ASSERT_EQ(filtered_tokens.size(), expected_output.size());
  ASSERT_EQ(filtered_indices.size(), expected_output.size());

  for (std::size_t i = 0; i < filtered_tokens.size(); i++) {
    // Ensure the index the tokens are pointing to do match
    EXPECT_EQ(cuda::std::get<0>(expected_output[i]), filtered_indices[i]) << "Mismatch at #" << i;
    // Ensure the token category is correct
    EXPECT_EQ(cuda::std::get<1>(expected_output[i]), filtered_tokens[i]) << "Mismatch at #" << i;
  }
}

TEST_P(JsonDelimiterParamTest, UTF_JSON)
{
  // Prepare cuda stream for data transfers & kernels
  auto const stream    = cudf::get_default_stream();
  auto mr              = cudf::get_current_device_resource_ref();
  auto json_parser     = cuio_json::detail::device_parse_nested_json;
  char const delimiter = GetParam();

  // Default parsing options
  cudf::io::json_reader_options default_options{};
  default_options.set_delimiter(delimiter);

  // Only ASCII string
  std::string ascii_pass = R"([
  {"a":1,"b":2,"c":[3], "d": {}},
  {"a":1,"b":4.0,"c":[], "d": {"year":1882,"author": "Bharathi"}},
  {"a":1,"b":6.0,"c":[5, 7], "d": null},
  {"a":1,"b":8.0,"c":null, "d": {}},
  {"a":1,"b":null,"c":null},
  {"a":1,"b":Infinity,"c":[null], "d": {"year":-600,"author": "Kaniyan"}}])";
  std::replace(ascii_pass.begin(), ascii_pass.end(), '\n', delimiter);

  auto const d_ascii_pass = cudf::detail::make_device_uvector(
    cudf::host_span<char const>{ascii_pass.c_str(), ascii_pass.size()},
    stream,
    cudf::get_current_device_resource_ref());

  CUDF_EXPECT_NO_THROW(json_parser(d_ascii_pass, default_options, stream, mr));

  // utf-8 string that fails parsing.
  std::string utf_failed = R"([
  {"a":1,"b":2,"c":[3], "d": {}},
  {"a":1,"b":4.0,"c":[], "d": {"year":1882,"author": "Bharathi"}},
  {"a":1,"b":6.0,"c":[5, 7], "d": null},
  {"a":1,"b":8.0,"c":null, "d": {}},
  {"a":1,"b":null,"c":null},
  {"a":1,"b":Infinity,"c":[null], "d": {"year":-600,"author": "filip ʒakotɛ"}}])";
  std::replace(utf_failed.begin(), utf_failed.end(), '\n', delimiter);

  auto const d_utf_failed = cudf::detail::make_device_uvector(
    cudf::host_span<char const>{utf_failed.c_str(), utf_failed.size()},
    stream,
    cudf::get_current_device_resource_ref());
  CUDF_EXPECT_NO_THROW(json_parser(d_utf_failed, default_options, stream, mr));

  // utf-8 string that passes parsing.
  std::string utf_pass = R"([
  {"a":1,"b":2,"c":[3], "d": {}},
  {"a":1,"b":4.0,"c":[], "d": {"year":1882,"author": "Bharathi"}},
  {"a":1,"b":6.0,"c":[5, 7], "d": null},
  {"a":1,"b":8.0,"c":null, "d": {}},
  {"a":1,"b":null,"c":null},
  {"a":1,"b":Infinity,"c":[null], "d": {"year":-600,"author": "Kaniyan"}},
  {"a":1,"b":NaN,"c":[null, null], "d": {"year": 2, "author": "filip ʒakotɛ"}}])";
  std::replace(utf_pass.begin(), utf_pass.end(), '\n', delimiter);

  auto const d_utf_pass = cudf::detail::make_device_uvector(
    cudf::host_span<char const>{utf_pass.c_str(), utf_pass.size()},
    stream,
    cudf::get_current_device_resource_ref());
  CUDF_EXPECT_NO_THROW(json_parser(d_utf_pass, default_options, stream, mr));
}

TEST_F(JsonParserTest, ExtractColumnWithQuotes)
{
  using cuio_json::SymbolT;
  auto json_parser = cuio_json::detail::device_parse_nested_json;

  // Prepare cuda stream for data transfers & kernels
  auto const stream = cudf::get_default_stream();
  auto mr           = cudf::get_current_device_resource_ref();

  // Default parsing options
  cudf::io::json_reader_options options{};
  options.enable_keep_quotes(true);

  std::string const input = R"( [{"a":"0.0", "b":1.0}, {"b":1.1}, {"b":2.1, "a":"2.0"}] )";
  auto const d_input      = cudf::detail::make_device_uvector_async(
    cudf::host_span<char const>{input.c_str(), input.size()},
    stream,
    cudf::get_current_device_resource_ref());
  // Get the JSON's tree representation
  auto const cudf_table = json_parser(d_input, options, stream, mr);

  auto constexpr expected_col_count = 2;
  EXPECT_EQ(cudf_table.tbl->num_columns(), expected_col_count);

  auto expected_col1 =
    cudf::test::strings_column_wrapper({R"("0.0")", R"()", R"("2.0")"}, {true, false, true});
  auto expected_col2            = cudf::test::fixed_width_column_wrapper<double>({1.0, 1.1, 2.1});
  cudf::column_view parsed_col1 = cudf_table.tbl->get_column(0);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_col1, parsed_col1);
  cudf::column_view parsed_col2 = cudf_table.tbl->get_column(1);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_col2, parsed_col2);
}

TEST_F(JsonParserTest, ExpectFailMixStructAndList)
{
  using cuio_json::SymbolT;
  auto json_parser = cuio_json::detail::device_parse_nested_json;

  // Prepare cuda stream for data transfers & kernels
  auto const stream = cudf::get_default_stream();
  auto mr           = cudf::get_current_device_resource_ref();

  // Default parsing options
  cudf::io::json_reader_options options{};
  options.enable_keep_quotes(true);

  std::vector<std::string> const inputs_fail{
    R"( [{"a":[123], "b":1.0}, {"b":1.1}, {"b":2.1, "a":{"0":123}}] )",
    R"( [{"a":{"0":"foo"}, "b":1.0}, {"b":1.1}, {"b":2.1, "a":[123]}] )",
    R"( [{"a":{"0":null}, "b":1.0}, {"b":1.1}, {"b":2.1, "a":[123]}] )"};

  std::vector<std::string> const inputs_succeed{
    R"( [{"a":[123, {"0": 123}], "b":1.0}, {"b":1.1}, {"b":2.1}] )",
    R"( [{"a":[123, "123"], "b":1.0}, {"b":1.1}, {"b":2.1}] )"};

  // libcudf does not currently support a mix of lists and structs.
  for (auto const& input : inputs_fail) {
    auto const d_input = cudf::detail::make_device_uvector_async(
      cudf::host_span<char const>{input.c_str(), input.size()},
      stream,
      cudf::get_current_device_resource_ref());
    EXPECT_THROW(auto const cudf_table = json_parser(d_input, options, stream, mr),
                 cudf::logic_error);
  }

  for (auto const& input : inputs_succeed) {
    auto const d_input = cudf::detail::make_device_uvector_async(
      cudf::host_span<char const>{input.c_str(), input.size()},
      stream,
      cudf::get_current_device_resource_ref());
    CUDF_EXPECT_NO_THROW(auto const cudf_table = json_parser(d_input, options, stream, mr));
  }
}

TEST_F(JsonParserTest, EmptyString)
{
  using cuio_json::SymbolT;
  auto json_parser = cuio_json::detail::device_parse_nested_json;

  // Prepare cuda stream for data transfers & kernels
  auto const stream = cudf::get_default_stream();
  auto mr           = cudf::get_current_device_resource_ref();

  // Default parsing options
  cudf::io::json_reader_options default_options{};

  std::string const input = R"([])";
  auto const d_input =
    cudf::detail::make_device_uvector(cudf::host_span<char const>{input.c_str(), input.size()},
                                      stream,
                                      cudf::get_current_device_resource_ref());
  // Get the JSON's tree representation
  auto const cudf_table = json_parser(d_input, default_options, stream, mr);

  auto const expected_col_count = 0;
  EXPECT_EQ(cudf_table.tbl->num_columns(), expected_col_count);
}

TEST_P(JsonDelimiterParamTest, RecoveringTokenStreamNewlineAndDelimiter)
{
  // Test input. Inline comments used to indicate character indexes
  //                           012345678 <= line 0
  char const delimiter = GetParam();

  /* Input:
   * {"a":2}
   * {"a":<delimiter>{"a":{"a":[321<delimiter>{"a":[1]}
   *
   * <delimiter>{"b":123}
   * {"b":123}
   */
  std::string input = R"({"a":2})"
                      "\n";
  // starting position 8 (zero indexed)
  input += R"({"a":)" + std::string(1, delimiter);
  // starting position 14 (zero indexed)
  input += R"({"a":{"a":[321)" + std::string(1, delimiter);
  // starting position 29 (zero indexed)
  input += R"({"a":[1]})" + std::string("\n\n") + std::string(1, delimiter);
  // starting position 41 (zero indexed)
  input += R"({"b":123})"
           "\n";
  // starting position 51 (zero indexed)
  input += R"({"b":123})";

  // Golden token stream sample
  using token_t = cuio_json::token_t;
  std::vector<std::pair<std::size_t, cuio_json::PdaTokenT>> golden_token_stream;
  if (delimiter != '\n') {
    golden_token_stream.resize(28);
    golden_token_stream = {// Line 0 (valid)
                           {0, token_t::StructBegin},
                           {1, token_t::StructMemberBegin},
                           {1, token_t::FieldNameBegin},
                           {3, token_t::FieldNameEnd},
                           {5, token_t::ValueBegin},
                           {6, token_t::ValueEnd},
                           {6, token_t::StructMemberEnd},
                           {6, token_t::StructEnd},
                           // Line 1 (invalid)
                           {0, token_t::StructBegin},
                           {0, token_t::StructEnd},
                           // Line 2 (valid)
                           {29, token_t::StructBegin},
                           {30, token_t::StructMemberBegin},
                           {30, token_t::FieldNameBegin},
                           {32, token_t::FieldNameEnd},
                           {34, token_t::ListBegin},
                           {35, token_t::ValueBegin},
                           {36, token_t::ValueEnd},
                           {36, token_t::ListEnd},
                           {37, token_t::StructMemberEnd},
                           {37, token_t::StructEnd},
                           // Line 3 (valid)
                           {41, token_t::StructBegin},
                           {42, token_t::StructMemberBegin},
                           {42, token_t::FieldNameBegin},
                           {44, token_t::FieldNameEnd},
                           {46, token_t::ValueBegin},
                           {49, token_t::ValueEnd},
                           {49, token_t::StructMemberEnd},
                           {49, token_t::StructEnd}};
  } else {
    /* Input:
     * {"a":2}
     * {"a":
     * {"a":{"a":[321
     * {"a":[1]}
     *
     *
     * {"b":123}
     * {"b":123}
     */
    golden_token_stream.resize(38);
    golden_token_stream = {// Line 0 (valid)
                           {0, token_t::StructBegin},
                           {1, token_t::StructMemberBegin},
                           {1, token_t::FieldNameBegin},
                           {3, token_t::FieldNameEnd},
                           {5, token_t::ValueBegin},
                           {6, token_t::ValueEnd},
                           {6, token_t::StructMemberEnd},
                           {6, token_t::StructEnd},
                           // Line 1 (invalid)
                           {0, token_t::StructBegin},
                           {0, token_t::StructEnd},
                           // Line 2 (invalid)
                           {0, token_t::StructBegin},
                           {0, token_t::StructEnd},
                           // Line 3 (valid)
                           {29, token_t::StructBegin},
                           {30, token_t::StructMemberBegin},
                           {30, token_t::FieldNameBegin},
                           {32, token_t::FieldNameEnd},
                           {34, token_t::ListBegin},
                           {35, token_t::ValueBegin},
                           {36, token_t::ValueEnd},
                           {36, token_t::ListEnd},
                           {37, token_t::StructMemberEnd},
                           {37, token_t::StructEnd},
                           // Line 4 (valid)
                           {41, token_t::StructBegin},
                           {42, token_t::StructMemberBegin},
                           {42, token_t::FieldNameBegin},
                           {44, token_t::FieldNameEnd},
                           {46, token_t::ValueBegin},
                           {49, token_t::ValueEnd},
                           {49, token_t::StructMemberEnd},
                           {49, token_t::StructEnd},
                           // Line 5 (valid)
                           {51, token_t::StructBegin},
                           {52, token_t::StructMemberBegin},
                           {52, token_t::FieldNameBegin},
                           {54, token_t::FieldNameEnd},
                           {56, token_t::ValueBegin},
                           {59, token_t::ValueEnd},
                           {59, token_t::StructMemberEnd},
                           {59, token_t::StructEnd}};
  }

  auto const stream = cudf::get_default_stream();

  // Default parsing options
  cudf::io::json_reader_options default_options{};
  default_options.set_recovery_mode(cudf::io::json_recovery_mode_t::RECOVER_WITH_NULL);
  default_options.enable_lines(true);
  default_options.set_delimiter(delimiter);

  // Prepare input & output buffers
  cudf::string_scalar const d_scalar(input, true, stream);
  auto const d_input = cudf::device_span<cuio_json::SymbolT const>{
    d_scalar.data(), static_cast<size_t>(d_scalar.size())};

  // Parse the JSON and get the token stream
  auto [d_tokens_gpu, d_token_indices_gpu] = cuio_json::detail::get_token_stream(
    d_input, default_options, stream, cudf::get_current_device_resource_ref());
  // Copy back the number of tokens that were written
  auto const tokens_gpu        = cudf::detail::make_std_vector_async(d_tokens_gpu, stream);
  auto const token_indices_gpu = cudf::detail::make_std_vector_async(d_token_indices_gpu, stream);

  stream.synchronize();
  // Verify the number of tokens matches
  ASSERT_EQ(golden_token_stream.size(), tokens_gpu.size());
  ASSERT_EQ(golden_token_stream.size(), token_indices_gpu.size());

  for (std::size_t i = 0; i < tokens_gpu.size(); i++) {
    // Ensure the index the tokens are pointing to do match
    EXPECT_EQ(golden_token_stream[i].first, token_indices_gpu[i]) << "Mismatch at #" << i;
    // Ensure the token category is correct
    EXPECT_EQ(golden_token_stream[i].second, tokens_gpu[i]) << "Mismatch at #" << i;
  }
}

TEST_P(JsonDelimiterParamTest, RecoveringTokenStreamNewlineAsWSAndDelimiter)
{
  // Test input. Inline comments used to indicate character indexes
  //                           012345678 <= line 0
  char const delimiter = GetParam();

  /* Input: (Note that \n is considered whitespace according to the JSON spec when it is not used as
   * a delimiter for JSONL)
   * {"a":2}
   * {"a":<delimiter>{"a":{"a":[321<delimiter>{"a":[1]}
   *
   * <delimiter>{"b":123}
   * {"b":123}<delimiter>
   * {"b"\n:\n\n\n123\n}
   */
  std::string input = R"({"a":2})"
                      "\n";
  // starting position 8 (zero indexed)
  input += R"({"a":)" + std::string(1, delimiter);
  // starting position 14 (zero indexed)
  input += R"({"a":{"a":[321)" + std::string(1, delimiter);
  // starting position 29 (zero indexed)
  input += R"({"a":[1]})" + std::string("\n\n") + std::string(1, delimiter);
  // starting position 41 (zero indexed)
  input += R"({"b":123})"
           "\n";
  // starting position 51 (zero indexed)
  input += R"({"b":123})" + std::string(1, delimiter);
  // starting position 61 (zero indexed)
  input += R"({"b")" + std::string("\n:\n\n\n123\n}");

  // Golden token stream sample
  using token_t = cuio_json::token_t;
  std::vector<std::pair<std::size_t, cuio_json::PdaTokenT>> golden_token_stream;
  if (delimiter != '\n') {
    golden_token_stream = {// Line 0 (valid)
                           {0, token_t::StructBegin},
                           {1, token_t::StructMemberBegin},
                           {1, token_t::FieldNameBegin},
                           {3, token_t::FieldNameEnd},
                           {5, token_t::ValueBegin},
                           {6, token_t::ValueEnd},
                           {6, token_t::StructMemberEnd},
                           {6, token_t::StructEnd},
                           // Line 1 (invalid)
                           {0, token_t::StructBegin},
                           {0, token_t::StructEnd},
                           // Line 2 (valid)
                           {29, token_t::StructBegin},
                           {30, token_t::StructMemberBegin},
                           {30, token_t::FieldNameBegin},
                           {32, token_t::FieldNameEnd},
                           {34, token_t::ListBegin},
                           {35, token_t::ValueBegin},
                           {36, token_t::ValueEnd},
                           {36, token_t::ListEnd},
                           {37, token_t::StructMemberEnd},
                           {37, token_t::StructEnd},
                           // Line 3 (valid)
                           {41, token_t::StructBegin},
                           {42, token_t::StructMemberBegin},
                           {42, token_t::FieldNameBegin},
                           {44, token_t::FieldNameEnd},
                           {46, token_t::ValueBegin},
                           {49, token_t::ValueEnd},
                           {49, token_t::StructMemberEnd},
                           {49, token_t::StructEnd},
                           // Line 4 (valid)
                           {61, token_t::StructBegin},
                           {62, token_t::StructMemberBegin},
                           {62, token_t::FieldNameBegin},
                           {64, token_t::FieldNameEnd},
                           {70, token_t::ValueBegin},
                           {73, token_t::ValueEnd},
                           {74, token_t::StructMemberEnd},
                           {74, token_t::StructEnd}};
  } else {
    /* Input:
     * {"a":2}
     * {"a":
     * {"a":{"a":[321
     * {"a":[1]}
     *
     *
     * {"b":123}
     * {"b":123}
     * {"b"\n:\n\n\n123\n}
     */
    golden_token_stream = {// Line 0 (valid)
                           {0, token_t::StructBegin},
                           {1, token_t::StructMemberBegin},
                           {1, token_t::FieldNameBegin},
                           {3, token_t::FieldNameEnd},
                           {5, token_t::ValueBegin},
                           {6, token_t::ValueEnd},
                           {6, token_t::StructMemberEnd},
                           {6, token_t::StructEnd},
                           // Line 1 (invalid)
                           {0, token_t::StructBegin},
                           {0, token_t::StructEnd},
                           // Line 2 (invalid)
                           {0, token_t::StructBegin},
                           {0, token_t::StructEnd},
                           // Line 3 (valid)
                           {29, token_t::StructBegin},
                           {30, token_t::StructMemberBegin},
                           {30, token_t::FieldNameBegin},
                           {32, token_t::FieldNameEnd},
                           {34, token_t::ListBegin},
                           {35, token_t::ValueBegin},
                           {36, token_t::ValueEnd},
                           {36, token_t::ListEnd},
                           {37, token_t::StructMemberEnd},
                           {37, token_t::StructEnd},
                           // Line 4 (valid)
                           {41, token_t::StructBegin},
                           {42, token_t::StructMemberBegin},
                           {42, token_t::FieldNameBegin},
                           {44, token_t::FieldNameEnd},
                           {46, token_t::ValueBegin},
                           {49, token_t::ValueEnd},
                           {49, token_t::StructMemberEnd},
                           {49, token_t::StructEnd},
                           // Line 5 (valid)
                           {51, token_t::StructBegin},
                           {52, token_t::StructMemberBegin},
                           {52, token_t::FieldNameBegin},
                           {54, token_t::FieldNameEnd},
                           {56, token_t::ValueBegin},
                           {59, token_t::ValueEnd},
                           {59, token_t::StructMemberEnd},
                           {59, token_t::StructEnd},
                           // Line 6 (invalid)
                           {0, token_t::StructBegin},
                           {0, token_t::StructEnd},
                           {0, token_t::StructBegin},
                           {0, token_t::StructEnd},
                           {0, token_t::StructBegin},
                           {0, token_t::StructEnd},
                           {0, token_t::StructBegin},
                           {0, token_t::StructEnd}};
  }

  auto const stream = cudf::get_default_stream();

  // Prepare input & output buffers
  cudf::string_scalar const d_scalar(input, true, stream);
  auto const d_input = cudf::device_span<cuio_json::SymbolT const>{
    d_scalar.data(), static_cast<size_t>(d_scalar.size())};

  // Default parsing options
  cudf::io::json_reader_options const in_opts =
    cudf::io::json_reader_options::builder(cudf::io::source_info{})
      .recovery_mode(cudf::io::json_recovery_mode_t::RECOVER_WITH_NULL)
      .delimiter(delimiter)
      .lines(true);

  // Parse the JSON and get the token stream
  auto [d_tokens_gpu, d_token_indices_gpu] = cuio_json::detail::get_token_stream(
    d_input, in_opts, stream, cudf::get_current_device_resource_ref());
  // Copy back the number of tokens that were written
  auto const tokens_gpu        = cudf::detail::make_std_vector_async(d_tokens_gpu, stream);
  auto const token_indices_gpu = cudf::detail::make_std_vector_async(d_token_indices_gpu, stream);

  stream.synchronize();
  // Verify the number of tokens matches
  ASSERT_EQ(golden_token_stream.size(), tokens_gpu.size());
  ASSERT_EQ(golden_token_stream.size(), token_indices_gpu.size());

  for (std::size_t i = 0; i < tokens_gpu.size(); i++) {
    // Ensure the index the tokens are pointing to do match
    EXPECT_EQ(golden_token_stream[i].first, token_indices_gpu[i]) << "Mismatch at #" << i;
    // Ensure the token category is correct
    EXPECT_EQ(golden_token_stream[i].second, tokens_gpu[i]) << "Mismatch at #" << i;
  }
}

TEST_F(JsonTest, StringContainingNonAsciiBytes)
{
  for (int b : {0x80, 0xA0, 0xC3, 0xE2, 0xF0, 0xFF}) {
    std::string const expected_cell{static_cast<char>(b)};
    std::string s{R"({"k":")"};
    s += expected_cell;
    s += R"("})";
    s += '\n';
    auto const opts = cudf::io::json_reader_options::builder(
                        cudf::io::source_info{cudf::host_span<std::byte const>{
                          reinterpret_cast<std::byte const*>(s.data()), s.size()}})
                        .lines(true)
                        .build();
    cudf::io::table_with_metadata tbl;
    ASSERT_NO_THROW(tbl = cudf::io::read_json(opts)) << "byte 0x" << std::hex << b;
    ASSERT_EQ(tbl.tbl->num_columns(), 1) << "byte 0x" << std::hex << b;
    ASSERT_EQ(tbl.tbl->num_rows(), 1) << "byte 0x" << std::hex << b;
    cudf::test::strings_column_wrapper expected({expected_cell});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(tbl.tbl->view().column(0), expected);
  }
}

// Rejects unquoted values whose first byte is neither '-' nor a digit.
TEST_F(JsonTest, RejectsUnquotedValuesWithInvalidLeadingChar)
{
  for (char const* bad : {"!", "+", " ", ".", "x", "/"}) {
    std::string s   = std::string{R"({"k":)"} + bad + "}\n";
    auto const opts = cudf::io::json_reader_options::builder(cudf::io::source_info{})
                        .lines(true)
                        .recovery_mode(cudf::io::json_recovery_mode_t::RECOVER_WITH_NULL)
                        .strict_validation(true)
                        .build();
    cudf::string_scalar const d_scalar(s, true);
    auto const d_input = cudf::device_span<cuio_json::SymbolT const>{
      d_scalar.data(), static_cast<size_t>(d_scalar.size())};
    auto const stream = cudf::get_default_stream();
    using token_t     = cuio_json::token_t;
    std::vector<cuio_json::PdaTokenT> tokens{token_t::StructBegin,
                                             token_t::StructMemberBegin,
                                             token_t::FieldNameBegin,
                                             token_t::FieldNameEnd,
                                             token_t::ValueBegin,
                                             token_t::ValueEnd,
                                             token_t::StructMemberEnd,
                                             token_t::StructEnd,
                                             token_t::LineEnd};
    std::vector<cuio_json::SymbolOffsetT> token_indices{0, 0, 1, 3, 5, 6, 6, 6, 7};
    auto d_tokens = cudf::detail::make_device_uvector_async(
      tokens, stream, cudf::get_current_device_resource_ref());
    auto d_token_indices = cudf::detail::make_device_uvector_async(
      token_indices, stream, cudf::get_current_device_resource_ref());

    cuio_json::detail::validate_token_stream(d_input, d_tokens, d_token_indices, opts, stream);
    auto const validated_tokens = cudf::detail::make_std_vector_async(d_tokens, stream);
    stream.synchronize();
    EXPECT_NE(std::find(validated_tokens.begin(), validated_tokens.end(), token_t::ErrorBegin),
              validated_tokens.end())
      << "value " << bad << " was unexpectedly accepted as a number";
  }
}

// Rejects JSON inputs whose nesting depth exceeds the supported maximum.
TEST_F(JsonTest, NestedInputAboveDepthLimit)
{
  std::size_t const depth = std::numeric_limits<int8_t>::max() + std::size_t{1};
  std::string s(depth, '[');
  s.append(depth, ']');
  auto const opts = cudf::io::json_reader_options::builder(
                      cudf::io::source_info{cudf::host_span<std::byte const>{
                        reinterpret_cast<std::byte const*>(s.data()), s.size()}})
                      .build();
  EXPECT_THROW(cudf::io::read_json(opts), cudf::logic_error);
}

// Accepts JSON inputs at the maximum supported nesting depth.
TEST_F(JsonTest, NestedInputAtDepthLimit)
{
  std::size_t const depth = std::numeric_limits<int8_t>::max();
  std::string s(depth, '[');
  s.append(depth, ']');
  auto const opts = cudf::io::json_reader_options::builder(
                      cudf::io::source_info{cudf::host_span<std::byte const>{
                        reinterpret_cast<std::byte const*>(s.data()), s.size()}})
                      .build();
  EXPECT_NO_THROW(cudf::io::read_json(opts));
}

CUDF_TEST_PROGRAM_MAIN()
