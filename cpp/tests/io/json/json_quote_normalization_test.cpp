/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/cudf_gtest.hpp>
#include <cudf_test/default_stream.hpp>
#include <cudf_test/table_utilities.hpp>
#include <cudf_test/testing_main.hpp>

#include <cudf/io/datasource.hpp>
#include <cudf/io/detail/json.hpp>
#include <cudf/io/json.hpp>
#include <cudf/io/types.hpp>

#include <rmm/device_buffer.hpp>

#include <string>

// Base test fixture for tests
struct JsonNormalizationTest : public cudf::test::BaseFixture {};

void run_test(std::string const& host_input,
              std::string const& expected_host_output,
              char delimiter = '\n')
{
  auto stream_view  = cudf::test::get_default_stream();
  auto device_input = rmm::device_buffer(host_input.c_str(), host_input.size(), stream_view);
  stream_view.synchronize();

  // Preprocessing FST
  cudf::io::datasource::owning_buffer<rmm::device_buffer> device_data(std::move(device_input));
  cudf::io::json::detail::normalize_single_quotes(
    device_data, delimiter, stream_view, cudf::get_current_device_resource_ref());

  std::string preprocessed_host_output(device_data.size(), 0);
  CUDF_CUDA_TRY(cudaMemcpyAsync(preprocessed_host_output.data(),
                                device_data.data(),
                                preprocessed_host_output.size(),
                                cudaMemcpyDefault,
                                stream_view.value()))
  stream_view.synchronize();
  CUDF_TEST_EXPECT_VECTOR_EQUAL(
    preprocessed_host_output, expected_host_output, preprocessed_host_output.size());
}

TEST_F(JsonNormalizationTest, GroundTruth_QuoteNormalization_Single)
{
  std::string input  = R"({'A':"TEST'"} ['OTHER STUFF'])";
  std::string output = R"({"A":"TEST'"} ["OTHER STUFF"])";
  run_test(input, output);
}

TEST_F(JsonNormalizationTest, GroundTruth_QuoteNormalization_MoreSingle)
{
  std::string input  = R"(['\t','\\t','\\','\\\"\'\\\\','\n','\b','\u0012'])";
  std::string output = R"(["\t","\\t","\\","\\\"'\\\\","\n","\b","\u0012"])";
  run_test(input, output);
}

TEST_F(JsonNormalizationTest, GroundTruth_QuoteNormalization_DoubleInSingle)
{
  std::string input  = R"({"A":'TEST"'})";
  std::string output = R"({"A":"TEST\""})";
  run_test(input, output);
}

TEST_F(JsonNormalizationTest, GroundTruth_QuoteNormalization_MoreDoubleInSingle)
{
  std::string input = R"({"ain't ain't a word and you ain't supposed to say it":'"""""""""""'})";
  std::string output =
    R"({"ain't ain't a word and you ain't supposed to say it":"\"\"\"\"\"\"\"\"\"\"\""})";
  run_test(input, output);
}

TEST_F(JsonNormalizationTest, GroundTruth_QuoteNormalization_StillMoreDoubleInSingle)
{
  std::string input  = R"([{"ABC':'CBA":'XYZ":"ZXY'}])";
  std::string output = R"([{"ABC':'CBA":"XYZ\":\"ZXY"}])";
  run_test(input, output);
}

TEST_F(JsonNormalizationTest, GroundTruth_QuoteNormalization_DoubleInSingleAndViceVersa)
{
  std::string input  = R"(['{"A": "B"}',"{'A': 'B'}"])";
  std::string output = R"(["{\"A\": \"B\"}","{'A': 'B'}"])";
  run_test(input, output);
}

TEST_F(JsonNormalizationTest, GroundTruth_QuoteNormalization_DoubleAndSingleInSingle)
{
  std::string input  = R"({"\"'\"'\"'\"'":'"\'"\'"\'"\'"'})";
  std::string output = R"({"\"'\"'\"'\"'":"\"'\"'\"'\"'\""})";
  run_test(input, output);
}

TEST_F(JsonNormalizationTest, GroundTruth_QuoteNormalization_EscapedSingleInDouble)
{
  std::string input  = R"(["\t","\\t","\\","\\\'\"\\\\","\n","\b"])";
  std::string output = R"(["\t","\\t","\\","\\'\"\\\\","\n","\b"])";
  run_test(input, output);
}

TEST_F(JsonNormalizationTest, GroundTruth_QuoteNormalization_EscapedDoubleInSingle)
{
  std::string input  = R"(["\t","\\t","\\",'\\\'\"\\\\',"\n","\b"])";
  std::string output = R"(["\t","\\t","\\","\\'\"\\\\","\n","\b"])";
  run_test(input, output);
}

TEST_F(JsonNormalizationTest, GroundTruth_QuoteNormalization_Invalid_MismatchedQuotes)
{
  std::string input  = R"(["THIS IS A TEST'])";
  std::string output = R"(["THIS IS A TEST'])";
  run_test(input, output);
}

TEST_F(JsonNormalizationTest, GroundTruth_QuoteNormalization_Invalid_MismatchedQuotesEscapedOutput)
{
  std::string input  = R"(['THIS IS A TEST"])";
  std::string output = R"(["THIS IS A TEST\"])";
  run_test(input, output);
}

TEST_F(JsonNormalizationTest, GroundTruth_QuoteNormalization_Invalid_MoreMismatchedQuotes)
{
  std::string input  = R"({"MORE TEST'N":'RESUL})";
  std::string output = R"({"MORE TEST'N":"RESUL})";
  run_test(input, output);
}

TEST_F(JsonNormalizationTest, GroundTruth_QuoteNormalization_Invalid_NoEndQuote)
{
  std::string input  = R"({"NUMBER":100'0,'STRING':'SOMETHING'})";
  std::string output = R"({"NUMBER":100"0,"STRING":"SOMETHING"})";
  run_test(input, output);
}

TEST_F(JsonNormalizationTest, GroundTruth_QuoteNormalization_InvalidJSON)
{
  std::string input  = R"({'NUMBER':100"0,"STRING":"SOMETHING"})";
  std::string output = R"({"NUMBER":100"0,"STRING":"SOMETHING"})";
  run_test(input, output);
}

TEST_F(JsonNormalizationTest, GroundTruth_QuoteNormalization_Invalid_WrongBackslash)
{
  std::string input  = R"({'a':'\\''})";
  std::string output = R"({"a":"\\""})";
  run_test(input, output);
}

TEST_F(JsonNormalizationTest, GroundTruth_QuoteNormalization_Invalid_WrongBraces)
{
  std::string input  = R"(}'a': 'b'{)";
  std::string output = R"(}"a": "b"{)";
  run_test(input, output);
}

TEST_F(JsonNormalizationTest, GroundTruth_QuoteNormalization_NonNewlineDelimiter)
{
  std::string input{"{\"a\": \"1\n2\"}z{\'a\': 12}"};
  std::string output{"{\"a\": \"1\n2\"}z{\"a\": 12}"};
  run_test(input, output, 'z');
}

TEST_F(JsonNormalizationTest, ReadJsonOption)
{
  // Test input
  std::string const host_input = R"({"a": "1\n2"}h{'a': 12})";
  cudf::io::json_reader_options input_options =
    cudf::io::json_reader_options::builder(
      cudf::io::source_info{cudf::host_span<std::byte const>{
        reinterpret_cast<std::byte const*>(host_input.data()), host_input.size()}})
      .lines(true)
      .delimiter('h')
      .normalize_single_quotes(true);

  cudf::io::table_with_metadata processed_table = cudf::io::read_json(input_options);

  // Expected table
  std::string const expected_input = R"({"a": "1\n2"}h{"a": 12})";
  cudf::io::json_reader_options expected_input_options =
    cudf::io::json_reader_options::builder(
      cudf::io::source_info{cudf::host_span<std::byte const>{
        reinterpret_cast<std::byte const*>(expected_input.data()), expected_input.size()}})
      .lines(true)
      .delimiter('h');

  cudf::io::table_with_metadata expected_table = cudf::io::read_json(expected_input_options);
  CUDF_TEST_EXPECT_TABLES_EQUAL(expected_table.tbl->view(), processed_table.tbl->view());
}

TEST_F(JsonNormalizationTest, ErrorCheck)
{
  // Test input
  std::string const host_input = R"({"A":'TEST"'})";
  cudf::io::json_reader_options input_options =
    cudf::io::json_reader_options::builder(
      cudf::io::source_info{cudf::host_span<std::byte const>{
        reinterpret_cast<std::byte const*>(host_input.data()), host_input.size()}})
      .lines(true);

  EXPECT_THROW(cudf::io::read_json(input_options), cudf::logic_error);
}

CUDF_TEST_PROGRAM_MAIN()
