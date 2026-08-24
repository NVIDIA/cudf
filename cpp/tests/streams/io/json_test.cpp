/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/default_stream.hpp>
#include <cudf_test/iterator_utilities.hpp>
#include <cudf_test/testing_main.hpp>

#include <cudf/io/json.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>

#include <string>
#include <vector>

class JSONTest : public cudf::test::BaseFixture {};

TEST_F(JSONTest, JSONreader)
{
  std::string data = "[1, 1.1]\n[2, 2.2]\n[3, 3.3]\n";
  cudf::io::json_reader_options in_options =
    cudf::io::json_reader_options::builder(
      cudf::io::source_info{cudf::host_span<std::byte const>{
        reinterpret_cast<std::byte const*>(data.data()), data.size()}})
      .dtypes(std::vector<cudf::data_type>{cudf::data_type{cudf::type_id::INT32},
                                           cudf::data_type{cudf::type_id::FLOAT64}})
      .lines(true);
  cudf::io::table_with_metadata result =
    cudf::io::read_json(in_options, cudf::test::get_default_stream());
}

TEST_F(JSONTest, JSONreaderWithDiagnostics)
{
  std::string data = "[1, 1.1]\n[2, 2.2]\n[3, 3.3]\n";
  cudf::io::json_reader_options in_options =
    cudf::io::json_reader_options::builder(
      cudf::io::source_info{cudf::host_span<std::byte const>{
        reinterpret_cast<std::byte const*>(data.data()), data.size()}})
      .dtypes(std::vector<cudf::data_type>{cudf::data_type{cudf::type_id::INT32},
                                           cudf::data_type{cudf::type_id::FLOAT64}})
      .lines(true);
  cudf::io::json_reader_result result =
    cudf::io::read_json_with_diagnostics(in_options, cudf::test::get_default_stream());
}

TEST_F(JSONTest, JSONreaderWithRowDiagnostics)
{
  std::string data = "[1, 1.1]\n[2, 2.2]\n[3, 3.3]\n";
  cudf::io::json_reader_options in_options =
    cudf::io::json_reader_options::builder(
      cudf::io::source_info{cudf::host_span<std::byte const>{
        reinterpret_cast<std::byte const*>(data.data()), data.size()}})
      .dtypes(std::vector<cudf::data_type>{cudf::data_type{cudf::type_id::INT32},
                                           cudf::data_type{cudf::type_id::FLOAT64}})
      .lines(true);
  cudf::io::json_reader_result_with_row_diagnostics result =
    cudf::io::read_json_with_row_diagnostics(in_options, cudf::test::get_default_stream());
}

TEST_F(JSONTest, JSONwriter)
{
  cudf::test::strings_column_wrapper col1{"a", "b", "c"};
  cudf::test::strings_column_wrapper col2{"d", "e", "f"};
  cudf::test::fixed_width_column_wrapper<int> col3{1, 2, 3};
  cudf::test::fixed_width_column_wrapper<float> col4{1.5, 2.5, 3.5};
  cudf::test::fixed_width_column_wrapper<int16_t> col5{{1, 2, 3},
                                                       cudf::test::iterators::nulls_at({0, 2})};
  cudf::table_view tbl_view{{col1, col2, col3, col4, col5}};
  cudf::io::table_metadata mt{{{"col1"}, {"col2"}, {"int"}, {"float"}, {"int16"}}};

  std::vector<char> out_buffer;
  auto destination     = cudf::io::sink_info(&out_buffer);
  auto options_builder = cudf::io::json_writer_options_builder(destination, tbl_view)
                           .include_nulls(true)
                           .metadata(mt)
                           .lines(false)
                           .na_rep("null");

  cudf::io::write_json(options_builder.build(), cudf::test::get_default_stream());
}

CUDF_TEST_PROGRAM_MAIN()
