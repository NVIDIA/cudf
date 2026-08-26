/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/default_stream.hpp>
#include <cudf_test/iterator_utilities.hpp>
#include <cudf_test/lists_column_wrapper.hpp>
#include <cudf_test/table_utilities.hpp>
#include <cudf_test/testing_main.hpp>

#include <cudf/detail/iterator.cuh>
#include <cudf/io/detail/codec.hpp>
#include <cudf/io/json.hpp>
#include <cudf/io/types.hpp>
#include <cudf/types.hpp>
#include <cudf/unary.hpp>

#include <cuda/iterator>

#include <string>
#include <vector>

struct JsonWriterTest : public cudf::test::BaseFixture {};

/**
 * @brief Test fixture for parametrized JSON reader tests
 */
struct JsonCompressedWriterTest : public cudf::test::BaseFixture,
                                  public testing::WithParamInterface<cudf::io::compression_type> {};

// Parametrize qualifying JSON tests for multiple compression types
INSTANTIATE_TEST_SUITE_P(JsonCompressedWriterTest,
                         JsonCompressedWriterTest,
                         ::testing::Values(cudf::io::compression_type::GZIP,
                                           cudf::io::compression_type::SNAPPY,
                                           cudf::io::compression_type::NONE));

void run_test(cudf::io::json_writer_options const& wopts, std::string const& expected)
{
  auto outbuf   = wopts.get_sink().buffers().front();
  auto comptype = wopts.get_compression();
  cudf::io::write_json(wopts, cudf::test::get_default_stream());
  if (comptype != cudf::io::compression_type::NONE) {
    auto decomp_out_buffer = cudf::io::detail::decompress(
      comptype,
      cudf::host_span<uint8_t const>(reinterpret_cast<uint8_t*>(outbuf->data()), outbuf->size()));
    EXPECT_EQ(expected,
              std::string_view(reinterpret_cast<char*>(decomp_out_buffer.data()),
                               decomp_out_buffer.size()));
  } else
    EXPECT_EQ(expected, std::string_view(outbuf->data(), outbuf->size()));
}

TEST_P(JsonCompressedWriterTest, EmptyInput)
{
  cudf::test::strings_column_wrapper col1;
  cudf::test::strings_column_wrapper col2;
  cudf::test::fixed_width_column_wrapper<int> col3;
  cudf::test::fixed_width_column_wrapper<float> col4;
  cudf::test::fixed_width_column_wrapper<int16_t> col5;
  cudf::table_view tbl_view{{col1, col2, col3, col4}};
  cudf::io::table_metadata mt{{{"col1"}, {"col2"}, {"int"}, {"float"}, {"int16"}}};

  std::vector<char> out_buffer;
  auto destination = cudf::io::sink_info(&out_buffer);
  auto out_options = cudf::io::json_writer_options_builder(destination, tbl_view)
                       .include_nulls(true)
                       .metadata(mt)
                       .lines(false)
                       .na_rep("null")
                       .build();
  run_test(out_options, "[]");

  // Empty columns in table - JSON Lines
  out_buffer.clear();
  out_options.enable_lines(true);
  run_test(out_options, "\n");

  // Empty table - JSON Lines
  cudf::table_view tbl_view2{};
  out_options.set_table(tbl_view2);
  out_buffer.clear();
  run_test(out_options, "\n");
}

TEST_P(JsonCompressedWriterTest, EmptyLeaf)
{
  cudf::test::strings_column_wrapper col1{""};
  cudf::test::fixed_width_column_wrapper<cudf::size_type> offsets{0, 0};
  auto col2 = make_lists_column(
    1, offsets.release(), cudf::test::strings_column_wrapper{}.release(), 0, rmm::device_buffer{});
  auto col3 = cudf::test::lists_column_wrapper<int>::make_one_empty_row_column();
  cudf::table_view tbl_view{{col1, *col2, col3}};
  cudf::io::table_metadata mt{{{"col1"}, {"col2"}, {"col3"}}};

  std::vector<char> out_buffer;
  auto destination = cudf::io::sink_info(&out_buffer);
  auto out_options = cudf::io::json_writer_options_builder(destination, tbl_view)
                       .include_nulls(true)
                       .metadata(mt)
                       .lines(false)
                       .na_rep("null")
                       .build();
  run_test(out_options, R"([{"col1":"","col2":[],"col3":[]}])");

  // Empty columns in table - JSON Lines
  out_buffer.clear();
  out_options.enable_lines(true);
  std::string const expected_lines = R"({"col1":"","col2":[],"col3":[]})"
                                     "\n";
  run_test(out_options, expected_lines);
}

TEST_F(JsonWriterTest, ErrorCases)
{
  cudf::test::strings_column_wrapper col1{"a", "b", "c"};
  cudf::test::strings_column_wrapper col2{"d", "e", "f"};
  cudf::test::fixed_width_column_wrapper<int> col3{1, 2, 3};
  cudf::test::fixed_width_column_wrapper<float> col4{1.5, 2.5, 3.5};
  cudf::test::fixed_width_column_wrapper<int16_t> col5{{1, 2, 3},
                                                       cudf::test::iterators::nulls_at({0, 2})};
  cudf::table_view tbl_view{{col1, col2, col3, col4, col5}};
  cudf::io::table_metadata mt{{{"col1"}, {"col2"}, {"int"}, {"float"}}};

  std::vector<char> out_buffer;
  auto destination = cudf::io::sink_info(&out_buffer);
  auto out_options = cudf::io::json_writer_options_builder(destination, tbl_view)
                       .include_nulls(true)
                       .metadata(mt)
                       .lines(false)
                       .na_rep("null")
                       .build();

  // not enough column names
  EXPECT_THROW(cudf::io::write_json(out_options, cudf::test::get_default_stream()),
               cudf::logic_error);

  mt.schema_info.emplace_back("int16");
  out_options.set_metadata(mt);
  EXPECT_NO_THROW(cudf::io::write_json(out_options, cudf::test::get_default_stream()));

  // chunk_rows must be at least 8
  out_options.set_rows_per_chunk(0);
  EXPECT_THROW(cudf::io::write_json(out_options, cudf::test::get_default_stream()),
               cudf::logic_error);
}

TEST_P(JsonCompressedWriterTest, PlainTable)
{
  cudf::io::compression_type const comptype = GetParam();
  cudf::test::strings_column_wrapper col1{"a", "b", "c"};
  cudf::test::strings_column_wrapper col2{"d", "e", "f"};
  cudf::test::fixed_width_column_wrapper<int64_t> col3{1, 2, 3};
  cudf::test::fixed_width_column_wrapper<double> col4{1.5, 2.5, 3.5};
  cudf::test::fixed_width_column_wrapper<int64_t> col5{{1, 2, 3},
                                                       cudf::test::iterators::nulls_at({0, 2})};
  cudf::table_view tbl_view{{col1, col2, col3, col4, col5}};
  cudf::io::table_metadata mt{{{"col1"}, {"col2"}, {"col3"}, {"col4"}, {"col5"}}};

  std::vector<char> out_buffer;
  auto destination = cudf::io::sink_info(&out_buffer);
  auto out_options = cudf::io::json_writer_options_builder(destination, tbl_view)
                       .include_nulls(true)
                       .metadata(mt)
                       .lines(false)
                       .compression(comptype)
                       .na_rep("null")
                       .build();

  std::string const expected =
    R"([{"col1":"a","col2":"d","col3":1,"col4":1.5,"col5":null},{"col1":"b","col2":"e","col3":2,"col4":2.5,"col5":2},{"col1":"c","col2":"f","col3":3,"col4":3.5,"col5":null}])";
  run_test(out_options, expected);
}

TEST_P(JsonCompressedWriterTest, SimpleNested)
{
  std::string const data = R"(
{"a": 1, "b": 2, "c": {"d": 3        }, "f": 5.5,  "g": [1]}
{"a": 6, "b": 7, "c": {"d": 8        }, "f": 10.5, "g": null}
{"a": 1, "b": 2, "c": {        "e": 4}, "f": 5.5,  "g": [2, null]}
{"a": 6, "b": 7, "c": {        "e": 9}, "f": 10.5, "g": [3, 4, 5]} )";
  cudf::io::json_reader_options in_options =
    cudf::io::json_reader_options::builder(
      cudf::io::source_info{cudf::host_span<std::byte const>{
        reinterpret_cast<std::byte const*>(data.data()), data.size()}})
      .lines(true);

  cudf::io::table_with_metadata result = cudf::io::read_json(in_options);
  cudf::table_view tbl_view            = result.tbl->view();
  cudf::io::table_metadata mt{result.metadata};

  std::vector<char> out_buffer;
  auto destination = cudf::io::sink_info(&out_buffer);
  auto out_options = cudf::io::json_writer_options_builder(destination, tbl_view)
                       .include_nulls(false)
                       .metadata(mt)
                       .lines(true)
                       .na_rep("null")
                       .build();

  std::string const expected = R"({"a":1,"b":2,"c":{"d":3},"f":5.5,"g":[1]}
{"a":6,"b":7,"c":{"d":8},"f":10.5}
{"a":1,"b":2,"c":{"e":4},"f":5.5,"g":[2,null]}
{"a":6,"b":7,"c":{"e":9},"f":10.5,"g":[3,4,5]}
)";
  run_test(out_options, expected);
}

TEST_P(JsonCompressedWriterTest, MixedNested)
{
  std::string const data = R"(
{"a": 1, "b": 2, "c": {"d": [3]      }, "f": 5.5,  "g": [ {"h": 1}]}
{"a": 6, "b": 7, "c": {"d": [8]      }, "f": 10.5, "g": null}
{"a": 1, "b": 2, "c": {        "e": 4}, "f": 5.5,  "g": [{"h": 2}, null]}
{"a": 6, "b": 7, "c": {        "e": 9}, "f": 10.5, "g": [{"h": 3}, {"h": 4}, {"h": 5}]} )";
  cudf::io::json_reader_options in_options =
    cudf::io::json_reader_options::builder(
      cudf::io::source_info{cudf::host_span<std::byte const>{
        reinterpret_cast<std::byte const*>(data.data()), data.size()}})
      .lines(true);

  cudf::io::table_with_metadata result = cudf::io::read_json(in_options);
  cudf::table_view tbl_view            = result.tbl->view();
  cudf::io::table_metadata mt{result.metadata};

  std::vector<char> out_buffer;
  auto destination = cudf::io::sink_info(&out_buffer);
  auto out_options = cudf::io::json_writer_options_builder(destination, tbl_view)
                       .include_nulls(false)
                       .metadata(mt)
                       .lines(false)
                       .na_rep("null")
                       .build();

  std::string const expected =
    R"([{"a":1,"b":2,"c":{"d":[3]},"f":5.5,"g":[{"h":1}]},)"
    R"({"a":6,"b":7,"c":{"d":[8]},"f":10.5},)"
    R"({"a":1,"b":2,"c":{"e":4},"f":5.5,"g":[{"h":2},null]},)"
    R"({"a":6,"b":7,"c":{"e":9},"f":10.5,"g":[{"h":3},{"h":4},{"h":5}]}])";
  run_test(out_options, expected);
}

TEST_F(JsonWriterTest, WriteReadNested)
{
  using namespace cudf::test::iterators;
  using LCW = cudf::test::lists_column_wrapper<int64_t>;
  cudf::test::fixed_width_column_wrapper<int> a{1, 6, 1, 6};
  cudf::test::fixed_width_column_wrapper<uint8_t> b{2, 7, 2, 7};
  cudf::test::fixed_width_column_wrapper<int64_t> d{{3, 8, 0, 0}, nulls_at({2, 3})};
  cudf::test::fixed_width_column_wrapper<int64_t> e{{0, 0, 4, 9}, nulls_at({0, 1})};
  cudf::test::structs_column_wrapper c{{d, e}};
  cudf::test::fixed_width_column_wrapper<float> f{5.5, 10.5, 5.5, 10.5};
  LCW g{{LCW{1}, LCW{0}, LCW{{2, 0}, null_at(1)}, LCW{3, 4, 5}}, null_at(1)};
  cudf::table_view tbl_view{{a, b, c, f, g}};
  cudf::io::table_metadata mt{{{"a"}, {"b"}, {"c"}, {"f"}, {"g"}}};
  mt.schema_info[2].children = {{"d"}, {"e"}};

  std::vector<char> out_buffer;
  auto destination = cudf::io::sink_info(&out_buffer);
  auto out_options = cudf::io::json_writer_options_builder(destination, tbl_view)
                       .include_nulls(false)
                       .metadata(mt)
                       .lines(true)
                       .na_rep("null")
                       .build();

  cudf::io::write_json(out_options, cudf::test::get_default_stream());
  std::string const expected = R"({"a":1,"b":2,"c":{"d":3},"f":5.5,"g":[1]}
{"a":6,"b":7,"c":{"d":8},"f":10.5}
{"a":1,"b":2,"c":{"e":4},"f":5.5,"g":[2,null]}
{"a":6,"b":7,"c":{"e":9},"f":10.5,"g":[3,4,5]}
)";
  auto const output_string   = std::string(out_buffer.data(), out_buffer.size());
  EXPECT_EQ(expected, output_string);

  // Read back the written JSON, and compare with the original table
  // Without type information
  auto in_options =
    cudf::io::json_reader_options::builder(
      cudf::io::source_info{cudf::host_span<std::byte const>{
        reinterpret_cast<std::byte const*>(output_string.data()), output_string.size()}})
      .lines(true)
      .build();

  auto result             = cudf::io::read_json(in_options);
  auto tbl_out            = result.tbl->view();
  auto const int64_dtype  = cudf::data_type{cudf::type_id::INT64};
  auto const double_dtype = cudf::data_type{cudf::type_id::FLOAT64};

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*cudf::cast(a, int64_dtype), tbl_out.column(0));
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*cudf::cast(b, int64_dtype), tbl_out.column(1));
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(c, tbl_out.column(2));
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*cudf::cast(f, double_dtype), tbl_out.column(3));
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(g, tbl_out.column(4));

  mt.schema_info[4].children = {{"offsets"}, {"element"}};  // list child column names
  EXPECT_EQ(mt.schema_info.size(), result.metadata.schema_info.size());
  for (auto i = 0UL; i < mt.schema_info.size(); i++) {
    EXPECT_EQ(mt.schema_info[i].name, result.metadata.schema_info[i].name) << "[" << i << "]";
    EXPECT_EQ(mt.schema_info[i].children.size(), result.metadata.schema_info[i].children.size())
      << "[" << i << "]";
    for (auto j = 0UL; j < mt.schema_info[i].children.size(); j++) {
      EXPECT_EQ(mt.schema_info[i].children[j].name, result.metadata.schema_info[i].children[j].name)
        << "[" << i << "][" << j << "]";
    }
  }

  // Read with type information
  std::map<std::string, cudf::io::schema_element> types;
  types["a"]                  = cudf::io::schema_element{cudf::data_type{cudf::type_id::INT32}};
  types["b"]                  = cudf::io::schema_element{cudf::data_type{cudf::type_id::UINT8}};
  types["c"]                  = cudf::io::schema_element{cudf::data_type{cudf::type_id::STRUCT}};
  types["c"].child_types["d"] = cudf::io::schema_element{cudf::data_type{cudf::type_id::INT64}};
  types["c"].child_types["e"] = cudf::io::schema_element{cudf::data_type{cudf::type_id::INT64}};
  types["f"]                  = cudf::io::schema_element{cudf::data_type{cudf::type_id::FLOAT32}};
  types["g"]                  = cudf::io::schema_element{cudf::data_type{cudf::type_id::LIST}};
  types["g"].child_types["element"] =
    cudf::io::schema_element{cudf::data_type{cudf::type_id::INT64}};

  in_options.set_dtypes(types);
  result  = cudf::io::read_json(in_options);
  tbl_out = result.tbl->view();

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(a, tbl_out.column(0));
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(b, tbl_out.column(1));
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(c, tbl_out.column(2));
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(f, tbl_out.column(3));
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(g, tbl_out.column(4));
  EXPECT_EQ(mt.schema_info.size(), result.metadata.schema_info.size());
  for (auto i = 0UL; i < mt.schema_info.size(); i++) {
    EXPECT_EQ(mt.schema_info[i].name, result.metadata.schema_info[i].name) << "[" << i << "]";
    EXPECT_EQ(mt.schema_info[i].children.size(), result.metadata.schema_info[i].children.size())
      << "[" << i << "]";
    for (auto j = 0UL; j < mt.schema_info[i].children.size(); j++) {
      EXPECT_EQ(mt.schema_info[i].children[j].name, result.metadata.schema_info[i].children[j].name)
        << "[" << i << "][" << j << "]";
    }
  }

  // Without children column names
  mt.schema_info[2].children.clear();
  out_options.set_metadata(mt);
  out_buffer.clear();
  cudf::io::write_json(out_options, cudf::test::get_default_stream());

  in_options = cudf::io::json_reader_options::builder(
                 cudf::io::source_info{cudf::host_span<std::byte const>{
                   reinterpret_cast<std::byte const*>(out_buffer.data()), out_buffer.size()}})
                 .lines(true)
                 .build();
  result = cudf::io::read_json(in_options);

  mt.schema_info[2].children = {{"0"}, {"1"}};
  EXPECT_EQ(mt.schema_info.size(), result.metadata.schema_info.size());
  for (auto i = 0UL; i < mt.schema_info.size(); i++) {
    EXPECT_EQ(mt.schema_info[i].name, result.metadata.schema_info[i].name) << "[" << i << "]";
    EXPECT_EQ(mt.schema_info[i].children.size(), result.metadata.schema_info[i].children.size())
      << "[" << i << "]";
    for (auto j = 0UL; j < mt.schema_info[i].children.size(); j++) {
      EXPECT_EQ(mt.schema_info[i].children[j].name, result.metadata.schema_info[i].children[j].name)
        << "[" << i << "][" << j << "]";
    }
  }

  // without column names
  out_options.set_metadata(cudf::io::table_metadata{});
  out_buffer.clear();
  cudf::io::write_json(out_options, cudf::test::get_default_stream());
  in_options = cudf::io::json_reader_options::builder(
                 cudf::io::source_info{cudf::host_span<std::byte const>{
                   reinterpret_cast<std::byte const*>(out_buffer.data()), out_buffer.size()}})
                 .lines(true)
                 .build();
  result = cudf::io::read_json(in_options);

  mt.schema_info             = {{"0"}, {"1"}, {"2"}, {"3"}, {"4"}};
  mt.schema_info[2].children = {{"0"}, {"1"}};
  mt.schema_info[4].children = {{"offsets"}, {"element"}};  // list child column names
  EXPECT_EQ(mt.schema_info.size(), result.metadata.schema_info.size());
  for (auto i = 0UL; i < mt.schema_info.size(); i++) {
    EXPECT_EQ(mt.schema_info[i].name, result.metadata.schema_info[i].name) << "[" << i << "]";
    EXPECT_EQ(mt.schema_info[i].children.size(), result.metadata.schema_info[i].children.size())
      << "[" << i << "]";
    for (auto j = 0UL; j < mt.schema_info[i].children.size(); j++) {
      EXPECT_EQ(mt.schema_info[i].children[j].name, result.metadata.schema_info[i].children[j].name)
        << "[" << i << "][" << j << "]";
    }
  }
}

TEST_P(JsonCompressedWriterTest, SpecialChars)
{
  cudf::test::fixed_width_column_wrapper<int> a{1, 6, 1, 6};
  cudf::test::strings_column_wrapper b{"abcd", "b\b\f\n\r\t", "\"c\"", "/\\"};
  cudf::table_view tbl_view{{a, b}};
  cudf::io::table_metadata mt{{{"\"a\""}, {"\'b\'"}}};

  std::vector<char> out_buffer;
  auto destination = cudf::io::sink_info(&out_buffer);
  auto out_options = cudf::io::json_writer_options_builder(destination, tbl_view)
                       .include_nulls(false)
                       .metadata(mt)
                       .lines(true)
                       .na_rep("null")
                       .build();

  std::string const expected = R"({"\"a\"":1,"'b'":"abcd"}
{"\"a\"":6,"'b'":"b\b\f\n\r\t"}
{"\"a\"":1,"'b'":"\"c\""}
{"\"a\"":6,"'b'":"\/\\"}
)";
  run_test(out_options, expected);
}

TEST_P(JsonCompressedWriterTest, NullList)
{
  std::string const data = R"(
{"a": [null], "b": [[1, 2, 3], [null], [null, null, null], [4, null, 5]]}
{"a": [2, null, null, 3] , "b": null}
{"a": [null, null, 4], "b": [[2, null], null]}
{"a": [5, null, null], "b": [null, [3, 4, 5]]} )";
  cudf::io::json_reader_options in_options =
    cudf::io::json_reader_options::builder(
      cudf::io::source_info{cudf::host_span<std::byte const>{
        reinterpret_cast<std::byte const*>(data.data()), data.size()}})
      .lines(true);

  cudf::io::table_with_metadata result = cudf::io::read_json(in_options);
  cudf::table_view tbl_view            = result.tbl->view();
  cudf::io::table_metadata mt{result.metadata};

  std::vector<char> out_buffer;
  auto destination = cudf::io::sink_info(&out_buffer);
  auto out_options = cudf::io::json_writer_options_builder(destination, tbl_view)
                       .include_nulls(true)
                       .metadata(mt)
                       .lines(true)
                       .na_rep("null")
                       .build();

  std::string const expected = R"({"a":[null],"b":[[1,2,3],[null],[null,null,null],[4,null,5]]}
{"a":[2,null,null,3],"b":null}
{"a":[null,null,4],"b":[[2,null],null]}
{"a":[5,null,null],"b":[null,[3,4,5]]}
)";
  run_test(out_options, expected);
}

TEST_P(JsonCompressedWriterTest, ChunkedNested)
{
  std::string const data = R"(
{"a": 1, "b": -2, "c": {      }, "e": [{"f": 1}]}
{"a": 2, "b": -2, "c": {      }, "e": null}
{"a": 3, "b": -2, "c": {"d": 9}, "e": [{"f": 2}, null]}
{"a": 4, "b": -2, "c": {"d": 16}, "e": [{"f": 3}, {"f": 4}, {"f": 5}]}
{"a": 5, "b": -2, "c": {      }, "e": []}
{"a": 6, "b": -2, "c": {"d": 36}, "e": [{"f": 6}]}
{"a": 7, "b": -2, "c": {"d": 49}, "e": [{"f": 7}]}
{"a": 8, "b": -2, "c": {"d": 64}, "e": [{"f": 8}]}
{"a": 9, "b": -2, "c": {"d": 81}, "e": [{"f": 9}]}
)";
  cudf::io::json_reader_options in_options =
    cudf::io::json_reader_options::builder(
      cudf::io::source_info{cudf::host_span<std::byte const>{
        reinterpret_cast<std::byte const*>(data.data()), data.size()}})
      .lines(true);

  cudf::io::table_with_metadata result = cudf::io::read_json(in_options);
  cudf::table_view tbl_view            = result.tbl->view();
  cudf::io::table_metadata mt{result.metadata};

  std::vector<char> out_buffer;
  auto destination = cudf::io::sink_info(&out_buffer);
  auto out_options = cudf::io::json_writer_options_builder(destination, tbl_view)
                       .include_nulls(false)
                       .metadata(mt)
                       .lines(true)
                       .na_rep("null")
                       .rows_per_chunk(8)
                       .build();

  std::string const expected =
    R"({"a":1,"b":-2,"c":{},"e":[{"f":1}]}
{"a":2,"b":-2,"c":{}}
{"a":3,"b":-2,"c":{"d":9},"e":[{"f":2},null]}
{"a":4,"b":-2,"c":{"d":16},"e":[{"f":3},{"f":4},{"f":5}]}
{"a":5,"b":-2,"c":{},"e":[]}
{"a":6,"b":-2,"c":{"d":36},"e":[{"f":6}]}
{"a":7,"b":-2,"c":{"d":49},"e":[{"f":7}]}
{"a":8,"b":-2,"c":{"d":64},"e":[{"f":8}]}
{"a":9,"b":-2,"c":{"d":81},"e":[{"f":9}]}
)";
  run_test(out_options, expected);
}

TEST_P(JsonCompressedWriterTest, StructAllNullCombinations)
{
  auto const_1_iter = cuda::make_constant_iterator(1);

  auto col_a = cudf::test::fixed_width_column_wrapper<int>(
    const_1_iter, const_1_iter + 32, cudf::detail::make_counting_transform_iterator(0, [](auto i) {
      return i / 16;
    }));

  auto col_b = cudf::test::fixed_width_column_wrapper<int>(
    const_1_iter, const_1_iter + 32, cudf::detail::make_counting_transform_iterator(0, [](auto i) {
      return (i / 8) % 2;
    }));

  auto col_c = cudf::test::fixed_width_column_wrapper<int>(
    const_1_iter, const_1_iter + 32, cudf::detail::make_counting_transform_iterator(0, [](auto i) {
      return (i / 4) % 2;
    }));

  auto col_d = cudf::test::fixed_width_column_wrapper<int>(
    const_1_iter, const_1_iter + 32, cudf::detail::make_counting_transform_iterator(0, [](auto i) {
      return (i / 2) % 2;
    }));

  auto col_e = cudf::test::fixed_width_column_wrapper<int>(
    const_1_iter, const_1_iter + 32, cudf::detail::make_counting_transform_iterator(0, [](auto i) {
      return i % 2;
    }));

  // The table has 32 rows with validity from 00000 to 11111
  cudf::table_view tbl_view = cudf::table_view({col_a, col_b, col_c, col_d, col_e});
  cudf::io::table_metadata mt{{{"a"}, {"b"}, {"c"}, {"d"}, {"e"}}};

  std::vector<char> out_buffer;
  auto destination = cudf::io::sink_info(&out_buffer);
  auto out_options = cudf::io::json_writer_options_builder(destination, tbl_view)
                       .include_nulls(false)
                       .metadata(mt)
                       .lines(true)
                       .na_rep("null")
                       .build();

  std::string const expected = R"({}
{"e":1}
{"d":1}
{"d":1,"e":1}
{"c":1}
{"c":1,"e":1}
{"c":1,"d":1}
{"c":1,"d":1,"e":1}
{"b":1}
{"b":1,"e":1}
{"b":1,"d":1}
{"b":1,"d":1,"e":1}
{"b":1,"c":1}
{"b":1,"c":1,"e":1}
{"b":1,"c":1,"d":1}
{"b":1,"c":1,"d":1,"e":1}
{"a":1}
{"a":1,"e":1}
{"a":1,"d":1}
{"a":1,"d":1,"e":1}
{"a":1,"c":1}
{"a":1,"c":1,"e":1}
{"a":1,"c":1,"d":1}
{"a":1,"c":1,"d":1,"e":1}
{"a":1,"b":1}
{"a":1,"b":1,"e":1}
{"a":1,"b":1,"d":1}
{"a":1,"b":1,"d":1,"e":1}
{"a":1,"b":1,"c":1}
{"a":1,"b":1,"c":1,"e":1}
{"a":1,"b":1,"c":1,"d":1}
{"a":1,"b":1,"c":1,"d":1,"e":1}
)";
  run_test(out_options, expected);
}

TEST_P(JsonCompressedWriterTest, Unicode)
{
  //                                       UTF-8,                      UTF-16
  cudf::test::strings_column_wrapper col1{"\"\\/\b\f\n\r\t", "ராபிட்ஸ்", "$€𐐷𤭢", "C𝞵𝓓𝒻"};
  // Unicode
  // 0000-FFFF     Basic Multilingual Plane
  // 10000-10FFFF  Supplementary Plane
  cudf::test::strings_column_wrapper col2{
    "CႮ≪ㇳ䍏凹沦王辿龸ꁗ믜스폶ﴠ",  //  0000-FFFF
    "𐀀𑿪𒐦𓃰𔙆 𖦆𗿿𘳕𚿾[↳] 𜽆𝓚𞤁🄰",                            // 10000-1FFFF
    "𠘨𡥌𢗉𣇊𤊩𥅽𦉱𧴱𨁲𩁹𪐢𫇭𬬭𭺷𮊦屮",                // 20000-2FFFF
    "𰾑𱔈𲍉"};                                         // 30000-3FFFF
  cudf::test::fixed_width_column_wrapper<int16_t> col3{{1, 2, 3, 4},
                                                       cudf::test::iterators::nulls_at({0, 2})};
  cudf::table_view tbl_view{{col1, col2, col3}};
  cudf::io::table_metadata mt{{{"col1"}, {"col2"}, {"int16"}}};

  std::vector<char> out_buffer;
  auto destination = cudf::io::sink_info(&out_buffer);
  auto out_options = cudf::io::json_writer_options_builder(destination, tbl_view)
                       .include_nulls(true)
                       .metadata(mt)
                       .lines(true)
                       .na_rep("null")
                       .build();

  std::string const expected =
    R"({"col1":"\"\\\/\b\f\n\r\t","col2":"C\u10ae\u226a\u31f3\u434f\u51f9\u6ca6\u738b\u8fbf\u9fb8\ua057\ubbdc\uc2a4\ud3f6\ue4fe\ufd20","int16":null}
{"col1":"\u0bb0\u0bbe\u0baa\u0bbf\u0b9f\u0bcd\u0bb8\u0bcd","col2":"\ud800\udc00\ud807\udfea\ud809\udc26\ud80c\udcf0\ud811\ude46 \ud81a\udd86\ud81f\udfff\ud823\udcd5\ud82b\udffe[\u21b3] \ud833\udf46\ud835\udcda\ud83a\udd01\ud83c\udd30","int16":2}
{"col1":"$\u20ac\ud801\udc37\ud852\udf62","col2":"\ud841\ude28\ud846\udd4c\ud849\uddc9\ud84c\uddca\ud850\udea9\ud854\udd7d\ud858\ude71\ud85f\udd31\ud860\udc72\ud864\udc79\ud869\udc22\ud86c\udded\ud872\udf2d\ud877\udeb7\ud878\udea6\u5c6e","int16":null}
{"col1":"C\ud835\udfb5\ud835\udcd3\ud835\udcbb","col2":"\ud883\udf91\ud885\udd08\ud888\udf49","int16":4}
)";
  run_test(out_options, expected);
}

TEST_P(JsonCompressedWriterTest, UnicodeUnescaped)
{
  cudf::test::strings_column_wrapper col1{"\"\\/\b\f\n\r\t", "ராபிட்ஸ்", "$€𐐷𤭢", "C𝞵𝓓𝒻"};
  cudf::test::strings_column_wrapper col2{"CႮ≪ㇳ䍏凹沦王辿龸ꁗ믜스폶ﴠ",
                                          "𐀀𑿪𒐦𓃰𔙆 𖦆𗿿𘳕𚿾[↳] 𜽆𝓚𞤁🄰",
                                          "𠘨𡥌𢗉𣇊𤊩𥅽𦉱𧴱𨁲𩁹𪐢𫇭𬬭𭺷𮊦屮",
                                          "𰾑𱔈𲍉"};
  cudf::test::fixed_width_column_wrapper<int16_t> col3{{1, 2, 3, 4},
                                                       cudf::test::iterators::nulls_at({0, 2})};
  cudf::table_view tbl_view{{col1, col2, col3}};
  cudf::io::table_metadata mt{{{"col1"}, {"col2"}, {"int16"}}};

  std::vector<char> out_buffer;
  auto destination = cudf::io::sink_info(&out_buffer);
  auto out_options = cudf::io::json_writer_options_builder(destination, tbl_view)
                       .include_nulls(true)
                       .metadata(mt)
                       .lines(true)
                       .na_rep("null")
                       .utf8_escaped(false)
                       .build();

  cudf::io::write_json(out_options, cudf::test::get_default_stream());

  std::string const expected =
    R"({"col1":"\"\\\/\b\f\n\r\t","col2":"CႮ≪ㇳ䍏凹沦王辿龸ꁗ믜스폶ﴠ","int16":null}
{"col1":"ராபிட்ஸ்","col2":"𐀀𑿪𒐦𓃰𔙆 𖦆𗿿𘳕𚿾[↳] 𜽆𝓚𞤁🄰","int16":2}
{"col1":"$€𐐷𤭢","col2":"𠘨𡥌𢗉𣇊𤊩𥅽𦉱𧴱𨁲𩁹𪐢𫇭𬬭𭺷𮊦屮","int16":null}
{"col1":"C𝞵𝓓𝒻","col2":"𰾑𱔈𲍉","int16":4}
)";

  std::string const output(out_buffer.data(), out_buffer.size());
  EXPECT_EQ(expected, output);
}

struct JsonWriterTypeSupportTest : public cudf::test::BaseFixture {};

TEST(JsonWriterTypeSupportTest, SupportedTypes)
{
  using cudf::io::is_supported_write_json;

  EXPECT_TRUE(is_supported_write_json(cudf::data_type{cudf::type_id::INT32}));
  EXPECT_TRUE(is_supported_write_json(cudf::data_type{cudf::type_id::FLOAT64}));
  EXPECT_TRUE(is_supported_write_json(cudf::data_type{cudf::type_id::STRING}));
  EXPECT_TRUE(is_supported_write_json(cudf::data_type{cudf::type_id::DECIMAL64}));
  EXPECT_TRUE(is_supported_write_json(cudf::data_type{cudf::type_id::TIMESTAMP_NANOSECONDS}));
  EXPECT_TRUE(is_supported_write_json(cudf::data_type{cudf::type_id::DURATION_SECONDS}));
}

TEST(JsonWriterTypeSupportTest, UnsupportedTypes)
{
  using cudf::io::is_supported_write_json;

  EXPECT_FALSE(is_supported_write_json(cudf::data_type{cudf::type_id::LIST}));
  EXPECT_FALSE(is_supported_write_json(cudf::data_type{cudf::type_id::STRUCT}));
  EXPECT_FALSE(is_supported_write_json(cudf::data_type{cudf::type_id::DICTIONARY32}));
}

CUDF_TEST_PROGRAM_MAIN()
