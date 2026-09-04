/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/testing_main.hpp>

#include <cudf/json/json.hpp>
#include <cudf/strings/replace.hpp>
#include <cudf/strings/strings_column_view.hpp>

#include <stdexcept>

// reference:  https://jsonpath.herokuapp.com/

// clang-format off
std::string json_string{
  "{"
    "\"store\": {""\"book\": ["
        "{"
          "\"category\": \"reference\","
          "\"author\": \"Nigel Rees\","
          "\"title\": \"Sayings of the Century\","
          "\"price\": 8.95"
        "},"
        "{"
          "\"category\": \"fiction\","
          "\"author\": \"Evelyn Waugh\","
          "\"title\": \"Sword of Honour\","
          "\"price\": 12.99"
        "},"
        "{"
          "\"category\": \"fiction\","
          "\"author\": \"Herman Melville\","
          "\"title\": \"Moby Dick\","
          "\"isbn\": \"0-553-21311-3\","
          "\"price\": 8.99"
        "},"
        "{"
          "\"category\": \"fiction\","
          "\"author\": \"J. R. R. Tolkien\","
          "\"title\": \"The Lord of the Rings\","
          "\"isbn\": \"0-395-19395-8\","
          "\"price\": 22.99"
        "}"
      "],"
      "\"bicycle\": {"
        "\"color\": \"red\","
        "\"price\": 19.95"
      "}"
    "},"
    "\"expensive\": 10"
  "}"
};
// clang-format on

std::unique_ptr<cudf::column> drop_whitespace(cudf::column_view const& col)
{
  cudf::test::strings_column_wrapper whitespace{"\n", "\r", "\t"};
  cudf::test::strings_column_wrapper repl{"", "", ""};

  cudf::strings_column_view strings(col);
  cudf::strings_column_view targets(whitespace);
  cudf::strings_column_view replacements(repl);
  return cudf::strings::replace_multiple(strings, targets, replacements);
}

struct JsonPathTests : public cudf::test::BaseFixture {};

TEST_F(JsonPathTests, GetJsonObjectRootOp)
{
  // root
  cudf::test::strings_column_wrapper input{json_string};
  std::string_view json_path("$");
  auto result_raw = cudf::get_json_object(cudf::strings_column_view(input), json_path);
  auto result     = drop_whitespace(*result_raw);

  auto expected = drop_whitespace(input);

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*result, *expected);
}

TEST_F(JsonPathTests, GetJsonObjectChildOp)
{
  {
    cudf::test::strings_column_wrapper input{json_string};
    std::string_view json_path("$.store");
    auto result_raw = cudf::get_json_object(cudf::strings_column_view(input), json_path);
    auto result     = drop_whitespace(*result_raw);

    // clang-format off
    cudf::test::strings_column_wrapper expected_raw{
      "{"
        "\"book\": ["
          "{"
            "\"category\": \"reference\","
            "\"author\": \"Nigel Rees\","
            "\"title\": \"Sayings of the Century\","
            "\"price\": 8.95"
          "},"
          "{"
            "\"category\": \"fiction\","
            "\"author\": \"Evelyn Waugh\","
            "\"title\": \"Sword of Honour\","
            "\"price\": 12.99"
          "},"
          "{"
            "\"category\": \"fiction\","
            "\"author\": \"Herman Melville\","
            "\"title\": \"Moby Dick\","
            "\"isbn\": \"0-553-21311-3\","
            "\"price\": 8.99"
          "},"
          "{"
            "\"category\": \"fiction\","
            "\"author\": \"J. R. R. Tolkien\","
            "\"title\": \"The Lord of the Rings\","
            "\"isbn\": \"0-395-19395-8\","
            "\"price\": 22.99"
          "}"
        "],"
        "\"bicycle\": {"
          "\"color\": \"red\","
          "\"price\": 19.95"
        "}"
      "}"
    };
    // clang-format on
    auto expected = drop_whitespace(expected_raw);

    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*result, *expected);
  }

  {
    cudf::test::strings_column_wrapper input{json_string};
    std::string_view json_path("$.store.book");
    auto result_raw = cudf::get_json_object(cudf::strings_column_view(input), json_path);
    auto result     = drop_whitespace(*result_raw);

    // clang-format off
    cudf::test::strings_column_wrapper expected_raw{
      "["
        "{"
          "\"category\": \"reference\","
          "\"author\": \"Nigel Rees\","
          "\"title\": \"Sayings of the Century\","
          "\"price\": 8.95"
        "},"
        "{"
          "\"category\": \"fiction\","
          "\"author\": \"Evelyn Waugh\","
          "\"title\": \"Sword of Honour\","
          "\"price\": 12.99"
        "},"
        "{"
          "\"category\": \"fiction\","
          "\"author\": \"Herman Melville\","
          "\"title\": \"Moby Dick\","
          "\"isbn\": \"0-553-21311-3\","
          "\"price\": 8.99"
        "},"
        "{"
          "\"category\": \"fiction\","
          "\"author\": \"J. R. R. Tolkien\","
          "\"title\": \"The Lord of the Rings\","
          "\"isbn\": \"0-395-19395-8\","
          "\"price\": 22.99"
        "}"
      "]"
    };
    // clang-format on
    auto expected = drop_whitespace(expected_raw);

    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*result, *expected);
  }
}

TEST_F(JsonPathTests, GetJsonObjectWildcardOp)
{
  {
    cudf::test::strings_column_wrapper input{json_string};
    std::string_view json_path("$.store.*");
    auto result_raw = cudf::get_json_object(cudf::strings_column_view(input), json_path);
    auto result     = drop_whitespace(*result_raw);

    // clang-format off
    cudf::test::strings_column_wrapper expected_raw{
      "["
        "["
          "{"
            "\"category\": \"reference\","
            "\"author\": \"Nigel Rees\","
            "\"title\": \"Sayings of the Century\","
            "\"price\": 8.95"
          "},"
          "{"
            "\"category\": \"fiction\","
            "\"author\": \"Evelyn Waugh\","
            "\"title\": \"Sword of Honour\","
            "\"price\": 12.99"
          "},"
          "{"
            "\"category\": \"fiction\","
            "\"author\": \"Herman Melville\","
            "\"title\": \"Moby Dick\","
            "\"isbn\": \"0-553-21311-3\","
            "\"price\": 8.99"
          "},"
          "{"
            "\"category\": \"fiction\","
            "\"author\": \"J. R. R. Tolkien\","
            "\"title\": \"The Lord of the Rings\","
            "\"isbn\": \"0-395-19395-8\","
            "\"price\": 22.99"
          "}"
        "],"
        "{"
          "\"color\": \"red\","
          "\"price\": 19.95"
        "}"
      "]"
    };
    // clang-format on
    auto expected = drop_whitespace(expected_raw);

    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*result, *expected);
  }

  {
    cudf::test::strings_column_wrapper input{json_string};
    std::string_view json_path("*");
    auto result_raw = cudf::get_json_object(cudf::strings_column_view(input), json_path);
    auto result     = drop_whitespace(*result_raw);

    // clang-format off
    cudf::test::strings_column_wrapper expected_raw{
      "["
        "{"
          "\"book\": ["
            "{"
              "\"category\": \"reference\","
              "\"author\": \"Nigel Rees\","
              "\"title\": \"Sayings of the Century\","
              "\"price\": 8.95"
            "},"
            "{"
              "\"category\": \"fiction\","
              "\"author\": \"Evelyn Waugh\","
              "\"title\": \"Sword of Honour\","
              "\"price\": 12.99"
            "},"
            "{"
              "\"category\": \"fiction\","
              "\"author\": \"Herman Melville\","
              "\"title\": \"Moby Dick\","
              "\"isbn\": \"0-553-21311-3\","
              "\"price\": 8.99"
            "},"
            "{"
              "\"category\": \"fiction\","
              "\"author\": \"J. R. R. Tolkien\","
              "\"title\": \"The Lord of the Rings\","
              "\"isbn\": \"0-395-19395-8\","
              "\"price\": 22.99"
            "}"
          "],"
          "\"bicycle\": {"
            "\"color\": \"red\","
            "\"price\": 19.95"
          "}"
        "},"
        "10"
      "]"
    };
    // clang-format on
    auto expected = drop_whitespace(expected_raw);

    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*result, *expected);
  }
}

TEST_F(JsonPathTests, GetJsonObjectSubscriptOp)
{
  {
    cudf::test::strings_column_wrapper input{json_string};
    std::string_view json_path("$.store.book[2]");
    auto result_raw = cudf::get_json_object(cudf::strings_column_view(input), json_path);
    auto result     = drop_whitespace(*result_raw);

    // clang-format off
    cudf::test::strings_column_wrapper expected_raw{
      "{"
        "\"category\": \"fiction\","
        "\"author\": \"Herman Melville\","
        "\"title\": \"Moby Dick\","
        "\"isbn\": \"0-553-21311-3\","
        "\"price\": 8.99"
      "}"
    };
    // clang-format on
    auto expected = drop_whitespace(expected_raw);

    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*result, *expected);
  }

  {
    cudf::test::strings_column_wrapper input{json_string};
    std::string_view json_path("$.store['bicycle']");
    auto result_raw = cudf::get_json_object(cudf::strings_column_view(input), json_path);
    auto result     = drop_whitespace(*result_raw);

    // clang-format off
    cudf::test::strings_column_wrapper expected_raw{
      "{"
        "\"color\": \"red\","
        "\"price\": 19.95"
      "}"
    };
    // clang-format on
    auto expected = drop_whitespace(expected_raw);

    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*result, *expected);
  }

  {
    cudf::test::strings_column_wrapper input{json_string};
    std::string_view json_path("$.store.book[*]");
    auto result_raw = cudf::get_json_object(cudf::strings_column_view(input), json_path);
    auto result     = drop_whitespace(*result_raw);

    // clang-format off
    cudf::test::strings_column_wrapper expected_raw{
      "["
        "{"
          "\"category\": \"reference\","
          "\"author\": \"Nigel Rees\","
          "\"title\": \"Sayings of the Century\","
          "\"price\": 8.95"
        "},"
        "{"
          "\"category\": \"fiction\","
          "\"author\": \"Evelyn Waugh\","
          "\"title\": \"Sword of Honour\","
          "\"price\": 12.99"
        "},"
        "{"
          "\"category\": \"fiction\","
          "\"author\": \"Herman Melville\","
          "\"title\": \"Moby Dick\","
          "\"isbn\": \"0-553-21311-3\","
          "\"price\": 8.99"
        "},"
        "{"
          "\"category\": \"fiction\","
          "\"author\": \"J. R. R. Tolkien\","
          "\"title\": \"The Lord of the Rings\","
          "\"isbn\": \"0-395-19395-8\","
          "\"price\": 22.99"
        "}"
      "]"
    };
    // clang-format on
    auto expected = drop_whitespace(expected_raw);

    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*result, *expected);
  }
}

TEST_F(JsonPathTests, GetJsonObjectFilter)
{
  // queries that result in filtering/collating results (mostly meaning - generates new
  // json instead of just returning parts of the existing string

  {
    cudf::test::strings_column_wrapper input{json_string};
    std::string_view json_path("$.store.book[*]['isbn']");
    auto result_raw = cudf::get_json_object(cudf::strings_column_view(input), json_path);
    auto result     = drop_whitespace(*result_raw);

    cudf::test::strings_column_wrapper expected_raw{R"(["0-553-21311-3","0-395-19395-8"])"};
    auto expected = drop_whitespace(expected_raw);

    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*result, *expected);
  }

  {
    cudf::test::strings_column_wrapper input{json_string};
    std::string_view json_path("$.store.book[*].category");
    auto result_raw = cudf::get_json_object(cudf::strings_column_view(input), json_path);
    auto result     = drop_whitespace(*result_raw);

    cudf::test::strings_column_wrapper expected_raw{
      R"(["reference","fiction","fiction","fiction"])"};
    auto expected = drop_whitespace(expected_raw);

    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*result, *expected);
  }

  {
    cudf::test::strings_column_wrapper input{json_string};
    std::string_view json_path("$.store.book[*].title");
    auto result_raw = cudf::get_json_object(cudf::strings_column_view(input), json_path);
    auto result     = drop_whitespace(*result_raw);

    cudf::test::strings_column_wrapper expected_raw{
      R"(["Sayings of the Century","Sword of Honour","Moby Dick","The Lord of the Rings"])"};
    auto expected = drop_whitespace(expected_raw);

    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*result, *expected);
  }

  {
    cudf::test::strings_column_wrapper input{json_string};
    std::string_view json_path("$.store.book.*.price");
    auto result_raw = cudf::get_json_object(cudf::strings_column_view(input), json_path);
    auto result     = drop_whitespace(*result_raw);

    cudf::test::strings_column_wrapper expected_raw{"[8.95,12.99,8.99,22.99]"};
    auto expected = drop_whitespace(expected_raw);

    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*result, *expected);
  }

  {
    // spark behavioral difference.
    //  standard:     "fiction"
    //  spark:        fiction
    cudf::test::strings_column_wrapper input{json_string};
    std::string_view json_path("$.store.book[2].category");
    auto result_raw = cudf::get_json_object(cudf::strings_column_view(input), json_path);
    auto result     = drop_whitespace(*result_raw);

    cudf::test::strings_column_wrapper expected_raw{"fiction"};
    auto expected = drop_whitespace(expected_raw);

    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*result, *expected);
  }
}

TEST_F(JsonPathTests, GetJsonObjectNullInputs)
{
  {
    std::string str(R"({"a" : "b"})");
    cudf::test::strings_column_wrapper input({str, "", str, ""}, {true, false, true, false});

    std::string_view json_path("$.a");
    auto result_raw = cudf::get_json_object(cudf::strings_column_view(input), json_path);
    auto result     = drop_whitespace(*result_raw);

    cudf::test::strings_column_wrapper expected_raw({"b", "", "b", ""}, {1, 0, 1, 0});
    auto expected = drop_whitespace(expected_raw);

    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*result, *expected);
  }
}

TEST_F(JsonPathTests, GetJsonObjectEmptyQuery)
{
  // empty query -> null
  {
    cudf::test::strings_column_wrapper input{R"({"a" : "b"})"};
    std::string_view json_path("");
    auto result = cudf::get_json_object(cudf::strings_column_view(input), json_path);

    cudf::test::strings_column_wrapper expected({""}, {0});

    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*result, expected);
  }
}

TEST_F(JsonPathTests, GetJsonObjectEmptyInputsAndOutputs)
{
  // empty string input -> null
  {
    cudf::test::strings_column_wrapper input{""};
    std::string_view json_path("$");
    auto result = cudf::get_json_object(cudf::strings_column_view(input), json_path);

    cudf::test::strings_column_wrapper expected({""}, {0});

    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*result, expected);
  }

  // slightly different from "empty output". in this case, we're
  // returning something, but it happens to be empty. so we expect
  // a valid, but empty row
  {
    cudf::test::strings_column_wrapper input{R"({"store": { "bicycle" : "" } })"};
    std::string_view json_path("$.store.bicycle");
    auto result = cudf::get_json_object(cudf::strings_column_view(input), json_path);

    cudf::test::strings_column_wrapper expected({""}, {1});

    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*result, expected);
  }
}

TEST_F(JsonPathTests, GetJsonObjectEmptyInput)
{
  cudf::test::strings_column_wrapper input{};
  std::string_view json_path("$");
  auto result = cudf::get_json_object(cudf::strings_column_view(input), json_path);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*result, input);
}

// badly formed JSONpath strings
TEST_F(JsonPathTests, GetJsonObjectIllegalQuery)
{
  // can't have more than one root operator, or a root operator anywhere other
  // than the beginning
  {
    cudf::test::strings_column_wrapper input{R"({"a": "b"})"};
    std::string_view json_path("$$");
    auto query = [&]() {
      auto result = cudf::get_json_object(cudf::strings_column_view(input), json_path);
    };
    EXPECT_THROW(query(), cudf::logic_error);
  }

  // invalid index
  {
    cudf::test::strings_column_wrapper input{R"({"a": "b"})"};
    std::string_view json_path("$[auh46h-]");
    auto query = [&]() {
      auto result = cudf::get_json_object(cudf::strings_column_view(input), json_path);
    };
    EXPECT_THROW(query(), cudf::logic_error);
  }

  // invalid index
  {
    cudf::test::strings_column_wrapper input{R"({"a": "b"})"};
    std::string_view json_path("$[[]]");
    auto query = [&]() {
      auto result = cudf::get_json_object(cudf::strings_column_view(input), json_path);
    };
    EXPECT_THROW(query(), cudf::logic_error);
  }

  // negative index
  {
    cudf::test::strings_column_wrapper input{R"({"a": "b"})"};
    std::string_view json_path("$[-1]");
    auto query = [&]() {
      auto result = cudf::get_json_object(cudf::strings_column_view(input), json_path);
    };
    EXPECT_THROW(query(), cudf::logic_error);
  }

  // child operator with no name specified
  {
    cudf::test::strings_column_wrapper input{R"({"a": "b"})"};
    std::string_view json_path(".");
    auto query = [&]() {
      auto result = cudf::get_json_object(cudf::strings_column_view(input), json_path);
    };
    EXPECT_THROW(query(), std::invalid_argument);
  }

  {
    cudf::test::strings_column_wrapper input{R"({"a": "b"})"};
    std::string_view json_path("][");
    auto query = [&]() {
      auto result = cudf::get_json_object(cudf::strings_column_view(input), json_path);
    };
    EXPECT_THROW(query(), std::invalid_argument);
  }

  {
    cudf::test::strings_column_wrapper input{R"({"a": "b"})"};
    std::string_view json_path("6hw6,56i3");
    auto query = [&]() {
      auto result = cudf::get_json_object(cudf::strings_column_view(input), json_path);
    };
    EXPECT_THROW(query(), std::invalid_argument);
  }

  {
    auto const input     = cudf::test::strings_column_wrapper{R"({"a": "b"})"};
    auto const json_path = std::string_view{"${a}"};
    auto const query     = [&]() {
      auto const result = cudf::get_json_object(cudf::strings_column_view(input), json_path);
    };
    EXPECT_THROW(query(), std::invalid_argument);
  }
}

// queries that are legal, but reference invalid parts of the input
TEST_F(JsonPathTests, GetJsonObjectInvalidQuery)
{
  // non-existent field
  {
    cudf::test::strings_column_wrapper input{R"({"a": "b"})"};
    std::string_view json_path("$[*].c");
    auto result = cudf::get_json_object(cudf::strings_column_view(input), json_path);

    cudf::test::strings_column_wrapper expected({""}, {0});

    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*result, expected);
  }

  // non-existent field
  {
    cudf::test::strings_column_wrapper input{R"({"a": "b"})"};
    std::string_view json_path("$[*].c[2]");
    auto result = cudf::get_json_object(cudf::strings_column_view(input), json_path);

    cudf::test::strings_column_wrapper expected({""}, {0});

    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*result, expected);
  }

  // non-existent field
  {
    cudf::test::strings_column_wrapper input{json_string};
    std::string_view json_path("$.store.book.price");
    auto result = cudf::get_json_object(cudf::strings_column_view(input), json_path);

    cudf::test::strings_column_wrapper expected({""}, {0});

    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*result, expected);
  }

  // out of bounds index
  {
    cudf::test::strings_column_wrapper input{json_string};
    std::string_view json_path("$.store.book[4]");
    auto result = cudf::get_json_object(cudf::strings_column_view(input), json_path);

    cudf::test::strings_column_wrapper expected({""}, {0});

    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*result, expected);
  }
}

TEST_F(JsonPathTests, MixedOutput)
{
  // various queries on:
  // clang-format off
  std::vector<std::string> input_strings {
    R"({"a": {"b" : "c"}})",

    "{"
      "\"a\": {\"b\" : \"c\"},"
      "\"d\": [{\"e\":123}, {\"f\":-10}]"
    "}",

    "{"
      "\"b\": 123"
    "}",

    "{"
      "\"a\": [\"y\",500]"
    "}",

    "{"
      "\"a\": \"\""
    "}",

    "{"
      "\"a\": {"
                "\"z\": {\"i\": 10, \"j\": 100},"
                "\"b\": [\"c\",null,true,-1]"
              "}"
    "}"
  };
  // clang-format on
  cudf::test::strings_column_wrapper input(input_strings.begin(), input_strings.end());
  {
    std::string_view json_path("$.a");
    auto result = cudf::get_json_object(cudf::strings_column_view(input), json_path);

    // clang-format off
    cudf::test::strings_column_wrapper expected({
      R"({"b" : "c"})",
      R"({"b" : "c"})",
      "",
      "[\"y\",500]",
      "",
      "{"
         "\"z\": {\"i\": 10, \"j\": 100},"
         "\"b\": [\"c\",null,true,-1]"
      "}"
      },
      {1, 1, 0, 1, 1, 1});
    // clang-format on

    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*result, expected);
  }

  {
    std::string_view json_path("$.a[1]");
    auto result = cudf::get_json_object(cudf::strings_column_view(input), json_path);

    // clang-format off
    cudf::test::strings_column_wrapper expected({
        "",
        "",
        "",
        "500",
        "",
        "",
      },
      {0, 0, 0, 1, 0, 0});
    // clang-format on

    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*result, expected);
  }

  {
    std::string_view json_path("$.a.b");
    auto result = cudf::get_json_object(cudf::strings_column_view(input), json_path);

    // clang-format off
    cudf::test::strings_column_wrapper expected({
      "c",
      "c",
      "",
      "",
      "",
      "[\"c\",null,true,-1]"},
      {1, 1, 0, 0, 0, 1});
    // clang-format on

    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*result, expected);
  }

  {
    std::string_view json_path("$.a[*]");
    auto result = cudf::get_json_object(cudf::strings_column_view(input), json_path);

    // clang-format off
    cudf::test::strings_column_wrapper expected({
      "[\"c\"]",
      "[\"c\"]",
      "",
      "[\"y\",500]",
      "[]",
      "["
        "{\"i\": 10, \"j\": 100},"
        "[\"c\",null,true,-1]"
      "]" },
      {1, 1, 0, 1, 1, 1});
    // clang-format on

    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*result, expected);
  }

  {
    std::string_view json_path("$.a.b[*]");
    auto result = cudf::get_json_object(cudf::strings_column_view(input), json_path);

    // clang-format off
    cudf::test::strings_column_wrapper expected({
      "[]",
      "[]",
      "",
      "",
      "",
      "[\"c\",null,true,-1]"},
      {1, 1, 0, 0, 0, 1});
    // clang-format on

    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*result, expected);
  }
}

TEST_F(JsonPathTests, StripQuotes)
{
  // we normally expect our outputs here to be
  // b     (no quotes)
  // but with string_quotes_from_single_strings false, we expect
  // "b"   (with quotes)
  {
    std::string str(R"({"a" : "b"})");
    cudf::test::strings_column_wrapper input({str, str});

    cudf::get_json_object_options options;
    options.set_strip_quotes_from_single_strings(false);

    std::string_view json_path("$.a");
    auto result_raw = cudf::get_json_object(cudf::strings_column_view(input), json_path, options);
    auto result     = drop_whitespace(*result_raw);

    cudf::test::strings_column_wrapper expected_raw({"\"b\"", "\"b\""});
    auto expected = drop_whitespace(expected_raw);

    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*result, *expected);
  }

  // a valid, but empty row
  {
    cudf::test::strings_column_wrapper input{R"({"store": { "bicycle" : "" } })"};
    std::string_view json_path("$.store.bicycle");

    cudf::get_json_object_options options;
    options.set_strip_quotes_from_single_strings(true);

    auto result = cudf::get_json_object(cudf::strings_column_view(input), json_path, options);

    cudf::test::strings_column_wrapper expected({""});

    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*result, expected);
  }
}

TEST_F(JsonPathTests, AllowSingleQuotes)
{
  // Tests allowing single quotes for strings.
  // Note:  this flag allows a mix of single and double quotes. it doesn't explicitly require
  // single-quotes only.

  // various queries on:
  std::vector<std::string> input_strings{
    // clang-format off
    R"({'a': {'b' : 'c'}})",

    "{"
      "\'a\': {\'b\' : \"c\"},"
      "\'d\': [{\"e\":123}, {\'f\':-10}]"
    "}",

    "{"
      "\'b\': 123"
    "}",

    "{"
      "\"a\": [\'y\',500]"
    "}",

    "{"
      "\'a\': \"\""
    "}",

    "{"
      "\"a\": {"
                "\'z\': {\'i\': 10, \'j\': 100},"
                "\'b\': [\'c\',null,true,-1]"
              "}"
    "}",

    "{"
      "\'a\': \"abc'def\""
    "}",

    "{"
      "\'a\': \"'abc'def'\""
    "}",
    // clang-format on
  };

  cudf::test::strings_column_wrapper input(input_strings.begin(), input_strings.end());
  {
    std::string_view json_path("$.a");

    cudf::get_json_object_options options;
    options.set_allow_single_quotes(true);

    auto result = cudf::get_json_object(cudf::strings_column_view(input), json_path, options);

    // clang-format off
    cudf::test::strings_column_wrapper expected({
      R"({'b' : 'c'})",
      R"({'b' : "c"})",
      "",
      "[\'y\',500]",
      "",
      "{"
         "\'z\': {\'i\': 10, \'j\': 100},"
         "\'b\': [\'c\',null,true,-1]"
      "}",
      "abc'def",
      "'abc'def'"
      },
      {1, 1, 0, 1, 1, 1, 1, 1});
    // clang-format on

    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*result, expected);
  }
}

TEST_F(JsonPathTests, StringsWithSpecialChars)
{
  // make sure we properly handle strings containing special characters
  // like { } [ ], etc
  // various queries on:

  {
    std::vector<std::string> input_strings{
      // clang-format off
      R"({"item" : [{"key" : "value["}]})",
      // clang-format on
    };

    cudf::test::strings_column_wrapper input(input_strings.begin(), input_strings.end());
    {
      std::string_view json_path("$.item");

      cudf::get_json_object_options options;
      options.set_allow_single_quotes(true);

      auto result = cudf::get_json_object(cudf::strings_column_view(input), json_path, options);

      // clang-format off
      cudf::test::strings_column_wrapper expected({
        R"([{"key" : "value["}])",
      });
      // clang-format on
      CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*result, expected);
    }
  }

  {
    std::vector<std::string> input_strings{
      // clang-format off
      R"({"a" : "[}{}][][{[\"}}[\"]"})",
      // clang-format on
    };

    cudf::test::strings_column_wrapper input(input_strings.begin(), input_strings.end());
    {
      std::string_view json_path("$.a");

      cudf::get_json_object_options options;
      options.set_allow_single_quotes(true);

      auto result = cudf::get_json_object(cudf::strings_column_view(input), json_path, options);

      // clang-format off
      cudf::test::strings_column_wrapper expected({
        R"([}{}][][{[\"}}[\"])",
      });
      // clang-format on
      CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*result, expected);
    }
  }
}

TEST_F(JsonPathTests, EscapeSequences)
{
  // valid escape sequences in JSON include
  // \" \\ \/ \b \f \n \r \t
  // \uXXXX  where X is a valid hex digit

  std::vector<std::string> input_strings{
    // clang-format off
    R"({"a" : "\" \\ \/ \b \f \n \r \t"})",
    R"({"a" : "\u1248 \uacdf \uACDF \u10EF"})"
    // clang-format on
  };

  cudf::test::strings_column_wrapper input(input_strings.begin(), input_strings.end());
  {
    std::string_view json_path("$.a");

    cudf::get_json_object_options options;
    options.set_allow_single_quotes(true);

    auto result = cudf::get_json_object(cudf::strings_column_view(input), json_path, options);

    // clang-format off
    cudf::test::strings_column_wrapper expected({
      R"(\" \\ \/ \b \f \n \r \t)",
      R"(\u1248 \uacdf \uACDF \u10EF)"
    });
    // clang-format on
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*result, expected);
  }
}

TEST_F(JsonPathTests, MissingFieldsAsNulls)
{
  std::string input_string{
    // clang-format off
    "{"
      "\"tup\":"
      "["
          "{\"id\":\"1\",\"array\":[1,2]},"
          "{\"id\":\"2\"},"
          "{\"id\":\"3\",\"array\":[3,4]},"
          "{\"id\":\"4\", \"a\": {\"x\": \"5\", \"y\": \"6\"}}"
      "]"
    "}"
    // clang-format on
  };
  auto do_test = [&input_string](auto const& json_path_string,
                                 auto const& default_output,
                                 auto const& missing_fields_output,
                                 bool default_valid = true) {
    cudf::test::strings_column_wrapper input{input_string};
    cudf::get_json_object_options options;

    // Test default behavior
    options.set_missing_fields_as_nulls(false);
    auto const default_result =
      cudf::get_json_object(cudf::strings_column_view(input), {json_path_string}, options);
    cudf::test::strings_column_wrapper default_expected({default_output}, {default_valid});

    CUDF_TEST_EXPECT_COLUMNS_EQUAL(default_expected, *default_result);

    // Test with missing fields as null
    options.set_missing_fields_as_nulls(true);
    auto const missing_fields_result =
      cudf::get_json_object(cudf::strings_column_view(input), {json_path_string}, options);
    cudf::test::strings_column_wrapper missing_fields_expected({missing_fields_output}, {1});

    CUDF_TEST_EXPECT_COLUMNS_EQUAL(missing_fields_expected, *missing_fields_result);
  };

  do_test("$.tup[1].array", "", "null", false);
  do_test("$.tup[*].array", "[[1,2],[3,4]]", "[[1,2],null,[3,4],null]");
  do_test("$.x[*].array", "", "null", false);
  do_test("$.tup[*].a.x", "[\"5\"]", "[null,null,null,\"5\"]");
}

TEST_F(JsonPathTests, QueriesContainingQuotes)
{
  std::string input_string = R"({"AB": 1, "A.B": 2, "'A": {"B'": 3}, "A": {"B": 4} })";

  auto do_test = [&input_string](auto const& json_path_string,
                                 auto const& expected_string,
                                 bool const& expect_null = false) {
    auto const input     = cudf::test::strings_column_wrapper{input_string};
    auto const json_path = std::string_view{json_path_string};
    cudf::get_json_object_options options;
    options.set_allow_single_quotes(true);
    auto const result = cudf::get_json_object(cudf::strings_column_view(input), json_path, options);
    auto const expected =
      cudf::test::strings_column_wrapper{std::initializer_list<std::string>{expected_string},
                                         std::initializer_list<bool>{!expect_null}};

    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*result, expected);
  };

  // Set 1
  do_test(R"($.AB)", "1");
  do_test(R"($['A.B'])", "2");
  do_test(R"($.'A.B')", "3");
  do_test(R"($.A.B)", "4");

  // Set 2
  do_test(R"($.'A)", R"({"B'": 3})");
}

TEST_F(JsonPathTests, DecodedAndQuotedFieldNames)
{
  auto const input = cudf::test::strings_column_wrapper{R"({"*":1,"\u0061":2})"};

  auto const literal_wildcard =
    cudf::get_json_object(cudf::strings_column_view(input), std::string_view{R"($['*'])"});
  auto const wildcard_expected = cudf::test::strings_column_wrapper{"1"};
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*literal_wildcard, wildcard_expected);

  auto const escaped_name =
    cudf::get_json_object(cudf::strings_column_view(input), std::string_view{"$.a"});
  auto const escaped_name_expected = cudf::test::strings_column_wrapper{"2"};
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*escaped_name, escaped_name_expected);
}

// Test that get_json_object creates valid string columns for empty/whitespace JSONPath queries
TEST_F(JsonPathTests, EmptyPathCreatesValidColumn)
{
  auto const input = cudf::test::strings_column_wrapper{R"({"a":"b"})"};

  // Test 1: Empty path should create VALID all-null column
  {
    std::string_view json_path("");
    auto result = cudf::get_json_object(cudf::strings_column_view(input), json_path);

    EXPECT_EQ(result->size(), 1);
    EXPECT_EQ(result->type().id(), cudf::type_id::STRING);
    EXPECT_EQ(result->null_count(), 1);
    EXPECT_GT(result->num_children(), 0);
  }

  // Test 2: Whitespace path should also create VALID all-null column
  {
    std::string_view json_path(" ");
    auto result = cudf::get_json_object(cudf::strings_column_view(input), json_path);

    EXPECT_EQ(result->size(), 1);
    EXPECT_EQ(result->null_count(), 1);
    EXPECT_GT(result->num_children(), 0);
  }
}

CUDF_TEST_PROGRAM_MAIN()
