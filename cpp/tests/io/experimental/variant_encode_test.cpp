/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/cudf_gtest.hpp>

#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/io/experimental/variant.hpp>
#include <cudf/io/experimental/variant_spec.hpp>
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/structs/structs_column_view.hpp>

#include <cmath>
#include <cstdint>
#include <initializer_list>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

/**
 * @brief Encode JSON object strings with the requested field names.
 *
 * @param json_rows Input JSON rows
 * @param column_names Field names to encode
 * @param valid Optional row validity
 * @return Encoded Parquet VARIANT column
 */
std::unique_ptr<cudf::column> encode(std::span<std::string const> json_rows,
                                     std::span<std::string const> column_names,
                                     std::span<bool const> valid = {})
{
  std::unique_ptr<cudf::column> input_col;
  if (valid.empty()) {
    cudf::test::strings_column_wrapper input(json_rows.begin(), json_rows.end());
    input_col = input.release();
  } else {
    cudf::test::strings_column_wrapper input(json_rows.begin(), json_rows.end(), valid.begin());
    input_col = input.release();
  }
  return cudf::io::parquet::experimental::encode_strings_to_variant(
    cudf::strings_column_view{input_col->view()}, column_names);
}

/**
 * @brief Convenience overload for inline test data.
 */
std::unique_ptr<cudf::column> encode(std::initializer_list<std::string> json_rows,
                                     std::initializer_list<std::string> column_names,
                                     std::initializer_list<bool> valid = {})
{
  return encode(std::span{json_rows.begin(), json_rows.size()},
                std::span{column_names.begin(), column_names.size()},
                std::span{valid.begin(), valid.size()});
}

/**
 * @brief Encode a slice of JSON object strings.
 *
 * @param json_rows Input JSON rows
 * @param column_names Field names to encode
 * @param offset First input row in the slice
 * @param size Number of rows in the slice
 * @param valid Optional row validity
 * @return Encoded Parquet VARIANT column
 */
std::unique_ptr<cudf::column> encode_sliced(std::span<std::string const> json_rows,
                                            std::span<std::string const> column_names,
                                            cudf::size_type offset,
                                            cudf::size_type size,
                                            std::span<bool const> valid = {})
{
  std::unique_ptr<cudf::column> input_col;
  if (valid.empty()) {
    cudf::test::strings_column_wrapper input(json_rows.begin(), json_rows.end());
    input_col = input.release();
  } else {
    cudf::test::strings_column_wrapper input(json_rows.begin(), json_rows.end(), valid.begin());
    input_col = input.release();
  }
  auto const sliced = cudf::slice(input_col->view(), {offset, offset + size})[0];
  return cudf::io::parquet::experimental::encode_strings_to_variant(
    cudf::strings_column_view{sliced}, column_names);
}

/**
 * @brief Convenience overload for sliced inline test data.
 */
std::unique_ptr<cudf::column> encode_sliced(std::initializer_list<std::string> json_rows,
                                            std::initializer_list<std::string> column_names,
                                            cudf::size_type offset,
                                            cudf::size_type size,
                                            std::initializer_list<bool> valid = {})
{
  return encode_sliced(std::span{json_rows.begin(), json_rows.size()},
                       std::span{column_names.begin(), column_names.size()},
                       offset,
                       size,
                       std::span{valid.begin(), valid.size()});
}

/**
 * @brief Extract and cast a field to the requested cudf type.
 */
template <cudf::type_id Type>
std::unique_ptr<cudf::column> extract(cudf::column_view const& variant, std::string_view path)
{
  return cudf::io::parquet::experimental::extract_variant_field(
    variant, path, cudf::data_type{Type});
}

}  // namespace

struct EncodeStringsToVariantSingleFieldTest : public cudf::test::BaseFixture {};

TEST_F(EncodeStringsToVariantSingleFieldTest, SingleRowInteger)
{
  auto variant = encode({R"({"a":42})"}, {"a"});

  auto ints = extract<cudf::type_id::INT64>(variant->view(), "$.a");
  cudf::test::fixed_width_column_wrapper<int64_t> expected{42};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*ints, expected);
}

TEST_F(EncodeStringsToVariantSingleFieldTest, SingleRowNegativeInteger)
{
  auto variant = encode({R"({"x":-100})"}, {"x"});

  auto ints = extract<cudf::type_id::INT64>(variant->view(), "$.x");
  cudf::test::fixed_width_column_wrapper<int64_t> expected{-100};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*ints, expected);
}

TEST_F(EncodeStringsToVariantSingleFieldTest, SingleRowFloat)
{
  auto variant = encode({R"({"f":3.14})"}, {"f"});

  auto floats = extract<cudf::type_id::FLOAT64>(variant->view(), "$.f");
  cudf::test::fixed_width_column_wrapper<double> expected{3.14};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*floats, expected);
}

TEST_F(EncodeStringsToVariantSingleFieldTest, SingleRowFloatExponent)
{
  auto variant = encode({R"({"f":1.5e2})"}, {"f"});

  auto floats = extract<cudf::type_id::FLOAT64>(variant->view(), "$.f");
  cudf::test::fixed_width_column_wrapper<double> expected{150.0};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*floats, expected);
}

TEST_F(EncodeStringsToVariantSingleFieldTest, ExtremeFloatExponents)
{
  auto variant = encode({R"({"f":0e401})",
                         R"({"f":0e309})",
                         R"({"f":1e-309})",
                         R"({"f":123456789012345678901234567890e-29})"},
                        {"f"});

  auto floats       = extract<cudf::type_id::FLOAT64>(variant->view(), "$.f");
  auto const values = cudf::test::to_host<double>(floats->view()).first;
  ASSERT_EQ(values.size(), 4);
  EXPECT_EQ(values[0], 0.0);
  EXPECT_TRUE(std::isfinite(values[0]));
  EXPECT_EQ(values[1], 0.0);
  EXPECT_TRUE(std::isfinite(values[1]));
  EXPECT_EQ(values[2], 1e-309);
  EXPECT_GT(values[2], 0.0);
  EXPECT_TRUE(std::isfinite(values[2]));
  EXPECT_DOUBLE_EQ(values[3], 1.23456789012345678901234567890);
  EXPECT_TRUE(std::isfinite(values[3]));
}

TEST_F(EncodeStringsToVariantSingleFieldTest, SingleRowBoolTrue)
{
  auto variant = encode({R"({"b":true})"}, {"b"});

  auto bools = extract<cudf::type_id::BOOL8>(variant->view(), "$.b");
  cudf::test::fixed_width_column_wrapper<bool> expected{true};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*bools, expected);
}

TEST_F(EncodeStringsToVariantSingleFieldTest, SingleRowBoolFalse)
{
  auto variant = encode({R"({"b":false})"}, {"b"});

  auto bools = extract<cudf::type_id::BOOL8>(variant->view(), "$.b");
  cudf::test::fixed_width_column_wrapper<bool> expected{false};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*bools, expected);
}

TEST_F(EncodeStringsToVariantSingleFieldTest, SingleRowShortString)
{
  auto variant = encode({R"({"s":"hello"})"}, {"s"});

  auto strs = extract<cudf::type_id::STRING>(variant->view(), "$.s");
  cudf::test::strings_column_wrapper expected{"hello"};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*strs, expected);
}

TEST_F(EncodeStringsToVariantSingleFieldTest, SingleRowShortStringBoundary)
{
  // 63 bytes is the maximum for the SHORT_STRING encoding path
  std::string boundary_str(63, 'x');
  auto json    = R"({"s":")" + boundary_str + R"("})";
  auto variant = encode({json}, {"s"});

  auto strs = extract<cudf::type_id::STRING>(variant->view(), "$.s");
  cudf::test::strings_column_wrapper expected{boundary_str};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*strs, expected);
}

TEST_F(EncodeStringsToVariantSingleFieldTest, SingleRowLongString)
{
  // Strings > 63 bytes use the LONG_STRING encoding path
  std::string long_str(70, 'x');
  auto json    = R"({"s":")" + long_str + R"("})";
  auto variant = encode({json}, {"s"});

  auto strs = extract<cudf::type_id::STRING>(variant->view(), "$.s");
  cudf::test::strings_column_wrapper expected{long_str};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*strs, expected);
}

TEST_F(EncodeStringsToVariantSingleFieldTest, SingleRowShortStringNonAscii)
{
  // "café" – é is U+00E9, encoded as 2 UTF-8 bytes (0xC3 0xA9), total 5 bytes
  auto variant = encode({R"({"s":"café"})"}, {"s"});

  auto strs = extract<cudf::type_id::STRING>(variant->view(), "$.s");
  cudf::test::strings_column_wrapper expected{"café"};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*strs, expected);
}

TEST_F(EncodeStringsToVariantSingleFieldTest, SingleRowLongStringNonAscii)
{
  // 35 × "é" (2 UTF-8 bytes each) = 70 bytes → LONG_STRING path
  std::string non_ascii_long;
  for (int i = 0; i < 35; ++i) {
    non_ascii_long += "\xC3\xA9";  // UTF-8 for é
  }
  auto json    = R"({"s":")" + non_ascii_long + R"("})";
  auto variant = encode({json}, {"s"});

  auto strs = extract<cudf::type_id::STRING>(variant->view(), "$.s");
  cudf::test::strings_column_wrapper expected{non_ascii_long};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*strs, expected);
}

TEST_F(EncodeStringsToVariantSingleFieldTest, SingleRowEscapedString)
{
  // The JSON string contains an escaped quote and an escaped backslash; the VARIANT payload
  // must contain the unescaped bytes a"b\c, not the raw escape sequences.
  auto variant = encode({R"({"s":"a\"b\\c"})"}, {"s"});

  auto strs = extract<cudf::type_id::STRING>(variant->view(), "$.s");
  cudf::test::strings_column_wrapper expected{"a\"b\\c"};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*strs, expected);
}

TEST_F(EncodeStringsToVariantSingleFieldTest, SingleRowCommonEscapes)
{
  // \n \t \r \b \f \/ all unescape to single control/ASCII bytes.
  auto variant = encode({R"({"s":"a\nb\tc\rd\be\ff\/g"})"}, {"s"});

  auto strs = extract<cudf::type_id::STRING>(variant->view(), "$.s");
  cudf::test::strings_column_wrapper expected{"a\nb\tc\rd\be\ff/g"};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*strs, expected);
}
TEST_F(EncodeStringsToVariantSingleFieldTest, SingleRowUnicodeEscape)
{
  // \u00e9 is the JSON \uXXXX escape for U+00E9 (e-acute), which must be unescaped to its
  // 2-byte UTF-8 encoding, not copied as the literal 6-character escape sequence.
  auto variant = encode({R"({"s":"caf\u00e9"})"}, {"s"});

  auto strs = extract<cudf::type_id::STRING>(variant->view(), "$.s");
  cudf::test::strings_column_wrapper expected{"café"};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*strs, expected);
}

TEST_F(EncodeStringsToVariantSingleFieldTest, SingleRowLiteralEmojiPassthrough)
{
  // A literal (non-escaped) 4-byte UTF-8 emoji is copied through unchanged.
  auto variant = encode({R"({"s":"😀"})"}, {"s"});

  auto strs = extract<cudf::type_id::STRING>(variant->view(), "$.s");
  cudf::test::strings_column_wrapper expected{"\xF0\x9F\x98\x80"};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*strs, expected);
}

TEST_F(EncodeStringsToVariantSingleFieldTest, SingleRowSurrogatePairEscape)
{
  // U+1F600 (grinning face emoji) as a JSON \uD83D\uDE00 UTF-16 surrogate pair escape, which
  // must be combined and unescaped to its 4-byte UTF-8 encoding.
  auto variant = encode({R"({"s":"\uD83D\uDE00"})"}, {"s"});

  auto strs = extract<cudf::type_id::STRING>(variant->view(), "$.s");
  cudf::test::strings_column_wrapper expected{"\xF0\x9F\x98\x80"};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*strs, expected);
}

TEST_F(EncodeStringsToVariantSingleFieldTest, UnpairedSurrogateEscapeRejected)
{
  // A high surrogate with no following low surrogate is not valid UTF-16/8 and must be rejected
  // rather than silently mis-encoded.
  EXPECT_THROW(encode({R"({"s":"\uD83D"})"}, {"s"}), std::invalid_argument);
}

TEST_F(EncodeStringsToVariantSingleFieldTest, NestedObjectFieldValueRejected)
{
  // The encoder only supports scalar, non-nested field values.
  EXPECT_THROW(encode({R"({"a":{"x":1}})"}, {"a"}), std::invalid_argument);
}

TEST_F(EncodeStringsToVariantSingleFieldTest, NestedArrayFieldValueRejected)
{
  EXPECT_THROW(encode({R"({"a":[1,2,3]})"}, {"a"}), std::invalid_argument);
}

TEST_F(EncodeStringsToVariantSingleFieldTest, SingleRowNullValue)
{
  // JSON null value → VARIANT null; cast_variant returns null for that row
  auto variant = encode({R"({"a":null})"}, {"a"});

  auto ints = extract<cudf::type_id::INT64>(variant->view(), "$.a");
  cudf::test::fixed_width_column_wrapper<int64_t> expected({0}, {false});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*ints, expected);
}

TEST_F(EncodeStringsToVariantSingleFieldTest, SlicedInputView)
{
  // Build a 5-row column and encode only rows 1–3 via a sliced strings_column_view.
  auto variant =
    encode_sliced({R"({"a":0})", R"({"a":10})", R"({"a":20})", R"({"a":30})", R"({"a":40})"},
                  {"a"},
                  /*offset=*/1,
                  /*size=*/3);

  ASSERT_EQ(variant->size(), 3);
  auto a_vals = extract<cudf::type_id::INT64>(variant->view(), "$.a");
  cudf::test::fixed_width_column_wrapper<int64_t> expected{10, 20, 30};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*a_vals, expected);
}

TEST_F(EncodeStringsToVariantSingleFieldTest, SlicedNullableInputView)
{
  auto variant =
    encode_sliced({R"({"a":0})", R"({"a":10})", R"({"a":20})", R"({"a":30})", R"({"a":40})"},
                  {"a"},
                  /*offset=*/1,
                  /*size=*/3,
                  {true, true, false, true, true});

  EXPECT_EQ(variant->null_count(), 1);
  auto values = extract<cudf::type_id::INT64>(variant->view(), "$.a");
  cudf::test::fixed_width_column_wrapper<int64_t> expected({10, 0, 30}, {true, false, true});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*values, expected);
}

struct EncodeStringsToVariantTest : public cudf::test::BaseFixture {};

TEST_F(EncodeStringsToVariantTest, MultiField)
{
  auto variant = encode({R"({"a":1,"b":"world","c":true})"}, {"a", "b", "c"});

  auto ints  = extract<cudf::type_id::INT64>(variant->view(), "$.a");
  auto strs  = extract<cudf::type_id::STRING>(variant->view(), "$.b");
  auto bools = extract<cudf::type_id::BOOL8>(variant->view(), "$.c");

  cudf::test::fixed_width_column_wrapper<int64_t> exp_ints{1};
  cudf::test::strings_column_wrapper exp_strs{"world"};
  cudf::test::fixed_width_column_wrapper<bool> exp_bools{true};

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*ints, exp_ints);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*strs, exp_strs);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*bools, exp_bools);
}

TEST_F(EncodeStringsToVariantTest, FieldOrderIndependentOfInputOrder)
{
  // column_names provided in reverse alphabetical order; should still encode correctly
  auto variant = encode({R"({"z":99,"a":7})"}, {"z", "a"});

  auto a_vals = extract<cudf::type_id::INT64>(variant->view(), "$.a");
  auto z_vals = extract<cudf::type_id::INT64>(variant->view(), "$.z");

  cudf::test::fixed_width_column_wrapper<int64_t> exp_a{7};
  cudf::test::fixed_width_column_wrapper<int64_t> exp_z{99};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*a_vals, exp_a);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*z_vals, exp_z);
}

TEST_F(EncodeStringsToVariantTest, MissingFieldIsAbsent)
{
  // "b" is listed in column_names but absent from the JSON object
  auto variant = encode({R"({"a":5})"}, {"a", "b"});

  auto a_vals = extract<cudf::type_id::INT64>(variant->view(), "$.a");
  auto b_vals = extract<cudf::type_id::INT64>(variant->view(), "$.b");

  cudf::test::fixed_width_column_wrapper<int64_t> exp_a{5};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*a_vals, exp_a);
  // b is absent → null
  EXPECT_EQ(b_vals->size(), 1);
  EXPECT_EQ(b_vals->null_count(), 1);
}

TEST_F(EncodeStringsToVariantTest, ExtraColumnsInNameList)
{
  // Many names provided; most absent from the JSON
  auto variant = encode({R"({"only":42})"}, {"only", "x", "y", "z"});

  auto only_vals = extract<cudf::type_id::INT64>(variant->view(), "$.only");
  cudf::test::fixed_width_column_wrapper<int64_t> expected{42};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*only_vals, expected);

  auto x_vals = extract<cudf::type_id::INT64>(variant->view(), "$.x");
  EXPECT_EQ(x_vals->null_count(), 1);
}

TEST_F(EncodeStringsToVariantTest, LiteralWildcardFieldName)
{
  auto variant = encode({R"({"*":42,"other":1})"}, {"*"});

  auto values = extract<cudf::type_id::INT64>(variant->view(), "$.*");
  cudf::test::fixed_width_column_wrapper<int64_t> expected{42};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*values, expected);
}

TEST_F(EncodeStringsToVariantTest, EscapedJsonFieldName)
{
  auto variant = encode({R"({"\u0061":7})"}, {"a"});

  auto values = extract<cudf::type_id::INT64>(variant->view(), "$.a");
  cudf::test::fixed_width_column_wrapper<int64_t> expected{7};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*values, expected);
}

TEST_F(EncodeStringsToVariantTest, ApostropheInFieldName)
{
  auto variant = encode({R"({"a'b":9})"}, {"a'b"});

  auto values = extract<cudf::type_id::INT64>(variant->view(), "$.a'b");
  cudf::test::fixed_width_column_wrapper<int64_t> expected{9};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*values, expected);
}

TEST_F(EncodeStringsToVariantTest, MultipleRows)
{
  auto variant =
    encode({R"({"a":1,"b":"foo"})", R"({"a":2,"b":"bar"})", R"({"a":3,"b":"baz"})"}, {"a", "b"});

  auto a_vals = extract<cudf::type_id::INT64>(variant->view(), "$.a");
  auto b_vals = extract<cudf::type_id::STRING>(variant->view(), "$.b");

  cudf::test::fixed_width_column_wrapper<int64_t> exp_a{1, 2, 3};
  cudf::test::strings_column_wrapper exp_b{"foo", "bar", "baz"};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*a_vals, exp_a);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*b_vals, exp_b);
}

TEST_F(EncodeStringsToVariantTest, MultipleRowsManyBlocks)
{
  // 512 rows ensures work is spread across multiple CUDA blocks (typically 256 threads each).
  constexpr int N = 512;
  std::vector<std::string> json_rows;
  json_rows.reserve(N);
  for (int i = 0; i < N; ++i) {
    json_rows.push_back(R"({"a":)" + std::to_string(i) + R"(})");
  }

  std::vector<std::string> const column_names{"a"};
  auto variant = encode(json_rows, column_names);
  ASSERT_EQ(variant->size(), N);

  auto a_vals = extract<cudf::type_id::INT64>(variant->view(), "$.a");
  ASSERT_EQ(a_vals->size(), N);
  EXPECT_EQ(a_vals->null_count(), 0);

  std::vector<int64_t> exp_vals(N);
  for (int i = 0; i < N; ++i) {
    exp_vals[i] = i;
  }
  cudf::test::fixed_width_column_wrapper<int64_t> expected(exp_vals.begin(), exp_vals.end());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*a_vals, expected);
}

TEST_F(EncodeStringsToVariantTest, MultipleRowsDifferentFieldsPresent)
{
  // Row 0 has a; row 1 has b; row 2 has both
  auto variant = encode({R"({"a":10})", R"({"b":20})", R"({"a":30,"b":40})"}, {"a", "b"});

  auto a_vals = extract<cudf::type_id::INT64>(variant->view(), "$.a");
  auto b_vals = extract<cudf::type_id::INT64>(variant->view(), "$.b");

  // Row 1 has no "a" → null
  cudf::test::fixed_width_column_wrapper<int64_t> exp_a({10, 0, 30}, {true, false, true});
  cudf::test::fixed_width_column_wrapper<int64_t> exp_b({0, 20, 40}, {false, true, true});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*a_vals, exp_a);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*b_vals, exp_b);
}

TEST_F(EncodeStringsToVariantTest, NullInputRow)
{
  // Row 1 is null
  auto variant = encode({R"({"a":7})", R"({"a":8})", R"({"a":9})"}, {"a"}, {true, false, true});

  ASSERT_EQ(variant->type().id(), cudf::type_id::STRUCT);
  EXPECT_EQ(variant->null_count(), 1);

  auto a_vals = extract<cudf::type_id::INT64>(variant->view(), "$.a");
  cudf::test::fixed_width_column_wrapper<int64_t> expected({7, 0, 9}, {true, false, true});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*a_vals, expected);
}

TEST_F(EncodeStringsToVariantTest, AllNullInputRows)
{
  auto variant = encode({R"({"a":1})", R"({"a":2})"}, {"a"}, {false, false});

  EXPECT_EQ(variant->null_count(), 2);
}

TEST_F(EncodeStringsToVariantTest, EmptyInput)
{
  auto variant = encode({}, {"a", "b"});

  ASSERT_EQ(variant->type().id(), cudf::type_id::STRUCT);
  EXPECT_EQ(variant->size(), 0);
}

TEST_F(EncodeStringsToVariantTest, OutputIsVariantStruct)
{
  auto variant = encode({R"({"x":1})"}, {"x"});

  // Must be struct<list<uint8>, list<uint8>>
  ASSERT_EQ(variant->type().id(), cudf::type_id::STRUCT);
  ASSERT_EQ(variant->num_children(), 2);

  cudf::structs_column_view sv{variant->view()};
  EXPECT_EQ(sv.child(0).type().id(), cudf::type_id::LIST);  // metadata
  EXPECT_EQ(sv.child(1).type().id(), cudf::type_id::LIST);  // value

  cudf::lists_column_view meta_lv{sv.child(0)};
  cudf::lists_column_view val_lv{sv.child(1)};
  EXPECT_EQ(meta_lv.child().type().id(), cudf::type_id::UINT8);
  EXPECT_EQ(val_lv.child().type().id(), cudf::type_id::UINT8);
}

TEST_F(EncodeStringsToVariantTest, NoColumnNames)
{
  // Empty field list → every row encodes as an empty VARIANT object
  auto variant = encode({R"({"a":1})", R"({"b":2})"}, {});

  ASSERT_EQ(variant->type().id(), cudf::type_id::STRUCT);
  EXPECT_EQ(variant->size(), 2);
  EXPECT_EQ(variant->null_count(), 0);
}

TEST_F(EncodeStringsToVariantTest, EmptyInputStillValidatesColumnNames)
{
  EXPECT_THROW(encode({}, {"a", "a"}), std::invalid_argument);
  EXPECT_THROW(encode({}, {""}), std::invalid_argument);
  EXPECT_THROW(encode({}, {"a.b"}), std::invalid_argument);
  EXPECT_THROW(encode({}, {"a[0]"}), std::invalid_argument);
}

TEST_F(EncodeStringsToVariantTest, FieldCountLimit)
{
  std::vector<std::string> names;
  names.reserve(256);
  for (int i = 0; i < 255; ++i) {
    names.push_back("f" + std::to_string(1000 + i).substr(1));
  }
  std::vector<std::string> const json_rows{R"({"f254":254})"};

  auto variant   = encode(json_rows, names);
  auto field_254 = extract<cudf::type_id::INT64>(variant->view(), "$.f254");
  cudf::test::fixed_width_column_wrapper<int64_t> expected{254};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*field_254, expected);

  names.push_back("f255");
  EXPECT_THROW(encode(json_rows, names), std::invalid_argument);
}
