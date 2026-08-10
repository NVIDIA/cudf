/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>

#include <cudf/column/column_factories.hpp>
#include <cudf/io/experimental/variant.hpp>
#include <cudf/io/experimental/variant_spec.hpp>
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/structs/structs_column_view.hpp>
#include <cudf/utilities/span.hpp>

#include <string>
#include <vector>

namespace {

// ─── helpers ─────────────────────────────────────────────────────────────────

// Encode a vector of JSON strings with the given column names.
std::unique_ptr<cudf::column> encode(std::vector<std::string> const& json_rows,
                                     std::vector<std::string> const& col_names,
                                     std::vector<bool> const& valid = {})
{
  std::unique_ptr<cudf::column> input_col;
  if (valid.empty()) {
    cudf::test::strings_column_wrapper w(json_rows.begin(), json_rows.end());
    input_col = w.release();
  } else {
    cudf::test::strings_column_wrapper w(json_rows.begin(), json_rows.end(), valid.begin());
    input_col = w.release();
  }
  cudf::strings_column_view scv{input_col->view()};
  std::vector<std::string> names(col_names);
  return cudf::io::parquet::experimental::encode_strings_to_variant(scv, names);
}

// Extract a field from a VARIANT struct column and cast to INT64.
std::unique_ptr<cudf::column> extract_int64(cudf::column_view const& variant,
                                            std::string const& path)
{
  using namespace cudf::io::parquet::experimental;
  return extract_variant_field(variant, path, cudf::data_type{cudf::type_id::INT64});
}

// Extract a field from a VARIANT struct column and cast to FLOAT64.
std::unique_ptr<cudf::column> extract_float64(cudf::column_view const& variant,
                                              std::string const& path)
{
  using namespace cudf::io::parquet::experimental;
  return extract_variant_field(variant, path, cudf::data_type{cudf::type_id::FLOAT64});
}

// Extract a field from a VARIANT struct column and cast to STRING.
std::unique_ptr<cudf::column> extract_string(cudf::column_view const& variant,
                                             std::string const& path)
{
  using namespace cudf::io::parquet::experimental;
  return extract_variant_field(variant, path, cudf::data_type{cudf::type_id::STRING});
}

// Extract a field from a VARIANT struct column and cast to BOOL8.
std::unique_ptr<cudf::column> extract_bool(cudf::column_view const& variant,
                                           std::string const& path)
{
  using namespace cudf::io::parquet::experimental;
  return extract_variant_field(variant, path, cudf::data_type{cudf::type_id::BOOL8});
}

}  // namespace

struct EncodeStringsToVariantTest : public cudf::test::BaseFixture {};

// ─── single-field tests ───────────────────────────────────────────────────────

TEST_F(EncodeStringsToVariantTest, SingleRowInteger)
{
  auto variant = encode({R"({"a":42})"}, {"a"});

  auto ints = extract_int64(variant->view(), "$.a");
  cudf::test::fixed_width_column_wrapper<int64_t> expected{42};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*ints, expected);
}

TEST_F(EncodeStringsToVariantTest, SingleRowNegativeInteger)
{
  auto variant = encode({R"({"x":-100})"}, {"x"});

  auto ints = extract_int64(variant->view(), "$.x");
  cudf::test::fixed_width_column_wrapper<int64_t> expected{-100};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*ints, expected);
}

TEST_F(EncodeStringsToVariantTest, SingleRowFloat)
{
  auto variant = encode({R"({"f":3.14})"}, {"f"});

  auto floats = extract_float64(variant->view(), "$.f");
  // Check that we get a non-null row
  EXPECT_EQ(floats->null_count(), 0);
  EXPECT_EQ(floats->size(), 1);
}

TEST_F(EncodeStringsToVariantTest, SingleRowFloatExponent)
{
  auto variant = encode({R"({"f":1.5e2})"}, {"f"});

  auto floats = extract_float64(variant->view(), "$.f");
  EXPECT_EQ(floats->null_count(), 0);
  EXPECT_EQ(floats->size(), 1);
}

TEST_F(EncodeStringsToVariantTest, SingleRowBoolTrue)
{
  auto variant = encode({R"({"b":true})"}, {"b"});

  auto bools = extract_bool(variant->view(), "$.b");
  cudf::test::fixed_width_column_wrapper<bool> expected{true};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*bools, expected);
}

TEST_F(EncodeStringsToVariantTest, SingleRowBoolFalse)
{
  auto variant = encode({R"({"b":false})"}, {"b"});

  auto bools = extract_bool(variant->view(), "$.b");
  cudf::test::fixed_width_column_wrapper<bool> expected{false};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*bools, expected);
}

TEST_F(EncodeStringsToVariantTest, SingleRowShortString)
{
  auto variant = encode({R"({"s":"hello"})"}, {"s"});

  auto strs = extract_string(variant->view(), "$.s");
  cudf::test::strings_column_wrapper expected{"hello"};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*strs, expected);
}

TEST_F(EncodeStringsToVariantTest, SingleRowLongString)
{
  // Strings > 63 bytes use the LONG_STRING encoding path
  std::string long_str(70, 'x');
  auto json    = R"({"s":")" + long_str + R"("})";
  auto variant = encode({json}, {"s"});

  auto strs = extract_string(variant->view(), "$.s");
  cudf::test::strings_column_wrapper expected{long_str};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*strs, expected);
}

TEST_F(EncodeStringsToVariantTest, SingleRowNullValue)
{
  // JSON null value → VARIANT null; cast_variant returns null for that row
  auto variant = encode({R"({"a":null})"}, {"a"});

  auto ints = extract_int64(variant->view(), "$.a");
  // null JSON value encodes as VARIANT null, which cast_variant cannot cast to INT64
  EXPECT_EQ(ints->size(), 1);
}

// ─── multi-field tests ────────────────────────────────────────────────────────

TEST_F(EncodeStringsToVariantTest, MultiField)
{
  auto variant = encode({R"({"a":1,"b":"world","c":true})"}, {"a", "b", "c"});

  auto ints  = extract_int64(variant->view(), "$.a");
  auto strs  = extract_string(variant->view(), "$.b");
  auto bools = extract_bool(variant->view(), "$.c");

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

  auto a_vals = extract_int64(variant->view(), "$.a");
  auto z_vals = extract_int64(variant->view(), "$.z");

  cudf::test::fixed_width_column_wrapper<int64_t> exp_a{7};
  cudf::test::fixed_width_column_wrapper<int64_t> exp_z{99};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*a_vals, exp_a);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*z_vals, exp_z);
}

// ─── missing fields ───────────────────────────────────────────────────────────

TEST_F(EncodeStringsToVariantTest, MissingFieldIsAbsent)
{
  // "b" is listed in column_names but absent from the JSON object
  auto variant = encode({R"({"a":5})"}, {"a", "b"});

  auto a_vals = extract_int64(variant->view(), "$.a");
  auto b_vals = extract_int64(variant->view(), "$.b");

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

  auto only_vals = extract_int64(variant->view(), "$.only");
  cudf::test::fixed_width_column_wrapper<int64_t> expected{42};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*only_vals, expected);

  auto x_vals = extract_int64(variant->view(), "$.x");
  EXPECT_EQ(x_vals->null_count(), 1);
}

// ─── multi-row tests ──────────────────────────────────────────────────────────

TEST_F(EncodeStringsToVariantTest, MultipleRows)
{
  auto variant =
    encode({R"({"a":1,"b":"foo"})", R"({"a":2,"b":"bar"})", R"({"a":3,"b":"baz"})"}, {"a", "b"});

  auto a_vals = extract_int64(variant->view(), "$.a");
  auto b_vals = extract_string(variant->view(), "$.b");

  cudf::test::fixed_width_column_wrapper<int64_t> exp_a{1, 2, 3};
  cudf::test::strings_column_wrapper exp_b{"foo", "bar", "baz"};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*a_vals, exp_a);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*b_vals, exp_b);
}

TEST_F(EncodeStringsToVariantTest, MultipleRowsDifferentFieldsPresent)
{
  // Row 0 has a; row 1 has b; row 2 has both
  auto variant = encode({R"({"a":10})", R"({"b":20})", R"({"a":30,"b":40})"}, {"a", "b"});

  auto a_vals = extract_int64(variant->view(), "$.a");
  auto b_vals = extract_int64(variant->view(), "$.b");

  // Row 1 has no "a" → null
  cudf::test::fixed_width_column_wrapper<int64_t> exp_a({10, 0, 30}, {true, false, true});
  cudf::test::fixed_width_column_wrapper<int64_t> exp_b({0, 20, 40}, {false, true, true});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*a_vals, exp_a);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*b_vals, exp_b);
}

// ─── null input rows ──────────────────────────────────────────────────────────

TEST_F(EncodeStringsToVariantTest, NullInputRow)
{
  // Row 1 is null
  auto variant = encode({R"({"a":7})", R"({"a":8})", R"({"a":9})"}, {"a"}, {true, false, true});

  ASSERT_EQ(variant->type().id(), cudf::type_id::STRUCT);
  EXPECT_EQ(variant->null_count(), 1);

  auto a_vals = extract_int64(variant->view(), "$.a");
  cudf::test::fixed_width_column_wrapper<int64_t> expected({7, 0, 9}, {true, false, true});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*a_vals, expected);
}

TEST_F(EncodeStringsToVariantTest, AllNullInputRows)
{
  auto variant = encode({R"({"a":1})", R"({"a":2})"}, {"a"}, {false, false});

  EXPECT_EQ(variant->null_count(), 2);
}

// ─── empty input ─────────────────────────────────────────────────────────────

TEST_F(EncodeStringsToVariantTest, EmptyInput)
{
  auto variant = encode({}, {"a", "b"});

  ASSERT_EQ(variant->type().id(), cudf::type_id::STRUCT);
  EXPECT_EQ(variant->size(), 0);
}

// ─── output structure ────────────────────────────────────────────────────────

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

// ─── zero column_names ────────────────────────────────────────────────────────

TEST_F(EncodeStringsToVariantTest, NoColumnNames)
{
  // Empty field list → every row encodes as an empty VARIANT object
  auto variant = encode({R"({"a":1})", R"({"b":2})"}, {});

  ASSERT_EQ(variant->type().id(), cudf::type_id::STRUCT);
  EXPECT_EQ(variant->size(), 2);
  EXPECT_EQ(variant->null_count(), 0);
}
