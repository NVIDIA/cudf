/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "json_to_variant_golden.hpp"

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>

#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/io/experimental/variant.hpp>
#include <cudf/io/experimental/variant_spec.hpp>
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/structs/structs_column_view.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <optional>
#include <string>
#include <vector>

namespace pqe = cudf::io::parquet::experimental;

using pqe::variant_operation_status;

namespace {

using bytes = std::vector<uint8_t>;

struct host_row {
  variant_operation_status status;
  bool valid;
  bytes metadata;
  bytes value;
};

std::vector<bytes> list_rows(cudf::column_view const& col)
{
  auto const stream = cudf::get_default_stream();
  cudf::lists_column_view const lists{col};
  if (lists.size() == 0) { return {}; }
  auto const offsets = cudf::detail::make_std_vector(
    cudf::device_span<cudf::size_type const>{lists.offsets_begin(),
                                             static_cast<std::size_t>(lists.size() + 1)},
    stream);
  auto const data = cudf::detail::make_std_vector(
    cudf::device_span<uint8_t const>{lists.child().data<uint8_t>(),
                                     static_cast<std::size_t>(lists.child().size())},
    stream);
  std::vector<bytes> rows;
  for (cudf::size_type r = 0; r < lists.size(); ++r) {
    rows.emplace_back(data.begin() + offsets[r], data.begin() + offsets[r + 1]);
  }
  return rows;
}

std::vector<host_row> convert(cudf::column_view const& input, bool allow_duplicate_keys)
{
  auto const stream = cudf::get_default_stream();
  auto status = cudf::make_numeric_column(cudf::data_type{cudf::type_id::UINT8}, input.size());
  auto const result = pqe::parse_json_to_variant(
    cudf::strings_column_view{input}, allow_duplicate_keys, status->mutable_view());
  EXPECT_EQ(result->size(), input.size());
  EXPECT_EQ(result->type().id(), cudf::type_id::STRUCT);
  if (input.size() == 0) { return {}; }

  auto const statuses = cudf::detail::make_std_vector(
    cudf::device_span<uint8_t const>{status->view().data<uint8_t>(),
                                     static_cast<std::size_t>(input.size())},
    stream);
  auto const valid = [&] {
    if (!result->nullable()) { return std::vector<bool>(input.size(), true); }
    auto const mask = cudf::detail::make_std_vector(
      cudf::device_span<cudf::bitmask_type const>{
        result->view().null_mask(),
        static_cast<std::size_t>(cudf::num_bitmask_words(input.size()))},
      stream);
    std::vector<bool> out(input.size());
    for (cudf::size_type r = 0; r < input.size(); ++r) {
      out[r] = (mask[r / 32] >> (r % 32)) & 1u;
    }
    return out;
  }();
  cudf::structs_column_view const sv{result->view()};
  auto const metadata = list_rows(sv.child(0));
  auto const values   = list_rows(sv.child(1));

  std::vector<host_row> rows;
  for (cudf::size_type r = 0; r < input.size(); ++r) {
    rows.push_back({static_cast<variant_operation_status>(statuses[r]),
                    valid[r],
                    valid[r] ? metadata[r] : bytes{},
                    valid[r] ? values[r] : bytes{}});
  }
  return rows;
}

host_row convert_one(std::string const& json, bool allow_duplicate_keys = false)
{
  cudf::test::strings_column_wrapper input({json});
  return convert(input, allow_duplicate_keys).front();
}

bytes from_hex(char const* hex)
{
  bytes out;
  for (std::size_t i = 0; hex[i] != '\0'; i += 2) {
    out.push_back(static_cast<uint8_t>(std::stoi(std::string(hex + i, 2), nullptr, 16)));
  }
  return out;
}

void append_le(bytes& out, uint64_t v, int width)
{
  for (int i = 0; i < width; ++i) {
    out.push_back(static_cast<uint8_t>(v >> (8 * i)));
  }
}

int width_of(uint64_t v) { return v <= 0xFF ? 1 : v <= 0xFFFF ? 2 : v <= 0xFF'FFFF ? 3 : 4; }

// Minimal reference encoders for the shapes used by the width tests
bytes encode_int(int64_t v)
{
  using T                  = pqe::variant_primitive_type;
  auto const [type, width] = (v >= -128 && v <= 127)       ? std::pair{T::INT8, 1}
                             : (v >= -32768 && v <= 32767) ? std::pair{T::INT16, 2}
                                                           : std::pair{T::INT32, 4};
  bytes out{static_cast<uint8_t>(static_cast<uint8_t>(type) << 2)};
  append_le(out, static_cast<uint64_t>(v), width);
  return out;
}

bytes encode_string(std::string const& s)
{
  bytes out;
  if (s.size() <= 63) {
    out.push_back(static_cast<uint8_t>(s.size() << 2 | 1));
  } else {
    out.push_back(static_cast<uint8_t>(pqe::variant_primitive_type::LONG_STRING) << 2);
    append_le(out, s.size(), 4);
  }
  out.insert(out.end(), s.begin(), s.end());
  return out;
}

bytes encode_array(std::vector<bytes> const& elements)
{
  uint64_t data_size = 0;
  for (auto const& e : elements) {
    data_size += e.size();
  }
  auto const large = elements.size() > 0xFF;
  auto const width = width_of(data_size);
  bytes out{static_cast<uint8_t>((large << 4) | ((width - 1) << 2) | 3)};
  append_le(out, elements.size(), large ? 4 : 1);
  uint64_t offset = 0;
  for (auto const& e : elements) {
    append_le(out, offset, width);
    offset += e.size();
  }
  append_le(out, offset, width);
  for (auto const& e : elements) {
    out.insert(out.end(), e.begin(), e.end());
  }
  return out;
}

/**
 * Field values are stored in document order, while the id and offset lists follow key order.
 * `ids[i]` and `positions[i]` are the dictionary id and the document position of the i-th field in
 * key order; `fields` holds the encoded values in document order.
 */
bytes encode_object(std::vector<uint32_t> const& ids,
                    std::vector<std::size_t> const& positions,
                    std::vector<bytes> const& fields)
{
  std::vector<uint64_t> offsets;
  uint64_t data_size = 0;
  for (auto const& f : fields) {
    offsets.push_back(data_size);
    data_size += f.size();
  }
  auto const large    = fields.size() > 0xFF;
  auto const id_width = width_of(ids.empty() ? 0 : *std::max_element(ids.begin(), ids.end()));
  auto const width    = width_of(data_size);
  bytes out{static_cast<uint8_t>((large << 6) | ((id_width - 1) << 4) | ((width - 1) << 2) | 2)};
  append_le(out, fields.size(), large ? 4 : 1);
  for (auto id : ids) {
    append_le(out, id, id_width);
  }
  for (auto p : positions) {
    append_le(out, offsets[p], width);
  }
  append_le(out, data_size, width);
  for (auto const& f : fields) {
    out.insert(out.end(), f.begin(), f.end());
  }
  return out;
}

bytes encode_metadata(std::vector<std::string> const& keys)
{
  uint64_t dict_size = 0;
  for (auto const& k : keys) {
    dict_size += k.size();
  }
  auto const width = width_of(std::max<uint64_t>(dict_size, keys.size()));
  bytes out{static_cast<uint8_t>(((width - 1) << 6) | 1)};
  append_le(out, keys.size(), width);
  uint64_t offset = 0;
  for (auto const& k : keys) {
    append_le(out, offset, width);
    offset += k.size();
  }
  append_le(out, offset, width);
  for (auto const& k : keys) {
    out.insert(out.end(), k.begin(), k.end());
  }
  return out;
}

bytes const empty_metadata{0x01, 0x00, 0x00};

std::vector<std::string> golden_inputs()
{
  std::vector<std::string> inputs;
  for (auto const& c : json_to_variant_golden_cases) {
    inputs.emplace_back(c.input);
  }
  return inputs;
}

void expect_golden(std::vector<host_row> const& rows,
                   std::size_t first_case,
                   bool allow_duplicate_keys)
{
  for (std::size_t i = 0; i < rows.size(); ++i) {
    auto const& c        = json_to_variant_golden_cases[first_case + i];
    auto const& row      = rows[i];
    auto const* metadata = allow_duplicate_keys ? c.dup_metadata : c.strict_metadata;
    auto const* value    = allow_duplicate_keys ? c.dup_value : c.strict_value;
    SCOPED_TRACE("golden case " + std::to_string(first_case + i) + ": " + std::string(c.input));
    if (metadata == nullptr) {
      EXPECT_FALSE(row.valid);
      EXPECT_NE(row.status, variant_operation_status::SUCCESS);
    } else {
      ASSERT_TRUE(row.valid);
      EXPECT_EQ(row.status, variant_operation_status::SUCCESS);
      EXPECT_EQ(row.metadata, from_hex(metadata));
      EXPECT_EQ(row.value, from_hex(value));
    }
  }
}

class scoped_env {
 public:
  scoped_env(char const* name, char const* value) : name_{name}
  {
    if (auto const* old = std::getenv(name); old != nullptr) { old_ = old; }
    setenv(name, value, 1);
  }
  ~scoped_env()
  {
    if (old_) {
      setenv(name_, old_->c_str(), 1);
    } else {
      unsetenv(name_);
    }
  }

 private:
  char const* name_;
  std::optional<std::string> old_;
};

// A slice that starts and ends away from the table's bounds, to exercise a non-zero offset
constexpr cudf::size_type slice_begin = 37;
cudf::size_type slice_end(std::size_t num_cases)
{
  return static_cast<cudf::size_type>(num_cases) - 7;
}

}  // namespace

struct JsonToVariantTest : public cudf::test::BaseFixture {};

TEST_F(JsonToVariantTest, GoldenStrict)
{
  auto const inputs = golden_inputs();
  cudf::test::strings_column_wrapper input(inputs.begin(), inputs.end());
  expect_golden(convert(input, false), 0, false);
}

TEST_F(JsonToVariantTest, GoldenAllowDuplicateKeys)
{
  auto const inputs = golden_inputs();
  cudf::test::strings_column_wrapper input(inputs.begin(), inputs.end());
  expect_golden(convert(input, true), 0, true);
}

TEST_F(JsonToVariantTest, GoldenSliced)
{
  auto const inputs = golden_inputs();
  cudf::test::strings_column_wrapper input(inputs.begin(), inputs.end());
  auto const sliced = cudf::slice(input, {slice_begin, slice_end(inputs.size())}).front();
  expect_golden(convert(sliced, false), slice_begin, false);
}

TEST_F(JsonToVariantTest, GoldenBatched)
{
  scoped_env const env{"LIBCUDF_JSON_TO_VARIANT_BATCH_SIZE", "300"};
  auto const inputs = golden_inputs();
  cudf::test::strings_column_wrapper input(inputs.begin(), inputs.end());
  expect_golden(convert(input, false), 0, false);
  expect_golden(convert(input, true), 0, true);
  auto const sliced = cudf::slice(input, {slice_begin, slice_end(inputs.size())}).front();
  expect_golden(convert(sliced, false), slice_begin, false);
}

TEST_F(JsonToVariantTest, Empty)
{
  cudf::test::strings_column_wrapper input{};
  auto status = cudf::make_numeric_column(cudf::data_type{cudf::type_id::UINT8}, 0);
  auto const result =
    pqe::parse_json_to_variant(cudf::strings_column_view{input}, false, status->mutable_view());
  EXPECT_EQ(result->size(), 0);
  ASSERT_EQ(result->num_children(), 2);
  EXPECT_EQ(result->child(0).type().id(), cudf::type_id::LIST);
  EXPECT_EQ(result->child(1).type().id(), cudf::type_id::LIST);
}

TEST_F(JsonToVariantTest, NullRows)
{
  cudf::test::strings_column_wrapper input({"1", "", "{\"a\":[1]}", "garbage", "", "\"x\""},
                                           {true, false, true, true, false, true});
  auto const rows = convert(input, false);
  EXPECT_EQ(rows[0].status, variant_operation_status::SUCCESS);
  EXPECT_EQ(rows[0].value, encode_int(1));
  EXPECT_EQ(rows[1].status, variant_operation_status::ROW_NULL);
  EXPECT_FALSE(rows[1].valid);
  EXPECT_EQ(rows[2].status, variant_operation_status::SUCCESS);
  EXPECT_EQ(rows[2].metadata, encode_metadata({"a"}));
  EXPECT_EQ(rows[2].value, encode_object({0}, {0}, {encode_array({encode_int(1)})}));
  EXPECT_EQ(rows[3].status, variant_operation_status::INVALID_JSON);
  EXPECT_FALSE(rows[3].valid);
  EXPECT_EQ(rows[4].status, variant_operation_status::ROW_NULL);
  EXPECT_EQ(rows[5].status, variant_operation_status::SUCCESS);
  EXPECT_EQ(rows[5].value, encode_string("x"));
}

TEST_F(JsonToVariantTest, NoStatusColumn)
{
  cudf::test::strings_column_wrapper input({"[1,2]", "[1,", "{\"a\":1,\"a\":2}"});
  auto const result = pqe::parse_json_to_variant(cudf::strings_column_view{input}, false);
  EXPECT_EQ(result->size(), 3);
  EXPECT_EQ(result->null_count(), 2);
}

TEST_F(JsonToVariantTest, InvalidStatusColumn)
{
  cudf::test::strings_column_wrapper input({"1", "2"});
  auto wrong_type = cudf::make_numeric_column(cudf::data_type{cudf::type_id::INT32}, 2);
  EXPECT_THROW((void)pqe::parse_json_to_variant(
                 cudf::strings_column_view{input}, false, wrong_type->mutable_view()),
               std::invalid_argument);
  auto wrong_size = cudf::make_numeric_column(cudf::data_type{cudf::type_id::UINT8}, 3);
  EXPECT_THROW((void)pqe::parse_json_to_variant(
                 cudf::strings_column_view{input}, false, wrong_size->mutable_view()),
               std::invalid_argument);
  auto nullable = cudf::make_numeric_column(
    cudf::data_type{cudf::type_id::UINT8}, 2, cudf::mask_state::ALL_VALID);
  EXPECT_THROW((void)pqe::parse_json_to_variant(
                 cudf::strings_column_view{input}, false, nullable->mutable_view()),
               std::invalid_argument);
}

TEST_F(JsonToVariantTest, DuplicateKeys)
{
  std::string const json = R"({"b":1,"a":2,"b":3})";
  auto const strict      = convert_one(json, false);
  EXPECT_EQ(strict.status, variant_operation_status::DUPLICATE_KEY);
  EXPECT_FALSE(strict.valid);

  // The last occurrence wins; the dictionary keeps keys in order of first use
  auto const dup = convert_one(json, true);
  EXPECT_EQ(dup.status, variant_operation_status::SUCCESS);
  EXPECT_EQ(dup.metadata, encode_metadata({"b", "a"}));
  EXPECT_EQ(dup.value, encode_object({1, 0}, {0, 1}, {encode_int(2), encode_int(3)}));
}

TEST_F(JsonToVariantTest, MixedTypesAcrossRows)
{
  cudf::test::strings_column_wrapper input(
    {R"({"a":1})", R"({"a":"x"})", R"({"a":[1,true]})", R"({"a":{"b":null}})", R"({"a":1.5})"});
  auto const rows = convert(input, false);
  bytes const true_value{static_cast<uint8_t>(pqe::variant_primitive_type::BOOLEAN_TRUE) << 2};
  bytes const decimal_value{
    static_cast<uint8_t>(static_cast<uint8_t>(pqe::variant_primitive_type::DECIMAL4) << 2),
    1,
    15,
    0,
    0,
    0};
  std::vector<bytes> const expected{
    encode_object({0}, {0}, {encode_int(1)}),
    encode_object({0}, {0}, {encode_string("x")}),
    encode_object({0}, {0}, {encode_array({encode_int(1), true_value})}),
    encode_object({0}, {0}, {encode_object({1}, {0}, {bytes{0}})}),
    encode_object({0}, {0}, {decimal_value})};
  for (std::size_t r = 0; r < rows.size(); ++r) {
    EXPECT_EQ(rows[r].status, variant_operation_status::SUCCESS) << r;
    EXPECT_EQ(
      rows[r].metadata,
      encode_metadata(r == 3 ? std::vector<std::string>{"a", "b"} : std::vector<std::string>{"a"}))
      << r;
    EXPECT_EQ(rows[r].value, expected[r]) << r;
  }
}

TEST_F(JsonToVariantTest, TrailingContent)
{
  cudf::test::strings_column_wrapper input(
    {R"("b":-1,"c":[]})", "[1]]", "1 2", "true]", "null x", "1,", "1x", "\"a\"\x1e[", "1\x1e"});
  auto const rows = convert(input, false);
  EXPECT_EQ(rows[0].value, encode_string("b"));
  EXPECT_EQ(rows[1].value, encode_array({encode_int(1)}));
  EXPECT_EQ(rows[2].value, encode_int(1));
  EXPECT_EQ(rows[3].value,
            bytes{static_cast<uint8_t>(pqe::variant_primitive_type::BOOLEAN_TRUE) << 2});
  EXPECT_EQ(rows[4].value, bytes{0});
  EXPECT_EQ(rows[5].status, variant_operation_status::INVALID_JSON);
  EXPECT_EQ(rows[6].status, variant_operation_status::INVALID_JSON);
  EXPECT_EQ(rows[7].value, encode_string("a"));
  EXPECT_EQ(rows[8].status, variant_operation_status::INVALID_JSON);
}

TEST_F(JsonToVariantTest, UnsupportedRootLiteral)
{
  auto const row = convert_one("true\xc3\xa9");
  EXPECT_EQ(row.status, variant_operation_status::UNSUPPORTED_INPUT);
  EXPECT_FALSE(row.valid);
}

TEST_F(JsonToVariantTest, NestingDepth)
{
  auto const nested = [](int lists, int objects) {
    std::string s;
    for (int i = 0; i < objects; ++i) {
      s += "{\"k\":";
    }
    s += std::string(lists, '[') + std::string(lists, ']');
    for (int i = 0; i < objects; ++i) {
      s += "}";
    }
    return s;
  };
  // Lists count once and objects twice towards the 127-level limit of the tree builder
  cudf::test::strings_column_wrapper input({nested(127, 0),
                                            nested(128, 0),
                                            nested(1, 63),
                                            nested(2, 63),
                                            nested(1000, 0),
                                            nested(1001, 0),
                                            "[\"" + std::string(500, '[') + "\"]"});
  auto const rows = convert(input, false);
  EXPECT_EQ(rows[0].status, variant_operation_status::SUCCESS);
  EXPECT_EQ(rows[1].status, variant_operation_status::UNSUPPORTED_INPUT);
  EXPECT_EQ(rows[2].status, variant_operation_status::SUCCESS);
  EXPECT_EQ(rows[3].status, variant_operation_status::UNSUPPORTED_INPUT);
  EXPECT_EQ(rows[4].status, variant_operation_status::UNSUPPORTED_INPUT);
  EXPECT_EQ(rows[5].status, variant_operation_status::INVALID_JSON);
  EXPECT_EQ(rows[6].status, variant_operation_status::SUCCESS);
  EXPECT_EQ(rows[6].value, encode_array({encode_string(std::string(500, '['))}));

  auto expected = encode_array({});
  for (int i = 1; i < 127; ++i) {
    expected = encode_array({expected});
  }
  EXPECT_EQ(rows[0].value, expected);
}

TEST_F(JsonToVariantTest, NestingDepthStringsAndTrailingContent)
{
  // Many brackets in total, but shallow: the exact depth check must not reject these
  std::string wide = "[";
  for (int i = 0; i < 300; ++i) {
    wide += (i ? "," : "") + std::string{R"({"a":[{}]})"};
  }
  wide += "]";
  std::vector<std::string> inputs{
    wide, "[]" + std::string(300, '['), "{} " + std::string(100, '{')};
  std::vector<variant_operation_status> expected(inputs.size(), variant_operation_status::SUCCESS);

  // Escapes before a quote, with the backslash run crossing every position of a 32-byte chunk
  auto const deep = std::string(130, '[') + std::string(130, ']');
  for (int pad = 0; pad < 40; ++pad) {
    auto const prefix = "[\"" + std::string(pad, 'x');
    // `\\"` closes the string, so the brackets after it nest
    inputs.push_back(prefix + R"(\\",)" + deep + "]");
    expected.push_back(variant_operation_status::UNSUPPORTED_INPUT);
    // `\"` does not close the string, so the brackets stay quoted
    inputs.push_back(prefix + R"(\")" + std::string(200, '[') + "\"]");
    expected.push_back(variant_operation_status::SUCCESS);
    // `\\\"` does not close it either
    inputs.push_back(prefix + R"(\\\")" + std::string(200, '{') + "\"]");
    expected.push_back(variant_operation_status::SUCCESS);
  }

  cudf::test::strings_column_wrapper input(inputs.begin(), inputs.end());
  auto const rows = convert(input, false);
  for (std::size_t r = 0; r < rows.size(); ++r) {
    EXPECT_EQ(rows[r].status, expected[r]) << inputs[r].substr(0, 60);
  }
  EXPECT_EQ(rows[1].value, encode_array({}));
}

TEST_F(JsonToVariantTest, WideArray)
{
  std::string json = "[";
  std::vector<bytes> elements;
  for (int i = 0; i < 300; ++i) {
    json += (i ? "," : "") + std::to_string(i * 250);
    elements.push_back(encode_int(i * 250));
  }
  json += "]";
  auto const row = convert_one(json);
  EXPECT_EQ(row.status, variant_operation_status::SUCCESS);
  EXPECT_EQ(row.metadata, empty_metadata);
  EXPECT_EQ(row.value, encode_array(elements));
}

TEST_F(JsonToVariantTest, WideObject)
{
  // 300 keys, written in reverse order so that sorting matters; the dictionary is in order of use
  std::string json = "{";
  std::vector<std::string> keys;
  for (int i = 299; i >= 0; --i) {
    auto const key = "k" + std::string(3 - std::to_string(i).size(), '0') + std::to_string(i);
    json += (i != 299 ? "," : "") + ("\"" + key + "\":" + std::to_string(i));
    keys.push_back(key);
  }
  json += "}";
  std::vector<uint32_t> ids;
  std::vector<std::size_t> positions;
  std::vector<bytes> fields;
  for (int i = 0; i < 300; ++i) {
    ids.push_back(299 - i);
    positions.push_back(299 - i);
    fields.push_back(encode_int(299 - i));
  }
  auto const row = convert_one(json);
  EXPECT_EQ(row.status, variant_operation_status::SUCCESS);
  EXPECT_EQ(row.metadata, encode_metadata(keys));
  EXPECT_EQ(row.value, encode_object(ids, positions, fields));
}

TEST_F(JsonToVariantTest, WideOffsets)
{
  // Offsets of 2 and 3 bytes
  std::string const s70k(70'000, 'x');
  std::string const s20m(17'000'000, 'y');
  cudf::test::strings_column_wrapper input(
    {"[\"" + s70k + "\",1]", "{\"a\":\"" + s20m + "\",\"b\":[\"" + s70k + "\"]}"});
  auto const rows = convert(input, false);
  EXPECT_EQ(rows[0].status, variant_operation_status::SUCCESS);
  EXPECT_EQ(rows[0].value, encode_array({encode_string(s70k), encode_int(1)}));
  EXPECT_EQ(rows[1].status, variant_operation_status::SUCCESS);
  EXPECT_EQ(rows[1].metadata, encode_metadata({"a", "b"}));
  EXPECT_EQ(
    rows[1].value,
    encode_object({0, 1}, {0, 1}, {encode_string(s20m), encode_array({encode_string(s70k)})}));
}

TEST_F(JsonToVariantTest, UnicodeKeyOrder)
{
  // UTF-16 order puts U+1F600 (a surrogate pair) before U+E000, unlike UTF-8 byte order
  auto const row = convert_one("{\"\xf0\x9f\x98\x80\":1,\"\xee\x80\x80\":2,\"z\":3}");
  EXPECT_EQ(row.status, variant_operation_status::SUCCESS);
  EXPECT_EQ(row.metadata, encode_metadata({"\xf0\x9f\x98\x80", "\xee\x80\x80", "z"}));
  EXPECT_EQ(row.value,
            encode_object({2, 0, 1}, {2, 0, 1}, {encode_int(1), encode_int(2), encode_int(3)}));
}

TEST_F(JsonToVariantTest, KeysWithPathSyntaxCharacters)
{
  // Keys come from the document itself, so the empty key and characters that are meaningful to
  // JSONPath are ordinary dictionary entries. Only the metadata differs between these rows.
  std::vector<std::pair<std::string, std::string>> const cases{{R"({"":1})", ""},
                                                               {R"({"a.b":1})", "a.b"},
                                                               {R"({"a[0]":1})", "a[0]"},
                                                               {R"({"$":1})", "$"},
                                                               {R"({"$.a":1})", "$.a"},
                                                               {R"({"a\"b":1})", "a\"b"}};
  for (auto const& [json, key] : cases) {
    auto const row = convert_one(json);
    EXPECT_EQ(row.status, variant_operation_status::SUCCESS) << json;
    EXPECT_EQ(row.metadata, encode_metadata({key})) << json;
    EXPECT_EQ(row.value, encode_object({0}, {0}, {encode_int(1)})) << json;
  }
}

TEST_F(JsonToVariantTest, Limits)
{
  std::string const long_name(50'001, 'n');
  cudf::test::strings_column_wrapper input({"\"" + std::string(20'000'000, 's') + "\"",
                                            "\"" + std::string(20'000'001, 's') + "\"",
                                            "{\"" + long_name.substr(1) + "\":1}",
                                            "{\"" + long_name + "\":1}",
                                            std::string(1000, '7'),
                                            std::string(1001, '7')});
  auto const rows = convert(input, false);
  EXPECT_EQ(rows[0].status, variant_operation_status::SUCCESS);
  EXPECT_EQ(rows[1].status, variant_operation_status::INVALID_JSON);
  EXPECT_EQ(rows[2].status, variant_operation_status::SUCCESS);
  EXPECT_EQ(rows[3].status, variant_operation_status::INVALID_JSON);
  EXPECT_EQ(rows[4].status, variant_operation_status::SUCCESS);
  EXPECT_EQ(rows[5].status, variant_operation_status::INVALID_JSON);
}

TEST_F(JsonToVariantTest, SizeLimit)
{
  // Eight strings of 17 MB encode to more than 128 MiB
  std::string const s(17'000'000, 'q');
  std::string json = "[";
  for (int i = 0; i < 8; ++i) {
    json += (i ? ",\"" : "\"") + s + "\"";
  }
  json += "]";
  cudf::test::strings_column_wrapper input({json, "[1]"});
  auto const rows = convert(input, false);
  EXPECT_EQ(rows[0].status, variant_operation_status::SIZE_LIMIT);
  EXPECT_FALSE(rows[0].valid);
  EXPECT_EQ(rows[1].status, variant_operation_status::SUCCESS);
  EXPECT_EQ(rows[1].value, encode_array({encode_int(1)}));
}

TEST_F(JsonToVariantTest, Numbers)
{
  using T             = pqe::variant_primitive_type;
  auto const decimal4 = [](uint8_t scale, int32_t unscaled) {
    bytes out{static_cast<uint8_t>(static_cast<uint8_t>(T::DECIMAL4) << 2), scale};
    append_le(out, static_cast<uint32_t>(unscaled), 4);
    return out;
  };
  auto const float64 = [](uint64_t bits) {
    bytes out{static_cast<uint8_t>(static_cast<uint8_t>(T::FLOAT64) << 2)};
    append_le(out, bits, 8);
    return out;
  };
  cudf::test::strings_column_wrapper input(
    {"127",
     "-129",
     "2147483648",
     "1.50",
     "1e0",
     "2.2250738585072011e-308",
     "1.00000000000000011102230246251565404236316680908203125",
     "1.00000000000000011102230246251565404236316680908203125e0",
     "2.4703282292062327e-324",
     "1e400",
     "01",
     "1.",
     "1.e5",
     "-",
     "NaN"});
  auto const rows = convert(input, false);
  EXPECT_EQ(rows[0].value, encode_int(127));
  EXPECT_EQ(rows[1].value, encode_int(-129));
  bytes int64_value{static_cast<uint8_t>(static_cast<uint8_t>(T::INT64) << 2)};
  append_le(int64_value, 2147483648ull, 8);
  EXPECT_EQ(rows[2].value, int64_value);
  EXPECT_EQ(rows[3].value, decimal4(2, 150));
  EXPECT_EQ(rows[4].value, float64(0x3FF0000000000000ull));
  // Just below the smallest normal double; rounds to the largest subnormal
  EXPECT_EQ(rows[5].value, float64(0x000FFFFFFFFFFFFFull));
  // 54 significant digits without an exponent is too precise for a decimal
  EXPECT_EQ(rows[6].value, float64(0x3FF0000000000000ull));
  // Exactly halfway between 1 and the next double: ties to even
  EXPECT_EQ(rows[7].value, float64(0x3FF0000000000000ull));
  // Just below half the smallest subnormal
  EXPECT_EQ(rows[8].value, float64(0));
  EXPECT_EQ(rows[9].value, float64(0x7FF0000000000000ull));
  for (int i = 10; i < 15; ++i) {
    EXPECT_EQ(rows[i].status, variant_operation_status::INVALID_JSON) << i;
  }
}
