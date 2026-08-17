/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "apache_variant_fixtures.hpp"

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>

#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/io/experimental/variant.hpp>
#include <cudf/io/experimental/variant_spec.hpp>
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/structs/structs_column_view.hpp>
#include <cudf/utilities/span.hpp>

#include <algorithm>
#include <array>
#include <bit>
#include <cstdio>
#include <cstring>
#include <format>
#include <memory>
#include <string>
#include <vector>

namespace avf = cudf::test::apache_variant_fixtures;

namespace {

// VARIANT value-header factory helpers.
//
// Every VARIANT value begins with a one-byte "value metadata" header. Its bits
// are (per the Apache Parquet variant spec [1]):
//
//   bit index:  7  6  5  4  3  2 | 1  0
//   field:      <- value_header ->|basic
//
//   - basic_type (low 2 bits): 0=primitive, 1=short_string, 2=object, 3=array
//   - value_header (high 6 bits): meaning depends on basic_type
//       * primitive    -> physical type id (variant_primitive_type below)
//       * short_string -> string length in bytes (0..63)
//       * object/array -> field-id / field-offset size flags
//
// The enums below let tests spell header bytes out by name (and avoid
// endianness ambiguity in the bit layout) instead of using magic numbers.
//
// [1] https://github.com/apache/parquet-format/blob/master/VariantEncoding.md
using cudf::io::parquet::experimental::variant_basic_type;
using cudf::io::parquet::experimental::variant_primitive_type;

// Compose a value-metadata header byte from a basic type and its 6-bit value_header.
constexpr uint8_t make_variant_header(variant_basic_type basic, uint8_t value_header)
{
  CUDF_EXPECTS(value_header <= 0x3F, "VARIANT value_header must fit in 6 bits");
  return static_cast<uint8_t>(static_cast<uint8_t>(basic) | (value_header << 2));
}

// Header byte for a primitive value of the given physical type.
constexpr uint8_t make_variant_primitive(variant_primitive_type type)
{
  return make_variant_header(variant_basic_type::PRIMITIVE, static_cast<uint8_t>(type));
}

// Header byte for a short string of the given length (must fit in 6 bits: 0..63).
constexpr uint8_t make_variant_short_string_header(std::size_t length)
{
  CUDF_EXPECTS(length <= 0x3F, "VARIANT short string length must fit in 6 bits");
  return make_variant_header(variant_basic_type::SHORT_STRING, static_cast<uint8_t>(length));
}

// Header byte for an object value with 1-byte field ids and 1-byte offsets
// (is_large=false), i.e. value_header == 0.
constexpr uint8_t make_variant_object_header()
{
  return make_variant_header(variant_basic_type::OBJECT, 0);
}

// Build a struct `column_view` over (metadata, value) without copying.
inline cudf::column_view wrap_variant_view(cudf::column_view const& metadata,
                                           cudf::column_view const& value)
{
  CUDF_EXPECTS(metadata.size() == value.size(),
               "metadata and value columns must have the same number of rows");
  return cudf::column_view{cudf::data_type{cudf::type_id::STRUCT},
                           value.size(),
                           nullptr,
                           nullptr,
                           0,
                           0,
                           {metadata, value}};
}

// Wrap a single-row (metadata, value) pair as a VARIANT struct column.
inline cudf::test::structs_column_wrapper wrap_single_variant(std::vector<uint8_t> const& meta,
                                                              std::vector<uint8_t> const& val)
{
  cudf::test::lists_column_wrapper<uint8_t> m(meta.begin(), meta.end());
  cudf::test::lists_column_wrapper<uint8_t> v(val.begin(), val.end());
  return cudf::test::structs_column_wrapper{{m, v}};
}

// Wrap an Apache parquet-testing fixture into a single-row VARIANT struct column.
template <std::size_t M, std::size_t V>
cudf::test::structs_column_wrapper make_apache_variant(avf::fixture<M, V> const& f)
{
  cudf::test::lists_column_wrapper<uint8_t> m(f.metadata.begin(), f.metadata.end());
  cudf::test::lists_column_wrapper<uint8_t> v(f.value.begin(), f.value.end());
  return cudf::test::structs_column_wrapper{{m, v}};
}

// Three-row VARIANT fixture reused by multiple multi-row tests below.
//   Row 0: dict {x,y}, value { x: INT32(7),  y: "hi"  }
//   Row 1: dict {x,z}, value { x: INT32(42), z: INT32(99) }
//   Row 2: dict {y},   value { y: "zzz" }
inline cudf::test::structs_column_wrapper make_xyz_three_row_variant()
{
  std::vector<uint8_t> const m1 = {0x01, 0x02, 0x00, 0x01, 0x02, 'x', 'y'};
  std::vector<uint8_t> const v1 = {
    0x02, 0x02, 0x00, 0x01, 0x00, 0x05, 0x08, 0x14, 0x07, 0x00, 0x00, 0x00, 0x09, 'h', 'i'};
  std::vector<uint8_t> const m2 = {0x01, 0x02, 0x00, 0x01, 0x02, 'x', 'z'};
  // clang-format off
  std::vector<uint8_t> const v2 = {
    0x02, 0x02, 0x00, 0x01, 0x00, 0x05, 0x0a,
    0x14, 0x2a, 0x00, 0x00, 0x00,
    0x14, 0x63, 0x00, 0x00, 0x00};
  // clang-format on
  std::vector<uint8_t> const m3 = {0x01, 0x01, 0x00, 0x01, 'y'};
  std::vector<uint8_t> const v3 = {0x02, 0x01, 0x00, 0x00, 0x04, 0x0d, 'z', 'z', 'z'};

  cudf::test::lists_column_wrapper<uint8_t> meta{
    {m1.begin(), m1.end()}, {m2.begin(), m2.end()}, {m3.begin(), m3.end()}};
  cudf::test::lists_column_wrapper<uint8_t> val{
    {v1.begin(), v1.end()}, {v2.begin(), v2.end()}, {v3.begin(), v3.end()}};
  return cudf::test::structs_column_wrapper{{meta, val}};
}

}  // namespace

struct ExtractVariantFieldTest : public cudf::test::BaseFixture {};

TEST_F(ExtractVariantFieldTest, NullStructRow)
{
  std::vector<uint8_t> const m = {0x01, 0x01, 0x00, 0x01, 'x'};
  // Row 0 object also exercises a mismatched field_id_size (2 bytes) vs field_offset_size (1 byte)
  std::vector<uint8_t> const v = {0x12, 0x01, 0x00, 0x00, 0x00, 0x05, 0x14, 0x07, 0x00, 0x00, 0x00};
  cudf::test::lists_column_wrapper<uint8_t> meta{{m.begin(), m.end()}, {0x00}};
  cudf::test::lists_column_wrapper<uint8_t> val{{v.begin(), v.end()}, {0x00}};
  // Use the validity vector to mask the second row null.
  cudf::test::structs_column_wrapper col{{meta, val}, std::vector<bool>{true, false}};

  auto got = cudf::io::parquet::experimental::extract_variant_field(
    col, "x", cudf::data_type{cudf::type_id::INT32}, cudf::test::get_default_stream());

  cudf::test::fixed_width_column_wrapper<int32_t> expected({7, 0}, {true, false});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*got, expected);
}

TEST_F(ExtractVariantFieldTest, NonObjectValueYieldsNull)
{
  std::vector<uint8_t> const metab = {0x01, 0x01, 0x00, 0x01, static_cast<uint8_t>('x')};
  // Primitive int32 only (not wrapped in object)
  std::vector<uint8_t> const valb = {0x14, 0x07, 0x00, 0x00, 0x00};
  auto col                        = wrap_single_variant(metab, valb);

  auto got = cudf::io::parquet::experimental::extract_variant_field(
    col, "x", cudf::data_type{cudf::type_id::INT32}, cudf::test::get_default_stream());

  cudf::test::fixed_width_column_wrapper<int32_t> expected({0}, {false});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*got, expected);
}

TEST_F(ExtractVariantFieldTest, InvalidMetadataYieldsNull)
{
  // Too short to be valid VARIANT metadata V1
  std::vector<uint8_t> const metab = {0x02};
  std::vector<uint8_t> const valb  = {0x02, 0x01, 0x00, 0x00, 0x05, 0x14, 0x07, 0x00, 0x00, 0x00};
  auto col                         = wrap_single_variant(metab, valb);

  auto got = cudf::io::parquet::experimental::extract_variant_field(
    col, "x", cudf::data_type{cudf::type_id::INT32}, cudf::test::get_default_stream());

  cudf::test::fixed_width_column_wrapper<int32_t> expected({0}, {false});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*got, expected);
}

TEST_F(ExtractVariantFieldTest, UnsupportedMetadataVersionYieldsNull)
{
  // Variant version != v1 must produce null on field lookup.
  std::vector<uint8_t> const metab = {0x02, 0x01, 0x00, 0x01, static_cast<uint8_t>('x')};
  std::vector<uint8_t> const valb  = {0x02, 0x01, 0x00, 0x00, 0x05, 0x14, 0x07, 0x00, 0x00, 0x00};
  auto col                         = wrap_single_variant(metab, valb);

  auto got = cudf::io::parquet::experimental::extract_variant_field(
    col, "x", cudf::data_type{cudf::type_id::INT32}, cudf::test::get_default_stream());

  cudf::test::fixed_width_column_wrapper<int32_t> expected({0}, {false});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*got, expected);
}

TEST_F(ExtractVariantFieldTest, TruncatedObjectValueYieldsNull)
{
  std::vector<uint8_t> const metab = {0x01, 0x01, 0x00, 0x01, static_cast<uint8_t>('x')};
  // Object header only (truncated)
  std::vector<uint8_t> const valb = {0x02};
  auto col                        = wrap_single_variant(metab, valb);

  auto got = cudf::io::parquet::experimental::extract_variant_field(
    col, "x", cudf::data_type{cudf::type_id::INT32}, cudf::test::get_default_stream());

  cudf::test::fixed_width_column_wrapper<int32_t> expected({0}, {false});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*got, expected);
}

TEST_F(ExtractVariantFieldTest, MultiRow)
{
  auto col    = make_xyz_three_row_variant();
  auto stream = cudf::test::get_default_stream();
  auto x      = cudf::io::parquet::experimental::extract_variant_field(
    col, "x", cudf::data_type{cudf::type_id::INT32}, stream);
  cudf::test::fixed_width_column_wrapper<int32_t> x_exp({7, 42, 0}, {true, true, false});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*x, x_exp);

  auto y = cudf::io::parquet::experimental::extract_variant_field(
    col, "y", cudf::data_type{cudf::type_id::STRING}, stream);
  cudf::test::strings_column_wrapper y_exp({"hi", "", "zzz"}, {true, false, true});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*y, y_exp);

  auto z = cudf::io::parquet::experimental::extract_variant_field(
    col, "z", cudf::data_type{cudf::type_id::INT32}, stream);
  cudf::test::fixed_width_column_wrapper<int32_t> z_exp({0, 99, 0}, {false, true, false});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*z, z_exp);
}

TEST_F(ExtractVariantFieldTest, SlicedInput)
{
  // Slice rows [1, 3); extracted column must reflect the slice, not the underlying child rows.
  auto const col    = make_xyz_three_row_variant();
  auto const sliced = cudf::slice(col, {1, 3}).front();

  auto got = cudf::io::parquet::experimental::extract_variant_field(
    sliced, "x", cudf::data_type{cudf::type_id::INT32}, cudf::test::get_default_stream());

  cudf::test::fixed_width_column_wrapper<int32_t> expected({42, 0}, {true, false});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*got, expected);
}

TEST_F(ExtractVariantFieldTest, ApacheObjectPrimitiveStringFields)
{
  auto col     = make_apache_variant(avf::object_primitive);
  auto stream  = cudf::test::get_default_stream();
  auto const s = cudf::data_type{cudf::type_id::STRING};

  for (auto const& [field, expected_str] :
       {std::pair{"string_field", "Apache Parquet"},
        std::pair{"timestamp_field", "2025-04-16T12:34:56.78"}}) {
    SCOPED_TRACE(std::string{"field: "} + field);
    auto got = cudf::io::parquet::experimental::extract_variant_field(col, field, s, stream);
    cudf::test::strings_column_wrapper expected({expected_str});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*got, expected);
  }
}

TEST_F(ExtractVariantFieldTest, ApacheObjectPrimitiveNullCases)
{
  auto col     = make_apache_variant(avf::object_primitive);
  auto stream  = cudf::test::get_default_stream();
  auto const s = cudf::data_type{cudf::type_id::STRING};

  for (auto const& field : {"no_such_field", "null_field"}) {
    SCOPED_TRACE(std::string{"field: "} + field);
    auto got = cudf::io::parquet::experimental::extract_variant_field(col, field, s, stream);
    ASSERT_EQ(got->size(), 1);
    EXPECT_EQ(got->null_count(), 1);
  }
}

TEST_F(ExtractVariantFieldTest, ApacheObjectPrimitiveIntField)
{
  auto col = make_apache_variant(avf::object_primitive);
  auto got = cudf::io::parquet::experimental::extract_variant_field(
    col, "int_field", cudf::data_type{cudf::type_id::INT8}, cudf::test::get_default_stream());
  cudf::test::fixed_width_column_wrapper<int8_t> expected{int8_t{1}};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*got, expected);
}

TEST_F(ExtractVariantFieldTest, ApacheObjectNested)
{
  auto col         = make_apache_variant(avf::object_nested);
  auto stream      = cudf::test::get_default_stream();
  auto const check = [&](char const* path, auto expected_val) {
    using T = decltype(expected_val);
    SCOPED_TRACE(std::string{"path: "} + path);
    if constexpr (std::is_same_v<T, char const*>) {
      auto got = cudf::io::parquet::experimental::extract_variant_field(
        col, path, cudf::data_type{cudf::type_id::STRING}, stream);
      cudf::test::strings_column_wrapper expected({expected_val});
      CUDF_TEST_EXPECT_COLUMNS_EQUAL(*got, expected);
    } else {
      auto got = cudf::io::parquet::experimental::extract_variant_field(
        col, path, cudf::data_type{cudf::type_to_id<T>()}, stream);
      cudf::test::fixed_width_column_wrapper<T> expected{expected_val};
      CUDF_TEST_EXPECT_COLUMNS_EQUAL(*got, expected);
    }
  };

  check("$.observation.location", "In the Volcano");
  check("$.observation.time", "12:34:56");
  check("$.observation.value.temperature", int8_t{123});
  check("$.species.name", "lava monster");
  check("$.species.population", int16_t{6789});
  check("$.id", int8_t{1});
}

TEST_F(ExtractVariantFieldTest, ApacheObjectEmpty)
{
  auto col = make_apache_variant(avf::object_empty);
  auto got = cudf::io::parquet::experimental::extract_variant_field(
    col, "foo", cudf::data_type{cudf::type_id::STRING}, cudf::test::get_default_stream());
  ASSERT_EQ(got->size(), 1);
  EXPECT_EQ(got->null_count(), 1);
}

TEST_F(ExtractVariantFieldTest, ApacheObjectNestedChainedCalls)
{
  auto col    = make_apache_variant(avf::object_nested);
  auto stream = cudf::test::get_default_stream();

  auto single = cudf::io::parquet::experimental::get_variant_field(
    col, "$.observation.value.temperature", stream);

  auto const meta_v = cudf::structs_column_view{col}.get_sliced_child(0, stream);
  auto obs  = cudf::io::parquet::experimental::get_variant_field(col, "observation", stream);
  auto vobj = cudf::io::parquet::experimental::get_variant_field(
    wrap_variant_view(meta_v, obs->view()), "value", stream);
  auto chained = cudf::io::parquet::experimental::get_variant_field(
    wrap_variant_view(meta_v, vobj->view()), "temperature", stream);

  EXPECT_EQ(single->type().id(), cudf::type_id::LIST);
  EXPECT_EQ(chained->type().id(), cudf::type_id::LIST);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*single, *chained);
}

TEST_F(ExtractVariantFieldTest, ApacheObjectNestedMissingIntermediate)
{
  auto col    = make_apache_variant(avf::object_nested);
  auto stream = cudf::test::get_default_stream();

  auto got = cudf::io::parquet::experimental::extract_variant_field(
    col, "$.species.nope", cudf::data_type{cudf::type_id::STRING}, stream);

  cudf::test::strings_column_wrapper expected({"donotread"}, {false});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*got, expected);
}

TEST_F(ExtractVariantFieldTest, NestedPathNonObjectIntermediate)
{
  // Dict = {a, b}; value = { a: INT32(5), b: "hi" }
  std::vector<uint8_t> const metab = {0x01, 0x02, 0x00, 0x01, 0x02, 'a', 'b'};
  std::vector<uint8_t> const valb  = {
    0x02, 0x02, 0x00, 0x01, 0x00, 0x05, 0x08, 0x14, 0x05, 0x00, 0x00, 0x00, 0x09, 'h', 'i'};

  auto col = wrap_single_variant(metab, valb);
  // Descending into "a" fails because it is a primitive, not an object.
  auto got = cudf::io::parquet::experimental::extract_variant_field(
    col, "$.a.b", cudf::data_type{cudf::type_id::INT32}, cudf::test::get_default_stream());

  cudf::test::fixed_width_column_wrapper<int32_t> expected({0}, {false});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*got, expected);
}

TEST_F(ExtractVariantFieldTest, BareNameEqualsDollarPath)
{
  auto col    = make_xyz_three_row_variant();
  auto stream = cudf::test::get_default_stream();

  auto bare   = cudf::io::parquet::experimental::get_variant_field(col, "x", stream);
  auto dollar = cudf::io::parquet::experimental::get_variant_field(col, "$.x", stream);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*bare, *dollar);
}

namespace {

// INT32 primitive blob: primitive int32 header + little-endian 4-byte payload.
inline std::vector<uint8_t> enc_int32(int32_t v)
{
  auto const u = static_cast<uint32_t>(v);
  return {make_variant_primitive(variant_primitive_type::INT32),
          static_cast<uint8_t>(u & 0xff),
          static_cast<uint8_t>((u >> 8) & 0xff),
          static_cast<uint8_t>((u >> 16) & 0xff),
          static_cast<uint8_t>((u >> 24) & 0xff)};
}

// Short-string primitive blob (single-byte header).
inline std::vector<uint8_t> enc_short_string(std::string_view s)
{
  CUDF_EXPECTS(s.size() < 64, "short-string length must fit in 6 bits of the single-byte header");
  std::vector<uint8_t> out{make_variant_short_string_header(s.size())};
  out.insert(out.end(), s.begin(), s.end());
  return out;
}

// Append `width` little-endian bytes of `bits` to `out`.
inline void append_le(std::vector<uint8_t>& out, uint64_t bits, int width)
{
  for (int i = 0; i < width; ++i) {
    out.push_back(static_cast<uint8_t>((bits >> (8 * i)) & 0xff));
  }
}

// Primitive value blobs (header + fixed payload) for every physical type the cast matrix exercises.
inline std::vector<uint8_t> enc_null()
{
  return {make_variant_primitive(variant_primitive_type::NULLVAL)};
}

inline std::vector<uint8_t> enc_bool(bool b)
{
  return {make_variant_primitive(b ? variant_primitive_type::BOOLEAN_TRUE
                                   : variant_primitive_type::BOOLEAN_FALSE)};
}

inline std::vector<uint8_t> enc_int8(int8_t v)
{
  return {make_variant_primitive(variant_primitive_type::INT8), static_cast<uint8_t>(v)};
}

inline std::vector<uint8_t> enc_int16(int16_t v)
{
  std::vector<uint8_t> out{make_variant_primitive(variant_primitive_type::INT16)};
  append_le(out, static_cast<uint16_t>(v), 2);
  return out;
}

inline std::vector<uint8_t> enc_int64(int64_t v)
{
  std::vector<uint8_t> out{make_variant_primitive(variant_primitive_type::INT64)};
  append_le(out, static_cast<uint64_t>(v), 8);
  return out;
}

inline std::vector<uint8_t> enc_float64(double v)
{
  std::vector<uint8_t> out{make_variant_primitive(variant_primitive_type::FLOAT64)};
  append_le(out, std::bit_cast<uint64_t>(v), 8);
  return out;
}

// Long-string primitive blob: header + 4-byte LE length + payload.
inline std::vector<uint8_t> enc_long_string(std::string_view s)
{
  std::vector<uint8_t> out{make_variant_primitive(variant_primitive_type::LONG_STRING)};
  append_le(out, s.size(), 4);
  out.insert(out.end(),
             reinterpret_cast<uint8_t const*>(s.begin()),
             reinterpret_cast<uint8_t const*>(s.end()));
  return out;
}

// Build a single-field object value wrapping `inner` under field id `fid`.
// field_off_size=1, field_id_size=1, is_large=false.
inline std::vector<uint8_t> build_single_field_object(uint8_t fid,
                                                      std::vector<uint8_t> const& inner)
{
  CUDF_EXPECTS(inner.size() < 256, "inner blob too large for 1-byte offset header");
  // Header, num_elements, field_id, offset 0, sentinel = inner.size().
  std::vector<uint8_t> out{
    make_variant_object_header(), 0x01, fid, 0x00, static_cast<uint8_t>(inner.size())};
  out.insert(out.end(), inner.begin(), inner.end());
  return out;
}

// Build a VARIANT object blob with `n_fields` fields.  Field ids are 0..n_fields-1
// (in ascending order, matching the dictionary positions) and each field holds a bare INT32 equal
// to its field id.  Uses 1-byte field_id_size and 1-byte field_off_size; n_fields must be
// <= 51 so the total value bytes (5 * n_fields) still fit in 1-byte offsets.
inline std::vector<uint8_t> build_sequential_int32_object(int n_fields)
{
  std::vector<uint8_t> out{make_variant_object_header(), static_cast<uint8_t>(n_fields)};
  for (int fid = 0; fid < n_fields; ++fid) {
    out.push_back(static_cast<uint8_t>(fid));
  }
  for (int i = 0; i <= n_fields; ++i) {
    out.push_back(static_cast<uint8_t>(i * 5));
  }
  for (int fid = 0; fid < n_fields; ++fid) {
    auto const v = enc_int32(fid);
    out.insert(out.end(), v.begin(), v.end());
  }
  return out;
}

// Lexicographically ordered dictionary of N zero-padded two-digit keys "k<NN>".
inline std::vector<std::string> make_numeric_keys(int n)
{
  std::vector<std::string> out;
  out.reserve(n);
  for (int i = 0; i < n; ++i) {
    std::array<char, 8> buf{};
    std::snprintf(buf.data(), buf.size(), "k%02d", i);
    out.emplace_back(buf.data());
  }
  return out;
}

// Wrap per-row (metadata, value) byte vectors into a VARIANT struct column.  Built
// with make_lists_column + structs_column_wrapper directly so the helper stays
// self-contained within this test file for dynamic row counts.
inline cudf::test::structs_column_wrapper wrap_multi_row_variant(
  std::vector<std::vector<uint8_t>> const& meta_rows,
  std::vector<std::vector<uint8_t>> const& val_rows)
{
  auto build_list = [](std::vector<std::vector<uint8_t>> const& rows) {
    auto const n = static_cast<cudf::size_type>(rows.size());
    std::vector<int32_t> offsets(n + 1, 0);
    std::vector<uint8_t> flat;
    for (cudf::size_type i = 0; i < n; ++i) {
      flat.insert(flat.end(), rows[i].begin(), rows[i].end());
      offsets[i + 1] = static_cast<int32_t>(flat.size());
    }
    auto offs =
      cudf::test::fixed_width_column_wrapper<int32_t>(offsets.begin(), offsets.end()).release();
    auto data = cudf::test::fixed_width_column_wrapper<uint8_t>(flat.begin(), flat.end()).release();
    return cudf::make_lists_column(n, std::move(offs), std::move(data), 0, {});
  };
  std::vector<std::unique_ptr<cudf::column>> children;
  children.emplace_back(build_list(meta_rows));
  children.emplace_back(build_list(val_rows));
  return cudf::test::structs_column_wrapper{std::move(children)};
}

// Build a V1 VARIANT metadata blob for the given ordered string dictionary.
// Uses 2-byte offsets when total string length exceeds 255 bytes; 1-byte otherwise.
// Header bits [7:6] = offset_size_minus_one; bits [3:0] = version (1).
inline std::vector<uint8_t> build_metadata(std::vector<std::string> const& keys)
{
  constexpr uint8_t kVariantMetadataVersion  = 0x01;
  constexpr int kMetadataOffsetSizeShift     = 6;
  constexpr uint32_t kMaxSingleByteOffsetSum = 255u;

  uint32_t total_key_bytes = 0;
  for (auto const& key : keys) {
    total_key_bytes += static_cast<uint32_t>(key.size());
  }

  int const offset_size = (total_key_bytes > kMaxSingleByteOffsetSum) ? 2 : 1;
  std::vector<uint8_t> out{static_cast<uint8_t>(kVariantMetadataVersion |
                                                ((offset_size - 1) << kMetadataOffsetSizeShift))};

  auto write_little_endian_offset = [&](uint32_t value) {
    for (int byte_index = 0; byte_index < offset_size; ++byte_index) {
      out.push_back(static_cast<uint8_t>(value >> (8 * byte_index)));
    }
  };
  write_little_endian_offset(static_cast<uint32_t>(keys.size()));

  uint32_t running_offset = 0;
  write_little_endian_offset(0u);
  for (auto const& key : keys) {
    running_offset += static_cast<uint32_t>(key.size());
    write_little_endian_offset(running_offset);
  }

  for (auto const& key : keys) {
    out.insert(out.end(), key.begin(), key.end());
  }
  return out;
}

}  // namespace

TEST_F(ExtractVariantFieldTest, NestedPathMultiRowMixedNulls)
{
  // Row 0: { 1st: { foo-bar: INT32(1) } } -> path "$.1st.foo-bar" = 1.  Dictionary strings are
  // stored in non-lexicographic order ({"foo-bar", "1st"})
  auto const m0 = build_metadata({"foo-bar", "1st"});
  auto const v0 = build_single_field_object(
    /*fid=1st*/ 1, build_single_field_object(/*fid=foo-bar*/ 0, enc_int32(1)));
  // Row 1: { 1st: INT32(5) } -> non-object intermediate at "1st" -> null
  auto const m1 = build_metadata({"1st"});
  auto const v1 = build_single_field_object(/*fid=1st*/ 0, enc_int32(5));
  // Row 2: { q: INT32(7) } -> key "1st" missing from dict -> null
  auto const m2 = build_metadata({"q"});
  auto const v2 = build_single_field_object(/*fid=q*/ 0, enc_int32(7));

  cudf::test::lists_column_wrapper<uint8_t> meta{
    {m0.begin(), m0.end()}, {m1.begin(), m1.end()}, {m2.begin(), m2.end()}};
  cudf::test::lists_column_wrapper<uint8_t> val{
    {v0.begin(), v0.end()}, {v1.begin(), v1.end()}, {v2.begin(), v2.end()}};
  cudf::test::structs_column_wrapper col{{meta, val}};

  auto got = cudf::io::parquet::experimental::extract_variant_field(
    col, "$.1st.foo-bar", cudf::data_type{cudf::type_id::INT32}, cudf::test::get_default_stream());

  cudf::test::fixed_width_column_wrapper<int32_t> expected({1, 0, 0}, {true, false, false});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*got, expected);
}

TEST_F(ExtractVariantFieldTest, EmptyPathRejected)
{
  auto col    = wrap_single_variant(build_metadata({}), enc_int32(1));
  auto stream = cudf::test::get_default_stream();
  EXPECT_THROW(
    static_cast<void>(cudf::io::parquet::experimental::get_variant_field(col, "", stream)),
    std::invalid_argument);
  EXPECT_THROW(
    static_cast<void>(cudf::io::parquet::experimental::get_variant_field(col, "$", stream)),
    std::invalid_argument);
  EXPECT_THROW(static_cast<void>(cudf::io::parquet::experimental::extract_variant_field(
                 col, "", cudf::data_type{cudf::type_id::INT32}, stream)),
               std::invalid_argument);
}

TEST_F(ExtractVariantFieldTest, SyntaxErrors)
{
  auto col    = wrap_single_variant(build_metadata({}), enc_int32(1));
  auto stream = cudf::test::get_default_stream();
  // Object-key descent and array-index steps are supported; wildcards, quoted keys, negative
  // indices, out-of-range indices, and other malformed bracket forms must throw.
  for (auto const* bad : {"$..a",
                          "$.a[",
                          "$.a[]",
                          "$.",
                          "$['x']",
                          "$.a[*]",
                          "$.a[-1]",
                          "$.a[+1]",
                          "$.a[ 1]",
                          "$.a[01x]",
                          "$.a[1",
                          "$.a[99999999999999999999]"}) {
    EXPECT_THROW(
      static_cast<void>(cudf::io::parquet::experimental::get_variant_field(col, bad, stream)),
      std::invalid_argument)
      << "path that should have thrown: " << bad;
  }
}

TEST_F(ExtractVariantFieldTest, ApacheArrayPrimitiveIndexing)
{
  // array_primitive encodes the int8 array [2, 1, 5, 9]; index into it via "[N]" steps.
  auto col       = make_apache_variant(avf::array_primitive);
  auto stream    = cudf::test::get_default_stream();
  auto const i8  = cudf::data_type{cudf::type_id::INT8};
  auto const get = [&](char const* path) {
    return cudf::io::parquet::experimental::extract_variant_field(col, path, i8, stream);
  };

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*get("$[0]"),
                                 cudf::test::fixed_width_column_wrapper<int8_t>{int8_t{2}});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*get("$[2]"),
                                 cudf::test::fixed_width_column_wrapper<int8_t>{int8_t{5}});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*get("$[3]"),
                                 cudf::test::fixed_width_column_wrapper<int8_t>{int8_t{9}});

  // Leading zeros are allowed
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*get("$[00]"),
                                 cudf::test::fixed_width_column_wrapper<int8_t>{int8_t{2}});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*get("$[01]"),
                                 cudf::test::fixed_width_column_wrapper<int8_t>{int8_t{1}});

  // Out-of-bounds index resolves to null.
  cudf::test::fixed_width_column_wrapper<int8_t> const null_expected({0}, {false});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*get("$[4]"), null_expected);

  // Exercise 2-, 3-, and 4-byte offsets with a four-byte element count.
  for (uint8_t offset_size = 2; offset_size <= 4; ++offset_size) {
    auto const value_header = static_cast<uint8_t>(0x04 | (offset_size - 1));
    std::vector<uint8_t> value{static_cast<uint8_t>(0x03 | (value_header << 2)), 1, 0, 0, 0};
    value.insert(value.end(), offset_size, 0);  // offsets[0]
    value.push_back(2);                         // offsets[1]
    value.insert(value.end(), offset_size - 1, 0);
    value.insert(value.end(), {0x0c, 42});  // INT8(42)

    auto wide_col = wrap_single_variant(build_metadata({}), value);
    auto got = cudf::io::parquet::experimental::extract_variant_field(wide_col, "$[0]", i8, stream);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*got,
                                   cudf::test::fixed_width_column_wrapper<int8_t>{int8_t{42}});
  }
}

TEST_F(ExtractVariantFieldTest, ArrayIndexingTypeMismatchAndBounds)
{
  // array_primitive is the int8 array [2, 1, 5, 9]. An object-key step against an array, an
  // out-of-bounds index, and an index step against a non-array element all resolve to null.
  auto col      = make_apache_variant(avf::array_primitive);
  auto stream   = cudf::test::get_default_stream();
  auto const i8 = cudf::data_type{cudf::type_id::INT8};
  cudf::test::fixed_width_column_wrapper<int8_t> const null_expected({0}, {false});

  // Object-key descent into an array value: no such key -> null.
  auto key_on_array =
    cudf::io::parquet::experimental::extract_variant_field(col, "$.foo", i8, stream);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*key_on_array, null_expected);

  // Index step against a primitive element (after first descending into it): non-array -> null.
  auto index_on_primitive =
    cudf::io::parquet::experimental::extract_variant_field(col, "$[0][0]", i8, stream);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*index_on_primitive, null_expected);
}

TEST_F(ExtractVariantFieldTest, EmptyArrayIndexing)
{
  auto col      = make_apache_variant(avf::array_empty);
  auto stream   = cudf::test::get_default_stream();
  auto const i8 = cudf::data_type{cudf::type_id::INT8};
  cudf::test::fixed_width_column_wrapper<int8_t> const null_expected({0}, {false});

  for (auto const* path : {"$[0]", "$[1]"}) {
    SCOPED_TRACE(std::string{"path: "} + path);
    auto got = cudf::io::parquet::experimental::extract_variant_field(col, path, i8, stream);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*got, null_expected);
  }

  // Truncated counts/tables, decreasing offsets, and offsets beyond the values region yield null.
  for (auto const& value : std::vector<std::vector<uint8_t>>{{0x13},
                                                             {0x03, 0x01, 0x00},
                                                             {0x03, 0x01, 0x02, 0x01, 0x0c, 42},
                                                             {0x03, 0x01, 0x00, 0x03, 0x0c, 42}}) {
    auto malformed_col = wrap_single_variant(build_metadata({}), value);
    auto got =
      cudf::io::parquet::experimental::extract_variant_field(malformed_col, "$[0]", i8, stream);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*got, null_expected);
  }
}

TEST_F(ExtractVariantFieldTest, MixedObjectArrayTraversal)
{
  // array_nested encodes:
  //   [ {id:1, thing:{names:["Contrarian","Spider"]}},
  //     null,
  //     {id:2, names:["Apple","Ray",null], type:"if"} ]
  auto col    = make_apache_variant(avf::array_nested);
  auto stream = cudf::test::get_default_stream();

  auto const check_str = [&](char const* path, char const* expected) {
    SCOPED_TRACE(std::string{"path: "} + path);
    auto got = cudf::io::parquet::experimental::extract_variant_field(
      col, path, cudf::data_type{cudf::type_id::STRING}, stream);
    cudf::test::strings_column_wrapper const expected_col({expected});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*got, expected_col);
  };
  auto const check_null = [&](char const* path) {
    SCOPED_TRACE(std::string{"path: "} + path);
    auto got = cudf::io::parquet::experimental::extract_variant_field(
      col, path, cudf::data_type{cudf::type_id::STRING}, stream);
    cudf::test::strings_column_wrapper const null_col({""}, {false});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*got, null_col);
  };

  check_str("$[2].type", "if");
  check_str("$[0].thing.names[0]", "Contrarian");
  check_str("$[0].thing.names[1]", "Spider");
  check_str("$[2].names[0]", "Apple");
  check_str("$[2].names[1]", "Ray");

  check_null("$[1].id");        // element 1 is a JSON null
  check_null("$[0].name");      // element 0 has no "name" key (it has "thing")
  check_null("$[2].names[2]");  // third name is null
  check_null("$[2].names[3]");  // out-of-bounds array index
}

TEST_F(ExtractVariantFieldTest, LargeDictionaryAndObjectScan)
{
  auto const keys        = make_numeric_keys(50);
  auto const meta        = build_metadata(keys);
  auto const val         = build_sequential_int32_object(50);
  auto col               = wrap_single_variant(meta, val);
  auto stream            = cudf::test::get_default_stream();
  auto const int32_dtype = cudf::data_type{cudf::type_id::INT32};

  // First, middle, and last keys each decode to their own field id.
  auto first =
    cudf::io::parquet::experimental::extract_variant_field(col, "k00", int32_dtype, stream);
  auto mid =
    cudf::io::parquet::experimental::extract_variant_field(col, "k24", int32_dtype, stream);
  auto last =
    cudf::io::parquet::experimental::extract_variant_field(col, "k49", int32_dtype, stream);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*first, cudf::test::fixed_width_column_wrapper<int32_t>{0});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*mid, cudf::test::fixed_width_column_wrapper<int32_t>{24});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*last, cudf::test::fixed_width_column_wrapper<int32_t>{49});
}

TEST_F(ExtractVariantFieldTest, LargeDictionary100FieldsExtractLast)
{
  // 100-key dictionary "k00"..."k99" totals 300 string bytes (> 255), so build_metadata must emit
  // 2-byte offsets. The value is a flat 100-field object where field 99 ("k99") holds INT32(99)
  // and all other fields hold BOOLEAN_TRUE (1 byte), keeping value offsets within 1-byte range
  // (99 * 1 + 5 = 104 bytes).
  auto const keys = make_numeric_keys(100);
  auto const meta = build_metadata(keys);

  constexpr int field_count = 100;
  constexpr int target_fid  = 99;
  auto const target_val     = enc_int32(target_fid);

  std::vector<uint8_t> val{make_variant_object_header(), static_cast<uint8_t>(field_count)};
  for (int fid = 0; fid < field_count; ++fid) {
    val.push_back(static_cast<uint8_t>(fid));
  }
  uint8_t field_offset = 0;
  for (int fid = 0; fid < field_count; ++fid) {
    val.push_back(field_offset);
    field_offset =
      static_cast<uint8_t>(field_offset + (fid == target_fid ? target_val.size() : 1u));
  }
  val.push_back(field_offset);
  for (int fid = 0; fid < field_count; ++fid) {
    if (fid == target_fid) {
      val.insert(val.end(), target_val.begin(), target_val.end());
    } else {
      val.push_back(make_variant_primitive(variant_primitive_type::BOOLEAN_TRUE));
    }
  }

  auto col = wrap_single_variant(meta, val);
  auto got = cudf::io::parquet::experimental::extract_variant_field(
    col, "k99", cudf::data_type{cudf::type_id::INT32}, cudf::test::get_default_stream());

  cudf::test::fixed_width_column_wrapper<int32_t> expected{int32_t{99}};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*got, expected);
}

TEST_F(ExtractVariantFieldTest, MetadataOffsetSizeThresholdBoundary)
{
  // Verifies build_metadata selects 1-byte offsets when total string bytes == 255 (still fits)
  // and 2-byte offsets when total == 256 (first value that overflows a uint8_t accumulator).
  auto stream                 = cudf::test::get_default_stream();
  auto const int32_dtype      = cudf::data_type{cudf::type_id::INT32};
  constexpr int32_t kExpected = 42;

  // Build a flat field_count-field object with 1-byte value offsets where field `target_fid`
  // holds INT32(kExpected) and all others hold BOOLEAN_TRUE.
  auto build_flat = [&](int field_count, int target_fid) {
    auto const payload = enc_int32(kExpected);
    std::vector<uint8_t> val{make_variant_object_header(), static_cast<uint8_t>(field_count)};
    for (int fid = 0; fid < field_count; ++fid) {
      val.push_back(static_cast<uint8_t>(fid));
    }
    uint8_t field_offset = 0;
    for (int fid = 0; fid < field_count; ++fid) {
      val.push_back(field_offset);
      field_offset = static_cast<uint8_t>(field_offset + (fid == target_fid ? payload.size() : 1u));
    }
    val.push_back(field_offset);
    for (int fid = 0; fid < field_count; ++fid) {
      if (fid == target_fid) {
        val.insert(val.end(), payload.begin(), payload.end());
      } else {
        val.push_back(make_variant_primitive(variant_primitive_type::BOOLEAN_TRUE));
      }
    }
    return val;
  };

  // Case 1: total == 255 (85 keys × 3 bytes). Stays at 1-byte offsets.
  // Extract the first key "k00" (field ID 0).
  {
    SCOPED_TRACE("total=255, 1-byte offsets");
    auto const keys = make_numeric_keys(85);
    auto col        = wrap_single_variant(build_metadata(keys), build_flat(85, /*target_fid=*/0));
    auto got =
      cudf::io::parquet::experimental::extract_variant_field(col, "k00", int32_dtype, stream);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*got,
                                   cudf::test::fixed_width_column_wrapper<int32_t>{kExpected});
  }

  // Case 2: total == 256 (84 keys × 3 bytes + "long" at 4 bytes). Switches to 2-byte offsets.
  // "long" sorts after all "kXX" keys ('l' > 'k'), so it becomes field ID 84.
  {
    SCOPED_TRACE("total=256, 2-byte offsets");
    auto keys = make_numeric_keys(84);
    keys.emplace_back("long");
    auto col = wrap_single_variant(build_metadata(keys), build_flat(85, /*target_fid=*/84));
    auto got =
      cudf::io::parquet::experimental::extract_variant_field(col, "long", int32_dtype, stream);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*got,
                                   cudf::test::fixed_width_column_wrapper<int32_t>{kExpected});
  }
}

TEST_F(ExtractVariantFieldTest, MalformedVariantDataYieldsNull)
{
  // The column shape is a valid STRUCT<list<uint8>, list<uint8>>, but the VARIANT bytes are
  // internally inconsistent. Each such row must resolve to a null result rather than throwing or
  // reading out of bounds.
  auto stream            = cudf::test::get_default_stream();
  auto const int32_dtype = cudf::data_type{cudf::type_id::INT32};

  struct data_case {
    std::string label;
    std::vector<uint8_t> meta;
    std::vector<uint8_t> val;
  };
  auto const valid_object = build_single_field_object(/*fid=*/0, enc_int32(1));
  std::vector<data_case> const cases{
    // Metadata claims 5 dictionary entries but carries no offset/string bytes for them.
    {"metadata dictionary size overruns the buffer", {0x01, 0x05}, valid_object},
    // Single-key dict whose trailing offset (0xFF) points far past the string payload.
    {"metadata offset points past the string payload", {0x01, 0x01, 0x00, 0xFF, 'x'}, valid_object},
    // Object header declares 255 fields but carries no field-id/offset bytes.
    {"object declares more fields than the value buffer holds",
     build_metadata({"x"}),
     {make_variant_object_header(), 0xFF}},
  };

  for (auto const& c : cases) {
    SCOPED_TRACE(c.label);
    auto col = wrap_single_variant(c.meta, c.val);
    auto got =
      cudf::io::parquet::experimental::extract_variant_field(col, "x", int32_dtype, stream);
    ASSERT_EQ(got->size(), 1);
    EXPECT_EQ(got->null_count(), 1);
  }
}

TEST_F(ExtractVariantFieldTest, NullsAtDifferentDepths)
{
  std::vector<std::string> const dict = {"a", "b", "c", "d"};  // fids: a=0,b=1,c=2,d=3
  auto const meta                     = build_metadata(dict);

  // Shape 0: intact — {a:{b:{c:{d:"leaf"}}}}
  auto const s0_d    = enc_short_string("leaf");
  auto const s0_cd   = build_single_field_object(/*fid=d*/ 3, s0_d);
  auto const s0_bc   = build_single_field_object(/*fid=c*/ 2, s0_cd);
  auto const s0_ab   = build_single_field_object(/*fid=b*/ 1, s0_bc);
  auto const s0_root = build_single_field_object(/*fid=a*/ 0, s0_ab);

  // Shape 1: missing key at depth 1 — root has "b" but no "a".
  auto const s1_b    = enc_int32(0);
  auto const s1_root = build_single_field_object(/*fid=b*/ 1, s1_b);

  // Shape 2: kind mismatch at depth 3 — {a:{b:INT32(7)}} so descending into "c" fails.
  auto const s2_bval = enc_int32(7);
  auto const s2_ab   = build_single_field_object(/*fid=b*/ 1, s2_bval);
  auto const s2_root = build_single_field_object(/*fid=a*/ 0, s2_ab);

  // Shape 3: missing key at depth 4 — {a:{b:{c:INT32(9)}}} so the final ".d" misses.
  auto const s3_cval = enc_int32(9);
  auto const s3_bc   = build_single_field_object(/*fid=c*/ 2, s3_cval);
  auto const s3_ab   = build_single_field_object(/*fid=b*/ 1, s3_bc);
  auto const s3_root = build_single_field_object(/*fid=a*/ 0, s3_ab);

  std::vector<std::vector<uint8_t> const*> const shapes{&s0_root, &s1_root, &s2_root, &s3_root};

  constexpr int num_rows = 128;
  std::vector<std::vector<uint8_t>> meta_rows(num_rows, meta);
  std::vector<std::vector<uint8_t>> val_rows(num_rows);
  std::vector<char const*> exp_strs(num_rows);
  std::vector<bool> exp_valid(num_rows);
  for (int i = 0; i < num_rows; ++i) {
    int const shape = i % 4;
    val_rows[i]     = *shapes[shape];
    exp_strs[i]     = (shape == 0) ? "leaf" : "";
    exp_valid[i]    = (shape == 0);
  }

  auto col = wrap_multi_row_variant(meta_rows, val_rows);

  auto got = cudf::io::parquet::experimental::extract_variant_field(
    col, "$.a.b.c.d", cudf::data_type{cudf::type_id::STRING}, cudf::test::get_default_stream());

  cudf::test::strings_column_wrapper expected(exp_strs.begin(), exp_strs.end(), exp_valid.begin());
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*got, expected);
}

TEST_F(ExtractVariantFieldTest, EmptyInput)
{
  auto const stream  = cudf::test::get_default_stream();
  auto const variant = cudf::empty_like(make_xyz_three_row_variant());

  auto got = cudf::io::parquet::experimental::extract_variant_field(
    *variant, "x", cudf::data_type{cudf::type_id::INT32}, stream);
  EXPECT_EQ(got->type().id(), cudf::type_id::INT32);
  EXPECT_EQ(got->size(), 0);
  EXPECT_EQ(got->null_count(), 0);
}

struct GetVariantFieldTest : public cudf::test::BaseFixture {};

TEST_F(GetVariantFieldTest, ApacheObjectPrimitive)
{
  auto col    = make_apache_variant(avf::object_primitive);
  auto stream = cudf::test::get_default_stream();

  auto got = cudf::io::parquet::experimental::get_variant_field(col, "int_field", stream);

  EXPECT_EQ(got->type().id(), cudf::type_id::LIST);
  EXPECT_EQ(got->size(), 1);
  EXPECT_EQ(cudf::lists_column_view{got->view()}.child().type().id(), cudf::type_id::UINT8);

  auto casted = cudf::io::parquet::experimental::cast_variant(
    got->view(), cudf::data_type{cudf::type_id::INT8}, stream);
  cudf::test::fixed_width_column_wrapper<int8_t> expected{int8_t{1}};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*casted, expected);
}

TEST_F(GetVariantFieldTest, ApacheObjectPrimitiveMissingKeyAllNull)
{
  auto col = make_apache_variant(avf::object_primitive);
  auto got = cudf::io::parquet::experimental::get_variant_field(
    col, "no_such_field", cudf::test::get_default_stream());

  EXPECT_EQ(got->type().id(), cudf::type_id::LIST);
  EXPECT_EQ(got->size(), 1);
  EXPECT_EQ(got->null_count(), 1);
}

TEST_F(GetVariantFieldTest, GetAndCastMatchesExtract)
{
  auto col    = make_xyz_three_row_variant();
  auto stream = cudf::test::get_default_stream();

  auto extract_x = cudf::io::parquet::experimental::extract_variant_field(
    col, "x", cudf::data_type{cudf::type_id::INT32}, stream);

  auto intermediate = cudf::io::parquet::experimental::get_variant_field(col, "x", stream);
  auto two_step_x   = cudf::io::parquet::experimental::cast_variant(
    intermediate->view(), cudf::data_type{cudf::type_id::INT32}, stream);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*extract_x, *two_step_x);
}

TEST_F(GetVariantFieldTest, EmptyInput)
{
  auto const stream  = cudf::test::get_default_stream();
  auto const variant = cudf::empty_like(make_xyz_three_row_variant());

  auto got = cudf::io::parquet::experimental::get_variant_field(*variant, "x", stream);
  EXPECT_EQ(got->type().id(), cudf::type_id::LIST);
  EXPECT_EQ(got->size(), 0);
  EXPECT_EQ(got->null_count(), 0);
  EXPECT_EQ(cudf::lists_column_view{got->view()}.child().type().id(), cudf::type_id::UINT8);
}

template <typename T, std::size_t M, std::size_t V>
std::unique_ptr<cudf::column> cast_apache_primitive(avf::fixture<M, V> const& fixture)
{
  auto const stream = cudf::test::get_default_stream();
  auto col          = make_apache_variant(fixture);
  auto const value  = cudf::structs_column_view{col}.get_sliced_child(1, stream);
  return cudf::io::parquet::experimental::cast_variant(
    value, cudf::data_type{cudf::type_to_id<T>()}, stream);
}

struct CastVariantTest : public cudf::test::BaseFixture {};

TEST_F(CastVariantTest, ApachePrimitiveInts)
{
  {
    auto got = cast_apache_primitive<int8_t>(avf::primitive_int8);
    cudf::test::fixed_width_column_wrapper<int8_t> expected{int8_t{42}};
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*got, expected);
  }
  {
    auto got = cast_apache_primitive<int16_t>(avf::primitive_int16);
    cudf::test::fixed_width_column_wrapper<int16_t> expected{int16_t{1234}};
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*got, expected);
  }
  {
    auto got = cast_apache_primitive<int32_t>(avf::primitive_int32);
    cudf::test::fixed_width_column_wrapper<int32_t> expected{int32_t{123456}};
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*got, expected);
  }
  {
    auto got = cast_apache_primitive<int64_t>(avf::primitive_int64);
    cudf::test::fixed_width_column_wrapper<int64_t> expected{int64_t{1234567890123456789LL}};
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*got, expected);
  }
}

TEST_F(CastVariantTest, ApachePrimitiveFloats)
{
  auto stream     = cudf::test::get_default_stream();
  auto const cast = [&](auto const& fixture, auto expected_val) {
    using T          = decltype(expected_val);
    auto col         = make_apache_variant(fixture);
    auto const value = cudf::structs_column_view{col}.get_sliced_child(1, stream);
    auto got         = cudf::io::parquet::experimental::cast_variant(
      value, cudf::data_type{cudf::type_to_id<T>()}, stream);
    cudf::test::fixed_width_column_wrapper<T> expected{expected_val};
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*got, expected);
  };

  cast(avf::primitive_float, float{1234567936.0f});
  cast(avf::primitive_double, double{1234567890.1234});
}

TEST_F(CastVariantTest, ApachePrimitiveBooleans)
{
  auto stream     = cudf::test::get_default_stream();
  auto const cast = [&](auto const& fixture, bool expected_val) {
    auto col         = make_apache_variant(fixture);
    auto const value = cudf::structs_column_view{col}.get_sliced_child(1, stream);
    auto got         = cudf::io::parquet::experimental::cast_variant(
      value, cudf::data_type{cudf::type_id::BOOL8}, stream);
    cudf::test::fixed_width_column_wrapper<bool> expected{expected_val};
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*got, expected);
  };

  cast(avf::primitive_boolean_true, true);
  cast(avf::primitive_boolean_false, false);

  // Null variant value must cast to a null BOOL8, not false.
  {
    auto col         = make_apache_variant(avf::primitive_null);
    auto const value = cudf::structs_column_view{col}.get_sliced_child(1, stream);
    auto got         = cudf::io::parquet::experimental::cast_variant(
      value, cudf::data_type{cudf::type_id::BOOL8}, stream);
    cudf::test::fixed_width_column_wrapper<bool> expected({false}, {false});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*got, expected);
  }

  // Sliced multi-row column exercises grid-stride paths with a non-zero slice offset.
  // num_rows / slice range is chosen so the sliced window (slice_end - slice_beg = 512)
  // spans more than one cast_variant_bool_kernel block (block_size = 256).
  {
    constexpr int num_rows  = 516;
    constexpr int slice_beg = 3;
    constexpr int slice_end = 515;

    std::vector<uint8_t> const true_bytes{avf::primitive_boolean_true.value.begin(),
                                          avf::primitive_boolean_true.value.end()};
    std::vector<uint8_t> const false_bytes{avf::primitive_boolean_false.value.begin(),
                                           avf::primitive_boolean_false.value.end()};
    std::vector<uint8_t> const null_bytes{avf::primitive_null.value.begin(),
                                          avf::primitive_null.value.end()};
    std::vector<uint8_t> const meta_bytes{avf::primitive_boolean_true.metadata.begin(),
                                          avf::primitive_boolean_true.metadata.end()};

    std::vector<std::vector<uint8_t>> metas(num_rows, meta_bytes);
    std::vector<std::vector<uint8_t>> vals(num_rows);
    std::vector<bool> exp_vals(num_rows);
    std::vector<bool> exp_valid(num_rows);

    for (int i = 0; i < num_rows; ++i) {
      int const pat = i % 3;
      if (pat == 0) {
        vals[i]      = true_bytes;
        exp_vals[i]  = true;
        exp_valid[i] = true;
      } else if (pat == 1) {
        vals[i]      = false_bytes;
        exp_vals[i]  = false;
        exp_valid[i] = true;
      } else {
        vals[i]      = null_bytes;
        exp_vals[i]  = false;
        exp_valid[i] = false;
      }
    }

    auto col          = wrap_multi_row_variant(metas, vals);
    auto const sliced = cudf::slice(col, {slice_beg, slice_end}).front();
    auto const value  = cudf::structs_column_view{sliced}.get_sliced_child(1, stream);
    auto got          = cudf::io::parquet::experimental::cast_variant(
      value, cudf::data_type{cudf::type_id::BOOL8}, stream);

    cudf::test::fixed_width_column_wrapper<bool> expected(
      exp_vals.begin() + slice_beg, exp_vals.begin() + slice_end, exp_valid.begin() + slice_beg);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*got, expected);
  }
}

TEST_F(CastVariantTest, ApacheShortString)
{
  auto col         = make_apache_variant(avf::short_string);
  auto stream      = cudf::test::get_default_stream();
  auto const value = cudf::structs_column_view{col}.get_sliced_child(1, stream);

  auto got = cudf::io::parquet::experimental::cast_variant(
    value, cudf::data_type{cudf::type_id::STRING}, stream);

  // Decoded from short_string.value: skip the 1-byte header, take the rest.
  std::string const expected_str(reinterpret_cast<char const*>(avf::short_string.value.data() + 1),
                                 avf::short_string.value.size() - 1);
  cudf::test::strings_column_wrapper expected({expected_str});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*got, expected);
}

TEST_F(CastVariantTest, ApachePrimitiveString)
{
  auto col         = make_apache_variant(avf::primitive_string);
  auto stream      = cudf::test::get_default_stream();
  auto const value = cudf::structs_column_view{col}.get_sliced_child(1, stream);

  auto got = cudf::io::parquet::experimental::cast_variant(
    value, cudf::data_type{cudf::type_id::STRING}, stream);

  // Long-string layout: 1 header byte + 4-byte LE length + payload.
  std::string const expected_str(
    reinterpret_cast<char const*>(avf::primitive_string.value.data() + 5),
    avf::primitive_string.value.size() - 5);
  cudf::test::strings_column_wrapper expected({expected_str});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*got, expected);
}

TEST_F(CastVariantTest, MismatchedTypeYieldsNull)
{
  // Casting an object value to a primitive must produce null, not throw
  auto stream      = cudf::test::get_default_stream();
  auto col         = make_apache_variant(avf::object_primitive);
  auto const value = cudf::structs_column_view{col}.get_sliced_child(1, stream);
  auto got         = cudf::io::parquet::experimental::cast_variant(
    value, cudf::data_type{cudf::type_id::INT32}, stream);
  ASSERT_EQ(got->size(), 1);
  EXPECT_EQ(got->null_count(), 1);
}

TEST_F(CastVariantTest, EmptyInput)
{
  auto const stream = cudf::test::get_default_stream();
  auto const values =
    cudf::empty_like(cudf::structs_column_view{make_xyz_three_row_variant()}.child(1));

  for (auto const id : {cudf::type_id::INT32,
                        cudf::type_id::STRING,
                        cudf::type_id::FLOAT32,
                        cudf::type_id::FLOAT64,
                        cudf::type_id::BOOL8}) {
    auto got = cudf::io::parquet::experimental::cast_variant(*values, cudf::data_type{id}, stream);
    EXPECT_EQ(got->type().id(), id);
    EXPECT_EQ(got->size(), 0);
    EXPECT_EQ(got->null_count(), 0);
  }
}

TEST_F(CastVariantTest, UnsupportedTypeThrows)
{
  // Unsupported target types must throw regardless of whether the input is empty or non-empty.
  auto stream = cudf::test::get_default_stream();

  std::vector<cudf::type_id> const ids{cudf::type_id::UINT8,
                                       cudf::type_id::UINT16,
                                       cudf::type_id::UINT32,
                                       cudf::type_id::UINT64,
                                       cudf::type_id::TIMESTAMP_DAYS,
                                       cudf::type_id::TIMESTAMP_SECONDS,
                                       cudf::type_id::TIMESTAMP_MICROSECONDS,
                                       cudf::type_id::DURATION_SECONDS,
                                       cudf::type_id::DECIMAL32,
                                       cudf::type_id::DECIMAL64,
                                       cudf::type_id::DECIMAL128};

  // Empty input: the early-return path must still validate the type.
  auto const empty_values =
    cudf::empty_like(cudf::structs_column_view{make_xyz_three_row_variant()}.child(1));
  for (auto const id : ids) {
    EXPECT_THROW(static_cast<void>(cudf::io::parquet::experimental::cast_variant(
                   *empty_values, cudf::data_type{id}, stream)),
                 std::invalid_argument)
      << std::format("expected throw for type_id {} on empty input", static_cast<int>(id));
  }

  // Non-empty input: the dispatch path must also throw for unsupported types.
  auto col         = make_apache_variant(avf::primitive_int32);
  auto const value = cudf::structs_column_view{col}.get_sliced_child(1, stream);
  for (auto const id : ids) {
    EXPECT_THROW(static_cast<void>(cudf::io::parquet::experimental::cast_variant(
                   value, cudf::data_type{id}, stream)),
                 std::invalid_argument)
      << std::format("expected throw for type_id {} on non-empty input", static_cast<int>(id));
  }
}

TEST_F(CastVariantTest, CastSourceTargetMatrix)
{
  // Exhaustively covers (source physical type) x (supported target) casts. The supported targets
  // are INT8/16/32/64 and STRING. Expected behaviour:
  //   - integer targets: only a source whose physical type has the *exact* same width decodes;
  //   every
  //     other source (including narrower/wider ints) yields null — cast_variant does not widen.
  //   - STRING target: short_string and long_string sources decode; every other source yields null.
  auto const stream = cudf::test::get_default_stream();

  struct source_blob {
    std::string label;
    std::vector<uint8_t> bytes;
  };
  std::vector<source_blob> const sources{
    {"null", enc_null()},
    {"bool_true", enc_bool(true)},
    {"bool_false", enc_bool(false)},
    {"int8", enc_int8(42)},
    {"int16", enc_int16(1234)},
    {"int32", enc_int32(123456)},
    {"int64", enc_int64(1234567890123456789LL)},
    {"float64", enc_float64(2.5)},
    {"short_string", enc_short_string("hi")},
    {"long_string", enc_long_string(std::string(70, 'a'))},
  };

  auto values_of = [](std::vector<uint8_t> const& b) {
    return cudf::test::lists_column_wrapper<uint8_t>(b.begin(), b.end());
  };

  // Integer targets: exactly one source label decodes to `match_value`; the rest are null.
  auto check_int_target = [&]<typename T>(char const* match_label, T match_value) {
    auto const target = cudf::data_type{cudf::type_to_id<T>()};
    for (auto const& src : sources) {
      SCOPED_TRACE(std::string{"int target "} + match_label + ", source " + src.label);
      auto values = values_of(src.bytes);
      auto got    = cudf::io::parquet::experimental::cast_variant(values, target, stream);
      if (std::string_view{src.label} == match_label) {
        cudf::test::fixed_width_column_wrapper<T> const expected{match_value};
        CUDF_TEST_EXPECT_COLUMNS_EQUAL(*got, expected);
      } else {
        ASSERT_EQ(got->size(), 1);
        EXPECT_EQ(got->null_count(), 1);
      }
    }
  };
  check_int_target.template operator()<int8_t>("int8", int8_t{42});
  check_int_target.template operator()<int16_t>("int16", int16_t{1234});
  check_int_target.template operator()<int32_t>("int32", int32_t{123456});
  check_int_target.template operator()<int64_t>("int64", int64_t{1234567890123456789LL});

  // STRING target: short_string and long_string decode; every other source is null.
  auto const string_type = cudf::data_type{cudf::type_id::STRING};
  for (auto const& src : sources) {
    SCOPED_TRACE(std::string{"string target, source "} + src.label);
    auto values = values_of(src.bytes);
    auto got    = cudf::io::parquet::experimental::cast_variant(values, string_type, stream);
    std::string_view const label{src.label};
    if (label == "short_string" || label == "long_string") {
      std::string const expected_str = (label == "short_string") ? "hi" : std::string(70, 'a');
      cudf::test::strings_column_wrapper const expected({expected_str});
      CUDF_TEST_EXPECT_COLUMNS_EQUAL(*got, expected);
    } else {
      ASSERT_EQ(got->size(), 1);
      EXPECT_EQ(got->null_count(), 1);
    }
  }
}

TEST_F(CastVariantTest, ShortStringLengthZero)
{
  // Short string with length 0 (lower boundary of the 6-bit length field): header only, no payload.
  auto stream = cudf::test::get_default_stream();
  std::vector<uint8_t> const val{make_variant_short_string_header(0)};
  cudf::test::lists_column_wrapper<uint8_t> values(val.begin(), val.end());
  auto got = cudf::io::parquet::experimental::cast_variant(
    values, cudf::data_type{cudf::type_id::STRING}, stream);
  cudf::test::strings_column_wrapper expected({""});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*got, expected);
}

TEST_F(CastVariantTest, ShortStringMaxLength)
{
  // Short string with length 63, the max value a 6-bit length field can hold, then 63 bytes.
  auto stream = cudf::test::get_default_stream();
  std::vector<uint8_t> val;
  val.push_back(make_variant_short_string_header(63));
  std::string const payload(63, 'z');
  val.insert(val.end(), payload.begin(), payload.end());
  cudf::test::lists_column_wrapper<uint8_t> values(val.begin(), val.end());
  auto got = cudf::io::parquet::experimental::cast_variant(
    values, cudf::data_type{cudf::type_id::STRING}, stream);
  cudf::test::strings_column_wrapper expected({payload});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*got, expected);
}

TEST_F(CastVariantTest, LongStringLengthZero)
{
  // Long string: primitive long_string header, 4-byte LE length = 0, no payload.
  auto stream = cudf::test::get_default_stream();
  std::vector<uint8_t> const val{
    make_variant_primitive(variant_primitive_type::LONG_STRING), 0x00, 0x00, 0x00, 0x00};
  cudf::test::lists_column_wrapper<uint8_t> values(val.begin(), val.end());
  auto got = cudf::io::parquet::experimental::cast_variant(
    values, cudf::data_type{cudf::type_id::STRING}, stream);
  cudf::test::strings_column_wrapper expected({""});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*got, expected);
}

TEST_F(CastVariantTest, LongStringDeclaredLengthExceedsPayloadYieldsNull)
{
  // decode_string rejects any long string whose declared LE length exceeds the payload bytes
  // actually present, whether the payload is partially present or entirely absent. Both shapes
  // below declare length=10 (0x0000000A).
  auto stream    = cudf::test::get_default_stream();
  auto const hdr = make_variant_primitive(variant_primitive_type::LONG_STRING);
  std::vector<std::vector<uint8_t>> const cases{
    {hdr, 0x0A, 0x00, 0x00, 0x00, 'a', 'b', 'c'},  // 3 of 10 payload bytes present
    {hdr, 0x0A, 0x00, 0x00, 0x00},                 // 0 of 10 payload bytes present
  };
  for (auto const& val : cases) {
    SCOPED_TRACE(std::string{"payload bytes present: "} + std::to_string(val.size() - 5));
    cudf::test::lists_column_wrapper<uint8_t> values(val.begin(), val.end());
    auto got = cudf::io::parquet::experimental::cast_variant(
      values, cudf::data_type{cudf::type_id::STRING}, stream);
    ASSERT_EQ(got->size(), 1);
    EXPECT_EQ(got->null_count(), 1);
  }
}

TEST_F(CastVariantTest, LongStringPayloadExceedsDeclaredLength)
{
  // When more bytes are present than the declared length, decode_string should read exactly the
  // declared number of bytes and ignore the trailing ones.
  auto stream    = cudf::test::get_default_stream();
  auto const hdr = make_variant_primitive(variant_primitive_type::LONG_STRING);
  // Declared length = 3 ("abc"), followed by 5 extra bytes that must be ignored.
  std::vector<uint8_t> const val{
    hdr, 0x03, 0x00, 0x00, 0x00, 'a', 'b', 'c', 'x', 'x', 'x', 'x', 'x'};
  cudf::test::lists_column_wrapper<uint8_t> values(val.begin(), val.end());
  auto got = cudf::io::parquet::experimental::cast_variant(
    values, cudf::data_type{cudf::type_id::STRING}, stream);
  cudf::test::strings_column_wrapper expected({"abc"});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*got, expected);
}

struct InvalidInputShapeTest : public cudf::test::BaseFixture {};

namespace {

// A well-formed VARIANT child: a single-row list<uint8> holding `bytes`.
inline std::unique_ptr<cudf::column> list_u8(std::vector<uint8_t> const& bytes)
{
  return cudf::test::lists_column_wrapper<uint8_t>(bytes.begin(), bytes.end()).release();
}

// A single-row list<int32> (wrong element type for a VARIANT child).
inline std::unique_ptr<cudf::column> list_i32(std::vector<int32_t> const& values)
{
  return cudf::test::lists_column_wrapper<int32_t>(values.begin(), values.end()).release();
}

// A single-row fixed-width int32 column (a non-list child).
inline std::unique_ptr<cudf::column> scalar_i32()
{
  return cudf::test::fixed_width_column_wrapper<int32_t>{42}.release();
}

// A single-row STRUCT column adopting `children`.
inline std::unique_ptr<cudf::column> struct_of(std::vector<std::unique_ptr<cudf::column>> children)
{
  return cudf::make_structs_column(1, std::move(children), 0, rmm::device_buffer{});
}

inline std::vector<std::unique_ptr<cudf::column>> two_children(std::unique_ptr<cudf::column> a,
                                                               std::unique_ptr<cudf::column> b)
{
  std::vector<std::unique_ptr<cudf::column>> v;
  v.push_back(std::move(a));
  v.push_back(std::move(b));
  return v;
}

// A malformed-shape case: a human-readable label plus the offending column.
struct broken_shape {
  std::string label;
  std::unique_ptr<cudf::column> column;
};

}  // namespace

// A VARIANT column must be a STRUCT whose first two children are each a list<uint8>. Enumerate the
// distinct ways that column-shape contract can be broken; get_variant_field must reject every one
// with std::invalid_argument.
TEST_F(InvalidInputShapeTest, GetVariantFieldRejectsMalformedInput)
{
  auto stream = cudf::test::get_default_stream();

  std::vector<broken_shape> cases;
  cases.push_back({"input column is not a struct", scalar_i32()});
  {
    std::vector<std::unique_ptr<cudf::column>> one;
    one.push_back(list_u8({0x01, 0x00, 0x00}));
    cases.push_back({"struct has fewer than two children", struct_of(std::move(one))});
  }
  cases.push_back({"metadata child has wrong column type (not a list)",
                   struct_of(two_children(scalar_i32(), list_u8({0x00})))});
  cases.push_back({"metadata child has wrong list element type (not uint8)",
                   struct_of(two_children(list_i32({1, 2, 3}), list_u8({0x00})))});
  cases.push_back({"value child has wrong column type (not a list)",
                   struct_of(two_children(list_u8({0x01, 0x00, 0x00}), scalar_i32()))});
  cases.push_back({"value child has wrong list element type (not uint8)",
                   struct_of(two_children(list_u8({0x01, 0x00, 0x00}), list_i32({1, 2, 3})))});

  for (auto const& c : cases) {
    SCOPED_TRACE(c.label);
    EXPECT_THROW(static_cast<void>(cudf::io::parquet::experimental::get_variant_field(
                   c.column->view(), "x", stream)),
                 std::invalid_argument);
  }
}

// cast_variant requires a list<uint8> input; every other shape must be rejected with
// std::invalid_argument.
TEST_F(InvalidInputShapeTest, CastVariantRejectsMalformedInput)
{
  auto stream = cudf::test::get_default_stream();

  std::vector<broken_shape> cases;
  cases.push_back({"input is not a list", scalar_i32()});
  cases.push_back({"input list has wrong element type (not uint8)", list_i32({1, 2, 3})});

  for (auto const& c : cases) {
    SCOPED_TRACE(c.label);
    EXPECT_THROW(static_cast<void>(cudf::io::parquet::experimental::cast_variant(
                   c.column->view(), cudf::data_type{cudf::type_id::INT32}, stream)),
                 std::invalid_argument);
  }
}

TEST_F(InvalidInputShapeTest, GetVariantTypeIdRejectsMalformedInput)
{
  // get_variant_type_id requires a list<uint8> input; every other shape must be rejected with
  // std::invalid_argument.
  auto stream = cudf::test::get_default_stream();

  std::vector<broken_shape> cases;
  cases.push_back({"input is not a list", scalar_i32()});
  cases.push_back({"input list has wrong element type (not uint8)", list_i32({1, 2, 3})});

  for (auto const& c : cases) {
    SCOPED_TRACE(c.label);
    EXPECT_THROW(static_cast<void>(
                   cudf::io::parquet::experimental::get_variant_type_id(c.column->view(), stream)),
                 std::invalid_argument);
  }
}

namespace {

/**
 * @brief Helper: run get_variant_type_id on the value child of an apache fixture.
 */
template <std::size_t M, std::size_t V>
[[nodiscard]] std::unique_ptr<cudf::column> apache_type_id(avf::fixture<M, V> const& fixture)
{
  auto const stream = cudf::test::get_default_stream();
  auto col          = make_apache_variant(fixture);
  auto const value  = cudf::structs_column_view{col}.get_sliced_child(1, stream);
  return cudf::io::parquet::experimental::get_variant_type_id(value, stream);
}

/**
 * @brief Build a list<uint8> column from blobs with per-row validity. Rows where valid[i] is false
 * are null at the list level (not an encoded Variant null — those are valid rows with a NULLVAL
 * blob).
 */
[[nodiscard]] std::unique_ptr<cudf::column> make_nullable_list_u8(
  cudf::host_span<std::vector<uint8_t> const> blobs, cudf::host_span<uint8_t const> valid)
{
  auto const num_rows = static_cast<cudf::size_type>(blobs.size());
  std::vector<int32_t> offsets(num_rows + 1, 0);
  std::vector<uint8_t> flat;
  for (cudf::size_type i = 0; i < num_rows; ++i) {
    flat.insert(flat.end(), blobs[i].begin(), blobs[i].end());
    offsets[i + 1] = static_cast<int32_t>(flat.size());
  }
  auto off_col =
    cudf::test::fixed_width_column_wrapper<int32_t>(offsets.begin(), offsets.end()).release();
  auto dat_col =
    cudf::test::fixed_width_column_wrapper<uint8_t>(flat.begin(), flat.end()).release();
  auto [d_mask, null_count] = cudf::test::detail::make_null_mask(valid.begin(), valid.end());
  return cudf::make_lists_column(
    num_rows, std::move(off_col), std::move(dat_col), null_count, std::move(d_mask));
}

}  // namespace

struct GetVariantTypeIdTest : public cudf::test::BaseFixture {};

using LT = cudf::io::parquet::experimental::variant_logical_type;

// Apache fixtures: one test per logical-type category.

TEST_F(GetVariantTypeIdTest, NullValue)
{
  auto got = apache_type_id(avf::primitive_null);
  cudf::test::fixed_width_column_wrapper<uint8_t> expected{static_cast<uint8_t>(LT::NULL_VALUE)};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*got, expected);
}

TEST_F(GetVariantTypeIdTest, Boolean)
{
  cudf::test::fixed_width_column_wrapper<uint8_t> const expected{static_cast<uint8_t>(LT::BOOLEAN)};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*apache_type_id(avf::primitive_boolean_true), expected);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*apache_type_id(avf::primitive_boolean_false), expected);
}

TEST_F(GetVariantTypeIdTest, LongValueAllIntWidths)
{
  // INT8, INT16, INT32, INT64 all map to long_value regardless of physical width.
  cudf::test::fixed_width_column_wrapper<uint8_t> const expected{
    static_cast<uint8_t>(LT::LONG_VALUE)};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*apache_type_id(avf::primitive_int8), expected);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*apache_type_id(avf::primitive_int16), expected);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*apache_type_id(avf::primitive_int32), expected);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*apache_type_id(avf::primitive_int64), expected);
}

TEST_F(GetVariantTypeIdTest, StringBothEncodings)
{
  // SHORT_STRING and primitive LONG_STRING both map to string.
  cudf::test::fixed_width_column_wrapper<uint8_t> const expected{static_cast<uint8_t>(LT::STRING)};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*apache_type_id(avf::short_string), expected);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*apache_type_id(avf::primitive_string), expected);
}

TEST_F(GetVariantTypeIdTest, FloatTypes)
{
  {
    auto got = apache_type_id(avf::primitive_float);
    cudf::test::fixed_width_column_wrapper<uint8_t> expected{static_cast<uint8_t>(LT::FLOAT_VALUE)};
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*got, expected);
  }
  {
    auto got = apache_type_id(avf::primitive_double);
    cudf::test::fixed_width_column_wrapper<uint8_t> expected{
      static_cast<uint8_t>(LT::DOUBLE_VALUE)};
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*got, expected);
  }
}

TEST_F(GetVariantTypeIdTest, Decimal)
{
  cudf::test::fixed_width_column_wrapper<uint8_t> const expected{static_cast<uint8_t>(LT::DECIMAL)};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*apache_type_id(avf::primitive_decimal4), expected);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*apache_type_id(avf::primitive_decimal8), expected);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*apache_type_id(avf::primitive_decimal16), expected);
}

TEST_F(GetVariantTypeIdTest, Date)
{
  auto got = apache_type_id(avf::primitive_date);
  cudf::test::fixed_width_column_wrapper<uint8_t> expected{static_cast<uint8_t>(LT::DATE)};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*got, expected);
}

TEST_F(GetVariantTypeIdTest, TimestampBothNanos)
{
  // TIMESTAMP_MICROS and TIMESTAMP_NANOS both map to timestamp.
  cudf::test::fixed_width_column_wrapper<uint8_t> const expected{
    static_cast<uint8_t>(LT::TIMESTAMP)};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*apache_type_id(avf::primitive_timestamp), expected);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*apache_type_id(avf::primitive_timestamp_nanos), expected);
}

TEST_F(GetVariantTypeIdTest, TimestampNtzBothNanos)
{
  // TIMESTAMP_NTZ_MICROS and TIMESTAMP_NTZ_NANOS both map to timestamp_ntz.
  cudf::test::fixed_width_column_wrapper<uint8_t> const expected{
    static_cast<uint8_t>(LT::TIMESTAMP_NTZ)};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*apache_type_id(avf::primitive_timestampntz), expected);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*apache_type_id(avf::primitive_timestampntz_nanos), expected);
}

TEST_F(GetVariantTypeIdTest, Binary)
{
  auto got = apache_type_id(avf::primitive_binary);
  cudf::test::fixed_width_column_wrapper<uint8_t> expected{static_cast<uint8_t>(LT::BINARY)};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*got, expected);
}

TEST_F(GetVariantTypeIdTest, Uuid)
{
  auto got = apache_type_id(avf::primitive_uuid);
  cudf::test::fixed_width_column_wrapper<uint8_t> expected{static_cast<uint8_t>(LT::UUID)};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*got, expected);
}

TEST_F(GetVariantTypeIdTest, TimeNtz)
{
  auto got = apache_type_id(avf::primitive_time);
  cudf::test::fixed_width_column_wrapper<uint8_t> expected{static_cast<uint8_t>(LT::TIME_NTZ)};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*got, expected);
}

TEST_F(GetVariantTypeIdTest, ObjectAndArray)
{
  {
    cudf::test::fixed_width_column_wrapper<uint8_t> const expected{
      static_cast<uint8_t>(LT::OBJECT)};
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*apache_type_id(avf::object_primitive), expected);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*apache_type_id(avf::object_nested), expected);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*apache_type_id(avf::object_empty), expected);
  }
  {
    cudf::test::fixed_width_column_wrapper<uint8_t> const expected{static_cast<uint8_t>(LT::ARRAY)};
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*apache_type_id(avf::array_primitive), expected);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*apache_type_id(avf::array_nested), expected);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*apache_type_id(avf::array_empty), expected);
  }
}

// Null and unknown-type behavior.

TEST_F(GetVariantTypeIdTest, UnknownPhysicalTypeProducesNull)
{
  // Primitive header byte 0xFC = (63 << 2) | 0: type_id 63 is not in the spec.
  auto const stream = cudf::test::get_default_stream();
  auto values =
    make_nullable_list_u8(std::vector<std::vector<uint8_t>>{{0xFC}}, std::vector<uint8_t>{1});
  auto got = cudf::io::parquet::experimental::get_variant_type_id(*values, stream);
  ASSERT_EQ(got->size(), 1);
  EXPECT_EQ(got->null_count(), 1);
}

TEST_F(GetVariantTypeIdTest, InputNullRowPropagates)
{
  // A null row in the input list<uint8> column propagates to the output.
  auto const stream = cudf::test::get_default_stream();
  auto values       = make_nullable_list_u8(
    std::vector<std::vector<uint8_t>>{enc_int32(1), enc_int32(2), enc_int32(3)},
    std::vector<uint8_t>{1, 0, 1});

  auto got = cudf::io::parquet::experimental::get_variant_type_id(*values, stream);

  std::initializer_list<uint8_t> expected_vals1{
    static_cast<uint8_t>(LT::LONG_VALUE), 0, static_cast<uint8_t>(LT::LONG_VALUE)};
  std::initializer_list<bool> validity1{true, false, true};
  cudf::test::fixed_width_column_wrapper<uint8_t> expected(expected_vals1, validity1);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*got, expected);
}

TEST_F(GetVariantTypeIdTest, EncodedNullIsNotInputNull)
{
  // An encoded Variant NULLVAL blob is a valid row whose type is null_value, not a null row.
  auto const stream = cudf::test::get_default_stream();
  auto val          = enc_null();
  cudf::test::lists_column_wrapper<uint8_t> values(val.begin(), val.end());
  auto got = cudf::io::parquet::experimental::get_variant_type_id(values, stream);

  ASSERT_EQ(got->size(), 1);
  EXPECT_EQ(got->null_count(), 0);
  cudf::test::fixed_width_column_wrapper<uint8_t> expected{static_cast<uint8_t>(LT::NULL_VALUE)};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*got, expected);
}

TEST_F(GetVariantTypeIdTest, EmptyValueBlobProducesNull)
{
  // An empty list row (zero bytes) has no header byte to decode → null.
  auto const stream = cudf::test::get_default_stream();
  auto values =
    make_nullable_list_u8(std::vector<std::vector<uint8_t>>{enc_int32(1), {}, enc_int32(3)},
                          std::vector<uint8_t>{1, 1, 1});

  auto got = cudf::io::parquet::experimental::get_variant_type_id(*values, stream);

  std::initializer_list<uint8_t> expected_vals2{
    static_cast<uint8_t>(LT::LONG_VALUE), 0, static_cast<uint8_t>(LT::LONG_VALUE)};
  std::initializer_list<bool> validity2{true, false, true};
  cudf::test::fixed_width_column_wrapper<uint8_t> expected(expected_vals2, validity2);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*got, expected);
}

// Multi-row and structural tests.

TEST_F(GetVariantTypeIdTest, MixedTypesColumn)
{
  auto const stream = cudf::test::get_default_stream();

  auto null_val = enc_null();
  auto bool_val = enc_bool(true);
  auto int_val  = enc_int64(999);
  auto str_val  = enc_short_string("hi");
  auto dbl_val  = enc_float64(3.14);

  cudf::test::lists_column_wrapper<uint8_t> values{
    {null_val.begin(), null_val.end()},
    {bool_val.begin(), bool_val.end()},
    {int_val.begin(), int_val.end()},
    {str_val.begin(), str_val.end()},
    {dbl_val.begin(), dbl_val.end()},
  };
  auto got = cudf::io::parquet::experimental::get_variant_type_id(values, stream);

  cudf::test::fixed_width_column_wrapper<uint8_t> expected{
    static_cast<uint8_t>(LT::NULL_VALUE),
    static_cast<uint8_t>(LT::BOOLEAN),
    static_cast<uint8_t>(LT::LONG_VALUE),
    static_cast<uint8_t>(LT::STRING),
    static_cast<uint8_t>(LT::DOUBLE_VALUE),
  };
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*got, expected);
}

TEST_F(GetVariantTypeIdTest, AllNullInputColumn)
{
  // All rows are null at the list level → all output rows are null.
  auto const stream = cudf::test::get_default_stream();
  auto values       = make_nullable_list_u8(
    std::vector<std::vector<uint8_t>>{enc_int32(1), enc_int32(2), enc_int32(3)},
    std::vector<uint8_t>{0, 0, 0});

  auto got = cudf::io::parquet::experimental::get_variant_type_id(*values, stream);

  ASSERT_EQ(got->size(), 3);
  EXPECT_EQ(got->null_count(), 3);
}

TEST_F(GetVariantTypeIdTest, EmptyInput)
{
  auto const stream = cudf::test::get_default_stream();
  auto const values =
    cudf::empty_like(cudf::structs_column_view{make_xyz_three_row_variant()}.child(1));

  auto got = cudf::io::parquet::experimental::get_variant_type_id(*values, stream);
  EXPECT_EQ(got->type().id(), cudf::type_id::UINT8);
  EXPECT_EQ(got->size(), 0);
  EXPECT_EQ(got->null_count(), 0);
}

TEST_F(GetVariantTypeIdTest, SlicedValuesColumn)
{
  // Verify that a sliced input produces correct results for the slice only.
  auto const stream      = cudf::test::get_default_stream();
  auto col               = make_xyz_three_row_variant();
  auto const value_child = cudf::structs_column_view{col}.get_sliced_child(1, stream);

  // The xyz variant has object rows; slicing [1,3) gives 2 object rows.
  auto const sliced_values = cudf::slice(value_child, {1, 3}).front();
  auto got = cudf::io::parquet::experimental::get_variant_type_id(sliced_values, stream);

  cudf::test::fixed_width_column_wrapper<uint8_t> expected{static_cast<uint8_t>(LT::OBJECT),
                                                           static_cast<uint8_t>(LT::OBJECT)};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*got, expected);
}

TEST_F(GetVariantTypeIdTest, MultiWordNullMask)
{
  // 600 rows cycling through recognized types, with invalid rows scattered across null-mask word
  // boundaries (32-bit words: the 31/32 and 63/64 boundaries) and within the trailing partial
  // word (rows 576..599), mixing list-level nulls with valid rows that carry an unrecognized
  // primitive header (0xFC).
  auto const stream = cudf::test::get_default_stream();

  struct row_spec {
    std::vector<uint8_t> blob;
    uint8_t expected_id;
  };

  std::vector<row_spec> const types{
    {enc_null(), static_cast<uint8_t>(LT::NULL_VALUE)},
    {enc_bool(false), static_cast<uint8_t>(LT::BOOLEAN)},
    {enc_int8(1), static_cast<uint8_t>(LT::LONG_VALUE)},
    {enc_int16(2), static_cast<uint8_t>(LT::LONG_VALUE)},
    {enc_int32(3), static_cast<uint8_t>(LT::LONG_VALUE)},
    {enc_int64(4), static_cast<uint8_t>(LT::LONG_VALUE)},
    {enc_float64(5.0), static_cast<uint8_t>(LT::DOUBLE_VALUE)},
    {enc_short_string("x"), static_cast<uint8_t>(LT::STRING)},
    {enc_long_string(std::string(70, 'z')), static_cast<uint8_t>(LT::STRING)},
  };
  constexpr int num_rows = 600;

  // Rows null at the list level (not an encoded Variant null).
  std::array<int, 4> const list_null_rows{31, 63, 585, 599};
  // Rows with a valid list but an unrecognized primitive header, which also produce a null
  // output row.
  std::array<int, 3> const unknown_header_rows{32, 64, 590};

  std::vector<std::vector<uint8_t>> blobs(num_rows);
  std::vector<uint8_t> list_valid(num_rows, 1);
  std::vector<uint8_t> expected_ids(num_rows, 0);
  std::vector<bool> expected_valid(num_rows, true);

  for (int i = 0; i < num_rows; ++i) {
    auto const& spec = types[i % types.size()];
    blobs[i]         = spec.blob;
    expected_ids[i]  = spec.expected_id;

    if (std::find(unknown_header_rows.begin(), unknown_header_rows.end(), i) !=
        unknown_header_rows.end()) {
      blobs[i]          = {0xFC};
      expected_ids[i]   = 0;
      expected_valid[i] = false;
    } else if (std::find(list_null_rows.begin(), list_null_rows.end(), i) != list_null_rows.end()) {
      list_valid[i]     = 0;
      expected_ids[i]   = 0;
      expected_valid[i] = false;
    }
  }

  auto values = make_nullable_list_u8(blobs, list_valid);
  auto got    = cudf::io::parquet::experimental::get_variant_type_id(*values, stream);

  cudf::test::fixed_width_column_wrapper<uint8_t> expected(
    expected_ids.begin(), expected_ids.end(), expected_valid.begin());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*got, expected);
  EXPECT_EQ(got->null_count(),
            static_cast<cudf::size_type>(list_null_rows.size() + unknown_header_rows.size()));
}
