/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/cudf_gtest.hpp>
#include <cudf_test/table_utilities.hpp>
#include <cudf_test/testing_main.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/column/column_factories.hpp>
#include <cudf/io/protobuf.hpp>
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/table/table.hpp>
#include <cudf/types.hpp>

#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

namespace pb = cudf::io::protobuf;

// ============================================================================
// Protobuf wire format encoding helpers
// ============================================================================
namespace {

constexpr int WT_VARINT = 0;
constexpr int WT_LEN    = 2;

std::vector<uint8_t> encode_varint(uint64_t value)
{
  std::vector<uint8_t> out;
  while (value > 0x7F) {
    out.push_back(static_cast<uint8_t>((value & 0x7F) | 0x80));
    value >>= 7;
  }
  out.push_back(static_cast<uint8_t>(value));
  return out;
}

std::vector<uint8_t> concat(std::initializer_list<std::vector<uint8_t>> parts)
{
  std::vector<uint8_t> out;
  for (auto const& p : parts) {
    out.insert(out.end(), p.begin(), p.end());
  }
  return out;
}

std::vector<uint8_t> tag(int field_number, int wire_type)
{
  return encode_varint((static_cast<uint64_t>(field_number) << 3) |
                       static_cast<uint64_t>(wire_type));
}

std::vector<uint8_t> encode_varint_field(int field_number, uint64_t value)
{
  return concat({tag(field_number, WT_VARINT), encode_varint(value)});
}

std::vector<uint8_t> encode_string_field(int field_number, std::string const& s)
{
  auto t   = tag(field_number, WT_LEN);
  auto len = encode_varint(s.size());
  auto out = concat({t, len});
  out.insert(out.end(), s.begin(), s.end());
  return out;
}

std::unique_ptr<cudf::column> make_binary_column(std::vector<std::vector<uint8_t>> const& messages,
                                                 std::vector<bool> const& validity = {})
{
  std::vector<int32_t> offsets;
  offsets.reserve(messages.size() + 1);
  offsets.push_back(0);
  for (auto const& m : messages) {
    offsets.push_back(offsets.back() + static_cast<int32_t>(m.size()));
  }

  std::vector<uint8_t> flat_data;
  flat_data.reserve(offsets.back());
  for (auto const& m : messages) {
    flat_data.insert(flat_data.end(), m.begin(), m.end());
  }

  auto offsets_col =
    cudf::test::fixed_width_column_wrapper<int32_t>(offsets.begin(), offsets.end()).release();
  auto data_col =
    cudf::test::fixed_width_column_wrapper<uint8_t>(flat_data.begin(), flat_data.end()).release();

  auto num_rows = static_cast<cudf::size_type>(messages.size());

  if (!validity.empty()) {
    auto [null_mask, null_count] =
      cudf::test::detail::make_null_mask(validity.begin(), validity.end());
    return cudf::make_lists_column(
      num_rows, std::move(offsets_col), std::move(data_col), null_count, std::move(null_mask));
  }

  return cudf::make_lists_column(
    num_rows, std::move(offsets_col), std::move(data_col), 0, rmm::device_buffer{});
}

pb::decode_protobuf_options make_scalar_options(std::vector<int> const& field_numbers,
                                                std::vector<cudf::type_id> const& types,
                                                std::vector<int> const& encodings,
                                                bool fail_on_errors = true)
{
  int const n = static_cast<int>(field_numbers.size());

  auto derive_wire_type = [](cudf::type_id type, int enc) -> pb::proto_wire_type {
    if (enc == static_cast<int>(pb::proto_encoding::FIXED)) {
      if (type == cudf::type_id::INT64 || type == cudf::type_id::UINT64 ||
          type == cudf::type_id::FLOAT64) {
        return pb::proto_wire_type::I64BIT;
      }
      return pb::proto_wire_type::I32BIT;
    }
    switch (type) {
      case cudf::type_id::FLOAT32: return pb::proto_wire_type::I32BIT;
      case cudf::type_id::FLOAT64: return pb::proto_wire_type::I64BIT;
      case cudf::type_id::STRING:
      case cudf::type_id::LIST:
      case cudf::type_id::STRUCT: return pb::proto_wire_type::LEN;
      default: return pb::proto_wire_type::VARINT;
    }
  };

  std::vector<pb::nested_field_descriptor> schema;
  schema.reserve(n);
  for (int i = 0; i < n; ++i) {
    schema.push_back({field_numbers[i],
                      -1,
                      0,
                      derive_wire_type(types[i], encodings[i]),
                      types[i],
                      static_cast<pb::proto_encoding>(encodings[i]),
                      false,
                      false,
                      false});
  }

  return pb::decode_protobuf_options::builder(std::move(schema))
    .fail_on_errors(fail_on_errors)
    .build();
}

}  // anonymous namespace

// ============================================================================
// Test fixture
// ============================================================================

struct ProtobufReaderTest : public cudf::test::BaseFixture {};

// ============================================================================
// Part0 tests: output shape, type structure, and null propagation
// (Stub decode returns all-null columns with correct types)
// ============================================================================

TEST_F(ProtobufReaderTest, EmptySchema)
{
  auto input = make_binary_column({encode_varint_field(1, 42), encode_varint_field(1, 7)});

  pb::decode_protobuf_options options{{}};

  auto result = pb::decode_protobuf(*input, options);

  ASSERT_EQ(result->type().id(), cudf::type_id::STRUCT);
  ASSERT_EQ(result->size(), 2);
  ASSERT_EQ(result->num_children(), 0);
}

TEST_F(ProtobufReaderTest, ZeroRows)
{
  auto input   = make_binary_column({});
  auto options = make_scalar_options({1, 2}, {cudf::type_id::INT64, cudf::type_id::STRING}, {0, 0});

  auto result = pb::decode_protobuf(*input, options);

  ASSERT_EQ(result->type().id(), cudf::type_id::STRUCT);
  ASSERT_EQ(result->size(), 0);
  ASSERT_EQ(result->num_children(), 2);
  EXPECT_EQ(result->child(0).type().id(), cudf::type_id::INT64);
  EXPECT_EQ(result->child(1).type().id(), cudf::type_id::STRING);
}

TEST_F(ProtobufReaderTest, ZeroRowsNestedSchema)
{
  // [0: id(INT32), 1: inner(STRUCT), 2: name(STRING, parent=1)]
  std::vector<pb::nested_field_descriptor> schema = {
    {1,
     -1,
     0,
     pb::proto_wire_type::VARINT,
     cudf::type_id::INT32,
     pb::proto_encoding::DEFAULT,
     false,
     false,
     false},
    {2,
     -1,
     0,
     pb::proto_wire_type::LEN,
     cudf::type_id::STRUCT,
     pb::proto_encoding::DEFAULT,
     false,
     false,
     false},
    {1,
     1,
     1,
     pb::proto_wire_type::LEN,
     cudf::type_id::STRING,
     pb::proto_encoding::DEFAULT,
     false,
     false,
     false},
  };

  auto options = pb::decode_protobuf_options::builder(std::move(schema)).build();

  auto result = pb::decode_protobuf(*make_binary_column({}), options);

  ASSERT_EQ(result->size(), 0);
  ASSERT_EQ(result->num_children(), 2);
  EXPECT_EQ(result->child(0).type().id(), cudf::type_id::INT32);
  EXPECT_EQ(result->child(1).type().id(), cudf::type_id::STRUCT);
  EXPECT_EQ(result->child(1).num_children(), 1);
  EXPECT_EQ(result->child(1).child(0).type().id(), cudf::type_id::STRING);
}

TEST_F(ProtobufReaderTest, ZeroRowsRepeatedSchema)
{
  std::vector<pb::nested_field_descriptor> schema = {
    {1,
     -1,
     0,
     pb::proto_wire_type::VARINT,
     cudf::type_id::INT32,
     pb::proto_encoding::DEFAULT,
     true,
     false,
     false},
  };

  auto options = pb::decode_protobuf_options::builder(std::move(schema)).build();

  auto result = pb::decode_protobuf(*make_binary_column({}), options);

  ASSERT_EQ(result->size(), 0);
  ASSERT_EQ(result->num_children(), 1);
  EXPECT_EQ(result->child(0).type().id(), cudf::type_id::LIST);
}

TEST_F(ProtobufReaderTest, StubReturnsAllNullWithCorrectTypes)
{
  auto input = make_binary_column({encode_varint_field(1, 42), encode_string_field(2, "hello")});

  auto options = make_scalar_options({1, 2}, {cudf::type_id::INT64, cudf::type_id::STRING}, {0, 0});

  auto result = pb::decode_protobuf(*input, options);

  ASSERT_EQ(result->type().id(), cudf::type_id::STRUCT);
  ASSERT_EQ(result->size(), 2);
  ASSERT_EQ(result->num_children(), 2);
  EXPECT_EQ(result->child(0).type().id(), cudf::type_id::INT64);
  EXPECT_EQ(result->child(1).type().id(), cudf::type_id::STRING);
  EXPECT_EQ(result->child(0).null_count(), 2);
  EXPECT_EQ(result->child(1).null_count(), 2);
}

TEST_F(ProtobufReaderTest, NullInputRowsPropagateToStruct)
{
  auto msg   = encode_varint_field(1, 42);
  auto input = make_binary_column({msg, {}, msg}, {true, false, true});

  auto options = make_scalar_options({1}, {cudf::type_id::INT64}, {0});

  auto result = pb::decode_protobuf(*input, options);

  ASSERT_EQ(result->size(), 3);
  EXPECT_EQ(result->null_count(), 1);
}

TEST_F(ProtobufReaderTest, MultipleNumericTypesShape)
{
  auto input = make_binary_column({encode_varint_field(1, 1)});

  auto options = make_scalar_options({1, 2, 3, 4, 5},
                                     {cudf::type_id::BOOL8,
                                      cudf::type_id::INT32,
                                      cudf::type_id::INT64,
                                      cudf::type_id::FLOAT32,
                                      cudf::type_id::FLOAT64},
                                     {0, 0, 0, 0, 0});

  auto result = pb::decode_protobuf(*input, options);

  ASSERT_EQ(result->num_children(), 5);
  EXPECT_EQ(result->child(0).type().id(), cudf::type_id::BOOL8);
  EXPECT_EQ(result->child(1).type().id(), cudf::type_id::INT32);
  EXPECT_EQ(result->child(2).type().id(), cudf::type_id::INT64);
  EXPECT_EQ(result->child(3).type().id(), cudf::type_id::FLOAT32);
  EXPECT_EQ(result->child(4).type().id(), cudf::type_id::FLOAT64);
}

CUDF_TEST_PROGRAM_MAIN()
