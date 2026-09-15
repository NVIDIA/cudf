/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * @file arrow_schema_writer.cpp
 * @brief Arrow IPC schema writer implementation
 */

#include "arrow_schema_writer.hpp"

#include "io/parquet/parquet_common.hpp"
#include "io/utilities/base64_utilities.hpp"
#include "ipc/Message_generated.h"
#include "ipc/Schema_generated.h"
#include "writer_impl_helpers.hpp"

#include <cudf/utilities/error.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <cuda/iterator>
#include <cuda/std/tuple>

namespace cudf::io::parquet::detail {

using namespace cudf::io::detail;

namespace {

// Copied over from arrow source for better code readability
namespace flatbuf       = cudf::io::parquet::flatbuf;
using FlatBufferBuilder = flatbuffers::FlatBufferBuilder;
using DictionaryOffset  = flatbuffers::Offset<flatbuf::DictionaryEncoding>;
using FieldOffset       = flatbuffers::Offset<flatbuf::Field>;
using Offset            = flatbuffers::Offset<void>;
using FBString          = flatbuffers::Offset<flatbuffers::String>;

/**
 * @brief Recursively construct the arrow schema (fields) tree
 *
 * @param fbb The root flatbuffer builder object instance
 * @param column A view of the column
 * @param column_metadata Metadata of the column
 * @param write_mode Flag to indicate that we are guaranteeing a single table write
 * @param utc_timestamps Flag to indicate if timestamps are UTC
 *
 * @return Flatbuffer offset to the constructed field
 */
FieldOffset make_arrow_schema_fields(FlatBufferBuilder& fbb,
                                     cudf::detail::LinkedColPtr const& column,
                                     column_in_metadata const& column_metadata,
                                     single_write_mode const write_mode,
                                     bool const utc_timestamps);

/**
 * @brief Functor to convert cudf column metadata to arrow schema field metadata
 */
struct dispatch_to_flatbuf {
  FlatBufferBuilder& fbb;
  cudf::detail::LinkedColPtr const& col;
  column_in_metadata const& col_meta;
  single_write_mode const write_mode;
  bool const utc_timestamps;
  Offset& field_offset;
  flatbuf::Type& field_type_id;
  std::vector<FieldOffset>& children;

  template <typename T>
  void operator()()
    requires(std::is_same_v<T, bool>)
  {
    field_type_id = flatbuf::Type_Bool;
    field_offset  = flatbuf::CreateBool(fbb).Union();
  }

  template <typename T>
  void operator()()
    requires(std::is_same_v<T, int8_t>)
  {
    field_type_id = flatbuf::Type_Int;
    field_offset  = flatbuf::CreateInt(fbb, 8, std::numeric_limits<T>::is_signed).Union();
  }

  template <typename T>
  void operator()()
    requires(std::is_same_v<T, int16_t>)
  {
    field_type_id = flatbuf::Type_Int;
    field_offset  = flatbuf::CreateInt(fbb, 16, std::numeric_limits<T>::is_signed).Union();
  }

  template <typename T>
  void operator()()
    requires(std::is_same_v<T, int32_t>)
  {
    field_type_id = flatbuf::Type_Int;
    field_offset  = flatbuf::CreateInt(fbb, 32, std::numeric_limits<T>::is_signed).Union();
  }

  template <typename T>
  void operator()()
    requires(std::is_same_v<T, int64_t>)
  {
    field_type_id = flatbuf::Type_Int;
    field_offset  = flatbuf::CreateInt(fbb, 64, std::numeric_limits<T>::is_signed).Union();
  }

  template <typename T>
  void operator()()
    requires(std::is_same_v<T, uint8_t>)
  {
    field_type_id = flatbuf::Type_Int;
    field_offset  = flatbuf::CreateInt(fbb, 8, std::numeric_limits<T>::is_signed).Union();
  }

  template <typename T>
  void operator()()
    requires(std::is_same_v<T, uint16_t>)
  {
    field_type_id = flatbuf::Type_Int;
    field_offset  = flatbuf::CreateInt(fbb, 16, std::numeric_limits<T>::is_signed).Union();
  }

  template <typename T>
  void operator()()
    requires(std::is_same_v<T, uint32_t>)
  {
    field_type_id = flatbuf::Type_Int;
    field_offset  = flatbuf::CreateInt(fbb, 32, std::numeric_limits<T>::is_signed).Union();
  }

  template <typename T>
  void operator()()
    requires(std::is_same_v<T, uint64_t>)
  {
    field_type_id = flatbuf::Type_Int;
    field_offset  = flatbuf::CreateInt(fbb, 64, std::numeric_limits<T>::is_signed).Union();
  }

  template <typename T>
  void operator()()
    requires(std::is_same_v<T, float>)
  {
    field_type_id = flatbuf::Type_FloatingPoint;
    field_offset  = flatbuf::CreateFloatingPoint(fbb, flatbuf::Precision::Precision_SINGLE).Union();
  }

  template <typename T>
  void operator()()
    requires(std::is_same_v<T, double>)
  {
    field_type_id = flatbuf::Type_FloatingPoint;
    field_offset  = flatbuf::CreateFloatingPoint(fbb, flatbuf::Precision::Precision_DOUBLE).Union();
  }

  template <typename T>
  void operator()()
    requires(std::is_same_v<T, cudf::string_view>)
  {
    field_type_id = flatbuf::Type_Utf8View;
    field_offset  = flatbuf::CreateUtf8View(fbb).Union();
  }

  template <typename T>
  void operator()()
    requires(std::is_same_v<T, cudf::timestamp_D>)
  {
    field_type_id = flatbuf::Type_Date;
    // Date type (Set unit type to DAY for arrows's Date32)
    field_offset = flatbuf::CreateDate(fbb, flatbuf::DateUnit_DAY).Union();
  }

  template <typename T>
  void operator()()
    requires(std::is_same_v<T, cudf::timestamp_s>)
  {
    field_type_id = flatbuf::Type_Timestamp;
    // Use one of the strings: "UTC", "Etc/UTC" or "+00:00" to indicate a native UTC timestamp
    field_offset = flatbuf::CreateTimestamp(
                     fbb, flatbuf::TimeUnit_SECOND, (utc_timestamps) ? fbb.CreateString("UTC") : 0)
                     .Union();
  }

  template <typename T>
  void operator()()
    requires(std::is_same_v<T, cudf::timestamp_ms>)
  {
    field_type_id = flatbuf::Type_Timestamp;
    // Use one of the strings: "UTC", "Etc/UTC" or "+00:00" to indicate a native UTC timestamp
    field_offset =
      flatbuf::CreateTimestamp(
        fbb, flatbuf::TimeUnit_MILLISECOND, (utc_timestamps) ? fbb.CreateString("UTC") : 0)
        .Union();
  }

  template <typename T>
  void operator()()
    requires(std::is_same_v<T, cudf::timestamp_us>)
  {
    field_type_id = flatbuf::Type_Timestamp;
    // Use one of the strings: "UTC", "Etc/UTC" or "+00:00" to indicate a native UTC timestamp
    field_offset =
      flatbuf::CreateTimestamp(
        fbb, flatbuf::TimeUnit_MICROSECOND, (utc_timestamps) ? fbb.CreateString("UTC") : 0)
        .Union();
  }

  template <typename T>
  void operator()()
    requires(std::is_same_v<T, cudf::timestamp_ns>)
  {
    field_type_id = flatbuf::Type_Timestamp;
    // Use one of the strings: "UTC", "Etc/UTC" or "+00:00" to indicate a native UTC timestamp
    field_offset =
      flatbuf::CreateTimestamp(
        fbb, flatbuf::TimeUnit_NANOSECOND, (utc_timestamps) ? fbb.CreateString("UTC") : 0)
        .Union();
  }

  template <typename T>
  void operator()()
    requires(std::is_same_v<T, cudf::duration_D>)
  {
    // `duration_D` is written as TimeType as `duration_D` is not a valid arrow type.
    //  This also allows for easy and faithful roundtripping with cudf.
    field_type_id = flatbuf::Type_Time;
    field_offset  = flatbuf::CreateTime(fbb, flatbuf::TimeUnit_MILLISECOND).Union();
  }

  template <typename T>
  void operator()()
    requires(std::is_same_v<T, cudf::duration_s>)
  {
    field_type_id = flatbuf::Type_Duration;
    field_offset  = flatbuf::CreateDuration(fbb, flatbuf::TimeUnit_SECOND).Union();
  }

  template <typename T>
  void operator()()
    requires(std::is_same_v<T, cudf::duration_ms>)
  {
    field_type_id = flatbuf::Type_Duration;
    field_offset  = flatbuf::CreateDuration(fbb, flatbuf::TimeUnit_MILLISECOND).Union();
  }

  template <typename T>
  void operator()()
    requires(std::is_same_v<T, cudf::duration_us>)
  {
    field_type_id = flatbuf::Type_Duration;
    field_offset  = flatbuf::CreateDuration(fbb, flatbuf::TimeUnit_MICROSECOND).Union();
  }

  template <typename T>
  void operator()()
    requires(std::is_same_v<T, cudf::duration_ns>)
  {
    field_type_id = flatbuf::Type_Duration;
    field_offset  = flatbuf::CreateDuration(fbb, flatbuf::TimeUnit_NANOSECOND).Union();
  }

  template <typename T>
  void operator()()
    requires(cudf::is_fixed_point<T>())
  {
    field_type_id = flatbuf::Type_Decimal;

    auto const [max_precision, bitwidth] = []() constexpr -> std::pair<int32_t, int32_t> {
      if constexpr (std::is_same_v<T, numeric::decimal32>) {
        return {MAX_DECIMAL32_PRECISION, 32};
      } else if constexpr (std::is_same_v<T, numeric::decimal64>) {
        return {MAX_DECIMAL64_PRECISION, 64};
      } else if constexpr (std::is_same_v<T, numeric::decimal128>) {
        return {MAX_DECIMAL128_PRECISION, 128};
      } else {
        CUDF_FAIL("Unsupported fixed point type for arrow schema writer");
      }
    }();

    field_offset =
      flatbuf::CreateDecimal(
        fbb,
        (col_meta.is_decimal_precision_set()) ? col_meta.get_decimal_precision() : max_precision,
        col->type().scale(),
        bitwidth)
        .Union();
  }

  template <typename T>
  void operator()()
    requires(cudf::is_nested<T>())
  {
    // Lists are represented differently in arrow and cuDF.
    // cuDF representation: List<int>: "col_name" : { "list", "element:int" } (2 children)
    // arrow schema representation: List<int>: "col_name" : { "list<item:int>" } (1 child)
    // Hence, we only need to process the second child of the list.
    if constexpr (std::is_same_v<T, cudf::list_view>) {
      children.emplace_back(make_arrow_schema_fields(
        fbb, col->children[1], col_meta.child(1), write_mode, utc_timestamps));
      field_type_id = flatbuf::Type_List;
      field_offset  = flatbuf::CreateList(fbb).Union();
    }

    // Traverse the struct in DFS manner and process children fields.
    else if constexpr (std::is_same_v<T, cudf::struct_view>) {
      std::transform(cuda::counting_iterator<std::size_t>{0},
                     cuda::counting_iterator{col->children.size()},
                     std::back_inserter(children),
                     [&](auto const idx) {
                       return make_arrow_schema_fields(
                         fbb, col->children[idx], col_meta.child(idx), write_mode, utc_timestamps);
                     });
      field_type_id = flatbuf::Type_Struct_;
      field_offset  = flatbuf::CreateStruct_(fbb).Union();
    }
  }

  template <typename T>
  void operator()()
    requires(cudf::is_dictionary<T>())
  {
    // `dictionary32` columns are not written to parquet by cudf.
    CUDF_FAIL("Dictionary columns are not supported for writing");
  }
};

FieldOffset make_arrow_schema_fields(FlatBufferBuilder& fbb,
                                     cudf::detail::LinkedColPtr const& column,
                                     column_in_metadata const& column_metadata,
                                     single_write_mode const write_mode,
                                     bool const utc_timestamps)
{
  // Variables to be set by the dispatch_to_flatbuf functor
  Offset field_offset         = 0;
  flatbuf::Type field_type_id = flatbuf::Type_NONE;
  std::vector<FieldOffset> children;

  cudf::type_dispatcher(column->type(),
                        dispatch_to_flatbuf{fbb,
                                            column,
                                            column_metadata,
                                            write_mode,
                                            utc_timestamps,
                                            field_offset,
                                            field_type_id,
                                            children});

  // push to field offsets vector
  return flatbuf::CreateField(
    fbb,
    fbb.CreateString(column_metadata.get_name()),                    // name
    is_output_column_nullable(column, column_metadata, write_mode),  // nullable
    field_type_id,                                                   // type id
    field_offset,                                                    // field offset
    {0},                                                             // DictionaryOffset
    fbb.CreateVector(children.data(), children.size()));             // children vector
}

}  // namespace

std::string construct_arrow_schema_ipc_message(cudf::detail::LinkedColVector const& linked_columns,
                                               table_input_metadata const& metadata,
                                               single_write_mode const write_mode,
                                               bool const utc_timestamps)
{
  // Lambda function to convert int32 to a string of uint8 bytes
  auto const convert_int32_to_byte_string = [&](int32_t const value) {
    std::array<uint8_t, sizeof(int32_t)> buffer{};
    std::memcpy(buffer.data(), &value, sizeof(int32_t));
    return std::string(reinterpret_cast<char*>(buffer.data()), buffer.size());
  };

  // Instantiate a flatbuffer builder
  FlatBufferBuilder fbb;

  // Create an empty field offset vector and reserve space for linked columns
  std::vector<FieldOffset> field_offsets;
  field_offsets.reserve(linked_columns.size());

  // populate field offsets (aka schema fields)
  std::transform(
    cuda::make_zip_iterator(
      cuda::std::make_tuple(linked_columns.begin(), metadata.column_metadata.begin())),
    cuda::make_zip_iterator(
      cuda::std::make_tuple(linked_columns.end(), metadata.column_metadata.end())),
    std::back_inserter(field_offsets),
    [&](auto const& elem) {
      return make_arrow_schema_fields(
        fbb, cuda::std::get<0>(elem), cuda::std::get<1>(elem), write_mode, utc_timestamps);
    });

  // Build an arrow:schema flatbuffer using the field offset vector and use it as the header to
  // create an ipc message flatbuffer
  fbb.Finish(flatbuf::CreateMessage(
    fbb,
    flatbuf::MetadataVersion_V5,    // Metadata version V5 (latest)
    flatbuf::MessageHeader_Schema,  // Schema type message header
    flatbuf::CreateSchema(fbb,
                          flatbuf::Endianness::Endianness_Little,
                          fbb.CreateVector(field_offsets))
      .Union(),                                // arrow:schema built from the field vector
    SCHEMA_HEADER_TYPE_IPC_MESSAGE_BODYLENGTH  // Body length is zero for schema type ipc message
    ));

  // Construct the final string and store it here to use its view in base64_encode
  std::string const ipc_message =
    convert_int32_to_byte_string(IPC_CONTINUATION_TOKEN) +
    // Since the schema type ipc message doesn't have a body, the flatbuffer size is equal to the
    // ipc message's metadata length
    convert_int32_to_byte_string(fbb.GetSize()) +
    std::string(reinterpret_cast<char*>(fbb.GetBufferPointer()), fbb.GetSize());

  // Encode the final ipc message string to base64 and return
  return cudf::io::detail::base64_encode(ipc_message);
}

}  // namespace cudf::io::parquet::detail
