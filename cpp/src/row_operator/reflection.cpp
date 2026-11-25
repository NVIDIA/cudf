/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "reflection.hpp"

#include <cudf/column/column_child_offsets.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rtcx/rtcx.hpp>

#include <format>
#include <stdexcept>

namespace cudf::row_operators {

std::string_view to_string(algorithm alg)
{
  switch (alg) {
    case algorithm::xxhash_32: return "xxhash_32";
    case algorithm::xxhash_64: return "xxhash_64";
    case algorithm::murmurhash3_x86_32: return "murmurhash3_x86_32";
    case algorithm::murmurhash3_x64_128: return "murmurhash3_x64_128";
    case algorithm::spark_murmurhash3_x86_32: return "spark_murmurhash3_x86_32";
    default:
      CUDF_FAIL(std::format("Unsupported hash algorithm: {}", static_cast<int>(alg)),
                std::invalid_argument);
  }
}

namespace {

// Spell the ABI names explicitly: type_to_name() is intended for diagnostics, not code generation.
std::string_view element_type_name(data_type type)
{
  using enum type_id;
  switch (type.id()) {
    case INT8: return "cudf_abi::int8_t";
    case INT16: return "cudf_abi::int16_t";
    case INT32: return "cudf_abi::int32_t";
    case INT64: return "cudf_abi::int64_t";
    case UINT8: return "cudf_abi::uint8_t";
    case UINT16: return "cudf_abi::uint16_t";
    case UINT32: return "cudf_abi::uint32_t";
    case UINT64: return "cudf_abi::uint64_t";
    case FLOAT32: return "cudf_abi::float32_t";
    case FLOAT64: return "cudf_abi::float64_t";
    case BOOL8: return "cudf_abi::bool8_t";
    case STRING: return "cudf_abi::string_view";
    case DECIMAL32: return "cudf_abi::decimal32";
    case DECIMAL64: return "cudf_abi::decimal64";
    case DECIMAL128: return "cudf_abi::decimal128";
    case TIMESTAMP_DAYS: return "cudf_abi::timestamp_D";
    case TIMESTAMP_SECONDS: return "cudf_abi::timestamp_s";
    case TIMESTAMP_MILLISECONDS: return "cudf_abi::timestamp_ms";
    case TIMESTAMP_MICROSECONDS: return "cudf_abi::timestamp_us";
    case TIMESTAMP_NANOSECONDS: return "cudf_abi::timestamp_ns";
    case DURATION_DAYS: return "cudf_abi::duration_D";
    case DURATION_SECONDS: return "cudf_abi::duration_s";
    case DURATION_MILLISECONDS: return "cudf_abi::duration_ms";
    case DURATION_MICROSECONDS: return "cudf_abi::duration_us";
    case DURATION_NANOSECONDS: return "cudf_abi::duration_ns";
    default:
      CUDF_FAIL(std::format("Unsupported row operator element type '{}'", type_to_name(type)),
                data_type_error);
  }
}

struct column_info {
  std::string_view layout;
  data_type element;
};

column_info get_column_info(column_view column)
{
  bool const dictionary = column.type().id() == type_id::DICTIONARY32;
  if (dictionary) {
    CUDF_EXPECTS(column.num_children() > dictionary_keys_column_index,
                 "Dictionary columns require indices and keys",
                 data_type_error);
    column = column.child(dictionary_keys_column_index);
  }
  bool const list = column.type().id() == type_id::LIST;
  if (list) {
    CUDF_EXPECTS(column.num_children() > lists_column_view::child_column_index,
                 "List columns require offsets and values",
                 data_type_error);
    column = column.child(lists_column_view::child_column_index);
  }
  static_cast<void>(element_type_name(column.type()));
  return {dictionary ? (list ? "dictionary_list_layout" : "dictionary_layout")
                     : (list ? "list_layout" : "plain_layout"),
          column.type()};
}

}  // namespace

std::string reflect_hash_output(algorithm alg)
{
  switch (alg) {
    case algorithm::murmurhash3_x86_32:
    case algorithm::xxhash_32:
    case algorithm::spark_murmurhash3_x86_32: return "cudf::row_operators::Hash32";
    case algorithm::murmurhash3_x64_128: return "cudf::row_operators::Hash128";
    case algorithm::xxhash_64: return "cudf::row_operators::Hash64";
    default:
      CUDF_FAIL(std::format("Unsupported hash algorithm: {}", static_cast<int>(alg)),
                std::invalid_argument);
  }
}

std::string reflect_layout(column_view const& column)
{
  auto const info = get_column_info(column);
  return rtcx::reflect_template(std::format("cudf_abi::{}", info.layout),
                                std::string{element_type_name(info.element)});
}

std::string reflect_element(column_view const& column)
{
  return std::string{element_type_name(get_column_info(column).element)};
}

std::string reflect_hasher(algorithm alg, column_view const& column, size_type index)
{
  CUDF_EXPECTS(index >= 0, "Column index cannot be negative", std::invalid_argument);
  auto const shape = reflect_layout(column);
  return rtcx::reflect_template(
    std::format("cudf::row_operators::{}", to_string(alg)), std::to_string(index), shape);
}

}  // namespace cudf::row_operators
