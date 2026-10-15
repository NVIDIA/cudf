/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "row_hash_descriptor.hpp"

#include <cudf/column/column.hpp>
#include <cudf/dictionary/dictionary_column_view.hpp>
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/error.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <cuda_runtime.h>

#include <jit/cache.hpp>

#include <algorithm>
#include <format>

namespace cudf::hashing::detail::jit {
namespace {
void append_row_hash_types(column_view const& column,
                           std::vector<type_id>& types,
                           type_id& dictionary_index_type)
{
  auto current = column;
  while (true) {
    types.push_back(current.type().id());

    if (current.type().id() == type_id::DICTIONARY32) {
      dictionary_index_type =
        current.child(dictionary_column_view::indices_column_index).type().id();
      current = current.child(dictionary_column_view::keys_column_index);
    } else if ((current.type().id() == type_id::LIST || current.type().id() == type_id::STRUCT) &&
               current.num_children() > 0) {
      auto const child_index =
        current.type().id() == type_id::LIST ? lists_column_view::child_column_index : size_type{0};
      current = current.child(child_index);
    } else {
      return;
    }
  }
}

}  // namespace

row_hash_schema make_row_hash_schema(table_view const& table)
{
  row_hash_schema result;
  result.columns.reserve(table.num_columns());
  result.dictionary_index_types.reserve(table.num_columns());
  for (auto const& column : table) {
    auto& types      = result.columns.emplace_back();
    auto& index_type = result.dictionary_index_types.emplace_back(type_id::EMPTY);
    append_row_hash_types(column, types, index_type);
  }
  return result;
}

namespace {

char const* abi_type(type_id type)
{
  switch (type) {
    case type_id::INT8: return "i8";
    case type_id::INT16: return "i16";
    case type_id::INT32: return "i32";
    case type_id::INT64: return "i64";
    case type_id::UINT8: return "u8";
    case type_id::UINT16: return "u16";
    case type_id::UINT32: return "u32";
    case type_id::UINT64: return "u64";
    case type_id::FLOAT32: return "f32";
    case type_id::FLOAT64: return "f64";
    case type_id::BOOL8: return "bool8";
    case type_id::STRING: return "string_view";
    case type_id::DECIMAL32: return "decimal32";
    case type_id::DECIMAL64: return "decimal64";
    case type_id::DECIMAL128: return "decimal128";
    case type_id::TIMESTAMP_DAYS: return "timestamp_days";
    case type_id::TIMESTAMP_SECONDS: return "timestamp_seconds";
    case type_id::TIMESTAMP_MILLISECONDS: return "timestamp_milliseconds";
    case type_id::TIMESTAMP_MICROSECONDS: return "timestamp_microseconds";
    case type_id::TIMESTAMP_NANOSECONDS: return "timestamp_nanoseconds";
    case type_id::DURATION_DAYS: return "duration_days";
    case type_id::DURATION_SECONDS: return "duration_seconds";
    case type_id::DURATION_MILLISECONDS: return "duration_milliseconds";
    case type_id::DURATION_MICROSECONDS: return "duration_microseconds";
    case type_id::DURATION_NANOSECONDS: return "duration_nanoseconds";
    default: CUDF_FAIL("Unsupported type");
  }
}

/**
 * @brief Generates the schema-specific body of the `cudf_row_hash` device function.
 *
 * Each column becomes a direct call to a typed hash overload. The resulting source is included
 * after the shared declarations in the CUDA dispatch wrapper, compiled to LTO IR, and linked with
 * the precompiled hash definitions. Only hashing code is emitted.
 *
 * @param schema Logical type paths and physical dictionary index types for the input columns
 * @param options Selects null checks and the specialized FLOAT64 list hash implementation
 * @return CUDA C++ source defining `cudf_row_hash`
 * @throws std::invalid_argument If a column path is empty or has an unsupported nesting pattern
 * @throws cudf::logic_error If an element type has no ABI representation
 */
std::string make_row_hash_dispatch_source(row_hash_schema const& schema,
                                          row_function_options options)
{
  auto source = std::string{R"CUDA(using namespace cudf::abi_types;
using namespace cudf::row_operators::jit::functions;

extern "C" __device__ u32 cudf_row_hash(signed char const* columns, i32 row, u32 seed)
{
)CUDA"};
  for (std::size_t i = 0; i < schema.columns.size(); ++i) {
    auto const& path = schema.columns[i];
    CUDF_EXPECTS(!path.empty(), "Column schema cannot be empty", std::invalid_argument);
    auto const root = path.front();

    // The final path entry selects the logical element overload. Decimal and chrono tags keep
    // their own identities; the definitions load the corresponding cuDF element types.
    auto const suffix          = abi_type(path.back());
    bool const list            = root == type_id::LIST;
    bool const dictionary      = root == type_id::DICTIONARY32;
    bool const dictionary_list = dictionary && path.size() == 3 && path[1] == type_id::LIST;

    // Dispatch supports one list layer, optionally stored as dictionary keys. Reject reversed
    // nesting (list<dictionary<T>>), deeper nesting, and structs before emitting overload calls.
    CUDF_EXPECTS((list && path.size() == 2) ||
                   (dictionary && (path.size() == 2 || dictionary_list)) ||
                   (!list && !dictionary && root != type_id::STRUCT && path.size() == 1),
                 "LTO function dispatch supports flat, LIST<flat>, DICTIONARY<flat>, "
                 "and DICTIONARY<LIST<flat>> schemas",
                 std::invalid_argument);

    std::string encoding = dictionary_list ? "_dictionary_list"
                           : dictionary    ? "_dictionary"
                           : list          ? "_list"
                                           : "";
    std::string args     = std::format("type_tag<{}>{{}}", suffix);

    // Only dictionary<list<int32>> has explicit index-width overloads. Other element types, or
    // schemas without index metadata, use the generic dictionary index load in the definitions.
    if (dictionary_list && path.back() == type_id::INT32 &&
        i < schema.dictionary_index_types.size()) {
      auto index_type = schema.dictionary_index_types[i];
      if (index_type == type_id::INT8 || index_type == type_id::INT16 ||
          index_type == type_id::INT32) {
        args += std::format(", index_type_tag<{}>{{}}", abi_type(index_type));
      }
    }

    // The FLOAT64 list variant canonicalizes signed zero and NaNs using integer bit operations
    // while preserving MurmurHash results. Flat and dictionary<flat> calls use the usual overload.
    auto const function =
      "hash" + encoding +
      ((list || dictionary_list) && path.back() == type_id::FLOAT64 && options.float64_bits
         ? "_bits"
         : "");

    // Match row-hasher semantics: each column starts with the same seed, then column hashes are
    // combined in order. The first hash initializes the result without an extra combine.
    source += std::format("  auto h{} = {}(columns, {}, row, seed, {}, {});\n",
                          i,
                          function,
                          i,
                          options.check_nulls ? "true" : "false",
                          args);
    source += i == 0 ? "  auto result = h0;\n"
                     : std::format("  result = cudf_hash_combine(result, h{});\n", i);
  }

  // With no columns there is nothing to combine, so the row hash is the supplied seed.
  source += schema.columns.empty() ? "  return seed;\n" : "  return result;\n";
  source += "}\n";
  return source;
}

}  // namespace

rtcx::blob get_row_function_dispatch_fragment(std::string const& name,
                                              row_hash_schema const& schema,
                                              row_function_options options)
{
  auto const source = make_row_hash_dispatch_source(schema, options);
  // Keep schema-dependent code in a separate in-memory header after the wrapper's PCH boundary.
  // Its contents participate in the fragment cache key, distinguishing schemas and options.
  char const* names[]   = {"row_function_dispatch_generated.cuh"};
  char const* headers[] = {source.c_str()};
  return get_kernel_fragment(
    name, "cudf/cpp/src/row_operator/jit/row_function_dispatch.cu", names, headers, name);
}

}  // namespace cudf::hashing::detail::jit
