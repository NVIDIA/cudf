/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <benchmarks/common/generate_input.hpp>
#include <benchmarks/io/cuio_common.hpp>

#include <cudf/io/types.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <nvbench/nvbench.cuh>

enum class data_type : int32_t {
  INTEGRAL        = static_cast<int32_t>(type_group_id::INTEGRAL),
  INTEGRAL_SIGNED = static_cast<int32_t>(type_group_id::INTEGRAL_SIGNED),
  FLOAT           = static_cast<int32_t>(type_group_id::FLOATING_POINT),
  BOOL8           = static_cast<int32_t>(cudf::type_id::BOOL8),
  DECIMAL         = static_cast<int32_t>(type_group_id::FIXED_POINT),
  TIMESTAMP       = static_cast<int32_t>(type_group_id::TIMESTAMP),
  DURATION        = static_cast<int32_t>(type_group_id::DURATION),
  STRING          = static_cast<int32_t>(cudf::type_id::STRING),
  LIST            = static_cast<int32_t>(cudf::type_id::LIST),
  STRUCT          = static_cast<int32_t>(cudf::type_id::STRUCT)
};

// NVBENCH_DECLARE_ENUM_TYPE_STRINGS macro must be used from global namespace scope
NVBENCH_DECLARE_ENUM_TYPE_STRINGS(
  data_type,
  [](data_type value) {
    switch (value) {
      case data_type::INTEGRAL: return "INTEGRAL";
      case data_type::INTEGRAL_SIGNED: return "INTEGRAL_SIGNED";
      case data_type::FLOAT: return "FLOAT";
      case data_type::BOOL8: return "BOOL8";
      case data_type::DECIMAL: return "DECIMAL";
      case data_type::TIMESTAMP: return "TIMESTAMP";
      case data_type::DURATION: return "DURATION";
      case data_type::STRING: return "STRING";
      case data_type::LIST: return "LIST";
      case data_type::STRUCT: return "STRUCT";
      default: return "Unknown";
    }
  },
  [](auto) { return std::string{}; })

NVBENCH_DECLARE_ENUM_TYPE_STRINGS(
  io_type,
  [](auto value) {
    switch (value) {
      case io_type::FILEPATH: return "FILEPATH";
      case io_type::HOST_BUFFER: return "HOST_BUFFER";
      case io_type::PINNED_BUFFER: return "PINNED_BUFFER";
      case io_type::DEVICE_BUFFER: return "DEVICE_BUFFER";
      case io_type::VOID: return "VOID";
      default: return "Unknown";
    }
  },
  [](auto) { return std::string{}; })

NVBENCH_DECLARE_ENUM_TYPE_STRINGS(
  cudf::io::compression_type,
  [](auto value) {
    switch (value) {
      case cudf::io::compression_type::SNAPPY: return "SNAPPY";
      case cudf::io::compression_type::GZIP: return "GZIP";
      case cudf::io::compression_type::ZSTD: return "ZSTD";
      case cudf::io::compression_type::ZLIB: return "ZLIB";
      case cudf::io::compression_type::LZ4: return "LZ4";
      case cudf::io::compression_type::NONE: return "NONE";
      default: return "Unknown";
    }
  },
  [](auto) { return std::string{}; })

enum class uses_index : bool { YES, NO };

enum class uses_numpy_dtype : bool { YES, NO };

NVBENCH_DECLARE_ENUM_TYPE_STRINGS(
  uses_index,
  [](auto value) {
    switch (value) {
      case uses_index::YES: return "YES";
      case uses_index::NO: return "NO";
      default: return "Unknown";
    }
  },
  [](auto) { return std::string{}; })

NVBENCH_DECLARE_ENUM_TYPE_STRINGS(
  uses_numpy_dtype,
  [](auto value) {
    switch (value) {
      case uses_numpy_dtype::YES: return "YES";
      case uses_numpy_dtype::NO: return "NO";
      default: return "Unknown";
    }
  },
  [](auto) { return std::string{}; })

NVBENCH_DECLARE_ENUM_TYPE_STRINGS(
  column_selection,
  [](auto value) {
    switch (value) {
      case column_selection::ALL: return "ALL";
      case column_selection::ALTERNATE: return "ALTERNATE";
      case column_selection::FIRST_HALF: return "FIRST_HALF";
      case column_selection::SECOND_HALF: return "SECOND_HALF";
      default: return "Unknown";
    }
  },
  [](auto) { return std::string{}; })

NVBENCH_DECLARE_ENUM_TYPE_STRINGS(
  row_selection,
  [](auto value) {
    switch (value) {
      case row_selection::ALL: return "ALL";
      case row_selection::BYTE_RANGE: return "BYTE_RANGE";
      case row_selection::NROWS: return "NROWS";
      case row_selection::SKIPFOOTER: return "SKIPFOOTER";
      case row_selection::STRIPES: return "STRIPES";
      case row_selection::ROW_GROUPS: return "ROW_GROUPS";
      default: return "Unknown";
    }
  },
  [](auto) { return std::string{}; })

NVBENCH_DECLARE_ENUM_TYPE_STRINGS(
  cudf::type_id,
  [](auto value) { return cudf::type_to_name(cudf::data_type{value}); },
  [](auto) { return std::string{}; })

enum class converts_strings : bool { YES, NO };

enum class uses_pandas_metadata : bool { YES, NO };

NVBENCH_DECLARE_ENUM_TYPE_STRINGS(
  converts_strings,
  [](auto value) {
    switch (value) {
      case converts_strings::YES: return "YES";
      case converts_strings::NO: return "NO";
      default: return "Unknown";
    }
  },
  [](auto) { return std::string{}; })

NVBENCH_DECLARE_ENUM_TYPE_STRINGS(
  uses_pandas_metadata,
  [](auto value) {
    switch (value) {
      case uses_pandas_metadata::YES: return "YES";
      case uses_pandas_metadata::NO: return "NO";
      default: return "Unknown";
    }
  },
  [](auto) { return std::string{}; })

enum class json_lines : bool { YES, NO };

enum class normalize_single_quotes : bool { YES, NO };

enum class normalize_whitespace : bool { YES, NO };

enum class mixed_types_as_string : bool { YES, NO };

enum class recovery_mode : bool { FAIL, RECOVER_WITH_NULL };

NVBENCH_DECLARE_ENUM_TYPE_STRINGS(
  json_lines,
  [](auto value) {
    switch (value) {
      case json_lines::YES: return "YES";
      case json_lines::NO: return "NO";
      default: return "Unknown";
    }
  },
  [](auto) { return std::string{}; })

NVBENCH_DECLARE_ENUM_TYPE_STRINGS(
  normalize_single_quotes,
  [](auto value) {
    switch (value) {
      case normalize_single_quotes::YES: return "YES";
      case normalize_single_quotes::NO: return "NO";
      default: return "Unknown";
    }
  },
  [](auto) { return std::string{}; })

NVBENCH_DECLARE_ENUM_TYPE_STRINGS(
  normalize_whitespace,
  [](auto value) {
    switch (value) {
      case normalize_whitespace::YES: return "YES";
      case normalize_whitespace::NO: return "NO";
      default: return "Unknown";
    }
  },
  [](auto) { return std::string{}; })

NVBENCH_DECLARE_ENUM_TYPE_STRINGS(
  mixed_types_as_string,
  [](auto value) {
    switch (value) {
      case mixed_types_as_string::YES: return "YES";
      case mixed_types_as_string::NO: return "NO";
      default: return "Unknown";
    }
  },
  [](auto) { return std::string{}; })

NVBENCH_DECLARE_ENUM_TYPE_STRINGS(
  recovery_mode,
  [](auto value) {
    switch (value) {
      case recovery_mode::FAIL: return "FAIL";
      case recovery_mode::RECOVER_WITH_NULL: return "RECOVER_WITH_NULL";
      default: return "Unknown";
    }
  },
  [](auto) { return std::string{}; })
