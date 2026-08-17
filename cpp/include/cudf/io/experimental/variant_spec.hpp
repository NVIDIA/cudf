/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>

namespace cudf::io::parquet::experimental {

/**
 * @brief Low 2 bits of a VARIANT value's metadata byte: the basic type.
 */
enum class variant_basic_type : uint8_t {
  PRIMITIVE    = 0,
  SHORT_STRING = 1,
  OBJECT       = 2,
  ARRAY        = 3,
};

/**
 * @brief Physical type ID carried in the value_header of a primitive VARIANT value.
 */
enum class variant_primitive_type : uint8_t {
  NULLVAL              = 0,
  BOOLEAN_TRUE         = 1,
  BOOLEAN_FALSE        = 2,
  INT8                 = 3,
  INT16                = 4,
  INT32                = 5,
  INT64                = 6,
  FLOAT64              = 7,
  DECIMAL4             = 8,
  DECIMAL8             = 9,
  DECIMAL16            = 10,
  DATE                 = 11,
  TIMESTAMP_MICROS     = 12,
  TIMESTAMP_NTZ_MICROS = 13,
  FLOAT32              = 14,
  BINARY               = 15,
  LONG_STRING          = 16,
  TIME_NTZ_MICROS      = 17,
  TIMESTAMP_NANOS      = 18,
  TIMESTAMP_NTZ_NANOS  = 19,
  UUID                 = 20,
};

/**
 * @brief Logical type of a VARIANT value as returned by get_variant_type_id.
 */
enum class variant_logical_type : uint8_t {
  OBJECT        = 0,
  ARRAY         = 1,
  NULL_VALUE    = 2,
  BOOLEAN       = 3,
  LONG_VALUE    = 4,
  STRING        = 5,
  DOUBLE_VALUE  = 6,
  DECIMAL       = 7,
  DATE          = 8,
  TIMESTAMP     = 9,
  TIMESTAMP_NTZ = 10,
  FLOAT_VALUE   = 11,
  BINARY        = 12,
  UUID          = 13,
  TIME_NTZ      = 14,
};

}  // namespace cudf::io::parquet::experimental
