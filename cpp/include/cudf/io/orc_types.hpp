/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/utilities/export.hpp>

#include <cstdint>

/**
 * @file
 * @brief Enums for ORC-specific compression and encoding types.
 */

namespace CUDF_EXPORT cudf {
namespace io::orc {
/**
 * @addtogroup io_types
 * @{
 */

/**
 * @brief Identifies a compression algorithm
 */
enum CompressionKind : uint8_t {
  NONE   = 0,
  ZLIB   = 1,
  SNAPPY = 2,
  LZO    = 3,
  LZ4    = 4,
  ZSTD   = 5,
};

/**
 * @brief Identifies a data type in an orc file
 */
enum TypeKind : int8_t {
  INVALID_TYPE_KIND = -1,
  BOOLEAN           = 0,
  BYTE              = 1,
  SHORT             = 2,
  INT               = 3,
  LONG              = 4,
  FLOAT             = 5,
  DOUBLE            = 6,
  STRING            = 7,
  BINARY            = 8,
  TIMESTAMP         = 9,
  LIST              = 10,
  MAP               = 11,
  STRUCT            = 12,
  UNION             = 13,
  DECIMAL           = 14,
  DATE              = 15,
  VARCHAR           = 16,
  CHAR              = 17,
};

/**
 * @brief Identifies the type of data stream
 */
enum StreamKind : int8_t {
  INVALID_STREAM_KIND = -1,
  PRESENT             = 0,  // boolean stream of whether the next value is non-null
  DATA                = 1,  // the primary data stream
  LENGTH              = 2,  // the length of each value for variable length data
  DICTIONARY_DATA     = 3,  // the dictionary blob
  DICTIONARY_COUNT    = 4,  // deprecated prior to Hive 0.11
  SECONDARY           = 5,  // a secondary data stream
  ROW_INDEX           = 6,  // the index for seeking to particular row groups
  BLOOM_FILTER        = 7,  // original bloom filters used before ORC-101
  BLOOM_FILTER_UTF8   = 8,  // bloom filters that consistently use utf8
};

/**
 * @brief Identifies the encoding of columns
 */
enum ColumnEncodingKind : int8_t {
  INVALID_ENCODING_KIND = -1,
  DIRECT                = 0,  // the encoding is mapped directly to the stream using RLE v1
  DICTIONARY            = 1,  // the encoding uses a dictionary of unique values using RLE v1
  DIRECT_V2             = 2,  // the encoding is direct using RLE v2
  DICTIONARY_V2         = 3,  // the encoding is dictionary-based using RLE v2
};

/**
 * @brief Identifies the type of encoding in a protocol buffer
 */
enum ProtofType : uint8_t {
  VARINT      = 0,
  FIXED64     = 1,
  FIXEDLEN    = 2,
  START_GROUP = 3,  // deprecated
  END_GROUP   = 4,  // deprecated
  FIXED32     = 5,
  INVALID_6   = 6,
  INVALID_7   = 7,
};

/** @} */  // end of group
}  // namespace io::orc
}  // namespace CUDF_EXPORT cudf
