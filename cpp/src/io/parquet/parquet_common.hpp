/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <string>

namespace cudf::io::parquet::detail {

// Parquet 4-byte magic number "PAR1"
constexpr uint32_t parquet_magic = (('P' << 0) | ('A' << 8) | ('R' << 16) | ('1' << 24));

// Shared-memory window size in bytes for parsing BYTE_ARRAY string dictionaries.
// Default value chosen for best balance between memory usage, occupancy and performance.
constexpr int string_dict_index_window_size = 4096;

// Max decimal precisions according to the parquet spec:
// https://github.com/apache/parquet-format/blob/master/LogicalTypes.md#decimal
auto constexpr MAX_DECIMAL32_PRECISION  = 9;
auto constexpr MAX_DECIMAL64_PRECISION  = 18;
auto constexpr MAX_DECIMAL128_PRECISION = 38;  // log10(2^(sizeof(int128_t) * 8 - 1) - 1)

// Constants copied from arrow source and renamed to match the case
int32_t constexpr MESSAGE_DECODER_NEXT_REQUIRED_SIZE_INITIAL         = sizeof(int32_t);
int32_t constexpr MESSAGE_DECODER_NEXT_REQUIRED_SIZE_METADATA_LENGTH = sizeof(int32_t);
int32_t constexpr IPC_CONTINUATION_TOKEN                             = -1;
std::string const ARROW_SCHEMA_KEY                                   = "ARROW:schema";

// Schema type ipc message has zero length body
int64_t constexpr SCHEMA_HEADER_TYPE_IPC_MESSAGE_BODYLENGTH = 0;

// bit space we are reserving in column_buffer::user_data
constexpr uint32_t PARQUET_COLUMN_BUFFER_SCHEMA_MASK          = (0xff'ffffu);
constexpr uint32_t PARQUET_COLUMN_BUFFER_FLAG_LIST_TERMINATED = (1 << 24);
// if this column has a list parent anywhere above it in the hierarchy
constexpr uint32_t PARQUET_COLUMN_BUFFER_FLAG_HAS_LIST_PARENT = (1 << 25);

}  // namespace cudf::io::parquet::detail
