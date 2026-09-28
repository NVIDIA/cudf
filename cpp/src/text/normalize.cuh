/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "text/detail/cp_data.h"

#include <rmm/device_uvector.hpp>

#include <cuda/stream>

#include <cstdint>

namespace nvtext {
namespace detail {

/**
 * @brief Bit used to filter out invalid code points.
 *
 * When normalizing characters to code point values, if this bit is set,
 * the code point should be filtered out before returning from the normalizer.
 */
constexpr uint32_t FILTER_BIT = 22;

/**
 * @brief Retrieve new code point from metadata value.
 *
 * @param metadata Value from the codepoint_metadata table.
 * @return The replacement character if appropriate.
 */
__device__ constexpr uint32_t get_first_cp(uint32_t metadata) { return metadata & NEW_CP_MASK; }

/**
 * @brief Retrieve token category from the metadata value.
 *
 * Category values are 0-5:
 * 0 - character should be padded
 * 1 - pad character if lower-case
 * 2 - character should be removed
 * 3 - remove character if lower-case
 * 4 - whitespace character -- always replace
 * 5 - uncategorized
 *
 * @param metadata Value from the codepoint_metadata table.
 * @return Category value.
 */
__device__ constexpr uint32_t extract_token_cat(uint32_t metadata)
{
  return (metadata >> TOKEN_CAT_SHIFT) & TOKEN_CAT_MASK;
}

/**
 * @brief Return true if category of metadata value specifies the character should be replaced.
 */
__device__ constexpr bool should_remove_cp(uint32_t metadata, bool lower_case)
{
  auto const cat = extract_token_cat(metadata);
  return (cat == TOKEN_CAT_REMOVE_CHAR) || (lower_case && (cat == TOKEN_CAT_REMOVE_CHAR_IF_LOWER));
}

/**
 * @brief Return true if category of metadata value specifies the character should be padded.
 */
__device__ constexpr bool should_add_spaces(uint32_t metadata, bool lower_case)
{
  auto const cat = extract_token_cat(metadata);
  return (cat == TOKEN_CAT_ADD_SPACE) || (lower_case && (cat == TOKEN_CAT_ADD_SPACE_IF_LOWER));
}

/**
 * @brief Return true if category of metadata value specifies the character should be replaced.
 */
__device__ constexpr bool always_replace(uint32_t metadata)
{
  return extract_token_cat(metadata) == TOKEN_CAT_ALWAYS_REPLACE;
}

/**
 * @brief Returns true if metadata value includes a multi-character transform bit equal to 1.
 */
__device__ constexpr bool is_multi_char_transform(uint32_t metadata)
{
  return (metadata >> MULTICHAR_SHIFT) & MULTICHAR_MASK;
}

/**
 * @brief Returns true if the byte passed in could be a valid head byte for
 * a utf8 character. That is, not binary `10xxxxxx`
 */
__device__ constexpr bool is_head_byte(unsigned char utf8_byte) { return (utf8_byte >> 6) != 2; }

/**
 * @brief Retrieve the code point metadata table.
 *
 * @param stream CUDA stream used for device memory operations and kernel launches.
 */
rmm::device_uvector<codepoint_metadata_type> get_codepoint_metadata(cuda::stream_ref stream);

/**
 * @brief Retrieve the auxiliary code point metadata table.
 *
 * @param stream CUDA stream used for device memory operations and kernel launches.
 */
rmm::device_uvector<aux_codepoint_data_type> get_aux_codepoint_data(cuda::stream_ref stream);

}  // namespace detail
}  // namespace nvtext
