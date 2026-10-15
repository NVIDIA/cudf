/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

/**
 * @file
 * @brief Lightweight element representations and overload tags for CUDA JIT dispatch.
 *
 * Explicit fundamental types give NVCC and NVRTC the same overload signatures and name mangling.
 * Aggregates use the CUDA compiler's natural alignment and padding; they are not packed wire
 * formats. Decimal objects and string views describe materialized elements, rather than the
 * layout of their column buffers. This header has no dependencies so it can be included directly
 * by generated CUDA source.
 */

/** @brief Element representations and type tags shared by precompiled and JIT CUDA code. */
namespace cudf::abi_types {

using i8    = signed char;         ///< Signed 8-bit integer.
using i16   = short;               ///< Signed 16-bit integer.
using i32   = int;                 ///< Signed 32-bit integer.
using i64   = long long;           ///< Signed 64-bit integer.
using i128  = __int128;            ///< Signed 128-bit integer.
using u8    = unsigned char;       ///< Unsigned 8-bit integer.
using u16   = unsigned short;      ///< Unsigned 16-bit integer.
using u32   = unsigned int;        ///< Unsigned 32-bit integer.
using u64   = unsigned long long;  ///< Unsigned 64-bit integer.
using f32   = float;               ///< 32-bit floating-point value.
using f64   = double;              ///< 64-bit floating-point value.
using bool8 = bool;                ///< Boolean element using CUDA's `bool` representation.

using decimal32_storage  = i32;   ///< Unscaled 32-bit decimal column representation.
using decimal64_storage  = i64;   ///< Unscaled 64-bit decimal column representation.
using decimal128_storage = i128;  ///< Unscaled 128-bit decimal column representation.

/**
 * @brief Materialized 32-bit decimal with logical value `value * 10^scale`.
 *
 * Column data stores only @ref decimal32_storage; the scale is column metadata.
 */
struct decimal32 {
  i32 value;  ///< Signed unscaled representation.
  i32 scale;  ///< Base-10 exponent applied to the representation.
};

/**
 * @brief Materialized 64-bit decimal with logical value `value * 10^scale`.
 *
 * Column data stores only @ref decimal64_storage; the scale is column metadata.
 */
struct decimal64 {
  i64 value;  ///< Signed unscaled representation.
  i32 scale;  ///< Base-10 exponent applied to the representation.
};

/**
 * @brief Materialized 128-bit decimal with logical value `value * 10^scale`.
 *
 * Column data stores only @ref decimal128_storage; the scale is column metadata.
 */
struct decimal128 {
  i128 value;  ///< Signed unscaled representation.
  i32 scale;   ///< Base-10 exponent applied to the representation.
};

/**
 * @brief Non-owning device view of a UTF-8 string, matching the `cudf::string_view` layout.
 *
 * The referenced bytes must remain valid while the view is used. The byte sequence need not be
 * null-terminated; a length of -1 indicates that the character count has not been computed.
 */
struct string_view {
  char const* data;  ///< Pointer to UTF-8 bytes in device memory.
  i32 bytes;         ///< Number of bytes in the string.
  i32 length;        ///< Cached Unicode character count, or -1 when unknown.
};

/** @brief Timestamp expressed as days since the UNIX epoch. */
struct timestamp_days {
  i32 value;  ///< Signed count of days.
};

/** @brief Timestamp expressed as seconds since the UNIX epoch. */
struct timestamp_seconds {
  i64 value;  ///< Signed count of seconds.
};

/** @brief Timestamp expressed as milliseconds since the UNIX epoch. */
struct timestamp_milliseconds {
  i64 value;  ///< Signed count of milliseconds.
};

/** @brief Timestamp expressed as microseconds since the UNIX epoch. */
struct timestamp_microseconds {
  i64 value;  ///< Signed count of microseconds.
};

/** @brief Timestamp expressed as nanoseconds since the UNIX epoch. */
struct timestamp_nanoseconds {
  i64 value;  ///< Signed count of nanoseconds.
};

/** @brief Duration expressed as a count of days. */
struct duration_days {
  i32 value;  ///< Signed count of days.
};

/** @brief Duration expressed as a count of seconds. */
struct duration_seconds {
  i64 value;  ///< Signed count of seconds.
};

/** @brief Duration expressed as a count of milliseconds. */
struct duration_milliseconds {
  i64 value;  ///< Signed count of milliseconds.
};

/** @brief Duration expressed as a count of microseconds. */
struct duration_microseconds {
  i64 value;  ///< Signed count of microseconds.
};

/** @brief Duration expressed as a count of nanoseconds. */
struct duration_nanoseconds {
  i64 value;  ///< Signed count of nanoseconds.
};

/**
 * @brief Nullable element represented by a payload and an explicit validity byte.
 *
 * This aggregate has its own layout, distinct from `std::optional`. Natural alignment may add
 * trailing padding. A null payload is unspecified and must not be read. Column null masks remain
 * separate from column data; this wrapper represents an individual nullable element.
 *
 * @tparam T Payload representation.
 */
template <typename T>
struct optional {
  T value;      ///< Payload; meaningful only when validity is 1.
  u8 validity;  ///< 0 for null, 1 for a valid payload.
};

/**
 * @brief Empty tag selecting the element-type overload of a row operation.
 * @tparam T Element representation used for overload selection.
 */
template <typename T>
struct type_tag {};

/**
 * @brief Empty tag selecting a dictionary index-storage specialization.
 * @tparam T Physical dictionary index type, independently of the dictionary's element type.
 */
template <typename T>
struct index_type_tag {};

}  // namespace cudf::abi_types
