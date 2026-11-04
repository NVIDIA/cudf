/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/utilities/export.hpp>

#include <cstdint>
#include <memory>
#include <span>

/**
 * @file
 * @brief APIs for compiling and sharing device code for user-defined functions
 */

namespace rtcx {
struct blob_t;
}

namespace CUDF_EXPORT cudf {

/**
 * @addtogroup udf
 * @{
 */

/// @brief Device-code formats accepted by LTO execution APIs.
enum class lto_binary_type : uint8_t {
  LTO_IR = 0,  ///< LTO IR binary
  FATBIN = 1   ///< FATBIN binary
};

/**
 * @brief Non-owning device-code fragment.
 *
 * Keep the bytes alive throughout the calling operation.
 *
 */
struct lto_fragment {
  lto_binary_type type = lto_binary_type::LTO_IR;  ///< Device-code format
  std::span<uint8_t const> data{};                 ///< Compiled bytes
};

/// @brief Non-owning collection of compiled device-code fragments.
using lto_fragments = std::span<lto_fragment const>;

/**
 * @brief Shares ownership of compiled device code; copies retain the same storage.
 *
 * Byte views do not own storage. Keep an owner alive while passing a view to an LTO API.
 */
struct [[nodiscard]] compiled_fragment {
  /// @brief Returns a non-owning view of the compiled fragment
  /// @return Non-owning view of the compiled fragment
  [[nodiscard]] lto_fragment view() const noexcept;

 private:
  std::shared_ptr<rtcx::blob_t> blob_;

  compiled_fragment(std::shared_ptr<rtcx::blob_t> blob) : blob_(std::move(blob)) {}

  friend compiled_fragment compile_cuda_to_lto_ir(char const*,
                                                  std::span<char const* const>,
                                                  std::span<char const* const>,
                                                  std::span<char const* const>,
                                                  bool);
};

/**
 * @brief Compiles CUDA device source into an owning LTO IR fragment.
 *
 * Compiled fragments are cached by source, in-memory headers, ordered include paths and PCH use.
 *
 * Filesystem headers and their dependencies must remain unchanged while caches may be reused;
 * use versioned include paths when changing filesystem dependencies.
 *
 * When PCH is enabled, headers used to build it must remain unchanged while it may be reused.
 * Place changing headers after `#pragma nv_hdrstop` or disable PCH.
 *
 * @param source_code Null-terminated CUDA source
 * @param include_names Null-terminated names of additional in-memory headers
 * @param includes Matching null-terminated header contents
 * @param include_paths Filesystem header directories, searched before bundled header directories
 * @param use_pch Enable automatic precompiled headers on supported NVRTC versions
 * @return Shared ownership of the compiled bytes
 * @throws std::invalid_argument for mismatched header spans
 * @throws std::runtime_error if compilation fails
 */
compiled_fragment compile_cuda_to_lto_ir(char const* source_code,
                                         std::span<char const* const> include_names,
                                         std::span<char const* const> includes,
                                         std::span<char const* const> include_paths,
                                         bool use_pch = true);

/** @} */  // end of group
}  // namespace CUDF_EXPORT cudf
