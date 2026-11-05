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
 * @addtogroup udfs
 * @{
 */

/// @brief Device-code formats accepted by LTO execution APIs.
enum class lto_binary_type : uint8_t { LTO_IR = 0, FATBIN = 1, PTX = 2 };

/**
 * @brief Non-owning device-code fragment.
 *
 * A module may define multiple functions and reference functions in other modules. Keep its bytes
 * alive until evaluation completes or a reusable transform program has been constructed.
 * PTX must include its terminating zero.
 * Declare callback symbols with @c extern "C" to avoid C++ name mangling.
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
  [[nodiscard]] lto_fragment view() const noexcept;

  /// @brief Returns a non-owning view of the compiled fragment
  [[nodiscard]] operator lto_fragment() const noexcept { return view(); }

 private:
  std::shared_ptr<rtcx::blob_t> blob_;

  compiled_fragment(std::shared_ptr<rtcx::blob_t> blob) : blob_(std::move(blob)) {}

  friend compiled_fragment compile_cuda_to_lto_ir(char const*,
                                                  std::span<char const*>,
                                                  std::span<char const*>,
                                                  std::span<char const* const>);
};

/**
 * @brief Compiles CUDA device source into an owning LTO IR fragment.
 *
 * @param source_code Null-terminated CUDA source
 * @param include_names Null-terminated names of additional in-memory headers
 * @param includes Matching null-terminated header contents
 * @param include_paths Filesystem header directories
 * @return Shared ownership of the compiled bytes
 * @throws std::invalid_argument for null source or mismatched header spans
 * @throws std::runtime_error if compilation fails
 */
compiled_fragment compile_cuda_to_lto_ir(char const* source_code,
                                         std::span<char const*> include_names,
                                         std::span<char const*> includes,
                                         std::span<char const* const> include_paths);

/** @} */  // end of group

}  // namespace CUDF_EXPORT cudf
