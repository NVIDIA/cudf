/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/utilities/export.hpp>

#include <cstdint>
#include <memory>
#include <span>

namespace rtcx {
struct blob_t;
}

namespace CUDF_EXPORT cudf {

/// @brief Device-code formats accepted by LTO execution APIs.
enum class lto_binary_type : uint8_t { LTO_IR = 0, FATBIN = 1, PTX = 2 };

/**
 * @brief Non-owning device-code fragment.
 *
 * Keep the bytes alive throughout the calling operation. PTX must include its terminating zero.
 * Declare callback symbols with @c extern "C" to avoid C++ name mangling. The calling API defines
 * parameter order and whether row or segment indices precede the values.
 *
 * Error-returning callbacks return @c cudf::errc. Inputs are values; outputs and mutable state
 * are pointers. Nullable values use @c cuda::std::optional<T>, with output pointers outside the
 * optional. Empty optionals represent nulls; index and user-data parameters are not optional.
 *
 * String inputs and unmaterialized results use @c cudf::string_view. Keep their device character
 * storage alive until result materialization completes. Transform outputs with preallocated
 * offsets instead receive @c cuda::std::span<char> referring to their writable character range.
 * The UDF must stay within that range and cannot change its offsets or length.
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
  /// @brief Returns compiled bytes, or an empty span for a default-constructed object.
  [[nodiscard]] std::span<uint8_t const> view() const noexcept;

  /// @brief Returns the same non-owning byte view as @c view().
  [[nodiscard]] operator std::span<uint8_t const>() const noexcept { return view(); }

 private:
  std::shared_ptr<rtcx::blob_t> blob_;

  friend compiled_fragment compile_cuda_to_lto_ir(char const*,
                                                  bool,
                                                  std::span<char const*>,
                                                  std::span<char const*>,
                                                  std::span<char const* const>);
};

/**
 * @brief Compiles CUDA device source into an owning LTO IR fragment.
 *
 * @param source_code Null-terminated CUDA source
 * @param includes_cudf Whether to provide the bundled cuDF headers
 * @param include_names Null-terminated names of additional in-memory headers
 * @param includes Matching null-terminated header contents
 * @param include_paths Filesystem header directories, searched before bundled directories;
 * each entry must be null-terminated and omit the @c -I prefix
 * @return Shared ownership of the compiled bytes
 * @throws std::invalid_argument for null source or mismatched header spans
 * @throws std::runtime_error if compilation fails
 */
compiled_fragment compile_cuda_to_lto_ir(char const* source_code,
                                         bool includes_cudf,
                                         std::span<char const*> include_names,
                                         std::span<char const*> includes,
                                         std::span<char const* const> include_paths);

}  // namespace CUDF_EXPORT cudf
