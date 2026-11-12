/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/types.hpp>
#include <cudf/utilities/export.hpp>

#include <cstdint>
#include <memory>
#include <span>
#include <string_view>

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

/**
 * @brief The type of LTO Binary
 */
enum class lto_binary_type : uint8_t {
  LTO_IR = 0,  //< LTO-IR binary
  FATBIN = 1,  //< FATBIN binary
  PTX    = 2   //< PTX source
};

/**
 * @brief A compiled device module to link into a JIT kernel.
 *
 * A module may define multiple functions and reference functions in other modules. Keep its bytes
 * alive until evaluation completes or a reusable transform program has been constructed.
 */
struct lto_fragment {
  lto_binary_type type          = lto_binary_type::LTO_IR;  ///< Binary format of the module
  std::span<uint8_t const> data = {};                       ///< The module's bytes
};

/// @brief Device modules linked into a JIT kernel.
using lto_fragments = std::span<lto_fragment const>;

/**
 * @brief LTO fragments defining a transform user-defined-function.
 *
 * The CUDF UDF ABI is:
 *
 * @code{.cpp}
 * cudf::errc udf(Outputs* ... outputs, Inputs... inputs);
 * cudf::errc indexed_udf(cudf::size_type ... indices, Outputs* ... outputs, Inputs... inputs);
 * @endcode
 *
 * The UDFs have the following calling conventions:
 *
 * - When indexed, one transform row index or two consecutive left/right join row indices are placed
 * before the output pointers. Indices use `cudf::size_type` and are never optional.
 * - Output pointers precede all inputs.
 * - Null-aware calls wrap column and scalar values in `cuda::std::optional<T>` and pass output
 * pointers as `cuda::std::optional<T>*`.
 * - User-data pointers are always raw `void*` values.
 * - The UDF must assign its outputs on every execution path and return `cudf::errc::SUCCESS` on
 * success or a `cudf::errc` indicating failure. User-defined error codes must be within the range
 * (`cudf::errc::USER_DEFINED_BEGIN` , `cudf::errc::USER_DEFINED_END`]
 *
 *
 * Fragments may export multiple symbols and reference symbols in other supplied fragments.
 * Neither this descriptor nor its fragments own storage.
 */
struct transform_udf {
  std::string_view symbol = "";  ///< Exact exported device symbol, expressible as a CUDA identifier
  lto_fragments fragments = {};  ///< Modules defining the function and its dependencies
  null_aware is_null_aware = null_aware::NO;  ///< Whether column values use optional types
  bool indexed             = false;  ///< Whether logical row indices precede all other arguments
};

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
                                                  bool,
                                                  std::string_view);
};

/**
 * @brief Compiles CUDA device source into an owning LTO IR fragment.
 *
 * Compiled fragments are cached by source, in-memory headers, ordered include paths and PCH use.
 *
 * CUDA UDFs can include `<cudf/udf_includes.hpp>` for the common cuDF value types,
 * cstring, optional, span, and device operator headers.
 *
 * Filesystem headers and their dependencies must remain unchanged while caches may be reused;
 * use versioned include paths when changing filesystem dependencies.
 *
 * When PCH is enabled, headers used to build it must remain unchanged while it may be reused.
 * Place changing headers after `#pragma nv_hdrstop` or disable PCH.
 * Source filenames do not isolate PCH reuse: NVRTC may reuse a compatible PCH across filenames.
 *
 * @param source_code Null-terminated CUDA source
 * @param include_names Null-terminated names of additional in-memory headers
 * @param includes Matching null-terminated header contents
 * @param include_paths Filesystem header directories, searched before bundled header directories
 * @param use_pch Enable automatic precompiled headers on supported NVRTC versions
 * @param pch_key PCH key used to identify or create the precompiled header
 * @return Shared ownership of the compiled bytes
 * @throws std::invalid_argument for mismatched header spans
 * @throws std::runtime_error if compilation fails
 */
compiled_fragment compile_cuda_to_lto_ir(char const* source_code,
                                         std::span<char const* const> include_names = {},
                                         std::span<char const* const> includes      = {},
                                         std::span<char const* const> include_paths = {},
                                         bool use_pch                               = true,
                                         std::string_view pch_key = "cudf_user_fragment");

/** @} */  // end of group

}  // namespace CUDF_EXPORT cudf
