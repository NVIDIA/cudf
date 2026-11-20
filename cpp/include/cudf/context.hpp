/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/utilities/export.hpp>

#include <cstdint>
#include <type_traits>

namespace CUDF_EXPORT cudf {
namespace detail {

/// @brief Flags for controlling initialization steps
enum class init_flags : std::uint32_t {
  /// @brief No initialization steps
  NONE = 0,
  /// @brief Load the nvCOMP library during initialization
  LOAD_NVCOMP = 1 << 0,
  /// @brief Default initialization steps
  DEFAULT = NONE,
  /// @brief All initialization steps
  ALL = LOAD_NVCOMP
};

/// @brief Bitwise OR operator for init_flags
/// @param lhs The left-hand side of the operator
/// @param rhs The right-hand side of the operator
/// @return The result of the bitwise OR operation
constexpr init_flags operator|(init_flags lhs, init_flags rhs) noexcept
{
  using underlying_t = std::underlying_type_t<init_flags>;
  return static_cast<init_flags>(static_cast<underlying_t>(lhs) | static_cast<underlying_t>(rhs));
}

/// @brief Bitwise AND operator for init_flags
/// @param lhs The left-hand side of the operator
/// @param rhs The right-hand side of the operator
/// @return The result of the bitwise AND operation
constexpr init_flags operator&(init_flags lhs, init_flags rhs) noexcept
{
  using underlying_t = std::underlying_type_t<init_flags>;
  return static_cast<init_flags>(static_cast<underlying_t>(lhs) & static_cast<underlying_t>(rhs));
}

/// @brief Bitwise NOT operator for init_flags
/// @param flags The flags to negate
/// @return The result of the bitwise NOT operation, only flipping bits that are part of
/// init_flags::ALL
constexpr init_flags operator~(init_flags flags) noexcept
{
  using underlying_t = std::underlying_type_t<init_flags>;
  return static_cast<init_flags>(static_cast<underlying_t>(init_flags::ALL) &
                                 ~static_cast<underlying_t>(flags));
}

/// @brief Check if a flag is set
/// @param flags The flags to check against
/// @param flag The specific flag to check for
/// @return true if all bits in `flag` are set in `flags`, false otherwise
constexpr bool has_flag(init_flags flags, init_flags flag) noexcept
{
  return (flags | flag) == flags;
}

/// @brief Ensure the cudf global context is initialized. Only the first call to this function will
/// have an effect, subsequent calls are no-ops regardless of the initialization flags.
///  This function is thread-safe and can be called from multiple threads concurrently.
///
/// It is intended for advanced users who need to explicitly control the initialization order of the
/// cuDF context. Most users should not need to call this function directly, as the context is
/// automatically initialized when needed.
///
/// @param flags Flags controlling which components to initialize
void initialize(init_flags flags = init_flags::DEFAULT);

}  // namespace detail

/**
 * @brief Enable or disable the JIT program cache
 *
 * When disabled, the cache will not be used for
 * storing or retrieving compiled programs, effectively bypassing the cache. When enabled, the
 * cache will be used as normal. This can be used to temporarily disable caching without clearing
 * the existing cache contents, allowing for easy re-enabling of the cache later.
 *
 * @param enable If `true`, the JIT program cache is enabled; if `false`, it is disabled.
 */
void enable_jit_cache(bool enable);

/**
 * @brief Clear the JIT program cache, removing all cached programs from memory and disk.
 *
 * This is a more expensive operation than simply disabling the cache, as it involves deleting
 * cached files from disk, but it also frees up any memory used by the cached programs. Use
 * `enable_jit_cache(false)` if you want to temporarily disable caching without clearing existing
 * cache contents.
 *
 * @warning For benchmarking or testing purposes, prefer `enable_jit_cache`.
 */
void clear_jit_cache();

}  // namespace CUDF_EXPORT cudf
